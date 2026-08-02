"""D2: an INDEPENDENT cross-check of the production solver's EV conventions.

The existing oracle (reference_solver.ReferenceCFR) only covers the flop-only FlopSolver, not the
BatchedCFR that actually generates the shipped packs — and BatchedCFR's only cross-check
(multistreet.py) shares its EV/utility conventions. So a *shared-convention* bug (a sign flip,
a wrong pot term, bad card-removal) would pass every existing test.

This test closes that gap without reusing any solver math. We solve a small real game on a
complete RIVER board (n_streets is derived from board length, so a 5-card board has NO runout —
the showdown is a deterministic 5-card eval, and the single betting street's four decision nodes
fully specify the averaged profile). Then we recompute each OOP hand's root check/bet EV FROM
SCRATCH — our own pot arithmetic, fold utilities and card-removal, and the raw `evaluate()` for
showdowns — under that same profile, and assert it matches the solver's reported per-hand EV.
Any convention error (sign flip, wrong pot term, bad card-removal) makes the two disagree.
"""
import numpy as np

from pokertrainer.cards import parse_cards, hand_str
from pokertrainer.evaluator import evaluate
from pokertrainer.presets import BB_SRP, BTN_SRP
from pokertrainer.ranges import expand_range
from pokertrainer.validate_flop import _make_solver, subsample


def _equity_matrix(oc, ic, board):
    """Ground-truth OOP-vs-IP win/tie/loss equity, computed independently of the solver.
    5-card board: deterministic showdown. 4-card (turn) board: average over every valid river
    card (excluding the board and BOTH hole pairs) — this independently exercises the runout
    denominator convention."""
    board = list(board)
    no, ni = len(oc), len(ic)
    if len(board) == 5:
        ro = np.array([evaluate((int(a), int(c), *board)) for a, c in oc])
        ri = np.array([evaluate((int(a), int(c), *board)) for a, c in ic])
        return np.where(ro[:, None] > ri[None, :], 1.0,
                        np.where(ro[:, None] == ri[None, :], 0.5, 0.0))
    E = np.zeros((no, ni))
    for i in range(no):
        a, c = int(oc[i, 0]), int(oc[i, 1])
        for j in range(ni):
            d, e = int(ic[j, 0]), int(ic[j, 1])
            if a in (d, e) or c in (d, e):
                continue                      # incompatible; B zeroes it anyway
            used = {a, c, d, e, *board}
            wins = ties = n = 0
            for r in range(52):
                if r in used:
                    continue
                b5 = board + [r]
                ov, iv = evaluate((a, c, *b5)), evaluate((d, e, *b5))
                wins += ov > iv
                ties += ov == iv
                n += 1
            E[i, j] = (wins + 0.5 * ties) / n if n else 0.0
    return E


def _independent_root_ev(oc, ic, w_o, w_i, board, pot, bet_frac, rep):
    """Recompute OOP's per-hand root EV(check)/EV(bet) independently of the solver, given the
    solver's averaged strategy `rep` (freqs at the four decision nodes)."""
    b = bet_frac * pot                       # bet size (street pot == pot, eo=ei=0 at entry)
    no, ni = len(oc), len(ic)
    E = _equity_matrix(oc, ic, board)        # ground-truth equity (with runout averaging if turn)
    # card-removal compatibility (no shared card between the two hole pairs).
    B = np.ones((no, ni))
    for i in range(no):
        a, c = int(oc[i, 0]), int(oc[i, 1])
        B[i, (ic[:, 0] == a) | (ic[:, 1] == a) | (ic[:, 0] == c) | (ic[:, 1] == c)] = 0.0

    # averaged strategy from the solver's report, indexed by our combo order.
    def col(node, act):
        m = {r["hand"]: r["freq"][act] for r in rep if r["node"] == node}
        return m
    root_bet = col("bb_first", "bet")
    ipc_bet = col("btn_vs_check", "bet")
    ovb_call = col("bb_vs_bet", "call")
    ivb_call = col("btn_vs_bet", "call")
    s_ipc_bet = np.array([ipc_bet[hand_str((int(ic[j, 0]), int(ic[j, 1])))] for j in range(ni)])
    s_ivb_call = np.array([ivb_call[hand_str((int(ic[j, 0]), int(ic[j, 1])))] for j in range(ni)])
    ev_check, ev_bet = {}, {}
    for i in range(no):
        h = hand_str((int(oc[i, 0]), int(oc[i, 1])))
        mass = float((B[i] * w_i).sum())
        if mass <= 1e-12:
            continue
        ovbc = ovb_call[h]                    # OOP's own call freq after checking into a bet
        # OOP checks: IP checks back -> showdown at pot; IP bets -> OOP folds (0, invested nothing)
        # or calls -> showdown at pot+2b having invested b.
        vchk = ((1 - s_ipc_bet) * (pot * E[i])
                + s_ipc_bet * (ovbc * ((pot + 2 * b) * E[i] - b)))
        # OOP bets: IP folds -> OOP wins pot; IP calls -> showdown at pot+2b, invested b.
        vbet = ((1 - s_ivb_call) * pot
                + s_ivb_call * ((pot + 2 * b) * E[i] - b))
        ev_check[h] = float((B[i] * w_i * vchk).sum() / mass)
        ev_bet[h] = float((B[i] * w_i * vbet).sum() / mass)
    return ev_check, ev_bet


def _cross_check(board_str, n_streets_expected, size=24, iters=400, tol=0.02):
    board = parse_cards(board_str)
    oop = np.array(subsample([c for c, _ in expand_range(BB_SRP, board)], size), dtype=np.int64)
    ip = np.array(subsample([c for c, _ in expand_range(BTN_SRP, board)], size), dtype=np.int64)
    pot, bet_frac = 5.5, 0.66
    s = _make_solver("cpu", "float64")(board, oop, ip, np.ones(len(oop)), np.ones(len(ip)),
                                       pot, bet_frac, 1)   # bet_streets=1 (last arg)
    assert s.n_streets == n_streets_expected
    s.run(iters)
    solver_ev = {r["hand"]: r["ev"] for r in s.flop_decisions_report() if r["node"] == "bb_first"}
    ev_check, ev_bet = _independent_root_ev(s.oc, s.ic, s.w_o, s.w_i, list(board),
                                            pot, bet_frac, s.flop_decisions_report())
    assert ev_check, "no OOP hands recomputed"
    worst = max(max(abs(ev_check[h] - solver_ev[h]["check"]),
                    abs(ev_bet[h] - solver_ev[h]["bet"])) for h in ev_check)
    assert worst < tol, f"{board_str}: independent EV disagrees with solver by {worst:.4f} bb"


def test_river_showdown_conventions_match_independent_recompute():
    # 5-card board -> no runout -> validates pot / fold / showdown / card-removal conventions.
    _cross_check("Ah7d2cKsQd", n_streets_expected=1)


def test_turn_runout_convention_matches_independent_recompute():
    # 4-card board -> one river runout -> ALSO validates the chance-node/runout averaging.
    _cross_check("Ah7d2cKs", n_streets_expected=2)
