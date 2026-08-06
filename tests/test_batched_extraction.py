"""GPU/batched port correctness — anchored to a fully INDEPENDENT recompute, NOT to
MultiStreetSpike (which an independent oracle proved wrong at streets=3, the river-betting game;
see scratch/indep3). Two guarantees:

1. The batched solver's streets=3 root EV matches a from-scratch, first-principles recompute
   under uniform strategies (what it plays at iters=1). This is the streets=3 correctness gate
   the old suite lacked (it stopped at streets=2) and the one that caught the MultiStreetSpike bug.
2. The ported extraction read-side (decision/node_action_share) is self-consistent with the
   solver's own trusted flop_decisions_report.

The CuPy (GPU) backend is the same code with a different array module, checked on a device.
"""
from functools import lru_cache

import numpy as np

from pokertrainer.cards import parse_cards, parse_hand, hand_str
from pokertrainer.evaluator import evaluate
from pokertrainer.solver.batched_gpu import BatchedGPUCFR

P0, BET = 5.5, 0.66
FLOP = parse_cards("As7h2d")
OOP = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c", "AhKh", "Ts9s"]]
IP = [parse_hand(h) for h in ["AhQh", "KsKh", "JsJh", "AhTh", "Tc9c"]]


def _independent_uniform_root_ev():
    """OOP root EV of the flop+turn+river no-raise game under UNIFORM strategies, recomputed from
    first principles (own pot/fold/showdown/runout math; only `evaluate` is reused). Prize
    convention: OOP fold -> -eo; OOP bets & IP folds -> P0+ei (uncalled bet returned); showdown ->
    (P0+eo+ei)*equity - eo. Chance denom = (52-len(board))-4."""
    def pair_ev(hi, hj):
        i0, i1 = hi; j0, j1 = hj
        hole = {i0, i1, j0, j1}

        @lru_cache(maxsize=None)
        def eqv(board):
            ov, iv = evaluate((i0, i1, *board)), evaluate((j0, j1, *board))
            return 1.0 if ov > iv else (0.5 if ov == iv else 0.0)

        @lru_cache(maxsize=None)
        def adv(street, board, eo, ei):
            if street == 3:
                return (P0 + eo + ei) * eqv(board) - eo
            cards = [c for c in range(52) if c not in set(board) | hole]
            denom = (52 - len(board)) - 4
            return sum(V(street + 1, board + (c,), eo, ei) for c in cards) / denom

        @lru_cache(maxsize=None)
        def V(street, board, eo, ei):
            b = BET * (P0 + eo + ei)
            ip_vs_check = 0.5 * adv(street, board, eo, ei) \
                + 0.5 * (0.5 * (-eo) + 0.5 * adv(street, board, eo + b, ei + b))
            oop_bet = 0.5 * (P0 + ei) + 0.5 * adv(street, board, eo + b, ei + b)
            return 0.5 * ip_vs_check + 0.5 * oop_bet

        return V(1, tuple(FLOP), 0.0, 0.0)

    num = den = 0.0
    for hi in OOP:
        for hj in IP:
            if set(hi) & set(hj):
                continue
            num += pair_ev(hi, hj); den += 1.0
    return num / den


def _solver():
    wo, wi = np.ones(len(OOP)), np.ones(len(IP))
    return BatchedGPUCFR(FLOP, OOP, IP, wo, wi, P0, BET, streets=3, bet_streets=3,
                         backend="numpy", dtype="float64", raise_x=None)


def test_batched_streets3_matches_independent_oracle():
    # iters=1 => uniform strategies on both sides; batched root EV must equal the independent
    # from-scratch recompute. (MultiStreetSpike fails this by ~0.09bb — the bug this guards.)
    ind = _independent_uniform_root_ev()
    got = _solver().run(1)["root_ev_oop_bb"]
    assert abs(got - ind) < 1e-9, f"batched streets=3 {got} != independent {ind}"


def test_extraction_self_consistent():
    # The ported decision() extraction must equal the solver's own trusted flop report.
    s = _solver()
    s.run(20)
    bkey = ("", tuple(sorted(int(c) for c in FLOP)))
    s.eval_capture_targets({bkey})
    rep = {r["hand"]: r for r in s.flop_decisions_report() if r["node"] == "bb_first"}
    for h in range(len(OOP)):
        d = s.decision(bkey, "root", h, ["check", "bet"])
        r = rep[hand_str((int(OOP[h][0]), int(OOP[h][1])))]
        for a in ("check", "bet"):
            assert abs(d["ev"][a] - r["ev"][a]) < 1e-9
