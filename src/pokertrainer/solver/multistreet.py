"""Multi-street CFR spike (flop -> turn -> river) — MIT.

A FEASIBILITY SPIKE, not production. Goal: measure the compute cost and
convergence of full multi-street solving vs the flop-only POC, to decide whether
a permissive solver can practically generate a real (multi-street) library.

Simplifications (documented, deliberate):
- One bet size per street (fraction of pot); no raising.
- Full turn+river enumeration (exact, deterministic) — no Monte-Carlo sampling.
- Vectorised over the private-hand dimension; recursion over the public tree and
  chance (turn/river) nodes.

Per street the betting is the fixed 4-infoset tree:
    OOP: check | bet
      check -> IP: check | bet
        check            -> advance (next street / showdown)
        bet  -> OOP: fold | call -> advance
      bet  -> IP: fold | call    -> advance
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np

from .cfr import _strategy_from_regret

Combo = Tuple[int, int]
CHECK, BET = 0, 1
FOLD, CALL = 0, 1


class MultiStreetSpike:
    def __init__(self, flop: List[int], oop: List[Combo], ip: List[Combo],
                 w_oop: np.ndarray, w_ip: np.ndarray, pot_bb: float,
                 bet_frac: float = 0.66, streets: int = 3):
        self.flop = list(flop)
        self.oc = np.array(oop, dtype=np.int64)
        self.ic = np.array(ip, dtype=np.int64)
        self.no, self.ni = len(oop), len(ip)
        self.P0 = float(pot_bb)
        self.bet_frac = bet_frac
        self.n_streets = streets  # 1=flop only(showdown after flop), 2=+turn, 3=+river
        self.w_o = w_oop / w_oop.sum()
        self.w_i = w_ip / w_ip.sum()

        # Pairwise compatibility (board-independent): 1.0 if combos share no card.
        self.B = self._compat(self.oc, self.ic)

        # Regret / strategy-sum tables, keyed by (street, board_key, node).
        self.R: Dict[tuple, np.ndarray] = {}
        self.S: Dict[tuple, np.ndarray] = {}
        self._Ecache: Dict[frozenset, np.ndarray] = {}
        self._t = 0

    @staticmethod
    def _compat(oc, ic):
        no, ni = len(oc), len(ic)
        B = np.ones((no, ni))
        for i in range(no):
            a, b = oc[i]
            B[i, (ic[:, 0] == a) | (ic[:, 1] == a) | (ic[:, 0] == b) | (ic[:, 1] == b)] = 0.0
        return B

    # --- showdown on a complete 5-card board (cached) ---
    def _E(self, board5: List[int]) -> np.ndarray:
        key = frozenset(board5)
        E = self._Ecache.get(key)
        if E is not None:
            return E
        from ..evaluator import evaluate
        ro = np.array([evaluate((a, b, *board5)) for a, b in self.oc])
        ri = np.array([evaluate((a, b, *board5)) for a, b in self.ic])
        gt = ro[:, None] > ri[None, :]
        eqv = np.where(gt, 1.0, np.where(ro[:, None] == ri[None, :], 0.5, 0.0))
        E = self.B * eqv
        self._Ecache[key] = E
        return E

    def _reg(self, key, n_actions):
        r = self.R.get(key)
        if r is None:
            player_n = self.no if key[-1] in ("root", "ovb") else self.ni
            r = np.zeros((player_n, n_actions))
            self.R[key] = r
            self.S[key] = np.zeros_like(r)
        return r

    # --- advance: chance to next street, or showdown at the last street ---
    def _advance(self, street, board, eo, ei, ro, ri, path):
        if street >= self.n_streets:  # showdown
            E = self._E(board)
            pot = self.P0 + eo + ei
            uo = pot * (E @ ri) - eo * (self.B @ ri)
            ui = pot * ((self.B - E).T @ ro) - ei * (self.B.T @ ro)
            return uo, ui
        # chance node: deal next community card (betting history carried in path)
        used = set(board)
        deck = [c for c in range(52) if c not in used]
        UO = np.zeros(self.no)
        UI = np.zeros(self.ni)
        oc0, oc1 = self.oc[:, 0], self.oc[:, 1]
        ic0, ic1 = self.ic[:, 0], self.ic[:, 1]
        for c in deck:
            live_o = (oc0 != c) & (oc1 != c)
            live_i = (ic0 != c) & (ic1 != c)
            uo, ui = self._solve_street(street + 1, board + [c], eo, ei,
                                        ro * live_o, ri * live_i, path)
            UO += uo * live_o
            UI += ui * live_i
        denom = len(deck) - 2  # valid next cards per combo (uniform)
        return UO / denom, UI / denom

    # --- one street's betting; returns (uo, ui) counterfactual values ---
    def _solve_street(self, street, board, eo, ei, ro, ri, path=""):
        # Infoset key includes the betting-line path so different histories that
        # reach the same board are distinct infosets (full-history game).
        bkey = (path, tuple(sorted(board)))
        pot = self.P0 + eo + ei
        b = self.bet_frac * pot

        k_root = (bkey, "root")
        k_ipc = (bkey, "ipc")
        k_ovb = (bkey, "ovb")
        k_ivb = (bkey, "ivb")
        s_root = _strategy_from_regret(self._reg(k_root, 2))
        s_ipc = _strategy_from_regret(self._reg(k_ipc, 2))
        s_ovb = _strategy_from_regret(self._reg(k_ovb, 2))
        s_ivb = _strategy_from_regret(self._reg(k_ivb, 2))

        # Lines that reach advance (with reaches folded through strategies):
        # check-check
        uo_cc, ui_cc = self._advance(street, board, eo, ei,
                                     ro * s_root[:, CHECK], ri * s_ipc[:, CHECK], path + "1")
        # OOP check, IP bet, OOP call
        uo_ovbcall, ui_ovbcall = self._advance(
            street, board, eo + b, ei + b,
            ro * s_root[:, CHECK] * s_ovb[:, CALL], ri * s_ipc[:, BET], path + "2")
        # OOP bet, IP call
        uo_ivbcall, ui_ivbcall = self._advance(
            street, board, eo + b, ei + b,
            ro * s_root[:, BET], ri * s_ivb[:, CALL], path + "3")

        # --- node cfvs (fold utilities use prize convention: folder loses its
        #     prior investment; winner collects pot minus own prior investment) ---
        # ovb (OOP facing IP bet after check): opp reach = ri*s_ipc[:,BET]
        oppmass_ovb = self.B @ (ri * s_ipc[:, BET])
        u_ovb = np.stack([-eo * oppmass_ovb, uo_ovbcall], axis=1)  # OOP fold -> -eo
        # ivb (IP facing OOP bet): opp reach = ro*s_root[:,BET]
        oppmass_ivb = self.B.T @ (ro * s_root[:, BET])
        u_ivb = np.stack([-ei * oppmass_ivb, ui_ivbcall], axis=1)  # IP fold -> -ei

        # ipc (IP after OOP check): check -> advance(cc); bet -> OOP responds (ovb)
        # IP bet value = OOP folds (IP wins P0+eo) + OOP calls (advance)
        ro_checked = ro * s_root[:, CHECK]
        ip_bet_fold = (self.P0 + eo) * (self.B.T @ (ro_checked * s_ovb[:, FOLD]))
        u_ipc = np.stack([ui_cc, ip_bet_fold + ui_ovbcall], axis=1)

        # root (OOP): check -> ipc subtree; bet -> IP responds (ivb)
        # OOP check value = IP checks (advance cc) + IP bets (OOP plays ovb)
        oop_check_ipbet = s_ovb[:, CALL] * uo_ovbcall + s_ovb[:, FOLD] * (-eo * oppmass_ovb)
        u_check = uo_cc + oop_check_ipbet
        # OOP bet value = IP folds (OOP wins P0+ei) + IP calls (advance)
        oop_bet_fold = (self.P0 + ei) * (self.B @ (ri * s_ivb[:, FOLD]))
        u_bet = oop_bet_fold + uo_ivbcall
        u_root = np.stack([u_check, u_bet], axis=1)

        # --- regret + strategy-sum updates (CFR+, linear averaging) ---
        self._t_update(k_root, s_root, u_root, ro)
        self._t_update(k_ipc, s_ipc, u_ipc, ri)
        self._t_update(k_ovb, s_ovb, u_ovb, ro_checked)
        self._t_update(k_ivb, s_ivb, u_ivb, ri)

        # value to each player at street entry (both act per strategy)
        uo = (s_root * u_root).sum(axis=1)
        # IP's value at entry = over its infosets reached: but IP only acts after
        # OOP's move; combine ipc (OOP checked) and ivb (OOP bet):
        ui = (s_ipc * u_ipc).sum(axis=1) + (s_ivb * u_ivb).sum(axis=1)
        return uo, ui

    def _t_update(self, key, strat, u, reach):
        base = (strat * u).sum(axis=1, keepdims=True)
        self.R[key] = np.maximum(self.R[key] + (u - base), 0.0)
        self.S[key] += self._t * reach[:, None] * strat

    def run(self, iterations: int) -> Dict:
        tracemalloc.start()
        t0 = time.time()
        ev_curve = []
        for t in range(1, iterations + 1):
            self._t = t
            uo, ui = self._solve_street(1, self.flop, 0.0, 0.0, self.w_o.copy(), self.w_i.copy())
            if t % max(1, iterations // 15) == 0 or t == iterations:
                root_ev = float((self.w_o * uo).sum())
                ev_curve.append((t, root_ev))
        runtime = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "root_ev_oop_bb": ev_curve[-1][1],
            "root_ev_pct_pot": 100 * ev_curve[-1][1] / self.P0,
            "ev_curve": ev_curve,
            "iterations": iterations,
            "runtime_sec": runtime,
            "sec_per_iter": runtime / iterations,
            "peak_mem_mb": peak / 1e6,
            "n_infosets": len(self.R),
            "n_showdown_boards": len(self._Ecache),
            "streets": self.n_streets,
            "combos": f"{self.no}x{self.ni}",
        }
