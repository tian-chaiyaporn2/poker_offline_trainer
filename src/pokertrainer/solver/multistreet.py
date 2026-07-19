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
FOLD, CALL, RAISE = 0, 1, 2


class MultiStreetSpike:
    def __init__(self, flop: List[int], oop: List[Combo], ip: List[Combo],
                 w_oop: np.ndarray, w_ip: np.ndarray, pot_bb: float,
                 bet_frac: float = 0.66, streets: int = 3, raise_x=None):
        self.flop = list(flop)
        self.oc = np.array(oop, dtype=np.int64)
        self.ic = np.array(ip, dtype=np.int64)
        self.no, self.ni = len(oop), len(ip)
        self.P0 = float(pot_bb)
        self.bet_frac = bet_frac
        self.raise_x = raise_x       # raise-to multiple of the bet; None = no raise
        self.n_streets = streets  # 1=flop only(showdown after flop), 2=+turn, 3=+river
        self.w_o = (w_oop / w_oop.sum()).astype(np.float64)   # match batched dtype
        self.w_i = (w_ip / w_ip.sum()).astype(np.float64)

        # Pairwise compatibility (board-independent): 1.0 if combos share no card.
        self.B = self._compat(self.oc, self.ic)

        # Regret / strategy-sum tables, keyed by (street, board_key, node).
        self.R: Dict[tuple, np.ndarray] = {}
        self.S: Dict[tuple, np.ndarray] = {}
        self._Ecache: Dict[frozenset, np.ndarray] = {}
        self._t = 0
        self._eval = False

    @staticmethod
    def _compat(oc, ic):
        no, ni = len(oc), len(ic)
        B = np.ones((no, ni))
        for i in range(no):
            a, b = oc[i]
            B[i, (ic[:, 0] == a) | (ic[:, 1] == a) | (ic[:, 0] == b) | (ic[:, 1] == b)] = 0.0
        return B

    def _get_strat(self, key, n_actions):
        """Regret-matched strategy while training; average strategy in eval mode."""
        self._reg(key, n_actions)
        if self._eval:
            s = self.S[key]
            tot = s.sum(axis=1, keepdims=True)
            return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0),
                            np.full_like(s, 1.0 / n_actions))
        return _strategy_from_regret(self.R[key])

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
            # OOP-owned nodes: root, ovb (OOP vs IP bet), iroop (OOP vs IP raise)
            player_n = self.no if key[-1] in ("root", "ovb", "iroop") else self.ni
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
        # Cards that collide with neither private hand: |deck| - 4.
        # (|deck| - 2 under-weights showdown EV vs fold equity.)
        denom = len(deck) - 4
        return UO / denom, UI / denom

    # --- one street's betting; returns (uo, ui) counterfactual values ---
    def _solve_street(self, street, board, eo, ei, ro, ri, path=""):
        if self.raise_x is not None:
            return self._solve_street_raise(street, board, eo, ei, ro, ri, path)
        # Infoset key includes the betting-line path so different histories that
        # reach the same board are distinct infosets (full-history game).
        bkey = (path, tuple(sorted(board)))
        pot = self.P0 + eo + ei
        b = self.bet_frac * pot

        k_root = (bkey, "root")
        k_ipc = (bkey, "ipc")
        k_ovb = (bkey, "ovb")
        k_ivb = (bkey, "ivb")
        s_root = self._get_strat(k_root, 2)
        s_ipc = self._get_strat(k_ipc, 2)
        s_ovb = self._get_strat(k_ovb, 2)
        s_ivb = self._get_strat(k_ivb, 2)

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

    # --- betting with a raise facing a bet: fold/call/raise, one raise/street ---
    def _solve_street_raise(self, street, board, eo, ei, ro, ri, path=""):
        bkey = (path, tuple(sorted(board)))
        pot = self.P0 + eo + ei
        b = self.bet_frac * pot
        R = self.raise_x * b                     # raise-to (raiser's street investment)

        s_root = self._get_strat((bkey, "root"), 2)
        s_ipc = self._get_strat((bkey, "ipc"), 2)
        s_ovb = self._get_strat((bkey, "ovb"), 3)    # fold/call/raise
        s_ivb = self._get_strat((bkey, "ivb"), 3)
        s_orip = self._get_strat((bkey, "orip"), 2)  # IP vs OOP raise
        s_iroop = self._get_strat((bkey, "iroop"), 2)  # OOP vs IP raise

        ro_ck = ro * s_root[:, CHECK]; ro_bt = ro * s_root[:, BET]
        ri_ck = ri * s_ipc[:, CHECK]; ri_bt = ri * s_ipc[:, BET]

        # advance lines (uo uses opp=IP reach, ui uses opp=OOP reach passed in)
        uo_cc, ui_cc = self._advance(street, board, eo, ei, ro_ck, ri_ck, path + "1")
        uo_L2c, ui_L2c = self._advance(street, board, eo + b, ei + b,
                                       ro_ck * s_ovb[:, CALL], ri_bt, path + "2c")
        uo_L2r, ui_L2r = self._advance(street, board, eo + R, ei + R,
                                       ro_ck * s_ovb[:, RAISE], ri_bt * s_orip[:, CALL], path + "2r")
        uo_L3c, ui_L3c = self._advance(street, board, eo + b, ei + b,
                                       ro_bt, ri * s_ivb[:, CALL], path + "3c")
        uo_L3r, ui_L3r = self._advance(street, board, eo + R, ei + R,
                                       ro_bt * s_iroop[:, CALL], ri * s_ivb[:, RAISE], path + "3r")

        # orip: IP facing OOP raise (line check-bet-raise). opp = OOP that raised.
        oppmass_orip = self.B.T @ (ro_ck * s_ovb[:, RAISE])
        u_orip = np.stack([-(ei + b) * oppmass_orip, ui_L2r], axis=1)
        # iroop: OOP facing IP raise (line bet-raise). opp = IP that raised.
        oppmass_iroop = self.B @ (ri * s_ivb[:, RAISE])
        u_iroop = np.stack([-(eo + b) * oppmass_iroop, uo_L3r], axis=1)

        # ovb: OOP facing IP bet (fold/call/raise). opp = IP that bet = ri_bt.
        oppmass_ovb = self.B @ ri_bt
        ovb_raise = (self.P0 + ei + b) * (self.B @ (ri_bt * s_orip[:, FOLD])) + uo_L2r
        u_ovb = np.stack([-eo * oppmass_ovb, uo_L2c, ovb_raise], axis=1)
        # ivb: IP facing OOP bet (fold/call/raise). opp = OOP that bet = ro_bt.
        oppmass_ivb = self.B.T @ ro_bt
        ivb_raise = (self.P0 + eo + b) * (self.B.T @ (ro_bt * s_iroop[:, FOLD])) + ui_L3r
        u_ivb = np.stack([-ei * oppmass_ivb, ui_L3c, ivb_raise], axis=1)

        # ipc: IP after OOP check (check/bet)
        u_ipc_bet = ((self.P0 + eo) * (self.B.T @ (ro_ck * s_ovb[:, FOLD]))
                     + ui_L2c + (s_orip * u_orip).sum(axis=1))
        u_ipc = np.stack([ui_cc, u_ipc_bet], axis=1)

        # root: OOP (check/bet)
        u_check = uo_cc + (s_ovb * u_ovb).sum(axis=1)
        u_bet = ((self.P0 + ei) * (self.B @ (ri * s_ivb[:, FOLD]))
                 + uo_L3c + (s_iroop * u_iroop).sum(axis=1))
        u_root = np.stack([u_check, u_bet], axis=1)

        self._t_update((bkey, "root"), s_root, u_root, ro)
        self._t_update((bkey, "ipc"), s_ipc, u_ipc, ri)
        self._t_update((bkey, "ovb"), s_ovb, u_ovb, ro_ck)
        self._t_update((bkey, "ivb"), s_ivb, u_ivb, ri)
        self._t_update((bkey, "orip"), s_orip, u_orip, ri_bt)
        self._t_update((bkey, "iroop"), s_iroop, u_iroop, ro_bt)

        uo = (s_root * u_root).sum(axis=1)
        ui = (s_ipc * u_ipc).sum(axis=1) + (s_ivb * u_ivb).sum(axis=1)
        return uo, ui

    def _t_update(self, key, strat, u, reach):
        if self._eval:
            return
        base = (strat * u).sum(axis=1, keepdims=True)
        self.R[key] = np.maximum(self.R[key] + (u - base), 0.0)
        self.S[key] += self._t * reach[:, None] * strat

    def run(self, iterations: int) -> Dict:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        tracemalloc.start()
        t0 = time.time()
        ev_curve = []
        for t in range(1, iterations + 1):
            self._t = t
            uo, ui = self._solve_street(1, self.flop, 0.0, 0.0, self.w_o.copy(), self.w_i.copy())
            if t % max(1, iterations // 15) == 0 or t == iterations:
                ev_curve.append((t, float((self.w_o * uo).sum())))
        # Report EV under the averaged strategy (CFR guarantee), not last iterate.
        self._eval = True
        uo_avg, _ = self._solve_street(1, self.flop, 0.0, 0.0, self.w_o.copy(), self.w_i.copy())
        self._eval = False
        # See batched.py: condition root EV on compatible matchups (matches cfr.py).
        joint = float(self.w_o @ (self.B @ self.w_i))
        root_ev = float((self.w_o * uo_avg).sum()) / (joint if joint > 1e-12 else 1.0)
        runtime = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "root_ev_oop_bb": root_ev,
            "root_ev_pct_pot": 100 * root_ev / self.P0,
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
