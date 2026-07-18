"""Batched public-tree multi-street CFR+ (MIT).

Same game and math as solver/multistreet.py, but restructured so each chance
node builds ALL runout child-contexts as one batched tensor dimension and calls
the next street once. This replaces tens of thousands of tiny per-board NumPy
calls with a handful of large tensor ops — the fix identified by the
multistreet spike (docs/multistreet_spike.md).

Correctness is checked against multistreet.py (the naive oracle): identical
inputs + iterations must give identical EVs.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np

from ..cards import hand_str
from ..evaluator import evaluate

CHECK, BET = 0, 1
FOLD, CALL = 0, 1


def _strat(reg: np.ndarray) -> np.ndarray:
    pos = np.maximum(reg, 0.0)
    tot = pos.sum(axis=-1, keepdims=True)
    n = reg.shape[-1]
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / n)


class BatchedCFR:
    def __init__(self, flop: List[int], oop, ip, w_oop, w_ip, pot_bb,
                 bet_frac: float = 0.66, streets: int = 3, bet_streets=None):
        self.flop = list(flop)
        self.oc = np.array(oop, dtype=np.int64)
        self.ic = np.array(ip, dtype=np.int64)
        self.no, self.ni = len(oop), len(ip)
        self.P0 = float(pot_bb)
        self.bet_frac = bet_frac
        self.n_streets = streets
        # Betting happens only on streets 1..bet_streets; later streets are pure
        # chance runouts (both check) that still realize equity to showdown. So
        # flop-only-with-runout = streets=3, bet_streets=1; full = bet_streets=3.
        self.bet_streets = streets if bet_streets is None else bet_streets
        self.w_o = (w_oop / w_oop.sum()).astype(np.float64)
        self.w_i = (w_ip / w_ip.sum()).astype(np.float64)
        # combo card lookup for fast "combo contains card c" masks
        self.o_has = np.zeros((52, self.no), dtype=bool)
        self.i_has = np.zeros((52, self.ni), dtype=bool)
        for c in range(52):
            self.o_has[c] = (self.oc[:, 0] == c) | (self.oc[:, 1] == c)
            self.i_has[c] = (self.ic[:, 0] == c) | (self.ic[:, 1] == c)
        self.B = self._compat()
        self.R: Dict[str, np.ndarray] = {}
        self.S: Dict[str, np.ndarray] = {}
        self._Ecache: Dict[frozenset, np.ndarray] = {}
        self._t = 0
        self._done = 0               # total iterations run (persists across run() calls)
        self._eval = False           # evaluation pass: no regret/strategy updates
        self._cap = None             # captured (s_root, u_root) at the flop root

    def _compat(self):
        B = np.ones((self.no, self.ni))
        for i in range(self.no):
            a, b = self.oc[i]
            B[i, self.i_has[a] | self.i_has[b]] = 0.0
        return B

    def _E(self, board5) -> np.ndarray:
        key = frozenset(board5)
        E = self._Ecache.get(key)
        if E is None:
            ro = np.array([evaluate((a, b, *board5)) for a, b in self.oc])
            ri = np.array([evaluate((a, b, *board5)) for a, b in self.ic])
            gt = ro[:, None] > ri[None, :]
            E = (self.B * np.where(gt, 1.0, np.where(ro[:, None] == ri[None, :], 0.5, 0.0))
                 ).astype(np.float32)
            self._Ecache[key] = E
        return E

    def _reg(self, path, node, C, na):
        key = path + node
        r = self.R.get(key)
        if r is None:
            r = np.zeros((C, self.no if node in ("R", "V") else self.ni, na))
            # node letter: R=root(OOP), P=ipc(IP), V=ovb(OOP), I=ivb(IP)
            self.R[key] = r
            self.S[key] = np.zeros_like(r)
        return r

    # showdown over a batch of complete boards
    def _showdown(self, boards, eo, ei, ro, ri):
        C = len(boards)
        E = np.empty((C, self.no, self.ni), dtype=np.float32)
        for k, bd in enumerate(boards):
            E[k] = self._E(bd)
        pot = (self.P0 + eo + ei)[:, None]
        uo = pot * np.einsum("cij,cj->ci", E, ri) - eo[:, None] * (ri @ self.B.T)
        ui = pot * np.einsum("cij,ci->cj", (self.B - E), ro) - ei[:, None] * (ro @ self.B)
        return uo, ui

    # chance node: batch every valid runout card into the next-street solve
    def _chance(self, street, boards, eo, ei, ro, ri, path):
        C = len(boards)
        child_boards, parent_idx, cards = [], [], []
        ceo, cei, cro, cri = [], [], [], []
        for k in range(C):
            used = set(boards[k])
            for c in range(52):
                if c in used:
                    continue
                child_boards.append(boards[k] + [c])
                parent_idx.append(k)
                cards.append(c)
                ceo.append(eo[k]); cei.append(ei[k])
                cro.append(ro[k] * ~self.o_has[c])
                cri.append(ri[k] * ~self.i_has[c])
        parent_idx = np.array(parent_idx)
        cards = np.array(cards)
        uo_c, ui_c = self._solve(street + 1, child_boards,
                                 np.array(ceo), np.array(cei),
                                 np.array(cro), np.array(cri), path)
        # zero contributions where the combo holds the dealt card, then scatter-add
        uo_c = uo_c * (~self.o_has[cards])
        ui_c = ui_c * (~self.i_has[cards])
        UO = np.zeros((C, self.no)); UI = np.zeros((C, self.ni))
        np.add.at(UO, parent_idx, uo_c)
        np.add.at(UI, parent_idx, ui_c)
        # current board has (street+2) cards; valid next cards per combo:
        denom = (52 - (street + 2)) - 2
        return UO / denom, UI / denom

    def _get_strat(self, path, node, C, na):
        """Regret-matched current strategy while training; the iteration-averaged
        strategy (from the strategy-sum S) in eval mode. The average is the stable
        equilibrium readout — CFR+ last-iterate oscillates, so per-hand preferred
        actions must be read from the average."""
        self._reg(path, node, C, na)          # ensure R and S exist
        if self._eval:
            s = self.S[path + node]
            tot = s.sum(axis=-1, keepdims=True)
            return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / na)
        return _strat(self.R[path + node])

    def _solve(self, street, boards, eo, ei, ro, ri, path):
        C = len(boards)
        if street > self.bet_streets:
            # No betting this street: both check, realize runout / showdown.
            if street >= self.n_streets:
                return self._showdown(boards, eo, ei, ro, ri)
            return self._chance(street, boards, eo, ei, ro, ri, path + "c")
        b = self.bet_frac * (self.P0 + eo + ei)          # [C]
        s_root = self._get_strat(path, "R", C, 2)
        s_ipc = self._get_strat(path, "P", C, 2)
        s_ovb = self._get_strat(path, "V", C, 2)
        s_ivb = self._get_strat(path, "I", C, 2)

        # line reaches
        ro_ck = ro * s_root[:, :, CHECK]
        ro_bt = ro * s_root[:, :, BET]
        ri_ck = ri * s_ipc[:, :, CHECK]
        ri_bt = ri * s_ipc[:, :, BET]
        eo2, ei2 = eo + b, ei + b

        if street >= self.n_streets:
            adv = lambda bd, e1, e2, r1, r2, p: self._showdown(bd, e1, e2, r1, r2)
        else:
            adv = lambda bd, e1, e2, r1, r2, p: self._chance(street, bd, e1, e2, r1, r2, p)
        # L1 check-check ; L2 check-bet-call ; L3 bet-call
        uo_L1, ui_L1 = adv(boards, eo, ei, ro_ck, ri_ck, path + "1")
        uo_L2, ui_L2 = adv(boards, eo2, ei2, ro_ck * s_ovb[:, :, CALL], ri_bt, path + "2")
        uo_L3, ui_L3 = adv(boards, eo2, ei2, ro_bt, ri * s_ivb[:, :, CALL], path + "3")

        oppmass_ovb = ri_bt @ self.B.T                    # [C,no]
        oppmass_ivb = ro_bt @ self.B                      # [C,ni]

        u_ovb = np.stack([-eo[:, None] * oppmass_ovb, uo_L2], axis=2)
        u_ivb = np.stack([-ei[:, None] * oppmass_ivb, ui_L3], axis=2)

        ip_bet_fold = (self.P0 + eo)[:, None] * ((ro_ck * s_ovb[:, :, FOLD]) @ self.B)
        u_ipc = np.stack([ui_L1, ip_bet_fold + ui_L2], axis=2)

        u_check = (uo_L1 + s_ovb[:, :, CALL] * uo_L2
                   + s_ovb[:, :, FOLD] * (-eo[:, None] * oppmass_ovb))
        oop_bet_fold = (self.P0 + ei)[:, None] * ((ri * s_ivb[:, :, FOLD]) @ self.B.T)
        u_bet = oop_bet_fold + uo_L3
        u_root = np.stack([u_check, u_bet], axis=2)

        if self._eval and street == 1 and path == "":
            # snapshot all four flop decision nodes (both players, root + facing bet)
            self._cap = {
                "s_root": s_root.copy(), "u_root": u_root.copy(),   # BB check/bet
                "s_ipc": s_ipc.copy(), "u_ipc": u_ipc.copy(),       # BTN check/bet (vs check)
                "s_ovb": s_ovb.copy(), "u_ovb": u_ovb.copy(),       # BB fold/call (vs bet)
                "s_ivb": s_ivb.copy(), "u_ivb": u_ivb.copy(),       # BTN fold/call (vs bet)
            }

        self._update(path + "R", s_root, u_root, ro)
        self._update(path + "P", s_ipc, u_ipc, ri)
        self._update(path + "V", s_ovb, u_ovb, ro_ck)
        self._update(path + "I", s_ivb, u_ivb, ri)

        uo = (s_root * u_root).sum(axis=2)
        ui = (s_ipc * u_ipc).sum(axis=2) + (s_ivb * u_ivb).sum(axis=2)
        return uo, ui

    def _update(self, key, s, u, reach):
        if self._eval:
            return
        base = (s * u).sum(axis=2, keepdims=True)
        self.R[key] = np.maximum(self.R[key] + (u - base), 0.0)
        self.S[key] += self._t * reach[:, :, None] * s

    def _eval_capture(self):
        self._eval = True
        self._cap = None
        self._solve(1, [list(self.flop)], np.zeros(1), np.zeros(1),
                    self.w_o[None, :].copy(), self.w_i[None, :].copy(), "")
        self._eval = False
        return self._cap

    def flop_root_report(self) -> Dict[str, Dict]:
        """Per-OOP-hand flop-root (BB check/bet) decision under the averaged
        profile: conditional EV (bb), frequencies, preferred action."""
        cap = self._eval_capture()
        s_root, u_root = cap["s_root"], cap["u_root"]
        opp = self.B @ self.w_i
        opp = np.where(opp > 1e-12, opp, 1.0)
        out = {}
        for i in range(self.no):
            ev_ch = float(u_root[0, i, CHECK] / opp[i])
            ev_bt = float(u_root[0, i, BET] / opp[i])
            out[hand_str((int(self.oc[i, 0]), int(self.oc[i, 1])))] = {
                "ev": {"check": ev_ch, "bet": ev_bt},
                "freq": {"check": float(s_root[0, i, CHECK]), "bet": float(s_root[0, i, BET])},
                "preferred": "check" if ev_ch >= ev_bt else "bet",
            }
        return out

    def flop_decisions_report(self) -> List[Dict]:
        """All four flop decision nodes (both players, root + facing a bet) under
        the averaged profile. One record per (node, acting-player hand): actions,
        per-action conditional EV (bb), frequencies, preferred action, and the
        opponent reach mass into that node (0 => rarely reached)."""
        cap = self._eval_capture()
        s_root = cap["s_root"]; s_ipc = cap["s_ipc"]
        ro_ck = self.w_o * s_root[0, :, CHECK]         # OOP checked reach
        ro_bt = self.w_o * s_root[0, :, BET]           # OOP bet reach
        ri_bt = self.w_i * s_ipc[0, :, BET]            # IP bet (vs check) reach

        nodes = [
            # (key, acting hands, oc?, actions, cfv, strat, opp reach mass per hand)
            ("bb_first",    self.oc, "check", "bet", cap["u_root"], s_root, self.B @ self.w_i),
            ("btn_vs_check", self.ic, "check", "bet", cap["u_ipc"], s_ipc, self.B.T @ ro_ck),
            ("bb_vs_bet",   self.oc, "fold", "call", cap["u_ovb"], cap["s_ovb"], self.B @ ri_bt),
            ("btn_vs_bet",  self.ic, "fold", "call", cap["u_ivb"], cap["s_ivb"], self.B.T @ ro_bt),
        ]
        acting = {"bb_first": "BB", "btn_vs_check": "BTN", "bb_vs_bet": "BB", "btn_vs_bet": "BTN"}
        recs: List[Dict] = []
        for key, combos, a0, a1, u, s, opp_mass in nodes:
            safe = np.where(opp_mass > 1e-12, opp_mass, 1.0)
            for i in range(len(combos)):
                ev0 = float(u[0, i, 0] / safe[i])
                ev1 = float(u[0, i, 1] / safe[i])
                recs.append({
                    "node": key,
                    "acting_player": acting[key],
                    "hand": hand_str((int(combos[i, 0]), int(combos[i, 1]))),
                    "actions": [a0, a1],
                    "ev": {a0: ev0, a1: ev1},
                    "freq": {a0: float(s[0, i, 0]), a1: float(s[0, i, 1])},
                    "preferred": a0 if ev0 >= ev1 else a1,
                    "reach_mass": float(opp_mass[i]),
                })
        return recs

    def run(self, iterations: int) -> Dict:
        tracemalloc.start()
        t0 = time.time()
        ev = []
        ro0 = self.w_o[None, :].copy()
        ri0 = self.w_i[None, :].copy()
        for t in range(1, iterations + 1):
            self._t = self._done + t          # persistent weight for linear averaging
            uo, ui = self._solve(1, [list(self.flop)], np.zeros(1), np.zeros(1),
                                 ro0.copy(), ri0.copy(), "")
            if t % max(1, iterations // 12) == 0 or t == iterations:
                ev.append((t, float((self.w_o * uo[0]).sum())))
        self._done += iterations
        rt = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "root_ev_oop_bb": ev[-1][1],
            "root_ev_pct_pot": 100 * ev[-1][1] / self.P0,
            "ev_curve": ev,
            "iterations": iterations,
            "runtime_sec": rt,
            "sec_per_iter": rt / iterations,
            "peak_mem_mb": peak / 1e6,
            "n_infosets": len(self.R),
            "streets": self.n_streets,
            "combos": f"{self.no}x{self.ni}",
        }
