"""Independent reference solver: vanilla CFR (MIT).

The PRD wants comparison against an *independent* reference. TexasSolver's
licence and the permissibility of internal commercial benchmarking are unverified
(see docs/licenses.md), and it is not bundled. As a genuinely independent
cross-check we provide a second solver implemented with a **different algorithm**
(plain CFR — no CFR+ regret flooring, no linear averaging — and an independently
written node traversal). If two different algorithms converge to the same EVs
and preferred actions, that is strong evidence the strategy computation is
correct rather than an artefact of one implementation.

This shares the equity/showdown layer with the production solver; that shared
layer is validated *independently* by a Monte-Carlo equity check (mc_equity.py).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _regret_match(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    tot = pos.sum(axis=1, keepdims=True)
    n = regret.shape[1]
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0),
                    np.full_like(regret, 1.0 / n))


class ReferenceCFR:
    """Plain CFR (vanilla). Same game as FlopSolver, independently coded."""

    def __init__(self, equity, compat, w_oop, w_ip, pot_bb, small_frac, large_frac):
        self.eq = equity
        self.C = compat
        self.CE = compat * equity
        self.CEip = compat * (1.0 - equity)
        self.P0 = float(pot_bb)
        self.bs = small_frac * self.P0
        self.bl = large_frac * self.P0
        self.no, self.ni = equity.shape
        self.wo = w_oop / w_oop.sum()
        self.wi = w_ip / w_ip.sum()
        keys3 = {"root": self.no, "ipc": self.ni}
        keys2 = {"ovs": self.no, "ovl": self.no, "ivs": self.ni, "ivl": self.ni}
        self.R = {k: np.zeros((n, 3)) for k, n in keys3.items()}
        self.R.update({k: np.zeros((n, 2)) for k, n in keys2.items()})
        self.S = {k: np.zeros_like(v) for k, v in self.R.items()}

    # showdown utility for a player given opponent reach r
    def _sd_oop(self, inv, r):
        pot = self.P0 + 2 * inv
        return pot * (self.CE @ r) - inv * (self.C @ r)

    def _sd_ip(self, inv, r):
        pot = self.P0 + 2 * inv
        return pot * (self.CEip.T @ r) - inv * (self.C.T @ r)

    def step(self):
        C = self.C; P0, bs, bl = self.P0, self.bs, self.bl
        wo, wi = self.wo, self.wi
        s = {k: _regret_match(v) for k, v in self.R.items()}

        # --- OOP counterfactual values ---
        u_bet_s = P0 * (C @ (wi * s["ivs"][:, 0])) + self._sd_oop(bs, wi * s["ivs"][:, 1])
        u_bet_l = P0 * (C @ (wi * s["ivl"][:, 0])) + self._sd_oop(bl, wi * s["ivl"][:, 1])
        ov_s_call = self._sd_oop(bs, wi * s["ipc"][:, 1])
        ov_l_call = self._sd_oop(bl, wi * s["ipc"][:, 2])
        u_check = (self._sd_oop(0.0, wi * s["ipc"][:, 0])
                   + s["ovs"][:, 1] * ov_s_call + s["ovl"][:, 1] * ov_l_call)
        u_root = np.stack([u_check, u_bet_s, u_bet_l], axis=1)
        u_ovs = np.stack([np.zeros(self.no), ov_s_call], axis=1)
        u_ovl = np.stack([np.zeros(self.no), ov_l_call], axis=1)

        # --- IP counterfactual values ---
        ro_c = wo * s["root"][:, 0]; ro_s = wo * s["root"][:, 1]; ro_l = wo * s["root"][:, 2]
        u_ivs = np.stack([np.zeros(self.ni), self._sd_ip(bs, ro_s)], axis=1)
        u_ivl = np.stack([np.zeros(self.ni), self._sd_ip(bl, ro_l)], axis=1)
        u_ipc = np.stack([
            self._sd_ip(0.0, ro_c),
            P0 * (C.T @ (ro_c * s["ovs"][:, 0])) + self._sd_ip(bs, ro_c * s["ovs"][:, 1]),
            P0 * (C.T @ (ro_c * s["ovl"][:, 0])) + self._sd_ip(bl, ro_c * s["ovl"][:, 1]),
        ], axis=1)

        reach = {"root": wo, "ipc": wi, "ovs": ro_c, "ovl": ro_c, "ivs": wi, "ivl": wi}
        u = {"root": u_root, "ipc": u_ipc, "ovs": u_ovs, "ovl": u_ovl, "ivs": u_ivs, "ivl": u_ivl}
        for k in self.R:
            base = (s[k] * u[k]).sum(axis=1, keepdims=True)
            self.R[k] += (u[k] - base)                     # vanilla: no flooring
            self.S[k] += reach[k][:, None] * s[k]          # vanilla: uniform average

    def solve(self, iterations: int) -> Dict[str, np.ndarray]:
        for _ in range(iterations):
            self.step()
        avg = {}
        for k, v in self.S.items():
            tot = v.sum(axis=1, keepdims=True)
            n = v.shape[1]
            avg[k] = np.where(tot > 0, v / np.where(tot > 0, tot, 1.0),
                              np.full_like(v, 1.0 / n))
        return avg

    def root_action_ev(self, avg) -> np.ndarray:
        C = self.C; P0, bs, bl = self.P0, self.bs, self.bl; wi = self.wi
        u_bet_s = P0 * (C @ (wi * avg["ivs"][:, 0])) + self._sd_oop(bs, wi * avg["ivs"][:, 1])
        u_bet_l = P0 * (C @ (wi * avg["ivl"][:, 0])) + self._sd_oop(bl, wi * avg["ivl"][:, 1])
        u_check = (self._sd_oop(0.0, wi * avg["ipc"][:, 0])
                   + avg["ovs"][:, 1] * self._sd_oop(bs, wi * avg["ipc"][:, 1])
                   + avg["ovl"][:, 1] * self._sd_oop(bl, wi * avg["ipc"][:, 2]))
        u_root = np.stack([u_check, u_bet_s, u_bet_l], axis=1)
        opp_mass = np.where((C @ wi) > 1e-12, C @ wi, 1.0)
        return u_root / opp_mass[:, None]

    def root_ev_bb(self, avg) -> float:
        C = self.C; P0, bs, bl = self.P0, self.bs, self.bl; wi = self.wi; wo = self.wo
        u_bet_s = P0 * (C @ (wi * avg["ivs"][:, 0])) + self._sd_oop(bs, wi * avg["ivs"][:, 1])
        u_bet_l = P0 * (C @ (wi * avg["ivl"][:, 0])) + self._sd_oop(bl, wi * avg["ivl"][:, 1])
        u_check = (self._sd_oop(0.0, wi * avg["ipc"][:, 0])
                   + avg["ovs"][:, 1] * self._sd_oop(bs, wi * avg["ipc"][:, 1])
                   + avg["ovl"][:, 1] * self._sd_oop(bl, wi * avg["ipc"][:, 2]))
        u_root = np.stack([u_check, u_bet_s, u_bet_l], axis=1)
        joint = float(wo @ (C @ wi))
        joint = joint if joint > 1e-12 else 1.0
        return float((wo * (avg["root"] * u_root).sum(axis=1)).sum()) / joint
