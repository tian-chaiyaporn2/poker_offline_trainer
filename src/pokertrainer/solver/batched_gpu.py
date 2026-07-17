"""GPU-ready batched multi-street CFR+ (MIT).

Same algorithm and math as solver/batched.py, with two changes that make it fast
on a GPU:

1. **Pluggable array backend** (`xp`): CuPy if available (GPU), else NumPy (CPU).
   CuPy mirrors the NumPy API, so the body is written once. With the NumPy
   backend this reproduces batched.py **exactly** (see tests) — that is the
   correctness guarantee for the GPU path, which cannot be unit-tested without a
   device.
2. **Vectorised chance node** + **on-device showdown tensor.** The runout
   child-contexts are built with a single gather (no per-context Python loop),
   and all boards' win-matrices live in one device array, so the dominant
   showdown einsum runs as one big GPU kernel — exactly the batched matmul GPUs
   are good at.

The showdown was shown (docs/multistreet_spike.md) to be the ~n² bottleneck of
CPU batched CFR; running it on GPU is the intended path to minutes/board.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, List

import numpy as np

from ..evaluator import evaluate

CHECK, BET = 0, 1
FOLD, CALL = 0, 1


def get_backend(prefer: str = "auto"):
    """Return (xp, scatter_add, to_device, to_host, name)."""
    if prefer in ("auto", "cupy"):
        try:
            import cupy as cp  # type: ignore
            import cupyx  # type: ignore

            def scatter_add(target, idx, src):
                cupyx.scatter_add(target, idx, src)

            return (cp, scatter_add, cp.asarray, cp.asnumpy, "cupy")
        except Exception:
            if prefer == "cupy":
                raise
    # NumPy fallback
    def scatter_add(target, idx, src):
        np.add.at(target, idx, src)

    return (np, scatter_add, np.asarray, (lambda a: np.asarray(a)), "numpy")


def _strat(xp, reg):
    pos = xp.maximum(reg, 0.0)
    tot = pos.sum(axis=-1, keepdims=True)
    n = reg.shape[-1]
    return xp.where(tot > 0, pos / xp.where(tot > 0, tot, 1.0), 1.0 / n)


class BatchedGPUCFR:
    def __init__(self, flop, oop, ip, w_oop, w_ip, pot_bb,
                 bet_frac=0.66, streets=3, backend="auto"):
        self.xp, self.scatter_add, self.to_dev, self.to_host, self.backend = get_backend(backend)
        xp = self.xp
        self.flop = list(flop)
        self.oc = np.array(oop, dtype=np.int64)
        self.ic = np.array(ip, dtype=np.int64)
        self.no, self.ni = len(oop), len(ip)
        self.P0 = float(pot_bb)
        self.bet_frac = bet_frac
        self.n_streets = streets

        # host "alive" masks: combo does NOT contain card c
        o_alive = np.ones((52, self.no), np.float64)
        i_alive = np.ones((52, self.ni), np.float64)
        for c in range(52):
            o_alive[c, (self.oc[:, 0] == c) | (self.oc[:, 1] == c)] = 0.0
            i_alive[c, (self.ic[:, 0] == c) | (self.ic[:, 1] == c)] = 0.0
        self.o_alive = xp.asarray(o_alive)
        self.i_alive = xp.asarray(i_alive)

        B = np.ones((self.no, self.ni))
        for i in range(self.no):
            a, b = self.oc[i]
            B[i, (self.ic[:, 0] == a) | (self.ic[:, 1] == a)
               | (self.ic[:, 0] == b) | (self.ic[:, 1] == b)] = 0.0
        self.B = xp.asarray(B)

        self.w_o = xp.asarray(w_oop / w_oop.sum())
        self.w_i = xp.asarray(w_ip / w_ip.sum())

        self._build_showdown_tensor()      # E_all on device + board->idx map
        self.R: Dict[str, object] = {}
        self.S: Dict[str, object] = {}
        self._child: Dict[str, tuple] = {}   # cached chance structure per path
        self._bidx: Dict[str, object] = {}   # cached showdown board indices per path
        self._t = 0

    def _build_showdown_tensor(self):
        """Precompute win-matrix E for every showdown board -> one device tensor."""
        deck = [c for c in range(52) if c not in set(self.flop)]
        boards = [list(self.flop) + list(extra)
                  for extra in combinations(deck, self.n_streets - 1)]
        self._board_id = {frozenset(b): k for k, b in enumerate(boards)}
        E = np.empty((len(boards), self.no, self.ni), np.float32)
        for k, b5 in enumerate(boards):
            ro = np.array([evaluate((a, b, *b5)) for a, b in self.oc])
            ri = np.array([evaluate((a, b, *b5)) for a, b in self.ic])
            gt = ro[:, None] > ri[None, :]
            E[k] = self.B_host_win(gt, ro, ri)
        self.E_all = self.xp.asarray(E)

    def B_host_win(self, gt, ro, ri):
        Bh = self.to_host(self.B) if self.backend == "cupy" else np.asarray(self.B)
        return Bh * np.where(gt, 1.0, np.where(ro[:, None] == ri[None, :], 0.5, 0.0))

    def _reg(self, key, C, player_o, na):
        r = self.R.get(key)
        if r is None:
            r = self.xp.zeros((C, self.no if player_o else self.ni, na))
            self.R[key] = r
            self.S[key] = self.xp.zeros_like(r)
        return r

    def _board_idx(self, boards, path):
        idx = self._bidx.get(path)
        if idx is None:
            idx = self.xp.asarray(np.array([self._board_id[frozenset(b)] for b in boards]))
            self._bidx[path] = idx
        return idx

    def _showdown(self, boards, eo, ei, ro, ri, path):
        xp = self.xp
        idx = self._board_idx(boards, path)
        E = self.E_all[idx]                      # [C,no,ni] gathered on device
        pot = (self.P0 + eo + ei)[:, None]
        uo = pot * xp.einsum("cij,cj->ci", E, ri) - eo[:, None] * (ri @ self.B.T)
        ui = pot * xp.einsum("cij,ci->cj", (self.B - E), ro) - ei[:, None] * (ro @ self.B)
        return uo, ui

    def _chance(self, street, boards, eo, ei, ro, ri, path):
        xp = self.xp
        C = len(boards)
        cm = self._child.get(path)
        if cm is None:
            pk, pc, child_boards = [], [], []
            for k in range(C):
                used = set(boards[k])
                for c in range(52):
                    if c in used:
                        continue
                    pk.append(k); pc.append(c); child_boards.append(boards[k] + [c])
            pk_h = np.array(pk); pc_h = np.array(pc)
            denom = (52 - (street + 2)) - 2
            cm = (xp.asarray(pk_h), xp.asarray(pc_h), pk_h, pc_h, child_boards, denom)
            self._child[path] = cm
        pk_d, pc_d, pk_h, pc_h, child_boards, denom = cm

        child_ro = ro[pk_d] * self.o_alive[pc_d]      # [M,no] one gather+mul
        child_ri = ri[pk_d] * self.i_alive[pc_d]
        uo_c, ui_c = self._solve(street + 1, child_boards, eo[pk_d], ei[pk_d],
                                 child_ro, child_ri, path)
        uo_c = uo_c * self.o_alive[pc_d]
        ui_c = ui_c * self.i_alive[pc_d]
        UO = xp.zeros((C, self.no)); UI = xp.zeros((C, self.ni))
        self.scatter_add(UO, pk_d, uo_c)
        self.scatter_add(UI, pk_d, ui_c)
        return UO / denom, UI / denom

    def _solve(self, street, boards, eo, ei, ro, ri, path):
        xp = self.xp
        C = len(boards)
        b = self.bet_frac * (self.P0 + eo + ei)
        s_root = _strat(xp, self._reg(path + "R", C, True, 2))
        s_ipc = _strat(xp, self._reg(path + "P", C, False, 2))
        s_ovb = _strat(xp, self._reg(path + "V", C, True, 2))
        s_ivb = _strat(xp, self._reg(path + "I", C, False, 2))

        ro_ck = ro * s_root[:, :, CHECK]; ro_bt = ro * s_root[:, :, BET]
        ri_ck = ri * s_ipc[:, :, CHECK]; ri_bt = ri * s_ipc[:, :, BET]
        eo2, ei2 = eo + b, ei + b
        if street >= self.n_streets:
            adv = self._showdown
        else:
            adv = lambda bo, e1, e2, r1, r2, p: self._chance(street, bo, e1, e2, r1, r2, p)

        uo_L1, ui_L1 = adv(boards, eo, ei, ro_ck, ri_ck, path + "1")
        uo_L2, ui_L2 = adv(boards, eo2, ei2, ro_ck * s_ovb[:, :, CALL], ri_bt, path + "2")
        uo_L3, ui_L3 = adv(boards, eo2, ei2, ro_bt, ri * s_ivb[:, :, CALL], path + "3")

        oppmass_ovb = ri_bt @ self.B.T
        oppmass_ivb = ro_bt @ self.B
        u_ovb = xp.stack([-eo[:, None] * oppmass_ovb, uo_L2], axis=2)
        u_ivb = xp.stack([-ei[:, None] * oppmass_ivb, ui_L3], axis=2)
        ip_bet_fold = (self.P0 + eo)[:, None] * ((ro_ck * s_ovb[:, :, FOLD]) @ self.B)
        u_ipc = xp.stack([ui_L1, ip_bet_fold + ui_L2], axis=2)
        u_check = (uo_L1 + s_ovb[:, :, CALL] * uo_L2
                   + s_ovb[:, :, FOLD] * (-eo[:, None] * oppmass_ovb))
        oop_bet_fold = (self.P0 + ei)[:, None] * ((ri * s_ivb[:, :, FOLD]) @ self.B.T)
        u_root = xp.stack([u_check, oop_bet_fold + uo_L3], axis=2)

        self._update(path + "R", s_root, u_root, ro)
        self._update(path + "P", s_ipc, u_ipc, ri)
        self._update(path + "V", s_ovb, u_ovb, ro_ck)
        self._update(path + "I", s_ivb, u_ivb, ri)
        uo = (s_root * u_root).sum(axis=2)
        ui = (s_ipc * u_ipc).sum(axis=2) + (s_ivb * u_ivb).sum(axis=2)
        return uo, ui

    def _update(self, key, s, u, reach):
        base = (s * u).sum(axis=2, keepdims=True)
        self.R[key] = self.xp.maximum(self.R[key] + (u - base), 0.0)
        self.S[key] += self._t * reach[:, :, None] * s

    def run(self, iterations: int) -> Dict:
        xp = self.xp
        t0 = time.time()
        ev = []
        ro0 = self.w_o[None, :]
        ri0 = self.w_i[None, :]
        for t in range(1, iterations + 1):
            self._t = t
            uo, ui = self._solve(1, [list(self.flop)],
                                 xp.zeros(1), xp.zeros(1), ro0 + 0, ri0 + 0, "")
            if t == iterations:
                root_ev = float(self.to_host((self.w_o * uo[0]).sum()))
        if self.backend == "cupy":
            self.xp.cuda.Stream.null.synchronize()
        rt = time.time() - t0
        return {
            "backend": self.backend,
            "root_ev_oop_bb": root_ev,
            "root_ev_pct_pot": 100 * root_ev / self.P0,
            "iterations": iterations,
            "runtime_sec": rt,
            "sec_per_iter": rt / iterations,
            "n_infosets": len(self.R),
            "streets": self.n_streets,
            "combos": f"{self.no}x{self.ni}",
        }
