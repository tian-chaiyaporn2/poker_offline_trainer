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
FOLD, CALL, RAISE = 0, 1, 2


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
                 bet_frac=0.66, streets=3, backend="auto", dtype="float64",
                 bet_streets=None, raise_x=None):
        self.xp, self.scatter_add, self.to_dev, self.to_host, self.backend = get_backend(backend)
        xp = self.xp
        # Compute dtype for reaches/regrets/showdown. float32 is much faster on
        # consumer GPUs (which are FP64-crippled); float64 is exact.
        self.dtype = np.dtype(dtype)
        self.flop = list(flop)
        self.oc = np.array(oop, dtype=np.int64)
        self.ic = np.array(ip, dtype=np.int64)
        self.no, self.ni = len(oop), len(ip)
        self.P0 = float(pot_bb)
        self.bet_frac = bet_frac
        self.n_streets = streets
        self.bet_streets = streets if bet_streets is None else bet_streets
        self.raise_x = raise_x

        # host "alive" masks: combo does NOT contain card c
        o_alive = np.ones((52, self.no), np.float64)
        i_alive = np.ones((52, self.ni), np.float64)
        for c in range(52):
            o_alive[c, (self.oc[:, 0] == c) | (self.oc[:, 1] == c)] = 0.0
            i_alive[c, (self.ic[:, 0] == c) | (self.ic[:, 1] == c)] = 0.0
        self.o_alive = xp.asarray(o_alive.astype(self.dtype))
        self.i_alive = xp.asarray(i_alive.astype(self.dtype))

        B = np.ones((self.no, self.ni))
        for i in range(self.no):
            a, b = self.oc[i]
            B[i, (self.ic[:, 0] == a) | (self.ic[:, 1] == a)
               | (self.ic[:, 0] == b) | (self.ic[:, 1] == b)] = 0.0
        self.B = xp.asarray(B.astype(self.dtype))

        self.w_o = xp.asarray((w_oop / w_oop.sum()).astype(self.dtype))
        self.w_i = xp.asarray((w_ip / w_ip.sum()).astype(self.dtype))

        self._build_showdown_tensor()      # E_all on device + board->idx map
        self.R: Dict[str, object] = {}
        self.S: Dict[str, object] = {}
        self._child: Dict[str, tuple] = {}   # cached chance structure per path
        self._bidx: Dict[str, object] = {}   # cached showdown board indices per path
        self._t = 0
        self._done = 0
        self._eval = False
        self._cap = None

    def _build_showdown_tensor(self):
        """Precompute win-matrix E for every showdown board -> one device tensor."""
        deck = [c for c in range(52) if c not in set(self.flop)]
        boards = [list(self.flop) + list(extra)
                  for extra in combinations(deck, self.n_streets - 1)]
        self._board_id = {frozenset(b): k for k, b in enumerate(boards)}
        Bh = self.to_host(self.B)                    # host copy of B, once
        E = np.empty((len(boards), self.no, self.ni), np.float32)
        for k, b5 in enumerate(boards):
            ro = np.array([evaluate((a, b, *b5)) for a, b in self.oc])
            ri = np.array([evaluate((a, b, *b5)) for a, b in self.ic])
            win = np.where(ro[:, None] > ri[None, :], 1.0,
                           np.where(ro[:, None] == ri[None, :], 0.5, 0.0))
            E[k] = Bh * win
        self.E_all = self.xp.asarray(E)

    def _reg(self, key, C, player_o, na):
        r = self.R.get(key)
        if r is None:
            r = self.xp.zeros((C, self.no if player_o else self.ni, na), dtype=self.dtype)
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
            # Uniform over cards that collide with neither private hand.
            denom = (52 - (street + 2)) - 4
            cm = (xp.asarray(pk_h), xp.asarray(pc_h), pk_h, pc_h, child_boards, denom)
            self._child[path] = cm
        pk_d, pc_d, pk_h, pc_h, child_boards, denom = cm

        child_ro = ro[pk_d] * self.o_alive[pc_d]      # [M,no] one gather+mul
        child_ri = ri[pk_d] * self.i_alive[pc_d]
        uo_c, ui_c = self._solve(street + 1, child_boards, eo[pk_d], ei[pk_d],
                                 child_ro, child_ri, path)
        uo_c = uo_c * self.o_alive[pc_d]
        ui_c = ui_c * self.i_alive[pc_d]
        UO = xp.zeros((C, self.no), dtype=self.dtype); UI = xp.zeros((C, self.ni), dtype=self.dtype)
        self.scatter_add(UO, pk_d, uo_c)
        self.scatter_add(UI, pk_d, ui_c)
        return UO / denom, UI / denom

    def _get_strat(self, path, node, C, player_o, na=2):
        """Current strategy while training; iteration-averaged (from S) in eval
        mode — the stable equilibrium readout (last-iterate oscillates)."""
        self._reg(path + node, C, player_o, na)   # ensure R,S exist
        xp = self.xp
        if self._eval:
            s = self.S[path + node]
            tot = s.sum(axis=-1, keepdims=True)
            return xp.where(tot > 0, s / xp.where(tot > 0, tot, 1.0), 1.0 / na)
        return _strat(xp, self.R[path + node])

    def _solve_raise(self, street, boards, eo, ei, ro, ri, path):
        xp = self.xp
        C = len(boards)
        b = self.bet_frac * (self.P0 + eo + ei)
        Rz = self.raise_x * b
        s_root = self._get_strat(path, "R", C, True)
        s_ipc = self._get_strat(path, "P", C, False)
        s_ovb = self._get_strat(path, "V", C, True, 3)
        s_ivb = self._get_strat(path, "I", C, False, 3)
        s_orip = self._get_strat(path, "O", C, False)
        s_iroop = self._get_strat(path, "W", C, True)

        ro_ck = ro * s_root[:, :, CHECK]; ro_bt = ro * s_root[:, :, BET]
        ri_ck = ri * s_ipc[:, :, CHECK]; ri_bt = ri * s_ipc[:, :, BET]
        if street >= self.n_streets:
            adv = lambda bd, e1, e2, r1, r2, p: self._showdown(bd, e1, e2, r1, r2, p)
        else:
            adv = lambda bd, e1, e2, r1, r2, p: self._chance(street, bd, e1, e2, r1, r2, p)

        uo_cc, ui_cc = adv(boards, eo, ei, ro_ck, ri_ck, path + "1")
        uo_L2c, ui_L2c = adv(boards, eo + b, ei + b, ro_ck * s_ovb[:, :, CALL], ri_bt, path + "2c")
        uo_L2r, ui_L2r = adv(boards, eo + Rz, ei + Rz, ro_ck * s_ovb[:, :, RAISE],
                             ri_bt * s_orip[:, :, CALL], path + "2r")
        uo_L3c, ui_L3c = adv(boards, eo + b, ei + b, ro_bt, ri * s_ivb[:, :, CALL], path + "3c")
        uo_L3r, ui_L3r = adv(boards, eo + Rz, ei + Rz, ro_bt * s_iroop[:, :, CALL],
                             ri * s_ivb[:, :, RAISE], path + "3r")

        oppmass_orip = (ro_ck * s_ovb[:, :, RAISE]) @ self.B
        u_orip = xp.stack([-(ei + b)[:, None] * oppmass_orip, ui_L2r], axis=2)
        oppmass_iroop = (ri * s_ivb[:, :, RAISE]) @ self.B.T
        u_iroop = xp.stack([-(eo + b)[:, None] * oppmass_iroop, uo_L3r], axis=2)

        oppmass_ovb = ri_bt @ self.B.T
        ovb_raise = (self.P0 + ei + b)[:, None] * ((ri_bt * s_orip[:, :, FOLD]) @ self.B.T) + uo_L2r
        u_ovb = xp.stack([-eo[:, None] * oppmass_ovb, uo_L2c, ovb_raise], axis=2)
        oppmass_ivb = ro_bt @ self.B
        ivb_raise = (self.P0 + eo + b)[:, None] * ((ro_bt * s_iroop[:, :, FOLD]) @ self.B) + ui_L3r
        u_ivb = xp.stack([-ei[:, None] * oppmass_ivb, ui_L3c, ivb_raise], axis=2)

        u_ipc_bet = ((self.P0 + eo)[:, None] * ((ro_ck * s_ovb[:, :, FOLD]) @ self.B)
                     + ui_L2c + (s_orip * u_orip).sum(axis=2))
        u_ipc = xp.stack([ui_cc, u_ipc_bet], axis=2)
        u_check = uo_cc + (s_ovb * u_ovb).sum(axis=2)
        u_bet = ((self.P0 + ei)[:, None] * ((ri * s_ivb[:, :, FOLD]) @ self.B.T)
                 + uo_L3c + (s_iroop * u_iroop).sum(axis=2))
        u_root = xp.stack([u_check, u_bet], axis=2)

        if self._eval and street == 1 and path == "":
            self._cap = {"s_root": s_root, "u_root": u_root, "s_ipc": s_ipc, "u_ipc": u_ipc,
                         "s_ovb": s_ovb, "u_ovb": u_ovb, "s_ivb": s_ivb, "u_ivb": u_ivb}
        self._update(path + "R", s_root, u_root, ro)
        self._update(path + "P", s_ipc, u_ipc, ri)
        self._update(path + "V", s_ovb, u_ovb, ro_ck)
        self._update(path + "I", s_ivb, u_ivb, ri)
        self._update(path + "O", s_orip, u_orip, ri_bt)
        self._update(path + "W", s_iroop, u_iroop, ro_bt)
        uo = (s_root * u_root).sum(axis=2)
        ui = (s_ipc * u_ipc).sum(axis=2) + (s_ivb * u_ivb).sum(axis=2)
        return uo, ui

    def _solve(self, street, boards, eo, ei, ro, ri, path):
        xp = self.xp
        C = len(boards)
        if street > self.bet_streets:
            # distinct path suffix so per-path caches don't collide with the
            # betting-street chance node at the same prefix
            if street >= self.n_streets:
                return self._showdown(boards, eo, ei, ro, ri, path + "s")
            return self._chance(street, boards, eo, ei, ro, ri, path + "c")
        if self.raise_x is not None:
            return self._solve_raise(street, boards, eo, ei, ro, ri, path)
        b = self.bet_frac * (self.P0 + eo + ei)
        s_root = self._get_strat(path, "R", C, True)
        s_ipc = self._get_strat(path, "P", C, False)
        s_ovb = self._get_strat(path, "V", C, True)
        s_ivb = self._get_strat(path, "I", C, False)

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

        if self._eval and street == 1 and path == "":
            self._cap = {"s_root": s_root, "u_root": u_root, "s_ipc": s_ipc, "u_ipc": u_ipc,
                         "s_ovb": s_ovb, "u_ovb": u_ovb, "s_ivb": s_ivb, "u_ivb": u_ivb}

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
        # keep regrets in self.dtype — a stray float64 here would silently promote
        # the whole float32 run and cripple throughput on FP64-limited GPUs (T4).
        self.R[key] = self.xp.maximum(self.R[key] + (u - base), 0.0).astype(self.dtype, copy=False)
        self.S[key] += self._t * reach[:, :, None] * s

    def _eval_capture(self):
        xp = self.xp
        self._eval = True
        self._cap = None
        self._solve(1, [list(self.flop)], xp.zeros(1, dtype=self.dtype),
                    xp.zeros(1, dtype=self.dtype), self.w_o[None, :] + 0,
                    self.w_i[None, :] + 0, "")
        self._eval = False
        return {k: self.to_host(v) for k, v in self._cap.items()}

    def flop_root_report(self):
        """Per-OOP-hand flop-root (BB check/bet) decision under the averaged
        profile. Matches the CPU solver's report."""
        cap = self._eval_capture()
        s_root, u_root = cap["s_root"], cap["u_root"]
        opp = np.where((o := self.to_host(self.B @ self.w_i)) > 1e-12, o, 1.0)
        from ..cards import hand_str
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

    def flop_decisions_report(self):
        """All four flop decision nodes (both players, root + facing bet) under the
        averaged profile. Matches CPU solver; response nodes carry the raise action
        when raise_x is set."""
        from .batched import _flop_decisions_from_cap
        return _flop_decisions_from_cap(self)

    def run(self, iterations: int) -> Dict:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        xp = self.xp
        t0 = time.time()
        ro0 = self.w_o[None, :]
        ri0 = self.w_i[None, :]
        for t in range(1, iterations + 1):
            self._t = self._done + t
            self._solve(1, [list(self.flop)],
                        xp.zeros(1, dtype=self.dtype), xp.zeros(1, dtype=self.dtype),
                        ro0 + 0, ri0 + 0, "")
        self._done += iterations
        self._eval = True
        uo_avg, _ = self._solve(1, [list(self.flop)],
                                xp.zeros(1, dtype=self.dtype), xp.zeros(1, dtype=self.dtype),
                                self.w_o[None, :] + 0, self.w_i[None, :] + 0, "")
        self._eval = False
        root_ev = float(self.to_host((self.w_o * uo_avg[0]).sum()))
        if self.backend == "cupy":
            self.xp.cuda.Stream.null.synchronize()
        rt = time.time() - t0
        return {
            "backend": self.backend,
            "dtype": str(self.dtype),
            "root_ev_oop_bb": root_ev,
            "root_ev_pct_pot": 100 * root_ev / self.P0,
            "iterations": iterations,
            "runtime_sec": rt,
            "sec_per_iter": rt / iterations,
            "n_infosets": len(self.R),
            "streets": self.n_streets,
            "combos": f"{self.no}x{self.ni}",
        }
