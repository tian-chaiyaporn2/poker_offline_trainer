"""Multi-size betting street shared by the batched solvers (MIT).

ONE implementation of the multi-size tree recursion for both `BatchedCFR` (NumPy)
and `BatchedGPUCFR` (NumPy/CuPy via `xp`). The two backends differ only in:

- the array module (`_mx`) and the dtype of the one fresh accumulator (`_mdtype`;
  the GPU keeps every accumulator in `self.dtype` so float32 never promotes),
- the shape of their `_get_strat` / `_showdown` signatures (bridged by
  `_mstrat` / `_mshowdown`),
- whether the root capture is copied (`_mkeep_cap`).

`multistreet.MultiStreetSpike._solve_street_multi` is deliberately NOT built on
this: it is the independent per-board oracle the batched solvers are checked
against (tests/test_bet_sizes.py), so it must not share their code.
"""

from __future__ import annotations

CHECK = 0
FOLD, CALL, RAISE = 0, 1, 2


class MultiSizeStreetMixin:
    """Provides `_solve_multi` for a batched solver.

    The host class supplies: `bet_fracs`, `raise_x`, `n_streets`, `P0`, `B`, `ni`,
    `_eval`, `_capbet`, `_chance`, `_update`, plus the backend hooks `_mx`,
    `_mdtype`, `_mstrat(path, node, C, oop, na)`, `_mshowdown(boards, eo, ei, ro,
    ri, path)` and `_mkeep_cap(cap)`.
    """

    def _solve_multi(self, street, boards, eo, ei, ro, ri, path):
        """Multi-size betting street (K = len(bet_fracs) >= 2), with or without raises.
        Batched mirror of MultiStreetSpike._solve_street_multi: check | bet_1..bet_K at
        root/ipc, one fold/call[/raise] response node per size (V{k}/I{k}) and, with
        raises, one fold/call node per raise-over-size-k (O{k}/W{k}). Path tokens carry
        the size index ("2{k}", "3{k}", "2{k}c", "2{k}r", ...) so every line is distinct."""
        xp = self._mx
        C = len(boards)
        K = len(self.bet_fracs)
        rz = self.raise_x is not None
        na_resp = 3 if rz else 2
        pot = self.P0 + eo + ei
        s_root = self._mstrat(path, "R", C, True, K + 1)
        s_ipc = self._mstrat(path, "P", C, False, K + 1)
        ro_ck = ro * s_root[:, :, CHECK]
        ri_ck = ri * s_ipc[:, :, CHECK]
        if street >= self.n_streets:
            adv = self._mshowdown
        else:
            adv = lambda bd, e1, e2, r1, r2, p: self._chance(street, bd, e1, e2, r1, r2, p)
        uo_cc, ui_cc = adv(boards, eo, ei, ro_ck, ri_ck, path + "1")

        u_check = uo_cc
        ui_ivb = xp.zeros((C, self.ni), dtype=self._mdtype)
        root_bets, ipc_bets, updates, cap = [], [], [], {}
        for k, frac in enumerate(self.bet_fracs):
            b = self._capbet(frac * pot, eo, ei)                      # [C]
            ro_bt = ro * s_root[:, :, 1 + k]
            ri_bt = ri * s_ipc[:, :, 1 + k]
            s_ovb = self._mstrat(path, f"V{k}", C, True, na_resp)
            s_ivb = self._mstrat(path, f"I{k}", C, False, na_resp)
            oppmass_ovb = ri_bt @ self.B.T                            # [C,no]
            oppmass_ivb = ro_bt @ self.B                              # [C,ni]
            if not rz:
                uo_L2, ui_L2 = adv(boards, eo + b, ei + b, ro_ck * s_ovb[:, :, CALL], ri_bt,
                                   path + f"2{k}")
                uo_L3, ui_L3 = adv(boards, eo + b, ei + b, ro_bt, ri * s_ivb[:, :, CALL],
                                   path + f"3{k}")
                u_ovb = xp.stack([-eo[:, None] * oppmass_ovb, uo_L2], axis=2)
                u_ivb = xp.stack([-ei[:, None] * oppmass_ivb, ui_L3], axis=2)
                ipc_bet = (self.P0 + eo)[:, None] * ((ro_ck * s_ovb[:, :, FOLD]) @ self.B) + ui_L2
                root_bet = (self.P0 + ei)[:, None] * ((ri * s_ivb[:, :, FOLD]) @ self.B.T) + uo_L3
            else:
                Rz = self._capbet(self.raise_x * b, eo, ei)
                s_orip = self._mstrat(path, f"O{k}", C, False, 2)
                s_iroop = self._mstrat(path, f"W{k}", C, True, 2)
                uo_L2c, ui_L2c = adv(boards, eo + b, ei + b, ro_ck * s_ovb[:, :, CALL], ri_bt,
                                     path + f"2{k}c")
                uo_L2r, ui_L2r = adv(boards, eo + Rz, ei + Rz, ro_ck * s_ovb[:, :, RAISE],
                                     ri_bt * s_orip[:, :, CALL], path + f"2{k}r")
                uo_L3c, ui_L3c = adv(boards, eo + b, ei + b, ro_bt, ri * s_ivb[:, :, CALL],
                                     path + f"3{k}c")
                uo_L3r, ui_L3r = adv(boards, eo + Rz, ei + Rz, ro_bt * s_iroop[:, :, CALL],
                                     ri * s_ivb[:, :, RAISE], path + f"3{k}r")
                oppmass_orip = (ro_ck * s_ovb[:, :, RAISE]) @ self.B
                u_orip = xp.stack([-(ei + b)[:, None] * oppmass_orip, ui_L2r], axis=2)
                oppmass_iroop = (ri * s_ivb[:, :, RAISE]) @ self.B.T
                u_iroop = xp.stack([-(eo + b)[:, None] * oppmass_iroop, uo_L3r], axis=2)
                ovb_raise = ((self.P0 + ei + b)[:, None]
                             * ((ri_bt * s_orip[:, :, FOLD]) @ self.B.T) + uo_L2r)
                u_ovb = xp.stack([-eo[:, None] * oppmass_ovb, uo_L2c, ovb_raise], axis=2)
                ivb_raise = ((self.P0 + eo + b)[:, None]
                             * ((ro_bt * s_iroop[:, :, FOLD]) @ self.B) + ui_L3r)
                u_ivb = xp.stack([-ei[:, None] * oppmass_ivb, ui_L3c, ivb_raise], axis=2)
                ipc_bet = ((self.P0 + eo)[:, None] * ((ro_ck * s_ovb[:, :, FOLD]) @ self.B)
                           + ui_L2c + (s_orip * u_orip).sum(axis=2))
                root_bet = ((self.P0 + ei)[:, None] * ((ri * s_ivb[:, :, FOLD]) @ self.B.T)
                            + uo_L3c + (s_iroop * u_iroop).sum(axis=2))
                updates.append((path + f"O{k}", s_orip, u_orip, ri_bt))
                updates.append((path + f"W{k}", s_iroop, u_iroop, ro_bt))
            u_check = u_check + (s_ovb * u_ovb).sum(axis=2)
            ui_ivb = ui_ivb + (s_ivb * u_ivb).sum(axis=2)
            root_bets.append(root_bet)
            ipc_bets.append(ipc_bet)
            updates.append((path + f"V{k}", s_ovb, u_ovb, ro_ck))
            updates.append((path + f"I{k}", s_ivb, u_ivb, ri))
            cap.update({f"s_ovb_{k}": s_ovb, f"u_ovb_{k}": u_ovb,
                        f"s_ivb_{k}": s_ivb, f"u_ivb_{k}": u_ivb})

        u_root = xp.stack([u_check] + root_bets, axis=2)
        u_ipc = xp.stack([ui_cc] + ipc_bets, axis=2)
        if self._eval and street == 1 and path == "":
            cap.update({"s_root": s_root, "u_root": u_root, "s_ipc": s_ipc, "u_ipc": u_ipc})
            self._mkeep_cap(cap)
        self._update(path + "R", s_root, u_root, ro)
        self._update(path + "P", s_ipc, u_ipc, ri)
        for key, s_, u_, reach in updates:
            self._update(key, s_, u_, reach)
        uo = (s_root * u_root).sum(axis=2)
        ui = (s_ipc * u_ipc).sum(axis=2) + ui_ivb
        return uo, ui
