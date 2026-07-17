"""Vectorised CFR+ solver for the POC flop-only game (MIT).

The game and utility conventions are specified in docs/solver_design.md. The
private-hand dimension is vectorised with NumPy; the public betting tree is the
fixed 6-infoset tree below. CFR+ is deterministic (no sampling), so repeated
runs with the same inputs and iteration count are identical — this is what makes
the stability requirement in PRD §6 trivially satisfiable.

Infosets (owner, actions):
  root         OOP  [check, bet_small, bet_large]
  ip_vs_check  IP   [check, bet_small, bet_large]   (after OOP checks)
  oop_vs_s     OOP  [fold, call]                    (after OOP check, IP bets small)
  oop_vs_l     OOP  [fold, call]                    (after OOP check, IP bets large)
  ip_vs_s      IP   [fold, call]                    (after OOP bets small)
  ip_vs_l      IP   [fold, call]                    (after OOP bets large)
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Action label tables (index order matters — matches array columns).
ROOT_ACTIONS = ["check", "bet_small", "bet_large"]
IPCHECK_ACTIONS = ["check", "bet_small", "bet_large"]
RESPONSE_ACTIONS = ["fold", "call"]


def _strategy_from_regret(regret: np.ndarray) -> np.ndarray:
    """Regret matching: normalise positive regrets; uniform if none positive."""
    pos = np.maximum(regret, 0.0)
    total = pos.sum(axis=1, keepdims=True)
    uniform = np.full_like(regret, 1.0 / regret.shape[1])
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), uniform)


@dataclass
class SolveResult:
    # Average strategies (probabilities), one row per combo.
    strategies: Dict[str, np.ndarray]
    # Per-hand EV of each action (bb), conditional on holding that hand.
    action_ev: Dict[str, np.ndarray]
    action_labels: Dict[str, List[str]]
    root_ev_oop_bb: float
    root_ev_oop_pct_pot: float
    pot_bb: float
    iterations: int
    exploitability_curve: List[Tuple[int, float]]  # (iteration, exploitability_bb)
    final_exploitability_bb: float
    final_exploitability_pct_pot: float
    runtime_sec: float
    peak_mem_mb: float
    oop_combos: List[Tuple[int, int]] = field(default_factory=list)
    ip_combos: List[Tuple[int, int]] = field(default_factory=list)


class FlopSolver:
    def __init__(
        self,
        equity: np.ndarray,        # [n_o, n_i] OOP win prob (ties 0.5)
        compat: np.ndarray,        # [n_o, n_i] 1.0 if compatible
        w_oop: np.ndarray,         # [n_o] prior weights (will be normalised)
        w_ip: np.ndarray,          # [n_i]
        pot_bb: float,
        small_frac: float,
        large_frac: float,
    ) -> None:
        self.eq = equity
        self.C = compat
        self.CE = compat * equity              # OOP-perspective showdown
        self.CEip = compat * (1.0 - equity)    # IP-perspective showdown
        self.CT = compat.T.copy()
        self.CEipT = self.CEip.T.copy()
        self.P0 = float(pot_bb)
        self.bs = small_frac * self.P0
        self.bl = large_frac * self.P0
        self.n_o = equity.shape[0]
        self.n_i = equity.shape[1]

        self.w_o = w_oop / w_oop.sum()
        self.w_i = w_ip / w_ip.sum()

        # Regret (Q) and strategy-sum (S) accumulators per infoset.
        self.Q = {
            "root": np.zeros((self.n_o, 3)),
            "ip_vs_check": np.zeros((self.n_i, 3)),
            "oop_vs_s": np.zeros((self.n_o, 2)),
            "oop_vs_l": np.zeros((self.n_o, 2)),
            "ip_vs_s": np.zeros((self.n_i, 2)),
            "ip_vs_l": np.zeros((self.n_i, 2)),
        }
        self.S = {k: np.zeros_like(v) for k, v in self.Q.items()}

    # --- terminal utility building blocks (per-hand, opponent-reach weighted) ---

    def _showdown_oop(self, invest: float, ri: np.ndarray) -> np.ndarray:
        pot = self.P0 + 2 * invest
        return pot * (self.CE @ ri) - invest * (self.C @ ri)

    def _showdown_ip(self, invest: float, ro: np.ndarray) -> np.ndarray:
        pot = self.P0 + 2 * invest
        return pot * (self.CEipT @ ro) - invest * (self.CT @ ro)

    # --- one CFR+ iteration -------------------------------------------------

    def _iteration(self, t: int) -> None:
        C, CT = self.C, self.CT
        P0, bs, bl = self.P0, self.bs, self.bl
        w_o, w_i = self.w_o, self.w_i

        s_root = _strategy_from_regret(self.Q["root"])
        s_ipc = _strategy_from_regret(self.Q["ip_vs_check"])
        s_ovs = _strategy_from_regret(self.Q["oop_vs_s"])
        s_ovl = _strategy_from_regret(self.Q["oop_vs_l"])
        s_ivs = _strategy_from_regret(self.Q["ip_vs_s"])
        s_ivl = _strategy_from_regret(self.Q["ip_vs_l"])

        # ---- OOP-perspective counterfactual values (opp = IP, reach w_i) ----
        # Node: OOP bets small -> IP fold/call (s_ivs)
        ri_f = w_i * s_ivs[:, 0]
        ri_c = w_i * s_ivs[:, 1]
        u_oop_bet_s = P0 * (C @ ri_f) + self._showdown_oop(bs, ri_c)
        ri_f = w_i * s_ivl[:, 0]
        ri_c = w_i * s_ivl[:, 1]
        u_oop_bet_l = P0 * (C @ ri_f) + self._showdown_oop(bl, ri_c)

        # Node: OOP checks -> IP (s_ipc) check / bet_s / bet_l
        # IP checks -> showdown at 0 invest
        u_check_ipcheck = self._showdown_oop(0.0, w_i * s_ipc[:, 0])
        # IP bets small -> OOP decides (s_ovs). cfv of OOP call:
        ri_s = w_i * s_ipc[:, 1]
        cfv_ovs_call = self._showdown_oop(bs, ri_s)      # fold cfv = 0
        node_val_ovs = s_ovs[:, 1] * cfv_ovs_call
        ri_l = w_i * s_ipc[:, 2]
        cfv_ovl_call = self._showdown_oop(bl, ri_l)
        node_val_ovl = s_ovl[:, 1] * cfv_ovl_call
        u_oop_check = u_check_ipcheck + node_val_ovs + node_val_ovl

        u_root = np.stack([u_oop_check, u_oop_bet_s, u_oop_bet_l], axis=1)

        # ---- OOP response infosets (owner OOP, opp reach into node) ----
        # oop_vs_s: reached after OOP check + IP bet_s; cfv already = cfv_ovs_call
        u_ovs = np.stack([np.zeros(self.n_o), cfv_ovs_call], axis=1)
        u_ovl = np.stack([np.zeros(self.n_o), cfv_ovl_call], axis=1)

        # ---- IP-perspective counterfactual values (opp = OOP, reach w_o) ----
        # ip_vs_s: OOP bet small, OOP reach = w_o * s_root[:,1]
        ro_s = w_o * s_root[:, 1]
        cfv_ivs_fold = np.zeros(self.n_i)
        cfv_ivs_call = self._showdown_ip(bs, ro_s)
        u_ivs = np.stack([cfv_ivs_fold, cfv_ivs_call], axis=1)
        ro_l = w_o * s_root[:, 2]
        cfv_ivl_call = self._showdown_ip(bl, ro_l)
        u_ivl = np.stack([np.zeros(self.n_i), cfv_ivl_call], axis=1)

        # ip_vs_check: OOP checked, OOP reach = w_o * s_root[:,0]
        ro_c = w_o * s_root[:, 0]
        u_ipc_check = self._showdown_ip(0.0, ro_c)
        # IP bet small -> OOP responds s_ovs (fold->IP wins P0, call->showdown)
        u_ipc_bet_s = (
            P0 * (CT @ (ro_c * s_ovs[:, 0]))
            + self._showdown_ip(bs, ro_c * s_ovs[:, 1])
        )
        u_ipc_bet_l = (
            P0 * (CT @ (ro_c * s_ovl[:, 0]))
            + self._showdown_ip(bl, ro_c * s_ovl[:, 1])
        )
        u_ipc = np.stack([u_ipc_check, u_ipc_bet_s, u_ipc_bet_l], axis=1)

        # ---- CFR+ regret & linear-averaged strategy updates ----
        reach = {
            "root": w_o,
            "ip_vs_check": w_i,
            "oop_vs_s": ro_c,          # OOP reach into node (checked)
            "oop_vs_l": ro_c,
            "ip_vs_s": w_i,
            "ip_vs_l": w_i,
        }
        cfv = {
            "root": u_root,
            "ip_vs_check": u_ipc,
            "oop_vs_s": u_ovs,
            "oop_vs_l": u_ovl,
            "ip_vs_s": u_ivs,
            "ip_vs_l": u_ivl,
        }
        strat = {
            "root": s_root,
            "ip_vs_check": s_ipc,
            "oop_vs_s": s_ovs,
            "oop_vs_l": s_ovl,
            "ip_vs_s": s_ivs,
            "ip_vs_l": s_ivl,
        }
        for key in self.Q:
            u = cfv[key]
            s = strat[key]
            baseline = (s * u).sum(axis=1, keepdims=True)
            # CFR+: floor cumulative regret at 0
            self.Q[key] = np.maximum(self.Q[key] + (u - baseline), 0.0)
            # Linear averaging (weight by iteration t)
            self.S[key] += t * reach[key][:, None] * s

    # --- best-response exploitability (convergence metric) ------------------

    def _avg_strategies(self) -> Dict[str, np.ndarray]:
        out = {}
        for key, s in self.S.items():
            total = s.sum(axis=1, keepdims=True)
            n = s.shape[1]
            out[key] = np.where(total > 0, s / np.where(total > 0, total, 1.0),
                                np.full_like(s, 1.0 / n))
        return out

    def _values_and_br(self, avg: Dict[str, np.ndarray]):
        """Return (v_oop, v_ip, br_oop, br_ip) in bb under the average profile.

        v_* are values when both play the average profile; br_* are best-response
        values against the average opponent. Uses per-infoset cfvs so IP's value
        is summed over its disjoint infosets.
        """
        C, CT = self.C, self.CT
        P0, bs, bl = self.P0, self.bs, self.bl
        w_o, w_i = self.w_o, self.w_i

        s_root = avg["root"]; s_ipc = avg["ip_vs_check"]
        s_ovs = avg["oop_vs_s"]; s_ovl = avg["oop_vs_l"]
        s_ivs = avg["ip_vs_s"]; s_ivl = avg["ip_vs_l"]

        # ---- OOP-perspective cfvs (opp = IP average) ----
        u_bet_s = P0 * (C @ (w_i * s_ivs[:, 0])) + self._showdown_oop(bs, w_i * s_ivs[:, 1])
        u_bet_l = P0 * (C @ (w_i * s_ivl[:, 0])) + self._showdown_oop(bl, w_i * s_ivl[:, 1])
        u_check_ipcheck = self._showdown_oop(0.0, w_i * s_ipc[:, 0])
        # OOP's own response nodes (value under avg vs BR):
        cfv_ovs_call = self._showdown_oop(bs, w_i * s_ipc[:, 1])
        cfv_ovl_call = self._showdown_oop(bl, w_i * s_ipc[:, 2])
        node_ovs_avg = s_ovs[:, 1] * cfv_ovs_call          # fold cfv = 0
        node_ovl_avg = s_ovl[:, 1] * cfv_ovl_call
        node_ovs_br = np.maximum(0.0, cfv_ovs_call)        # fold=0 vs call
        node_ovl_br = np.maximum(0.0, cfv_ovl_call)
        u_check_avg = u_check_ipcheck + node_ovs_avg + node_ovl_avg
        u_check_br = u_check_ipcheck + node_ovs_br + node_ovl_br
        u_root_avg = np.stack([u_check_avg, u_bet_s, u_bet_l], axis=1)

        v_oop = float((w_o * (s_root * u_root_avg).sum(axis=1)).sum())
        br_oop_val = np.maximum.reduce([u_check_br, u_bet_s, u_bet_l])
        br_oop = float((w_o * br_oop_val).sum())

        # ---- IP-perspective cfvs (opp = OOP average) ----
        ro_c = w_o * s_root[:, 0]
        ro_s = w_o * s_root[:, 1]
        ro_l = w_o * s_root[:, 2]
        # ip_vs_s / ip_vs_l (fold cfv = 0)
        u_ivs = np.stack([np.zeros(self.n_i), self._showdown_ip(bs, ro_s)], axis=1)
        u_ivl = np.stack([np.zeros(self.n_i), self._showdown_ip(bl, ro_l)], axis=1)
        # ip_vs_check
        u_ipc_check = self._showdown_ip(0.0, ro_c)
        u_ipc_bet_s = P0 * (CT @ (ro_c * s_ovs[:, 0])) + self._showdown_ip(bs, ro_c * s_ovs[:, 1])
        u_ipc_bet_l = P0 * (CT @ (ro_c * s_ovl[:, 0])) + self._showdown_ip(bl, ro_c * s_ovl[:, 1])
        u_ipc = np.stack([u_ipc_check, u_ipc_bet_s, u_ipc_bet_l], axis=1)

        # IP value under avg = sum over its disjoint infosets.
        v_ip = float(
            (w_i * (s_ipc * u_ipc).sum(axis=1)).sum()
            + (w_i * (s_ivs * u_ivs).sum(axis=1)).sum()
            + (w_i * (s_ivl * u_ivl).sum(axis=1)).sum()
        )
        br_ip = float(
            (w_i * u_ipc.max(axis=1)).sum()
            + (w_i * u_ivs.max(axis=1)).sum()
            + (w_i * u_ivl.max(axis=1)).sum()
        )
        return v_oop, v_ip, br_oop, br_ip

    def _exploitability_bb(self, avg: Dict[str, np.ndarray]) -> float:
        """Nash exploitability (bb): total best-response gain over the average
        profile. Tends to 0 at equilibrium."""
        v_oop, v_ip, br_oop, br_ip = self._values_and_br(avg)
        return (br_oop - v_oop) + (br_ip - v_ip)

    # --- driver -------------------------------------------------------------

    def solve(self, iterations: int, checkpoints: int = 12) -> SolveResult:
        tracemalloc.start()
        t0 = time.time()
        curve: List[Tuple[int, float]] = []
        check_every = max(1, iterations // checkpoints)
        for t in range(1, iterations + 1):
            self._iteration(t)
            if t % check_every == 0 or t == iterations:
                avg = self._avg_strategies()
                curve.append((t, self._exploitability_bb(avg)))
        runtime = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        avg = self._avg_strategies()
        action_ev, root_ev = self._report_action_evs(avg)
        expl = self._exploitability_bb(avg)
        return SolveResult(
            strategies=avg,
            action_ev=action_ev,
            action_labels={
                "root": ROOT_ACTIONS, "ip_vs_check": IPCHECK_ACTIONS,
                "oop_vs_s": RESPONSE_ACTIONS, "oop_vs_l": RESPONSE_ACTIONS,
                "ip_vs_s": RESPONSE_ACTIONS, "ip_vs_l": RESPONSE_ACTIONS,
            },
            root_ev_oop_bb=root_ev,
            root_ev_oop_pct_pot=100.0 * root_ev / self.P0,
            pot_bb=self.P0,
            iterations=iterations,
            exploitability_curve=curve,
            final_exploitability_bb=expl,
            final_exploitability_pct_pot=100.0 * expl / self.P0,
            runtime_sec=runtime,
            peak_mem_mb=peak / 1e6,
        )

    def _report_action_evs(self, avg: Dict[str, np.ndarray]):
        """Per-hand action EVs (bb) conditional on holding the hand, plus root EV."""
        C, CT = self.C, self.CT
        P0, bs, bl = self.P0, self.bs, self.bl
        w_o, w_i = self.w_o, self.w_i
        s_root = avg["root"]; s_ipc = avg["ip_vs_check"]
        s_ovs = avg["oop_vs_s"]; s_ovl = avg["oop_vs_l"]
        s_ivs = avg["ip_vs_s"]; s_ivl = avg["ip_vs_l"]

        # OOP root action cfvs (unconditional, opp-reach-weighted)
        u_bet_s = P0 * (C @ (w_i * s_ivs[:, 0])) + self._showdown_oop(bs, w_i * s_ivs[:, 1])
        u_bet_l = P0 * (C @ (w_i * s_ivl[:, 0])) + self._showdown_oop(bl, w_i * s_ivl[:, 1])
        u_check_ipcheck = self._showdown_oop(0.0, w_i * s_ipc[:, 0])
        node_ovs = s_ovs[:, 1] * self._showdown_oop(bs, w_i * s_ipc[:, 1])
        node_ovl = s_ovl[:, 1] * self._showdown_oop(bl, w_i * s_ipc[:, 2])
        u_check = u_check_ipcheck + node_ovs + node_ovl
        u_root = np.stack([u_check, u_bet_s, u_bet_l], axis=1)

        opp_mass_o = C @ w_i            # compatible IP mass per OOP hand
        opp_mass_o = np.where(opp_mass_o > 1e-12, opp_mass_o, 1.0)
        root_action_ev = u_root / opp_mass_o[:, None]

        # IP-vs-check action cfvs
        ro_c = w_o * s_root[:, 0]
        u_ipc_check = self._showdown_ip(0.0, ro_c)
        u_ipc_bet_s = P0 * (CT @ (ro_c * s_ovs[:, 0])) + self._showdown_ip(bs, ro_c * s_ovs[:, 1])
        u_ipc_bet_l = P0 * (CT @ (ro_c * s_ovl[:, 0])) + self._showdown_ip(bl, ro_c * s_ovl[:, 1])
        u_ipc = np.stack([u_ipc_check, u_ipc_bet_s, u_ipc_bet_l], axis=1)
        opp_mass_i = CT @ w_o
        opp_mass_i = np.where(opp_mass_i > 1e-12, opp_mass_i, 1.0)
        ipc_action_ev = u_ipc / opp_mass_i[:, None]

        # Root EV to OOP (bb): expected over OOP hands of chosen-strategy value.
        root_ev = float((w_o * (s_root * u_root).sum(axis=1)).sum())

        action_ev = {"root": root_action_ev, "ip_vs_check": ipc_action_ev}
        return action_ev, root_ev
