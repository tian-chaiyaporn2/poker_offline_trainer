"""Permissive-solver runner (Deliverable 3, PRD §5 Step 3) — MIT.

Runs one canonical scenario end to end through our own solver and returns a
JSON-serialisable result with everything the PRD requires:
action frequencies by hand, per-action EV, overall range frequencies, root EV,
convergence info, and runtime / memory.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .cards import hand_str
from .scenario import Scenario, load_scenario, validate_solution
from .showdown import equity_matrix
from .solver import FlopSolver


def _range_frequencies(strategy: np.ndarray, weights: np.ndarray) -> List[float]:
    w = weights / weights.sum()
    return [round(float(x), 6) for x in (strategy * w[:, None]).sum(axis=0)]


def run_scenario(raw: Dict) -> Dict:
    scn: Scenario = load_scenario(raw)
    equity, compat = equity_matrix(scn.board, scn.oop_combos, scn.ip_combos)
    solver = FlopSolver(
        equity, compat, scn.w_oop, scn.w_ip,
        pot_bb=scn.pot_bb, small_frac=scn.small_frac, large_frac=scn.large_frac,
    )
    res = solver.solve(iterations=scn.iterations)
    validate_solution(res.strategies)

    root_labels = res.action_labels["root"]
    root_strat = res.strategies["root"]
    root_ev = res.action_ev["root"]

    per_hand: List[Dict] = []
    for idx, combo in enumerate(scn.oop_combos):
        per_hand.append({
            "hand": hand_str(combo),
            "weight": float(scn.w_oop[idx]),
            "strategy": {a: float(root_strat[idx, k]) for k, a in enumerate(root_labels)},
            "action_ev_bb": {a: float(root_ev[idx, k]) for k, a in enumerate(root_labels)},
        })

    return {
        "scenario_id": scn.id,
        "solver": "pokertrainer_cfr_plus",
        "board": list(scn.raw["board"]),
        "acting_player": "BB",
        "actions": root_labels,
        "pot_bb": scn.pot_bb,
        "n_oop_combos": len(scn.oop_combos),
        "n_ip_combos": len(scn.ip_combos),
        "range_action_frequencies": dict(zip(root_labels, _range_frequencies(root_strat, scn.w_oop))),
        "root_ev_oop_bb": round(res.root_ev_oop_bb, 6),
        "root_ev_oop_pct_pot": round(res.root_ev_oop_pct_pot, 4),
        "convergence": {
            "algorithm": "cfr_plus",
            "iterations": res.iterations,
            "final_exploitability_bb": round(res.final_exploitability_bb, 8),
            "final_exploitability_pct_pot": round(res.final_exploitability_pct_pot, 6),
            "exploitability_curve": [[it, round(e, 8)] for it, e in res.exploitability_curve],
        },
        "resources": {
            "runtime_sec": round(res.runtime_sec, 3),
            "peak_mem_mb": round(res.peak_mem_mb, 2),
        },
        "per_hand": per_hand,
    }
