"""Output normalization (Deliverable 5, PRD §5 Step 5) — MIT.

Both solvers are mapped into ONE common format so that formatting differences
(hand notation, suit order, action names, EV units, probability format) can never
masquerade as strategy disagreement. Canonical conventions (see
docs/scenario_format.md): hands high-card-first in 'shdc' notation, action names
[check, bet_small, bet_large], EV in big blinds, probabilities in [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .cards import hand_str

CANON_ACTIONS = ["check", "bet_small", "bet_large"]


@dataclass
class NormalizedSolve:
    scenario_id: str
    source: str                       # e.g. "cfr_plus" or "reference_cfr"
    board: List[str]
    actions: List[str]                # canonical order
    root_ev_oop_bb: float
    range_freqs: Dict[str, float]
    per_hand: Dict[str, Dict[str, Dict[str, float]]]  # hand -> {"strategy":{...},"ev":{...}}
    runtime_sec: float = 0.0
    peak_mem_mb: float = 0.0


def from_runner_output(out: Dict, source: str = "cfr_plus") -> NormalizedSolve:
    per_hand = {}
    for h in out["per_hand"]:
        per_hand[h["hand"]] = {
            "strategy": {a: float(h["strategy"][a]) for a in CANON_ACTIONS},
            "ev": {a: float(h["action_ev_bb"][a]) for a in CANON_ACTIONS},
        }
    return NormalizedSolve(
        scenario_id=out["scenario_id"],
        source=source,
        board=list(out["board"]),
        actions=list(CANON_ACTIONS),
        root_ev_oop_bb=float(out["root_ev_oop_bb"]),
        range_freqs={a: float(out["range_action_frequencies"][a]) for a in CANON_ACTIONS},
        per_hand=per_hand,
        runtime_sec=float(out["resources"]["runtime_sec"]),
        peak_mem_mb=float(out["resources"]["peak_mem_mb"]),
    )


def from_solver_arrays(scenario_id: str, board: List[str], oop_combos,
                       root_strategy: np.ndarray, root_action_ev: np.ndarray,
                       w_oop: np.ndarray, root_ev_bb: float, source: str,
                       runtime_sec: float = 0.0, peak_mem_mb: float = 0.0) -> NormalizedSolve:
    """Normalize either solver's raw OOP-root arrays into the common format."""
    w = w_oop / w_oop.sum()
    range_freqs = {a: float((root_strategy[:, k] * w).sum()) for k, a in enumerate(CANON_ACTIONS)}
    per_hand = {}
    for i, combo in enumerate(oop_combos):
        per_hand[hand_str(combo)] = {
            "strategy": {a: float(root_strategy[i, k]) for k, a in enumerate(CANON_ACTIONS)},
            "ev": {a: float(root_action_ev[i, k]) for k, a in enumerate(CANON_ACTIONS)},
        }
    return NormalizedSolve(
        scenario_id=scenario_id, source=source, board=list(board),
        actions=list(CANON_ACTIONS), root_ev_oop_bb=root_ev_bb,
        range_freqs=range_freqs, per_hand=per_hand, runtime_sec=runtime_sec,
        peak_mem_mb=peak_mem_mb,
    )
