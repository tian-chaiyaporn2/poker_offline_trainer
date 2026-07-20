"""Scenario loading, validation, and range expansion (MIT).

Turns a canonical scenario dict (docs/scenario_format.md) into concrete inputs
for the solver, and enforces the technical validation rules from PRD §6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .cards import parse_cards
from .ranges import expand_range

Combo = Tuple[int, int]


class ValidationError(Exception):
    pass


def _require_finite(name: str, value) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass
class Scenario:
    raw: Dict
    board: List[int]
    oop_combos: List[Combo]
    ip_combos: List[Combo]
    w_oop: np.ndarray
    w_ip: np.ndarray
    pot_bb: float
    small_frac: float
    large_frac: float
    iterations: int

    @property
    def id(self) -> str:
        return self.raw["id"]


def load_scenario(raw: Dict) -> Scenario:
    board = parse_cards(raw["board"])

    # PRD §6: no duplicate or impossible cards on the board.
    if len(set(board)) != len(board):
        raise ValidationError(f"duplicate board cards in {raw['id']}")
    if len(board) != 3:
        raise ValidationError(f"flop must be 3 cards in {raw['id']}")

    # FlopSolver implements a fixed tree. Reject scenario fields that would
    # silently describe a different game than the one being solved.
    _SUPPORTED_ALLOWED = {"check", "bet_small", "bet_large", "call", "fold"}
    _SUPPORTED_RAISE = {"no_raise_v1"}
    _SUPPORTED_POSITIONS = {"ip": "BTN", "oop": "BB"}
    acting = raw.get("acting_player", "BB")
    if acting != "BB":
        raise ValidationError(
            f"acting_player={acting!r} unsupported; FlopSolver is BB-first only"
        )
    positions = raw.get("positions", _SUPPORTED_POSITIONS)
    if positions != _SUPPORTED_POSITIONS:
        raise ValidationError(
            f"positions={positions!r} unsupported; FlopSolver is "
            f"{_SUPPORTED_POSITIONS} only (refusing a mismatched matchup label "
            f"while still loading BB/BTN ranges)"
        )
    actions = raw.get("actions", {})
    allowed = set(actions.get("allowed", list(_SUPPORTED_ALLOWED)))
    if not allowed.issubset(_SUPPORTED_ALLOWED):
        raise ValidationError(
            f"unsupported actions {sorted(allowed - _SUPPORTED_ALLOWED)}; "
            f"FlopSolver supports {sorted(_SUPPORTED_ALLOWED)}"
        )
    # FlopSolver always solves the full fixed tree — a partial `allowed` list
    # would silently describe a different game than the one being solved.
    if allowed != _SUPPORTED_ALLOWED:
        raise ValidationError(
            f"actions.allowed must be the full FlopSolver set "
            f"{sorted(_SUPPORTED_ALLOWED)}; got {sorted(allowed)}"
        )
    raise_rule = actions.get("raise_rule", "no_raise_v1")
    if raise_rule not in _SUPPORTED_RAISE:
        raise ValidationError(
            f"raise_rule={raise_rule!r} unsupported; use one of {sorted(_SUPPORTED_RAISE)}"
        )
    tree = raw.get("tree", {})
    streets = tree.get("streets", ["flop"])
    if streets != ["flop"]:
        raise ValidationError(
            f"tree.streets={streets!r} unsupported by FlopSolver "
            f"(flop-only realized-equity model); use BatchedCFR for multi-street"
        )

    oop_wc = expand_range(raw["ranges"]["BB"]["combos"], board)
    ip_wc = expand_range(raw["ranges"]["BTN"]["combos"], board)
    if not oop_wc or not ip_wc:
        raise ValidationError(f"empty range after board removal in {raw['id']}")

    oop_combos = [c for c, _ in oop_wc]
    ip_combos = [c for c, _ in ip_wc]
    w_oop = np.array([w for _, w in oop_wc], dtype=np.float64)
    w_ip = np.array([w for _, w in ip_wc], dtype=np.float64)

    sizes = raw["actions"]["bet_sizes_pct_pot"]
    pot_bb = _require_finite("pot_bb", raw["pot_bb"])
    small = _require_finite("bet_sizes_pct_pot.small", sizes["small"])
    large = _require_finite("bet_sizes_pct_pot.large", sizes["large"])
    iterations = int(_require_finite("solver.iterations", raw["solver"]["iterations"]))
    if iterations <= 0:
        raise ValidationError(f"solver.iterations must be > 0, got {iterations}")
    if pot_bb <= 0:
        raise ValidationError(f"pot_bb must be > 0, got {pot_bb}")
    return Scenario(
        raw=raw,
        board=board,
        oop_combos=oop_combos,
        ip_combos=ip_combos,
        w_oop=w_oop,
        w_ip=w_ip,
        pot_bb=pot_bb,
        small_frac=small / 100.0,
        large_frac=large / 100.0,
        iterations=iterations,
    )


def validate_solution(strategies: Dict[str, np.ndarray], tol: float = 1e-6) -> None:
    """PRD §6 technical validation on solver output.

    - No illegal actions (arrays have the right number of columns handled by
      the solver's fixed tree).
    - Action probabilities total ~100%.
    """
    for key, arr in strategies.items():
        sums = arr.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            bad = np.where(~np.isclose(sums, 1.0, atol=1e-4))[0]
            raise ValidationError(
                f"infoset {key}: {len(bad)} rows do not sum to 1 (e.g. {sums[bad[:3]]})"
            )
        if (arr < -tol).any():
            raise ValidationError(f"infoset {key}: negative probability")
