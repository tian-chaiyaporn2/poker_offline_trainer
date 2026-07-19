"""Preset ranges and the 12 POC flop boards (MIT).

Ranges are condensed but poker-sound HU single-raised-pot ranges:
- BTN (in position / preflop raiser) has a slight range advantage at the top.
- BB (out of position / caller) defends wide but caps the very top (3-bets
  removed) — mirroring a real single-raised pot.

They are intentionally moderate in size to keep one-time generation practical
(PRD §7 "compute practical for a larger library"). Exact ranges are a POC
simplification; preflop solving is out of scope (PRD §8).
"""

from __future__ import annotations

from typing import Dict, List

# --- BTN (IP) single-raised-pot range -------------------------------------
BTN_SRP: Dict[str, float] = {c: 1.0 for c in [
    # pocket pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # suited aces (broad)
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    # other suited
    "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
    # offsuit broadways
    "AKo", "AQo", "AJo", "ATo", "KQo", "KJo", "QJo",
]}

# --- BB (OOP) single-raised-pot calling range -----------------------------
BB_SRP: Dict[str, float] = {c: 1.0 for c in [
    # pocket pairs (QQ+ mostly 3-bets, so cap at JJ here)
    "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # suited aces (AKs/AQs partly 3-bet -> capped at AJs)
    "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    # other suited
    "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "T8s",
    "98s", "97s", "87s", "86s", "76s", "65s", "54s",
    # offsuit
    "AJo", "ATo", "KQo", "KJo", "KTo", "QJo", "QTo", "JTo", "T9o", "98o",
]}


# --- The 12 POC boards (PRD §4: dry, connected, paired, two-suit, monotone,
#     low-card must all be represented) -----------------------------------
BOARDS: List[Dict] = [
    {"board": "As7h2d", "categories": ["dry", "high_card", "rainbow"]},
    {"board": "Kd9c4h", "categories": ["dry", "high_card", "rainbow"]},
    {"board": "Th9h8d", "categories": ["connected", "two_suit", "draw_heavy"]},
    {"board": "8s7s6c", "categories": ["connected", "two_suit", "low"]},
    {"board": "KsKd6h", "categories": ["paired", "high_card"]},
    {"board": "5c5d2h", "categories": ["paired", "low"]},
    {"board": "AhQh7c", "categories": ["two_suit", "high_card", "draw_heavy"]},
    {"board": "Jd8d3s", "categories": ["two_suit", "mid"]},
    {"board": "Qh8h3h", "categories": ["monotone"]},
    {"board": "7s5s2s", "categories": ["monotone", "low"]},
    {"board": "6h4d2c", "categories": ["dry", "low", "rainbow"]},
    {"board": "JcTs9d", "categories": ["connected", "high_card", "rainbow"]},
    # Coverage additions (indices 12-16): the highest-frequency textures the
    # priority scorer flagged as uncovered by boards 0-11 (low + paired
    # disconnected/connected families). Solve as a targeted second run.
    {"board": "9h6h2c", "categories": ["two_suit", "low", "dry"]},
    {"board": "QhQc6h", "categories": ["paired", "two_suit", "high_card", "dry"]},
    {"board": "9s6d2c", "categories": ["rainbow", "low", "dry"]},
    {"board": "8h8c7h", "categories": ["paired", "two_suit", "low", "connected"]},
    {"board": "JhJcTd", "categories": ["paired", "rainbow", "high_card", "connected"]},
]


def board_id(board: str) -> str:
    return f"srp_btn_bb_100bb_flop_{board}"


def build_scenario(entry: Dict, iterations: int = 1500, seed: int = 12345) -> Dict:
    """Assemble a canonical scenario dict (see docs/scenario_format.md)."""
    board = entry["board"]
    return {
        "id": board_id(board),
        "game": "nlhe",
        "format": "heads_up",
        "spot": "single_raised_pot",
        "positions": {"ip": "BTN", "oop": "BB"},
        "stacks_bb": 100,
        "rake": 0,
        "board": [board[i:i + 2] for i in range(0, len(board), 2)],
        "board_categories": entry["categories"],
        "pot_bb": 5.5,
        "effective_stack_bb": 97.5,
        "ranges": {
            "BTN": {"notation": "range_v1", "combos": BTN_SRP},
            "BB": {"notation": "range_v1", "combos": BB_SRP},
        },
        "acting_player": "BB",
        "actions": {
            "bet_sizes_pct_pot": {"small": 33, "large": 75},
            "allowed": ["check", "bet_small", "bet_large", "call", "fold"],
            "raise_rule": "no_raise_v1",
        },
        "tree": {"streets": ["flop"], "model": "flop_only_realized_equity"},
        "solver": {
            "algorithm": "cfr_plus",
            "iterations": iterations,
            "convergence_target_exploitability_pct_pot": 1.0,
            "seed": seed,
        },
    }
