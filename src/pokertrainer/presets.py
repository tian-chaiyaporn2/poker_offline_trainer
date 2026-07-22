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

# --- SB-vs-BB single-raised pot (blind vs blind). Postflop the SB acts first
#     (OOP) and the BB acts last (IP) — note the OOP player is the PRE-FLOP
#     AGGRESSOR here, inverting the BTN-vs-BB dynamic. Ranges are v1 engineering
#     estimates for a poker reviewer to validate (like BB_SRP/BTN_SRP above).
# SB opening (raise-first-in) range that goes to the flop as the raiser (OOP):
SB_SRP: Dict[str, float] = {c: 1.0 for c in [
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "QJs", "QTs", "Q9s", "Q8s",
    "JTs", "J9s", "J8s", "T9s", "T8s", "98s", "97s", "87s", "86s", "76s", "65s", "54s",
    "AKo", "AQo", "AJo", "ATo", "A9o", "KQo", "KJo", "KTo", "QJo", "QTo", "JTo", "T9o", "98o",
]}
# BB flat-call range vs the SB open (IP) — defends wide with position + odds:
BB_vs_SB: Dict[str, float] = {c: 1.0 for c in [
    "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "QJs", "QTs", "Q9s", "Q8s", "JTs", "J9s", "J8s",
    "T9s", "T8s", "97s", "98s", "87s", "86s", "76s", "75s", "65s", "54s", "43s",
    "AJo", "ATo", "A9o", "KQo", "KJo", "KTo", "QJo", "QTo", "JTo", "T9o", "98o", "87o",
]}

# --- CO-vs-BB single-raised pot. CO opens, folds to BB, BB calls. Postflop the BB
#     acts first (OOP) and the CO acts last (IP) — same shape as BTN-vs-BB but the CO
#     opens a touch tighter (more players still behind). v1 estimates to validate.
CO_SRP: Dict[str, float] = {c: 1.0 for c in [  # CO open (~26%)
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "T8s",
    "98s", "87s", "76s", "65s", "54s",
    "AKo", "AQo", "AJo", "KQo", "KJo", "QJo",
]}
BB_vs_CO: Dict[str, float] = {c: 1.0 for c in [  # BB defends vs CO open (a touch tighter than vs BTN)
    "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "T8s",
    "98s", "87s", "76s", "65s", "54s",
    "AJo", "ATo", "KQo", "KJo", "KTo", "QJo", "QTo", "JTo",
]}

# --- BTN-vs-SB single-raised pot. BTN opens, BB folds, SB flat-calls. Postflop the
#     SB acts first (OOP caller) and the BTN acts last (IP) — the SB flats a medium
#     range (strong hands 3-bet instead). v1 estimates to validate.
SB_vs_BTN: Dict[str, float] = {c: 1.0 for c in [  # SB flat-call vs BTN open (OOP)
    "99", "88", "77", "66", "55", "44", "33", "22",
    "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KTs", "KJs", "QTs", "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
    "AJo", "KQo", "KJo", "QJo",
]}

# --- UTG-vs-BB and HJ-vs-BB single-raised pots. Completes the "any position opens, BB
#     defends" family (UTG tightest -> BTN widest). Same shape as CO/BTN-vs-BB (BB is OOP,
#     opener is IP); the opener tightens and BB defends a touch tighter for earlier seats.
#     v1 engineering estimates for a reviewer to validate.
UTG_SRP: Dict[str, float] = {c: 1.0 for c in [  # UTG open (~15%, tightest)
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AQs", "AJs", "ATs", "A5s", "A4s",
    "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s",
    "AKo", "AQo", "AJo", "KQo",
]}
BB_vs_UTG: Dict[str, float] = {c: 1.0 for c in [  # BB defends vs UTG (tightest defense)
    "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "98s", "87s", "76s", "65s",
    "AJo", "ATo", "KQo", "KJo", "QJo", "JTo",
]}
HJ_SRP: Dict[str, float] = {c: 1.0 for c in [  # HJ open (~18%, between UTG and CO)
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s",
    "98s", "87s", "76s", "65s",
    "AKo", "AQo", "AJo", "KQo", "KJo", "QJo",
]}
BB_vs_HJ: Dict[str, float] = {c: 1.0 for c in [  # BB defends vs HJ (between vs-UTG and vs-CO)
    "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "T8s",
    "98s", "87s", "76s", "65s",
    "AJo", "ATo", "KQo", "KJo", "QJo", "JTo",
]}

# --- BTN-vs-BB 3-BET pot. BTN opens, BB 3-bets, BTN calls. Postflop the BB is OOP
#     (the 3-bettor / aggressor, acts first) and the BTN is IP (caller). The whole
#     point is the LOW SPR: pot ~20bb with ~88bb behind (SPR ~4.4) vs the deep ~18
#     SPR of a single-raised pot — so stacks get in by the river (the eff_stack cap
#     models this). Ranges are v1 polarized estimates to validate.
BB_3BET: Dict[str, float] = {c: 1.0 for c in [  # BB 3-bets vs BTN open (polarized, OOP)
    "AA", "KK", "QQ", "JJ", "TT",                      # value pairs
    "AKs", "AQs", "AKo",                               # value broadways
    "A5s", "A4s", "A3s", "A2s",                        # nut-blocker bluffs
    "KJs", "QJs", "JTs", "76s", "65s",                 # suited bluffs
]}
BTN_vs_3BET: Dict[str, float] = {c: 1.0 for c in [  # BTN calls the 3-bet (IP, flats)
    "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ",
    "AKs", "AQs", "AJs", "ATs",
    "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s",
]}

# Scenario registry — parameterizes the content pipeline (--scenario). Each entry
# names the OOP (first-to-act) and IP (last-to-act) ranges + position labels; the
# solver is scenario-agnostic (it just takes the two ranges). Single-raised pots have
# no eff_stack (deep, SPR ~18); the 3-bet pot sets eff_stack so bets hit all-in (SPR ~4).
SCENARIOS: Dict[str, Dict] = {
    "btn_vs_bb_srp": {"oop_range": BB_SRP, "ip_range": BTN_SRP,
                      "oop_pos": "BB", "ip_pos": "BTN", "pot": 5.5, "bet_frac": 0.66,
                      "label": "BTN opens, BB calls (single-raised pot)"},
    "sb_vs_bb_srp": {"oop_range": SB_SRP, "ip_range": BB_vs_SB,
                     "oop_pos": "SB", "ip_pos": "BB", "pot": 6.0, "bet_frac": 0.66,
                     "label": "SB opens, BB calls (blind vs blind, single-raised pot)"},
    "co_vs_bb_srp": {"oop_range": BB_vs_CO, "ip_range": CO_SRP,
                     "oop_pos": "BB", "ip_pos": "CO", "pot": 5.5, "bet_frac": 0.66,
                     "label": "CO opens, BB calls (single-raised pot)"},
    "utg_vs_bb_srp": {"oop_range": BB_vs_UTG, "ip_range": UTG_SRP,
                      "oop_pos": "BB", "ip_pos": "UTG", "pot": 5.5, "bet_frac": 0.66,
                      "label": "UTG opens, BB calls (single-raised pot)"},
    "hj_vs_bb_srp": {"oop_range": BB_vs_HJ, "ip_range": HJ_SRP,
                     "oop_pos": "BB", "ip_pos": "HJ", "pot": 5.5, "bet_frac": 0.66,
                     "label": "HJ opens, BB calls (single-raised pot)"},
    "btn_vs_sb_srp": {"oop_range": SB_vs_BTN, "ip_range": BTN_SRP,
                      "oop_pos": "SB", "ip_pos": "BTN", "pot": 5.5, "bet_frac": 0.66,
                      "label": "BTN opens, SB calls (single-raised pot)"},
    "btn_bb_3bet": {"oop_range": BB_3BET, "ip_range": BTN_vs_3BET,
                    "oop_pos": "BB", "ip_pos": "BTN", "pot": 20.0, "bet_frac": 0.66,
                    "eff_stack": 88.0,
                    "label": "BTN opens, BB 3-bets, BTN calls (3-bet pot, low SPR)"},
}


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
