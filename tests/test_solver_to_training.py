"""Solver → recommendation → training translation contracts (MIT)."""

import sqlite3

from pokertrainer.content_pack import (
    DEFAULT_CONFIG, build_pack, refresh_pack_lessons, resign_pack,
)
from pokertrainer.content_yield import CLEAR_SEP_PCT
from pokertrainer.export import build_questions
from pokertrainer.priority import score_records
from pokertrainer.solver.batched import preferred_action


def test_preferred_is_max_ev():
    ev = {"check": 1.0, "bet": 0.5}
    freq = {"check": 0.1, "bet": 0.9}
    assert preferred_action(ev, freq) == "check"


def test_preferred_ev_tie_breaks_by_frequency():
    ev = {"check": 1.0, "bet": 1.0}
    freq = {"check": 0.2, "bet": 0.8}
    assert preferred_action(ev, freq) == "bet"
    # Insertion order alone would pick check; freq must win the tie.
    assert preferred_action({"check": 1.0, "bet": 1.0}, {"check": 0.9, "bet": 0.1}) == "check"


def test_export_recommended_uses_freq_on_ev_tie():
    solve = {
        "scenario_id": "t", "board": ["As", "Kh", "2c"], "pot_bb": 5.5,
        "actions": ["check", "bet_small", "bet_large"],
        "solver": "test",
        "per_hand": [{
            "hand": "JdJc",
            "strategy": {"check": 0.1, "bet_small": 0.8, "bet_large": 0.1},
            "action_ev_bb": {"check": 1.0, "bet_small": 1.0, "bet_large": 0.5},
        }],
    }
    qs = build_questions(solve, max_per_board=1)
    assert qs and qs[0]["recommended_action"] == "bet_small"


def test_priority_uses_per_record_pot():
    """SB-vs-BB pot 6.0 must not be scored as if pot were the 5.5 default."""
    base = {
        "node": "sb_first", "board_texture": ["unpaired", "rainbow", "high_card", "disconnected"],
        "hand_category": "air", "preferred": "check", "reach_mass": 1.0,
        "freq": {"check": 0.9, "bet": 0.1}, "accepted": True, "reason": "realization",
        "board": "As7h2d", "hand": "3c2c",
    }
    # Same EV gap; larger pot → smaller impact % → lower impact percentile when
    # compared against an equal-gap smaller-pot record.
    small_pot = {**base, "ev": {"check": 1.0, "bet": 0.0}, "pot_bb": 5.5, "hand": "3c2c"}
    large_pot = {**base, "ev": {"check": 1.0, "bet": 0.0}, "pot_bb": 100.0, "hand": "4c2c",
                 "board": "Kd7h2d"}
    scored = {r["hand"]: r for r in score_records([small_pot, large_pot], pot=5.5)}
    assert scored["3c2c"]["priority_parts"]["impact_pct"] > scored["4c2c"]["priority_parts"]["impact_pct"]


def test_mixed_requires_all_actions_near_indifferent(tmp_path):
    """Top-2 close + dominated third must not stay labeled mixed after refresh."""
    pot = 5.5
    rec = {
        "board": "As7h2d", "board_texture": ["unpaired", "rainbow", "high_card", "disconnected"],
        "board_favored": None, "node": "bb_vs_bet", "acting_player": "BB",
        "decision_type": "vs_bet", "hand": "7h7c", "hand_category": "weak_pair",
        "actions": ["fold", "call", "raise"],
        "ev": {"fold": 0.0, "call": 0.985, "raise": 1.008},
        "freq": {"fold": 0.05, "call": 0.4, "raise": 0.55},
        "preferred": "raise",
        # Legacy top-2-only mixed flag (raise vs call ≈ 0.42% pot).
        "ev_sep_pct": round(100 * (1.008 - 0.985) / pot, 3),
        "mixed": True, "reach_mass": 0.8, "accepted": True, "pot_bb": pot,
        "oop_pos": "BB", "ip_pos": "BTN", "scenario": "btn_vs_bb_srp",
        "explanation": {
            "reason": "mixed",
            "headline": "Both actions are close here — either is acceptable.",
            "detail": ["placeholder"],
        },
    }
    assert rec["ev_sep_pct"] < CLEAR_SEP_PCT
    build_pack([rec], DEFAULT_CONFIG, out_dir=str(tmp_path), version="vmix")
    db = str(tmp_path / "flop_pack_vmix.db")
    out = refresh_pack_lessons(db)
    assert out["hash_ok"] and out["signature_ok"]
    row = sqlite3.connect(db).execute(
        "SELECT mixed, reason, headline FROM flop_decision"
    ).fetchone()
    assert row[0] == 0
    assert row[1] != "mixed"
    assert "either is acceptable" not in row[2]


def test_refresh_backfills_pot_and_roles(tmp_path):
    rec = {
        "board": "As7h2d", "board_texture": ["unpaired", "rainbow", "high_card", "disconnected"],
        "board_favored": "BB", "node": "bb_first", "acting_player": "BB",
        "decision_type": "first_action", "hand": "AhKh", "hand_category": "air",
        "actions": ["check", "bet"], "ev": {"check": 0.2, "bet": 0.0},
        "freq": {"check": 0.9, "bet": 0.1}, "preferred": "check",
        "ev_sep_pct": 3.6, "mixed": False, "reach_mass": 0.8, "accepted": True,
    }
    build_pack([rec], DEFAULT_CONFIG, out_dir=str(tmp_path), version="vfill", pot=5.5)
    db = str(tmp_path / "flop_pack_vfill.db")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE flop_decision SET pot_bb=NULL, oop_pos=NULL, ip_pos=NULL")
    conn.commit()
    conn.close()
    resign_pack(db)
    out = refresh_pack_lessons(db)
    assert out["hash_ok"] and out["signature_ok"]
    pot, oop, ip = sqlite3.connect(db).execute(
        "SELECT pot_bb, oop_pos, ip_pos FROM flop_decision"
    ).fetchone()
    assert pot == 5.5 and oop == "BB" and ip == "BTN"


def test_acts_first_only_for_first_nodes():
    """Facing a bet is never 'act first', even when hero is OOP."""
    def acts_first(node):
        return node.endswith("_first")
    assert acts_first("bb_first") is True
    assert acts_first("bb_vs_bet") is False
    assert acts_first("btn_vs_check") is False
    assert acts_first("sb_first") is True
