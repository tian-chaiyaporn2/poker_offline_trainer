"""Solver → recommendation → training translation contracts (MIT)."""

import sqlite3

from pokertrainer.content_pack import (
    DEFAULT_CONFIG, build_pack, refresh_pack_lessons, resign_pack,
)
from pokertrainer.content_yield import CLEAR_SEP_PCT
from pokertrainer.explanations import classify_reason, explain, freq_pct_ints
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
    base = {
        "node": "sb_first", "board_texture": ["unpaired", "rainbow", "high_card", "disconnected"],
        "hand_category": "air", "preferred": "check", "reach_mass": 1.0,
        "freq": {"check": 0.9, "bet": 0.1}, "accepted": True, "reason": "realization",
        "board": "As7h2d", "hand": "3c2c",
    }
    small_pot = {**base, "ev": {"check": 1.0, "bet": 0.0}, "pot_bb": 5.5, "hand": "3c2c"}
    large_pot = {**base, "ev": {"check": 1.0, "bet": 0.0}, "pot_bb": 100.0, "hand": "4c2c",
                 "board": "Kd7h2d"}
    scored = {r["hand"]: r for r in score_records([small_pot, large_pot], pot=5.5)}
    assert scored["3c2c"]["priority_parts"]["impact_pct"] > scored["4c2c"]["priority_parts"]["impact_pct"]


def test_mixed_requires_all_actions_near_indifferent(tmp_path):
    pot = 5.5
    rec = {
        "board": "As7h2d", "board_texture": ["unpaired", "rainbow", "high_card", "disconnected"],
        "board_favored": None, "node": "bb_vs_bet", "acting_player": "BB",
        "decision_type": "vs_bet", "hand": "7h7c", "hand_category": "weak_pair",
        "actions": ["fold", "call", "raise"],
        "ev": {"fold": 0.0, "call": 0.985, "raise": 1.008},
        "freq": {"fold": 0.05, "call": 0.4, "raise": 0.55},
        "preferred": "raise",
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
    """OOP facing a bet must not be labeled acts_first (regression from SB-vs-BB wiring)."""
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    path = root / "demo" / "build_trainer.py"
    spec = importlib.util.spec_from_file_location("build_trainer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = {
        "board": "As7h2d", "hand": "JhJc", "node": "bb_vs_bet",
        "acting_player": "BB",
        "actions": '["fold","call"]',
        "ev": '{"fold":0.0,"call":1.0}',
        "freq": '{"fold":0.2,"call":0.8}',
        "preferred_action": "call",
        "action_grades": '{"fold":"major_error","call":"best"}',
        "reason": "value_call", "headline": "x", "detail": '["d"]',
        "mixed": 0,
    }
    q = mod._to_q(row, oop_pos="BB", ip_pos="BTN")
    assert q["acts_first"] is False and q["is_oop"] is True
    q2 = mod._to_q({**row, "node": "bb_first", "actions": '["check","bet"]',
                    "ev": '{"check":1.0,"bet":0.5}', "freq": '{"check":0.8,"bet":0.2}',
                    "preferred_action": "check",
                    "action_grades": '{"check":"best","bet":"costly"}',
                    "reason": "pot_control"}, oop_pos="BB", ip_pos="BTN")
    assert q2["acts_first"] is True
    # SB-vs-BB: BB is IP — still not acts_first on vs_bet, and is_oop False.
    q3 = mod._to_q({**row, "acting_player": "BB", "node": "bb_vs_bet"},
                   oop_pos="SB", ip_pos="BB")
    assert q3["acts_first"] is False and q3["is_oop"] is False


def test_freq_pct_ints_sum_to_100():
    fp = freq_pct_ints({"fold": 0.694, "call": 0.296, "raise": 0.01},
                       order=["fold", "call", "raise"])
    assert sum(fp.values()) == 100
    assert all(v >= 0 for v in fp.values())
    thirds = freq_pct_ints({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, order=["a", "b", "c"])
    assert sum(thirds.values()) == 100
    assert set(thirds.values()) <= {33, 34}


def test_first_action_raise_does_not_become_pot_control():
    r = {
        "node": "bb_first", "acting_player": "BB", "hand": "AhKh",
        "hand_category": "weak_pair", "preferred": "raise",
        "actions": ["check", "bet"], "ev": {"check": 0.0, "bet": 1.0},
        "freq": {"check": 0.1, "bet": 0.9}, "mixed": False,
        "board_texture": ["rainbow"], "decision_type": "first_action",
    }
    assert classify_reason(r) == "fold"


def test_mixed_detail_lists_all_three_actions():
    r = {
        "node": "bb_vs_bet", "acting_player": "BB", "hand": "7h7c",
        "hand_category": "weak_pair", "preferred": "call",
        "actions": ["fold", "call", "raise"],
        "ev": {"fold": 1.0, "call": 1.0, "raise": 1.0},
        "freq": {"fold": 0.33, "call": 0.34, "raise": 0.33},
        "ev_sep_pct": 0.0, "mixed": True,
        "board_texture": ["rainbow"], "decision_type": "vs_bet",
    }
    e = explain(r)
    assert e["reason"] == "mixed"
    assert "fold" in e["detail"][0] and "call" in e["detail"][0] and "raise" in e["detail"][0]
    parts = e["detail"][1].replace("Solver frequency: ", "").split(", ")
    pcts = [int(p.rsplit(" ", 1)[1].rstrip("%")) for p in parts]
    assert sum(pcts) == 100


def test_river_realization_headline_has_no_free_card():
    r = {
        "node": "bb_first", "acting_player": "BB", "hand": "Ah5h",
        "board": "Th9h8d2c7c", "hand_category": "air", "preferred": "check",
        "actions": ["check", "bet"], "ev": {"check": 0.2, "bet": 0.0},
        "freq": {"check": 0.9, "bet": 0.1}, "ev_sep_pct": 3.6, "mixed": False,
        "board_texture": ["rainbow"], "decision_type": "first_action",
    }
    e = explain(r)
    assert e["reason"] == "realization"
    assert "free card" not in e["headline"].lower()
    assert "improve" not in e["headline"].lower()
