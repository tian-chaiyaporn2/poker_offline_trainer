"""Content pack build/verify/integrity tests (MIT)."""

import json
import os
import sqlite3

from pokertrainer.content_pack import build_pack, verify_pack, grade_action


def _rec(hand, node, hc, pref, ev0, ev1, actions):
    a0, a1 = actions
    return {
        "board": "As7h2d", "board_texture": ["rainbow", "high_card"], "board_favored": "BTN",
        "node": node, "acting_player": "BB" if node.startswith("bb") else "BTN",
        "decision_type": "first_action" if node in ("bb_first", "btn_vs_check") else "vs_bet",
        "hand": hand, "hand_category": hc, "actions": actions,
        "ev": {a0: ev0, a1: ev1}, "freq": {a0: 0.9, a1: 0.1},
        "preferred": pref, "ev_sep_pct": abs(ev0 - ev1) / 5.5 * 100,
        "mixed": False, "reach_mass": 0.8, "accepted": True,
    }


CONFIG = {"positions": {"ip": "BTN", "oop": "BB"}, "stack_bb": 100, "pot_bb": 5.5,
          "bet_pct_pot": 66, "rake": 0, "solver_model": "full_street_cfr_plus"}


def _pack(tmp_path):
    recs = [
        _rec("AhKh", "bb_first", "air", "check", 0.2, 0.0, ("check", "bet")),
        _rec("AsAd", "btn_vs_check", "strong_made", "bet", 2.5, 3.0, ("check", "bet")),
        _rec("7s7c", "bb_vs_bet", "weak_pair", "call", -0.1, 0.4, ("fold", "call")),
    ]
    return build_pack(recs, CONFIG, out_dir=str(tmp_path), version="vtest", pot=5.5)


def test_build_and_verify(tmp_path):
    rep = _pack(tmp_path)
    assert rep["record_count"] == "3"
    db = os.path.join(str(tmp_path), "flop_pack_vtest.db")
    v = verify_pack(db)
    assert v["hash_ok"] and v["signature_ok"]


def test_precomputed_grades(tmp_path):
    _pack(tmp_path)
    db = os.path.join(str(tmp_path), "flop_pack_vtest.db")
    conn = sqlite3.connect(db)
    grades = json.loads(conn.execute(
        "SELECT action_grades FROM flop_decision WHERE hand='AsAd'").fetchone()[0])
    conn.close()
    assert grades["bet"] == "best"          # bet is the top action
    assert grades["check"] in ("good", "acceptable", "costly")  # 0.5bb behind


def test_grade_thresholds():
    assert grade_action(3.0, 3.0, 5.5) == "best"          # zero regret
    assert grade_action(0.0, 3.0, 5.5) == "major_error"   # ~55% pot regret


def test_tamper_breaks_signature(tmp_path):
    _pack(tmp_path)
    db = os.path.join(str(tmp_path), "flop_pack_vtest.db")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE flop_decision SET preferred_action='bet' WHERE hand='AhKh'")
    conn.commit(); conn.close()
    v = verify_pack(db)
    assert not v["hash_ok"] and not v["signature_ok"]      # tamper detected


def test_gzip_and_report(tmp_path):
    rep = _pack(tmp_path)
    assert rep["gz_bytes"] < rep["db_bytes"]
    assert os.path.exists(os.path.join(str(tmp_path), "flop_pack_vtest.db.gz"))
    assert os.path.exists(os.path.join(str(tmp_path), "build_report_vtest.json"))
