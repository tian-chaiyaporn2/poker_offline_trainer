"""Explanation classifier poker-sanity tests (MIT)."""

from pokertrainer.explanations import classify_reason, explain


def rec(node, hand_cat, preferred, ev0, ev1, actions, texture=("rainbow", "high_card"),
        mixed=False):
    a0, a1 = actions
    return {
        "node": node, "acting_player": "BB" if node.startswith("bb") else "BTN",
        "hand": "AhKh", "hand_category": hand_cat, "preferred": preferred,
        "actions": actions, "ev": {a0: ev0, a1: ev1},
        "freq": {a0: 0.9, a1: 0.1}, "ev_sep_pct": 0.1 if mixed else 5.0,
        "board_texture": list(texture), "mixed": mixed,
    }


def test_value_bet():
    r = rec("bb_first", "strong_made", "bet", 3.0, 2.0, ("check", "bet"))
    assert classify_reason(r) == "value"


def test_air_bets_is_bluff():
    r = rec("btn_vs_check", "air", "bet", 1.0, 1.5, ("check", "bet"))
    assert classify_reason(r) == "bluff"


def test_draw_bets_is_semibluff():
    r = rec("bb_first", "draw", "bet", 1.0, 1.5, ("check", "bet"))
    assert classify_reason(r) == "semi_bluff"


def test_strong_check_is_trap():
    r = rec("bb_first", "strong_made", "check", 3.0, 2.0, ("check", "bet"))
    assert classify_reason(r) == "trap"


def test_pair_bets_wet_is_protection():
    r = rec("bb_first", "top_pair", "bet", 3.0, 2.0, ("check", "bet"),
            texture=("two_tone", "connected"))
    assert classify_reason(r) == "protection"


def test_weak_check_is_pot_control():
    r = rec("bb_first", "weak_pair", "check", 1.0, 0.5, ("check", "bet"))
    assert classify_reason(r) == "pot_control"


def test_fold_and_call_reasons():
    assert classify_reason(rec("bb_vs_bet", "air", "fold", 0.0, -1.0, ("fold", "call"))) == "fold"
    assert classify_reason(rec("bb_vs_bet", "top_pair", "call", -0.1, 2.0, ("fold", "call"))) == "value_call"
    assert classify_reason(rec("bb_vs_bet", "draw", "call", -0.1, 0.5, ("fold", "call"))) == "call_odds"


def test_mixed_headline_has_no_frequency_claim():
    r = rec("bb_vs_bet", "weak_pair", "fold", 1.0, 1.0, ("fold", "call"), mixed=True)
    e = explain(r)
    assert e["reason"] == "mixed"
    assert "acceptable" in e["headline"]
    # headline must not assert which action is used more (that lives in detail)
    assert "used a bit more" not in e["headline"]


def test_three_action_near_top2_is_not_mixed_reason():
    """Raise≈call with fold dominated must not teach 'any action is acceptable'."""
    r = {
        "node": "bb_vs_bet", "acting_player": "BB", "hand": "7h7c",
        "hand_category": "weak_pair", "preferred": "raise",
        "actions": ["fold", "call", "raise"],
        "ev": {"fold": 0.0, "call": 0.985, "raise": 1.008},
        "freq": {"fold": 0.05, "call": 0.4, "raise": 0.55},
        "ev_sep_pct": 0.42, "mixed": False,
        "board_texture": ["rainbow", "high_card"], "decision_type": "vs_bet",
    }
    assert classify_reason(r) == "raise_bluff"
    assert explain(r)["reason"] != "mixed"


def test_explanation_shape():
    e = explain(rec("bb_first", "strong_made", "bet", 3.0, 2.0, ("check", "bet")),
                board_favored="BB")
    assert e["reason"] and e["headline"] and isinstance(e["detail"], list) and e["detail"]


def test_relabeled_first_action_is_value_not_fold():
    """SB-vs-BB relabel must not misclassify first-action bets as folds."""
    r = rec("sb_first", "strong_made", "bet", 3.0, 2.0, ("check", "bet"))
    r["decision_type"] = "first_action"
    r["acting_player"] = "SB"
    assert classify_reason(r) == "value"


def test_board_flush_detail_notes_shared_board():
    """River board-flush spots must not teach nut-level strength silently."""
    r = {
        "node": "bb_first", "acting_player": "BB", "hand": "JhJc",
        "board": "Qh8h3h2hAh", "hand_category": "strong_made", "preferred": "bet",
        "actions": ["check", "bet"], "ev": {"check": 1.0, "bet": 1.5},
        "freq": {"check": 0.2, "bet": 0.8}, "ev_sep_pct": 5.0,
        "board_texture": ["monotone", "high_card"], "mixed": False,
        "decision_type": "first_action", "pot_bb": 5.5,
    }
    e = explain(r)
    assert e["reason"] == "value"
    assert any("board alone is a flush" in d for d in e["detail"])
    assert "ahead of the hands that call" not in e["headline"]
    assert "thin value" in e["headline"].lower() or "fold equity" in e["headline"].lower()


def test_river_realization_headline_has_no_free_card():
    r = rec("bb_first", "air", "check", 1.0, 0.5, ("check", "bet"))
    r["board"] = "As 7h 2d Ks 9c"
    e = explain(r)
    assert e["reason"] == "realization"
    assert "free card" not in e["headline"].lower()
    assert "no more cards" in e["headline"].lower() or "no more" in e["headline"].lower()


def test_river_trap_headline_has_no_catch_up():
    r = rec("bb_first", "strong_made", "check", 3.0, 2.0, ("check", "bet"))
    r["board"] = "As 7h 2d Ks 9c"
    e = explain(r)
    assert e["reason"] == "trap"
    assert "let them catch up" not in e["headline"].lower()
    assert "catch up or bluff" not in e["headline"].lower()
    assert "completed board" in e["headline"].lower() or "induce" in e["headline"].lower()
