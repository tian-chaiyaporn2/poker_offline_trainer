"""Scenario parameterization: the default is unchanged, new positions relabel (MIT)."""

from pokertrainer.cards import parse_hand
from pokertrainer.content_yield import extract_records
from pokertrainer.presets import SCENARIOS
from pokertrainer.validate_flop import _make_solver

OOP = [parse_hand(h) for h in ["AhKh", "QsQd", "7c6c"]]
IP = [parse_hand(h) for h in ["AsKd", "JhJc", "9h8h"]]


def test_registry_has_both_scenarios():
    for name in ("btn_vs_bb_srp", "sb_vs_bb_srp"):
        sc = SCENARIOS[name]
        assert {"oop_range", "ip_range", "oop_pos", "ip_pos", "pot", "bet_frac"} <= set(sc)


def test_default_scenario_labels_unchanged():
    """BTN-vs-BB must keep the historical node names + positions so existing packs
    and merges stay compatible."""
    make = _make_solver("cpu", "float64")
    recs = extract_records("As7h2d", OOP, IP, 4, make, 5.5, 0.66)
    assert {r["node"] for r in recs} == {"bb_first", "btn_vs_check", "bb_vs_bet", "btn_vs_bet"}
    assert {r["acting_player"] for r in recs} == {"BB", "BTN"}
    assert all(r["scenario"] == "btn_vs_bb_srp" for r in recs)


def test_sb_scenario_relabels_nodes_and_positions():
    make = _make_solver("cpu", "float64")
    recs = extract_records("As7h2d", OOP, IP, 4, make, 6.0, 0.66,
                           oop_pos="SB", ip_pos="BB", scenario="sb_vs_bb_srp")
    assert {r["node"] for r in recs} == {"sb_first", "bb_vs_check", "sb_vs_bet", "bb_vs_bet"}
    assert {r["acting_player"] for r in recs} == {"SB", "BB"}
    assert all(r["scenario"] == "sb_vs_bb_srp" for r in recs)
    # first-action classification still works on relabeled nodes
    dt = {r["node"]: r["decision_type"] for r in recs}
    assert dt["sb_first"] == "first_action" and dt["sb_vs_bet"] == "vs_bet"
