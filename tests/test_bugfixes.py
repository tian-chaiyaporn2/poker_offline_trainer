"""Regression tests for bugs found in the comprehensive review (MIT)."""

import numpy as np
import pytest

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.compare import compare
from pokertrainer.handinfo import describe_hand
from pokertrainer.normalize import NormalizedSolve
from pokertrainer.showdown import equity_matrix
from pokertrainer.solver.batched import BatchedCFR
from pokertrainer.solver.cfr import FlopSolver
from pokertrainer.validate_flop import hand_category


def test_chance_node_pure_runout_matches_exact_equity():
    """Multi-street chance nodes must average over 52-board-4 cards, not -2."""
    flop = parse_cards("AsKd2c")
    oop = [parse_hand("AhAd")]
    ip = [parse_hand("KhQc")]
    exact = float(equity_matrix(flop, oop, ip)[0][0, 0])
    # Pure runout (no betting): root EV with pot=1 is exact equity.
    res = BatchedCFR(
        flop, oop, ip, np.ones(1), np.ones(1), pot_bb=1.0,
        streets=3, bet_streets=0,
    ).run(1)
    assert abs(res["root_ev_oop_bb"] - exact) < 1e-9


def test_hand_category_draw_not_strong_made():
    desc = describe_hand(parse_hand("6h5h"), parse_cards("Th9h8d"))
    assert "draw" in desc
    assert hand_category(desc) == "draw"


def test_board_pair_only_is_air_not_weak_pair():
    desc = describe_hand(parse_hand("AsJd"), parse_cards("KsKd6h"))
    assert desc.startswith("high card") or "draw" in desc
    assert hand_category(desc) == "air" or hand_category(desc) == "draw"
    assert hand_category(desc) != "weak_pair"


def test_equity_matrix_rejects_board_collision():
    board = parse_cards("AsKd2c")
    with pytest.raises(ValueError, match="collides"):
        equity_matrix(board, [parse_hand("AsAd")], [parse_hand("KhQc")])


def test_equity_matrix_incompatible_pairs_are_half():
    board = parse_cards("2c3d4h")
    eq, compat = equity_matrix(board, [parse_hand("AsAd")], [parse_hand("AsKh")])
    assert compat[0, 0] == 0.0
    assert eq[0, 0] == 0.5


def test_root_ev_normalized_by_joint_mass():
    board = parse_cards("2c3d4h")
    oop = [parse_hand("AsAd")]
    # One IP combo shares As (impossible); one is live.
    ip = [parse_hand("AsKh"), parse_hand("KhKd")]
    eq, compat = equity_matrix(board, oop, ip)
    assert float(compat[0].sum()) == 1.0
    s = FlopSolver(eq, compat, np.ones(1), np.ones(2), 6.5, 0.33, 0.75)
    # Force near-checkdown by many iters; conditional EV ≈ pot * equity vs KhKd.
    res = s.solve(400)
    live_eq = float(eq[0, 1])
    # With check-heavy play, root EV should be near pot * live equity, not half.
    assert res.root_ev_oop_bb > 6.5 * live_eq * 0.7


def test_compare_rejects_disjoint_hands():
    a = NormalizedSolve(
        scenario_id="x", source="a", board=["As", "Kd", "2c"],
        actions=["check", "bet_small", "bet_large"],
        root_ev_oop_bb=1.0,
        range_freqs={"check": 1, "bet_small": 0, "bet_large": 0},
        per_hand={"AhAd": {"strategy": {"check": 1, "bet_small": 0, "bet_large": 0},
                           "ev": {"check": 1, "bet_small": 0, "bet_large": 0}}},
    )
    b = NormalizedSolve(
        scenario_id="x", source="b", board=["As", "Kd", "2c"],
        actions=["check", "bet_small", "bet_large"],
        root_ev_oop_bb=1.0,
        range_freqs={"check": 1, "bet_small": 0, "bet_large": 0},
        per_hand={"KhKd": {"strategy": {"check": 1, "bet_small": 0, "bet_large": 0},
                           "ev": {"check": 1, "bet_small": 0, "bet_large": 0}}},
    )
    with pytest.raises(ValueError, match="shared hand"):
        compare(a, b, pot_bb=5.5)


def test_compare_rejects_board_mismatch():
    hand = {"AhAd": {"strategy": {"check": 1, "bet_small": 0, "bet_large": 0},
                     "ev": {"check": 1, "bet_small": 0, "bet_large": 0}}}
    a = NormalizedSolve("x", "a", ["As", "Kd", "2c"],
                        ["check", "bet_small", "bet_large"], 1.0,
                        {"check": 1, "bet_small": 0, "bet_large": 0}, hand)
    b = NormalizedSolve("x", "b", ["Qs", "Jd", "2c"],
                        ["check", "bet_small", "bet_large"], 1.0,
                        {"check": 1, "bet_small": 0, "bet_large": 0}, hand)
    with pytest.raises(ValueError, match="board mismatch"):
        compare(a, b, pot_bb=5.5)


def test_range_rejects_weight_above_one():
    from pokertrainer.ranges import expand_range
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expand_range({"AA": 1.5}, [])


def test_range_rejects_duplicate_aliases():
    from pokertrainer.ranges import expand_range
    with pytest.raises(ValueError, match="duplicate combo"):
        expand_range({"AKs": 1.0, "KAs": 1.0}, [])


def test_scenario_rejects_multistreet_tree():
    from pokertrainer.presets import BOARDS, build_scenario
    from pokertrainer.scenario import load_scenario, ValidationError
    raw = build_scenario(BOARDS[0])
    raw["tree"] = {"streets": ["flop", "turn", "river"]}
    with pytest.raises(ValidationError, match="tree.streets"):
        load_scenario(raw)


def test_solve_result_has_action_ev_for_all_infosets():
    board = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c"]]
    ip = [parse_hand(h) for h in ["AhQh", "JsJh", "Tc9c"]]
    eq, C = equity_matrix(board, oop, ip)
    res = FlopSolver(eq, C, np.ones(3), np.ones(3), 5.5, 0.33, 0.75).solve(80)
    for key in res.action_labels:
        assert key in res.action_ev
        assert res.action_ev[key].shape[1] == len(res.action_labels[key])


def test_run_reports_average_strategy_ev():
    """Batched and multistreet run() must agree on average-strategy root EV."""
    from pokertrainer.solver.multistreet import MultiStreetSpike
    flop = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c"]]
    ip = [parse_hand(h) for h in ["AhQh", "JsJh", "Tc9c"]]
    wo, wi = np.ones(3), np.ones(3)
    a = MultiStreetSpike(flop, oop, ip, wo, wi, 5.5, 0.66, streets=2).run(60)
    b = BatchedCFR(flop, oop, ip, wo, wi, 5.5, 0.66, streets=2).run(60)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9

