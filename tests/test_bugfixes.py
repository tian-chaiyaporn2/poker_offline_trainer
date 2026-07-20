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
    """run() must report the iteration-AVERAGED root EV — not the last iterate —
    consistently across solvers. (A pure agreement check would also pass under the
    old last-iterate behavior, so also pin it to the average and to ground truth.)"""
    from pokertrainer.solver.multistreet import MultiStreetSpike
    flop = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c"]]
    ip = [parse_hand(h) for h in ["AhQh", "JsJh", "Tc9c"]]
    wo, wi = np.ones(3), np.ones(3)
    a = MultiStreetSpike(flop, oop, ip, wo, wi, 5.5, 0.66, streets=2).run(60)
    b = BatchedCFR(flop, oop, ip, wo, wi, 5.5, 0.66, streets=2).run(60)
    # (1) both solvers agree on the averaged-strategy EV
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9
    # (2) the reported EV is the AVERAGE, provably distinct from the last iterate
    #     (ev_curve records the per-iteration last-iterate EVs). If run() reported
    #     the last iterate these would be equal.
    assert abs(a["root_ev_oop_bb"] - a["ev_curve"][-1][1]) > 1e-4
    assert abs(b["root_ev_oop_bb"] - b["ev_curve"][-1][1]) > 1e-4


def test_checkdown_root_ev_matches_true_equity():
    """Ground truth for the averaged EV + chance denominator together: with no
    betting (bet_frac=0) the game is a pure check-down, so OOP's reported pot
    share must equal OOP's full-runout equity."""
    from pokertrainer.mc_equity import mc_equity
    flop = parse_cards("As7h2d")
    oh, ih = parse_hand("KhQh"), parse_hand("JcTc")
    r = BatchedCFR(flop, [oh], [ih], np.ones(1), np.ones(1), 5.5, 0.0, streets=3).run(40)
    eq = mc_equity(flop, oh, ih, samples=120000, seed=3)
    assert abs(r["root_ev_pct_pot"] / 100.0 - eq) < 0.01


def test_root_ev_conditioned_on_compatible_matchups():
    """With OVERLAPPING ranges (card collisions -> joint mass < 1), the reported
    root EV must be normalized by that compatible mass. On a check-down the pot
    share then equals the reach-weighted equity over compatible matchups; the
    un-normalized value understates it by ~1-joint."""
    from pokertrainer.mc_equity import mc_equity
    flop = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhKh", "AhQd", "KsKc", "Td9d", "QdJd", "Kh5h"]]
    ip = [parse_hand(h) for h in ["AhJc", "QhQc", "KsTs", "Td8c", "9s8s", "KhQs"]]
    wo, wi = np.ones(len(oop)), np.ones(len(ip))
    s = BatchedCFR(flop, oop, ip, wo, wi, 5.5, 0.0, streets=3)
    r = s.run(40)
    joint = float(s.w_o @ (s.B @ s.w_i))
    assert joint < 0.95, "test needs colliding ranges to be meaningful"
    num = den = 0.0
    for i, ho in enumerate(oop):
        for j, hi in enumerate(ip):
            if s.B[i, j] == 0:
                continue
            eq = mc_equity(flop, ho, hi, samples=30000, seed=100 + i * 10 + j)
            num += s.w_o[i] * s.w_i[j] * eq
            den += s.w_o[i] * s.w_i[j]
    cond_eq = num / den
    assert abs(r["root_ev_pct_pot"] / 100.0 - cond_eq) < 0.01



def test_content_yield_checkpoint_config_mismatch(tmp_path):
    from pokertrainer.content_yield import (
        _atomic_write_json, _ensure_checkpoint_config, _solve_config,
    )
    out = tmp_path / "cy"
    out.mkdir()
    (out / "boards").mkdir()
    cfg_a = _solve_config(40, 100, "cpu", "float64", 5.5, 0.66, None, [0, 1])
    cfg_b = _solve_config(40, 200, "cpu", "float64", 5.5, 0.66, None, [0, 1])
    _ensure_checkpoint_config(str(out), cfg_a, fresh=False)
    try:
        _ensure_checkpoint_config(str(out), cfg_b, fresh=False)
        assert False, "expected SystemExit on config mismatch"
    except SystemExit as e:
        assert "mismatch" in str(e)


def test_content_yield_fresh_clears_stale_boards(tmp_path):
    from pokertrainer.content_yield import (
        _atomic_write_json, _ensure_checkpoint_config, _solve_config,
    )
    out = tmp_path / "cy"
    boards = out / "boards"
    boards.mkdir(parents=True)
    stale = boards / "board_00.json"
    _atomic_write_json(str(stale), [{"hand": "AhAd", "accepted": True}])
    cfg = _solve_config(40, 100, "cpu", "float64", 5.5, 0.66, None, [0])
    _ensure_checkpoint_config(str(out), cfg, fresh=True)
    assert not stale.exists()
    assert (out / "solve_config.json").exists()


def test_content_yield_refuses_orphan_checkpoints(tmp_path):
    from pokertrainer.content_yield import _atomic_write_json, _ensure_checkpoint_config, _solve_config
    out = tmp_path / "cy"
    boards = out / "boards"
    boards.mkdir(parents=True)
    _atomic_write_json(str(boards / "board_00.json"), [{"hand": "AhAd"}])
    cfg = _solve_config(40, 100, "cpu", "float64", 5.5, 0.66, None, [0])
    try:
        _ensure_checkpoint_config(str(out), cfg, fresh=False)
        assert False, "expected SystemExit for orphan checkpoints"
    except SystemExit as e:
        assert "solve_config.json" in str(e)


def test_validate_records_flags_bad_freq():
    from pokertrainer.content_yield import validate_records
    recs = [{
        "node": "bb_first", "hand": "AhAd",
        "ev": {"check": 1.0, "bet": 0.5},
        "freq": {"check": 0.5, "bet": 0.3},
        "preferred": "check",
    }]
    problems = validate_records(recs)
    assert any("freq sums" in p for p in problems)


def test_validate_flop_write_empty_rows(tmp_path):
    from pokertrainer.validate_flop import _write
    _write([], str(tmp_path), {"n": 1})
    import json
    summary = json.load(open(tmp_path / "summary.json"))
    assert summary["totals"]["hand_decisions"] == 0


def test_scenario_rejects_partial_allowed():
    from pokertrainer.presets import BOARDS, build_scenario
    from pokertrainer.scenario import load_scenario, ValidationError
    raw = build_scenario(BOARDS[0])
    raw["actions"]["allowed"] = ["check"]
    with pytest.raises(ValidationError, match="full FlopSolver set"):
        load_scenario(raw)


def test_export_acceptable_includes_acceptable_grade():
    from pokertrainer.export import build_questions
    # Synthetic solve payload: second action is within 2% pot of best → acceptable
    solve = {
        "scenario_id": "t", "board": ["As", "7h", "2d"], "pot_bb": 5.5,
        "actions": ["check", "bet_small"],
        "solver": "test",
        "per_hand": [{
            "hand": "AcJc",  # first free AJs combo vs As7h2d board
            "strategy": {"check": 0.8, "bet_small": 0.2},
            "action_ev_bb": {"check": 1.0, "bet_small": 0.95},  # ~0.9% pot loss
        }],
    }
    qs = build_questions(solve, max_per_board=1)
    assert qs
    assert "bet_small" in qs[0]["acceptable_actions"]
    assert qs[0]["action_grade"]["bet_small"] == "acceptable"


def test_turn_board_uses_correct_street_count():
    """Turn/river demos must not hard-code a 3-street flop tree (6-card boards)."""
    from pokertrainer.validate_flop import _make_solver
    flop = parse_cards("Th9h8d2c")
    oop = [parse_hand(h) for h in ["AhAc", "KsKc"]]
    ip = [parse_hand(h) for h in ["AhQh", "JsJh"]]
    make = _make_solver("cpu", "float64")
    s = make(flop, oop, ip, np.ones(2), np.ones(2), 5.5, 0.66, 2)
    assert s.n_streets == 2 and s.bet_streets == 2
    s.run(4)
    sizes = {len(k) for k in s._Ecache}
    assert sizes == {5}, f"expected 5-card showdowns, got {sizes}"


def test_aggregate_only_checks_checkpoint_config(tmp_path):
    from pokertrainer.content_yield import (
        _atomic_write_json, run as cy_run, _solve_config,
    )
    out = tmp_path / "cy"
    boards = out / "boards"
    boards.mkdir(parents=True)
    cfg = _solve_config(40, 100, "cpu", "float64", 5.5, 0.66, None, [0])
    cfg["scenario"] = "btn_vs_bb_srp"
    _atomic_write_json(str(out / "solve_config.json"), cfg)
    _atomic_write_json(str(boards / "board_00.json"), [{
        "accepted": True, "node": "bb_first", "board_texture": [],
        "hand_category": "air", "preferred": "check",
    }])
    try:
        cy_run(n=40, iters=100, roots=[0], out=str(out), aggregate_only=True,
               scenario="sb_vs_bb_srp")
        assert False, "expected SystemExit on aggregate-only scenario mismatch"
    except SystemExit as e:
        assert "mismatch" in str(e)


def test_find_pack_prefers_mtime_not_lexicographic(tmp_path, monkeypatch):
    import time
    import trainer.pack_server as ps
    packs = tmp_path / "output" / "packs"
    packs.mkdir(parents=True)
    older = packs / "flop_pack_v10.db"
    newer = packs / "flop_pack_v9.db"
    older.write_text("x"); time.sleep(0.02); newer.write_text("y")
    monkeypatch.setattr(ps, "ROOT", str(tmp_path))
    assert ps.find_pack() == str(newer)
