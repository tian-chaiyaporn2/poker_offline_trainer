"""End-to-end pipeline, export, and comparison tests (MIT)."""

import numpy as np

from pokertrainer.compare import compare
from pokertrainer.export import build_questions
from pokertrainer.normalize import from_solver_arrays
from pokertrainer.presets import BOARDS, build_scenario
from pokertrainer.reference_solver import ReferenceCFR
from pokertrainer.runner import run_scenario
from pokertrainer.scenario import load_scenario
from pokertrainer.showdown import equity_matrix
from pokertrainer.solver import FlopSolver


def test_run_scenario_and_export():
    raw = build_scenario(BOARDS[0], iterations=600)
    out = run_scenario(raw)
    assert out["n_oop_combos"] > 100
    # range frequencies sum to ~1
    assert abs(sum(out["range_action_frequencies"].values()) - 1.0) < 1e-6
    qs = build_questions(out, max_per_board=10)
    assert len(qs) >= 8
    for q in qs:
        assert q["recommended_action"] in q["available_actions"]
        assert abs(sum(q["action_frequencies"].values()) - 1.0) < 1e-3
        assert q["validation_status"] == "passed"


def test_cfr_plus_agrees_with_reference():
    raw = build_scenario(BOARDS[2], iterations=1200)   # connected board
    scn = load_scenario(raw)
    eq, C = equity_matrix(scn.board, scn.oop_combos, scn.ip_combos)

    r = FlopSolver(eq, C, scn.w_oop, scn.w_ip, scn.pot_bb, scn.small_frac, scn.large_frac).solve(1200)
    a = from_solver_arrays(scn.id, raw["board"], scn.oop_combos,
                           r.strategies["root"], r.action_ev["root"],
                           scn.w_oop, r.root_ev_oop_bb, "cfr_plus")

    ref = ReferenceCFR(eq, C, scn.w_oop, scn.w_ip, scn.pot_bb, scn.small_frac, scn.large_frac)
    avg = ref.solve(4000)
    b = from_solver_arrays(scn.id, raw["board"], scn.oop_combos,
                           avg["root"], ref.root_action_ev(avg),
                           scn.w_oop, ref.root_ev_bb(avg), "reference_cfr")

    rep = compare(a, b, scn.pot_bb)
    assert abs(rep["root_ev"]["diff_pct_pot"]) < 1.0        # target
    assert rep["preferred_action_agreement"] >= 0.90        # target
    assert len(rep["major_disagreements"]) == 0
    assert rep["all_targets_pass"]
