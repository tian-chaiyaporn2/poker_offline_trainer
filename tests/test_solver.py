"""Solver convergence, stability, and validation-rule tests (PRD §6) — MIT."""

import numpy as np
import pytest

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.presets import BOARDS, build_scenario
from pokertrainer.scenario import ValidationError, load_scenario, validate_solution
from pokertrainer.showdown import equity_matrix
from pokertrainer.solver import FlopSolver


def _small_solver(iters=800):
    board = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c", "AhKh", "Ts9s", "6s5s", "9c8d"]]
    ip = [parse_hand(h) for h in ["AhQh", "KsKh", "JsJh", "AhTh", "Tc9c", "6h5h", "9s8s"]]
    eq, C = equity_matrix(board, oop, ip)
    s = FlopSolver(eq, C, np.ones(len(oop)), np.ones(len(ip)), 5.5, 0.33, 0.75)
    return s, s.solve(iterations=iters)


def test_probabilities_sum_to_one():
    _, res = _small_solver()
    validate_solution(res.strategies)   # raises if any row != 1
    for arr in res.strategies.values():
        assert np.allclose(arr.sum(axis=1), 1.0)
        assert (arr >= -1e-9).all()


def test_exploitability_improves_and_stabilizes():
    """PRD §6: increasing iterations improves OR stabilizes the solution."""
    _, res = _small_solver(iters=1500)
    curve = [e for _, e in res.exploitability_curve]
    # Large net improvement from start to end.
    assert curve[-1] < curve[0] * 0.05
    # Well under the 1%-of-pot target.
    assert res.final_exploitability_pct_pot < 0.5
    # Tail has stabilized: last checkpoints tiny and within a narrow band.
    tail = curve[-4:]
    assert max(tail) < 1e-3
    assert max(tail) - min(tail) < 5e-4


def test_more_iterations_more_stable():
    _, low = _small_solver(iters=300)
    _, high = _small_solver(iters=3000)
    assert high.final_exploitability_bb <= low.final_exploitability_bb


def test_determinism():
    _, r1 = _small_solver(600)
    _, r2 = _small_solver(600)
    assert r1.root_ev_oop_bb == r2.root_ev_oop_bb
    assert np.array_equal(r1.strategies["root"], r2.strategies["root"])


def test_suit_isomorphism_root_ev():
    """PRD §6 / stop condition: suit-equivalent configs must match."""
    board = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhKh", "7s7c", "Ts9s", "6c5c"]]
    ip = [parse_hand(h) for h in ["QdQc", "KhQh", "9c8d", "JsTs"]]
    eq1, C1 = equity_matrix(board, oop, ip)
    ev1 = FlopSolver(eq1, C1, np.ones(4), np.ones(4), 5.5, 0.33, 0.75).solve(1000).root_ev_oop_bb

    perm = {0: 1, 1: 0, 2: 3, 3: 2}
    rl = lambda c: (c // 4) * 4 + perm[c % 4]
    b2 = [rl(c) for c in board]
    o2 = [tuple(sorted((rl(a), rl(b)), reverse=True)) for a, b in oop]
    i2 = [tuple(sorted((rl(a), rl(b)), reverse=True)) for a, b in ip]
    eq2, C2 = equity_matrix(b2, o2, i2)
    ev2 = FlopSolver(eq2, C2, np.ones(4), np.ones(4), 5.5, 0.33, 0.75).solve(1000).root_ev_oop_bb
    assert abs(ev1 - ev2) < 1e-9


def test_validation_rejects_duplicate_board():
    raw = build_scenario(BOARDS[0])
    raw["board"] = ["As", "As", "2d"]   # duplicate
    with pytest.raises(ValidationError):
        load_scenario(raw)
