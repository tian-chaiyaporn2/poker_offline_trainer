"""Solver → recommendation → training translation contracts (MIT)."""

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
