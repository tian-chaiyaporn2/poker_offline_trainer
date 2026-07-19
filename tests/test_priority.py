"""Priority scorer must be a correct, sensible ranking (MIT)."""

from pokertrainer import priority as P


def _rec(node="bb_first", texture=("unpaired", "rainbow", "high_card", "disconnected"),
         hc="air", reason="realization", ev=None, reach=0.8, board="As7h2d", accepted=True):
    return {
        "node": node, "board": board, "board_texture": list(texture),
        "hand_category": hc, "preferred": "check", "reach_mass": reach,
        "ev": ev or {"check": 0.3, "bet": -0.2}, "freq": {"check": 0.9, "bet": 0.1},
        "accepted": accepted, "explanation": {"reason": reason},
    }


def test_texture_freqs_sum_to_one():
    freqs = P.flop_texture_freqs()
    assert abs(sum(freqs.values()) - 1.0) < 1e-9
    # every tuple is (pair, suit, height, connect)
    assert all(len(t) == 4 for t in freqs)
    # two-tone is the single most common suit pattern -> its buckets are sizeable
    assert max(freqs.values()) > 0.05


def test_scores_in_range_and_have_parts():
    recs = [_rec(reason=r) for r in ("value", "trap", "bluff", "fold", "realization")]
    scored = P.score_records(recs)
    assert len(scored) == 5
    for r in scored:
        assert 0.0 <= r["priority"] <= 1.0
        assert set(r["priority_parts"]) >= {"frequency", "impact", "intuition", "impact_pct"}


def test_intuition_ranks_counterintuitive_plays_higher():
    # a trap (check a monster) must out-score a value bet on the intuition axis
    assert P.INTUITION["trap"] > P.INTUITION["value"]
    assert P.INTUITION["bluff_catch"] > P.INTUITION["fold"]


def test_impact_tracks_ev_spread():
    small = _rec(ev={"check": 0.10, "bet": 0.05})        # 0.05 spread
    big = _rec(ev={"fold": 0.0, "call": 3.0})            # 3.0 spread
    scored = P.score_records([small, big])
    by_id = {tuple(r["ev"].items()): r for r in scored}
    assert (by_id[(("fold", 0.0), ("call", 3.0))]["priority_parts"]["impact_pct"]
            > by_id[(("check", 0.10), ("bet", 0.05))]["priority_parts"]["impact_pct"])


def test_solve_backlog_flags_uncovered_common_textures():
    # only cover a rainbow board; the common two-tone bucket must show as uncovered
    recs = [_rec(texture=("unpaired", "rainbow", "high_card", "disconnected"))]
    backlog = P.solve_backlog(recs, top=25)
    covered = [b for b in backlog if b["board_texture"] == ["unpaired", "rainbow", "high_card", "disconnected"]]
    uncovered = [b for b in backlog if not b["covered"]]
    assert covered and covered[0]["boards_solved"] == 1
    assert uncovered and all(u["records"] == 0 for u in uncovered)
    # backlog is frequency-ranked among uncovered -> first uncovered is high-frequency
    assert uncovered[0]["occurrence_pct"] > 5.0


def test_lesson_backlog_ranks_by_total_value():
    recs = [_rec(reason="trap") for _ in range(5)] + [_rec(node="btn_vs_check", reason="bluff")]
    scored = P.score_records(recs)
    backlog = P.lesson_backlog(scored)
    assert backlog and backlog[0]["total_value"] >= backlog[-1]["total_value"]
    assert all({"node", "reason", "n_records", "total_value"} <= set(b) for b in backlog)
