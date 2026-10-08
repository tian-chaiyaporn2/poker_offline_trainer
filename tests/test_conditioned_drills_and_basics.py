"""Turn/river drills come from conditioned (solved-line) spots; Basics questions are embedded."""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def bt():
    spec = importlib.util.spec_from_file_location("build_trainer", ROOT / "demo" / "build_trainer.py")
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(ROOT)                     # pack paths are repo-relative
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def cond(bt):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return bt.load_conditioned_turnriver()
    finally:
        os.chdir(cwd)


def test_conditioned_spots_are_turn_and_river_only(bt, cond):
    streets = {bt.STREET[len(d["board"])] for d in cond}
    assert streets == {"turn", "river"}


def test_every_conditioned_spot_has_the_completed_earlier_streets(bt, cond):
    for d in cond:
        street = bt.STREET[len(d["board"])]
        want = ["flop"] if street == "turn" else ["flop", "turn"]
        assert [s["street"] for s in d["line"]] == want, d["id"]
        for s in d["line"]:
            assert s["acts"], d["id"]
            for who, act in s["acts"]:
                assert who in ("you", "opp") and act in ("check", "bet", "call", "fold")
            # a completed street that reached the next one never ends in a fold
            assert s["acts"][-1][1] in ("check", "call"), (d["id"], s)


def test_conditioned_spots_are_re_explained_with_real_numbers(cond):
    reasons = {d["reason"] for d in cond}
    assert "continuation" not in reasons
    assert len(reasons) >= 6                      # a real spread of teaching reasons
    for d in cond:
        assert d["headline"]
        assert "None" not in d["detail"], d["id"]   # ev gap derived, never missing


def test_turnriver_drill_mixes_both_streets(bt):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        qs = bt.load_turnriver()
    finally:
        os.chdir(cwd)
    assert {q["street"] for q in qs} == {"turn", "river"}
    assert all(q["line"] for q in qs)


def test_line_events_follow_the_node_and_villain_reply(bt):
    ev = bt._line_events({"node": "bb_first"}, "check", "Opponent bets 66% of the pot.")
    assert ev == [("you", "check")]                          # the bet arrives with the vs_bet step
    ev = bt._line_events({"node": "bb_vs_bet"}, "call", "You call — the turn comes.")
    assert ev == [("opp", "bet"), ("you", "call")]
    ev = bt._line_events({"node": "btn_vs_check"}, "bet", "Opponent calls — the river comes.")
    assert ev == [("opp", "check"), ("you", "bet"), ("opp", "call")]
    ev = bt._line_events({"node": "bb_first"}, "bet", "Opponent folds — you win the pot.")
    assert ev == [("you", "bet"), ("opp", "fold")]


def test_basics_questions_are_embedded_and_gradeable():
    html = (ROOT / "index.html").read_text()
    m = re.search(r"const FOUND = (\[[^\n]+\]);", html)
    assert m, "basics questions not embedded"
    found = json.loads(m.group(1))
    assert len(found) >= 60
    assert {q["unit"] for q in found} == {"board_reading", "hand_reading", "pot_odds", "equity"}
    for q in found:
        assert q["basics"] is True and q["answer"] in q["actions"]
        assert q["ask"] and q["why"]
        if q["unit"] != "pot_odds":
            assert q["board"], q["id"]                  # the app draws the cards
    assert 'data-c="basics"' in html


def test_basics_copy_is_beginner_language():
    from pokertrainer.foundations import generate_all
    for q in generate_all():
        text = q["ask"] + " " + q["explanation"]
        assert not re.search(r"\bbb\b", text), q["id"]          # chips, not bb
        assert "equity" not in text.lower(), q["id"]
        assert "Monte-Carlo" not in text and "makes:" not in text, q["id"]


def test_multi_size_records_render_as_sized_actions(bt):
    base = {"board": "As7h2d", "hand": "KdKc", "acting_player": "BB", "mixed": 0,
            "reason": "value", "headline": "h", "detail": "[]"}
    first = bt._to_q(dict(base, node="bb_first", actions='["check","bet_33","bet_75"]',
                          ev='{"check":1,"bet_33":2,"bet_75":1.5}',
                          freq='{"check":0.1,"bet_33":0.8,"bet_75":0.1}',
                          preferred_action="bet_33",
                          action_grades='{"check":"costly","bet_33":"best","bet_75":"good"}'))
    assert first["labels"] == {"check": "Check", "bet_33": "Bet 33%", "bet_75": "Bet 75%"}
    facing = bt._to_q(dict(base, node="bb_vs_bet_75", actions='["fold","call"]',
                           ev='{"fold":0,"call":1}', freq='{"fold":0.2,"call":0.8}',
                           preferred_action="call",
                           action_grades='{"fold":"costly","call":"best"}'), bet_pct=33)
    assert facing["node"] == "bb_vs_bet" and facing["bet_pct"] == 75
