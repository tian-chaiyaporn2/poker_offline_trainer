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


def test_line_events_come_from_structure_not_narration(bt):
    FLOP, TURN = "As7h2d", "As7h2dKc"
    st = lambda node, board=FLOP: {"node": node, "board": board}
    # OOP check, then this street's facing-a-bet step: the bet comes from that step.
    assert bt._line_events(st("bb_first"), "check", st("bb_vs_bet")) == [("you", "check")]
    # OOP check, then the next street: the opponent checked back.
    assert bt._line_events(st("bb_first"), "check", st("bb_first", TURN)) == \
        [("you", "check"), ("opp", "check")]
    # IP check-back closes the street itself — exactly two checks, never three.
    assert bt._line_events(st("btn_vs_check"), "check", st("btn_vs_check", TURN)) == \
        [("opp", "check"), ("you", "check")]
    # A bet followed by a later street was called.
    assert bt._line_events(st("btn_vs_check"), "bet", st("btn_vs_check", TURN)) == \
        [("opp", "check"), ("you", "bet"), ("opp", "call")]
    assert bt._line_events(st("bb_vs_bet"), "call", st("bb_first", TURN)) == \
        [("opp", "bet"), ("you", "call")]
    # Last step of the hand: no opponent reply is invented.
    assert bt._line_events(st("bb_first"), "bet", None) == [("you", "bet")]
    # A sized facing node is still "they bet, then you answer".
    assert bt._line_events(st("bb_vs_bet_33"), "fold", None) == [("opp", "bet"), ("you", "fold")]


def test_hero_checked_back_rewrites_only_display_text(bt):
    assert bt._hero_checked_back(
        "btn_vs_check", "check", "Opponent checks back — the turn comes."
    ) == "You check back — the turn comes."
    assert bt._hero_checked_back(
        "bb_first", "check", "Opponent checks back — the turn comes."
    ) == "Opponent checks back — the turn comes."


def test_no_completed_street_checks_three_times(cond):
    for d in cond:
        for s in d["line"]:
            checks = [a for _, a in s["acts"] if a == "check"]
            assert len(checks) <= 2, (d["id"], s)


def test_ip_checkback_narration_names_the_hero(bt):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        hands = bt.load_continuation()
        exploit = bt.load_exploit()
    finally:
        os.chdir(cwd)
    seen = 0
    for steps in hands:
        for s in steps:
            if str(s["node"]).endswith("_vs_check") and s["preferred"] == "check":
                assert s["villain_action"].startswith("You check back"), s["villain_action"]
                seen += 1
    assert seen
    ex_seen = 0
    for hands in exploit.values():
        for steps in hands:
            for s in steps:
                if str(s["node"]).endswith("_vs_check") and s["preferred"] == "check":
                    assert s["villain_action"].startswith("You check back"), s["villain_action"]
                    ex_seen += 1
    assert ex_seen


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


def test_basics_session_persists_and_pot_odds_are_possible():
    """Basics answers must survive a reload, and a 'how often' choice can't exceed 100%."""
    for path in (ROOT / "index.html", ROOT / "demo" / "trainer_demo.html"):
        html = path.read_text()
        assert "const b=streets.basics;" in html, path          # Basics row survives a reload
        assert 'basics:"Basics"' in html, path
        assert 'id="session-unit"' in html, path
        assert "every question was right" in html, path
        assert "Why is that the answer?" in html, path
        m = re.search(r"const FOUND = (\[[^\n]+\]);", html)
        assert m, path
        for q in json.loads(m.group(1)):
            if q["unit"] != "pot_odds":
                continue
            seen = set()
            for opt in q["actions"]:
                assert opt not in seen, (path, q["id"])
                seen.add(opt)
                assert 1 <= int(opt.rstrip("%")) <= 100, (path, q["id"], opt)


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


def test_conditioned_mixed_uses_the_ev_rule(cond):
    """Turn/river close calls use the flop packs' EV rule (all actions within CLEAR_SEP_PCT)."""
    import sqlite3
    from pokertrainer.content_yield import CLEAR_SEP_PCT
    c = sqlite3.connect(ROOT / "output" / "packs" / "flop_pack_continuation_full.db")
    pots = dict(c.execute("SELECT id, pot_bb FROM flop_decision"))
    c.close()
    assert any(d["mixed"] for d in cond) and any(not d["mixed"] for d in cond)
    for d in cond:
        ev = json.loads(d["ev"])
        close = all(100.0 * (max(ev.values()) - v) / pots[d["id"]] < CLEAR_SEP_PCT
                    for v in ev.values())
        assert bool(d["mixed"]) == close, d["id"]


def test_basics_stays_out_of_decision_quality():
    src = (ROOT / "demo" / "build_trainer.py").read_text()
    # quiz answers update the session summary + the Basics row only, never lifetime n/solid/leak
    assert "recordGrade(stats,tier,hit);trackStreet(lifetime,hit);saveLifetime();" in src
    assert "out.street.basics={n:n,hit:hit}" in src
    # answers an earlier build folded into n/solid/leak are pulled back out on load
    assert "pokerN+n===out.n&&pokerN<out.n" in src


def _pack_bet_pct(path):
    import sqlite3
    c = sqlite3.connect(ROOT / "output" / "packs" / path)
    cfg = json.loads(dict(c.execute("SELECT key, value FROM pack_meta"))["config"])
    c.close()
    return int(cfg["bet_pct_pot"])


def test_play_modes_use_the_pack_bet_size(bt):
    """Play-a-hand, exploit, and turn/river drills name the size the pack was solved at."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        hands = bt.load_continuation()
        exploit = bt.load_exploit()
        drills = bt.load_turnriver()
    finally:
        os.chdir(cwd)
    cont_pct = _pack_bet_pct("flop_pack_continuation_full.db")
    exp_pct = _pack_bet_pct("flop_pack_exploit_full.db")
    for steps in hands:
        for s in steps:
            assert s["bet_pct"] == cont_pct
            if "bet" in s["actions"]:
                assert s["labels"]["bet"] == f"Bet {cont_pct}%"
    for groups in exploit.values():
        for steps in groups:
            for s in steps:
                assert s["bet_pct"] == exp_pct
                if "bet" in s["actions"]:
                    assert s["labels"]["bet"] == f"Bet {exp_pct}%"
    assert drills
    for q in drills:
        assert q["bet_pct"] == cont_pct, q["node"]
        if "bet" in q["actions"]:
            assert q["labels"]["bet"] == f"Bet {cont_pct}%"


def test_play_modes_use_the_ev_close_call(bt):
    """Play-a-hand and exploit call a spot mixed only when every action is within CLEAR_SEP_PCT."""
    import sqlite3
    from pokertrainer.content_yield import CLEAR_SEP_PCT
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        hands = bt.load_continuation()
        exploit = bt.load_exploit()
    finally:
        os.chdir(cwd)

    def expect(path):
        c = sqlite3.connect(ROOT / "output" / "packs" / path)
        out = {}
        for ev_s, pot, detail, mixed in c.execute(
                "SELECT ev, pot_bb, detail, mixed FROM flop_decision"):
            det = json.loads(detail or "{}")
            ev = json.loads(ev_s)
            close = all(100.0 * (max(ev.values()) - v) / pot < CLEAR_SEP_PCT
                        for v in ev.values())
            out[(det.get("hand_id"), int(det.get("step_index", 0)))] = (close, bool(mixed))
        c.close()
        return out

    def check(groups, table):
        changed = 0
        for steps in groups:
            for s in steps:
                close, pack = table[(s["hand_id"], s["step_index"])]
                assert bool(s["mixed"]) == close, s["hand_id"]
                changed += close != pack
        return changed

    assert check(hands, expect("flop_pack_continuation_full.db"))
    exploit_steps = [steps for groups in exploit.values() for steps in groups]
    assert check(exploit_steps, expect("flop_pack_exploit_full.db"))


def test_equity_explanation_names_the_graded_band():
    from pokertrainer.foundations import generate_all
    for q in generate_all():
        if q["unit"] != "equity":
            continue
        lo_hi = q["answer"].split("%")[0]                      # e.g. "20–40"
        assert f"in the {lo_hi}% range" in q["explanation"], q["id"]


def test_equity_text_never_crosses_a_band_or_goes_blank():
    from pokertrainer.foundations import _equity_why
    assert "you win 0% " in _equity_why(0.0)
    assert "you win 19.9% " in _equity_why(0.1996) and "0–20% range" in _equity_why(0.1996)
    assert "you win 29% " in _equity_why(0.29)            # no float floor artefact (28.9)


def test_generators_flag_close_calls_by_ev():
    from pokertrainer.content_yield import ev_close
    src = "".join((ROOT / "demo" / f).read_text() for f in ("gen_continuation.py", "gen_exploit.py"))
    assert "second >= 0.35" not in src and src.count('"mixed": ev_close(d["ev"]') == 2
    assert ev_close({"check": 1.0, "bet": 1.004}, 5.5) and not ev_close({"check": 1.0, "bet": 1.2}, 5.5)
