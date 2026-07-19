"""Foundations content generators must be correct + deterministic (MIT)."""

from pokertrainer import foundations as F
from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.content_yield import board_texture
from pokertrainer.handinfo import describe_hand
from pokertrainer.mc_equity import mc_equity


def test_all_questions_are_auto_gradeable():
    qs = F.generate_all()
    assert len(qs) > 40
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    for q in qs:
        assert q["answer"] in q["options"], f"{q['id']}: answer not an option"
        assert 2 <= len(q["options"]) <= 5
        assert q["prompt"] and q["explanation"]


def test_generation_is_deterministic():
    a = F.generate_all()
    b = F.generate_all()
    assert a == b, "generators must be reproducible (signed-pack requirement)"


def test_board_reading_matches_texture():
    for q in F.board_reading_questions():
        tags = board_texture(parse_cards(q["data"]["board"]))
        if q["kind"] == "pairing":
            assert q["answer"] == ("Paired" if "paired" in tags else "Unpaired")
        elif q["kind"] == "connectedness":
            assert q["answer"] == ("Connected" if "connected" in tags else "Disconnected")


def test_pot_odds_arithmetic_is_correct():
    for q in F.pot_odds_questions():
        pot, bet = q["data"]["pot"], q["data"]["bet"]
        assert abs(q["data"]["break_even"] - bet / (pot + 2 * bet)) < 1e-3  # stored rounded to 4dp
        assert q["answer"] == f"{round(100 * bet / (pot + 2 * bet))}%"


def test_hand_reading_matches_evaluator():
    for q in F.hand_reading_questions():
        h = parse_hand(q["data"]["hand"])
        b = parse_cards(q["data"]["board"])
        assert q["answer"] == describe_hand(h, b)


def test_equity_answer_matches_band():
    for q in F.equity_questions():
        eq = mc_equity(parse_cards(q["data"]["board"]), parse_hand(q["data"]["hero"]),
                       parse_hand(q["data"]["villain"]), samples=60000, seed=99)
        # answer band should contain the (independently sampled) equity within slack
        lo_hi = {"0–20% (big underdog)": (0, .2), "20–40% (behind)": (.2, .4),
                 "40–60% (coin flip)": (.4, .6), "60–80% (ahead)": (.6, .8),
                 "80–100% (big favorite)": (.8, 1.01)}
        lo, hi = lo_hi[q["answer"]]
        assert lo - 0.05 <= eq <= hi + 0.05, f"{q['id']}: eq {eq:.3f} not near band {q['answer']}"
