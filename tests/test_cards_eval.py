"""Card model and evaluator tests (MIT)."""

import random

import pytest

from pokertrainer.cards import (cards_str, hand_str, parse_card, parse_cards,
                                parse_hand, card_str)
from pokertrainer.evaluator import (category_name, evaluate, evaluate_ref)


def test_card_roundtrip():
    for text in ["Ah", "Td", "2c", "Ks", "9h"]:
        assert card_str(parse_card(text)) == text


def test_parse_hand_high_first():
    a, b = parse_hand("2cAh")          # ace should sort first
    assert card_str(a) == "Ah"
    assert hand_str((a, b)) == "Ah2c"


def test_parse_cards_forms():
    assert cards_str(parse_cards("AhKd")) == "AhKd"
    assert cards_str(parse_cards(["Ah", "Kd"])) == "AhKd"
    assert cards_str(parse_cards("Ah Kd")) == "AhKd"


@pytest.mark.parametrize("text,cat", [
    ("AhKhQhJhTh", "straight flush"),
    ("AsAhAdAc2h", "four of a kind"),
    ("AsAhAd2h2c", "full house"),
    ("Ah9h7h4h2h", "flush"),
    ("5h4d3c2sAh", "straight"),        # wheel
    ("AsAhAd7h2c", "three of a kind"),
    ("AsAhKdKc2c", "two pair"),
    ("AsAh9d7c2s", "one pair"),
    ("AsKh9d7c2s", "high card"),
])
def test_categories(text, cat):
    assert category_name(evaluate(parse_cards(text))) == cat


def test_wheel_below_six_high():
    assert evaluate(parse_cards("6h5d4c3s2h")) > evaluate(parse_cards("5h4d3c2sAh"))


def test_fast_matches_reference_random():
    rng = random.Random(123)
    for _ in range(5000):
        cards = rng.sample(range(52), 7)
        assert evaluate(cards) == evaluate_ref(cards)


def test_seven_card_best_of_five():
    # straight flush hidden among 7 cards
    assert category_name(evaluate(parse_cards("AhKhQhJhTh2c3d"))) == "straight flush"
