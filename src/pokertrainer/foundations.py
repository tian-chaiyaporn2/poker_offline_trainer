"""Foundations content generators (PRD v1.3 §6 fundamentals) — MIT.

Turns the `foundation_template` seeds (board reading, pot odds, hand reading,
equity) into concrete, auto-gradeable practice questions. Everything here is
**deterministic** (fixed inputs + seeds) so a generated set is reproducible and
can go into a signed pack. Nothing invents strategy — each answer is computed
from the same primitives the solver pipeline uses (evaluator, board texture,
pot-odds arithmetic, exact showdown equity).

Each question is a dict:
    {id, unit, kind, prompt, ask, options:[...], answer, explanation, data:{...}}
`prompt` names the cards in text (CLI / server); `ask` is the same question without them,
for the app, which draws the cards. Explanations are beginner-facing plain English ("chips",
not "bb"; "how often you win", not "equity"). `answer` is always one of `options`, so the
trainer grades by exact match.

CLI:  PYTHONPATH=src python -m pokertrainer.foundations --out output/foundations
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

from .cards import parse_cards, parse_hand, card_str
from .content_yield import board_texture
from .handinfo import describe_hand
from .presets import BOARDS
from .showdown import equity_matrix


def _board_label(board_str: str) -> str:
    cs = parse_cards(board_str)
    return " ".join(card_str(c) for c in cs)


def _opts(answer: str, distractors: List[str], seed: int, k: int = 3) -> List[str]:
    """Answer + up to k distractors (never equal to the answer), order fixed by seed."""
    pool = [d for d in distractors if d != answer]
    rng = random.Random(seed)
    rng.shuffle(pool)
    opts = [answer] + pool[:k]
    rng.shuffle(opts)
    return opts


# --------------------------------------------------------------------------- #
# 1) Board reading — suit texture, pairing, connectedness (from board_texture) #
# --------------------------------------------------------------------------- #

_SUIT_Q = {"monotone": "Monotone (one suit)", "two_tone": "Two-tone (two suits)",
           "rainbow": "Rainbow (three suits)"}
_SUIT_WHY = {
    "monotone": "All three cards are the same suit, so anyone holding two of that suit "
                "already has a flush.",
    "two_tone": "Two cards share a suit, so anyone holding two more of that suit has a flush "
                "draw — one more card of it makes a flush.",
    "rainbow": "Every card is a different suit, so nobody can have a flush or even a "
               "one-card flush draw yet.",
}


def board_reading_questions() -> List[Dict]:
    out = []
    for bi, entry in enumerate(BOARDS):
        bstr = entry["board"]
        tags = board_texture(parse_cards(bstr))
        lbl = _board_label(bstr)
        suit = next(t for t in tags if t in _SUIT_Q)
        out.append({
            "id": f"found_board_suit_{bi:02d}", "unit": "board_reading", "kind": "suit_texture",
            "prompt": f"How many suits are on the flop {lbl}?",
            "ask": "How many suits are on this flop?",
            "options": list(_SUIT_Q.values()),
            "answer": _SUIT_Q[suit],
            "explanation": _SUIT_WHY[suit],
            "data": {"board": bstr, "tags": tags},
        })
        paired = "paired" in tags
        out.append({
            "id": f"found_board_pair_{bi:02d}", "unit": "board_reading", "kind": "pairing",
            "prompt": f"Is the flop {lbl} paired?",
            "ask": "Is this flop paired?",
            "options": ["Paired", "Unpaired"],
            "answer": "Paired" if paired else "Unpaired",
            "explanation": ("Two of the cards share a rank, so anyone holding the third one "
                            "already has three of a kind." if paired
                            else "All three cards are different ranks — no pair on the board."),
            "data": {"board": bstr, "tags": tags},
        })
        connected = "connected" in tags
        out.append({
            "id": f"found_board_conn_{bi:02d}", "unit": "board_reading", "kind": "connectedness",
            "prompt": f"Is the flop {lbl} connected (coordinated for straights)?",
            "ask": "Are these cards close enough together to make straights likely?",
            "options": ["Connected", "Disconnected"],
            "answer": "Connected" if connected else "Disconnected",
            "explanation": ("The cards are close in rank, so many two-card hands make a "
                            "straight or a straight draw." if connected
                            else "The cards are far apart in rank, so straights are unlikely."),
            "data": {"board": bstr, "tags": tags},
        })
    return out


# --------------------------------------------------------------------------- #
# 2) Pot odds — break-even calling equity (arithmetic)                          #
# --------------------------------------------------------------------------- #

# (pot, bet) in bb. Break-even equity to call = bet / (pot + 2*bet).
_POT_ODDS_SPOTS = [(6, 2), (6, 3), (6, 4), (6, 6), (10, 5), (10, 7.5), (4, 2), (8, 12)]


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def pot_odds_questions() -> List[Dict]:
    out = []
    for i, (pot, bet) in enumerate(_POT_ODDS_SPOTS):
        correct = bet / (pot + 2 * bet)
        # Common wrong calcs: bet/(pot+bet) (ignores your own call), bet/pot, and
        # bet/(pot+3*bet). "How often" is a share of the time, so a result over 100%
        # (bet bigger than the pot) is replaced with the complement — how often you lose.
        answer = _pct(correct)
        distractors = []
        for w in (bet / (pot + bet), bet / pot, bet / (pot + 3 * bet)):
            if round(100 * w) > 100 or round(100 * w) < 1:
                w = 1.0 - correct
            label = _pct(w)
            if label != answer and label not in distractors and 1 <= round(100 * w) <= 100:
                distractors.append(label)
        if len(distractors) < 3:
            raise RuntimeError(f"pot-odds spot {(pot, bet)} produced {distractors}")
        out.append({
            "id": f"found_pot_odds_{i:02d}", "unit": "pot_odds", "kind": "arithmetic",
            "prompt": (f"The pot is {pot:g} bb and your opponent bets {bet:g} bb. "
                       "What equity do you need to profitably call?"),
            "ask": (f"The pot is {pot:g} chips and your opponent bets {bet:g}. How often do "
                    "you need to win for calling to pay off?"),
            "options": _opts(answer, distractors, seed=1000 + i),
            "answer": answer,
            "explanation": (f"Calling costs {bet:g}. If you call, the final pot is "
                            f"{pot:g} + {bet:g} + {bet:g} = {pot + 2 * bet:g} chips. You break "
                            f"even when you win your share of it: {bet:g} ÷ {pot + 2 * bet:g} = "
                            f"{answer}. Win more often than that and calling makes money."),
            "data": {"pot": pot, "bet": bet, "break_even": round(correct, 4)},
        })
    return out


# --------------------------------------------------------------------------- #
# 3) Hand reading — made hand / draws from the evaluator (describe_hand)         #
# --------------------------------------------------------------------------- #

_HAND_SPOTS = [
    ("AhKh", "Qh8h3h"), ("As5s", "Ah7d2c"), ("7c7d", "As7h2d"), ("KdQd", "KsQh4c"),
    ("Jh Th".replace(" ", ""), "9h8h2c"), ("AcQc", "Qs9d4h"), ("6s6d", "As7h2d"),
    ("KhQs", "Jd Td 3c".replace(" ", "")), ("AhAd", "Ks8c2h"), ("9c8c", "Ah7c2c"),
    ("TsTh", "Ac Kd 5s".replace(" ", "")), ("JsJd", "Th9h2c"),
]
_HAND_POOL = ["high card", "top pair", "middle/bottom pair", "overpair", "pocket pair",
              "two pair", "three of a kind", "straight", "flush",
              "top pair + flush draw", "flush draw", "straight draw",
              "high card + flush draw", "high card + straight draw"]


# Each label piece -> (how to say it in a sentence, what it means), so the answer teaches the words.
_HAND_GLOSS = {
    "high card": ("no pair", "just your highest card"),
    "top pair": ("top pair", "one of your cards pairs the highest card on the board"),
    "middle/bottom pair": ("a middle or bottom pair", "one of your cards pairs a lower board card"),
    "overpair": ("an overpair", "a pocket pair higher than every board card"),
    "pocket pair": ("a pocket pair", "your pair is below the top board card"),
    "two pair": ("two pair", "two different pairs"),
    "three of a kind": ("three of a kind", "three cards of the same rank"),
    "straight": ("a straight", "five cards in a row"),
    "flush": ("a flush", "five cards of the same suit"),
    "full house": ("a full house", "three of a kind plus a pair"),
    "four of a kind": ("four of a kind", "all four cards of one rank"),
    "straight flush": ("a straight flush", "five in a row, all one suit"),
    "flush draw": ("a flush draw", "four cards of one suit, so one more makes a flush"),
    "straight draw": ("a straight draw", "four cards in a row, so the right next card makes "
                                         "a straight"),
}


def _says(piece: str) -> str:
    said, means = _HAND_GLOSS.get(piece, (piece, ""))
    return f"{said} ({means})" if means else said


def _hand_why(label: str) -> str:
    made, *draws = label.split(" + ")
    if not draws:
        return "You have " + _says(made) + "."
    lead = "No pair yet, but you have " if made == "high card" else "You have " + _says(made) + ", plus "
    return lead + " and ".join(_says(d) for d in draws) + "."


def hand_reading_questions() -> List[Dict]:
    out = []
    for i, (hand, board) in enumerate(_HAND_SPOTS):
        h = parse_hand(hand)
        b = parse_cards(board)
        ans = describe_hand(h, b)
        out.append({
            "id": f"found_hand_read_{i:02d}", "unit": "hand_reading", "kind": "evaluator",
            "prompt": f"You hold {_board_label(hand)} on {_board_label(board)}. What is your hand?",
            "ask": "What hand do you have?",
            "options": _opts(ans, _HAND_POOL, seed=2000 + i),
            "answer": ans,
            "explanation": _hand_why(ans),
            "data": {"hand": hand, "board": board},
        })
    return out


# --------------------------------------------------------------------------- #
# 4) Equity — exact showdown equity vs a specific hand, bucketed                 #
# --------------------------------------------------------------------------- #

_EQUITY_SPOTS = [
    ("AhKh", "JcTc", "Qh8h3h"), ("7c7d", "AhKd", "As7h2d"), ("QsQd", "Ah5h", "Kd9c4h"),
    ("AhKd", "7c7d", "Th9h8d"), ("KdQd", "As5s", "Ks8c2h"), ("9h8h", "AcAd", "7h6c2c"),
    ("AsQs", "KhKc", "Qd9d4c"), ("JhTh", "AcAd", "9h8h2c"),
]
_BANDS = [(0.0, 0.2, "0–20% (big underdog)"), (0.2, 0.4, "20–40% (behind)"),
          (0.4, 0.6, "40–60% (coin flip)"), (0.6, 0.8, "60–80% (ahead)"),
          (0.8, 1.01, "80–100% (big favorite)")]


def _band(eq: float) -> str:
    for lo, hi, label in _BANDS:
        if lo <= eq < hi:
            return label
    return _BANDS[-1][2]


def _equity_why(eq: float) -> str:
    """Name the exact figure AND the band it lands in, so an edge case like 20.2% can't read
    as "about 20%" while the graded answer is 20–40%."""
    # Round DOWN to 0.1 so the figure never crosses into the next band (19.96 -> "19.9", not
    # "20" next to "0–20%"); drop a trailing ".0" ("0", "100", "20" — never an empty string).
    pct = f"{math.floor(1000 * eq + 1e-9) / 10:.1f}"
    pct = pct[:-2] if pct.endswith(".0") else pct
    lo, hi = next((lo, min(hi, 1.0)) for lo, hi, lbl in _BANDS if lbl == _band(eq))
    # Flag a near-boundary figure (not the 0% / 100% ends of the scale, which aren't edges).
    near = (lo > 0 and eq - lo < 0.01) or (hi < 1.0 and hi - eq < 0.01)
    edge = " — just inside it" if near else ""
    return (f"Counting every possible turn and river, you win {pct}% of the time (a split pot "
            f"counts as half). That's in the {round(100 * lo)}–{round(100 * hi)}% range{edge}.")


def equity_questions() -> List[Dict]:
    out = []
    for i, (hero, vill, board) in enumerate(_EQUITY_SPOTS):
        h, v, b = parse_hand(hero), parse_hand(vill), parse_cards(board)
        # One pair vs one pair is 1,081 runouts. Enumerate them; sampling 200k was slower
        # and could not change the rounded percent.
        eq = float(equity_matrix(b, [h], [v])[0][0, 0])
        ans = _band(eq)
        out.append({
            "id": f"found_equity_{i:02d}", "unit": "equity", "kind": "exact",
            "prompt": (f"You hold {_board_label(hero)} on {_board_label(board)} against "
                       f"{_board_label(vill)}. Roughly what is your equity to the river?"),
            "ask": ("Your opponent shows their hand. If the turn and river are dealt, roughly "
                    "how often do you win?"),
            "options": [lbl for _, _, lbl in _BANDS],
            "answer": ans,
            "explanation": _equity_why(eq),
            "data": {"hero": hero, "villain": vill, "board": board, "equity": round(eq, 4)},
        })
    return out


GENERATORS = {
    "board_reading": board_reading_questions,
    "pot_odds": pot_odds_questions,
    "hand_reading": hand_reading_questions,
    "equity": equity_questions,
}


def generate_all() -> List[Dict]:
    out: List[Dict] = []
    for gen in GENERATORS.values():
        out.extend(gen())
    # invariant: answer must be one of the options (auto-gradeable)
    for q in out:
        assert q["answer"] in q["options"], f"{q['id']}: answer not in options"
    return out


def run(out_dir: str = "output/foundations") -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    qs = generate_all()
    with open(os.path.join(out_dir, "questions.json"), "w") as f:
        json.dump(qs, f, indent=1)
        f.write("\n")
    from collections import Counter
    by_unit = Counter(q["unit"] for q in qs)
    print(f"generated {len(qs)} foundation questions -> {out_dir}/questions.json")
    print("by unit:", dict(by_unit))
    return qs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/foundations")
    a = ap.parse_args()
    run(a.out)
