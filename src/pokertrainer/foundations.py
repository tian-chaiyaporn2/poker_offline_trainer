"""Foundations content generators (PRD v1.3 §6 fundamentals) — MIT.

Turns the `foundation_template` seeds (board reading, pot odds, hand reading,
equity) into concrete, auto-gradeable practice questions. Everything here is
**deterministic** (fixed inputs + seeds) so a generated set is reproducible and
can go into a signed pack. Nothing invents strategy — each answer is computed
from the same primitives the solver pipeline uses (evaluator, board texture,
pot-odds arithmetic, Monte-Carlo equity).

Each question is a dict:
    {id, unit, kind, prompt, options:[...], answer, explanation, data:{...}}
`answer` is always one of `options`, so the trainer grades by exact match.

CLI:  PYTHONPATH=src python -m pokertrainer.foundations --out output/foundations
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

from .cards import parse_cards, parse_hand, hand_str, card_str
from .content_yield import board_texture
from .handinfo import describe_hand
from .mc_equity import mc_equity
from .presets import BOARDS


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
            "options": list(_SUIT_Q.values()),
            "answer": _SUIT_Q[suit],
            "explanation": f"{lbl} is {_SUIT_Q[suit].lower()} — flush possibilities scale with shared suits.",
            "data": {"board": bstr, "tags": tags},
        })
        paired = "paired" in tags
        out.append({
            "id": f"found_board_pair_{bi:02d}", "unit": "board_reading", "kind": "pairing",
            "prompt": f"Is the flop {lbl} paired?",
            "options": ["Paired", "Unpaired"],
            "answer": "Paired" if paired else "Unpaired",
            "explanation": ("Two of the three cards share a rank." if paired
                            else "All three ranks are distinct."),
            "data": {"board": bstr, "tags": tags},
        })
        connected = "connected" in tags
        out.append({
            "id": f"found_board_conn_{bi:02d}", "unit": "board_reading", "kind": "connectedness",
            "prompt": f"Is the flop {lbl} connected (coordinated for straights)?",
            "options": ["Connected", "Disconnected"],
            "answer": "Connected" if connected else "Disconnected",
            "explanation": ("The ranks are close enough to make straights likely." if connected
                            else "The ranks are spread out, so straights are unlikely."),
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
        # common wrong calcs: bet/(pot+bet) (ignores your call in the pot) and bet/pot
        wrong1 = bet / (pot + bet)
        wrong2 = bet / pot
        wrong3 = bet / (pot + 3 * bet)
        answer = _pct(correct)
        distractors = [_pct(w) for w in (wrong1, wrong2, wrong3)]
        out.append({
            "id": f"found_pot_odds_{i:02d}", "unit": "pot_odds", "kind": "arithmetic",
            "prompt": (f"The pot is {pot:g} bb and your opponent bets {bet:g} bb. "
                       "What equity do you need to profitably call?"),
            "options": _opts(answer, distractors, seed=1000 + i),
            "answer": answer,
            "explanation": (f"You risk {bet:g} to win {pot + bet:g}; break-even = "
                            f"bet / (pot + 2·bet) = {bet:g}/{pot + 2 * bet:g} = {answer}."),
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


def hand_reading_questions() -> List[Dict]:
    out = []
    for i, (hand, board) in enumerate(_HAND_SPOTS):
        h = parse_hand(hand)
        b = parse_cards(board)
        ans = describe_hand(h, b)
        out.append({
            "id": f"found_hand_read_{i:02d}", "unit": "hand_reading", "kind": "evaluator",
            "prompt": f"You hold {_board_label(hand)} on {_board_label(board)}. What is your hand?",
            "options": _opts(ans, _HAND_POOL, seed=2000 + i),
            "answer": ans,
            "explanation": f"{_board_label(hand)} on {_board_label(board)} makes: {ans}.",
            "data": {"hand": hand, "board": board},
        })
    return out


# --------------------------------------------------------------------------- #
# 4) Equity — Monte-Carlo equity vs a specific hand, bucketed (montecarlo)       #
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


def equity_questions() -> List[Dict]:
    out = []
    for i, (hero, vill, board) in enumerate(_EQUITY_SPOTS):
        h, v, b = parse_hand(hero), parse_hand(vill), parse_cards(board)
        eq = mc_equity(b, h, v, samples=200000, seed=3000 + i)
        ans = _band(eq)
        out.append({
            "id": f"found_equity_{i:02d}", "unit": "equity", "kind": "montecarlo",
            "prompt": (f"You hold {_board_label(hero)} on {_board_label(board)} against "
                       f"{_board_label(vill)}. Roughly what is your equity to the river?"),
            "options": [lbl for _, _, lbl in _BANDS],
            "answer": ans,
            "explanation": f"Enumerated/Monte-Carlo equity is about {round(100 * eq)}% — {ans}.",
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
