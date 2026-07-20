"""Pre-flop all-in equity (MIT).

Board-free hot-and-cold equity: P(hero beats villain) + 0.5*P(tie) over a full
random 5-card runout. Unlike `mc_equity` (which deals a turn+river onto an existing
flop), this deals all five community cards, so it is the pre-flop matchup engine the
pre-flop solver stands on. Card-removal aware.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .evaluator import evaluate
from .ranges import class_to_combos

Combo = Tuple[int, int]

# The 169 canonical starting-hand classes, strongest-ish first is not required — this
# is just the label set (pairs, suited, offsuit).
_R = "AKQJT98765432"  # high -> low for readable class names


def hand_classes() -> List[str]:
    classes: List[str] = []
    for i, hi in enumerate(_R):
        for j, lo in enumerate(_R):
            if i == j:
                classes.append(hi + lo)          # pocket pair, e.g. "AA"
            elif i < j:
                classes.append(hi + lo + "s")     # suited, e.g. "AKs"
            else:
                classes.append(lo + hi + "o")     # offsuit, e.g. "AKo"
    return classes


def preflop_equity(hero: Combo, villain: Combo,
                   samples: int = 10000, seed: int = 7) -> float:
    """P(hero beats villain) + 0.5*P(tie) over a random 5-card board."""
    used = set(hero) | set(villain)
    if len(used) != 4:
        raise ValueError("card collision between hero and villain")
    deck = [c for c in range(52) if c not in used]
    rng = random.Random(seed)
    wins = 0.0
    for _ in range(samples):
        board = rng.sample(deck, 5)
        h = evaluate([hero[0], hero[1], *board])
        v = evaluate([villain[0], villain[1], *board])
        wins += 1.0 if h > v else (0.5 if h == v else 0.0)
    return wins / samples


def class_equity(class_a: str, class_b: str,
                 samples: int = 4000, seed: int = 7) -> float:
    """Average pre-flop equity of hand-class A vs hand-class B, averaged over every
    non-colliding combo pairing (so e.g. AA-vs-AKs correctly accounts for the shared
    aces via card removal)."""
    a_combos = class_to_combos(class_a)
    b_combos = class_to_combos(class_b)
    total, n = 0.0, 0
    for ai, a in enumerate(a_combos):
        for bi, b in enumerate(b_combos):
            if set(a) & set(b):
                continue  # card collision — impossible matchup
            total += preflop_equity(a, b, samples=samples, seed=seed + 31 * ai + bi)
            n += 1
    if n == 0:
        raise ValueError(f"no valid matchup for {class_a} vs {class_b}")
    return total / n
