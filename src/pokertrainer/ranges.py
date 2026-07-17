"""Preflop range expansion (MIT).

Ranges are written in the standard poker grid notation:

    "AA"   -> the 6 pocket-ace combos
    "AKs"  -> the 4 suited ace-king combos
    "AKo"  -> the 12 offsuit ace-king combos
    "T9s"  -> suited ten-nine

Each class carries a weight in [0, 1]. `expand_range` turns a class->weight dict
into concrete (combo, weight) pairs, dropping any combo that collides with the
known board (card removal).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from .cards import RANKS, make_card

_RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}

Combo = Tuple[int, int]  # (high_card, low_card), high by (rank, suit)


def _combo(c1: int, c2: int) -> Combo:
    return (c1, c2) if c1 > c2 else (c2, c1)


def class_to_combos(hand_class: str) -> List[Combo]:
    """Expand a grid class like 'AKs', 'QQ', 'T9o' to concrete combos."""
    hand_class = hand_class.strip()
    r1 = _RANK_TO_IDX[hand_class[0].upper()]
    r2 = _RANK_TO_IDX[hand_class[1].upper()]
    combos: List[Combo] = []
    if r1 == r2:  # pocket pair
        for s1, s2 in combinations(range(4), 2):
            combos.append(_combo(make_card(r1, s1), make_card(r1, s2)))
        return combos
    suited = hand_class[2:].lower() == "s"
    offsuit = hand_class[2:].lower() == "o"
    if not (suited or offsuit):
        raise ValueError(f"non-pair class needs s/o suffix: {hand_class!r}")
    for s1 in range(4):
        for s2 in range(4):
            if suited and s1 != s2:
                continue
            if offsuit and s1 == s2:
                continue
            combos.append(_combo(make_card(r1, s1), make_card(r2, s2)))
    return combos


def expand_range(
    class_weights: Dict[str, float], board: List[int]
) -> List[Tuple[Combo, float]]:
    """Expand class->weight into [(combo, weight)], removing board collisions."""
    blocked = set(board)
    out: List[Tuple[Combo, float]] = []
    for hand_class, weight in class_weights.items():
        if weight <= 0:
            continue
        for combo in class_to_combos(hand_class):
            if combo[0] in blocked or combo[1] in blocked:
                continue
            out.append((combo, float(weight)))
    return out
