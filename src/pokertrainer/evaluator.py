"""Hand evaluator (MIT, our own code).

Ranks any 5–7 card poker hand into a single integer where **higher is better**,
so two hands can be compared directly. This is deliberately dependency-free so
there is no third-party evaluator licence to audit (see docs/licenses.md).

The score packs a hand category (0=high card .. 8=straight flush) with up to
five rank kickers in base-16, guaranteeing correct ordering.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

from .cards import card_rank, card_suit

# Hand categories, higher is better.
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

_CATEGORY_NAMES = {
    HIGH_CARD: "high card",
    ONE_PAIR: "one pair",
    TWO_PAIR: "two pair",
    TRIPS: "three of a kind",
    STRAIGHT: "straight",
    FLUSH: "flush",
    FULL_HOUSE: "full house",
    QUADS: "four of a kind",
    STRAIGHT_FLUSH: "straight flush",
}


def _pack(category: int, kickers: Sequence[int]) -> int:
    """Pack category + up to 5 kickers (each 0..14) into one int."""
    value = category
    for k in kickers:
        value = value * 16 + k
    # Pad to a fixed width so categories with fewer kickers still compare right.
    for _ in range(5 - len(kickers)):
        value *= 16
    return value


def _straight_high(rank_set: frozenset) -> int:
    """Return the high rank index of the best straight, or -1 if none.

    Ranks are 0..12 (2..A). Wheel A-2-3-4-5 is handled by treating the ace as
    also filling the "below 2" slot; its straight-high is 3 (the five)."""
    # Ace (12) can act as low for the wheel.
    present = set(rank_set)
    if 12 in present:
        present.add(-1)  # ace-low sentinel, one below the deuce (0)
    high = -1
    for top in range(12, 2, -1):  # top of a 5-straight ranges 12..3
        if all((top - i) in present for i in range(5)):
            return top
    # explicit wheel check (top == 3 uses -1 sentinel)
    if all(r in present for r in (-1, 0, 1, 2, 3)):
        high = 3
    return high


def evaluate5(cards: Sequence[int]) -> int:
    """Evaluate exactly 5 cards -> comparable int (higher better)."""
    ranks = sorted((card_rank(c) for c in cards), reverse=True)
    suits = [card_suit(c) for c in cards]
    is_flush = len(set(suits)) == 1
    rank_set = frozenset(ranks)
    straight_high = _straight_high(rank_set)

    # Count rank multiplicities.
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Sort ranks by (count, rank) descending -> canonical kicker order.
    by_group = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(c for _, c in by_group)
    ordered_ranks = [r for r, _ in by_group]

    if is_flush and straight_high >= 0:
        return _pack(STRAIGHT_FLUSH, [straight_high])
    if shape == (4, 1):
        return _pack(QUADS, ordered_ranks)          # [quad_rank, kicker]
    if shape == (3, 2):
        return _pack(FULL_HOUSE, ordered_ranks)      # [trip_rank, pair_rank]
    if is_flush:
        return _pack(FLUSH, ranks)                    # 5 kickers
    if straight_high >= 0:
        return _pack(STRAIGHT, [straight_high])
    if shape == (3, 1, 1):
        return _pack(TRIPS, ordered_ranks)
    if shape == (2, 2, 1):
        return _pack(TWO_PAIR, ordered_ranks)         # [hi_pair, lo_pair, kicker]
    if shape == (2, 1, 1, 1):
        return _pack(ONE_PAIR, ordered_ranks)
    return _pack(HIGH_CARD, ranks)


def evaluate(cards: Sequence[int]) -> int:
    """Evaluate 5, 6, or 7 cards -> best 5-card score (higher better)."""
    n = len(cards)
    if n == 5:
        return evaluate5(cards)
    if n < 5:
        raise ValueError("need at least 5 cards")
    best = 0
    for combo in combinations(cards, 5):
        score = evaluate5(combo)
        if score > best:
            best = score
    return best


def category_of(score: int) -> int:
    """Recover the hand category from a packed score."""
    return score // (16 ** 5)


def category_name(score: int) -> str:
    return _CATEGORY_NAMES[category_of(score)]
