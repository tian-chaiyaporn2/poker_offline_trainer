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
    # top ranges 12..3; top==3 covers the wheel A-2-3-4-5 via the -1 sentinel.
    for top in range(12, 2, -1):
        if all((top - i) in present for i in range(5)):
            return top
    return -1


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


def evaluate_ref(cards: Sequence[int]) -> int:
    """Reference evaluator: best 5-card score by brute force. Used to validate
    the fast path in tests. Correct but slow."""
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


# --- Fast bitmask evaluator (hot path for the solver) ---------------------

# Straight windows over rank bits 0..12; value is the straight-high rank index.
_STRAIGHT_WINDOWS = [(0b11111 << (high - 4), high) for high in range(12, 3, -1)]
_WHEEL_MASK = (1 << 12) | (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0)  # A-2-3-4-5


def _straight_high_from_mask(mask: int) -> int:
    """Highest straight-high rank present in a 13-bit rank mask, or -1."""
    for window, high in _STRAIGHT_WINDOWS:
        if (mask & window) == window:
            return high
    if (mask & _WHEEL_MASK) == _WHEEL_MASK:
        return 3  # five-high wheel
    return -1


def _top_bits(mask: int, k: int) -> List[int]:
    """The k highest set bit positions (ranks), descending."""
    out: List[int] = []
    for r in range(12, -1, -1):
        if mask & (1 << r):
            out.append(r)
            if len(out) == k:
                break
    return out


def evaluate(cards: Sequence[int]) -> int:
    """Evaluate 5, 6, or 7 cards -> best 5-card score (higher better).

    Fast path using rank/suit bitmasks; no per-combination looping.
    Produces the same packed scores as `evaluate5`/`evaluate_ref`.
    """
    if len(cards) < 5:
        raise ValueError("need at least 5 cards")

    rank_count = [0] * 13
    suit_masks = [0, 0, 0, 0]
    all_mask = 0
    for c in cards:
        r = c >> 2          # card // 4
        s = c & 3           # card % 4
        rank_count[r] += 1
        suit_masks[s] |= 1 << r
        all_mask |= 1 << r

    # Flush / straight flush.
    flush_suit = -1
    for s in range(4):
        if suit_masks[s].bit_count() >= 5:
            flush_suit = s
            break
    if flush_suit >= 0:
        sf_high = _straight_high_from_mask(suit_masks[flush_suit])
        if sf_high >= 0:
            return _pack(STRAIGHT_FLUSH, [sf_high])

    # Group ranks by multiplicity.
    quads = [r for r in range(12, -1, -1) if rank_count[r] == 4]
    trips = [r for r in range(12, -1, -1) if rank_count[r] == 3]
    pairs = [r for r in range(12, -1, -1) if rank_count[r] == 2]

    if quads:
        q = quads[0]
        kicker = _top_bits(all_mask & ~(1 << q), 1)[0]
        return _pack(QUADS, [q, kicker])

    if trips and (len(trips) >= 2 or pairs):
        t = trips[0]
        # best available pair: a second trip counts as a pair.
        pair_candidates = [r for r in trips[1:]] + pairs
        p = max(pair_candidates)
        return _pack(FULL_HOUSE, [t, p])

    if flush_suit >= 0:
        return _pack(FLUSH, _top_bits(suit_masks[flush_suit], 5))

    straight_high = _straight_high_from_mask(all_mask)
    if straight_high >= 0:
        return _pack(STRAIGHT, [straight_high])

    if trips:
        t = trips[0]
        kickers = _top_bits(all_mask & ~(1 << t), 2)
        return _pack(TRIPS, [t, *kickers])

    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        kicker = _top_bits(all_mask & ~(1 << hi) & ~(1 << lo), 1)[0]
        return _pack(TWO_PAIR, [hi, lo, kicker])

    if pairs:
        p = pairs[0]
        kickers = _top_bits(all_mask & ~(1 << p), 3)
        return _pack(ONE_PAIR, [p, *kickers])

    return _pack(HIGH_CARD, _top_bits(all_mask, 5))


def category_of(score: int) -> int:
    """Recover the hand category from a packed score."""
    return score // (16 ** 5)


def category_name(score: int) -> str:
    return _CATEGORY_NAMES[category_of(score)]
