"""Light hand descriptors for human-readable training explanations (MIT)."""

from __future__ import annotations

from typing import List, Tuple

from .cards import card_rank, card_suit
from .evaluator import category_name, evaluate

Combo = Tuple[int, int]


def _has_flush_draw(cards: List[int]) -> bool:
    counts = [0, 0, 0, 0]
    for c in cards:
        counts[card_suit(c)] += 1
    return max(counts) == 4  # exactly four to a suit = draw (5+ is a made flush)


def _has_open_ended(cards: List[int]) -> bool:
    ranks = set(card_rank(c) for c in cards)
    if 12 in ranks:
        ranks.add(-1)  # wheel ace
    runs = 0
    for r in range(-1, 13):
        if r in ranks:
            runs += 1
            if runs >= 4:
                return True
        else:
            runs = 0
    return False


def describe_hand(hole: Combo, board: List[int]) -> str:
    """A short label like 'top pair', 'overpair', 'flush draw', 'ace high'."""
    five = list(hole) + list(board)
    made = category_name(evaluate(five)) if len(five) >= 5 else "high card"
    labels = [made]

    hole_ranks = sorted((card_rank(c) for c in hole), reverse=True)
    board_ranks = sorted((card_rank(c) for c in board), reverse=True)
    top_board = board_ranks[0] if board_ranks else -1

    if made == "one pair":
        # Distinguish overpair / top pair / lower.
        if hole_ranks[0] == hole_ranks[1]:  # pocket pair
            labels = ["overpair" if hole_ranks[0] > top_board else "pocket pair"]
        else:
            paired_rank = next((r for r in hole_ranks if r in board_ranks), None)
            if paired_rank == top_board:
                labels = ["top pair"]
            elif paired_rank is not None:
                labels = ["middle/bottom pair"]

    draws = []
    if _has_flush_draw(five) and "flush" not in made:
        draws.append("flush draw")
    if _has_open_ended(five) and "straight" not in made:
        draws.append("straight draw")
    return " + ".join(labels + draws)
