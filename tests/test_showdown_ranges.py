"""Showdown equity, ranges, and card-removal tests (MIT)."""

import numpy as np

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.mc_equity import mc_equity
from pokertrainer.ranges import class_to_combos, expand_range
from pokertrainer.showdown import compat_matrix, equity_matrix


def test_class_expansion_counts():
    assert len(class_to_combos("AA")) == 6      # pocket pair
    assert len(class_to_combos("AKs")) == 4     # suited
    assert len(class_to_combos("AKo")) == 12    # offsuit


def test_board_removal():
    board = parse_cards("AsKd2c")
    combos = expand_range({"AA": 1.0}, board)   # As removed -> 3 AA combos left
    assert len(combos) == 3


def test_compat_blocking():
    oop = [parse_hand("AsAd")]
    ip = [parse_hand("AsKh"), parse_hand("KhQh")]
    C = compat_matrix(oop, ip)
    assert C[0, 0] == 0.0    # shares As
    assert C[0, 1] == 1.0    # disjoint


def test_equity_matches_monte_carlo():
    board = parse_cards("2c7dTh")
    hero = parse_hand("AsAd")
    vill = parse_hand("KsKd")
    eq, _ = equity_matrix(board, [hero], [vill])
    mc = mc_equity(board, hero, vill, samples=40000, seed=3)
    assert abs(float(eq[0, 0]) - mc) < 0.01     # within sampling error


def test_suit_isomorphism_equity():
    """PRD §6: equivalent suit arrangements produce equivalent results.

    Apply a global suit permutation to board + both combos; enumerated equity
    must be identical."""
    board = parse_cards("As7h2d")
    oop = [parse_hand(h) for h in ["AhKh", "7s7c", "Ts9s"]]
    ip = [parse_hand(h) for h in ["QdQc", "Kh Qh".replace(" ", ""), "9c8d"]]
    eq1, _ = equity_matrix(board, oop, ip)

    # Permute suits: s<->h, d<->c  (a valid relabeling of all four suits)
    perm = {0: 1, 1: 0, 2: 3, 3: 2}   # suit indices c,d,h,s = 0,1,2,3

    def relabel(card):
        return (card // 4) * 4 + perm[card % 4]

    board2 = [relabel(c) for c in board]
    oop2 = [tuple(sorted((relabel(a), relabel(b)), reverse=True)) for a, b in oop]
    ip2 = [tuple(sorted((relabel(a), relabel(b)), reverse=True)) for a, b in ip]
    eq2, _ = equity_matrix(board2, oop2, ip2)
    assert np.allclose(eq1, eq2)
