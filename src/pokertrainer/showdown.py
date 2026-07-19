"""Showdown equity precomputation (MIT).

For a fixed flop and two combo lists, compute:

  equity[i, j]  -- P(OOP combo i beats IP combo j) + 0.5 * P(tie), averaged over
                   every turn+river runout that blocks neither combo.
  compat[i, j]  -- 1.0 if combos i and j share no card (can co-occur), else 0.0.

These are computed once per scenario (independent of CFR iterations), so the CFR
inner loop stays cheap. Full enumeration => exact and deterministic.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence, Tuple

import numpy as np

from .evaluator import evaluate

Combo = Tuple[int, int]


def _combo_cards(combos: Sequence[Combo]) -> np.ndarray:
    return np.array([[a, b] for a, b in combos], dtype=np.int64)


def compat_matrix(oop: Sequence[Combo], ip: Sequence[Combo]) -> np.ndarray:
    """compat[i,j] = 1.0 if oop[i] and ip[j] share no card, else 0.0."""
    n_o, n_i = len(oop), len(ip)
    m = np.ones((n_o, n_i), dtype=np.float64)
    oc = _combo_cards(oop)
    ic = _combo_cards(ip)
    for i in range(n_o):
        a, b = oc[i]
        blocked = (ic[:, 0] == a) | (ic[:, 1] == a) | (ic[:, 0] == b) | (ic[:, 1] == b)
        m[i, blocked] = 0.0
    return m


def equity_matrix(
    board: Sequence[int], oop: Sequence[Combo], ip: Sequence[Combo]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (equity, compat) matrices for a 3-card flop.

    equity[i,j] in [0,1] = OOP win probability (ties count 0.5) over all valid
    turn+river runouts. Pairs that share a card get equity 0.5 (never used;
    masked by compat).
    """
    board = list(board)
    assert len(board) == 3, "flop must be 3 cards"
    used_by_board = set(board)
    if len(used_by_board) != 3:
        raise ValueError("board cards must be unique")
    for label, combos in (("oop", oop), ("ip", ip)):
        for combo in combos:
            if combo[0] in used_by_board or combo[1] in used_by_board:
                raise ValueError(f"{label} combo {combo} collides with board")
            if combo[0] == combo[1]:
                raise ValueError(f"{label} combo has duplicate cards: {combo}")
    n_o, n_i = len(oop), len(ip)
    oc = _combo_cards(oop)
    ic = _combo_cards(ip)
    compat = compat_matrix(oop, ip)

    wins = np.zeros((n_o, n_i), dtype=np.float64)   # accumulates 1.0 win, 0.5 tie
    count = np.zeros((n_o, n_i), dtype=np.float64)   # valid runouts per pair

    deck = [c for c in range(52) if c not in used_by_board]

    oc0, oc1 = oc[:, 0], oc[:, 1]
    ic0, ic1 = ic[:, 0], ic[:, 1]
    rank_o = np.full(n_o, -1, dtype=np.int64)
    rank_i = np.full(n_i, -1, dtype=np.int64)

    # Precompute, per runout, the rank of every live combo (7-card best-of-5).
    for turn, river in combinations(deck, 2):
        b0, b1, b2 = board
        live_o = (oc0 != turn) & (oc1 != turn) & (oc0 != river) & (oc1 != river)
        live_i = (ic0 != turn) & (ic1 != turn) & (ic0 != river) & (ic1 != river)
        for i in np.nonzero(live_o)[0]:
            rank_o[i] = evaluate((oc0[i], oc1[i], b0, b1, b2, turn, river))
        for j in np.nonzero(live_i)[0]:
            rank_i[j] = evaluate((ic0[j], ic1[j], b0, b1, b2, turn, river))

        # Live pairs only; incompatible (card-sharing) pairs are masked later
        # via `compat`, so they need not be excluded here.
        lm = live_o[:, None] & live_i[None, :]
        ro = rank_o[:, None]
        ri = rank_i[None, :]
        wins += lm * ((ro > ri) + 0.5 * (ro == ri))
        count += lm

    with np.errstate(invalid="ignore", divide="ignore"):
        equity = np.where(count > 0, wins / count, 0.5)
    # Incompatible private pairs are unused by CFR (masked by compat), but the
    # API contract promises equity 0.5 for them.
    equity = np.where(compat > 0, equity, 0.5)
    return equity, compat
