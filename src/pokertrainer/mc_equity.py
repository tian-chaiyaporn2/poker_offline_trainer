"""Independent Monte-Carlo equity check (MIT).

The exact equity engine (showdown.py) and both solvers share the same 7-card
evaluator + runout logic. To validate that shared layer *independently*, this
module estimates a single combo-vs-combo equity by random sampling of runouts —
a different code path (random sampling vs full enumeration). Agreement to within
sampling error confirms the enumerated equities are correct.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .evaluator import evaluate

Combo = Tuple[int, int]


def mc_equity(board: List[int], hero: Combo, villain: Combo,
              samples: int = 20000, seed: int = 7) -> float:
    """Random-sampling estimate of P(hero beats villain) + 0.5 P(tie)."""
    used = set(board) | set(hero) | set(villain)
    if len(used) != len(board) + 4:
        raise ValueError("card collision between board/hero/villain")
    deck = [c for c in range(52) if c not in used]
    rng = random.Random(seed)
    wins = 0.0
    for _ in range(samples):
        turn, river = rng.sample(deck, 2)
        full = board + [turn, river]
        h = evaluate([hero[0], hero[1], *full])
        v = evaluate([villain[0], villain[1], *full])
        wins += 1.0 if h > v else (0.5 if h == v else 0.0)
    return wins / samples
