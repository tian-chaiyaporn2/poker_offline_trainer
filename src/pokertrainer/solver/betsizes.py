"""Bet-size menu shared by the three multi-street solvers (MIT).

The legacy tree has exactly one bet size (`bet_frac`). Passing `bet_fracs` with
two or more sizes switches the solvers to the multi-size tree:

    OOP: check | bet_s1 | ... | bet_sK
      check -> IP: check | bet_s1 | ... | bet_sK
        check   -> advance
        bet_sk  -> OOP facing size k: fold | call (| raise) -> advance
      bet_sk -> IP facing size k: fold | call (| raise)     -> advance

Every size gets its own response node(s) (and its own raise-response node when
raise_x is set). `bet_fracs=None` or a single size keeps the legacy code path
untouched (bit-for-bit identical results, plain "bet" label).

Action labels: "bet_<round(100*frac)>", e.g. "bet_33", "bet_75". Response nodes are
named after the faced size ("bb_vs_bet_33"). Sizes are sorted ascending so the action
order is canonical regardless of the order passed in.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

MAX_SIZES = 9      # path tokens encode the size index as ONE digit (see solvers)


def bet_label(frac: float) -> str:
    return f"bet_{int(round(100 * frac))}"


def normalize_bet_fracs(bet_frac: float, bet_fracs: Optional[Sequence[float]],
                        pot_bb: float, eff_stack: Optional[float]
                        ) -> Tuple[List[float], List[str]]:
    """Validate the bet-size menu -> (sorted fracs, action labels).

    `bet_fracs=None` -> ([bet_frac], ["bet"]) (legacy single-size tree). A single size
    in `bet_fracs` is also the legacy tree with that size. Raises ValueError on
    non-positive / non-finite / duplicate sizes, on two sizes that share a label
    (e.g. 0.331 and 0.334 -> both "bet_33"), on more than MAX_SIZES sizes, and when two
    sizes cap to the same all-in amount at the street-entry root (eff_stack) — those
    would be the same action under two labels (identical EV, arbitrary frequency split),
    which is a config mistake for a content run. (Deeper streets, where a bigger
    accumulated pot can make two sizes both all-in, keep both actions: their subtrees are
    then identical, which is harmless to the equilibrium and never extracted.)
    """
    if bet_fracs is None:
        return [float(bet_frac)], ["bet"]
    fracs = [float(f) for f in bet_fracs]
    if not fracs:
        raise ValueError("bet_fracs must contain at least one size")
    for f in fracs:
        if not math.isfinite(f) or f <= 0:
            raise ValueError(f"bet sizes must be positive finite pot fractions, got {f!r}")
    fracs = sorted(fracs)
    if len(fracs) == 1:
        return fracs, ["bet"]
    if len(fracs) > MAX_SIZES:
        raise ValueError(f"at most {MAX_SIZES} bet sizes are supported, got {len(fracs)}")
    labels = [bet_label(f) for f in fracs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"bet sizes {fracs} are not distinct at 1% resolution "
                         f"(labels {labels})")
    if eff_stack:
        capped = [min(f * float(pot_bb), float(eff_stack)) for f in fracs]
        for a in range(len(fracs)):
            for b in range(a + 1, len(fracs)):
                if abs(capped[a] - capped[b]) < 1e-9:
                    raise ValueError(
                        f"bet sizes {fracs[a]} and {fracs[b]} both cap to the same all-in "
                        f"amount ({capped[a]}bb, eff_stack={eff_stack}) at the root — they "
                        f"would be one action under two labels; drop one")
    return fracs, labels
