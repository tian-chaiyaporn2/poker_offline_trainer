"""Calibrated 6-max pre-flop ranges (MIT).

Our own strength+playability ORDERING (from the equity table + realization model), cut to
standard per-position WIDTHS. This is 'solver-approximate, tuned to standard frequencies'
— NOT the exact-CFR guarantee the postflop packs carry. Opening ranges are nested
(a later seat opens a superset of an earlier one), matching how real charts behave.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .solver.preflop import load_equity_table, combo_weights, type_bonus

# Standard 6-max, 100bb, raise-first-in opening frequencies (%). Common-knowledge targets.
RFI_FREQ: Dict[str, float] = {
    "UTG": 16.0, "HJ": 20.0, "CO": 27.0, "BTN": 48.0, "SB": 44.0,
}
POSITIONS: List[str] = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]


def strength_order(classes: List[str], E: np.ndarray, tb: np.ndarray) -> List[int]:
    """Class indices from best to worst by equity-vs-field + playability bonus."""
    wn = combo_weights(classes)
    wn = wn / wn.sum()
    score = (E @ wn) + tb                      # raw strength + how well it plays postflop
    return sorted(range(len(classes)), key=lambda i: -score[i])


def strength_cumulative() -> Dict[str, float]:
    """{hand_class: cumulative combo-% when hands are added strongest-first}. Used to tell
    how close a hand sits to a frequency threshold (near a cutoff => a mixed/close spot)."""
    classes, E = load_equity_table()
    w = combo_weights(classes)
    order = strength_order(classes, E, type_bonus(classes))
    tot, acc, cum = w.sum(), 0.0, {}
    for i in order:
        acc += w[i]
        cum[classes[i]] = float(100.0 * acc / tot)
    return cum


def _top_pct(classes, w, order, pct) -> List[str]:
    target = pct / 100.0 * w.sum()
    acc, out = 0.0, []
    for i in order:
        out.append(classes[i])
        acc += w[i]
        if acc >= target:
            break
    return out


# BB defending vs a single open: (total defend %, value-3bet %) by opener seat. BB defends
# tighter vs early opens (stronger ranges) and wide vs the SB (where BB is in position).
# Linear 3-bet (value the top of the defense) — the clearest lesson for a trainer.
BB_DEFENSE: Dict[str, tuple] = {
    "UTG": (28.0, 7.0), "HJ": (33.0, 8.0), "CO": (40.0, 9.0),
    "BTN": (55.0, 11.0), "SB": (66.0, 13.0),
}


def bb_defense_ranges() -> Dict[str, Dict[str, str]]:
    """Per opener seat -> {hand_class: '3bet'|'call'|'fold'} for the Big Blind."""
    classes, E = load_equity_table()
    w = combo_weights(classes)
    tb = type_bonus(classes)
    order = strength_order(classes, E, tb)
    out: Dict[str, Dict[str, str]] = {}
    for opener, (dfd, tb3) in BB_DEFENSE.items():
        threebet = set(_top_pct(classes, w, order, tb3))
        defend = set(_top_pct(classes, w, order, dfd))
        out[opener] = {c: ("3bet" if c in threebet else "call" if c in defend else "fold")
                       for c in classes}
    return out


def rfi_ranges() -> Dict[str, Dict[str, str]]:
    """Per opening position -> {hand_class: 'open'|'fold'}. BB has no RFI (it defends)."""
    classes, E = load_equity_table()
    w = combo_weights(classes)
    tb = type_bonus(classes)
    order = strength_order(classes, E, tb)
    out: Dict[str, Dict[str, str]] = {}
    for pos, pct in RFI_FREQ.items():
        opens = set(_top_pct(classes, w, order, pct))
        out[pos] = {c: ("open" if c in opens else "fold") for c in classes}
    return out
