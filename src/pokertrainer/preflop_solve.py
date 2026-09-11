"""Attach solved preflop action EVs to chart spots (MIT).

Builds (or loads) the 169×169 class equity table, runs the 2-player raise-ladder
CFR (BTN opener vs BB defender — the heads-up-by-flop model), and returns
per-class action EVs at RFI / vs-open / vs-3bet. Chart spots stay the quiz
sample; recommended action becomes max-EV when the table is present.

This is CPU work (minutes for the table, seconds for CFR). Not a Kaggle hold.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from .preflop_equity import build_class_equity_table, hand_classes
from .solver.preflop import (
    PreflopCFR, combo_weights, load_equity_table, raise_ladder_game, type_bonus,
)

TABLE_PATH = os.path.join("output", "preflop", "equity_169.npz")

# Node keys in raise_ladder_game (action-name tuples).
_RFI_KEY = ("fold", "open")
_DEF_KEY = ("fold", "call", "3bet")
_VS3_KEY = ("fold", "call", "4bet")


def ensure_equity_table(path: str = TABLE_PATH, samples: int = 400,
                        seed: int = 7) -> Tuple[list, np.ndarray]:
    """Load the cached 169×169 table, or build it (CPU, a few minutes at samples=400)."""
    if os.path.exists(path):
        return load_equity_table(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    classes, E = build_class_equity_table(samples=samples, seed=seed)
    np.savez_compressed(path, classes=np.array(classes, dtype=object), E=E)
    return classes, E


def solve_action_evs(iters: int = 800, samples: int = 400,
                     path: str = TABLE_PATH) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return {node_kind: {hand_class: {action: ev_bb}}} for rfi / def / vs3bet."""
    classes, E = ensure_equity_table(path, samples=samples)
    w = combo_weights(classes)
    tb = type_bonus(classes)
    # Opener is IP (BTN vs BB). Realization is the documented modeled term.
    cfr = PreflopCFR(raise_ladder_game(), E, w, ip_player=0, realize=0.04, tb=tb)
    avg = cfr.run(iters=iters)
    evs = cfr.action_evs(avg)
    keys = cfr.nodes_by_actions()
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for kind, key in (("rfi", _RFI_KEY), ("def", _DEF_KEY), ("vs3bet", _VS3_KEY)):
        nid = keys[key]
        names = list(key)
        mat = evs[nid]
        out[kind] = {
            classes[i]: {names[k]: float(mat[i, k]) for k in range(len(names))}
            for i in range(len(classes))
        }
    return out


def apply_evs(actions, cls: str, ctx: str, ev_book: Dict, fallback) -> Dict[str, float]:
    """Look up solved EVs for this spot; fall back to the chart sentinels."""
    kind = "rfi" if ctx == "rfi" else ("vs3bet" if ctx == "vs3bet" else "def")
    solved = (ev_book.get(kind) or {}).get(cls)
    if not solved:
        return {a: fallback(a) for a in actions}
    return {a: float(solved[a]) for a in actions if a in solved}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--table", default=TABLE_PATH)
    a = ap.parse_args()
    book = solve_action_evs(iters=a.iters, samples=a.samples, path=a.table)
    aa = book["rfi"]["AA"]
    print("AA RFI", {k: round(v, 3) for k, v in aa.items()})
    print("72o RFI", {k: round(v, 3) for k, v in book["rfi"]["72o"].items()})
    print("wrote/used", a.table)
    print(json.dumps({k: len(v) for k, v in book.items()}))
