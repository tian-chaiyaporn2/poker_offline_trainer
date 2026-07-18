"""Content-yield gate (PRD v1.3 §5.3, §9.4) — MIT.

Measures **accepted full-street flop decision records per solved root** and
projects them against the launch targets (≥1,200 accepted records; ≥30 distinct
10-question sessions; coverage across nodes, board families, hand categories).

For each flop root it runs a full-street solve, extracts all four flop decision
nodes (both players, root + facing a bet), tags each record, applies the §9.3
acceptance rules, and aggregates. Reduced-range/CPU runs are for pipeline
validation + projection only; the authoritative gate needs full ranges on GPU
(§13.1) — run via colab/kaggle_content_yield.ipynb.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .cards import card_rank, card_suit, parse_cards, parse_hand
from .explanations import explain
from .handinfo import describe_hand
from .presets import BOARDS
from .ranges import expand_range
from .presets import BB_SRP, BTN_SRP
from .validate_flop import _make_solver, hand_category, subsample

# Acceptance thresholds (§9.3; engineering starting points).
MIN_REACH = 0.05          # node must be practically reached (opp mass)
CLEAR_SEP_PCT = 0.5       # EV gap above this = a clear (scorable) lesson; else "mixed"
DEDUP_CAP = 30            # max accepted records per (node, board_cat, hand_cat, preferred)


def board_texture(flop: List[int]) -> List[str]:
    ranks = sorted((card_rank(c) for c in flop), reverse=True)
    suits = [card_suit(c) for c in flop]
    tags = []
    tags.append("paired" if len(set(ranks)) < 3 else "unpaired")
    nsuit = len(set(suits))
    tags.append("monotone" if nsuit == 1 else "two_tone" if nsuit == 2 else "rainbow")
    tags.append("high_card" if ranks[0] >= 8 else "low")           # T=8
    span = ranks[0] - ranks[2]
    tags.append("connected" if span <= 4 else "disconnected")
    return tags


def extract_records(flop_str, oop, ip, iters, make, pot, bet_frac) -> List[Dict]:
    flop = parse_cards(flop_str)
    s = make(flop, oop, ip, np.ones(len(oop)), np.ones(len(ip)), pot, bet_frac, 3)
    res = s.run(iters)
    ev_pct = res.get("root_ev_pct_pot", 50.0)     # BB (OOP) share of pot
    board_favored = "BTN" if ev_pct < 45 else ("BB" if ev_pct > 55 else None)
    recs = s.flop_decisions_report()
    btags = board_texture(flop)
    for r in recs:
        r["board"] = flop_str
        r["board_texture"] = btags
        r["board_favored"] = board_favored
        r["hand_category"] = hand_category(describe_hand(parse_hand(r["hand"]), flop))
        r["decision_type"] = "first_action" if r["node"] in ("bb_first", "btn_vs_check") else "vs_bet"
        top2 = sorted(r["ev"].values(), reverse=True)[:2]     # best vs 2nd-best action
        r["ev_sep_pct"] = round(100 * (top2[0] - top2[1]) / pot, 3)
        r["mixed"] = r["ev_sep_pct"] < CLEAR_SEP_PCT
        # Accepted: practically reached (unstable/reduced-range handled elsewhere).
        r["accepted"] = r["reach_mass"] >= MIN_REACH
        r["explanation"] = explain(r, board_favored)
    # free GPU memory between roots (prevents pool accumulation / OOM)
    xp = getattr(s, "xp", None)
    backend = getattr(s, "backend", "")
    del s
    if xp is not None and backend == "cupy":
        try:
            import gc
            gc.collect()
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    return recs


def _dedup(recs: List[Dict]) -> List[Dict]:
    """Cap near-identical concepts so a few hands don't dominate a lesson."""
    buckets = defaultdict(list)
    for r in recs:
        key = (r["node"], tuple(r["board_texture"]), r["hand_category"], r["preferred"])
        buckets[key].append(r)
    out = []
    for key, group in buckets.items():
        out.extend(group[:DEDUP_CAP])
    return out


def yield_report(all_recs, n_solved, roots, hands_per_side, full_range_size) -> Dict:
    accepted = [r for r in all_recs if r["accepted"]]
    deduped = _dedup(accepted)
    per_node = Counter(r["node"] for r in accepted)
    mixed = sum(r["mixed"] for r in accepted)
    # projection: accepted scales ~linearly with hands/side (records = 4 nodes x hands)
    scale = full_range_size / hands_per_side if hands_per_side else 1.0
    acc_per_root = len(deduped) / roots if roots else 0
    proj_per_root_full = acc_per_root * scale
    concepts = len({(r["node"], tuple(r["board_texture"]), r["hand_category"], r["preferred"])
                    for r in deduped})
    return {
        "config": {"roots_solved": roots, "hands_per_side": hands_per_side,
                   "full_range_size": full_range_size, "note": "reduced-range projection"
                   if hands_per_side < full_range_size else "full-range"},
        "records_raw": len(all_recs),
        "accepted": len(accepted),
        "accepted_deduped": len(deduped),
        "accepted_rate": round(len(accepted) / len(all_recs), 3) if all_recs else 0,
        "mixed_share": round(mixed / len(accepted), 3) if accepted else 0,
        "per_node_accepted": dict(per_node),
        "mean_reach_by_node": {n: round(float(np.mean([r["reach_mass"] for r in accepted if r["node"] == n])), 3)
                               for n in per_node},
        "accepted_per_root_deduped": round(acc_per_root, 1),
        "projected_accepted_per_root_full_range": round(proj_per_root_full, 1),
        "projection": {
            "at_40_roots": round(proj_per_root_full * 40),
            "at_60_roots": round(proj_per_root_full * 60),
            "target_1200_met_at_40": proj_per_root_full * 40 >= 1200,
            "target_1200_met_at_60": proj_per_root_full * 60 >= 1200,
        },
        "distinct_concepts": concepts,
        "est_10q_sessions_at_60_roots": round(proj_per_root_full * 60 / 10),
        "coverage": {
            "by_node": dict(per_node),
            "by_board_texture": dict(Counter(t for r in deduped for t in r["board_texture"])),
            "by_hand_category": dict(Counter(r["hand_category"] for r in deduped)),
            "by_decision_type": dict(Counter(r["decision_type"] for r in deduped)),
            "by_reason": dict(Counter(r["explanation"]["reason"] for r in deduped
                                      if "explanation" in r)),
        },
    }


def run(n=40, iters=300, roots=None, solver="cpu", dtype="float64",
        out="output/content_yield", full_range_size=250, pot=5.5, bet_frac=0.66,
        raise_x=None):
    os.makedirs(out, exist_ok=True)
    make = _make_solver(solver, dtype, raise_x=raise_x)
    board_list = [BOARDS[i]["board"] for i in roots] if roots else [b["board"] for b in BOARDS]
    all_recs: List[Dict] = []
    t0 = time.time()
    for k, bstr in enumerate(board_list, 1):
        flop = parse_cards(bstr)
        oop = subsample([c for c, _ in expand_range(BB_SRP, flop)], n)
        ip = subsample([c for c, _ in expand_range(BTN_SRP, flop)], n)
        recs = extract_records(bstr, oop, ip, iters, make, pot, bet_frac)
        all_recs.extend(recs)
        print(f"[{k}/{len(board_list)}] {bstr}: {len(recs)} records "
              f"({sum(r['accepted'] for r in recs)} accepted) [{time.time()-t0:.0f}s]", flush=True)

    rep = yield_report(all_recs, len(all_recs), len(board_list), n, full_range_size)
    with open(os.path.join(out, "yield_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    with open(os.path.join(out, "records.json"), "w") as f:
        json.dump(all_recs, f)
    print("\n=== content-yield ===")
    print(json.dumps({k: rep[k] for k in ("accepted_per_root_deduped",
          "projected_accepted_per_root_full_range", "projection",
          "distinct_concepts", "per_node_accepted")}, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--roots", default=None, help="comma-separated board indices")
    ap.add_argument("--solver", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--full-range-size", type=int, default=250)
    ap.add_argument("--raise-x", type=float, default=None,
                    help="enable fold/call/raise; raise-to multiple of the bet, e.g. 3")
    ap.add_argument("--out", default="output/content_yield")
    a = ap.parse_args()
    roots = [int(x) for x in a.roots.split(",")] if a.roots else None
    run(n=a.n, iters=a.iters, roots=roots, solver=a.solver, dtype=a.dtype,
        out=a.out, full_range_size=a.full_range_size, raise_x=a.raise_x)
