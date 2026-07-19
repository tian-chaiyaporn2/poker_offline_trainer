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
import math
import os
import time
import traceback
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


def yield_report(all_recs, roots, hands_per_side, full_range_size) -> Dict:
    accepted = [r for r in all_recs if r["accepted"]]
    deduped = _dedup(accepted)
    per_node = Counter(r["node"] for r in accepted)
    mixed = sum(r["mixed"] for r in accepted)
    # Two different quantities:
    #  - RAW accepted records scale ~linearly with hands/side and with root count
    #    (each root contributes ~4 nodes x hands). Use this for the linear
    #    projection of available records.
    #  - DEDUPED records collapse to concepts (node x board_texture x hand_category
    #    x preferred), so they SATURATE across boards/hands rather than scaling
    #    linearly. Report separately as the diversity measure.
    scale = full_range_size / hands_per_side if hands_per_side else 1.0
    raw_per_root_full = (len(accepted) / roots * scale) if roots else 0
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
        "projected_raw_accepted_per_root_full_range": round(raw_per_root_full, 1),
        "projection_note": "raw records scale linearly; after concept-dedup the "
                           "usable count is bounded by distinct concepts x cap",
        "projection": {
            "raw_at_40_roots": round(raw_per_root_full * 40),
            "raw_at_60_roots": round(raw_per_root_full * 60),
            # raw records scale linearly with roots; a 40-root launch library
            # projects well past 1,200 (the concept-dedup ceiling is separately
            # large — see distinct_concepts_measured).
            "target_1200_met_at_40_roots": raw_per_root_full * 40 >= 1200,
        },
        "distinct_concepts_measured": concepts,
        "est_10q_sessions_raw_at_60_roots": round(raw_per_root_full * 60 / 10),
        "coverage": {
            "by_node": dict(per_node),
            "by_board_texture": dict(Counter(t for r in deduped for t in r["board_texture"])),
            "by_hand_category": dict(Counter(r["hand_category"] for r in deduped)),
            "by_decision_type": dict(Counter(r["decision_type"] for r in deduped)),
            "by_reason": dict(Counter(r["explanation"]["reason"] for r in deduped
                                      if "explanation" in r)),
        },
    }


# ---------------------------------------------------------------------------
# Checkpointing (so a long GPU run never loses completed boards to a late crash
# or Kaggle's session time-limit). Each board is solved, validated, then written
# to boards/board_<idx>.json; records.json + yield_report.json are refreshed
# after EVERY board so partial output is always present and downloadable. A
# re-run skips boards that already have a valid checkpoint (resume).
# ---------------------------------------------------------------------------

def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _is_finite_record(r: Dict) -> bool:
    ev, fr = r.get("ev", {}), r.get("freq", {})
    if not ev or not fr:
        return False
    if not (all(_finite(v) for v in ev.values()) and all(_finite(v) for v in fr.values())):
        return False
    # reach_mass / ev_sep_pct also become SQLite REAL columns; a NaN there would
    # be coerced to NULL and silently break the pack signature at the very end of
    # a long run. They're always finite when ev/freq are, but guard anyway.
    return all(_finite(r[k]) for k in ("reach_mass", "ev_sep_pct") if k in r)


def _mean_range_size(board_idx: List[int]) -> float:
    """Mean combos-per-side across the requested boards (full expanded ranges,
    no subsample) — the true 'full range' for the yield projection."""
    sizes: List[int] = []
    for i in board_idx:
        flop = parse_cards(BOARDS[i]["board"])
        sizes.append(len(expand_range(BB_SRP, flop)))
        sizes.append(len(expand_range(BTN_SRP, flop)))
    return sum(sizes) / len(sizes) if sizes else 0.0


def validate_records(recs: List[Dict]) -> List[str]:
    """Catch NaN/inf EVs and malformed strategies immediately, per board, so a
    numerically bad solve is flagged now instead of poisoning the final pack."""
    problems: List[str] = []
    if not recs:
        return ["no records produced for this board"]
    for r in recs:
        tag = f"{r.get('node')} {r.get('hand')}"
        ev = r.get("ev", {})
        fr = r.get("freq", {})
        if not ev or not fr:
            problems.append(f"{tag}: missing ev/freq")
            continue
        for a, v in ev.items():
            if not _finite(v):
                problems.append(f"{tag}: ev[{a}]={v}")
        for a, v in fr.items():
            if not _finite(v):
                problems.append(f"{tag}: freq[{a}]={v}")
        fs = sum(v for v in fr.values() if _finite(v))
        if not _finite(fs) or abs(fs - 1.0) > 0.02:
            problems.append(f"{tag}: freq sums to {fs:.4f} (expected 1.0)")
        if r.get("preferred") not in ev:
            problems.append(f"{tag}: preferred={r.get('preferred')!r} not in ev keys {list(ev)}")
    return problems


def _valid_checkpoint(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
        return isinstance(d, list) and len(d) > 0
    except Exception:
        return False


def _atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _aggregate(out: str, boards_dir: str, board_idx: List[int], hands_per_side: int,
               full_range_size: int) -> Dict:
    """Rebuild records.json + yield_report.json from whatever board checkpoints
    exist so far. Safe to call after every board and as an --aggregate-only pass."""
    all_recs: List[Dict] = []
    done: List[int] = []
    for i in board_idx:
        bpath = os.path.join(boards_dir, f"board_{i:02d}.json")
        if _valid_checkpoint(bpath):
            all_recs.extend(json.load(open(bpath)))
            done.append(i)
    _atomic_write_json(os.path.join(out, "records.json"), all_recs)
    if all_recs:
        rep = yield_report(all_recs, len(done), hands_per_side, full_range_size)
    else:
        rep = {"accepted": 0, "note": "no boards completed yet"}
    rep["boards_completed"] = done
    rep["boards_requested"] = list(board_idx)
    rep["boards_missing"] = [i for i in board_idx if i not in done]
    with open(os.path.join(out, "yield_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


def run(n=40, iters=300, roots=None, solver="cpu", dtype="float64",
        out="output/content_yield", full_range_size=250, pot=5.5, bet_frac=0.66,
        raise_x=None, fresh=False, aggregate_only=False):
    os.makedirs(out, exist_ok=True)
    boards_dir = os.path.join(out, "boards")
    os.makedirs(boards_dir, exist_ok=True)
    board_idx = list(roots) if roots is not None else list(range(len(BOARDS)))
    # True full range for the projection: when --n >= the range, we solve every
    # combo, so hands_per_side == full range and the scale factor is 1.0 (avoids
    # the old bug where --n 400 vs a ~250-combo range deflated projections ~1.6x).
    mean_full = _mean_range_size(board_idx)
    eff_full = int(round(mean_full)) if mean_full else full_range_size
    eff_hands = min(n, eff_full) if eff_full else n

    if not aggregate_only:
        make = _make_solver(solver, dtype, raise_x=raise_x)
        t0 = time.time()
        for k, i in enumerate(board_idx, 1):
            bstr = BOARDS[i]["board"]
            bpath = os.path.join(boards_dir, f"board_{i:02d}.json")
            if not fresh and _valid_checkpoint(bpath):
                cnt = len(json.load(open(bpath)))
                print(f"[{k}/{len(board_idx)}] board {i:02d} {bstr}: cached ({cnt} recs) — skip", flush=True)
                continue
            try:
                flop = parse_cards(bstr)
                oop = subsample([c for c, _ in expand_range(BB_SRP, flop)], n)
                ip = subsample([c for c, _ in expand_range(BTN_SRP, flop)], n)
                recs = extract_records(bstr, oop, ip, iters, make, pot, bet_frac)
                # Drop only the individual records with non-finite EV/freq (a
                # float32 blow-up on one hand must not discard the whole board's
                # 30-min solve, nor leak a NaN into the signed pack).
                clean = [r for r in recs if _is_finite_record(r)]
                dropped = len(recs) - len(clean)
                if not clean:
                    _atomic_write_json(os.path.join(boards_dir, f"board_{i:02d}.raw.json"), recs)
                    json.dump({"board": bstr, "reason": "no finite records",
                               "n_records": len(recs)},
                              open(os.path.join(boards_dir, f"board_{i:02d}.PROBLEM.json"), "w"), indent=2)
                    print(f"[{k}/{len(board_idx)}] board {i:02d} {bstr}: {len(recs)} recs but ALL "
                          f"non-finite — not checkpointed (see board_{i:02d}.PROBLEM.json)", flush=True)
                else:
                    _atomic_write_json(bpath, clean)
                    extra = f", {dropped} dropped NaN/inf" if dropped else ""
                    print(f"[{k}/{len(board_idx)}] board {i:02d} {bstr}: {len(clean)} recs "
                          f"({sum(r['accepted'] for r in clean)} accepted{extra}) OK "
                          f"[{time.time()-t0:.0f}s]", flush=True)
                    if dropped:
                        json.dump({"board": bstr, "dropped_non_finite": dropped, "kept": len(clean)},
                                  open(os.path.join(boards_dir, f"board_{i:02d}.DROPPED.json"), "w"), indent=2)
            except Exception:
                # one board crashing must not abort the run — log and continue so
                # the remaining boards still complete and get saved.
                with open(os.path.join(boards_dir, f"board_{i:02d}.ERROR.txt"), "w") as f:
                    f.write(traceback.format_exc())
                print(f"[{k}/{len(board_idx)}] board {i:02d} {bstr}: CRASHED — logged "
                      f"board_{i:02d}.ERROR.txt, continuing", flush=True)
            # refresh combined outputs after every board (survives a later timeout)
            _aggregate(out, boards_dir, board_idx, eff_hands, eff_full)

    rep = _aggregate(out, boards_dir, board_idx, eff_hands, eff_full)
    print("\n=== content-yield ===")
    print(f"boards completed: {rep['boards_completed']}  missing: {rep['boards_missing']}")
    if rep.get("accepted"):
        print(json.dumps({k: rep[k] for k in ("accepted", "accepted_deduped",
              "projected_raw_accepted_per_root_full_range", "projection",
              "distinct_concepts_measured", "per_node_accepted")}, indent=2))
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
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing board checkpoints and re-solve every board")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip solving; just rebuild records.json + yield_report.json "
                         "from existing board checkpoints")
    a = ap.parse_args()
    roots = [int(x) for x in a.roots.split(",")] if a.roots else None
    run(n=a.n, iters=a.iters, roots=roots, solver=a.solver, dtype=a.dtype,
        out=a.out, full_range_size=a.full_range_size, raise_x=a.raise_x,
        fresh=a.fresh, aggregate_only=a.aggregate_only)
