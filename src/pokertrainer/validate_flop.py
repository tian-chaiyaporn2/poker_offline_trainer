"""Flop-only vs full-street validation (MIT).

Executes the team's validation plan (docs/flop_training_validation_plan.md):
does the flop-only abstraction change the flop recommendation vs a full
flop->turn->river solve under IDENTICAL assumptions?

Design: both models use `streets=3` (turn+river always dealt). Flop-only is
`bet_streets=1` (flop betting, then pure runout); full-street is `bet_streets=3`.
They share the exact same betting tree, ranges, pot, stacks, bet size, and acting
player — the ONLY difference is whether turn/river betting follows the flop.

Stability (§7.5): the full-street model is snapshotted at a mid checkpoint and at
the end (same solve); any hand whose full-street preferred action flips between
them is marked unstable -> Red. This detects non-convergence per-hand.

Outputs: per-hand CSV (§11) + a JSON aggregate for the findings report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import traceback
from typing import Dict, List

import numpy as np

from .cards import parse_cards
from .handinfo import describe_hand
from .presets import BOARDS
from .ranges import expand_range
from .presets import BB_SRP, BTN_SRP
from .solver.batched import BatchedCFR
from .solver.batched_gpu import BatchedGPUCFR

# Classification thresholds (§8; engineering starting points).
GREEN_REGRET_PCT = 0.25
AMBER_REGRET_PCT = 1.0
INDIFF_PCT = 0.25          # full-street EV gap below this -> indifferent
CLEAR_GAP_PCT = 0.5        # each model's EV gap above this -> a "clear" preference
FREQ_MATERIAL_PP = 25.0    # bet-freq difference above this (pp) -> material


def subsample(lst, n):
    if n >= len(lst):
        return lst
    idx = np.linspace(0, len(lst) - 1, n).astype(int)
    return [lst[i] for i in idx]


def hand_category(descriptor: str) -> str:
    # Classify from the made-hand token only. Checking the full string matches
    # substrings inside draw labels (e.g. "high card + flush draw" → "flush").
    made = descriptor.split(" + ")[0]
    if any(k in made for k in ("two pair", "three of", "straight", "flush", "full",
                               "four of", "overpair")):   # "four of a kind" (evaluator name)
        return "strong_made"
    if "top pair" in made:
        return "top_pair"
    if "pair" in made:                      # middle/bottom/pocket underpair
        return "weak_pair"
    if "draw" in descriptor:
        return "draw"
    return "air"


def classify(agree, regret_pct, indiff, freq_pp, clear_disagree, unstable):
    if unstable:
        return "red"
    if regret_pct > AMBER_REGRET_PCT:
        return "red"
    if clear_disagree:
        return "red"
    if agree and regret_pct <= GREEN_REGRET_PCT and not indiff and freq_pp < FREQ_MATERIAL_PP:
        return "green"
    return "amber"


def _streets_for_board(flop) -> int:
    """Betting-street count for a starting board: flop=3, turn=2, river=1."""
    n = 6 - len(flop)
    if n not in (1, 2, 3):
        raise ValueError(f"unsupported board length {len(flop)}; need 3–5 cards")
    return n


def _make_solver(solver, dtype, raise_x=None):
    """Return a factory building a flop-root-reporting solver on cpu or gpu.

    bet_streets controls which streets have betting; raise_x enables fold/call/raise.
    n_streets is derived from the starting board length so turn/river demos deal
    the correct number of runout cards (not a hard-coded flop tree).
    """
    if solver == "gpu":
        return lambda f, o, i, wo, wi, pot, bf, bet_streets: BatchedGPUCFR(
            f, o, i, wo, wi, pot, bf, streets=_streets_for_board(f),
            bet_streets=bet_streets, backend="auto", dtype=dtype, raise_x=raise_x)
    return lambda f, o, i, wo, wi, pot, bf, bet_streets: BatchedCFR(
        f, o, i, wo, wi, pot, bf, streets=_streets_for_board(f),
        bet_streets=bet_streets, raise_x=raise_x)


def solve_board(flop, oop, ip, pot, bet_frac, iters, make):
    wo, wi = np.ones(len(oop)), np.ones(len(ip))
    # FLOP-ONLY = flop betting, then turn+river dealt as pure chance (no betting),
    # equity realized at showdown. (bet_streets=1; full 3-card->5-card runout.)
    s1 = make(flop, oop, ip, wo, wi, pot, bet_frac, 1)
    s1.run(max(200, iters))
    r1 = s1.flop_root_report()
    # FULL-STREET = betting on all three streets. Snapshot mid+end for stability.
    s3 = make(flop, oop, ip, wo, wi, pot, bet_frac, 3)
    half = iters // 2
    s3.run(half)
    r3_mid = s3.flop_root_report()
    s3.run(iters - half)
    r3 = s3.flop_root_report()
    return r1, r3, r3_mid


def validate(n=20, iters=280, pot=5.5, bet_frac=0.66, out="output/validation",
             solver="cpu", dtype="float64", max_boards=len(BOARDS), board_indices=None):
    os.makedirs(out, exist_ok=True)
    make = _make_solver(solver, dtype)
    rows: List[Dict] = []
    t0 = time.time()
    boards = ([BOARDS[i] for i in board_indices] if board_indices
              else BOARDS[:max_boards])

    failed = []
    for bi, entry in enumerate(boards, 1):
        board_str = entry["board"]
        try:
            flop = parse_cards(board_str)
            oop = subsample([c for c, _ in expand_range(BB_SRP, flop)], n)
            ip = subsample([c for c, _ in expand_range(BTN_SRP, flop)], n)
            r1, r3, r3_mid = solve_board(flop, oop, ip, pot, bet_frac, iters, make)

            for h in r1:
                fo, fs, fm = r1[h], r3[h], r3_mid[h]
                fo_pref, fs_pref = fo["preferred"], fs["preferred"]
                regret_bb = fs["ev"][fs_pref] - fs["ev"][fo_pref]
                regret_pct = 100 * regret_bb / pot
                fs_gap = abs(fs["ev"]["check"] - fs["ev"]["bet"])
                fo_gap = abs(fo["ev"]["check"] - fo["ev"]["bet"])
                indiff = (100 * fs_gap / pot) <= INDIFF_PCT
                agree = fo_pref == fs_pref
                clear_disagree = (not agree and not indiff
                                  and 100 * fo_gap / pot > CLEAR_GAP_PCT
                                  and 100 * fs_gap / pot > CLEAR_GAP_PCT)
                freq_pp = abs(fo["freq"]["bet"] - fs["freq"]["bet"]) * 100
                # Unstable = preferred action flipped between the mid and final
                # snapshots AND the decision is non-indifferent (a flip between two
                # near-equal actions is expected noise, not instability).
                unstable = (fm["preferred"] != fs_pref) and not indiff
                cls = classify(agree, regret_pct, indiff, freq_pp, clear_disagree, unstable)
                hc0, hc1 = parse_cards(h)
                desc = describe_hand((hc0, hc1), flop)
                rows.append({
                    "scenario_id": f"srp_btn_bb_100bb_flop_{board_str}",
                    "board": board_str,
                    "hand": h,
                    "board_category": ";".join(entry["categories"]),
                    "hand_category": hand_category(desc),
                    "hand_descriptor": desc,
                    "flop_only_preferred_action": fo_pref,
                    "full_street_preferred_action": fs_pref,
                    "flop_only_bet_freq": round(fo["freq"]["bet"], 4),
                    "full_street_bet_freq": round(fs["freq"]["bet"], 4),
                    "flop_only_ev_check": round(fo["ev"]["check"], 4),
                    "flop_only_ev_bet": round(fo["ev"]["bet"], 4),
                    "full_street_ev_check": round(fs["ev"]["check"], 4),
                    "full_street_ev_bet": round(fs["ev"]["bet"], 4),
                    "full_street_regret_bb": round(regret_bb, 4),
                    "full_street_regret_pct_pot": round(regret_pct, 4),
                    "preferred_action_agreement": agree,
                    "indifference_flag": indiff,
                    "stability_flag": "unstable" if unstable else "stable",
                    "suit_isomorphism_flag": "validated_globally",
                    "classification": cls,
                    "reviewer_status": "pending",
                    "reviewer_notes": "",
                    "publishable": cls == "green",
                })
        except Exception:
            failed.append(board_str)
            err_path = os.path.join(out, f"board_{bi:02d}.ERROR.txt")
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
            print(f"[{bi:2d}/{len(boards)}] {board_str:8s} CRASHED — logged "
                  f"{os.path.basename(err_path)}, continuing", flush=True)
            continue
        # Incremental write after each board so a long run survives interruption.
        _write(rows, out, dict(n=n, iters=iters, pot=pot, bet_frac=bet_frac,
                               boards_done=bi, boards_failed=failed))
        print(f"[{bi:2d}/{len(boards)}] {board_str:8s} done ({time.time()-t0:.0f}s "
              f"elapsed, {len(rows)} rows)", flush=True)

    if rows:
        _write(rows, out, dict(n=n, iters=iters, pot=pot, bet_frac=bet_frac,
                               boards_done=len(boards) - len(failed),
                               boards_failed=failed))
    print(f"\nTotal: {len(rows)} hand-decisions across {len(boards) - len(failed)}/"
          f"{len(boards)} boards in {time.time()-t0:.0f}s"
          + (f" (failed: {failed})" if failed else ""))


def _write(rows, out, cfg):
    if not rows:
        summary = {"config": cfg, "totals": {"hand_decisions": 0}, "note": "no boards completed"}
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("wrote summary.json (empty — no successful boards)")
        return
    # CSV (§11)
    csv_path = os.path.join(out, "flop_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Aggregate summary for the findings report
    def pct(sub, tot):
        return round(100 * sub / tot, 1) if tot else 0.0
    tot = len(rows)
    non_indiff = [r for r in rows if not r["indifference_flag"]]
    agree_ni = sum(r["preferred_action_agreement"] for r in non_indiff)
    greens = [r for r in rows if r["classification"] == "green"]
    ambers = [r for r in rows if r["classification"] == "amber"]
    reds = [r for r in rows if r["classification"] == "red"]
    regrets = [r["full_street_regret_pct_pot"] for r in rows]
    unstable = [r for r in rows if r["stability_flag"] == "unstable"]

    def by(key):
        cats = {}
        for r in rows:
            for c in (r[key].split(";") if key == "board_category" else [r[key]]):
                d = cats.setdefault(c, {"n": 0, "agree_ni": 0, "ni": 0, "green": 0,
                                        "amber": 0, "red": 0, "regrets": []})
                d["n"] += 1
                d["regrets"].append(r["full_street_regret_pct_pot"])
                d[r["classification"]] += 1
                if not r["indifference_flag"]:
                    d["ni"] += 1
                    d["agree_ni"] += int(r["preferred_action_agreement"])
        for c, d in cats.items():
            d["agreement_ni_pct"] = pct(d["agree_ni"], d["ni"])
            d["green_pct"] = pct(d["green"], d["n"])
            d["median_regret_pct"] = round(statistics.median(d["regrets"]), 3)
            d.pop("regrets")
        return cats

    bias_flop = statistics.mean(r["flop_only_bet_freq"] for r in rows)
    bias_full = statistics.mean(r["full_street_bet_freq"] for r in rows)

    summary = {
        "config": cfg,
        "totals": {
            "hand_decisions": tot,
            "non_indifferent": len(non_indiff),
            "agreement_all_pct": pct(sum(r["preferred_action_agreement"] for r in rows), tot),
            "agreement_non_indiff_pct": pct(agree_ni, len(non_indiff)),
            "green": len(greens), "amber": len(ambers), "red": len(reds),
            "green_pct": pct(len(greens), tot),
            "amber_pct": pct(len(ambers), tot),
            "red_pct": pct(len(reds), tot),
            "median_regret_pct_pot": round(statistics.median(regrets), 3),
            "mean_regret_pct_pot": round(statistics.mean(regrets), 3),
            "p90_regret_pct_pot": round(
                sorted(regrets)[min(len(regrets) - 1, max(0, int(np.ceil(0.9 * len(regrets)) - 1)))],
                3),
            "max_regret_pct_pot": round(max(regrets), 3),
            "unstable_count": len(unstable),
            "unstable_pct": pct(len(unstable), tot),
            "betting_freq_bias_pp": round(100 * (bias_flop - bias_full), 1),
            "flop_only_avg_bet_freq": round(bias_flop, 3),
            "full_street_avg_bet_freq": round(bias_full, 3),
        },
        "by_board_category": by("board_category"),
        "by_hand_category": by("hand_category"),
        # largest disagreements
        "top_disagreements": [
            {k: r[k] for k in ("board", "hand", "hand_category",
                               "flop_only_preferred_action", "full_street_preferred_action",
                               "full_street_regret_pct_pot", "stability_flag", "classification")}
            for r in sorted(rows, key=lambda r: -r["full_street_regret_pct_pot"])[:15]
        ],
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {csv_path} and summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--iters", type=int, default=280)
    ap.add_argument("--out", default="output/validation")
    ap.add_argument("--solver", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--max-boards", type=int, default=len(BOARDS))
    ap.add_argument("--board-indices", default=None,
                    help="comma-separated board indices, e.g. 0,2,4,8")
    a = ap.parse_args()
    bidx = [int(x) for x in a.board_indices.split(",")] if a.board_indices else None
    validate(n=a.n, iters=a.iters, out=a.out, solver=a.solver, dtype=a.dtype,
             max_boards=a.max_boards, board_indices=bidx)
