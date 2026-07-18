"""Automated comparison driver (Deliverable 6) — MIT.

Runs our CFR+ solver against the independent reference CFR on all boards, adds an
independent Monte-Carlo equity spot-check and a determinism/stability check, and
writes output/comparison_report.json + a markdown summary.

Usage:
    python -m pokertrainer.benchmark [--iterations N] [--ref-iterations M]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

from .compare import compare
from .mc_equity import mc_equity
from .normalize import from_solver_arrays
from .presets import BOARDS, build_scenario
from .reference_solver import ReferenceCFR
from .scenario import load_scenario
from .showdown import equity_matrix
from .solver import FlopSolver


def _mc_spotcheck(scn, equity, n_pairs=6, seed=11) -> Dict:
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_pairs):
        i = rng.randrange(len(scn.oop_combos))
        j = rng.randrange(len(scn.ip_combos))
        hero, vill = scn.oop_combos[i], scn.ip_combos[j]
        if set(hero) & set(vill):
            continue
        est = mc_equity(scn.board, hero, vill, samples=15000, seed=seed)
        diffs.append(abs(est - float(equity[i, j])))
    return {"pairs": len(diffs), "max_abs_diff": round(max(diffs), 4) if diffs else 0.0,
            "mean_abs_diff": round(sum(diffs) / len(diffs), 4) if diffs else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--ref-iterations", type=int, default=4000)  # vanilla CFR is slower
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    reports: List[Dict] = []
    for i, entry in enumerate(BOARDS, 1):
        raw = build_scenario(entry, iterations=args.iterations)
        scn = load_scenario(raw)
        equity, compat = equity_matrix(scn.board, scn.oop_combos, scn.ip_combos)

        # Our CFR+ (twice on the same equity -> determinism check)
        def solve_cfr_plus():
            s = FlopSolver(equity, compat, scn.w_oop, scn.w_ip,
                           scn.pot_bb, scn.small_frac, scn.large_frac)
            return s.solve(iterations=args.iterations)
        r1 = solve_cfr_plus()
        r2 = solve_cfr_plus()
        det_diff = abs(r1.root_ev_oop_bb - r2.root_ev_oop_bb)
        a = from_solver_arrays(scn.id, raw["board"], scn.oop_combos,
                               r1.strategies["root"], r1.action_ev["root"],
                               scn.w_oop, r1.root_ev_oop_bb, "cfr_plus",
                               r1.runtime_sec, r1.peak_mem_mb)

        # Independent reference (vanilla CFR)
        t0 = time.time()
        ref = ReferenceCFR(equity, compat, scn.w_oop, scn.w_ip,
                           scn.pot_bb, scn.small_frac, scn.large_frac)
        avg = ref.solve(args.ref_iterations)
        ref_rt = time.time() - t0
        b = from_solver_arrays(scn.id, raw["board"], scn.oop_combos,
                               avg["root"], ref.root_action_ev(avg),
                               scn.w_oop, ref.root_ev_bb(avg), "reference_cfr", ref_rt)

        rep = compare(a, b, scn.pot_bb)
        rep["determinism_root_ev_diff_bb"] = round(det_diff, 8)
        rep["mc_equity_spotcheck"] = _mc_spotcheck(scn, equity)
        reports.append(rep)

        print(f"[{i:2d}/12] {''.join(rep['board']):8s} "
              f"EV diff {rep['root_ev']['diff_pct_pot']:+.3f}% pot | "
              f"agree {rep['preferred_action_agreement']*100:5.1f}% "
              f"({rep['non_indifferent_decisions']} non-indiff) | "
              f"freqΔ {rep['avg_freq_diff_pp_non_indiff']:.2f}pp | "
              f"MCΔ {rep['mc_equity_spotcheck']['max_abs_diff']:.3f} | "
              f"{'PASS' if rep['all_targets_pass'] else 'CHECK'}")

    _write_reports(reports, args)


def _write_reports(reports: List[Dict], args) -> None:
    n = len(reports)
    agg = {
        "boards": n,
        "all_pass": sum(r["all_targets_pass"] for r in reports),
        "max_root_ev_diff_pct_pot": max(abs(r["root_ev"]["diff_pct_pot"]) for r in reports),
        "min_agreement": min(r["preferred_action_agreement"] for r in reports),
        "max_avg_freq_diff_pp": max(r["avg_freq_diff_pp_non_indiff"] for r in reports),
        "max_mc_equity_diff": max(r["mc_equity_spotcheck"]["max_abs_diff"] for r in reports),
        "max_determinism_diff_bb": max(r["determinism_root_ev_diff_bb"] for r in reports),
        "total_major_disagreements": sum(len(r["major_disagreements"]) for r in reports),
    }
    with open(os.path.join(args.out, "comparison_report.json"), "w") as f:
        json.dump({"aggregate": agg, "iterations": args.iterations,
                   "ref_iterations": args.ref_iterations, "boards": reports}, f, indent=2)

    lines = ["# Automated Comparison Report (CFR+ vs independent reference CFR)\n",
             f"Boards: {n} · All-targets pass: {agg['all_pass']}/{n}\n",
             "| Board | EV diff (%pot) | Agreement | Non-indiff | freqΔ (pp) | MC eqΔ | Pass |",
             "|-------|----------------|-----------|-----------|-----------|--------|------|"]
    for r in reports:
        lines.append(
            f"| {''.join(r['board'])} | {r['root_ev']['diff_pct_pot']:+.3f} | "
            f"{r['preferred_action_agreement']*100:.1f}% | "
            f"{r['non_indifferent_decisions']} | {r['avg_freq_diff_pp_non_indiff']:.2f} | "
            f"{r['mc_equity_spotcheck']['max_abs_diff']:.3f} | "
            f"{'✅' if r['all_targets_pass'] else '⚠️'} |")
    lines += ["",
              "## Aggregate", "",
              f"- Max root-EV difference: **{agg['max_root_ev_diff_pct_pot']:.3f}% of pot** (target < 1%)",
              f"- Min preferred-action agreement: **{agg['min_agreement']*100:.1f}%** (target ≥ 90%)",
              f"- Max avg frequency diff (non-indiff): **{agg['max_avg_freq_diff_pp']:.2f} pp** (target < 5pp)",
              f"- Major disagreements (strong vs strong): **{agg['total_major_disagreements']}** (target 0)",
              f"- Max MC-vs-enumerated equity diff: **{agg['max_mc_equity_diff']:.3f}** (sampling error ~0.004)",
              f"- Max determinism EV diff across repeated CFR+ runs: **{agg['max_determinism_diff_bb']:.2e} bb**",
              ""]
    with open(os.path.join(args.out, "comparison_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nAggregate: {agg['all_pass']}/{n} boards pass all targets. "
          f"Max EV diff {agg['max_root_ev_diff_pct_pot']:.3f}% pot, "
          f"min agreement {agg['min_agreement']*100:.1f}%.")


if __name__ == "__main__":
    main()
