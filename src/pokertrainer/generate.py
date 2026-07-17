"""Generate the full POC library: solve all 12 boards, export questions (MIT).

Usage:
    python -m pokertrainer.generate [--iterations N] [--per-board K] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

from .export import build_questions, write_json, write_sqlite
from .presets import BOARDS, build_scenario
from .runner import run_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--per-board", type=int, default=10)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    solves_dir = os.path.join(args.out, "solves")
    os.makedirs(solves_dir, exist_ok=True)

    all_questions: List[Dict] = []
    summary: List[Dict] = []
    t_all = time.time()

    for i, entry in enumerate(BOARDS, 1):
        scn = build_scenario(entry, iterations=args.iterations)
        t0 = time.time()
        solve = run_scenario(scn)
        wall = time.time() - t0
        with open(os.path.join(solves_dir, f"{solve['scenario_id']}.json"), "w") as f:
            json.dump(solve, f, indent=2)

        questions = build_questions(solve, max_per_board=args.per_board)
        all_questions.extend(questions)

        conv = solve["convergence"]
        row = {
            "board": "".join(solve["board"]),
            "categories": entry["categories"],
            "combos": f"{solve['n_oop_combos']}x{solve['n_ip_combos']}",
            "root_ev_pct_pot": solve["root_ev_oop_pct_pot"],
            "exploit_pct_pot": conv["final_exploitability_pct_pot"],
            "range_freqs": solve["range_action_frequencies"],
            "questions": len(questions),
            "wall_sec": round(wall, 2),
        }
        summary.append(row)
        print(f"[{i:2d}/12] {row['board']:8s} {str(entry['categories']):45s} "
              f"combos {row['combos']:9s} EV {row['root_ev_pct_pot']:5.1f}%  "
              f"expl {row['exploit_pct_pot']:.4f}%  q={row['questions']}  "
              f"{row['wall_sec']:.1f}s")

    write_json(all_questions, os.path.join(args.out, "questions.json"))
    write_sqlite(all_questions, os.path.join(args.out, "trainer.db"))
    with open(os.path.join(args.out, "generation_summary.json"), "w") as f:
        json.dump({
            "total_questions": len(all_questions),
            "total_wall_sec": round(time.time() - t_all, 2),
            "iterations": args.iterations,
            "boards": summary,
        }, f, indent=2)

    print(f"\nTotal: {len(all_questions)} questions across {len(BOARDS)} boards "
          f"in {time.time() - t_all:.1f}s")
    worst = max(summary, key=lambda r: r["exploit_pct_pot"])
    print(f"Worst-converged board: {worst['board']} "
          f"({worst['exploit_pct_pot']:.4f}% pot exploitability)")


if __name__ == "__main__":
    main()
