"""Content prioritization (PRD roadmap) — MIT.

Ranks candidate lessons by **how much they're worth teaching**, so board/runout/
lesson selection is data-driven instead of guessed. Score per record combines
three axes, each read from data the solve already produces:

  frequency  = P(board texture) x reach_mass    -- how often the spot occurs
  impact     = (best_ev - worst_ev) / pot       -- how costly the mistake is
  intuition  = reason -> gap in [0,1]            -- how non-obvious the GTO play is

Each axis is percentile-ranked across all records, then combined by weight
(default 0.4 / 0.4 / 0.2). Output is two backlogs:
  * lesson_backlog -- spot-types (node x texture x hand-category x reason) ranked
    by total teaching value: what to surface first.
  * solve_backlog  -- board textures ranked by real-world frequency, flagged by
    how well the current records cover them: what to SOLVE next.

CLI:  PYTHONPATH=src python -m pokertrainer.priority --records <records.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from .cards import card_str
from .content_yield import board_texture

# How counterintuitive the GTO play is to a casual player (teaching value is
# highest where intuition fails). Obvious plays low, advanced/mixed plays high.
INTUITION = {
    "value": 0.20, "value_call": 0.30, "fold": 0.15, "realization": 0.35,
    "protection": 0.60, "call_odds": 0.60, "raise_value": 0.60,
    "pot_control": 0.70, "semi_bluff": 0.70, "bluff": 0.75, "raise_semibluff": 0.75,
    "bluff_catch": 0.80, "mixed": 0.85, "trap": 0.90, "raise_bluff": 0.90,
}
DEFAULT_WEIGHTS = {"frequency": 0.4, "impact": 0.4, "intuition": 0.2}

_TEXTURE_FREQ: Dict[Tuple[str, ...], float] = {}
_TEXTURE_SAMPLE: Dict[Tuple[str, ...], str] = {}


def flop_texture_freqs() -> Dict[Tuple[str, ...], float]:
    """P(texture tuple) over all C(52,3)=22,100 flops (real-world occurrence)."""
    if _TEXTURE_FREQ:
        return _TEXTURE_FREQ
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    total = 0
    for flop in combinations(range(52), 3):
        t = tuple(board_texture(list(flop)))
        counts[t] += 1
        total += 1
        if t not in _TEXTURE_SAMPLE:
            _TEXTURE_SAMPLE[t] = " ".join(card_str(c) for c in flop)
    for t, c in counts.items():
        _TEXTURE_FREQ[t] = c / total
    return _TEXTURE_FREQ


def _pctrank(values: List[float]) -> List[float]:
    """Percentile rank in [0,1] (average ties) for combining unlike scales."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _reason(r: Dict) -> str:
    return (r.get("explanation") or {}).get("reason", "value")


def score_records(records: List[Dict], pot: float = 5.5, weights=None) -> List[Dict]:
    """Attach frequency/impact/intuition components + combined priority to each
    accepted record. Returns the scored (accepted) records."""
    weights = weights or DEFAULT_WEIGHTS
    freqs = flop_texture_freqs()
    recs = [r for r in records if r.get("accepted", True)]
    if not recs:
        return []
    raw_freq, raw_impact, raw_intu = [], [], []
    for r in recs:
        tex = tuple(r.get("board_texture", []))
        evs = list(r["ev"].values())
        raw_freq.append(freqs.get(tex, 0.0) * float(r.get("reach_mass", 0.0)))
        raw_impact.append((max(evs) - min(evs)) / pot if evs else 0.0)
        raw_intu.append(INTUITION.get(_reason(r), 0.5))
    fr, im, it = _pctrank(raw_freq), _pctrank(raw_impact), _pctrank(raw_intu)
    for i, r in enumerate(recs):
        r["priority"] = round(weights["frequency"] * fr[i]
                              + weights["impact"] * im[i]
                              + weights["intuition"] * it[i], 4)
        r["priority_parts"] = {"frequency": round(fr[i], 3), "impact": round(im[i], 3),
                               "intuition": round(it[i], 3),
                               "freq_raw": round(raw_freq[i], 5),
                               "impact_pct": round(100 * raw_impact[i], 2)}
    return recs


def lesson_backlog(scored: List[Dict], top=40) -> List[Dict]:
    """Spot-types ranked by TOTAL teaching value (what to surface/teach first)."""
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in scored:
        key = (r["node"], tuple(r.get("board_texture", [])), r["hand_category"], _reason(r))
        groups[key].append(r)
    rows = []
    for (node, tex, hc, reason), g in groups.items():
        total = sum(x["priority"] for x in g)
        rows.append({
            "node": node, "board_texture": list(tex), "hand_category": hc, "reason": reason,
            "n_records": len(g),
            "total_value": round(total, 3),
            "mean_priority": round(total / len(g), 3),
            "mean_impact_pct": round(sum(x["priority_parts"]["impact_pct"] for x in g) / len(g), 2),
        })
    rows.sort(key=lambda x: -x["total_value"])
    return rows[:top]


def solve_backlog(records: List[Dict], top=25) -> List[Dict]:
    """Board textures ranked by real-world frequency, annotated with how well the
    current records cover them: high-frequency + low-coverage = solve next."""
    freqs = flop_texture_freqs()
    covered_boards = defaultdict(set)     # texture -> set of board strings present
    covered_recs = defaultdict(int)
    for r in records:
        if not r.get("accepted", True):
            continue
        tex = tuple(r.get("board_texture", []))
        covered_boards[tex].add(r.get("board"))
        covered_recs[tex] += 1
    rows = []
    for tex, p in freqs.items():
        rows.append({
            "board_texture": list(tex),
            "occurrence_pct": round(100 * p, 3),
            "boards_solved": len(covered_boards.get(tex, ())),
            "records": covered_recs.get(tex, 0),
            "example_board": _TEXTURE_SAMPLE.get(tex, ""),
            "covered": tex in covered_boards,
        })
    # rank by frequency among UNDER-covered textures first, then by frequency
    rows.sort(key=lambda x: (x["boards_solved"] > 0, -x["occurrence_pct"]))
    return rows[:top]


def build_report(records: List[Dict], pot: float = 5.5, weights=None) -> Dict:
    scored = score_records(records, pot, weights)
    scored_sorted = sorted(scored, key=lambda r: -r["priority"])
    top_lessons = [{
        "priority": r["priority"], "parts": r["priority_parts"],
        "node": r["node"], "board": r.get("board"), "hand": r.get("hand"),
        "hand_category": r["hand_category"], "preferred": r["preferred"],
        "reason": _reason(r),
        "headline": (r.get("explanation") or {}).get("headline", ""),
    } for r in scored_sorted[:30]]
    return {
        "params": {"pot": pot, "weights": weights or DEFAULT_WEIGHTS},
        "n_accepted": len(scored),
        "top_lessons": top_lessons,
        "lesson_backlog": lesson_backlog(scored),
        "solve_backlog": solve_backlog(records),
        "coverage": {
            "textures_covered": len({tuple(r.get("board_texture", []))
                                     for r in records if r.get("accepted", True)}),
            "textures_total": len(flop_texture_freqs()),
        },
    }


def run(records_path: str, out_dir: str = "output/priority", pot: float = 5.5) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    records = json.load(open(records_path))
    rep = build_report(records, pot=pot)
    with open(os.path.join(out_dir, "priority_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"scored {rep['n_accepted']} accepted records from {records_path}")
    print(f"texture coverage: {rep['coverage']['textures_covered']}/{rep['coverage']['textures_total']}\n")
    print("== TOP LESSONS (frequency x impact x intuition) ==")
    for r in rep["top_lessons"][:10]:
        p = r["parts"]
        print(f"  {r['priority']:.3f}  {r['node']:<13} {r['board']} {r['hand']} "
              f"[{r['reason']}]  freq={p['frequency']:.2f} impact={p['impact']:.2f}({p['impact_pct']:.0f}%) intu={p['intuition']:.2f}")
    print("\n== LESSON BACKLOG (spot-types by total value) ==")
    for b in rep["lesson_backlog"][:10]:
        print(f"  {b['total_value']:6.2f}  {b['node']:<13} {'/'.join(b['board_texture'])[:22]:<22} "
              f"{b['hand_category']:<14} {b['reason']:<12} (n={b['n_records']}, impact~{b['mean_impact_pct']:.0f}%)")
    print("\n== SOLVE BACKLOG (common textures, coverage) ==")
    for s in rep["solve_backlog"][:10]:
        flag = "" if s["boards_solved"] else "  <-- UNCOVERED"
        print(f"  {s['occurrence_pct']:5.2f}%  {'/'.join(s['board_texture']):<34} "
              f"solved={s['boards_solved']} recs={s['records']} eg={s['example_board']}{flag}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="records.json from content_yield")
    ap.add_argument("--out", default="output/priority")
    ap.add_argument("--pot", type=float, default=5.5)
    a = ap.parse_args()
    run(a.records, a.out, a.pot)
