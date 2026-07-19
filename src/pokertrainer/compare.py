"""Automated solver comparison (Deliverable 6, PRD §5 Step 6, §6 targets) — MIT.

Compares two NormalizedSolve objects (same scenario, same abstraction) and
reports the PRD metrics. The comparison prioritises EV agreement over exact
frequency matching, and only judges preferred-action agreement on
*non-indifferent* decisions (EV gap between best and 2nd-best > threshold).
"""

from __future__ import annotations

from typing import Dict, List

from .normalize import CANON_ACTIONS, NormalizedSolve

# PRD §6 initial targets
TARGET_ROOT_EV_DIFF_PCT = 1.0        # root EV within ~1% of pot
TARGET_AGREEMENT = 0.90              # >=90% agreement on non-indifferent spots
TARGET_FREQ_DIFF_PP = 5.0            # avg aggregate freq diff < 5pp (non-indiff)
INDIFF_THRESHOLD_PCT = 1.0           # non-indifferent if best-2nd EV gap > 1% pot
MAJOR_DISAGREE_PCT = 5.0             # "strongly" profitable/losing threshold


def _best_and_gap(ev: Dict[str, float]):
    ranked = sorted(CANON_ACTIONS, key=lambda a: ev[a], reverse=True)
    best, second = ranked[0], ranked[1]
    return best, ev[best] - ev[second]


def compare(a: NormalizedSolve, b: NormalizedSolve, pot_bb: float) -> Dict:
    indiff_bb = INDIFF_THRESHOLD_PCT / 100.0 * pot_bb
    major_bb = MAJOR_DISAGREE_PCT / 100.0 * pot_bb

    if a.scenario_id != b.scenario_id:
        raise ValueError(f"scenario_id mismatch: {a.scenario_id!r} vs {b.scenario_id!r}")
    if a.board != b.board:
        raise ValueError(f"board mismatch: {a.board} vs {b.board}")
    if a.actions != b.actions:
        raise ValueError(f"actions mismatch: {a.actions} vs {b.actions}")

    hands_a = set(a.per_hand)
    hands_b = set(b.per_hand)
    shared = sorted(hands_a & hands_b)
    missing_in_b = sorted(hands_a - hands_b)
    missing_in_a = sorted(hands_b - hands_a)
    if not shared:
        raise ValueError("compare() requires at least one shared hand")

    root_ev_diff = a.root_ev_oop_bb - b.root_ev_oop_bb

    non_indiff = 0
    agree = 0
    freq_diffs: List[float] = []
    ev_diffs: List[float] = []
    disagreements: List[Dict] = []
    major: List[Dict] = []
    prob_ok = True

    for h in shared:
        sa, sb = a.per_hand[h]["strategy"], b.per_hand[h]["strategy"]
        ea, eb = a.per_hand[h]["ev"], b.per_hand[h]["ev"]
        if abs(sum(sa.values()) - 1) > 1e-3 or abs(sum(sb.values()) - 1) > 1e-3:
            prob_ok = False
        if any(v < -1e-9 or v > 1.0 + 1e-9 for v in sa.values()) or \
           any(v < -1e-9 or v > 1.0 + 1e-9 for v in sb.values()):
            prob_ok = False

        # max per-action EV difference for this hand
        ev_diffs.append(max(abs(ea[x] - eb[x]) for x in CANON_ACTIONS))

        best_a, gap_a = _best_and_gap(ea)
        best_b, gap_b = _best_and_gap(eb)
        # A decision is non-indifferent if EITHER solver sees a clear best.
        if gap_a > indiff_bb or gap_b > indiff_bb:
            non_indiff += 1
            # aggregate frequency difference on this decision (pp, summed abs / 2)
            fd = 0.5 * sum(abs(sa[x] - sb[x]) for x in CANON_ACTIONS) * 100
            freq_diffs.append(fd)
            if best_a == best_b:
                agree += 1
            else:
                disagreements.append({
                    "hand": h, "cfr_plus_best": best_a, "reference_best": best_b,
                    "gap_a_pct_pot": round(100 * gap_a / pot_bb, 2),
                    "gap_b_pct_pot": round(100 * gap_b / pot_bb, 2),
                })
                # Major: A strongly prefers X while B strongly prefers Y (both clear)
                if gap_a > major_bb and gap_b > major_bb:
                    major.append({"hand": h, "cfr_plus": best_a, "reference": best_b})

    agreement = (agree / non_indiff) if non_indiff else 1.0
    avg_freq_diff = (sum(freq_diffs) / len(freq_diffs)) if freq_diffs else 0.0
    max_ev_diff = max(ev_diffs) if ev_diffs else 0.0
    mean_ev_diff = (sum(ev_diffs) / len(ev_diffs)) if ev_diffs else 0.0

    passes = {
        "root_ev_within_target": abs(root_ev_diff) <= TARGET_ROOT_EV_DIFF_PCT / 100 * pot_bb,
        "agreement_target": agreement >= TARGET_AGREEMENT,
        "freq_diff_target": avg_freq_diff <= TARGET_FREQ_DIFF_PP,
        "no_major_disagreement": len(major) == 0,
        "probabilities_valid": prob_ok,
        "hands_complete": not missing_in_a and not missing_in_b,
    }
    return {
        "scenario_id": a.scenario_id,
        "board": a.board,
        "root_ev": {
            "cfr_plus_bb": round(a.root_ev_oop_bb, 4),
            "reference_bb": round(b.root_ev_oop_bb, 4),
            "diff_bb": round(root_ev_diff, 4),
            "diff_pct_pot": round(100 * root_ev_diff / pot_bb, 4),
        },
        "range_freqs": {"cfr_plus": a.range_freqs, "reference": b.range_freqs},
        "hands": {
            "shared": len(shared),
            "missing_in_reference": missing_in_b,
            "missing_in_cfr_plus": missing_in_a,
        },
        "action_ev_diff_bb": {"max": round(max_ev_diff, 4), "mean": round(mean_ev_diff, 4)},
        "non_indifferent_decisions": non_indiff,
        "preferred_action_agreement": round(agreement, 4),
        "avg_freq_diff_pp_non_indiff": round(avg_freq_diff, 3),
        "disagreements": disagreements,
        "major_disagreements": major,
        "runtime_sec": {"cfr_plus": a.runtime_sec, "reference": b.runtime_sec},
        "peak_mem_mb": {"cfr_plus": a.peak_mem_mb},
        "targets_pass": passes,
        "all_targets_pass": all(passes.values()),
    }
