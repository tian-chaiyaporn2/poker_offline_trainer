"""Explanation generation (PRD v1.3 §10.1, §6.4, FR-012) — MIT.

Turns a validated flop decision record into a plain-language reason. Casual
language first: one practical headline; EVs/frequencies are expandable detail.
The `reason` tag is drawn from the app's reason-classification set so it can
drive the "identify the main reason" question (FR-012).

Nothing here invents strategy — the recommended action and EVs come from the
full-street solve; this only *labels and phrases* them with poker-standard
heuristics (hand strength × action × board texture × node).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

# Reason tags. First-action (bet/check) reasons align with FR-012's set;
# response (fold/call) nodes add bluff-catch / odds / fold.
HEADLINES = {
    "value":         "Bet for value — you're ahead of the hands that call, so get money in.",
    "protection":    "Bet to protect — vulnerable to draws and overcards, so charge them.",
    "bluff":         "Bet as a bluff — little showdown value, so pressure better hands to fold.",
    "semi_bluff":    "Bet as a semi-bluff — you can fold out better hands now and still improve.",
    "pot_control":   "Check to keep the pot small — decent showdown value, but don't bloat it.",
    "trap":          "Check to trap — you're very strong; let them catch up or bluff.",
    "realization":   "Check — weak holding; take a free card and try to improve.",
    "value_call":    "Call — you're ahead of enough of their betting range to continue.",
    "bluff_catch":   "Call to catch bluffs — you beat the hands they'd bluff with.",
    "call_odds":     "Call — your draw has the equity and pot odds to continue.",
    "raise_value":   "Raise for value — you're strong, so build the pot against worse.",
    "raise_semibluff": "Raise as a semi-bluff — fold out better hands now and improve if called.",
    "raise_bluff":   "Raise as a bluff — represent strength and pressure their bets.",
    "fold":          "Fold — not enough equity or showdown value against this bet.",
    "mixed":         "All actions are close here — any of them is acceptable.",
}

# River has no more cards — drop "improve" / "free card" / draw-chase wording.
RIVER_HEADLINES = {
    "realization":   "Check — weak holding; there's no more cards to come.",
    "semi_bluff":    "Bet as a bluff — little showdown value, so pressure better hands to fold.",
    "call_odds":     "Call — you have the pot odds to continue.",
    "raise_semibluff": "Raise as a bluff — represent strength and pressure their bets.",
    "protection":    "Bet for thin value / denial — charge worse hands while the board is final.",
}

# Casual board-texture phrasing (§6.4 style).
_TEXTURE = {
    "monotone": "all one suit", "two_tone": "two of a suit", "rainbow": "three suits",
    "paired": "a paired board", "connected": "a connected board",
}


def _street_from_rec(rec: Dict) -> str:
    board = rec.get("board") or ""
    if isinstance(board, list):
        n = len(board)
    else:
        cards = board.split() if " " in str(board) else [str(board)[i:i + 2]
                                                         for i in range(0, len(str(board)), 2)]
        n = len(cards)
    return {3: "flop", 4: "turn", 5: "river"}.get(n, "flop")


def freq_pct_ints(freq: Dict[str, float],
                  order: Optional[List[str]] = None) -> Dict[str, int]:
    """Integer percentages that sum to 100 (largest-remainder method).

    Independent `round(100 * p)` can yield 99 or 101 and mis-teach the mix.
    """
    keys = [k for k in (order or list(freq.keys())) if k in freq]
    if not keys:
        return {}
    raw = [100.0 * float(freq[k]) for k in keys]
    floors = [int(math.floor(x + 1e-12)) for x in raw]
    need = 100 - sum(floors)
    # Assign leftover points to the largest fractional parts.
    by_frac = sorted(
        range(len(keys)),
        key=lambda i: (raw[i] - floors[i], raw[i]),
        reverse=True,
    )
    out = {keys[i]: floors[i] for i in range(len(keys))}
    if need > 0:
        for j in range(need):
            out[keys[by_frac[j % len(keys)]]] += 1
    elif need < 0:
        # Floor sum can exceed 100 with slightly-over-1 inputs; peel from the smallest
        # fractional parts, but SKIP zero entries and keep going (never leave a bad
        # sum, never push a count negative).
        smallest = list(reversed(by_frac))
        removed, idx, guard = 0, 0, 0
        while removed < -need and guard < 10 * len(keys):
            i = smallest[idx % len(keys)]
            if out[keys[i]] > 0:
                out[keys[i]] -= 1
                removed += 1
            idx += 1
            guard += 1
    return out


def _wet(texture: List[str]) -> bool:
    return any(t in texture for t in ("two_tone", "monotone", "connected"))


def _is_first_action(rec: Dict) -> bool:
    """True for bet/check nodes (not fold/call/raise responses).

    Prefer decision_type when present so relabeled scenario nodes
    (sb_first, bb_vs_check, ...) are classified correctly.
    """
    dt = rec.get("decision_type")
    if dt:
        return dt == "first_action"
    node = rec.get("node", "")
    return node in ("bb_first", "btn_vs_check") or node.endswith("_first") \
        or node.endswith("_vs_check")


def classify_reason(rec: Dict) -> str:
    hc = rec["hand_category"]
    act = rec["preferred"]
    first_action = _is_first_action(rec)
    wet = _wet(rec.get("board_texture", []))
    if rec.get("mixed"):
        return "mixed"
    if first_action:
        if act == "bet":
            if hc == "strong_made":
                return "value"
            if hc == "draw":
                return "semi_bluff"
            if hc == "air":
                return "bluff"
            if hc in ("top_pair", "weak_pair"):
                return "protection" if wet else "value"
            return "value"
        if act == "check":
            if hc == "strong_made":
                return "trap"
            if hc in ("top_pair", "weak_pair"):
                return "pot_control"
            return "realization"          # draw or air checking
        # Mis-tagged first_action with a response action — do not call it a check.
        return "fold"
    # response node (fold / call / raise)
    if act == "raise":
        if hc in ("strong_made", "top_pair"):
            return "raise_value"
        if hc == "draw":
            return "raise_semibluff"
        return "raise_bluff"          # weak_pair / air raising
    if act == "call":
        if hc in ("strong_made", "top_pair"):
            return "value_call"
        if hc == "draw":
            return "call_odds"
        return "bluff_catch"          # weak_pair / air continuing
    return "fold"


def explain(rec: Dict, board_favored: Optional[str] = None) -> Dict:
    """Return {reason, headline, detail:[...]} for a decision record."""
    reason = classify_reason(rec)
    pref = rec["preferred"]
    street = _street_from_rec(rec)
    headline = RIVER_HEADLINES.get(reason, HEADLINES[reason]) if street == "river" \
        else HEADLINES[reason]

    # EV detail (expandable): how much better the preferred action is.
    evs = rec["ev"]
    ranked = sorted(evs, key=lambda a: evs[a], reverse=True)
    best, second = ranked[0], ranked[1]
    gap_pct = rec.get("ev_sep_pct")
    freq = rec.get("freq", {})
    detail: List[str] = []
    if reason == "mixed":
        close = [_action_word(a) for a in ranked]
        if len(close) <= 2:
            detail.append(f"{close[0]} and {close[1]} are within "
                          f"{gap_pct}% of the pot — treat both as acceptable.")
        else:
            # gap_pct is only the gap to 2nd-best; for 3+ actions quote the true
            # spread from best to worst so the number isn't understated.
            pot = rec.get("pot_bb") or 0.0
            spread = round(100.0 * (evs[ranked[0]] - evs[ranked[-1]]) / pot, 2) if pot else gap_pct
            detail.append(
                f"All {len(close)} actions ({', '.join(close)}) are within "
                f"{spread}% of the pot — any is acceptable."
            )
    else:
        detail.append(f"{_action_word(best).capitalize()} is best; "
                      f"{_action_word(second)} gives up ~{gap_pct}% of the pot.")
    # round() raises on NaN/inf, so only emit the frequency line when every value
    # is finite (a non-finite strategy is dropped upstream, but never crash here).
    if freq and all(isinstance(freq.get(a), (int, float)) and math.isfinite(freq[a]) for a in freq):
        fp = freq_pct_ints(freq, order=ranked)
        detail.append("Solver frequency: " + ", ".join(f"{_action_word(a)} {fp[a]}%" for a in ranked))
    # board-level range-advantage note where relevant
    if board_favored and _is_first_action(rec) and pref == "bet":
        if board_favored == rec["acting_player"]:
            detail.append(f"{rec['acting_player']}'s range is stronger on this board, "
                          f"which supports betting.")
    tex = rec.get("board_texture", [])
    tex_words = [ _TEXTURE[t] for t in tex if t in _TEXTURE ]
    if tex_words:
        detail.append("Board: " + ", ".join(tex_words) + ".")
    return {"reason": reason, "headline": headline, "detail": detail}


def _action_word(a: str) -> str:
    return {"bet": "bet", "check": "check", "call": "call", "fold": "fold",
            "raise": "raise"}.get(a, a)
