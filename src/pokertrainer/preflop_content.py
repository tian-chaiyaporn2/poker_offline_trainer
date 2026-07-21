"""Pre-flop training content (MIT).

Generates quiz spots (open/fold RFI + BB defense) from the calibrated ranges, with
plain-language reads, position-aware reasons, mixed/close flags, and rules of thumb.
Shared by the standalone pre-flop trainer and the integrated main trainer.
"""
import random

from .preflop_ranges import (
    rfi_ranges, bb_defense_ranges, sb_defense_ranges, vs_3bet_ranges,
    RFI_FREQ, BB_DEFENSE, SB_DEFENSE, VS_3BET, strength_cumulative)
from .preflop_equity import hand_classes

MARGIN = 3.5  # a hand within this % of a range cutoff is a "close"/mixed spot

RANK = "AKQJT98765432"
RI = {r: i for i, r in enumerate("23456789TJQKA")}
NAMES = {"A": "Aces", "K": "Kings", "Q": "Queens", "J": "Jacks", "T": "Tens",
         "9": "Nines", "8": "Eights", "7": "Sevens", "6": "Sixes", "5": "Fives",
         "4": "Fours", "3": "Threes", "2": "Twos"}
ONE = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack", "T": "Ten", "9": "Nine",
       "8": "Eight", "7": "Seven", "6": "Six", "5": "Five", "4": "Four", "3": "Three", "2": "Two"}
POS_FULL = {"UTG": "UTG (first to act)", "HJ": "the Hijack", "CO": "the Cutoff",
            "BTN": "the Button", "SB": "the Small Blind", "BB": "the Big Blind"}


def hand_read(cls):
    """Plain description of a starting hand class."""
    if len(cls) == 2:                              # pair
        r = cls[0]
        tier = ("a premium pair" if r in "AKQ" else
                "a strong pair" if r in "JT9" else
                "a medium pair" if r in "8765" else "a small pair")
        return f"a pair of {NAMES[r]} — {tier}"
    hi, lo, suit = cls[0], cls[1], cls[2]
    suited = suit == "s"
    gap = RI[hi] - RI[lo]
    both_broadway = RI[hi] >= RI["T"] and RI[lo] >= RI["T"]
    kind = "suited" if suited else "offsuit"
    if hi == "A":
        return f"{hi}{lo}{suit} — {'a suited' if suited else 'an offsuit'} ace"
    if both_broadway:
        return f"{hi}{lo}{suit} — {kind} broadway cards"
    if suited and gap <= 1:
        return f"{hi}{lo}{suit} — suited connectors"
    if suited and gap == 2:
        return f"{hi}{lo}{suit} — a suited one-gapper"
    if suited:
        return f"{hi}{lo}{suit} — a suited hand"
    if gap >= 4:
        return f"{hi}{lo}o — a weak, disconnected offsuit hand"
    return f"{hi}{lo}o — an offsuit hand"


def _hand_cat(cls):
    if len(cls) == 2:
        return "pair"
    hi, lo, suit = cls[0], cls[1], cls[2]
    if suit == "s" and RI[hi] - RI[lo] <= 2:
        return "suited_conn"
    if suit == "s":
        return "suited"
    return "offsuit"


RFI_WHY = {
    "open": {
        "pair": "A pocket pair plays well and makes strong hands — open it.",
        "suited": "Suited and playable — a good hand to open and take the lead.",
        "suited_conn": "Suited and connected — opens well for its playability (straights and flushes).",
        "offsuit": "Strong enough to open and take the initiative.",
    },
    "fold": {
        "pair": "Too small to open from here — set-mining alone isn't enough. Fold.",
        "suited": "Playable, but too weak to open from this position — fold.",
        "suited_conn": "Nice shape, but too weak to open from this early — fold.",
        "offsuit": "Offsuit and too weak to open — fold and wait for a better spot.",
    },
}
DEF_WHY = {
    "3bet": "Strong enough to re-raise (3-bet) for value — you want the money in.",
    "call": "Good enough to call and see a flop — but not strong enough to 3-bet.",
    "fold": "Too weak to continue against this open — fold and save your chips.",
}
VS3BET_WHY = {
    "4bet": "A premium — 4-bet (re-raise the 3-bet) for value; you want the money in.",
    "call": "Strong enough to call the 3-bet and see a flop — but not to 4-bet.",
    "fold": "Good enough to open, but not to continue against a 3-bet — fold.",
}
RULES_RFI = "The later your seat, the wider you open — fewer players left to wake up with a hand."
RULES_DEF = "Defend wider in the Big Blind (you get a price), but 3-bet only your strongest — and fold the junk."
RULES_SB = "In the Small Blind you're out of position with the BB still behind — defend tighter than the BB, leaning toward 3-bet-or-fold."
RULES_3BET = "Against a 3-bet, continue only with your strongest: 4-bet the premiums, call a few, fold the rest of your opens."


def combo_for(cls, rng):
    """Pick concrete cards (suits) for a hand class, for display."""
    suits = "shdc"
    if len(cls) == 2:
        s = rng.sample(suits, 2)
        return [cls[0] + s[0], cls[0] + s[1]]
    hi, lo, suit = cls[0], cls[1], cls[2]
    if suit == "s":
        s = rng.choice(suits)
        return [hi + s, lo + s]
    s = rng.sample(suits, 2)
    return [hi + s[0], lo + s[1]]


def _earliest_open(cls, rfi):
    for p in ["UTG", "HJ", "CO", "BTN", "SB"]:
        if rfi[p][cls] == "open":
            return p
    return None


def build_questions():
    rng = random.Random(11)
    rfi = rfi_ranges()
    dfn = bb_defense_ranges()
    sbd = sb_defense_ranges()
    v3 = vs_3bet_ranges()
    classes = hand_classes()
    cum = strength_cumulative()
    qs = []

    # RFI: for each position, pick instructive opens + folds (favour boundary hands)
    for pos in ["UTG", "HJ", "CO", "BTN", "SB"]:
        opens = [c for c in classes if rfi[pos][c] == "open"]
        folds = [c for c in classes if rfi[pos][c] == "fold"]
        pick = rng.sample(opens, min(4, len(opens))) + rng.sample(folds, min(3, len(folds)))
        for cls in pick:
            act = rfi[pos][cls]
            close = bool(abs(cum[cls] - RFI_FREQ[pos]) < MARGIN)
            why = RFI_WHY[act][_hand_cat(cls)]
            if act == "fold":                                # teach the positional nuance
                eo = _earliest_open(cls, rfi)
                why += (f" You'd open it from {POS_FULL[eo]}." if eo
                        else " It's a fold from every seat.")
            qs.append({
                "kind": "rfi", "ctx": "rfi", "pos": pos, "hand": combo_for(cls, rng), "cls": cls,
                "actions": ["fold", "open"], "answer": act,
                "mixed": close, "alt": ("open" if act == "fold" else "fold"),
                "read": hand_read(cls),
                "why": why, "rule": RULES_RFI,
                "situation": f"You're on {POS_FULL[pos]}. It folds to you.",
            })

    # BB defense: for each opener, pick 3bet / call / fold hands
    for opener in ["UTG", "CO", "BTN", "SB"]:
        d = dfn[opener]
        dfd, tb3 = BB_DEFENSE[opener]
        for act in ("3bet", "call", "fold"):
            pool = [c for c in classes if d[c] == act]
            for cls in rng.sample(pool, min(2, len(pool))):
                near3 = abs(cum[cls] - tb3) < MARGIN         # 3bet/call boundary
                nearD = abs(cum[cls] - dfd) < MARGIN         # call/fold boundary
                alt = None
                if near3:
                    alt = "call" if act == "3bet" else "3bet"
                elif nearD:
                    alt = "fold" if act == "call" else "call"
                qs.append({
                    "kind": "def", "ctx": "def", "pos": "BB", "opener": opener,
                    "hand": combo_for(cls, rng), "cls": cls,
                    "actions": ["fold", "call", "3bet"], "answer": act,
                    "mixed": bool(alt), "alt": alt,
                    "read": hand_read(cls),
                    "why": DEF_WHY[act], "rule": RULES_DEF,
                    "situation": f"You're in the Big Blind, and {POS_FULL[opener]} opens. It's on you.",
                })

    # SB defense: out of position with the BB behind (tighter than the BB)
    for opener in ["CO", "BTN"]:
        d = sbd[opener]
        dfd, tb3 = SB_DEFENSE[opener]
        for act in ("3bet", "call", "fold"):
            pool = [c for c in classes if d[c] == act]
            for cls in rng.sample(pool, min(2, len(pool))):
                near3 = abs(cum[cls] - tb3) < MARGIN
                nearD = abs(cum[cls] - dfd) < MARGIN
                alt = ("call" if act == "3bet" else "3bet") if near3 else \
                      ("fold" if act == "call" else "call") if nearD else None
                qs.append({
                    "kind": "sbdef", "ctx": "def", "pos": "SB", "opener": opener,
                    "hand": combo_for(cls, rng), "cls": cls,
                    "actions": ["fold", "call", "3bet"], "answer": act,
                    "mixed": bool(alt), "alt": alt, "read": hand_read(cls),
                    "why": DEF_WHY[act], "rule": RULES_SB,
                    "situation": f"You're in the Small Blind, and {POS_FULL[opener]} opens. It's on you.",
                })

    # Facing a 3-bet: you opened, a blind 3-bets — 4-bet / call / fold (over your opens)
    cont, fb = VS_3BET
    for (seat, seat_full, tbettor, tb_seat) in [("CO", "the Cutoff", "the Big Blind", "BB"),
                                                ("BTN", "the Button", "the Small Blind", "SB")]:
        opens = [c for c in classes if rfi[seat][c] == "open"]
        for act in ("4bet", "call", "fold"):
            pool = [c for c in opens if v3[c] == act]
            for cls in rng.sample(pool, min(2, len(pool))):
                near4 = abs(cum[cls] - fb) < MARGIN
                nearC = abs(cum[cls] - cont) < MARGIN
                alt = ("call" if act == "4bet" else "4bet") if near4 else \
                      ("fold" if act == "call" else "call") if nearC else None
                qs.append({
                    "kind": "vs3bet", "ctx": "vs3bet", "pos": seat, "tbettor": tb_seat,
                    "hand": combo_for(cls, rng), "cls": cls,
                    "actions": ["fold", "call", "4bet"], "answer": act,
                    "mixed": bool(alt), "alt": alt, "read": hand_read(cls),
                    "why": VS3BET_WHY[act], "rule": RULES_3BET,
                    "situation": f"You opened from {seat_full}, and {tbettor} 3-bets. It's back on you.",
                })
    rng.shuffle(qs)
    return qs


