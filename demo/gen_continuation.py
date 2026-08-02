"""Generate LINKED flop->turn->river trajectory content ("continuation mode") — MIT.

One hand played across streets: the hero's hole cards persist, the board grows a card at a
time, and the villain plays its SOLVED main line — at each of its decision nodes it takes the
action its whole range predominantly takes (bet if the range's aggregate bet share >= 0.5, else
check; call if its call share >= 0.5, else fold), driven by `node_action_share`. So the villain
genuinely bets into the hero and the hero faces real fold/call spots; a fold on either side ends
the hand there (hands are variable length). Each street the hero acts on yields one decision
record for the hero's specific combo, carrying real per-action solver EV/frequency, and the steps
of one hand share a `hand_id`. Both hero seats (OOP and IP) are generated from each solve.

Engine: `MultiStreetSpike` (flop->turn->river CFR, no raising). Because it enumerates the full
turn+river runout, a single CPU solve is EXPENSIVE (tens of minutes). Defaults here produce a
small seed; the full-variety library is meant to be generated on GPU/Kaggle (raise `--flops`,
`--n`, `--iters`). See docs/continuation_mode_scope.md.

Run (small CPU seed):  PYTHONPATH=src python demo/gen_continuation.py --flops 2 --n 40 --iters 160
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pokertrainer.cards import parse_cards, hand_str            # noqa: E402
from pokertrainer.content_pack import build_pack, verify_pack, record_id  # noqa: E402
from pokertrainer.content_yield import validate_records         # noqa: E402
from pokertrainer.evaluator import evaluate                     # noqa: E402
from pokertrainer.presets import BB_SRP, BTN_SRP                # noqa: E402
from pokertrainer.ranges import expand_range                    # noqa: E402
from pokertrainer.solver.multistreet import MultiStreetSpike    # noqa: E402
from pokertrainer.validate_flop import subsample                # noqa: E402

POT, BET = 5.5, 0.66
CHECK, BET_I = 0, 1
ACTS = ["check", "bet"]
SEAT_POS = {"OOP": ("BB", "BTN"), "IP": ("BTN", "BB")}   # (hero seat code, villain seat code)

# Curated runouts: (flop, turn, river). Dry brick, ace-high with a runout scare, wet board.
CURATED = [
    ("7c6d2s", "Kh", "3c"),
    ("Ah7d2c", "Ts", "9h"),
    ("Qd9s4h", "Jc", "2d"),
    ("Ks8h5d", "5s", "Ad"),
]


def _bk(path, board):
    return (path, tuple(sorted(board)))


def _candidate_bkeys(flopb, turnb, riverb):
    """All 13 (path, board) nodes reachable along any villain-passive line for this runout."""
    tg = {_bk("", flopb)}
    for d1 in "123":
        tg.add(_bk(d1, turnb))
        for d2 in "123":
            tg.add(_bk(d1 + d2, riverb))
    return tg


def _pot_after(path):
    """Pot (bb) entering the NEXT street given the path digits taken so far. Each '2'/'3'
    street added one bet+call of the then-current pot fraction; '1' added nothing."""
    pot = POT
    for d in path:
        if d in ("2", "3"):
            pot += 2 * (BET * pot)     # hero bet + villain call, both BET*pot of the street pot
    return pot


STREETS = ["flop", "turn", "river"]
PCT = int(round(BET * 100))
# node the trainer renders for each (seat, situation): first-to-act, facing a check, facing a bet.
_NODE = {("OOP", "root"): "bb_first", ("IP", "ipc"): "btn_vs_check",
         ("OOP", "ovb"): "bb_vs_bet", ("IP", "ivb"): "btn_vs_bet"}


def _build_trajectory(s, flopb, turn, river, flop_s, seat, hero_idx, version):
    """Walk ONE played line where the villain acts on its solved range strategy (it bets when its
    range predominantly bets) and the hero responds with the solver's play. Each hero decision is
    a linked step — including facing-a-bet fold/call spots. A fold (either side) ends the hand."""
    combos = s.oc if seat == "OOP" else s.ic
    hand = hand_str((int(combos[hero_idx, 0]), int(combos[hero_idx, 1])))
    hero_pos, vill_pos = SEAT_POS[seat]
    hand_id = record_id(flop_s + _c(turn) + _c(river), seat, hand, version)
    boards = [flopb, flopb + [turn], flopb + [turn, river]]
    board_strs = [flop_s, flop_s + _c(turn), flop_s + _c(turn) + _c(river)]

    recs, path, prev_len = [], "", len(flopb)

    def rec(street, node_key, actions, d, newcard):
        second = sorted(d["freq"].values())[-2] if len(d["freq"]) > 1 else 0.0
        step_index = len(recs)
        recs.append({
            "board": board_strs[street], "board_texture": [], "board_favored": None,
            "node": _NODE[(seat, node_key)], "acting_player": hero_pos,
            "decision_type": "continuation", "hand": hand, "hand_category": "cont",
            "actions": actions, "ev": d["ev"], "freq": d["freq"], "preferred": d["preferred"],
            "reach_mass": d["reach_mass"], "mixed": bool(second >= 0.35),
            "pot_bb": _pot_after(path), "scenario": f"cont|{hand_id}|s{step_index}|{seat}",
            "oop_pos": "BB", "ip_pos": "BTN",
            "explanation": {"reason": "continuation",
                            "headline": ("On the " + STREETS[street] + ", the solver's strongest "
                                         "play with this hand is to " + d["preferred"] + "."),
                            "detail": {"hand_id": hand_id, "step_index": step_index,
                                       "hero_seat": seat, "street": STREETS[street],
                                       "villain_action": "", "board_full": board_strs[street],
                                       "hero_cards": hand, "is_oop": seat == "OOP",
                                       "villain": vill_pos, "newcard": newcard, "last": False}}})
        return recs[-1]

    def after(r, text):
        r["explanation"]["detail"]["villain_action"] = text

    def nxt(street):
        return STREETS[street + 1] if street < 2 else "showdown"

    ended = False
    for street in range(3):
        if ended:
            break
        bkey = _bk(path, boards[street])
        newcard = len(boards[street]) > prev_len
        prev_len = len(boards[street])
        if seat == "OOP":                                  # OOP hero acts first at root
            d = s.decision(bkey, "root", hero_idx, ["check", "bet"])
            if d["reach_mass"] < 0.05:
                return []
            r = rec(street, "root", ["check", "bet"], d, newcard)
            if d["preferred"] == "bet":
                if s.node_action_share(bkey, "ivb", 1) >= 0.5:       # villain calls
                    after(r, "Opponent calls — the " + nxt(street) + " comes."); path += "3"
                else:
                    after(r, "Opponent folds — you take the pot."); ended = True
            else:                                                    # OOP checks
                if s.node_action_share(bkey, "ipc", 1) >= 0.5:       # villain bets -> hero faces it
                    after(r, "Opponent bets " + str(PCT) + "% of the pot.")
                    d2 = s.decision(bkey, "ovb", hero_idx, ["fold", "call"])
                    r2 = rec(street, "ovb", ["fold", "call"], d2, False)
                    if d2["preferred"] == "call":
                        after(r2, "You call — the " + nxt(street) + " comes."); path += "2"
                    else:
                        after(r2, "You fold — the hand's over."); ended = True
                else:
                    after(r, "Opponent checks back — the " + nxt(street) + " comes."); path += "1"
        else:                                              # IP hero; villain (OOP) acts first
            if s.node_action_share(bkey, "root", 1) >= 0.5:          # villain bets -> hero faces it
                d = s.decision(bkey, "ivb", hero_idx, ["fold", "call"])
                if d["reach_mass"] < 0.05:
                    return []
                r = rec(street, "ivb", ["fold", "call"], d, newcard)
                if d["preferred"] == "call":
                    after(r, "You call — the " + nxt(street) + " comes."); path += "3"
                else:
                    after(r, "You fold — the hand's over."); ended = True
            else:                                                    # villain checks -> hero decides
                d = s.decision(bkey, "ipc", hero_idx, ["check", "bet"])
                if d["reach_mass"] < 0.05:
                    return []
                r = rec(street, "ipc", ["check", "bet"], d, newcard)
                if d["preferred"] == "bet":
                    if s.node_action_share(bkey, "ovb", 1) >= 0.5:   # villain calls the hero's bet
                        after(r, "Opponent calls — the " + nxt(street) + " comes."); path += "2"
                    else:
                        after(r, "Opponent folds — you take the pot."); ended = True
                else:
                    after(r, "Opponent checks back — the " + nxt(street) + " comes."); path += "1"
    if recs:
        recs[-1]["explanation"]["detail"]["last"] = True
    return recs


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _c(code):
    return _RANKS[code // 4] + _SUITS[code % 4]


def _pick_hero_indices(combos, board5, avoid, k=6):
    """Pick k hero combos spanning strong->weak by made-hand rank on the final board,
    skipping combos that collide with the dealt runout cards."""
    ranked = []
    for i in range(len(combos)):
        a, b = int(combos[i, 0]), int(combos[i, 1])
        if a in avoid or b in avoid:
            continue
        ranked.append((evaluate((a, b, *board5)), i))
    ranked.sort(reverse=True)
    if len(ranked) <= k:
        return [i for _, i in ranked]
    step = (len(ranked) - 1) / (k - 1)
    return [ranked[round(j * step)][1] for j in range(k)]


def run(flops=2, n=40, iters=160, version="continuation_seed",
        note="linked flop->turn->river; villain plays its solved main line (range argmax); CPU seed"):
    recs, conv = [], []
    for fi, (flop_s, turn_s, river_s) in enumerate(CURATED[:flops], 1):
        flop = parse_cards(flop_s)
        turn = parse_cards(turn_s)[0]
        river = parse_cards(river_s)[0]
        flopb = list(flop)
        oop = subsample([c for c, _ in expand_range(BB_SRP, flop)], n)
        ip = subsample([c for c, _ in expand_range(BTN_SRP, flop)], n)
        s = MultiStreetSpike(flop, oop, ip, np.ones(len(oop)), np.ones(len(ip)),
                             POT, BET, streets=3, raise_x=None)
        res = s.run(iters)
        tail = res["ev_curve"]
        stable = len(tail) >= 2 and abs(tail[-1][1] - tail[-2][1]) / POT < 0.01
        conv.append({"flop": flop_s, "root_ev_pct_pot": round(res["root_ev_pct_pot"], 2),
                     "stable": stable, "runtime_sec": round(res["runtime_sec"], 1)})
        s.eval_capture_targets(_candidate_bkeys(flopb, flopb + [turn], flopb + [turn, river]))
        board5 = flopb + [turn, river]
        avoid = {turn, river}
        for seat in ("OOP", "IP"):
            combos = s.oc if seat == "OOP" else s.ic
            for idx in _pick_hero_indices(combos, board5, avoid):
                recs.extend(_build_trajectory(s, flopb, turn, river, flop_s, seat, idx, version))
        print(f"[{fi}/{flops}] {flop_s}->{turn_s}{river_s}: "
              f"{len(recs)} records so far, stable={stable}", flush=True)

    errs = validate_records(recs)
    if errs:
        print("VALIDATE WARNINGS:", errs[:5])
    config = {"positions": {"ip": "BTN", "oop": "BB"}, "stack_bb": 100, "pot_bb": POT,
              "bet_pct_pot": 66, "line": "continuation_solved_villain",
              "note": note, "solver_model": "multistreet_spike_cfr_plus",
              "convergence": conv}
    build_pack(recs, config, "output/packs", version, pot=POT, dedup_cap=999)
    verdict = verify_pack(f"output/packs/flop_pack_{version}.db")
    hands = len({r["scenario"].split("|")[1] for r in recs})
    print(f"wrote continuation pack: {len(recs)} records / {hands} hands; VERIFY: {verdict}")
    if not (verdict.get("hash_ok") and verdict.get("signature_ok")):
        raise SystemExit("continuation pack failed integrity verification")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", type=int, default=2)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--version", default="continuation_seed")
    run(**{k: v for k, v in vars(ap.parse_args()).items()})
