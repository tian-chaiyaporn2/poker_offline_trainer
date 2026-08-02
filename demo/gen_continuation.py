"""Generate LINKED flop->turn->river trajectory content ("continuation mode") — MIT.

One hand played across streets: the hero's hole cards persist, the board grows a card at a
time, and the villain plays a fixed PASSIVE line (never bets/raises: checks behind, calls when
the hero bets) so a single hand advances on one runout. Each street yields one decision record
for the hero's specific combo, carrying real per-action solver EV/frequency, all three sharing a
`hand_id`. Both hero seats (OOP and IP) are generated from each solve.

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


def _advance_digit(seat, preferred):
    """Villain-passive advance digit after the hero's action (see plan Stage 1 table)."""
    if seat == "OOP":                     # hero at root
        return "1" if preferred == "check" else "3"    # IP checks / IP calls
    return "1" if preferred == "check" else "2"        # IP hero at ipc: OOP checks / check-calls


def _node_for(seat):
    return "root" if seat == "OOP" else "ipc"


def _villain_label(seat, preferred):
    """What the villain does after the hero's action, for the between-street interstitial."""
    if preferred == "bet":
        return "calls your bet"
    return "checks behind" if seat == "OOP" else "checks"


def _pot_after(path):
    """Pot (bb) entering the NEXT street given the path digits taken so far. Each '2'/'3'
    street added one bet+call of the then-current pot fraction; '1' added nothing."""
    pot = POT
    for d in path:
        if d in ("2", "3"):
            pot += 2 * (BET * pot)     # hero bet + villain call, both BET*pot of the street pot
    return pot


def _build_trajectory(s, flopb, turn, river, flop_s, seat, hero_idx, version):
    """Walk one hero combo's flop->turn->river villain-passive line into 3 linked records."""
    turnb, riverb = flopb + [turn], flopb + [turn, river]
    combos = s.oc if seat == "OOP" else s.ic
    hero_cards = (int(combos[hero_idx, 0]), int(combos[hero_idx, 1]))
    hand = hand_str(hero_cards)
    node = _node_for(seat)
    hero_pos, vill_pos = SEAT_POS[seat]
    hand_id = record_id(flop_s, seat, hand, version)

    boards = [flopb, turnb, riverb]
    board_strs = [flop_s, flop_s + "".join(_c(x) for x in [turn]),
                  flop_s + "".join(_c(x) for x in [turn, river])]
    streets = ["flop", "turn", "river"]

    recs, path = [], ""
    for step in range(3):
        d = s.decision(_bk(path, boards[step]), node, hero_idx, ACTS)
        if d["reach_mass"] < 0.05:       # node barely reached — skip whole trajectory
            return []
        pot_here = _pot_after(path)
        second = sorted(d["freq"].values())[-2] if len(d["freq"]) > 1 else 0.0
        villain = _villain_label(seat, d["preferred"])
        recs.append({
            "board": board_strs[step], "board_texture": [], "board_favored": None,
            "node": f"{hero_pos.lower()}_first" if seat == "OOP" else f"{hero_pos.lower()}_vs_check",
            "acting_player": hero_pos, "decision_type": "continuation",
            "hand": hand, "hand_category": "cont",
            "actions": ACTS, "ev": d["ev"], "freq": d["freq"], "preferred": d["preferred"],
            "reach_mass": d["reach_mass"], "mixed": bool(second >= 0.35),
            "pot_bb": pot_here,
            "scenario": f"cont|{hand_id}|s{step}|{seat}",
            "oop_pos": "BB", "ip_pos": "BTN",
            "explanation": {
                "reason": "continuation",
                "headline": f"{streets[step].title()} — you're {'out of' if seat=='OOP' else 'in'} position",
                "detail": {
                    "hand_id": hand_id, "step_index": step, "hero_seat": seat,
                    "street": streets[step], "villain_action": villain,
                    "board_full": board_strs[step], "hero_cards": hand,
                    "is_oop": seat == "OOP", "villain": vill_pos,
                },
            },
        })
        if step < 2:
            path += _advance_digit(seat, d["preferred"])
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
        note="linked flop->turn->river; villain passive (checks/calls); CPU seed"):
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
              "bet_pct_pot": 66, "line": "continuation_passive_villain",
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
