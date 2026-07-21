"""Generate turn + river decision content (later-street prototype) — MIT.

A turn decision is the same 4-node solve on a 4-card board; a river decision is
the same solve on a 5-card board. This is an *unconditioned* later-street toy
demo: ranges are the preflop SRP ranges with board card-removal only — they are
NOT filtered by a prior check/check line. UI copy must not claim a checked-
through range.

`_make_solver` derives n_streets from board length (turn→2, river→1) so runouts
stop at a real 5-card board.

Run:  PYTHONPATH=src python demo/gen_turn_river.py
"""
import numpy as np

from pokertrainer.cards import parse_cards
from pokertrainer.content_pack import build_pack, verify_pack
from pokertrainer.content_yield import extract_records
from pokertrainer.presets import BB_SRP, BTN_SRP
from pokertrainer.ranges import expand_range
from pokertrainer.validate_flop import _make_solver, subsample

# (board, bet_streets): 4-card board -> turn (2), 5-card board -> river (1).
# Curated representative runouts: bricks, flush-completers, board-pairers, overcards.
RUNOUTS = [
    ("Th9h8d2c", 2), ("Th9h8dJh", 2), ("Th9h8dTs", 2),      # turn: brick / flush+straight / pairs top
    ("As7h2dKs", 2), ("As7h2d7c", 2),                        # turn: overcard / pairs board
    ("Kd9c4hQs", 2), ("Kd9c4h4d", 2),                        # turn: overcard / pairs board
    ("Qh8h3h2h", 2), ("Qh8h3hKd", 2),                        # turn: 4th heart / overcard
    ("Th9h8d2c7c", 1), ("Th9h8d2cKd", 1),                    # river: straight-filler / blank
    ("As7h2dKs9c", 1), ("As7h2dKsQh", 1),                    # river: blanks
    ("Qh8h3h2hAh", 1), ("Kd9c4hQs2s", 1),                    # river: nut-flush card / blank
]
POT, BET = 5.5, 0.66


def run(solver="cpu", dtype="float64", n=90, iters=200, version="turnriver_demo",
        note="turn/river demo (reduced range; NOT check-check filtered)", raise_x=None):
    """Solve the curated turn/river runouts. Demo defaults (cpu, n=90) run locally;
    pass --solver gpu --n 400 --iters 300 (Kaggle) for the full-range pack. Pass
    --raise-x 3 to add Fold/Call/Raise on the facing-a-bet nodes (the raise pass)."""
    make = _make_solver(solver, dtype, raise_x=raise_x)
    recs = []
    for i, (board, streets) in enumerate(RUNOUTS, 1):
        flop = parse_cards(board)
        oop = subsample([c for c, _ in expand_range(BB_SRP, flop)], n)
        ip = subsample([c for c, _ in expand_range(BTN_SRP, flop)], n)
        r = [x for x in extract_records(board, oop, ip, iters, make, POT, BET, streets=streets)
             if x.get("accepted")]
        recs.extend(r)
        street = {2: "turn", 1: "river"}[streets]
        print(f"[{i}/{len(RUNOUTS)}] {board} ({street}): {len(r)} records", flush=True)
    config = {"positions": {"ip": "BTN", "oop": "BB"}, "stack_bb": 100, "pot_bb": POT,
              "bet_pct_pot": 66, "line": "unconditioned_later_street",
              "note": note, "solver_model": "full_street_cfr_plus"}
    build_pack(recs, config, "output/packs", version)
    verdict = verify_pack(f"output/packs/flop_pack_{version}.db")
    print("VERIFY:", verdict)
    if not (verdict.get("hash_ok") and verdict.get("signature_ok")):
        raise SystemExit("turn/river pack failed integrity verification")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--n", type=int, default=90, help="combos per range (400 ~ full-range)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--version", default="turnriver_demo")
    ap.add_argument("--note", default="turn/river demo (reduced range; NOT check-check filtered)")
    ap.add_argument("--raise-x", dest="raise_x", type=float, default=None,
                    help="raise-to multiple of the bet (e.g. 3) — adds Fold/Call/Raise on vs-bet nodes")
    a = ap.parse_args()
    run(a.solver, a.dtype, a.n, a.iters, a.version, a.note, raise_x=a.raise_x)
