"""Wrap the pre-flop chart ranges into a signed, verifiable content pack (MIT).

Spots are sampled from the calibrated 6-max charts. When the 169×169 equity table
is available (or built with --solve), per-action EVs come from the 2-player
raise-ladder CFR and preferred is max-EV. Without --solve, EVs stay honest
chart sentinels (not solver numbers).

Run:  PYTHONPATH=src python demo/gen_preflop.py --solve
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pokertrainer.content_pack import build_pack, verify_pack   # noqa: E402
from pokertrainer.preflop_content import pack_records           # noqa: E402

VERSION = "preflop_v1"
OUT_DIR = "output/packs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", action="store_true",
                    help="attach 2-player CFR EVs (builds equity table if missing)")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--iters", type=int, default=800)
    a = ap.parse_args()
    ev_book = None
    solver_model = "preflop_chart_ranges_v1"
    note = ("Chart-based opening/defense ranges; freq is a pure strategy, "
            "ev is flat (not solver EV).")
    if a.solve:
        from pokertrainer.preflop_solve import solve_action_evs
        ev_book = solve_action_evs(iters=a.iters, samples=a.samples)
        solver_model = "preflop_cfr_hu_subgame_v1"
        note = ("Chart-sampled spots; ev/preferred from 2-player raise-ladder CFR "
                "(BTN-vs-BB tree, realization model). Not exact 6-max GTO.")
    records = pack_records(ev_book=ev_book)
    config = {
        "content_kind": "preflop_ranges",
        "solver_model": solver_model,
        "positions": {"format": "6-max", "seats": ["UTG", "HJ", "CO", "BTN", "SB", "BB"]},
        "note": note,
    }
    report = build_pack(records, config, OUT_DIR, VERSION,
                        pot=2.5, dedup_cap=99)   # 2.5bb ~ preflop pot; cap high so no spots drop
    db = os.path.join(OUT_DIR, f"flop_pack_{VERSION}.db")
    verdict = verify_pack(db)
    print(f"wrote {db}")
    print(f"  spots={len(records)} accepted={report['records_accepted']} "
          f"after_dedup={report['records_after_dedup']} db_bytes={report['db_bytes']}")
    print(f"  verify: hash_ok={verdict['hash_ok']} signature_ok={verdict['signature_ok']} "
          f"records={verdict['records']}")
    if not (verdict["hash_ok"] and verdict["signature_ok"]):
        raise SystemExit("pack failed self-verification")


if __name__ == "__main__":
    main()
