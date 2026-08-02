"""Wrap the pre-flop chart ranges into a signed, verifiable content pack (MIT).

A5 — parity with the post-flop packs on the *distribution / provenance* axis: the
pre-flop spots that `preflop_content.build_questions()` produces are chart-based (a single
correct action + plain-language coaching, no per-action solver EV). We store them in the
same signed SQLite schema so the trainer loads + verifies them exactly like every other
pack, instead of importing the range code at build time.

Because the schema (and `content_pack._require_finite`) expects an `ev`/`freq` map keyed by
the action list with `preferred` a max-EV action, we encode each chart decision honestly as
a PURE strategy: freq = 1.0 on the correct action, and a FLAT neutral EV (all 0.0) that is
explicitly *not* solver-derived. The pre-flop-only fields (ctx / opener / tbettor / alt /
why / rule / situation) ride in the row's `detail` JSON so they round-trip losslessly.

Run:  python demo/gen_preflop.py       # writes output/packs/flop_pack_preflop_v1.db(.gz)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pokertrainer.content_pack import build_pack, verify_pack   # noqa: E402
from pokertrainer.preflop_content import pack_records           # noqa: E402

VERSION = "preflop_v1"
OUT_DIR = "output/packs"


def main():
    records = pack_records()
    config = {
        "content_kind": "preflop_ranges",
        "solver_model": "preflop_chart_ranges_v1",
        "positions": {"format": "6-max", "seats": ["UTG", "HJ", "CO", "BTN", "SB", "BB"]},
        "note": "Chart-based opening/defense ranges; freq is a pure strategy, ev is flat (not solver EV).",
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
