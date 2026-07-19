"""Content pack build (PRD v1.3 §10.1 pack build, §11.1 entities) — MIT.

Assembles accepted full-street flop decision records (+ explanations) into a
versioned, signed, provenance-stamped SQLite pack the app loads on-device:

- `flop_decision`  — one row per record: state, actions, EVs, frequencies, tags,
  explanation, and **precomputed per-action grades** (§8.2) for trivial on-device
  scoring.
- `foundation_template` — algorithmic supporting content (seeded).
- `pack_meta` — version, config, provenance, grade thresholds, content hash and
  signature (integrity: signed + verifiable, §11.2).

The pack is deterministic and reproducible (§10.2): same records + config + code
commit produce the same content hash. Build also emits `build_report.json` and a
gzipped copy of the pack.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import subprocess
from typing import Dict, List, Optional

from .explanations import explain

# Grade thresholds as % of pot of the normalized EV regret of the chosen action
# (§8.2; engineering starting points, stored in the pack for transparency).
GRADE_THRESHOLDS = [("best", 0.25), ("good", 1.0), ("acceptable", 3.0), ("costly", 10.0)]
DEV_SIGNING_KEY = b"pokertrainer-dev-signing-key"   # production: asymmetric key


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def record_id(board: str, node: str, hand: str, version: str) -> str:
    return hashlib.sha1(f"{board}|{node}|{hand}|{version}".encode()).hexdigest()[:16]


def grade_action(action_ev: float, best_ev: float, pot: float) -> str:
    regret = 100.0 * (best_ev - action_ev) / pot
    for name, thr in GRADE_THRESHOLDS:
        if regret <= thr:
            return name
    return "major_error"


def _action_grades(rec: Dict, pot: float) -> Dict[str, str]:
    evs = rec["ev"]
    best = max(evs.values())
    return {a: grade_action(evs[a], best, pot) for a in evs}


def _canonical(rec_rows: List[tuple], id_idx: int = 0) -> str:
    """Deterministic serialization for hashing (sorted by id)."""
    return json.dumps(sorted(rec_rows, key=lambda r: r[id_idx]), separators=(",", ":"),
                      default=str)


SCHEMA = """
CREATE TABLE pack_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE flop_decision (
  id TEXT PRIMARY KEY, board TEXT, board_texture TEXT, board_favored TEXT,
  node TEXT, acting_player TEXT, decision_type TEXT,
  hand TEXT, hand_category TEXT,
  actions TEXT, ev TEXT, freq TEXT, preferred_action TEXT, action_grades TEXT,
  ev_sep_pct REAL, mixed INTEGER, reach_mass REAL,
  reason TEXT, headline TEXT, detail TEXT,
  solver_model TEXT, validation_status TEXT
);
CREATE INDEX idx_node ON flop_decision(node);
CREATE INDEX idx_hand_cat ON flop_decision(hand_category);
CREATE INDEX idx_decision_type ON flop_decision(decision_type);
CREATE INDEX idx_board ON flop_decision(board);
CREATE INDEX idx_reason ON flop_decision(reason);
CREATE TABLE foundation_template (id TEXT PRIMARY KEY, unit TEXT, kind TEXT, spec TEXT);
"""

FOUNDATION_SEEDS = [
    ("found_board_texture", "board_reading", "template",
     json.dumps({"asks": "Classify the board (dry/paired/connected/two-tone/monotone).",
                 "generator": "board_texture"})),
    ("found_pot_odds", "pot_odds", "arithmetic",
     json.dumps({"asks": "What equity do you need to call?", "generator": "pot_odds"})),
    ("found_hand_reading", "hand_reading", "evaluator",
     json.dumps({"asks": "Identify made hand / draws.", "generator": "describe_hand"})),
    ("found_equity", "equity", "montecarlo",
     json.dumps({"asks": "Estimate equity vs a range.", "generator": "mc_equity"})),
]


def _resolve_signing_key(signing_key: Optional[bytes] = None) -> bytes:
    """Prefer an explicit key, then POKERTRAINER_SIGNING_KEY, else the dev key."""
    if signing_key is not None:
        return signing_key
    env = os.environ.get("POKERTRAINER_SIGNING_KEY")
    if env:
        return env.encode() if isinstance(env, str) else env
    return DEV_SIGNING_KEY


def build_pack(records: List[Dict], config: Dict, out_dir: str, version: str,
               signing_key: Optional[bytes] = None, pot: float = 5.5,
               dedup_cap: int = 30) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    signing_key = _resolve_signing_key(signing_key)

    accepted = [r for r in records if r.get("accepted", True)]
    # concept dedup (cap near-identical records)
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in accepted:
        key = (r["node"], tuple(r.get("board_texture", [])), r["hand_category"], r["preferred"])
        buckets[key].append(r)
    deduped = [r for g in buckets.values() for r in g[:dedup_cap]]

    rows = []
    for r in deduped:
        expl = r.get("explanation") or explain(r, r.get("board_favored"))
        rid = record_id(r["board"], r["node"], r["hand"], version)
        rows.append((
            rid, r["board"], json.dumps(r.get("board_texture", [])), r.get("board_favored"),
            r["node"], r["acting_player"], r.get("decision_type", ""),
            r["hand"], r["hand_category"],
            json.dumps(r["actions"]), json.dumps(r["ev"]), json.dumps(r["freq"]),
            r["preferred"], json.dumps(_action_grades(r, pot)),
            r.get("ev_sep_pct"), int(bool(r.get("mixed"))), r.get("reach_mass"),
            expl["reason"], expl["headline"], json.dumps(expl["detail"]),
            config.get("solver_model", "full_street_cfr_plus"), "passed",
        ))

    content_hash = hashlib.sha256(_canonical(rows).encode()).hexdigest()
    signature = hmac.new(signing_key, content_hash.encode(), hashlib.sha256).hexdigest()

    db_path = os.path.join(out_dir, f"flop_pack_{version}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO flop_decision VALUES (%s)" % ",".join("?" * 22), rows)
    conn.executemany("INSERT INTO foundation_template VALUES (?,?,?,?)", FOUNDATION_SEEDS)
    meta = {
        "pack_id": f"flop_srp_btn_bb_{version}",
        "version": version,
        "config": json.dumps(config),
        "provenance": json.dumps({"git_commit": _git_commit(),
                                  "solver_model": config.get("solver_model", "full_street_cfr_plus")}),
        "grade_thresholds_pct_pot": json.dumps(dict(GRADE_THRESHOLDS)),
        "record_count": str(len(rows)),
        "content_hash": content_hash,
        "signature": signature,
        "signing_scheme": "hmac-sha256-dev",
    }
    conn.executemany("INSERT INTO pack_meta VALUES (?,?)", list(meta.items()))
    conn.commit()
    conn.close()

    # gzip the pack (compressed distribution, §10.1)
    gz_path = db_path + ".gz"
    with open(db_path, "rb") as f, gzip.open(gz_path, "wb") as g:
        shutil.copyfileobj(f, g)

    report = {**{k: json.loads(v) if k in ("config", "provenance", "grade_thresholds_pct_pot")
                 else v for k, v in meta.items()},
              "db_bytes": os.path.getsize(db_path), "gz_bytes": os.path.getsize(gz_path),
              "records_accepted": len(accepted), "records_after_dedup": len(rows)}
    with open(os.path.join(out_dir, f"build_report_{version}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def verify_pack(db_path: str, signing_key: Optional[bytes] = None) -> Dict:
    """Recompute the content hash + signature and compare to stored (integrity)."""
    signing_key = _resolve_signing_key(signing_key)
    conn = sqlite3.connect(db_path)
    cols = [d[1] for d in conn.execute("PRAGMA table_info(flop_decision)")]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM flop_decision").fetchall()
    meta = dict(conn.execute("SELECT key, value FROM pack_meta").fetchall())
    conn.close()
    content_hash = hashlib.sha256(_canonical(rows).encode()).hexdigest()
    signature = hmac.new(signing_key, content_hash.encode(), hashlib.sha256).hexdigest()
    return {
        "records": len(rows),
        "hash_ok": content_hash == meta.get("content_hash"),
        "signature_ok": signature == meta.get("signature"),
        "version": meta.get("version"),
    }


DEFAULT_CONFIG = {
    "positions": {"ip": "BTN", "oop": "BB"}, "stack_bb": 100, "pot_bb": 5.5,
    "bet_pct_pot": 66, "rake": 0, "solver_model": "full_street_cfr_plus",
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="records.json from content_yield")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--out", default="output/packs")
    ap.add_argument("--pot", type=float, default=5.5)
    a = ap.parse_args()
    recs = json.load(open(a.records))
    rep = build_pack(recs, DEFAULT_CONFIG, out_dir=a.out, version=a.version, pot=a.pot)
    print(json.dumps(rep, indent=2))
    db = os.path.join(a.out, f"flop_pack_{a.version}.db")
    print("verify:", verify_pack(db))
