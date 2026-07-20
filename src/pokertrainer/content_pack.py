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

Integrity covers flop_decision rows, foundation_template rows, and all pack_meta
keys except content_hash/signature themselves — so metadata and foundations
cannot be tampered while still verifying as signed.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import math
import os
import shutil
import sqlite3
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from .explanations import explain

# Grade thresholds as % of pot of the normalized EV regret of the chosen action
# (§8.2; engineering starting points, stored in the pack for transparency).
GRADE_THRESHOLDS = [("best", 0.25), ("good", 1.0), ("acceptable", 3.0), ("costly", 10.0)]
DEV_SIGNING_KEY = b"pokertrainer-dev-signing-key"   # production: asymmetric key
_META_UNSIGNED = frozenset({"content_hash", "signature"})


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def record_id(board: str, node: str, hand: str, version: str,
              scenario: str = "") -> str:
    return hashlib.sha1(
        f"{board}|{node}|{hand}|{version}|{scenario}".encode()
    ).hexdigest()[:16]


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


def _require_finite(rec: Dict) -> None:
    """Refuse non-finite / incoherent numerics before they enter a signed pack."""
    for key in ("ev_sep_pct", "reach_mass"):
        val = rec.get(key)
        if val is not None and isinstance(val, (int, float)) and not math.isfinite(val):
            raise ValueError(f"non-finite {key}={val!r} in record {rec.get('hand')}")
    actions = rec.get("actions") or []
    for group in ("ev", "freq"):
        mapping = rec.get(group) or {}
        if set(mapping.keys()) != set(actions):
            raise ValueError(
                f"{group} keys {sorted(mapping)} != actions {actions} "
                f"in record {rec.get('hand')}"
            )
        for a, val in mapping.items():
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"non-finite {group}[{a}]={val!r} in record {rec.get('hand')}"
                )
    freq = rec.get("freq") or {}
    if abs(sum(freq.values()) - 1.0) > 0.02:
        raise ValueError(
            f"freq sums to {sum(freq.values()):.4f} (expected ~1) "
            f"in record {rec.get('hand')}"
        )
    for a, val in freq.items():
        if val < -1e-9 or val > 1.0 + 1e-9:
            raise ValueError(
                f"freq[{a}]={val!r} outside [0, 1] in record {rec.get('hand')}"
            )
    ev = rec.get("ev") or {}
    pref = rec.get("preferred")
    if pref not in ev:
        raise ValueError(
            f"preferred={pref!r} missing from ev in record {rec.get('hand')}"
        )
    # Preferred must be a max-EV action — otherwise grades/explanations contradict
    # the recommended action the trainer shows.
    best = max(ev.values())
    if ev[pref] < best - 1e-9:
        raise ValueError(
            f"preferred={pref!r} EV {ev[pref]} < best EV {best} "
            f"in record {rec.get('hand')}"
        )


def _canonical(decision_rows: Sequence[tuple], foundation_rows: Sequence[tuple],
               meta_items: Sequence[Tuple[str, str]]) -> str:
    """Deterministic serialization covering every integrity-relevant table."""
    signed_meta = sorted(
        ((k, v) for k, v in meta_items if k not in _META_UNSIGNED),
        key=lambda kv: kv[0],
    )
    payload = {
        "flop_decision": sorted(decision_rows, key=lambda r: r[0]),
        "foundation_template": sorted(foundation_rows, key=lambda r: r[0]),
        "pack_meta": signed_meta,
    }
    return json.dumps(payload, separators=(",", ":"), default=str, allow_nan=False)


SCHEMA = """
CREATE TABLE pack_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE flop_decision (
  id TEXT PRIMARY KEY, board TEXT, board_texture TEXT, board_favored TEXT,
  node TEXT, acting_player TEXT, decision_type TEXT,
  hand TEXT, hand_category TEXT,
  actions TEXT, ev TEXT, freq TEXT, preferred_action TEXT, action_grades TEXT,
  ev_sep_pct REAL, mixed INTEGER, reach_mass REAL,
  reason TEXT, headline TEXT, detail TEXT,
  solver_model TEXT, validation_status TEXT,
  scenario TEXT, pot_bb REAL, oop_pos TEXT, ip_pos TEXT
);
CREATE INDEX idx_node ON flop_decision(node);
CREATE INDEX idx_hand_cat ON flop_decision(hand_category);
CREATE INDEX idx_decision_type ON flop_decision(decision_type);
CREATE INDEX idx_board ON flop_decision(board);
CREATE INDEX idx_reason ON flop_decision(reason);
CREATE INDEX idx_scenario ON flop_decision(scenario);
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
    for r in accepted:
        _require_finite(r)
    # concept dedup (cap near-identical records); scenario keeps matchups apart
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in accepted:
        key = (r.get("scenario", ""), r["node"], tuple(r.get("board_texture", [])),
               r["hand_category"], r["preferred"])
        buckets[key].append(r)
    deduped = [r for g in buckets.values() for r in g[:dedup_cap]]

    rows = []
    for r in deduped:
        expl = r.get("explanation") or explain(r, r.get("board_favored"))
        scenario = r.get("scenario", "")
        rec_pot = float(r["pot_bb"]) if r.get("pot_bb") is not None else float(pot)
        rid = record_id(r["board"], r["node"], r["hand"], version, scenario)
        rows.append((
            rid, r["board"], json.dumps(r.get("board_texture", [])), r.get("board_favored"),
            r["node"], r["acting_player"], r.get("decision_type", ""),
            r["hand"], r["hand_category"],
            json.dumps(r["actions"]), json.dumps(r["ev"]), json.dumps(r["freq"]),
            r["preferred"], json.dumps(_action_grades(r, rec_pot)),
            r.get("ev_sep_pct"), int(bool(r.get("mixed"))), r.get("reach_mass"),
            expl["reason"], expl["headline"], json.dumps(expl["detail"]),
            config.get("solver_model", "full_street_cfr_plus"), "passed",
            scenario, rec_pot, r.get("oop_pos"), r.get("ip_pos"),
        ))

    foundation_rows = list(FOUNDATION_SEEDS)
    meta = {
        "pack_id": f"flop_srp_btn_bb_{version}",
        "version": version,
        "config": json.dumps(config),
        "provenance": json.dumps({"git_commit": _git_commit(),
                                  "solver_model": config.get("solver_model", "full_street_cfr_plus")}),
        "grade_thresholds_pct_pot": json.dumps(dict(GRADE_THRESHOLDS)),
        "record_count": str(len(rows)),
        "signing_scheme": "hmac-sha256-dev",
        "default_pot_bb": str(pot),
    }
    content_hash = hashlib.sha256(
        _canonical(rows, foundation_rows, list(meta.items())).encode()
    ).hexdigest()
    signature = hmac.new(signing_key, content_hash.encode(), hashlib.sha256).hexdigest()
    meta["content_hash"] = content_hash
    meta["signature"] = signature

    db_path = os.path.join(out_dir, f"flop_pack_{version}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO flop_decision VALUES (%s)" % ",".join("?" * 26), rows)
    conn.executemany("INSERT INTO foundation_template VALUES (?,?,?,?)", foundation_rows)
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


def _read_pack_payload(conn: sqlite3.Connection):
    cols = [d[1] for d in conn.execute("PRAGMA table_info(flop_decision)")]
    decision_rows = conn.execute(f"SELECT {','.join(cols)} FROM flop_decision").fetchall()
    foundation_rows = conn.execute(
        "SELECT id, unit, kind, spec FROM foundation_template"
    ).fetchall()
    meta = dict(conn.execute("SELECT key, value FROM pack_meta").fetchall())
    return decision_rows, foundation_rows, meta


def verify_pack(db_path: str, signing_key: Optional[bytes] = None) -> Dict:
    """Recompute the content hash + signature and compare to stored (integrity)."""
    signing_key = _resolve_signing_key(signing_key)
    conn = sqlite3.connect(db_path)
    decision_rows, foundation_rows, meta = _read_pack_payload(conn)
    conn.close()
    content_hash = hashlib.sha256(
        _canonical(decision_rows, foundation_rows, list(meta.items())).encode()
    ).hexdigest()
    signature = hmac.new(signing_key, content_hash.encode(), hashlib.sha256).hexdigest()
    return {
        "records": len(decision_rows),
        "hash_ok": content_hash == meta.get("content_hash"),
        "signature_ok": signature == meta.get("signature"),
        "version": meta.get("version"),
    }


def resign_pack(db_path: str, signing_key: Optional[bytes] = None) -> Dict:
    """Rewrite content_hash/signature for an existing pack under current rules.

    Used to migrate packs built before foundations/meta were part of the signed
    payload. Does not alter decision or foundation rows.
    """
    signing_key = _resolve_signing_key(signing_key)
    conn = sqlite3.connect(db_path)
    decision_rows, foundation_rows, meta = _read_pack_payload(conn)
    content_hash = hashlib.sha256(
        _canonical(decision_rows, foundation_rows, list(meta.items())).encode()
    ).hexdigest()
    signature = hmac.new(signing_key, content_hash.encode(), hashlib.sha256).hexdigest()
    conn.execute("INSERT OR REPLACE INTO pack_meta VALUES (?, ?)",
                 ("content_hash", content_hash))
    conn.execute("INSERT OR REPLACE INTO pack_meta VALUES (?, ?)",
                 ("signature", signature))
    conn.commit()
    conn.close()
    # refresh gzip sidecar (always rewrite so .db and .db.gz stay in sync)
    gz_path = db_path + ".gz"
    with open(db_path, "rb") as f, gzip.open(gz_path, "wb") as g:
        shutil.copyfileobj(f, g)
    return verify_pack(db_path, signing_key=signing_key)


# Indifference threshold mirrored from content_yield (keep in sync).
_CLEAR_SEP_PCT = 0.5


def refresh_pack_lessons(db_path: str, signing_key: Optional[bytes] = None,
                         pot_default: float = 5.5) -> Dict:
    """Recompute mixed / explanations from stored EVs and backfill role/pot cols.

    Does not re-solve. Fixes packs where top-2 indifference labeled a 3-action
    spot "mixed" while a third action was dominated, and adds pot_bb / oop_pos /
    ip_pos when missing so multi-scenario tooling stays consistent.
    """
    signing_key = _resolve_signing_key(signing_key)
    conn = sqlite3.connect(db_path)
    cols = {d[1] for d in conn.execute("PRAGMA table_info(flop_decision)")}
    meta = dict(conn.execute("SELECT key, value FROM pack_meta").fetchall())
    try:
        cfg = json.loads(meta.get("config") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    pot_fallback = float(cfg.get("pot_bb") or meta.get("default_pot_bb") or pot_default)
    positions = cfg.get("positions") or {}
    oop_fallback = positions.get("oop")
    ip_fallback = positions.get("ip")

    for col, typ in (("pot_bb", "REAL"), ("oop_pos", "TEXT"), ("ip_pos", "TEXT"),
                     ("scenario", "TEXT")):
        if col not in cols:
            conn.execute(f"ALTER TABLE flop_decision ADD COLUMN {col} {typ}")
            cols.add(col)

    select_cols = [
        "id", "board", "board_texture", "board_favored", "node", "acting_player",
        "decision_type", "hand", "hand_category", "actions", "ev", "freq",
        "preferred_action", "ev_sep_pct", "mixed", "reason", "headline", "detail",
        "pot_bb", "oop_pos", "ip_pos", "scenario",
    ]
    rows = conn.execute(f"SELECT {','.join(select_cols)} FROM flop_decision").fetchall()
    updated = 0
    for row in rows:
        (rid, board, btex, bfav, node, actor, dtype, hand, hcat, actions_s, ev_s,
         freq_s, pref, sep, mixed, reason, headline, detail_s, pot_bb, oop_pos,
         ip_pos, scenario) = row
        ev = json.loads(ev_s)
        freq = json.loads(freq_s)
        actions = json.loads(actions_s)
        texture = json.loads(btex) if btex else []
        rec_pot = float(pot_bb) if pot_bb is not None else pot_fallback
        best = max(ev.values())
        regrets = [100.0 * (best - v) / rec_pot for v in ev.values()]
        new_sep = round(sorted(regrets)[1], 3) if len(regrets) > 1 else 0.0
        new_mixed = all(g < _CLEAR_SEP_PCT for g in regrets)
        rec = {
            "node": node, "acting_player": actor, "hand": hand,
            "hand_category": hcat, "preferred": pref, "actions": actions,
            "ev": ev, "freq": freq, "ev_sep_pct": new_sep, "mixed": new_mixed,
            "board_texture": texture,
            "decision_type": dtype or (
                "first_action" if str(node).endswith(("_first", "_vs_check"))
                else "vs_bet"),
        }
        expl = explain(rec, bfav)
        new_oop = oop_pos or oop_fallback
        new_ip = ip_pos or ip_fallback
        changed = (
            bool(mixed) != new_mixed or sep != new_sep or reason != expl["reason"]
            or headline != expl["headline"] or detail_s != json.dumps(expl["detail"])
            or pot_bb is None or oop_pos != new_oop or ip_pos != new_ip
        )
        if not changed:
            continue
        conn.execute(
            "UPDATE flop_decision SET ev_sep_pct=?, mixed=?, reason=?, headline=?, "
            "detail=?, pot_bb=?, oop_pos=?, ip_pos=? WHERE id=?",
            (new_sep, int(new_mixed), expl["reason"], expl["headline"],
             json.dumps(expl["detail"]), rec_pot, new_oop, new_ip, rid),
        )
        updated += 1
    conn.commit()
    conn.close()
    verdict = resign_pack(db_path, signing_key=signing_key)
    return {"updated": updated, **verdict}


DEFAULT_CONFIG = {
    "positions": {"ip": "BTN", "oop": "BB"}, "stack_bb": 100, "pot_bb": 5.5,
    "bet_pct_pot": 66, "rake": 0, "solver_model": "full_street_cfr_plus",
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", help="records.json from content_yield")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--out", default="output/packs")
    ap.add_argument("--pot", type=float, default=5.5)
    ap.add_argument("--resign", help="re-sign an existing pack DB in place")
    ap.add_argument("--refresh-lessons",
                    help="recompute mixed/explanations (+ backfill pot/roles) in place")
    a = ap.parse_args()
    if a.refresh_lessons:
        print("refresh:", refresh_pack_lessons(a.refresh_lessons))
    elif a.resign:
        print("resign:", resign_pack(a.resign))
    else:
        if not a.records:
            ap.error("--records is required unless --resign/--refresh-lessons is set")
        recs = json.load(open(a.records))
        rep = build_pack(recs, DEFAULT_CONFIG, out_dir=a.out, version=a.version, pot=a.pot)
        print(json.dumps(rep, indent=2))
        db = os.path.join(a.out, f"flop_pack_{a.version}.db")
        print("verify:", verify_pack(db))
