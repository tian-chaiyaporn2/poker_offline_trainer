"""Offline trainer for the full-street content pack (PRD v1.3) — MIT, stdlib only.

Serves flop decision questions from a signed content pack (output/packs/*.db):
all four decision nodes (BB/BTN, first action + facing a bet), graded on-device
using the pack's precomputed action grades, with the plain-language explanation.

Run:  python trainer/pack_server.py   → http://127.0.0.1:8000
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import random
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DB = os.path.join(HERE, "pack_results.db")
INDEX = os.path.join(HERE, "pack_index.html")
MAX_BODY = 64 * 1024
_results_lock = __import__("threading").Lock()

import sys
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
from pokertrainer.content_pack import verify_pack

ACTION_LABEL = {"check": "Check", "bet": "Bet 66%", "fold": "Fold",
                "call": "Call", "raise": "Raise 3x"}
QUESTIONS = {}
PACK_META = {}


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _situation(node: str, actor: str, bet_pct: int = 66) -> str:
    """Build situation text from structured fields — works across scenarios."""
    if node.endswith("_first"):
        return f"You are the {actor}, first to act on the flop."
    if node.endswith("_vs_check"):
        return f"You are the {actor}. The opponent checked to you."
    if node.endswith("_vs_bet"):
        return (f"You are the {actor}. You face a {bet_pct}% pot bet "
                f"after checking.")
    return f"You are the {actor} ({node})."


def find_pack() -> str:
    env = os.environ.get("POKERTRAINER_PACK")
    if env:
        if not os.path.exists(env):
            raise SystemExit(f"POKERTRAINER_PACK not found: {env}")
        return env
    packs = glob.glob(os.path.join(ROOT, "output", "packs", "flop_pack_*.db"))
    if not packs:
        raise SystemExit("No content pack found. Build one:\n"
                         "  PYTHONPATH=src python -m pokertrainer.content_pack "
                         "--records output/content_yield_preview/records.json --version v0_preview")
    # Prefer non-demo/non-preview packs; pick newest by mtime (not lexicographic —
    # v9 sorts after v10).
    primary = [p for p in packs
               if "demo" not in os.path.basename(p)
               and "preview" not in os.path.basename(p)]
    candidates = primary or packs
    return max(candidates, key=os.path.getmtime)


def load_pack():
    path = find_pack()
    verdict = verify_pack(path)
    if not (verdict.get("hash_ok") and verdict.get("signature_ok")):
        # Common non-tamper cause: a POKERTRAINER_SIGNING_KEY is exported that
        # differs from the key the pack was signed with (e.g. the shipped packs
        # are dev-signed). Call that out so it isn't mistaken for tampering.
        hint = ""
        if os.environ.get("POKERTRAINER_SIGNING_KEY") and verdict.get("hash_ok") \
                and not verdict.get("signature_ok"):
            hint = ("\n  NOTE: POKERTRAINER_SIGNING_KEY is set but does not match this "
                    "pack's signature.\n  Unset it to serve a dev-signed pack, or rebuild "
                    "the pack with that key (content_pack --records ...).")
        raise SystemExit(
            f"Content pack failed integrity check: {path}\n"
            f"  verify={verdict}\n"
            "Refuse to serve a tampered or unsigned pack." + hint
        )
    conn = sqlite3.connect(path)
    meta = dict(conn.execute("SELECT key, value FROM pack_meta").fetchall())
    rows = conn.execute(
        "SELECT id, board, node, acting_player, hand, actions, ev, freq, "
        "preferred_action, action_grades, reason, headline, detail, mixed "
        "FROM flop_decision").fetchall()
    conn.close()
    try:
        cfg = json.loads(meta.get("config") or "{}")
        bet_pct = int(cfg.get("bet_pct_pot", 66))
    except (TypeError, ValueError, json.JSONDecodeError):
        bet_pct = 66
    q = {}
    for (rid, board, node, actor, hand, actions, ev, freq, pref, grades,
         reason, headline, detail, mixed) in rows:
        cards = board.split() if " " in board else [board[i:i + 2] for i in range(0, len(board), 2)]
        q[rid] = {
            "id": rid, "board": cards, "node": node, "acting_player": actor,
            "hero_cards": [hand[0:2], hand[2:4]],
            "situation": _situation(node, actor, bet_pct),
            "actions": json.loads(actions), "ev": json.loads(ev), "freq": json.loads(freq),
            "preferred_action": pref, "action_grades": json.loads(grades),
            "reason": reason, "headline": headline, "detail": json.loads(detail),
            "mixed": bool(mixed),
        }
    return os.path.basename(path), meta, q, verdict


def public_question(q):
    return {"id": q["id"], "board": q["board"], "node": q["node"],
            "acting_player": q["acting_player"], "hero_cards": q["hero_cards"],
            "situation": q["situation"],
            "actions": [{"key": a, "label": ACTION_LABEL.get(a, a)} for a in q["actions"]]}


def grade_answer(q, action):
    grade = q["action_grades"].get(action, "major_error")
    verdicts = {
        "best": "Best — top action (or effectively tied).",
        "good": "Good — a small concession.",
        "acceptable": "Acceptable — playable, not preferred.",
        "costly": "Costly — a meaningful recurring leak.",
        "major_error": "Major error — clearly dominated here.",
    }
    return {
        "grade": grade, "verdict": verdicts.get(grade, grade),
        "recommended_action": q["preferred_action"], "mixed": q["mixed"],
        "headline": q["headline"], "detail": q["detail"], "reason": q["reason"],
        "action_grades": q["action_grades"],
        "ev_bb": {a: round(q["ev"][a], 2) for a in q["actions"]},
        "freq_pct": {a: round(100 * q["freq"][a]) for a in q["actions"]},
        "labels": {a: ACTION_LABEL.get(a, a) for a in q["actions"]},
    }


def init_results():
    conn = _connect(RESULTS_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "question_id TEXT, node TEXT, chosen TEXT, grade TEXT, ts TEXT)")
    conn.commit(); conn.close()


def record(qid, node, action, grade):
    with _results_lock:
        conn = _connect(RESULTS_DB)
        conn.execute("INSERT INTO results (question_id,node,chosen,grade,ts) VALUES (?,?,?,?,?)",
                     (qid, node, action, grade, dt.datetime.now().isoformat(timespec="seconds")))
        conn.commit(); conn.close()


def stats():
    conn = _connect(RESULTS_DB)
    rows = dict(conn.execute("SELECT grade, COUNT(*) FROM results GROUP BY grade").fetchall())
    total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    conn.close()
    good = rows.get("best", 0) + rows.get("good", 0)
    return {"total": total, "best": rows.get("best", 0), "good": good,
            "acceptable": rows.get("acceptable", 0),
            "costly": rows.get("costly", 0) + rows.get("major_error", 0)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif p == "/api/next":
            if not QUESTIONS:
                self._json({"error": "no questions available"}, 503); return
            self._json(public_question(random.choice(list(QUESTIONS.values()))))
        elif p == "/api/stats":
            self._json({**stats(), "pack": PACK_META})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._json({"error": "bad content-length"}, 400); return
        if length < 0 or length > MAX_BODY:
            self._json({"error": "payload too large"}, 413); return
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid json"}, 400); return
        if urlparse(self.path).path == "/api/answer":
            q = QUESTIONS.get(data.get("question_id"))
            action = data.get("action")
            if not q or action not in q["actions"]:
                self._json({"error": "bad request"}, 400); return
            fb = grade_answer(q, action)
            record(q["id"], q["node"], action, fb["grade"])
            self._json(fb)
        else:
            self._json({"error": "not found"}, 404)


def main():
    global QUESTIONS, PACK_META
    name, meta, QUESTIONS, verdict = load_pack()
    if not QUESTIONS:
        raise SystemExit(f"Content pack {name} has zero decision records.")
    # records count comes from verified row count, never unsigned pack_meta
    PACK_META = {"file": name, "version": meta.get("version"),
                 "records": verdict.get("records"),
                 "signed": bool(verdict.get("hash_ok") and verdict.get("signature_ok"))}
    init_results()
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Full-street trainer — pack {name} ({len(QUESTIONS)} questions).")
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
