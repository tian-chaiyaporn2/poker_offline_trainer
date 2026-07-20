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

ACTION_LABEL_BASE = {"check": "Check", "bet": "Bet 66%", "fold": "Fold", "call": "Call"}
QUESTIONS = {}
PACK_META = {}
ACTION_LABEL = dict(ACTION_LABEL_BASE)
ACTION_LABEL["raise"] = "Raise 3x"


def _action_labels(bet_pct: int = 66, raise_x=None) -> dict:
    labels = {
        "check": "Check",
        "bet": f"Bet {bet_pct}%",
        "fold": "Fold",
        "call": "Call",
    }
    if raise_x is None:
        # Legacy packs omit raise_x; demos historically used 3x.
        labels["raise"] = "Raise 3x"
    else:
        x = float(raise_x)
        labels["raise"] = f"Raise {x:g}x" if x != int(x) else f"Raise {int(x)}x"
    return labels


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _situation(node: str, actor: str, bet_pct: int = 66,
               oop: str | None = None, street: str = "flop") -> str:
    """Build situation text from structured fields — works across scenarios.

    Facing-bet copy differs for OOP (checked, then faced a bet) vs IP (opponent
    led into them). Infer OOP from pack config when provided. Street comes from
    board length so turn/river packs are not labeled as flop.
    """
    on = {"flop": "on the flop", "turn": "on the turn",
          "river": "on the river"}.get(street, "on the board")
    if node.endswith("_first"):
        return f"You are the {actor}, first to act {on}."
    if node.endswith("_vs_check"):
        return f"You are the {actor}. The opponent checked to you {on}."
    if node.endswith("_vs_bet"):
        if oop is not None and actor == oop:
            return (f"You are the {actor}. You checked and face a "
                    f"{bet_pct}% pot bet {on}.")
        return (f"You are the {actor}. The opponent led into you for "
                f"{bet_pct}% of the pot {on}.")
    return f"You are the {actor} ({node}) {on}."


def _street_from_board(cards: list) -> str:
    return {3: "flop", 4: "turn", 5: "river"}.get(len(cards), "flop")


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
    cols = {d[1] for d in conn.execute("PRAGMA table_info(flop_decision)")}
    has_roles = "oop_pos" in cols
    select = (
        "SELECT id, board, node, acting_player, hand, actions, ev, freq, "
        "preferred_action, action_grades, reason, headline, detail, mixed"
        + (", oop_pos, ip_pos" if has_roles else "")
        + " FROM flop_decision"
    )
    rows = conn.execute(select).fetchall()
    conn.close()
    try:
        cfg = json.loads(meta.get("config") or "{}")
        bet_pct = int(cfg.get("bet_pct_pot", 66))
        positions = cfg.get("positions") or {}
        cfg_oop = positions.get("oop")
        raise_x = cfg.get("raise_x")
    except (TypeError, ValueError, json.JSONDecodeError):
        bet_pct = 66
        cfg_oop = None
        raise_x = None
    global ACTION_LABEL
    ACTION_LABEL = _action_labels(bet_pct, raise_x)
    q = {}
    for row in rows:
        if has_roles:
            (rid, board, node, actor, hand, actions, ev, freq, pref, grades,
             reason, headline, detail, mixed, oop_pos, ip_pos) = row
            oop = oop_pos or cfg_oop
        else:
            (rid, board, node, actor, hand, actions, ev, freq, pref, grades,
             reason, headline, detail, mixed) = row
            oop = cfg_oop
        cards = board.split() if " " in board else [board[i:i + 2] for i in range(0, len(board), 2)]
        street = _street_from_board(cards)
        q[rid] = {
            "id": rid, "board": cards, "node": node, "acting_player": actor,
            "hero_cards": [hand[0:2], hand[2:4]], "street": street,
            "situation": _situation(node, actor, bet_pct, oop=oop, street=street),
            "actions": json.loads(actions), "ev": json.loads(ev), "freq": json.loads(freq),
            "preferred_action": pref, "action_grades": json.loads(grades),
            "reason": reason, "headline": headline, "detail": json.loads(detail),
            "mixed": bool(mixed),
        }
    return os.path.basename(path), meta, q, verdict


def public_question(q):
    return {"id": q["id"], "board": q["board"], "node": q["node"],
            "acting_player": q["acting_player"], "hero_cards": q["hero_cards"],
            "situation": q["situation"], "street": q.get("street", "flop"),
            "actions": [{"key": a, "label": ACTION_LABEL.get(a, a)} for a in q["actions"]]}


def grade_answer(q, action):
    from pokertrainer.explanations import freq_pct_ints
    grade = q["action_grades"].get(action, "major_error")
    pref = q["preferred_action"]
    # Grade is EV-loss based; preferred is max-EV (freq on ties). When the pick
    # grades "best" but isn't the starred action, it is still a top play.
    if action == pref:
        verdict = "Best — top action (or effectively tied)."
    elif grade == "best":
        verdict = f"Best — effectively tied with {_label(pref)}."
    elif q.get("mixed") and grade in ("good", "acceptable"):
        verdict = f"Close — any play is fine here (listed preferred: {_label(pref)})."
    else:
        verdicts = {
            "good": "Good — a small concession.",
            "acceptable": "Acceptable — playable, not preferred.",
            "costly": "Costly — a meaningful recurring leak.",
            "major_error": "Major error — clearly dominated here.",
        }
        verdict = verdicts.get(grade, grade)
    return {
        "grade": grade, "verdict": verdict,
        "recommended_action": pref, "mixed": q["mixed"],
        "headline": q["headline"], "detail": q["detail"], "reason": q["reason"],
        "action_grades": q["action_grades"],
        "ev_bb": {a: round(q["ev"][a], 2) for a in q["actions"]},
        "freq_pct": freq_pct_ints(q["freq"], order=q["actions"]),
        "labels": {a: ACTION_LABEL.get(a, a) for a in q["actions"]},
    }


def _label(action: str) -> str:
    return ACTION_LABEL.get(action, action)


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
