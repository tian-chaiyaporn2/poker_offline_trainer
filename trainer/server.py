"""Offline local poker trainer (PRD §5 Step 8, Deliverable 8) — MIT.

A dependency-free web server (Python standard library only). No account, no
cloud, no ads, no network calls. Serves training questions from the generated
SQLite database and records the user's results locally.

Run:
    python trainer/server.py            # then open http://127.0.0.1:8000

The answer key (frequencies, EVs, grading) is only sent AFTER the user submits
an action, so questions are a fair test.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QUESTIONS_DB = os.path.join(ROOT, "output", "trainer.db")
RESULTS_DB = os.path.join(HERE, "results.db")
INDEX_HTML = os.path.join(HERE, "index.html")
MAX_BODY = 64 * 1024
_results_lock = __import__("threading").Lock()


def _connect_rw(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _connect_ro(path: str) -> sqlite3.Connection:
    """Read-only open — works when the questions DB lives on a read-only volume."""
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30, check_same_thread=False)


def load_questions() -> dict:
    if not os.path.exists(QUESTIONS_DB):
        raise SystemExit(
            f"Question DB not found at {QUESTIONS_DB}.\n"
            "Generate it first:  PYTHONPATH=src python -m pokertrainer.generate"
        )
    conn = _connect_ro(QUESTIONS_DB)
    rows = conn.execute("SELECT id, payload FROM questions").fetchall()
    conn.close()
    return {qid: json.loads(payload) for qid, payload in rows}


def init_results_db() -> None:
    conn = _connect_rw(RESULTS_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT, chosen_action TEXT, grade TEXT,
            ev_loss_pct_pot REAL, ts TEXT
        )"""
    )
    conn.commit()
    conn.close()


QUESTIONS = {}


def public_question(q: dict) -> dict:
    """Question fields safe to reveal before the user answers (no answer key)."""
    return {
        "id": q["id"],
        "situation": q["situation"],
        "board": q["board"],
        "hero_position": q["hero_position"],
        "hero_cards": q["hero_cards"],
        "pot_bb": q["pot_bb"],
        "available_actions": q["available_actions"],
    }


def grade_answer(q: dict, action: str) -> dict:
    grade = q["action_grade"].get(action, "costly")
    ev_loss = q["action_ev_loss_pct_pot"].get(action, 0.0)
    recommended = q["recommended_action"]
    # Grade is EV-loss; recommended is max-EV. A non-recommended "good" is close,
    # not "GTO-optimal" — that wording belongs only to the listed recommendation.
    if action == recommended and grade == "good":
        verdict = "Good — top action (or within 0.5% of the pot of it)."
    elif grade == "good":
        verdict = (f"Good — close to the best action "
                   f"({_label(recommended)}); a small concession.")
    elif grade == "acceptable":
        verdict = (f"Acceptable — slightly suboptimal, costing about "
                   f"{ev_loss:.1f}% of the pot vs the best action.")
    else:
        verdict = (f"Costly — this loses about {ev_loss:.1f}% of the pot vs the "
                   f"best action ({_label(recommended)}).")
    return {
        "grade": grade,
        "verdict": verdict,
        "ev_loss_pct_pot": ev_loss,
        "recommended_action": recommended,
        "acceptable_actions": q["acceptable_actions"],
        "action_frequencies": q["action_frequencies"],
        "action_ev_bb": q["action_ev_bb"],
        "action_grade": q["action_grade"],
        "hand_descriptor": q["hand_descriptor"],
        "model_abstraction": q.get("model_abstraction", ""),
    }


def _label(a: str) -> str:
    return {"check": "check", "bet_small": "small bet (33%)",
            "bet_large": "large bet (75%)"}.get(a, a)


def record_result(question_id: str, action: str, grade: str, ev_loss: float) -> None:
    with _results_lock:
        conn = _connect_rw(RESULTS_DB)
        conn.execute(
            "INSERT INTO results (question_id, chosen_action, grade, ev_loss_pct_pot, ts)"
            " VALUES (?,?,?,?,?)",
            (question_id, action, grade, ev_loss, dt.datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()


def stats() -> dict:
    conn = _connect_rw(RESULTS_DB)
    rows = conn.execute("SELECT grade, COUNT(*) FROM results GROUP BY grade").fetchall()
    total = conn.execute("SELECT COUNT(*), AVG(ev_loss_pct_pot) FROM results").fetchone()
    conn.close()
    counts = {g: c for g, c in rows}
    n = total[0] or 0
    return {
        "total": n,
        "good": counts.get("good", 0),
        "acceptable": counts.get("acceptable", 0),
        "costly": counts.get("costly", 0),
        "avg_ev_loss_pct_pot": round(total[1], 3) if total[1] is not None else 0.0,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(INDEX_HTML, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/next":
            if not QUESTIONS:
                self._send_json({"error": "no questions available"}, 503)
                return
            q = random.choice(list(QUESTIONS.values()))
            self._send_json(public_question(q))
        elif path == "/api/stats":
            self._send_json(stats())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_json({"error": "bad content-length"}, 400)
            return
        if length < 0 or length > MAX_BODY:
            self._send_json({"error": "payload too large"}, 413)
            return
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return
        if path == "/api/answer":
            q = QUESTIONS.get(data.get("question_id"))
            action = data.get("action")
            if q is None or action not in q["available_actions"]:
                self._send_json({"error": "bad request"}, 400)
                return
            feedback = grade_answer(q, action)
            record_result(q["id"], action, feedback["grade"], feedback["ev_loss_pct_pot"])
            self._send_json(feedback)
        else:
            self._send_json({"error": "not found"}, 404)


def main():
    global QUESTIONS
    QUESTIONS = load_questions()
    if not QUESTIONS:
        raise SystemExit(f"No questions loaded from {QUESTIONS_DB}.")
    init_results_db()
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Poker Offline Trainer — {len(QUESTIONS)} questions loaded.")
    print(f"Open  http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
