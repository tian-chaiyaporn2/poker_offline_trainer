"""Training-question export (Deliverable 7, PRD §5 Step 7) — MIT.

Converts a solved scenario into training-question records and writes them to
JSON and SQLite. Each question is the OOP (BB) flop decision for one hand.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, List

from .cards import card_str, parse_hand, hand_str
from .handinfo import describe_hand

# Preference order of hero hands to sample per board (variety of strengths).
_SAMPLE_CLASSES = [
    "AJs", "ATs", "KQs", "QJs", "JTs", "T9s", "98s", "76s", "54s", "A5s",
    "KJo", "QTo", "JJ", "TT", "99", "77", "55", "22", "A9s", "T8s", "65s", "K9s",
]

# EV-loss thresholds (as fraction of pot) for grading a chosen action.
GOOD_TOL = 0.005      # within 0.5% of pot of the best action
ACCEPT_TOL = 0.02     # within 2% of pot


def _pick_hero_combo(hand_class: str, board_cards: List[str]) -> str | None:
    from .ranges import class_to_combos
    from .cards import parse_cards
    board = set(parse_cards(board_cards))
    for combo in class_to_combos(hand_class):
        if combo[0] not in board and combo[1] not in board:
            return hand_str(combo)
    return None


def _grade(ev_loss_pct_pot: float) -> str:
    if ev_loss_pct_pot <= GOOD_TOL * 100:
        return "good"
    if ev_loss_pct_pot <= ACCEPT_TOL * 100:
        return "acceptable"
    return "costly"


def build_questions(solve: Dict, max_per_board: int = 10) -> List[Dict]:
    board = solve["board"]
    pot = solve["pot_bb"]
    actions = solve["actions"]
    by_hand = {h["hand"]: h for h in solve["per_hand"]}

    questions: List[Dict] = []
    for hand_class in _SAMPLE_CLASSES:
        if len(questions) >= max_per_board:
            break
        hero = _pick_hero_combo(hand_class, board)
        if hero is None or hero not in by_hand:
            continue
        h = by_hand[hero]
        evs = h["action_ev_bb"]
        best_ev = max(evs.values())
        ev_loss = {a: best_ev - evs[a] for a in actions}
        ev_loss_pct = {a: round(100.0 * ev_loss[a] / pot, 4) for a in actions}
        grade = {a: _grade(ev_loss_pct[a]) for a in actions}
        # Recommended = highest-EV action (matches EV-loss grading / "best action" UI).
        # On exact EV ties, prefer the higher solver frequency so the star matches the mix.
        recommended = max(actions, key=lambda a: (evs[a], h["strategy"][a]))
        acceptable = [a for a in actions if grade[a] in ("good", "acceptable")]

        hole = parse_hand(hero)
        descriptor = describe_hand(hole, [_pc(c) for c in board])
        questions.append({
            "id": f"{solve['scenario_id']}__{hero}",
            "scenario_id": solve["scenario_id"],
            "situation": (
                "Heads-up NLHE, single-raised pot, 100bb. You are BB "
                "(out of position) first to act on the flop."
            ),
            "board": board,
            "hero_position": "BB",
            "hero_cards": [hero[0:2], hero[2:4]],
            "hand_class": hand_class,
            "hand_descriptor": descriptor,
            "pot_bb": pot,
            "available_actions": actions,
            "action_frequencies": {a: round(h["strategy"][a], 4) for a in actions},
            "action_ev_bb": {a: round(evs[a], 4) for a in actions},
            "action_ev_loss_pct_pot": ev_loss_pct,
            "action_grade": grade,
            "recommended_action": recommended,
            "acceptable_actions": acceptable,
            "solver_model": solve.get("solver", "pokertrainer_cfr_plus"),
            "model_abstraction": "flop_only_realized_equity",
            "validation_status": "passed",
        })
    return questions


def _pc(card_text: str) -> int:
    from .cards import parse_card
    return parse_card(card_text)


def write_json(questions: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump({"count": len(questions), "questions": questions}, f, indent=2,
                  allow_nan=False)


def write_sqlite(questions: List[Dict], path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS questions")
    conn.execute(
        """CREATE TABLE questions (
            id TEXT PRIMARY KEY,
            scenario_id TEXT,
            board TEXT,
            hero_cards TEXT,
            hand_class TEXT,
            hand_descriptor TEXT,
            pot_bb REAL,
            recommended_action TEXT,
            acceptable_actions TEXT,
            action_frequencies TEXT,
            action_ev_bb TEXT,
            action_grade TEXT,
            validation_status TEXT,
            payload TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT,
            chosen_action TEXT,
            grade TEXT,
            ts TEXT
        )"""
    )
    for q in questions:
        conn.execute(
            "INSERT OR REPLACE INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                q["id"], q["scenario_id"], " ".join(q["board"]),
                " ".join(q["hero_cards"]), q["hand_class"], q["hand_descriptor"],
                q["pot_bb"], q["recommended_action"],
                json.dumps(q["acceptable_actions"]),
                json.dumps(q["action_frequencies"]),
                json.dumps(q["action_ev_bb"]),
                json.dumps(q["action_grade"]),
                q["validation_status"], json.dumps(q, allow_nan=False),
            ),
        )
    conn.commit()
    conn.close()
