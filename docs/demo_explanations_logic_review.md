# Demo App Explanations — Logic Review

Review date: 2026-07-22. Scope: every explanation surface the **demo trainer**
shows a learner — pack-baked `reason` / `headline` / `detail`, plus live JS
(`plainHead`, `RULES`, standing text, factors, mixed `closeExplain`, contrast,
preflop `why`). Kaggle notebooks are out of scope (see `docs/kaggle_bug_review.md`
only as prior-art for review format).

Skips pure solver math. Builds on `docs/solver_to_training_review.md` and the
2026-07-22 contrast / river-trap plain pass.

---

## Surfaces reviewed

| Surface | Source |
|---------|--------|
| Reason tag + poker headline + detail bullets | `explanations.explain` → pack → `_to_q` |
| Beginner plain “why” | `PLAIN_HEAD` / `RIVER_PLAIN` / `bcReframe` |
| Learning term line | `TERMS.learning` (+ river overrides) |
| Rule of thumb (“Explain more”) | `RULES` / `RIVER_RULES` |
| Where you stand | `standingText(handRead(...))` |
| Decision factors | `decisionFactors` |
| Mixed near-tie why | `closeExplain` ← `inferReason` |
| Contrast twin | `findContrast` / `AXIS_WHY` |
| Preflop why / rule | `preflop_content` |

Demo packs embedded by `demo/build_trainer.py`: fullrange flop, raise demo,
turn/river fullrange, SB-vs-BB, BTN-vs-SB (and optional CO/UTG/HJ when present).

---

## Contract checks (all green)

Recomputed over raise-demo + turn/river packs and the live `index.html` deck:

1. **Pack `reason` ≡ `classify_reason(rec)`** — 0 mismatches.
2. **Preferred action ≡ max EV** — 0 mismatches.
3. **`mixed` ⇒ every legal action within 0.5% pot** — 0 mismatches.
4. **Reason ↔ preferred action matrix** (value→bet, fold→fold, …) — 0 mismatches.
5. **River pack headlines** use `RIVER_HEADLINES` for remapped reasons — OK.
6. **Preflop “open … fold from earlier seat”** contrast copy — intentional, not a bug.

Explanations remain **labels** (category × action × texture), not a second solver.

---

## Findings (this PR)

### H1. Board-flush standing claimed nut strength — **FIXED**
- **Where:** demo `standingText` / `handRead`; pack `explain` detail
- **Bug:** On a five-flush board (e.g. `JhJc` on `Qh8h3h2hAh` in the live deck),
  JS treated any flush as “only a full house or better beats you.” Higher flush
  cards still beat you; the board itself is shared.
- **Impact:** Beginner “where you stand” contradicted reality on every board-flush
  river spot (84 improving flush rows in `flop_pack_turnriver_fullrange.db` still
  correctly prefer bet/raise for fold equity / thin value — the *standing* copy
  was the lie).
- **Fix:** Detect `boardFlushAlone` / `boardStraightAlone`; standing + factor
  panel teach shared-board; `explain()` adds a board-alone detail bullet.
  Refresh turn/river packs.

### H2. River “Explain more” / Learning still said cards can come — **FIXED**
- **Where:** demo `RULES`, `TERMS.learning` path in `renderFeedback`
- **Bug:** `plainHead` already had `RIVER_PLAIN` (incl. trap from the prior pass),
  but the rule-of-thumb block and Learning headline still used flop copy
  (“free card”, “still improve”, “catch up”, “dangerous cards could still come”)
  on river protection / trap (and any realization / semi-bluff / odds raise).
- **Fix:** `RIVER_RULES` + `RIVER_LEARNING`; `ruleFor(q)` / learning head
  river-aware.

### M1. Standing said “not value-betting” while solver raised for value — **FIXED**
- **Where:** `standingText` coordinated-board branch
- **Bug:** KK on `Th9h8d2c7c` with reason `raise_value` still read “mostly
  bluff-catching, **not value-betting**.”
- **Fix:** Keep the vulnerability warning; drop the action prescription.

### M2. Mixed `inferReason` called no-pair calls “chase odds” — **FIXED**
- **Where:** `inferReason` → `closeExplain`
- **Bug:** Any no-pair call mapped to `call_odds` (“could improve / cheap enough
  to chase”), including air with no draw and river calls. Pack classifier correctly
  uses `bluff_catch` for air continues.
- **Fix:** `call` + no made hand → `call_odds` only if `rd.draw`, else `bluff_catch`.

### M3. Factor panel said “cards can still come” on the river — **FIXED**
- **Where:** `decisionFactors` board `why`
- **Fix:** Street-aware copy; board-flush / board-straight captions.

---

## Design notes (not bugs)

1. **Board-flush + `value` / `raise_value` reasons** — Heuristic still labels
   improving flushes as value when the solver bets/raises. That can be thin / for
   fold equity; the new detail + standing teach the shared-board caveat without
   forcing every such row to `bluff`.
2. **`bcReframe` only covers pair / two pair / trips** — Flush/straight monsters
   on wet boards are handled by the board-alone standing path instead.
3. **Air `bluff_catch` (“you beat their bluffs”)** — Ace-high can be a fine
   bluff-catcher; headline is intentionally coarse.
4. **Folding overpairs on four-straight rivers** — Solver can correctly fold AA;
   standing now warns vulnerability without saying “you must call.”

---

## Tests / rebuild

- `tests/test_explanations.py` — board-flush detail; river realization headline.
- `python -m pokertrainer.content_pack --refresh-lessons` on turn/river packs.
- `PYTHONPATH=src python demo/build_trainer.py` regenerates `index.html` /
  `demo/trainer_demo.html`.
