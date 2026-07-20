# Solver → Recommendation → Training Review

Review date: 2026-07-20. Scope: how full-street CFR outputs become the
**recommended action**, **grades**, **explanations**, and **trainer feedback**.

---

## Pipeline (production path)

```
BatchedCFR / BatchedGPUCFR  (averaged strategy S + action utilities U)
        │
        ▼
_flop_decisions_from_cap / preferred_action()
        │  preferred = max EV; EV ties → higher frequency
        ▼
content_yield.extract_records()
        │  reach gate, mixed flag, hand/board tags, explain()
        ▼
content_pack.build_pack()
        │  EV-regret grades, dedup, signed SQLite
        ▼
trainer (pack_server / demo build_trainer)
```

Legacy flop-only path (`runner` → `export.build_questions` → `trainer.db`) still
exists with different action names and coarser grades; do not mix contracts.

---

## Contract: what “recommended” means

| Layer | Rule |
|-------|------|
| Preferred / recommended | **Max EV** under the averaged strategy |
| EV ties | Higher **solver frequency** (new; was insertion order) |
| Grades | EV regret as % of pot vs best EV |
| Mixed lesson | EV gap &lt; 0.5% pot → `mixed=True`, reason `mixed` |
| Frequency bars | Teaching detail — **not** the recommendation selector |

Integrity checks reject `preferred` that is not max-EV before a pack is signed.

---

## Findings fixed in this pass

### H1. Trainer called tied-best picks a “costly leak” — **FIXED**
- **Where:** `demo/build_trainer.py` feedback; also `trainer/pack_server.grade_answer`
- **Bug:** Verdict branched on `chosen === preferred`, then `good` / `acceptable`,
  else “costly leak”. ~11% of full-range pack rows have **two** actions graded
  `best` (gap ≤ 0.25% pot). Picking the non-starred one showed grade color
  `best` but text “costly leak”.
- **Fix:** Honor grade first: `best` (even if not starred) is a top play; mixed
  + good/acceptable gets soft “either is fine” copy.

### H2. `mixed` flag dropped before the trainer UI — **FIXED**
- **Where:** `demo/build_trainer._to_q`
- **Bug:** Pack column `mixed` was selected then discarded, so the UI could not
  soften near-indifferent spots.
- **Fix:** Pass `mixed` into question JSON; use it in feedback.

### M1. EV ties preferred the first action key — **FIXED**
- **Where:** `solver/batched.preferred_action`, GPU `flop_root_report`, `export`
- **Bug:** `max(ev, key=ev.get)` / `check if ev_ch >= ev_bt` preferred check/fold
  on exact ties even when the mix put more mass on bet/call.
- **Fix:** Tie-break by frequency so the star matches what the solver plays more.

### M2. Priority impact used a global pot — **FIXED**
- **Where:** `priority.score_records`
- **Bug:** After per-record `pot_bb` landed in packs, priority still divided by
  the CLI default (5.5), skewing SB-vs-BB impact.
- **Fix:** Use `record["pot_bb"]` when present.

### L1. Legacy trainer labeled recommended as “GTO pick” — **FIXED**
- Max-EV recommendation is not necessarily the modal frequency; tag is now
  “recommended”.

---

## Design notes (not bugs)

1. **★ vs frequency bars** — In ~6% of full-range rows (and ~29% of the reduced
   raise demo), preferred ≠ highest frequency. Bars sort by freq; ★ marks max EV.
   That is intentional for EV-loss grading; the star tooltip now says “Best EV”.
2. **Explanations are labels** — `explanations.classify_reason` is a heuristic
   matrix over hand category × action × texture. It does not invent strategy,
   but reason tags can disagree with nuanced GTO motives.
3. **Two pipelines** — Full-street pack vs flop-only export still differ in
   actions, grade thresholds, and schemas.
4. **Indifference thresholds** — 0.25% (grade `best`) / 0.5% (`mixed`) / 1.0%
   (cross-solver compare) remain intentionally layered.

---

## Tests

- `tests/test_solver_to_training.py` — preferred selection, export tie-break,
  per-record priority pot.
