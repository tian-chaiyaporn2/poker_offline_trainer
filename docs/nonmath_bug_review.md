# Non-Math / Non-Kaggle Bug Review

Review date: 2026-07-20. Scope: demo builders, trainer UI/servers, content
pack/yield/export, foundations/explanations/priority wiring, scenario loading,
and generation orchestration — **excluding** pure solver math and Kaggle
notebooks (see `docs/bug_review.md`, `docs/kaggle_bug_review.md`).

All items below were independently reproduced. Status reflects this PR.

---

## Critical / High

### H1. Pack signature covered only `flop_decision` rows — **FIXED**
- **Where:** `content_pack.py` `build_pack` / `verify_pack`
- **Bug:** `content_hash` / `signature` hashed decision rows only.
  `foundation_template` and almost all `pack_meta` (including `record_count`)
  could be rewritten while `verify_pack()` still returned `hash_ok` +
  `signature_ok`.
- **Impact:** Integrity claim was false for metadata and foundations.
  Combined with `pack_index.html` inserting `s.pack.records` via `innerHTML`,
  a tampered `record_count` became XSS on the trainer origin.
- **Fix:** Canonical payload now covers decisions + foundations + all meta
  except `content_hash`/`signature`. `resign_pack()` migrates existing DBs.
  Pack server reports verified row count, not unsigned meta. Stats UI uses
  DOM/`textContent`.

### H2. Turn/river demo solved a 6–7 card “board” — **FIXED**
- **Where:** `validate_flop._make_solver`, `demo/gen_turn_river.py`,
  chance denom in `solver/batched.py` + `batched_gpu.py`
- **Bug:** `_make_solver` hard-coded `streets=3` and treated the caller’s
  `streets=` arg as `bet_streets` only. A 4-card turn start still dealt two
  more cards → 6-card showdowns. Chance denom used `street+2` (flop-shaped)
  instead of `len(board)`.
- **Impact:** Turn/river demo EVs and recommendations were not the stated game.
- **Fix:** Derive `n_streets` from board length (flop→3, turn→2, river→1);
  chance denom uses `52 - len(board) - 4`. Regenerated
  `flop_pack_turnriver_demo.db`.

### H3. Demo builders never verified packs — **FIXED**
- **Where:** `demo/build_preview.py`, `demo/build_trainer.py`,
  `demo/gen_turn_river.py`
- **Bug:** Static pages claimed “Signed & integrity-verified” while reading
  SQLite directly. `gen_turn_river` printed verify but did not fail.
- **Fix:** `verify_pack()` required before embed; abort unless both checks
  pass. Generator exits non-zero on verify failure.

### H4. Static trainer XSS via embedded pack JSON / `innerHTML` — **FIXED**
- **Where:** `demo/build_trainer.py`, `demo/build_preview.py`
- **Bug:** Pack strings embedded with raw `json.dumps` (no `</script>` escape)
  and rendered with `innerHTML` for labels/cards/bars. Preview interpolated
  meta/hand fields into HTML without escaping.
- **Fix:** Escape `<` as `\u003c` when embedding JSON; build DOM with
  `textContent`; `html.escape` preview fields.

---

## Medium

### M1. Multi-scenario packs collided on `record_id` — **FIXED**
- **Where:** `content_pack.record_id` / schema / dedup
- **Bug:** IDs hashed `(board, node, hand, version)` only; schema omitted
  `scenario`. Combining BTN-vs-BB and SB-vs-BB could `UNIQUE` collide or mix
  concepts.
- **Fix:** Include `scenario` in id, row, index, and dedup key.

### M2. Explanations mislabeled relabeled first-action nodes — **FIXED**
- **Where:** `explanations.classify_reason`
- **Bug:** Only `bb_first` / `btn_vs_check` counted as first-action. Relabeled
  `sb_first` / `bb_vs_check` fell through to the response branch → value bets
  labeled `fold`.
- **Fix:** Prefer `decision_type == "first_action"`; also accept `_first` /
  `_vs_check` suffixes.

### M3. Export omitted `"acceptable"` from `acceptable_actions` — **FIXED**
- **Where:** `export.build_questions`
- **Bug:** Grades included `"acceptable"`, but the list kept only `"good"`.
- **Fix:** Include both `"good"` and `"acceptable"`.

### M4. `--aggregate-only` skipped checkpoint config checks — **FIXED**
- **Where:** `content_yield.run`
- **Bug:** Aggregate-only rebuilds ignored `solve_config.json`, reopening the
  checkpoint-mixing hole fixed for normal resumes.
- **Fix:** Always call `_ensure_checkpoint_config` (never `--fresh` clear when
  aggregating).

### M5. Scenario loader accepted partial `actions.allowed` — **FIXED**
- **Where:** `scenario.load_scenario`
- **Bug:** Any subset of supported actions was accepted, but `FlopSolver`
  always solves the full fixed tree.
- **Fix:** Require the full supported `allowed` set (same spirit as M6 in
  `bug_review.md`).

### M6. Pack server “newest pack” was lexicographic — **FIXED**
- **Where:** `trainer/pack_server.find_pack`
- **Bug:** `sorted(...)[-1]` preferred `v9` over `v10`; demos could win.
- **Fix:** Prefer non-demo/non-preview packs; pick by mtime; honor
  `POKERTRAINER_PACK`.

### M7. Hard-coded BTN-vs-BB situation text — **FIXED**
- **Where:** `trainer/pack_server.py`
- **Bug:** `bb_vs_bet` always said “BTN bet”, wrong for SB-vs-BB packs.
- **Fix:** Build situation from node suffix + acting player + pack bet %.

### M8. NaN records built packs that failed their own verify — **FIXED**
- **Where:** `content_pack.build_pack`
- **Bug:** SQLite stores NaN REAL as NULL; hash was over Python rows → immediate
  verify failure.
- **Fix:** Reject non-finite numerics before insert; `allow_nan=False` in
  canonical JSON.

### M9. Turn/river UI claimed “checked through” ranges — **FIXED**
- **Where:** `demo/gen_turn_river.py`, `demo/build_trainer.py`
- **Bug:** Generator used unconditioned SRP ranges but copy claimed check/check
  filtered ranges.
- **Fix:** Relabel as unconditioned later-street demo; update trainer footer /
  situation text.

---

## Low

### L1. Empty question sets crashed `/api/next` — **FIXED**
- Fail fast at startup; HTTP 503 if empty.

### L2. Answer fetch errors left buttons disabled — **FIXED**
- `try/catch` + re-enable / error verdict in both trainer UIs.

### L3. Questions DB open always set WAL — **FIXED**
- Read-only URI for `trainer.db`; WAL only on writable results DB.

### L4. `generate.py` progress said `/12` while `BOARDS` has 17 — **FIXED**
- Progress uses `len(BOARDS)`; docstring updated.

---

## Artifacts updated

- Re-signed: `flop_pack_v1_fullrange.db`, `v0_preview`, `v1_raise_demo`
  (expanded integrity payload).
- Regenerated: `flop_pack_turnriver_demo.db` (correct street tree; reduced-range
  demo params `N=30`, `ITERS=60` for this pass).
- Rebuilt: `demo/trainer_demo.html`, `demo/content_preview.html`, `index.html`,
  `preview.html`.

## Not bugs / deferred

- **Public dev signing key fallback** — still intentional for local demos;
  production should set `POKERTRAINER_SIGNING_KEY`.
- **Unauthenticated localhost trainer APIs** — acceptable for single-user
  offline POC; Origin checks would be a later hardening step.
- **Full-range raise / check-check-filtered turn-river packs** — still future
  depth work; demos remain reduced-range and (for turn/river) unconditioned.
