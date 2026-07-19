# Comprehensive Bug Review

Review date: 2026-07-19. Scope: full repository (solver, equity, export,
trainer UI/servers, content pack, benches, notebooks, tests).

All items below were independently reproduced. Status reflects this PR.

---

## Critical / High

### H1. Multi-street chance nodes used the wrong card-removal denominator — **FIXED**
- **Where:** `solver/batched.py`, `solver/batched_gpu.py`, `solver/multistreet.py`
- **Bug:** Chance nodes divided by `52 - board - 2` (47 on the flop) while
  masking cards that collide with *either* private hand. Correct average is
  over `52 - board - 4` (45) cards.
- **Impact:** Pure runout EV scaled by `(45/47)*(44/46) ≈ 0.916`. Fold payoffs
  were not scaled → overvalued fold equity in full-street strategies/packs.
- **Fix:** Denominator is now `52 - board - 4`. Regression in `test_bugfixes.py`.

### H2. Flush/straight *draws* classified as `strong_made` — **FIXED**
- **Where:** `validate_flop.hand_category`
- **Fix:** Classify from the made-hand token only (`descriptor.split(" + ")[0]`).

### H3. Pack trainer never verified pack integrity — **FIXED**
- **Where:** `trainer/pack_server.py`
- **Fix:** `verify_pack()` at load; refuse to serve unless hash+signature OK.

### H4. Aggregate root EV / exploitability not divided by joint mass — **FIXED**
- **Where:** `solver/cfr.py`, `reference_solver.py`
- **Fix:** Divide by `w_o @ (C @ w_i)`.

### H5. Embedded Colab/Kaggle notebooks shipped stale solver code — **FIXED**
- **Where:** `colab/*.ipynb` embeds
- **Fix:** `scripts/refresh_notebook_embeds.py` regenerates ZIPs from current
  `src/` + `bench/`; all four embedded notebooks refreshed.

### H6. `compare()` could pass on incompatible solves — **FIXED**
- **Where:** `compare.py`
- **Fix:** Raise on scenario/board/action mismatch or empty shared set; reject
  out-of-range probabilities; require complete hand coverage.

---

## Medium

### M1. IP-vs-check action EVs used the wrong opponent mass — **FIXED**
- **Fix:** Denominator is checked reach `C.T @ (w_o * s_root[:,0])`.

### M2. Board-pair air labeled as `weak_pair` — **FIXED**
- **Fix:** `describe_hand` labels board-pair-only holdings as `high card`.

### M3. Pack signing key hard-coded — **FIXED** (dev-key fallback retained)
- **Fix:** `POKERTRAINER_SIGNING_KEY` env overrides; explicit key still accepted;
  tests/local keep the documented dev key as fallback.

### M4. XSS via pack fields in `pack_index.html` — **FIXED**
- **Fix:** Pack strings rendered with `textContent`.

### M5. `run()` reported last-iterate EV, not average-strategy EV — **FIXED**
- **Where:** `batched.py`, `batched_gpu.py`, `multistreet.py`
- **Fix:** After training, eval-mode traversal reports average-strategy root EV.
  Oracle parity preserved (both solvers updated together).

### M6. Scenario tree fields silently ignored — **FIXED**
- **Where:** `scenario.load_scenario`
- **Fix:** Reject unsupported `acting_player`, `allowed`, `raise_rule`, and
  multi-street `tree.streets` instead of solving a different game silently.

### M7. Hard-coded absolute path in C kernel bench — **FIXED**

### M8. Recommended action ≠ max-EV action in flop trainer export — **FIXED**
- **Fix:** Recommend `argmax(ev)`; `output/trainer.db` regenerated.

---

## Low

### L1. `equity_matrix` board collisions / API contract — **FIXED**
### L2. `iterations <= 0` / `samples <= 0` — **FIXED**
### L3. HTTP body/JSON error handling — **FIXED**
- 400/413 responses; 64 KiB body cap on both trainers.
### L4. p90 regret index — **FIXED**
### L5. Range weights `> 1` / duplicate aliases — **FIXED**
- Reject weights outside `[0,1]` and overlapping classes (e.g. `AKs`+`KAs`).
### L6. `action_ev` missing response infosets — **FIXED**
- All six infosets now report action EVs.
### L7. GPU bench timing — **FIXED**
- Warmup on the same instance; ms/iter = `runtime_sec / K`.
### L8. MC equity spot-check false zero — **FIXED**
- Resample until `n_pairs` valid; per-pair seeds; `None` if none found.
### L9. SQLite threading — **FIXED**
- WAL + timeout + write lock on both trainer servers.

---

## Artifacts regenerated

- `output/questions.json`, `output/trainer.db`, `output/solves/*`,
  `output/generation_summary.json` (flop-only library)
- `output/content_yield_preview/*` + `output/packs/flop_pack_v0_preview.db`
- `output/cy_raise/*` + `output/packs/flop_pack_v1_raise_demo.db`
- Colab/Kaggle notebook embeds via `scripts/refresh_notebook_embeds.py`

---

## Tests

- Full pytest suite green
- New coverage in `tests/test_bugfixes.py` for chance denom, categories,
  equity API, joint-mass EV, compare guards, ranges, scenario rejection,
  full action_ev contract, average-strategy `run()` parity
