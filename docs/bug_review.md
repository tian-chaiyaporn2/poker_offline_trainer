# Comprehensive Bug Review

Review date: 2026-07-19. Scope: full repository (solver, equity, export,
trainer UI/servers, content pack, benches, notebooks, tests).

All items below were independently reproduced unless marked *open / residual*.
Fixes landed in the same change set are marked **FIXED**.

---

## Critical / High

### H1. Multi-street chance nodes used the wrong card-removal denominator — **FIXED**
- **Where:** `solver/batched.py`, `solver/batched_gpu.py`, `solver/multistreet.py`
- **Bug:** Chance nodes divided by `52 - board - 2` (47 on the flop) while
  masking cards that collide with *either* private hand. For a live combo pair
  the correct average is over `52 - board - 4` (45) cards.
- **Impact:** Pure runout EV was scaled by `(45/47)*(44/46) ≈ 0.916`. Fold
  payoffs were *not* scaled, so betting lines that buy folds were overvalued
  relative to check-check / call lines. This biases full-street strategies and
  every content pack built from `BatchedCFR` / `BatchedGPUCFR`.
- **Evidence:** `AhAd` vs `KhQc` on `AsKd2c`: exact equity `0.9828…`, pre-fix
  solver returned `0.9001…` (= exact × scale).
- **Fix:** Denominator is now `52 - board - 4` (and `len(deck) - 4` in the
  multistreet oracle). Regression: `tests/test_bugfixes.py`.

### H2. Flush/straight *draws* classified as `strong_made` — **FIXED**
- **Where:** `validate_flop.hand_category`, consumed by `content_yield` /
  explanations
- **Bug:** Substring match for `"flush"` / `"straight"` ran before `"draw"`, so
  `"high card + flush draw"` became `strong_made` and got “value” / “trap”
  explanations.
- **Fix:** Classify from the made-hand token only (`descriptor.split(" + ")[0]`).

### H3. Pack trainer never verified pack integrity — **FIXED**
- **Where:** `trainer/pack_server.py`
- **Bug:** Loaded the newest `flop_pack_*.db` and showed `signed: true` if a
  `signature` metadata key existed, without calling `verify_pack()`.
- **Fix:** `verify_pack()` runs at load; refuse to serve unless `hash_ok` and
  `signature_ok`. UI `signed` flag comes from verification.

### H4. Aggregate root EV / exploitability not divided by joint mass — **FIXED**
- **Where:** `solver/cfr.py`, `reference_solver.py`
- **Bug:** OOP/IP ranges are normalized independently; incompatible pairs are
  zeroed by `compat`, but aggregate EV used the un-normalized product measure.
  With 50% blocker overlap, reported root EV was ~½ of the conditional EV.
- **Fix:** Divide root EV and exploitability terms by `w_o @ (C @ w_i)`.

### H5. Embedded Colab/Kaggle notebooks ship stale solver code — *open*
- **Where:** `colab/*.ipynb` embedded ZIP payloads
- **Bug:** Copies of `batched_gpu.py` / `validate_flop.py` predate the float32
  regret cast and the corrected flop-only model (`streets=3, bet_streets=1`).
  Notebook “flop-only” runs were actually 5-card flop showdowns.
- **Impact:** Validation conclusions from those notebooks can disagree with
  current `src/`.
- **Recommended:** Regenerate embedded sources from `src/`, or drop embeds and
  install the package.

### H6. `compare()` could pass on incompatible solves — **FIXED**
- **Where:** `compare.py`
- **Bug:** Empty shared-hand set → agreement `1.0`, EV diffs `0.0`,
  `all_targets_pass=True`. Boards / scenario IDs were never checked. Negative
  probabilities summing to 1 counted as valid.
- **Fix:** Raise on scenario/board/action mismatch or empty shared set; reject
  out-of-range probabilities; require complete hand coverage for pass.

---

## Medium

### M1. IP-vs-check action EVs used the wrong opponent mass — **FIXED**
- **Where:** `solver/cfr.py` `_report_action_evs`
- **Bug:** Divided by full OOP mass `C.T @ w_o` instead of the checked reach
  `C.T @ (w_o * s_root[:,0])`, scaling IP EVs by OOP’s check frequency.
- **Fix:** Use checked reach as the denominator.

### M2. Board-pair air labeled as `weak_pair` — **FIXED**
- **Where:** `handinfo.describe_hand`
- **Bug:** On paired boards, hole cards that don’t pair anything still evaluate
  as `"one pair"` (board pair), which mapped to `weak_pair`.
- **Fix:** If no hole rank pairs the board and hero isn’t a pocket pair, label
  `high card`.

### M3. Pack signing key is hard-coded in source — *open*
- **Where:** `content_pack.DEV_SIGNING_KEY`
- **Bug:** Anyone with the repo can forge a verifying HMAC signature.
- **Recommended:** Asymmetric signing (private key out of repo, public key
  pinned) or require a secret via env and refuse the dev key outside tests.

### M4. XSS via pack fields in `pack_index.html` — **FIXED**
- **Where:** `trainer/pack_index.html`
- **Bug:** `headline` / `reason` / action labels inserted with `innerHTML`.
- **Fix:** Build those nodes with `textContent`.

### M5. `run()` reports last-iterate EV, not average-strategy EV — *open*
- **Where:** `batched.py` / `batched_gpu.py` / `multistreet.py` `run()`
- **Bug:** Training traversal uses current regret-matched strategy; CFR
  guarantees apply to the average. `flop_decisions_report()` already uses
  average strategies, so `run()["root_ev_*"]` can disagree with reports.
- **Note:** Left unchanged so oracle parity tests (batched ≡ multistreet)
  stay bit-identical; fix both together.

### M6. Scenario tree fields silently ignored — *open*
- **Where:** `scenario.py` / `runner.py`
- **Bug:** `allowed` actions, `raise_rule`, acting player, etc. are not wired;
  `FlopSolver` always uses the hard-coded 6-infoset tree.
- **Recommended:** Reject unsupported fields or route to the matching solver.

### M7. Hard-coded absolute path in C kernel bench — **FIXED**
- **Where:** `bench/bench_kernel.py`
- **Fix:** Load `kernel.so` next to `kernel.c`; compare both OOP and IP outputs.

### M8. Recommended action ≠ max-EV action in flop trainer export — **FIXED**
- **Where:** `export.py`
- **Bug:** Graded by EV regret but recommended the highest-*frequency* action;
  3/120 committed questions disagreed with max EV while the UI said “best”.
- **Fix:** Recommend `argmax(ev)`.

---

## Low

### L1. `equity_matrix` accepted board-colliding hands / broke API contract — **FIXED**
- Raises on board collisions; incompatible pairs now return equity `0.5` as
  documented.

### L2. `iterations <= 0` / `samples <= 0` crashed unclearly — **FIXED**
- `ValueError` in batched/GPU/multistreet `run()` and `mc_equity`.

### L3. HTTP servers: unbounded body / no JSON error handling — *open*
- **Where:** `trainer/server.py`, `trainer/pack_server.py`
- Malformed `Content-Length` / JSON drops the connection; no size cap.

### L4. p90 regret used a biased index — **FIXED**
- **Where:** `validate_flop.py` summary — nearest-rank `ceil(0.9*n)-1`.

### L5. Range expansion accepts weights `> 1` and duplicate aliases — *open*
- **Where:** `ranges.py` — `AKs`+`KAs` duplicates combos; schema says `[0,1]`.

### L6. `SolveResult.action_labels` lists 6 infosets but `action_ev` has 2 — *open*
- **Where:** `solver/cfr.py`

### L7. GPU bench timing subtracts unrelated one-iter cost — *open*
- **Where:** `bench/gpu_bench.py`

### L8. MC equity spot-check can return 0 error if all samples collide — *open*
- **Where:** `benchmark.py` `_mc_spotcheck`

### L9. SQLite writes from `ThreadingHTTPServer` without timeout/WAL — *open*
- Concurrent answers can raise `database is locked`.

---

## Test coverage gaps (not bugs, but missed regressions)

- No prior test for chance-node runout EV vs exact equity.
- No `compare()` negative cases.
- GPU parity tests only cover `streets in (1, 2)`, not full 3-street.
- Notebooks are not exercised in CI.

New coverage: `tests/test_bugfixes.py` (chance denom, categories, equity API,
joint-mass EV, compare guards).

---

## What was verified clean

- 5–7 card evaluator (wheel, categories) — no confirmed ranking bug.
- Card encoding / parsing — OK for validated scenario inputs.
- Flop-only `FlopSolver` strategy learning math (aside from reporting bugs above).
- Existing pytest suite: all green after fixes.

---

## Residual risk for committed artifacts

- `output/trainer.db` / `output/questions.json` were generated with the old
  frequency-based `recommended_action` (3 mismatches). Regenerate with
  `python -m pokertrainer.generate` if you want the export fix reflected.
- Full-street packs under `output/packs/` were built with the wrong chance
  denominator; rebuild packs after this fix before treating them as authoritative.
- Colab/Kaggle notebook embeds remain stale until regenerated.
