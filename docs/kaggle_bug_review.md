# Kaggle Run Bug Review

Review date: 2026-07-19. Scope: the two notebooks we actually run on Kaggle
(`colab/kaggle_fullrange_validation.ipynb`, `colab/kaggle_content_yield.ipynb`)
and the modules they invoke (`validate_flop`, `content_yield`, `content_pack`,
GPU solvers).

Prior solver bugs from `docs/bug_review.md` (H1 chance denom, H4 joint mass, M5
average strategy, M1 opponent mass, H2/M2 categories) remain fixed and tested.
This pass focuses on **run-orchestration and resume safety** that would burn a
GPU commit.

---

## Findings (this PR)

### K1. Optional raise half always ran on Save & Run All — **FIXED**
- **Where:** `colab/kaggle_content_yield.ipynb` final cell
- **Bug:** Markdown called the raise path “Optional”, but the code cell had no
  gate. Run All did the full 12-board pack (~5–7 h) **then** started
  `--raise-x 3` boards 0–5 (multi-hour), risking the 12 h session limit and a
  failed commit even when the main pack succeeded.
- **Fix:** `RUN_RAISE_HALF = False` by default; raise solve only when explicitly
  enabled.

### K2. Checkpoint resume ignored solve settings — **FIXED**
- **Where:** `content_yield.py`
- **Bug:** `_valid_checkpoint` only checked “non-empty JSON list”. Re-running
  into the same `--out` with different `iters` / `dtype` / `raise_x` / boards
  reused old board files and mixed them into `records.json`.
- **Fix:** Write `solve_config.json` fingerprint; refuse mismatch; `--fresh`
  clears board checkpoints. Orphan boards without a config also refuse resume.

### K3. `validate_records()` was never called — **FIXED**
- **Where:** `content_yield.py`
- **Bug:** Freq-sum / preferred-key checks existed but only `_is_finite_record`
  ran. Bad strategies would not be logged.
- **Fix:** Call `validate_records` per board; write `board_XX.VALIDATE.json`
  when warnings appear (still keep finite records — do not discard a whole
  board for soft warnings).

### K4. Validation notebook aborted the whole run on one board crash — **FIXED**
- **Where:** `validate_flop.py`
- **Bug:** Unlike content-yield, one CuPy/OOM error aborted the process. Prior
  boards were on disk via incremental write, but the commit failed and later
  boards were missing.
- **Fix:** Per-board try/except; log `board_XX.ERROR.txt`; continue; empty-row
  `_write` safe.

### K5. Validation notebook embed one file behind — **FIXED**
- **Where:** `kaggle_fullrange_validation.ipynb` embedded ZIP
- **Bug:** Embed matched solver/`validate_flop` but lagged on `content_yield.py`
  (unused by that notebook). `bug_review.md` H5 overstated “all refreshed”.
- **Fix:** Re-run `scripts/refresh_notebook_embeds.py`.

---

## Not bugs (clarifications)

- **Showdown `E_all` stored as float32 even when `--dtype float64`:** win values
  are `{0, 0.5, 1}` × binary card-removal — exact in float32. Not a correctness
  issue for these runs; the notebook’s “exact (float64)” wording refers to the
  CFR compute dtype.
- **Cross-version Kaggle resume:** checkpoints only help within the same
  `/kaggle/working` (or if prior output is remounted as input). A brand-new
  Save & Run All commit starts empty unless you attach prior output.
- **Solver math (chance denom / average strategy / joint mass):** still correct;
  covered by `tests/test_bugfixes.py`.

---

## What to run on Kaggle after this PR

1. **Content pack (primary):** `kaggle_content_yield.ipynb` — Save & Run All.
   Stops after pack build+verify; raise half stays skipped.
2. **Validation (optional):** `kaggle_fullrange_validation.ipynb` — Save & Run All
   (~4–6 h float64). One board crash no longer kills the commit.
