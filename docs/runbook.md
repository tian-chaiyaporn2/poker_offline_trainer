# Run book — solve → pack → serve

End-to-end operational guide for producing and serving the trainer's content.
Everything is MIT-licensed and runs from this repo; the only external dependency
for the *full-range* solve is a GPU (Kaggle T4). TexasSolver is a **dev-only**
reference and is never bundled.

## 0. Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q            # sanity: full suite should be green
```

## 1. Pipeline at a glance

```
ranges + boards ─▶ solve (CFR+, full street) ─▶ decision records ─▶ signed pack ─▶ trainer
                     validate_flop              content_yield        content_pack    pack_server
                                                                                     (+ demo page)
foundations (board reading / pot odds / hand reading / equity) ──▶ foundations.json
```

Two content streams:
- **Flop decisions** — solver-generated, graded, explained (the core).
- **Foundations** — deterministic fundamentals questions (no solver needed).

## 2. Solve + validate a spot (dev / spot-check)

```bash
# Cross-check the three solvers agree and match Monte-Carlo equity:
python -m pytest tests/test_batched.py tests/test_bugfixes.py -q
# Flop-only vs full-street comparison on a board subset:
python -m pokertrainer.validate_flop --solver cpu --n 40 --iters 300 --max-boards 1
```

## 3. Generate flop-decision records (the content-yield gate)

**Reduced range (CPU, minutes) — pipeline check only:**
```bash
python -m pokertrainer.content_yield --solver cpu --n 40 --iters 300 \
    --roots 0,1,2 --out output/content_yield_preview
# add --raise-x 3 to include the raise action (FR-011)
```

**Full range (GPU, ~5–7 h) — the authoritative data:** run on Kaggle, not locally
(full range is ~10 days on CPU). Open `colab/kaggle_content_yield.ipynb`:
1. Settings → **GPU T4 x1**, **Internet On**.
2. Run the **smoke cell** first (~20–35 min); confirm it prints a GPU name and
   `'note': 'full-range'`.
3. **Save & Run All (Commit)**, close the tab, download `flop_pack_v1_fullrange.db`
   (+ `records_v1_fullrange.json`) from the version's Output tab.

Checkpointing: each board is solved → validated → written to `boards/board_XX.json`;
`records.json`/`yield_report.json` refresh after every board. A single crashing
board is logged and **skipped**, and the run continues. Non-finite (NaN/inf)
records are dropped per board and never enter the pack. **Caveat:** Kaggle's GPU
commit limit is ~**9 h**, and a commit that *exceeds* it may be killed without
saving `/kaggle/working` — so keep the run inside the limit (the main run is
~5–7 h; smoke + raise cells auto-skip in a commit). Checkpoints let an
*interactive* re-run resume; a fresh commit starts over (`/kaggle/working` is
wiped between commits).

Resume / rebuild from partial checkpoints without re-solving:
```bash
python -m pokertrainer.content_yield --aggregate-only --out output/content_yield
```

**Full-range RAISE pass (FR-011) — gives every facing-a-bet spot a Raise option.**
The launch pack solves `*_vs_bet` nodes as Fold/Call only (raising blows up the tree).
To add raises, re-solve BTN-vs-BB with `--raise-x 3`. The raise tree is bigger, so it
runs in **3 parts** (~4–6 h each) via `colab/kaggle_content_raise.ipynb`: set `PART`
to `'A'` (boards 0–5), `'B'` (6–11), `'C'` (12–16), Save & Run All each time, and
download `records_raise_<PART>.json`. Merge the three board-wise, then build/sign as
in §4 (records → `v1_fullrange`, this becomes the new full-range pack — a superset:
same Check/Bet spots + Fold/Call/Raise on the vs-bet nodes). First-to-act and
checked-to spots stay Check/Bet (no bet to raise). Local one-board check:
```bash
python -m pokertrainer.content_yield --solver cpu --n 8 --iters 25 --roots 0 \
    --raise-x 3 --scenario btn_vs_bb_srp --out /tmp/raise_smoke   # vs_bet -> fold/call/raise
```

*Same pass for the other two scenarios that are still Fold/Call on vs-bet:*
- **SB-vs-BB raise pass** — `colab/kaggle_content_raise_sb.ipynb` (identical 2-part flow,
  `--scenario sb_vs_bb_srp`). Download `records_raise_sb_<PART>.json`, merge, build/sign a
  raise-enabled `sb_vs_bb` pack → replaces `flop_pack_sb_vs_bb.db`.
- **Turn/river raise pass** — `colab/kaggle_content_turnriver_raise.ipynb` (one commit,
  ~1–3 h — the raise tree is bigger): `demo/gen_turn_river.py … --raise-x 3`. Produces
  `flop_pack_turnriver_fullrange.db` (now a raise superset), drops straight in.

**Full-range TURN / RIVER pass (range only).** `colab/kaggle_content_turnriver.ipynb` (GPU,
one commit ~20–40 min) runs `demo/gen_turn_river.py --solver gpu --n 400 --iters 300
--version turnriver_fullrange` over the 16 curated runouts (Check/Bet + Fold/Call, no raise
— use the raise notebook above to add raises). **Shipped:** the full-range pack is live.
Local reduced-range default:
```bash
python demo/gen_turn_river.py            # cpu, N=90 -> turnriver_demo
python demo/gen_turn_river.py --raise-x 3 --n 6 --iters 15 --version tr_raise_smoke  # local raise smoke
```

## 4. Build + sign + verify the pack

```bash
python -m pokertrainer.content_pack \
    --records output/content_yield/records.json --version v1_fullrange \
    --out output/packs
# -> output/packs/flop_pack_v1_fullrange.db (+ .db.gz), HMAC-SHA256 signed
python -c "from pokertrainer.content_pack import verify_pack; \
    print(verify_pack('output/packs/flop_pack_v1_fullrange.db'))"
# expect: hash_ok=True, signature_ok=True
```

Signing key: defaults to a built-in **dev** key. To sign with a real key, export
`POKERTRAINER_SIGNING_KEY` **before building**; the server then needs the same key
(or it refuses to serve — the error says so explicitly).

## 5. Foundations content

```bash
python -m pokertrainer.foundations --out output/foundations
# -> output/foundations/questions.json  (board reading, pot odds, hand reading, equity)
```
Deterministic — same output every run, so it can go into a signed pack. Answers are
computed from the same primitives as the solver pipeline (evaluator, board texture,
pot-odds arithmetic, MC equity). Trainer integration (serving these alongside flop
decisions) is the next step.

## 5b. Prioritize what to solve/teach next

```bash
python -m pokertrainer.priority --records output/content_yield/records.json --out output/priority
```
Ranks every accepted record by **frequency × impact × intuition-gap** (all read
from the solve: P(board texture) × reach_mass, EV spread, and how non-obvious the
GTO play is). Outputs `priority_report.json` with two backlogs: `lesson_backlog`
(spot-types by total teaching value — what to surface first) and `solve_backlog`
(board textures by real-world frequency vs current coverage — **what to solve
next**; high-frequency + uncovered = top priority). Use it to steer board/runout
selection with data instead of guesses.

## 6. Serve / preview

```bash
python trainer/pack_server.py            # http://127.0.0.1:8000 — picks the newest pack
```
Shareable review page (no server), regenerated from a signed pack:
```bash
python demo/build_preview.py             # -> demo/content_preview.html + index.html
```
`index.html` is the GitHub Pages landing page; push to `main` and Pages rebuilds.

## 7. Invariants & gotchas (hard-won)

- **Chance-node denominator = `deck − 4`** (both players' hole cards). Verified
  against MC equity; `−2` understates multi-street EV ~4–9%. All three solvers
  must agree (oracle == batched == GPU, to machine precision).
- **Root EV is normalized by compatible-matchup mass** (`w_o @ B @ w_i`) so it's
  conditioned on a valid, non-colliding matchup. Per-action EVs/grades are separate
  and unaffected.
- **Reported EV uses the iteration-averaged strategy**, not the last iterate
  (CFR+ last-iterate oscillates).
- **GPU must stay float32.** A single un-typed accumulator once promoted the whole
  run to float64 (~32× slower on a T4). `tests/test_gpu_float32.py` guards this.
- **Grades come from EV-regret thresholds** stored in the pack; the deterministic
  "Green filter" is what makes the app safe without a human reviewer.

## 8. Tests

```bash
python -m pytest -q                      # full suite (~2–3 min; solver comparisons are slow)
```
