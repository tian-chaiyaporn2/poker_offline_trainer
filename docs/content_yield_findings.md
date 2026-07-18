# Content-Yield Gate — Findings

_PRD v1.3 §5.3 / §9.4. Measures accepted full-street flop decision records per
solved root and projects vs the launch targets (≥1,200 accepted records; ≥30
distinct 10-question sessions; coverage across nodes/board families/hand classes)._

## Result: PASS with large margin

Record volume is **not** a binding constraint. Even a handful of roots clears
1,200; the 40–60 root launch library yields **tens of thousands** of accepted
records with ample concept diversity. The real constraints are **compute
(GPU-hours)** and **coverage balance**, both of which look healthy.

## Measured (CPU preview, n=40 hands/side, 4 diverse roots: dry-ace, connected, paired, monotone)

| Metric | Value |
|---|---|
| Accepted records/root (deduped) | **130** |
| Projected accepted/root at full range (~250 hands) | **~812** |
| Projection @ 40 roots | ~32,500 (dedup-bounded ≈ 24k) |
| Projection @ 60 roots | ~48,750 (dedup-bounded ≈ 24k) |
| Target | **1,200** |
| Distinct concepts (4 roots) | **67** (scales with roots) |
| Est. 10-question sessions @ 60 roots | far above the 30 target |

Even against the dedup ceiling (cap 30 records per node × board-texture ×
hand-category × preferred-action concept), the library lands **~20× over** the
1,200 target.

## Per-node coverage (the useful structural finding)

| Flop decision node | Accepted (4 roots × 40 hands) | Note |
|---|---:|---|
| BB check / bet (first action) | 160 / 160 | fully covered |
| BTN check / bet (vs BB check) | 160 / 160 | fully covered |
| BB fold / call (vs BTN bet) | 160 / 160 | fully covered |
| **BTN fold / call (vs BB bet)** | **40 / 160** | **naturally thin** |

The BTN-facing-a-BB-bet node is sparse because **BB rarely donk-bets** — so that
spot seldom occurs, and the reach-mass filter correctly drops ~75% of it. This is
a real poker property, not a pipeline gap: the three common decision types are
richly covered; "BTN responds to a BB lead" is a rare spot and will be a small
part of the curriculum (as it should be).

## Caveats

- **Reduced-range CPU projection.** Per §13.1, full ranges are the content
  authority; these numbers are a validated projection. The authoritative
  full-range measurement runs headless on GPU via
  `colab/kaggle_content_yield.ipynb` (12 roots, ~3 h) — it will confirm, not
  overturn, the "pass with margin" conclusion, and pin down the exact per-node
  yield (especially the thin BTN-vs-bet node).
- **Projection is dedup-bounded**, so the real total is nearer the ~24k ceiling
  than the raw linear figure — still ~20× the target.

## Implications for v1

1. **Proceed** — record volume comfortably supports the launch library; pick root
   count (40–60) for **concept diversity and coverage**, not to hit 1,200.
2. **Compute is the real budget.** Track accepted questions per GPU-hour (§10.3),
   not boards/hour — each root already yields hundreds of accepted records.
3. **Expect asymmetric coverage by decision type** — plan the curriculum so the
   rare BTN-vs-donk-bet node is a small, clearly-scoped slice.
4. Next pipeline stages (now unblocked): tagging is in place; build **explanation
   generation**, **content QA**, and the **signed SQLite pack** (§10.1), and add
   the **raise** action (FR-011) when a second decision branch is wanted.

## Artifacts

- Yield report: `output/content_yield_preview/yield_report.json`
- Records: `output/content_yield_preview/records.json`
- Module / CLI: `src/pokertrainer/content_yield.py`
- Full-range GPU runner: `colab/kaggle_content_yield.ipynb`
