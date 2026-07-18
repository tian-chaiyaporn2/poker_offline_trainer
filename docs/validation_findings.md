# Findings — Flop-Only Training Usefulness Validation

_Executes `docs/flop_training_validation_plan.md`. One document: Reports A–D
(§12), acceptance criteria (§9), hypotheses, stop-condition review (§14), and the
config/compute caveats that bound the result._

_Corrected model (see §1): "flop-only" = flop betting, then turn+river dealt as a
pure chance runout (equity realized), no future betting. An earlier pass used a
degenerate flop-only model (immediate 3-card showdown, draws = 0 equity); those
numbers were withdrawn. The numbers below are from the corrected model._

---

## 0. Headline & recommendation

**Do not ship flop-only betting recommendations. Recommended path: Option B —
fundamentals-only, plus a narrow filtered-Green subset used only for
"when to check" decisions.**

The flop-only model **over-bets massively**: it bets **60% of range vs the
full-street model's 15% (+45 pp)**, and on non-indifferent decisions its
preferred action agrees with full-street only **48.6% of the time** — no better
than a coin flip. Almost every disagreement is the same shape: **flop-only bets,
full-street checks.** Only **36.7%** of decisions are safe (Green), and those are
concentrated in **checking** decisions (air, dry boards).

The actionable split: **flop-only's *check* recommendations are largely safe**
(full-street checks a superset of what flop-only checks); **its *bet*
recommendations are largely wrong.** So the product can teach "when to check" and
non-betting fundamentals — not "when to bet" — from flop-only data.

**Safety without a human reviewer** rests on the deterministic Green filter: ship
only Green as scored questions, everything else as non-prescriptive fundamentals,
explanations action-focused. See §Report D.

---

## 1. What was compared (and the caveats that bound it)

- **Flop-only** = `BatchedCFR(streets=3, bet_streets=1)` — flop betting, then
  turn+river dealt as pure chance, equity realized at showdown. **Full-street** =
  `bet_streets=3` — betting on all three streets. Same solver, same tree, one
  frozen config (`docs/validation_config.md`); the only difference is turn/river
  betting. This matches the product's flop-only solver, which likewise over-bets
  (~78% of range in the POC) — the corrected model faithfully represents it.
- **Preferred action** = highest-EV action; **full-street regret** = full-street
  EV(best) − full-street EV(flop-only's preferred action).
- **Strategies read from the iteration-averaged profile** (last-iterate
  oscillates). Converged: **0.4% unstable**; non-indiff preferred stable ~200
  iters; run used 400 iters, mid-snapshot at 200.
- **Caveat — reduced ranges (20 hands/side), CPU.** Full ~250-hand ranges need
  the GPU (`colab/kaggle_fullrange_validation.ipynb`, now using the corrected
  model). The over-betting is **inherent to the no-future-street abstraction**
  (confirmed by the POC's own ~78% betting), so full range is very unlikely to
  rescue the betting recommendations; it will refine the exact Green%.
- **Caveat — single bet size (66%)**, for a matched tree. Validating the literal
  3-bet-size shipped questions needs a 3-size full-street solver (follow-up).

---

## Report A — Strategy fidelity

_12 boards, 240 hand-decisions (222 non-indifferent)._

| Metric | Result | Option-A target |
|---|---|---|
| Preferred-action agreement (all / non-indiff) | 47.5% / **48.6%** | ≥ 90% ❌ |
| Full-street regret — median | **0.05% pot** | ≤ 0.25% ✅ |
| Full-street regret — mean / p90 / max | 1.94% / 6.0% / **29.5%** | — |
| Betting-frequency bias (flop-only − full-street) | **+45.2 pp** (0.60 vs 0.15) | ❌ severe |
| Unstable | **0.4%** | low ✅ |

Median regret is near 0 (the many *check* agreements), but the mean/p90/max are
large — a heavy tail of costly **over-bets** (regret to ~29% of pot).

**By hand category** — made hands and pairs disagree most:

| category | n | agree (non-indiff) | Green % |
|---|---:|---:|---:|
| air | 78 | **71.8%** | **65.4%** |
| top_pair | 22 | 43.8% | 31.8% |
| weak_pair | 73 | 36.8% | 24.7% |
| strong_made | 67 | **34.5%** | **17.9%** |

Air is the *safest* (both models check it); made/value hands are least safe —
full-street checks them (slow-play, pot control, checking-range protection) while
flop-only bets.

**By board category** — dry safe, connected/draw-heavy unsafe:

| category | agree (non-indiff) | Green % |
|---|---:|---:|
| dry | **79.2%** | **60.0%** |
| rainbow / low | 55.9% / 53.8% | 45.0% / 43.8% |
| monotone / paired | 52.5% / 55.0% | 45.0% / 42.5% |
| two_suit | 33.9% | 21.2% |
| draw_heavy | 30.8% | 15.0% |
| connected | **13.3%** | **6.7%** |

Dry boards (little betting incentive) are safe; connected/two-suit/draw-heavy
boards — where flop-only over-bets draws and marginal hands hardest — are unsafe.

**Largest disagreements — all flop-only-bet → full-street-check:**

| board | hand | category | regret %pot |
|---|---|---|---:|
| Qh8h3h | 6d6c | weak_pair | 29.5 |
| Qh8h3h | 8c7c | weak_pair | 25.1 |
| 7s5s2s | Ah6h | air (draw) | 23.2 |
| 7s5s2s | 3h3c | weak_pair | 21.5 |
| 7s5s2s | AhTh | air (draw) | 20.1 |

---

## Report B — Content viability

| Class | Count | % | Meaning |
|---|---:|---:|---|
| **Green** (publishable, scored) | 88 | **36.7%** | same action, regret ≤ 0.25% pot, stable |
| **Amber** (non-scored) | 76 | 31.7% | regret 0.25–1.0%, near-indifferent, or freq differs |
| **Red** (do not ship as strategy) | 76 | 31.7% | regret > 1.0%, clear disagreement, or unstable |

- **Publishable now (Green): 88 questions**, but strategically **narrow** — skewed
  to *check* decisions on air and dry boards. Prescriptive **betting** content is
  mostly Amber/Red.
- **Safe to teach:** when to check, clearly dominated actions, dry/air play, and
  **all non-betting fundamentals** (equity, hand strength, range interaction,
  board texture).
- **Requires full-street content:** value-betting, slow-play/pot-control, thin
  bets, draws, connected/two-suit board play, and bet-sizing.

---

## Report C — Compute feasibility

_From the GPU benchmark (`docs/validation_config.md` §6) plus this CPU run._

- **Full-street, full ranges (T4 GPU):** ~2.2 s/iter → **~22 min/board**; the
  corrected flop-only adds a runout pass, so a full 12-board validation is
  **~4–6 h** headless on Kaggle (within the 30 h quota).
- **Convergence:** non-indiff preferred actions stable from ~200 iters; 0.4%
  unstable at 400.
- **CPU vs GPU:** GPU ~10–15× faster; GPU EV == CPU EV (~0, exact). No material
  disagreement (§14 stop condition cleared).
- **This CPU run:** ~11 min/board (both models, n=20), 12 boards in ~2.2 h.
- **Library estimate (T4, 600 iters):** 100 boards ≈ 37 h, 1000 ≈ 367 h on one
  T4; A100/H100 ≈ 5–15× → ~1–3 days for 1000 boards, parallel across GPUs.

---

## Report D — Recommendation

**Option B — fundamentals-only, no prescriptive betting.** Agreement is ~50%
(coin-flip) and Green is 37%; the model is not safe for teaching *when to bet*.
Ship equity, hand-strength, range-interaction, and board-texture content, which
are correct and stable, and must not claim flop-only frequencies are GTO.

**Narrow filtered add-on (deterministic, no human reviewer):** the **88 Green
questions** may ship as scored — but they are dominated by *check* decisions on
air/dry boards, so treat this as a "when to check" module, not a general betting
curriculum. Gate: the **full-range GPU re-run** (confirms Green% at full
resolution). Safety mechanism = the Green filter alone (a Green question's
recommended action is machine-verified within ≤0.25% pot of full-street);
**never score Amber; keep explanations action-focused.**

**Do not** ship flop-only *bet* recommendations — the +45 pp over-bet is a
systematic, costly leak.

---

## Hypotheses (§3)

- **H1 — robust actions exist: CONFIRMED (checks).** Air/dry checks agree
  strongly; median regret ~0.
- **H2 — disagreement is concentrated: CONFIRMED**, in made/value hands, weak
  pairs, draws, and connected/two-suit boards — all flop-only-bet → full-street-
  check. Broader than H2 anticipated (it's ~half of all decisions).
- **H3 — a safe publishable subset exists: WEAKLY.** 37% Green, skewed to check
  decisions; the filter works but leaves a narrow, betting-poor curriculum.
- **H4 — fundamentals remain useful: CONFIRMED**, and is now the primary product.

## Acceptance criteria (§9)

**Option A (filtered strategy) — NOT met:** agreement 48.6% (< 90%), Green 36.7%
(< 70%). **Option B (fundamentals-only) — MET.**

## Stop-condition review (§14) — none triggered

CPU vs GPU exact; preferred actions stable with iterations (0.4% unstable);
suit-isomorphism unit-tested; prize identity holds; benchmark is a flop root;
ranges/trees identical between models (one frozen config).

## Definition of Done (§15) — status

**Met:** both (corrected) models under one frozen config; regret + safety class
per hand; disagreements investigated (all the over-betting shape); GPU
runtime/memory reproducible; CPU/GPU exact. **Outstanding:** the full-range GPU
re-run (this pass: 12 boards, 20-hand ranges) — runnable now via
`colab/kaggle_fullrange_validation.ipynb`. The human-review gate (§10) is dropped
by product decision; safety rests on the deterministic Green filter.

## Artifacts

- Per-hand dataset (§11): `output/validation_corrected/flop_validation.csv` (240 rows)
- Aggregate: `output/validation_corrected/summary.json`
- Config freeze + benchmark: `docs/validation_config.md`
- Harness (runs at full scale on GPU): `src/pokertrainer/validate_flop.py`
