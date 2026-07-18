# Findings — Flop-Only Training Usefulness Validation

_Executes `docs/flop_training_validation_plan.md`. One document: Reports A–D
(§12), acceptance criteria (§9), hypotheses, stop-condition review (§14), and the
config/compute caveats that bound the result._

---

## 0. Headline & recommendation

**Do not ship unfiltered flop-only betting frequencies. Recommended path:
Option B (fundamentals-only beta) now, with a *filtered-Green strategy subset* as
a fast-follow — gated on one automated check: the full-range GPU re-run.**

The flop-only model **systematically over-bets** (+14.4 pp vs full-street), and
the over-betting is concentrated in **made/value hands** that the full-street
model prefers to **check** (slow-play, pot control, protecting a checking range).
Checks, clearly dominated actions, and non-betting fundamentals are safe; the
prescriptive **bet** recommendations are where it fails.

Against the plan's Option-A bar the model does **not** qualify (agreement 77% <
90%; Green 58% < 70%), but median full-street regret is **0.0% of pot** and a
**138-question Green subset** is individually safe.

**Safety without a human reviewer.** The safety mechanism is the *deterministic
regret filter* (the Green classification), which needs no human. A Green question
is, by definition, one whose recommended action matches the full-street model
within ≤0.25% pot and is stable — its core claim is machine-verified. The
conservative, reviewer-free posture is therefore: **ship only Green as scored
questions; ship everything else as non-prescriptive fundamentals (never score
Amber); keep explanations action-focused** ("checking is best here") rather than
narrating multi-street logic the flop-only model can't justify. This removes the
two things a human was there to catch — misleading "several actions OK" framing
and over-claimed reasoning — by simply not shipping those as graded strategy.

---

## 1. What was compared (and the caveats that bound it)

- **Flop-only model** = `BatchedCFR(streets=1)`; **full-street reference** =
  `BatchedCFR(streets=3)` — the *same solver and betting tree*, generated from one
  frozen config (`docs/validation_config.md`). The **only** difference is whether
  turn/river betting follows the flop. This is the assumption-matched comparison
  §5 requires (no range/tree/stack/pot/bet-size mismatch possible).
- **Preferred action** = highest-EV action. **Full-street regret** = full-street
  EV(best) − full-street EV(flop-only's preferred action).
- **Strategies read from the iteration-averaged profile** (not CFR+ last-iterate,
  which oscillates and produced spurious instability). Convergence verified:
  non-indifferent preferred actions stable from ~200 iters; run used 400 iters
  with a mid-snapshot at 200 → **0.4% unstable** (§7.5 satisfied).

### Caveats (must be resolved before final sign-off — see §7)

1. **Single bet size (66%)**, required for a matched tree. This validates the
   *abstraction*, not the exact 3-bet-size (33/75%) shipped question set. The
   literal 120 shipped questions need a 3-size full-street solver (scoped
   follow-up).
2. **Reduced ranges (20 hands/side), CPU first pass.** Full ~250-hand ranges at
   full-street need the GPU. The **direction** of every finding (over-betting of
   made hands; safe checks/fundamentals) is robust and matches known solver
   behaviour, but exact percentages will shift at full range. The plan's
   Definition of Done (§15) requires the full-range 30–50 board GPU run.

---

## Report A — Strategy fidelity

_12 boards, 240 hand-decisions (210 non-indifferent)._

| Metric | Result | Option-A target |
|---|---|---|
| Preferred-action agreement (all) | **74.6%** | — |
| Preferred-action agreement (non-indifferent) | **77.1%** | ≥ 90% ❌ |
| Full-street regret — median | **0.00% pot** | ≤ 0.25% ✅ |
| Full-street regret — mean / p90 / max | 0.63% / 1.12% / **11.35%** | — |
| Betting-frequency bias (flop-only − full-street) | **+14.4 pp** (0.294 vs 0.150) | understood ✅ |
| Unstable (non-indiff preferred flips 200→400) | **0.4%** | low ✅ |

The regret distribution is **heavily right-skewed**: most decisions are fine
(median 0), but a tail (Red 13%) reaches ~11% of pot — enough to teach a costly
habit if shipped unfiltered.

**By hand category** — disagreement is concentrated exactly where H2 predicted:

| category | n | agree (non-indiff) | Green % |
|---|---:|---:|---:|
| top_pair | 22 | **50.0%** | **22.7%** |
| strong_made | 67 | **67.3%** | **34.3%** |
| weak_pair | 73 | 85.3% | 74.0% |
| air | 78 | 83.1% | 71.8% |

The counter-intuitive but correct signal: **made/value hands are the *least*
safe**, because the full-street model checks many of them (slow-play, pot
control, checking-range protection) while flop-only bets them. Air and weak pairs
agree far more often.

**By board category** — safest to least safe:

| category | agree (non-indiff) | Green % |
|---|---:|---:|
| monotone | 87.5% | **85.0%** |
| paired / low | 82.5% / 76.9% | 62.5% |
| high_card | 77.0% | 54.0% |
| dry | 68.8% | 50.0% |
| connected | 75.6% | **38.3%** |

Monotone/paired/low boards (both models check-heavy) are safest; connected and
dry boards (more marginal betting spots) are least safe.

**Largest disagreements — all the same shape (flop-only bets, full-street checks):**

| board | hand | category | regret %pot |
|---|---|---|---:|
| 6h4d2c | 9s8h | air | 11.35 |
| Qh8h3h | JdJc | weak_pair | 11.03 |
| As7h2d | AhTc | top_pair | 10.98 |
| KsKd6h | 9s8h | weak_pair | 9.93 |
| As7h2d | AcJh | top_pair | 9.02 |

Every one of the top disagreements is **flop-only "bet" → full-street "check"** —
the over-betting bias, made concrete.

---

## Report B — Content viability

| Class | Count | % | Meaning |
|---|---:|---:|---|
| **Green** (publishable, scored) | 138 | **57.5%** | same action, regret ≤ 0.25% pot, stable |
| **Amber** (teach as "several OK") | 70 | 29.2% | regret 0.25–1.0%, or near-indifferent, or freq differs |
| **Red** (do not ship as strategy) | 32 | 13.3% | regret > 1.0%, clear disagreement, or (0.4%) unstable |

- **Publishable now (Green): 138 questions** across all 12 boards — a real
  curriculum, but below the 70% bar for declaring the *model* shippable.
- **Concepts safe to teach:** checking decisions, clearly dominated actions,
  monotone/paired/low-board play, and all **non-betting fundamentals** (equity,
  hand strength, range interaction, board texture).
- **Concepts requiring full-street content:** value-betting made hands,
  slow-play/pot-control, thin bets on connected/dry boards, and bet-sizing.

---

## Report C — Compute feasibility

_From the GPU benchmark (`docs/validation_config.md` §6) plus this CPU run._

- **Full-street, full ranges (T4 GPU):** ~2.2 s/iter → **~22 min/board** at 600
  iters; **~4.4 h** for a 12-board library.
- **Convergence:** non-indifferent preferred actions stable from **~200 iters**
  (0 changes 200→700). 600 iters is conservative → real cost likely ~⅓ lower.
- **CPU vs GPU:** GPU ~10–15× faster; GPU EV == CPU EV (~1e-14). No material
  disagreement (a §14 stop condition — cleared).
- **This validation run (CPU, reduced n=20):** ~9.4 min/board for *both* models,
  12 boards in ~1.9 h. Memory: a few MB.
- **Library estimate (T4, 600 iters; halve if 200 suffice):** 100 boards ≈ 37 h,
  500 ≈ 183 h, 1000 ≈ 367 h on one T4 → an A100/H100 (~5–15×) does 1000 boards in
  ~1–3 days; board-level parallelism scales linearly (boards independent).

---

## Report D — Recommendation

**Primary: Option B — ship a fundamentals-only beta now.** With 77% agreement and
58% Green, the model as a whole is not safe for prescriptive betting instruction;
the +14.4 pp over-bet would teach a systematic leak. Equity, hand-strength,
range-interaction, and board-texture content is correct and stable and can ship.

**Fast-follow: filtered Option A** — ship the **138 Green questions** as scored
strategy content, gated on a single automated check: the **full-range GPU re-run**
confirming the Green subset holds at full resolution (this pass used 20-hand
ranges). No human reviewer required — the Green filter is the deterministic safety
gate (see §0). The app must not claim flop-only frequencies are full-street GTO
(§9 Option B rule), and only Green ships as *scored* content.

**Do not** ship unfiltered flop-only betting frequencies.

---

## Hypotheses (§3)

- **H1 — robust actions exist: CONFIRMED.** Median regret 0.0%; obvious
  checks/dominated actions agree; air & weak pairs 83–85% agreement.
- **H2 — disagreement is concentrated: CONFIRMED**, and localised more precisely
  than expected: in **made/value hands the full-street model checks**
  (slow-play/pot-control), plus draws/marginal spots. All top disagreements are
  flop-only-bet → full-street-check.
- **H3 — a safe publishable subset exists: PARTIALLY.** A deterministic
  regret-filter yields a 57.5% Green subset — real, but below the 70% bar; the
  filter works, the model just isn't clean enough to pass Option A wholesale.
- **H4 — fundamentals remain useful: CONFIRMED.** Non-prescriptive content is
  unaffected by the abstraction and is the safe immediate product.

## Acceptance criteria (§9)

**Option A (filtered strategy) — NOT met:** agreement 77.1% (< 90%), Green 57.5%
(< 70%). Median regret 0.0% (✅) is the one primary criterion met.
**Option B (fundamentals-only) — MET:** equity/classification/range/board metrics
are correct and stable; the app can ship these while avoiding GTO claims.

## Stop-condition review (§14) — none triggered

- CPU vs GPU differ materially: **NO** (EV equal to ~1e-14).
- Increasing iterations flips clear preferred actions repeatedly: **NO** (stable
  from ~200 iters; 0.4% unstable at 400).
- Suit-isomorphic states disagree: **NO** (evaluator/showdown suit-iso unit-tested).
- Full-street EVs fail internal consistency: **NO** (prize identity holds up to
  the expected card-removal normalisation).
- Benchmark ≠ full flop-root tree: **NO** (clarified in §6 — it is a flop root).
- Ranges/action trees differ between models: **NO** (same code, one frozen config).

## Definition of Done (§15) — status

**Met:** harness + both models under one frozen config; full-street regret +
safety class for every compared hand; largest disagreements investigated (all the
same over-betting shape); GPU runtime/memory reproducible; CPU/GPU agreement.
**Outstanding for final sign-off:** the full-range **GPU** re-run (this pass: 12
boards, 20-hand ranges) — runnable now via `colab/poker_fullrange_validation.ipynb`
(the harness runs on the GPU solver with `--solver gpu`). The plan's human-review
gate (§10) is **dropped** by product decision; safety instead rests on the
deterministic Green filter (see §0), which is why the reviewer-free posture ships
only Green as scored content. This pass establishes the method — converged,
stable, CPU/GPU-exact — and gives a decision-ready direction.

---

## Artifacts

- Per-hand dataset (§11 schema): `output/validation/flop_validation.csv` (240 rows)
- Machine-readable aggregate: `output/validation/summary.json`
- Config freeze + benchmark clarification: `docs/validation_config.md`
- Harness (re-runnable at full scale on GPU): `src/pokertrainer/validate_flop.py`
