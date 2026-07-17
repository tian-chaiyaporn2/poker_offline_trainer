# Decision Memo — Open-Source Poker Solver POC

**Question (PRD §2):** Can MIT/Apache-licensed software generate sufficiently
reliable strategy data for a commercial poker-training app?

**Recommendation: PROCEED — build the product on our own permissively-licensed
solver.** The flop-only POC is complete and passes; the multi-street extension
needed for full GTO is validated and made practical by a GPU (still zero
copyleft). No stop condition was triggered.

---

## What we found

### 1. The flop-only POC works and passes (PRD §7 acceptance)

- **Our own MIT solver** (vectorised CFR+) — no third-party solver licence to
  audit, the cleanest possible answer to the core question.
- All **12 boards** solved; **120 validated training questions**; a fully
  **offline local trainer** (no cloud/accounts/ads).
- **Accuracy:** 100% preferred-action agreement vs an *independent* solver;
  max root-EV difference **0.004% of pot** (target < 1%). Equities match an
  independent Monte-Carlo check.
- **Stability:** exploitability converges to **< 0.004% of pot** and is
  **bit-for-bit deterministic**; suit-isomorphic spots match.
- **Runtime:** the whole 12-board library generates in **~2.4 minutes** on a
  laptop. **32 automated tests** pass.
- **Licensing risk: effectively zero** — MIT solver + NumPy (BSD) + Python
  stdlib. TexasSolver is never bundled, imported, or required.

**Limitation:** the POC models a **flop-only** abstraction (no turn/river
betting), so strategies are exact for that game but more bet-heavy than full GTO.
Disclosed in-app and in every question.

### 2. Multi-street (full GTO) — validated, and made practical by GPU

We extended to full flop→turn→river and ran three spikes (details in
`docs/multistreet_spike.md`). Each corrected the last:

| Approach | Result |
|---|---|
| Naive pure-Python enumeration | Correct but impractical (~2.5 h/board at 15 hands) |
| Compile hot loop to C | **Dead end** — only ~4× (memory-bandwidth-bound; NumPy already uses BLAS) |
| **Batched public-tree CFR** (NumPy) | Right architecture; **validated exact** vs the naive oracle (~1e-15); ~10–30× faster, but CPU-bound at hours/board for full ranges |
| **Batched CFR + GPU** (CuPy) | **Validated exact** (GPU EV = CPU EV); **~15× over CPU**; measured on a free Colab T4 below |

**Measured GPU (Colab T4, full ~250-hand ranges, river solve):**

- **~22 minutes per board** (600 CFR iterations); a **12-board multi-street
  library in ~4.4 hours** on a *free* GPU, unattended.
- GPU is permissive (CuPy is BSD/MIT-style); the solver stays our MIT code.

---

## What this means for the product

- **Immediately practical:** a POC-scale multi-street library (tens of boards) is
  a **~4-hour overnight run on a free GPU** — no paid or copyleft software.
- **Commercial scale (hundreds–thousands of flops)** is not blocked; it needs a
  bigger GPU and/or known optimisations, each with a clear runway:
  - **Datacenter GPU** (A100/H100 ≈ 5–15× a T4) → **~2–4 min/board** → a
    1,000-board library in ~1–3 days on one GPU, trivially parallel across GPUs.
  - **Materialise-free showdown** — removes the `E`-gather copy we identified as
    the n=250 bottleneck (float32 confirmed the solve is memory-, not
    compute-bound there).
  - **Suit isomorphism** (fewer contexts on suited boards) and **iteration
    tuning** (600 was assumed; CFR+ often converges in 200–300 → ~halves runtime).

---

## Recommendation (maps to PRD §12)

**Proceed** — permissive software is sufficient and practical. Concretely:

1. **Ship the flop-only trainer as v1** now (complete, fast, disclosed).
2. **Build the multi-street library on batched CFR + GPU** (our MIT solver +
   CuPy). Architecture is proven and validated exact; the only work is the
   optimisation runway above, sized to the target library.
3. **Clear the TexasSolver licence question** and wire the (already-stubbed)
   adapter to spot-check multi-street output against the industry reference —
   valuable once multi-street exists, not before.

## Risks / open items

- Multi-street **convergence iteration count** not yet measured precisely (assume
  600; likely fewer) — quick to nail down and it directly scales runtime/cost.
- Commercial-scale runtime **depends on GPU tier**; budget an A100-class GPU (or
  the showdown optimisation) before committing to a large library.
- **Preflop ranges** are hand-crafted (out of POC scope); a product needs solved
  ranges.
- **Provenance:** solver and evaluator are our own MIT code — fully known.

**Bottom line:** the permissive approach is not the constraint. Flop-only ships
today; full multi-street is a proven, permissive, GPU-accelerated build with a
clear cost/runtime runway.
