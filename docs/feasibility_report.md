# Feasibility Report (POC Deliverable 9)

_Answers the core question (PRD §2): can MIT/Apache-licensed software generate
sufficiently reliable strategy data for a commercial poker-training app?_

**Short answer: Yes, for the modeled abstraction — with limitations. Recommended
outcome: _Proceed with limitations_ (PRD §12).**

The permissive stack produced exact, stable, reproducible, internally-validated
GTO strategy data with **zero licensing risk**, and the whole pipeline
(scenario → solve → compare → export → offline trainer) works end to end. The
main limitation is scope: the POC solves a **flop-only, no-future-betting**
abstraction. Extending to full multi-street solving — and benchmarking against a
licence-cleared TexasSolver — is the clear next step before production.

---

## 1. What was built

Our own MIT solver (Discounted **CFR+**), a dependency-free 7-card evaluator, an
exact showdown-equity engine, an independent reference solver (vanilla CFR) for
cross-checking, a normalization + automated comparison layer, a 120-question
training library, and a fully offline local trainer. The only third-party
runtime dependency is **NumPy (BSD)**; everything else is our MIT code or the
Python standard library (see `docs/licenses.md`).

## 2. Accuracy

Measured by an independent algorithm (vanilla CFR) solving the identical game,
plus an independent Monte-Carlo check of the equity layer. Across all 12 boards
(`output/comparison_report.md`):

| Metric | Result | PRD target |
|--------|--------|-----------|
| Max root-EV difference (CFR+ vs reference) | **0.004% of pot** | < ~1% of pot |
| Min preferred-action agreement (non-indifferent) | **100%** | ≥ 90% |
| Max avg frequency difference (non-indifferent) | **0.54 pp** | < 5 pp |
| Major disagreements (strong-vs-strong) | **0** | 0 |
| Max Monte-Carlo vs enumerated equity difference | **0.006** | ~sampling error |

Two independently written solvers using **different algorithms** converge to the
same equilibrium, and the shared equity engine matches independent random
sampling. This is strong evidence the strategy computation is correct.

## 3. Stability

- **Exploitability** falls monotonically (modulo ~1e-5 numerical noise) to
  **< 0.004% of pot** on every board — far inside the 1%-of-pot convergence
  target — and stabilizes as iterations increase (PRD §6).
- **Determinism:** CFR+ takes no random samples; repeated runs are
  **bit-for-bit identical** (root-EV diff = 0). This trivially satisfies
  "repeated runs produce reasonably stable output."
- **Suit isomorphism:** suit-equivalent board/range configurations produce
  identical equities and root EVs (dedicated tests) — clearing that stop
  condition (PRD §10).

## 4. Runtime & memory

On existing local hardware (single machine, Python 3.10 + NumPy), no cloud:

- **Per board:** ~9 s exact equity precompute + ~2.3 s CFR+ solve (1500 iters).
- **Full library (12 boards, 120 questions):** **~142 s total.**
- **Peak memory:** a few MB per solve (ranges ≈ 230–280 combos each).

The equity precompute dominates and is embarrassingly parallel across boards and
runouts; a larger library is clearly practical (PRD §7). Note this is for the
flop-only model — multi-street solving is materially more expensive (see §7).

## 5. Important disagreements / caveats

- **No disagreements** survived between CFR+ and the reference on
  non-indifferent decisions (100% agreement). Frequency differences appeared
  only where actions were near-indifferent in EV — exactly the case the PRD says
  is not a failure.
- **Strategic realism caveat (the key finding):** because the model has no turn
  or river betting, betting is rewarded mainly through fold equity, so the
  solved BB strategies are **more bet-heavy and less check-heavy** than a real
  multi-street GTO solution would be. The strategies are correct *for the modeled
  game* (exploitability ≈ 0) but should not be read as full-street GTO. This is
  disclosed in every training question (`model_abstraction` field) and in the
  trainer UI.

## 6. Licensing risk

**Effectively zero for the distributable.** The "permissive solver" is our own
MIT code — there is no third-party solver licence to audit, which is the
cleanest possible answer to the core question. Runtime deps: NumPy (BSD) + Python
stdlib (PSF). TexasSolver is **not bundled, not imported, and not required**; the
adapter is licence-gated and refuses to run without explicit acknowledgement
(`benchmark_texassolver.py`). Open item: confirm TexasSolver's licence and
internal-benchmarking permission before using it as a reference (PRD §3).

## 7. Limitations to resolve before production

1. **Multi-street solving.** Add turn/river betting for realistic GTO. This is
   the single most important extension; the CFR core, equity engine, scenario
   format, comparison, export, and trainer are all reusable. **A spike has now
   been run** (`docs/multistreet_spike.md`): the multi-street CFR math is correct
   and converges, but a **pure-Python full-enumeration implementation is not
   practical** — cost grows ~155× per street (0.7 → 109 → 18,390 ms/iter for
   flop → +turn → +river), making a *naive* river solve ~2.5 hours per board even
   at 15 hands. A follow-up compiled-kernel spike found the fix is **not**
   primarily "rewrite in C" — a C showdown kernel was only ~4× faster than NumPy
   (the work is memory-bandwidth-bound and NumPy already uses BLAS). The real cost
   is ~63,000 tiny NumPy calls per iteration from the naive recursive tree. We
   then **built the batched public-tree CFR** (`solver/batched.py`) and validated
   it **exactly** against the naive oracle (EV diff ~1e-15). It is ~10–30× faster,
   but the showdown einsum now dominates and scales ~n², so it is **hours/board at
   full ranges on CPU** — good for reduced ranges, not yet a full-range library.
   The remaining lever is **GPU** for the showdown (a batched matmul; `CuPy`/`JAX`,
   still permissive, plausibly 50–100× → minutes/board). Net: an in-house,
   fully-permissive multi-street solver is viable via **batched CFR + GPU**; the
   architecture is proven, the CPU-only version is the bottleneck. See
   `docs/multistreet_spike.md`.
2. **TexasSolver benchmark.** Once (1) exists and the licence is cleared, run the
   real cross-solver comparison the PRD envisions. The adapter and normalization
   format are already in place.
3. **Preflop ranges.** The POC uses condensed hand-crafted SRP ranges; a product
   needs solved/curated preflop ranges (out of POC scope, PRD §8).
4. **Raising.** v1 caps the tree at no-raise; add raise lines with the tree.

## 8. Stop-condition review (PRD §10)

None triggered: results are stable under more iterations, no significant
EV disagreements remained after confirming identical inputs, suit-equivalent
configs matched, output is interpretable, compute is practical, solver provenance
is fully known (our own MIT code), and nothing depends on TexasSolver.

## 9. Recommendation — **Proceed with limitations**

Permissive software is clearly sufficient to generate stable, reliable,
license-clean strategy data and to drive an offline trainer. Before scaling to a
commercial library, extend the solver to multiple streets and validate the
richer strategies against a licence-cleared independent reference. The scenario
spec, solver core, comparison harness, export, and trainer are all built to carry
forward unchanged.
