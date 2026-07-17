# Multi-Street Feasibility Spike — Findings

_Follow-up to the flop-only POC. Question: can we practically extend our own
permissive solver to full multi-street (turn + river betting)?_
Code: `src/pokertrainer/solver/multistreet.py`. Measurements on the ace-high
rainbow flop `As7h2d`, single machine, Python 3.10 + NumPy.

## TL;DR

The multi-street **method works** — the solver is correct and converges. But a
**pure-Python full-enumeration implementation is not practical**: cost explodes
~155× per street added, and a single river solve is already ~2.5 hours per board
at a tiny 15-hand range. Production multi-street needs a **compiled inner loop**
(and, for some boards, suit isomorphism). This is exactly why tools like
TexasSolver are written in C++.

## Correctness (not a divergence bug)

The flop+turn game converges to a stable equilibrium with the corrected
utility bookkeeping:

| Iterations | 50 | 200 | 600 | 1500 |
|-----------|----|----|----|------|
| OOP EV (% pot) | 18.25 | 18.09 | 18.20 | 17.99 |

Stable to within ~0.2 pp — the CFR math generalises correctly to multiple
streets. (Flop-only remains bit-for-bit reproducible; the same determinism
applies here.)

## The compute wall

Per-iteration cost and tree size grow ~155× **per street added**:

| Model | ms / iteration | Infosets | Showdown boards | Notes |
|-------|---------------:|---------:|----------------:|-------|
| Flop only (POC)          | **0.7** | 4 | 1 | ships today |
| Flop + turn              | **109** | 200 | 49 | converges in ~1 min |
| Flop + turn + **river**  | **18,390** | 4,904 | 1,176 | see below |

_(All at 15 combos per side, `As7h2d`.)_

A river solve does **~64,000 showdown evaluations per iteration** across ~4,900
infosets — every one a small Python-level NumPy call, so the runtime is
**overhead-bound, not arithmetic-bound** (15 combos vs 250 barely moves it).

**Extrapolation:** CFR+ needs ~500+ iterations to converge. At 18.4 s/iter that
is **~2.5 hours per board at just 15 combos**. Real ranges (~250 combos) add the
showdown's ~n² term on top; a full 12-board river library in pure Python would
take **days to weeks** — impractical (a PRD §10 stop condition for *this
implementation*, not for the approach).

## Why — and the levers that fix it

The bottleneck is Python/NumPy call overhead over an enormous node count, not the
algorithm. The levers, in order of impact:

1. **Compiled inner loop** (biggest). The per-node work is tiny; the overhead is
   the killer. A C++/Rust/Cython/Numba (or GPU via JAX/CuPy) showdown + regret
   kernel removes it — plausibly 50–1000×. This is how TexasSolver hits
   seconds-to-minutes on the *same* enumeration.
2. **Suit isomorphism on runouts** (board-dependent). Collapses equivalent
   turn/river suit patterns. Note it helps **monotone/two-tone** flops a lot but
   gives **~zero** reduction on a rainbow flop like `As7h2d`, whose three
   distinct suits already pin all symmetry. So it is a real but uneven lever.
3. **Coarser abstraction** (fewer bet sizes, bucketed runouts) — trades exactness
   for speed; use sparingly given the PRD's stability priority.

A compiled kernel (≈100×) plus isomorphism on suited boards (≈2–6×) would bring
18 s/iter into the tens-of-milliseconds range — i.e. a river solve in
seconds-to-minutes per board, matching TexasSolver-class performance while
staying fully in-house and permissively licensed.

## What this means for the product decision

- The **permissive approach is not the problem** — our own MIT solver is correct,
  convergent, deterministic, and license-clean. The pure-Python *implementation*
  is the problem.
- Three viable paths:
  1. **Ship flop-only v1** now (works, fast, strategically simplified — disclosed
     in-app), and treat multi-street as a later milestone.
  2. **Build an optimised permissive multi-street solver** (compiled kernel +
     isomorphism). Keeps zero license risk; meaningful but well-scoped eng work.
  3. **Use an existing compiled solver** as a generation backend — fastest to
     real multi-street data, but re-introduces the exact licensing question the
     PRD set out to avoid (TexasSolver is copyleft; verify before relying on it).

## Recommended next step

Before committing to build path (2), run one more **small spike: a compiled
(Numba or Rust) river-showdown + CFR kernel** on a single board, to confirm the
~100× is real and that a permissive multi-street solver can hit practical
runtimes. That is the concrete go/no-go for an in-house optimised solver vs.
falling back to flop-only or a licensed backend.
