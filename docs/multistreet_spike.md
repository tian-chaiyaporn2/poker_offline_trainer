# Multi-Street Feasibility Spike — Findings

_Follow-up to the flop-only POC. Question: can we practically extend our own
permissive solver to full multi-street (turn + river betting)?_
Code: `src/pokertrainer/solver/multistreet.py`. Measurements on the ace-high
rainbow flop `As7h2d`, single machine, Python 3.10 + NumPy.

## TL;DR

The multi-street **method works** — the solver is correct and converges. The
*naive* implementation is impractical (~2.5 h/board at 15 hands), but a
compiled-kernel spike showed the fix is **not** "rewrite in C" (that gave only
~4×, because NumPy already uses BLAS and the work is memory-bandwidth-bound). The
real bottleneck is **~63,000 tiny NumPy calls per iteration** from a naive
recursive tree; a **batched public-tree CFR rewrite (mostly in NumPy)** should
reach practical runtimes — ~1–3 min/board, a full library in under an hour —
while staying fully permissive with no new dependency. Details below.

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

## Compiled-kernel spike — result (this changed the recommendation)

We then compiled the river-showdown inner loop to a C shared library (clang
`-O3 -march=native`, called via ctypes) and benchmarked it against NumPy on the
full 1,176-board runout for `As7h2d` at 160×160 combos:

| Showdown pass (all 1,176 boards, 1 iteration) | Time |
|-----------------------------------------------|------|
| NumPy, naive per-board loop (both players)    | ~258 ms |
| NumPy, **batched** (`tensordot`, one big op)  | ~71 ms |
| **C kernel** (fused loops)                    | ~68 ms |

**The C kernel was only ~4× faster than naive NumPy, and ~equal to batched
NumPy.** Reason: NumPy already dispatches matvecs to optimised BLAS, and the
showdown is **memory-bandwidth-bound** (it streams the ~120 MB board tensor), so
compiled code can't pull far ahead. **"Just rewrite the kernel in C" is not the
100× lever we assumed.**

### The real diagnosis

The 18 s/iter of the naive solver is **not** showdown arithmetic (that floor is
~70–260 ms/iter for all boards). It is Python/dispatch overhead from doing the
showdown as **~63,000 tiny separate NumPy calls per iteration** — the recursive
tree re-solves each river board ~54× (once per betting line reaching it). The fix
is architectural: a **batched public-tree CFR** that carries reach vectors and
touches each board **once** per iteration, replacing tens of thousands of tiny
calls with a handful of large tensor ops.

## Revised recommendation

- The big lever is **batched vectorisation (a solver rewrite), largely in NumPy**
  — plausibly 50–250× (18 s/iter → ~0.1–0.3 s/iter), i.e. a river solve in
  ~1–3 min/board and a 12-board library in well under an hour. **No new
  dependency, still fully permissive.**
- A compiled kernel adds only ~4× on top and is **bandwidth-bound**, so it's a
  *secondary* optimisation, not the enabler. Reach for it (or GPU/`tensordot` on
  bigger batches) only after batching.
- **Suit isomorphism** still helps suited boards (and shrinks the 120–290 MB
  per-flop tensor), but not rainbow flops.

**Concrete next step:** rewrite the multi-street solver as a **batched
public-tree CFR** (vectorised over runouts), then measure a real river solve on
one board. That — not compiling — is the go/no-go for an in-house permissive
multi-street solver. Expected outcome: practical (minutes per board) in pure
NumPy.
