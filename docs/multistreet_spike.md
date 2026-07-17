# Multi-Street Feasibility Spike — Findings

_Follow-up to the flop-only POC. Question: can we practically extend our own
permissive solver to full multi-street (turn + river betting)?_
Code: `src/pokertrainer/solver/multistreet.py`. Measurements on the ace-high
rainbow flop `As7h2d`, single machine, Python 3.10 + NumPy.

## TL;DR

The multi-street **method works and is now built + validated** (batched
public-tree CFR, exact to machine precision vs the oracle). Three things we
learned, each correcting the last:

1. Naive pure-Python enumeration is impractical (~2.5 h/board at 15 hands).
2. "Just rewrite the kernel in C" is a **dead end** — only ~4×, because the
   showdown is memory-bandwidth-bound and NumPy already uses BLAS.
3. The real win is **batched public-tree CFR** (touch each board once/iter). We
   built it: correct, ~10–30× faster — but the showdown einsum now dominates and
   scales ~n², so it's **hours/board at full ranges on CPU**, not minutes.

**Net:** an in-house, fully-permissive multi-street solver is viable and the
batched CFR is the right foundation, but a practical full-range library needs
**batched CFR + GPU** (`CuPy`/`JAX`, still zero copyleft) — the showdown is a
batched matmul, ideal for GPU. Details below.

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
multi-street solver.

---

## Batched public-tree CFR — built and measured (`solver/batched.py`)

We built it. At each chance node it assembles **all** runout child-contexts as
one batched tensor dimension and calls the next street once, so a river solve is
9 batched passes instead of ~19,458 tiny per-board solves.

### Correctness — validated exactly

Building this surfaced a real bug in the naive oracle: it keyed turn/river
infosets **by board only**, merging different flop betting lines that reach the
same board. The batched solver keys by **full betting history** (the pot
matters) — the game-theoretically correct partition. After fixing the oracle to
match, the two solvers agree to **machine precision**:

| | streets=1 | streets=2 |
|---|---|---|
| naive vs batched EV diff | 0.0 | ~1e-15 |

(Regression test: `tests/test_batched.py`.) The batched solver converges and is
deterministic. The residual "prize" gap (~3%) is the expected card-removal
normalisation — a compatible pair co-occurs on ~45 of each player's 47 valid
runouts (45/47 ≈ 0.957), not a leak.

### Speed — a solid win, but a new (arithmetic) wall

Batched **river** solve, per iteration, on `As7h2d`:

| Combos/side | ms / iter | vs naive (18,390 ms) |
|------------:|----------:|---------------------|
| 20  | ~1,310  | ~14× faster |
| 50  | ~2,345  | — |
| 100 | ~6,269  | — |

Batching removed the call-overhead (~14× at small ranges) — but the **showdown
einsum now dominates and scales ~n²**, so it extrapolates to **~33 s/iter at
~250-hand ranges** → still **hours per board** for a full-range river library on
CPU. Batching moved us from *days/board* to *hours/board*; it is **not yet
minutes/board** at production range sizes.

### Corrected conclusion (this is the real answer)

- **Architecture: solved.** Batched public-tree CFR is correct (validated to
  machine precision) and is the right design.
- **Pure-CPU NumPy: partial.** ~10–30× over naive — good enough for reduced
  ranges, turn-only spots, or small batches; **not** a full-range × 12-board
  river library (hours/board).
- **To reach production (minutes/board), the remaining lever is the showdown
  einsum**, which is a batched matrix multiply — ideal for:
  1. **GPU** (`CuPy`/`JAX`, both permissive) — plausibly 50–100× on exactly this
     op → ~0.3–0.7 s/iter at full ranges → minutes/board.
  2. **Suit isomorphism** (shrinks the board/context count on suited flops).
  3. **Range / card abstraction** (reduces n, quadratic payoff).

**Bottom line:** an in-house, fully-permissive multi-street solver is viable, and
the batched CFR is the correct foundation — but a practical full-range library
needs **batched CFR + GPU** (still zero copyleft), not CPU NumPy alone. The
compile-to-C idea was a dead end (bandwidth-bound, ~4×); the GPU path is the one
worth spiking next (needs GPU hardware, unavailable in this environment).
