# Solver Design & Modeled Abstraction

_Our own MIT solver (PRD Deliverable 3). This document states exactly what game
is solved, so that "abstraction differences" are never mistaken for "solver
disagreement" (PRD §5, Step 5)._

## Algorithm

Vectorised **Discounted CFR** (Brown & Sandholm 2019-style discounting) over a
public betting tree, with the private-hand dimension held as NumPy arrays. CFR
is a deterministic, exact-in-expectation self-play method: no Monte-Carlo
sampling, so repeated runs with the same seed and iteration count are
bit-for-bit identical — directly satisfying the PRD's stability requirement.

## Modeled game (POC abstraction v1: flop-only, realized equity)

- Heads-up, BTN (in position, IP) vs BB (out of position, OOP).
- Single-raised pot, 100bb start → flop pot `5.5bb`, `97.5bb` behind. No rake.
- **OOP acts first on the flop** (Hold'em rule).
- Flop betting only. When the flop betting closes with a call or check/check,
  the remaining equity is **realized by full turn+river runout enumeration** as
  a single showdown — there is no turn or river betting.

### Betting tree (finite by construction)

```
OOP to act (pot P0):
├─ check
│   └─ IP to act:
│      ├─ check            → showdown
│      ├─ bet 33% (small)  → OOP: fold | call→showdown
│      └─ bet 75% (large)  → OOP: fold | call→showdown
├─ bet 33% (small)         → IP: fold | call→showdown
└─ bet 75% (large)         → IP: fold | call→showdown
```

No raising in v1 (documented simplification to cap the tree). The three permitted
acting-player actions — **check / small bet / large bet** — map exactly onto the
PRD's flop action set.

### Why flop-only for the POC

A true multi-street solve (turn + river betting) is the natural extension and is
where a TexasSolver comparison is most meaningful. It is also far more expensive
(47×46 runout betting subtrees). For a *feasibility* POC whose priorities are
**stability, reproducibility, a complete pipeline, and honest compute numbers**,
the flop-only / realized-equity model is:

- **Exact** (full runout enumeration, no sampling),
- **Fast** (≈6 decision infosets; sub-second CFR per board),
- **Stable** (deterministic; iteration-increase monotonically converges),
- and produces sensible flop check/bet strategies.

Its known deviation from full GTO: with no future betting, value betting is
under-rewarded, so strategies are more check-weighted and less polarised than a
full multi-street solver. This is disclosed in every exported question's
metadata and in the feasibility report. The PRD explicitly lists "claims of
complete GTO accuracy" as out of scope (§8).

## Utility convention

The pre-flop pot `P0` is the contested prize (pre-flop chips are sunk). A
player's subgame utility = chips collected − chips invested **on the flop**:

- **Showdown** (both invested `x`): `E[util_OOP] = equity·(P0 + oop_inv + ip_inv) − oop_inv`, where `equity ∈ [0,1]` includes half-credit for ties.
- **OOP folds:** `util_OOP = −oop_inv`, `util_IP = P0 + oop_inv`.
- **IP folds:** `util_IP = −ip_inv`, `util_OOP = P0 + ip_inv`.

EVs are reported in **big blinds** and as **% of the flop pot** for comparison.

## Card removal

Two combos that share a card never co-occur; the showdown module builds a
compatibility mask `B[i,j]`. Equity is averaged only over runouts that block
neither combo. Suit-isomorphic scenarios must yield equivalent results
(PRD §6) — enforced by the normaliser and a dedicated test.

## Outputs (per scenario, PRD §5 Step 3)

- Action frequencies by hand (per infoset).
- EV of each action, per hand.
- Overall range action frequencies.
- Root EV.
- Iteration / convergence info (exploitability curve).
- Runtime and peak memory.

## Extension path (post-POC)

`solver/cfr.py` isolates the terminal-utility provider behind an interface so the
same CFR core can drive a multi-street tree later without rewriting the search.
