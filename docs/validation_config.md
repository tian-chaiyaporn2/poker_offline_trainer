# Validation Configuration Freeze (plan §5) & Benchmark Clarification (§6)

This records the exact, matched configuration under which the flop-only and
full-street models are compared, and clarifies the GPU benchmark — both required
by the validation plan before results can be trusted.

## Matched configuration (both models generated from this)

The flop-only model is `BatchedCFR(streets=1)`; the full-street reference is
`BatchedCFR(streets=3)`. They are the **same code and the same betting tree**,
differing only in whether turn/river betting follows the flop. Every frozen field
below is therefore identical by construction (no possibility of a range,
action-tree, stack, pot, rake, or bet-size mismatch — the plan's §5/§14 hazard).

```yaml
scenario_id:        srp_btn_bb_100bb_flop_<board>
root_street:        flop
positions:          {ip: BTN, oop: BB}
ranges:             BB_SRP (OOP) vs BTN_SRP (IP)   # identical to both models
board:              <one of the 12 POC boards>
pot_bb:             5.5
effective_stack_bb: 97.5
acting_player:      BB (out of position, first to act)
legal_actions:      OOP {check, bet};  facing a bet {fold, call}
bet_sizes:          66% pot (single size per street)
raise_rules:        no raise (v1)
rake:               0
solver:             CFR+ (deterministic; last-iterate)
solver_version:     batched.py @ current commit
random_seed:        n/a (no sampling; fully deterministic)
```

### Deviations from the shipped POC — disclosed

- **Single bet size (66%)**, not the POC's check/small(33%)/large(75%). This is
  required to keep the flop tree identical between the two models (comparing the
  3-size POC questions against a 1-size full-street solve would be a bet-size
  mismatch and is invalid per §5). Consequence: this pass validates the
  **flop-only abstraction** (the effect of removing future streets), not the
  exact 3-size shipped question set. Validating the literal 120 shipped questions
  requires a 3-bet-size full-street solver (a scoped follow-up).
- **Reduced ranges (first pass, CPU).** Full ~250-hand ranges at full-street on
  CPU are hours/board; this first pass subsamples ranges for tractability. The
  plan's Definition of Done (§15) requires the full 30–50 board, full-range run,
  which needs the GPU path (below). This pass establishes the harness + direction.

## Benchmark clarification (plan §6)

The "~22 minutes per board" figure, stated precisely:

| Field | Value |
|---|---|
| Root street | **flop root** (full flop→turn→river public tree) |
| Turn states (chance) | 47 cards |
| River states (chance) | 46 cards (per turn) |
| Private hands | ~250 per side (full ranges) |
| Betting model | single bet size/street, no raise; OOP-first; check/bet, fold/call |
| River showdown boards | 1,176 unique (per flop) |
| CFR iterations | 600 (assumed; convergence measured separately — see Report C) |
| Convergence measure | EV stability / per-hand preferred-action stability |
| Numeric precision | float64 (float32 available; ~equal at n=250, memory-bound) |
| Hardware | free Colab **T4** GPU |
| Runtime | ~2.2 s/iter → ~22 min/board (600 iters); ~4.4 h for 12 boards |
| CPU vs GPU | GPU ~10–15× faster than CPU NumPy |
| Correctness | GPU EV == CPU EV (diff ~1e-14) |

End-to-end runtime (data transfer + export) is negligible vs the solve on the T4
(equity-cache build ≈ 2 s/board; export is milliseconds). This benchmark **does**
represent the intended full flop-root public tree — not a river-root subgame.
