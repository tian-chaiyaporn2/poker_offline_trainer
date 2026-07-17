# Canonical Scenario Format

_Deliverable 2 (PRD §9), Step 2 of the required workflow (PRD §5)._

One machine-readable format describes each solve. Both the production solver
(ours) and any reference solver (TexasSolver) must receive **equivalent inputs
derived from this single specification** so that formatting never masquerades as
strategy disagreement.

## Card & suit notation (canonical)

- Ranks: `2 3 4 5 6 7 8 9 T J Q K A` (uppercase `T` for ten).
- Suits: `s h d c` (spades, hearts, diamonds, clubs), always lowercase.
- A card is `<rank><suit>`, e.g. `Ah`, `Td`, `2c`.
- Canonical suit ordering for isomorphism/normalisation: `s > h > d > c`.
- Hands are two cards, higher card first by (rank, suit): e.g. `AhKh`, `7d7c`.

## Scenario JSON schema

```jsonc
{
  "id": "srp_btn_bb_100bb_flop_As7h2d",       // unique, stable
  "game": "nlhe",
  "format": "heads_up",
  "spot": "single_raised_pot",
  "positions": { "ip": "BTN", "oop": "BB" },   // in-position / out-of-position
  "stacks_bb": 100,                             // starting stacks (both), big blinds
  "rake": 0,

  "board": ["As", "7h", "2d"],                  // 3 flop cards, canonical notation

  "pot_bb": 5.5,                                // pot at start of flop (BTN 2.5 open, BB call)
  "effective_stack_bb": 97.5,                   // stack behind at start of flop

  "ranges": {
    // rank-pair grid notation with weights in [0,1]; expands to combos
    "BTN": { "notation": "range_v1", "combos": { "AA": 1.0, "AKs": 1.0, "...": 1.0 } },
    "BB":  { "notation": "range_v1", "combos": { "...": 1.0 } }
  },

  "acting_player": "BB",                         // who moves first on the flop (OOP acts first)

  "actions": {
    // Exact bet sizes MUST be identical across both solvers (PRD §4).
    "bet_sizes_pct_pot": { "small": 33, "large": 75 },
    "allowed": ["check", "bet_small", "bet_large", "call", "fold", "raise", "all_in"],
    "raise_rule": "single_raise_then_allin"      // caps tree depth for the POC
  },

  "tree": {
    "streets": ["flop", "turn", "river"],        // full street solve for correct flop EVs
    "turn_bet_sizes_pct_pot": { "large": 75 },
    "river_bet_sizes_pct_pot": { "large": 75 }
  },

  "solver": {
    "algorithm": "discounted_cfr",
    "iterations": 2000,
    "convergence_target_exploitability_pct_pot": 1.0,
    "seed": 12345
  }
}
```

## Field notes

- **`pot_bb` / `effective_stack_bb`** derive from a 2.5bb button open, big-blind
  call, 100bb start: pot = 2.5 + 2.5 + 0.5(dead?) — for the POC we use a clean
  `pot_bb = 5.5`, `effective_stack_bb = 97.5`. Documented so both solvers match.
- **`acting_player`** is OOP (BB) first to act on the flop, per Hold'em rules.
- **`bet_sizes_pct_pot`** — the three permitted flop actions map to
  `check`, `bet_small` (33% pot), `bet_large` (75% pot). Facing a bet, the
  responder may `fold`, `call`, or `raise` (single raise then all-in) to keep the
  tree finite.
- **Isomorphism**: two scenarios that differ only by a suit permutation must
  produce equivalent results (PRD §6). The normaliser canonicalises suits before
  comparison.

## Equivalence contract for the reference solver

When exporting to TexasSolver input, the adapter must reproduce, exactly:
hand notation, suit ordering, action names, bet sizes (in % pot), pot/stack in
the same units, and identical ranges. Any field that cannot be reproduced is
logged as a documented limitation (PRD §5, Step 4).
