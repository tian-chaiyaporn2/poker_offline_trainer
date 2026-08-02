# Scope — "Play the hand through": continuation mode across streets

**Goal.** Replace (or complement) the current *isolated-spot* drills with a mode where a
**single hand plays out street by street**: the hero's two cards persist, the board grows one
card at a time (flop → turn → river), villain acts between streets, and each street is a
decision that follows from the last. Teaches planning and how a new card changes a hand's
value — the skill isolated spots can't convey.

---

## The core change: isolated spots → linked trajectories

Today the trainer samples **independent, unconditioned** decisions:
- Flop packs (`content_yield` + `BatchedCFR`) and the turn/river pack (`demo/gen_turn_river.py`)
  are separate pools. The turn/river generator solves **curated representative runouts**
  (brick / flush-completer / board-pairer), each on a hand-picked board — *not* the
  continuation of any specific flop decision (`gen_turn_river.py` docstring: "unconditioned
  later-street toy demo").
- A session (`buildOrder()` in `demo/build_trainer.py`) picks N unrelated spots; each `q` is a
  standalone `(board, hand, node)`.

Continuation mode needs **trajectories**: an ordered sequence of steps sharing a `hand_id`,
same hero cards, growing board, each street's correct play solved *consistent with the line so
far* (including villain's between-street action).

**The engine already exists.** `src/pokertrainer/solver/multistreet.py` (`MultiStreetSpike`)
is a full flop→turn→river CFR: chance nodes deal the turn/river, tables keyed by
`(street, board_key, node)`, exact runout enumeration. It was built to prove multi-street
solving is practical. What's missing is (a) an **extractor that walks one line** down the
solved tree into ordered steps, and (b) a **"play a hand" session flow** in the trainer.

---

## Phased plan

### Phase 0 — MVP: scripted single-line continuation
The cheapest version that delivers the feeling. One deterministic line per hand.
- **Villain rule:** deterministic — "villain calls" after each hero action (no raising). This
  fixes the board runout so the hand advances on a single path.
- **Content:** for a curated set of (flop, hero-hand) starting points, solve with
  `MultiStreetSpike`, walk the *villain-calls* line, and emit ordered steps:
  `flop decision → (villain calls, deal turn) → turn decision → (villain calls, deal river) →
  river decision`. Each step is a normal decision record + a `hand_id` + `step_index` + the
  dealt card + villain's action label.
- **Pack:** reuse the signed `flop_decision` schema; add `hand_id`, `step_index`,
  `villain_action` (or fold them into `scenario`/`detail` to avoid a schema/signing change —
  preferred, mirrors the A5 approach).
- **Trainer:** a "hand session" that iterates a hand's steps: hero cards persist, board grows,
  a between-steps beat shows "Villain calls" and deals the next card (reuse the existing
  deal animation + the street-changing background rooms), then the next decision.
- **Effort:** ~3–5 days. Solver exists; work is the line-walker extractor + session flow.

### Phase 1 — Real solved trajectories, curated boards, signed pack
- Broaden to a solid set of instructive runouts per flop (brick / scare card / board-pair),
  each a proper solved line; validate exploitability→0 on the multi-street solve.
- Proper signed pack + build report; wire `load_*` like the other packs.
- **Effort:** ~1 week.

### Phase 2 — Branching villain (optional, ties into B/exploit)
- Villain sometimes raises or check-raises; sample villain's action from solved frequencies so
  lines vary, and let the hero face different runouts on replay. This is where continuation
  and **phase B (exploit vs archetypes)** naturally merge — an archetype's tendencies drive
  the between-street action, and you show the deviation *across a whole hand*.
- **Effort:** significant; do only after Phase 0/1 prove the format.

---

## Trainer session-flow changes (`demo/build_trainer.py`)
- `buildOrder()` currently fills `order` with independent question indices. Add a mode that
  fills it with **hands** (each a list of step-questions sharing `hand_id`), or keep `order`
  flat but tag consecutive entries with the same `hand_id` and advance the board between them.
- `renderHand()` / `renderQuestion()` already redraw board + hero per spot; extend to (a) keep
  hero cards fixed within a hand, (b) reveal one new board card per step, (c) show a short
  "Villain calls" interstitial before the next decision.
- The **stratified sampler** (3 preflop / 4 flop / 1 turn / 2 river) is replaced within a hand
  by natural street progression; at the session level you'd pick M full hands instead.

---

## Side benefits
- **Dissolves the flop-heavy imbalance** and the **"turn = 1 spot"** gap: turns and rivers are
  generated as continuations of flop hands, not a separate sparse pool.
- Uses assets we already built: street-changing background rooms, deal-in animation.
- More engaging — a hand has a narrative arc.

## Key decisions to make up front
1. **Villain model for MVP:** deterministic "calls" (simplest, recommended) vs sampled.
2. **Replace vs complement:** is continuation the *default* session, or a separate mode
   alongside the isolated drills? (Recommend: a mode, keep drills for targeted practice.)
3. **Schema:** fold `hand_id`/`step_index`/`villain_action` into `detail`/`scenario` (no
   signing change, A5-style) vs extend the table.
4. **Board curation:** which runouts per flop teach the most (scare cards, bricks, pairs).

## Out of scope (for now)
- Multi-street multiple bet sizes / all-in (separate, already scoped and parked).
- Full villain range-vs-range replay UI.

## Recommendation
Do this as its own phase, and **before phase B** — sequential play is more foundational than
exploit deviations, and B's contrast content is richer over a whole hand. Start with Phase 0
(scripted, villain-calls) to validate the format cheaply before investing in Phase 1's content.
