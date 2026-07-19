# PRD: Open-Source Poker Solver POC

## 1. Objective

Build a small proof of concept demonstrating that free, permissively licensed
software can:

1. Generate stable poker strategy data.
2. Produce results reasonably consistent with an independent solver.
3. Convert the results into usable training questions.
4. Run as a simple local trainer without paid infrastructure.

The POC is intended to test technical feasibility, not create a complete
product.

---

## 2. Core Question

> Can an MIT- or Apache-licensed solver generate sufficiently reliable strategy
> data for a commercial poker-training app?

---

## 3. Software Approach

### Production components

Use only dependencies whose licences are verified as commercially permissive,
preferably:

- MIT
- Apache-2.0
- BSD

Expected components:

- A permissively licensed solver candidate for strategy generation.
- A permissively licensed poker library for card validation and hand evaluation.
- SQLite or JSON for storing training questions.
- A basic local web interface.

### Independent reference

TexasSolver may be used as a separate comparison tool during development.

It must not be:

- Included in the distributable POC.
- Incorporated into the production codebase.
- Required for the trainer to operate.
- Used to generate the final production database without appropriate permission.

The team must verify whether internal commercial-project benchmarking is
permitted before relying on it.

---

## 4. POC Scope

### Poker configuration

The POC will cover only:

- No-limit Texas Hold'em
- Heads-up postflop
- Button versus big blind
- Single-raised pot
- 100-big-blind starting stacks
- No rake
- Flop decisions
- One acting player per question

Permitted actions:

- Check
- Small bet
- Large bet

Exact bet sizes must be identical across both solvers.

### Test scenarios

Create:

- 12 exact flop boards
- At least 8–10 private hands per board
- At least 100 playable training questions

Board selection should include:

- Dry high-card boards
- Connected boards
- Paired boards
- Two-suit boards
- Monotone boards
- Low-card boards

---

## 5. Required Workflow

### Step 1: Licence verification

Before development:

- Record the licence of every dependency.
- Preserve required copyright and licence notices.
- Reject dependencies whose commercial-use rights are unclear.
- Keep TexasSolver completely separate from the distributable project.

### Step 2: Canonical scenario format

Create one machine-readable format defining:

- Player ranges and combination weights
- Board cards
- Pot size
- Remaining stacks
- Acting player
- Permitted actions and bet sizes
- Raise rules
- Solver convergence settings

Both solvers must receive equivalent inputs derived from this specification.

### Step 3: Production solving

Run each scenario through the permissively licensed solver and export:

- Action frequencies by hand
- Expected value of each action
- Overall range action frequencies
- Root expected value
- Iteration or convergence information
- Runtime and memory usage

### Step 4: Reference solving

Run the same scenarios separately through TexasSolver, subject to licence
permission.

If TexasSolver cannot legally or technically be used, the POC may use another
independent reference, but this limitation must be documented.

### Step 5: Output normalization

Create a common comparison format so both solvers use the same:

- Hand notation
- Suit ordering
- Action names
- Bet-size representation
- Expected-value units
- Probability format

This step is required because formatting differences can otherwise appear to be
strategy disagreements.

### Step 6: Automated comparison

Generate a report showing:

- Root expected-value differences
- Action expected-value differences
- Preferred-action agreement
- Strategy-frequency differences
- Missing or invalid hands
- Illegal actions
- Probability totals
- Runtime and memory usage
- Results that change significantly between repeated runs

### Step 7: Training export

Convert accepted results into JSON or SQLite records containing:

- Situation description
- Board cards
- Player cards
- Available actions
- Action frequencies
- Action expected values
- Recommended action or acceptable action set
- Validation status

### Step 8: Local trainer

Build a simple local interface that:

1. Displays the cards, pot and situation.
2. Lets the user select an action.
3. Shows the strategy frequencies and expected values.
4. Explains whether the answer was good, acceptable or costly.
5. Records the result locally.
6. Loads another question.

No account, cloud system, advertising or mobile application is required.

---

## 6. Validation Rules

### Technical validation

The following must always pass:

- No duplicate or impossible cards.
- No illegal actions.
- Action probabilities total approximately 100%.
- Equivalent suit arrangements produce equivalent results.
- Repeated runs produce reasonably stable output.
- Increasing solver iterations improves or stabilizes the solution.

### Solver comparison

The comparison should prioritize expected-value agreement over exact frequency
matching.

Initial POC targets:

- Root expected-value difference below approximately 1% of the pot.
- At least 90% agreement in clearly non-indifferent decisions.
- No major disagreement where one solver considers an action strongly profitable
  and the other considers it strongly losing.
- Average aggregate frequency difference generally below five percentage points
  in non-indifferent situations.
- All significant disagreements are automatically flagged.

A decision is considered non-indifferent when the expected-value difference
between the best and second-best action exceeds a predefined threshold.

Frequency differences are not considered major failures when the actions have
nearly identical expected values.

---

## 7. POC Acceptance Criteria

The POC passes when:

- All 12 boards can be processed successfully.
- At least 100 valid questions are produced.
- The trainer runs locally without internet access.
- The generated strategies remain stable as solver iterations increase.
- Technical validation tests pass.
- Most strategically meaningful decisions agree with the independent reference.
- Major disagreements can be explained or isolated.
- No paid or restrictively licensed component is required in the distributable
  trainer.
- Compute requirements are practical for generating a larger library.

---

## 8. Out of Scope

The POC will not include:

- Three-to-six-player postflop solving
- Preflop solving
- Turn or river starting scenarios
- Multiple stack depths
- Rake
- Tournament structures
- Mobile packaging
- User accounts
- Payments
- Advertisements
- Adaptive learning
- AI-generated explanations
- Production graphics
- Claims of complete GTO accuracy

---

## 9. Deliverables

1. Dependency and licence inventory.
2. Canonical scenario specification.
3. Permissive-solver runner.
4. TexasSolver or alternative benchmark adapter.
5. Output-normalization script.
6. Automated comparison report.
7. At least 100 validated training records.
8. Simple local trainer.
9. Final feasibility report covering:
   - Accuracy
   - Stability
   - Runtime
   - Memory use
   - Important disagreements
   - Licensing risks
   - Recommendation

---

## 10. Stop Conditions

Stop or change direction if:

- The permissive solver produces unstable results after increasing iterations.
- Significant expected-value disagreements remain after confirming identical
  inputs.
- Equivalent suit configurations produce materially different results.
- Solver output cannot be interpreted reliably.
- Compute requirements make larger-scale generation impractical.
- The solver licence or source-code provenance cannot be verified.
- The POC depends on TexasSolver or another restrictive component to function.

---

## 11. Estimated Effort

Suggested team:

- One developer
- One knowledgeable poker reviewer for spot checks

Expected effort:

- Approximately two to four weeks
- Existing local hardware where possible
- No required paid software or cloud services

---

## 12. Final Decision

At completion, select one outcome:

**Proceed** — The permissive solver is sufficiently stable and accurate to
generate a larger training library.

**Proceed with limitations** — The solver is useful for specific configurations
but requires stronger validation, restricted scenarios or manual review.

**Replace the solver** — The current solver is unsuitable, but the scenario,
benchmarking and trainer pipeline remain reusable.

**Stop** — Available permissive software cannot produce sufficiently reliable or
practical results for the proposed product.

---

## 13. Content Strategy & Roadmap (post-POC direction)

The POC's core question is answered: permissively-licensed software **can**
generate reliable strategy data. The solver is cross-validated three ways to
machine precision and matches independent Monte-Carlo equity to <0.001 (see
`docs/runbook.md`, §7 invariants). This section records the direction for turning
that capability into a content library, so scope decisions are deliberate.

### 13.1 Full-street solving as the foundation

Content is generated from **full flop→turn→river solves** (betting on all three
streets), and the flop decisions are extracted from them so each recommendation
already accounts for future streets. A consequence worth stating explicitly: the
turn and river GTO strategies are **computed as a byproduct** of every solve — the
expensive compute is already spent, so reaching turn/river content is primarily an
*extraction* problem, not a *solve* problem.

### 13.2 Data-driven prioritization (do not cover everything)

The flop space is ~1,755 strategically-distinct boards and the turn/river space is
combinatorially larger; exhaustive coverage is neither feasible nor useful. Content
selection is therefore **prioritized by measured value**, not curated by hand. Each
candidate lesson is scored (`pokertrainer.priority`) on three axes read from the
solve output:

- **Frequency** — `P(board texture)` (exact combinatorics over all flops) ×
  `reach_mass` (how often the hand reaches the node). How often the spot occurs.
- **Impact** — best-vs-worst EV spread as a share of the pot. How costly the
  mistake is.
- **Intuition-gap** — how non-obvious the GTO play is (a trap or bluff-catch
  scores far above a value bet). Teaching value is highest where intuition fails.

The scorer emits two backlogs that drive planning:

- **Solve backlog** — board textures ranked by real-world frequency vs current
  coverage. **High-frequency + uncovered = solve next.**
- **Lesson backlog** — spot-types (node × texture × hand-category × reason) ranked
  by total teaching value. What to surface first.

### 13.3 Phased roadmap

1. **MVP (current)** — flop decisions for one scenario (BTN opens, BB calls,
   single-raised pot, 100 bb, 66% c-bet), on a prioritized board set. Launch gate:
   ≥1,200 accepted records with coverage across nodes, textures, hand categories,
   and reasons (measured by the content-yield report). **Status: met** — the
   full-range run produced 5,737 signed records; ~95.5% flop-texture coverage after
   the coverage boards (§13.2 solve backlog).
2. **Breadth first — additional positions (committed, next).** The highest-value
   expansion is *more scenarios*, not more depth on one: a player faces many
   position matchups daily, so covering them widens what the trainer teaches far
   more than adding an action to a single spot. The pipeline is scenario-
   parameterized (`--scenario`); each matchup supplies its own preflop ranges and
   reuses the board/priority machinery. **Next scenario: SB vs BB single-raised
   pot** (blind-vs-blind; note the OOP player is the pre-flop aggressor here,
   inverting the BTN-vs-BB range dynamic).
3. **Raise action (committed, depth pass).** fold/call/**raise** (FR-011) is
   important and *will* be solved — as a re-solve of each shipped scenario's boards
   with the raise action enabled (~3× cost, multi-commit). Sequenced after the
   first breadth expansion, then layered onto each scenario.
4. **Turn & river decisions** — extract from the *same* full-street solves via
   representative-runout sampling (brick / flush-completing / pairing / overcard),
   prioritized by the same scorer. No additional GPU budget beyond the flop runs.
5. **Further scope (compute-bounded)** — 3-bet/4-bet pots, stack depths, and
   multiple bet sizes. Each is a *new solve family* that multiplies compute and
   pushes GPU memory (near the T4 float32 limit at full range with raise), so these
   are sequenced deliberately, not bundled.

Prioritization: **breadth (positions) before depth (raise, turn/river)** for the
next investment, because it expands the product's real-world coverage most per
solve; raise and turn/river then deepen each shipped scenario. Items 2–5 move the
corresponding entries in §8 from "out of POC scope" to "roadmapped after
validation"; the single-scenario, flop-only §8 boundary describes the **MVP**, not
the product ceiling.

### 13.4 Foundations content (complementary stream)

Alongside solver-derived decisions, a deterministic **foundations** stream
(`pokertrainer.foundations`) generates fundamentals questions — board reading, pot
odds, hand reading, equity — computed from the same primitives (no invented
strategy). These require no solve and broaden the curriculum below the
spot-specific lessons.

### 13.5 Resourcing summary

The GPU cost is dominated by the flop full-street solves, which already exist.
Turn/river content and prioritization add **CPU-cheap extraction and analysis**,
not solve time. The genuine compute frontier is *scope broadening* (§13.3 item 4),
which is the decision to budget against — not board count or street depth within
the current scenario.
