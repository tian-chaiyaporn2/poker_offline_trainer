# Poker Offline Trainer — Open-Source Solver POC

A proof of concept testing whether **free, permissively licensed software** can
generate stable poker strategy data, cross-check it against an independent
solver, turn it into training questions, and serve them in a **fully offline**
local trainer — with no paid or copyleft dependencies.

See [`PRD.md`](PRD.md) for the full product requirements and
[`docs/feasibility_report.md`](docs/feasibility_report.md) for the results and
recommendation.

## Result at a glance

- ✅ All **12 boards** solved; **120 training questions** generated.
- ✅ Our own **MIT CFR+ solver** converges to **< 0.004% of pot** exploitability
  and is **deterministic** (bit-for-bit stable across runs).
- ✅ **100%** preferred-action agreement vs an **independent** vanilla-CFR solver
  (max root-EV diff **0.004% of pot**); equities match an independent Monte-Carlo
  check.
- ✅ **Zero licensing risk**: solver + evaluator are our MIT code; only NumPy
  (BSD) + Python stdlib at runtime. TexasSolver is never bundled or required.
- ⚠️ Scope limit: models a **flop-only** abstraction (no turn/river betting).
  Recommendation: **Proceed with limitations** — extend to multi-street next.

## The permissive-solver decision

Rather than depend on a third-party solver (the exact licensing risk the PRD
worries about), **the solver is our own MIT code** — a vectorised Discounted
CFR+. That makes the licensing story airtight and directly answers the core
question. TexasSolver stays a dev-only, out-of-process reference (its licence is
copyleft and unverified); the codebase enforces its isolation.

## Layout

```
PRD.md                     Product requirements
docs/
  licenses.md              Dependency & licence inventory (Deliverable 1)
  scenario_format.md       Canonical scenario spec (Deliverable 2)
  solver_design.md         The modeled game / abstraction
  feasibility_report.md    Final feasibility report (Deliverable 9)
src/pokertrainer/
  cards.py evaluator.py    Card model + fast 5–7 card evaluator (MIT, ours)
  ranges.py presets.py     Range expansion, 12 boards, HU SRP ranges
  scenario.py              Scenario loading + PRD §6 validation
  showdown.py mc_equity.py Exact equity (enumeration) + Monte-Carlo check
  solver/cfr.py            CFR+ solver (Deliverable 3)
  reference_solver.py      Independent vanilla-CFR reference
  normalize.py compare.py  Output normalization + comparison (Deliverables 5,6)
  benchmark_texassolver.py TexasSolver adapter stub (Deliverable 4, gated)
  runner.py export.py      Solve runner + training-question export (Deliverable 7)
  generate.py benchmark.py CLIs
trainer/                   Offline local web trainer (Deliverable 8)
tests/                     pytest suite (evaluator, equity, solver, pipeline)
output/                    Generated library + reports (committed)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate the training library (solves all 12 boards -> questions.json + trainer.db)
PYTHONPATH=src python -m pokertrainer.generate

# 2) Cross-check our CFR+ vs the independent reference solver
PYTHONPATH=src python -m pokertrainer.benchmark        # writes output/comparison_report.md

# 3) Run the offline trainer, then open http://127.0.0.1:8000
python trainer/server.py

# 4) Tests
python -m pytest
```

The generated library is committed under `output/`, so the trainer runs
out-of-the-box without regenerating.

## Licence

MIT — see [`LICENSE`](LICENSE). All runtime dependencies are permissive
(MIT / BSD / PSF); see [`docs/licenses.md`](docs/licenses.md).
