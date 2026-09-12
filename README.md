# Poker Offline Trainer — Open-Source Solver + Offline App

An MIT-licensed poker trainer: our own Discounted CFR+ generates strategy data,
signed packs turn it into graded questions, and a **fully offline** web app
serves them. No paid or copyleft runtime dependencies.

See [`PRD.md`](PRD.md) for product requirements and
[`docs/runbook.md`](docs/runbook.md) for the solve → pack → serve pipeline.

## What's in the app today

The GitHub Pages landing page (`index.html`) is a self-contained trainer:

- **Play a hand** (default) — flop→turn→river continuation on a 48-board GPU library.
- **Exploit** — same linked hands vs 5 opponent archetypes (station / nit / maniac / LAG / reg).
- **Isolated drills** — flop, turn, river, plus preflop and **foundations** (board reading, pot odds, hand reading, equity).
- **Position packs** — BTN-vs-BB, SB-vs-BB, BTN-vs-SB, CO/UTG/HJ-vs-BB, and a low-SPR 3-bet pot.
- Language ladder (Beginner / Learning / Pro / Adaptive) and optional BYOK coach.

The original flop-only POC (12 boards, 120 questions) is still in `output/trainer.db`
and `trainer/server.py`. The live product is the full-street pack + demo trainer.

## Held (needs a Kaggle GPU commit)

These are implemented in code or notebooks, but the **production library** is
not regenerated yet:

- **Raise pass (FR-011)** — solver supports `--raise-x 3`; only a 240-record demo pack is shipped. Full-range re-solves of BTN-vs-BB, SB-vs-BB, and turn/river wait on GPU.
- **Continuation Phase 2** — branching / sampled villain (raises, check-raises). Phase 0/1 (solved main line) is shipped.
- **Next solve families** — 4-bet pots, extra stack depths, multiple bet sizes.

## Layout

```
PRD.md                     Product requirements + current roadmap status
docs/
  runbook.md               Solve → pack → serve
  licenses.md              Dependency & licence inventory
  scenario_format.md       Canonical scenario spec
  solver_design.md         The modeled game / abstraction
  preflop_and_exploit_plan.md   Preflop + exploit (both shipped; raise held)
  continuation_mode_scope.md    Play-a-hand (Phase 0/1 shipped; Phase 2 held)
src/pokertrainer/          Solver, packs, explanations, foundations, preflop
demo/build_trainer.py      Builds the self-contained app → index.html
colab/                     Kaggle/Colab GPU notebooks
output/packs/              Signed SQLite content packs
tests/                     pytest suite
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline trainer (uses committed packs — no regenerate required)
python trainer/server.py          # http://127.0.0.1:8000  (legacy flop-only)
python trainer/pack_server.py     # newest signed pack
# or open index.html / demo/trainer_demo.html (no server)

# Rebuild the self-contained app after pack or UI changes
PYTHONPATH=src python demo/build_trainer.py

# Tests
python -m pytest
```

GPU library generation (full-range / raise / bulk continuation) is documented
in [`docs/runbook.md`](docs/runbook.md) and `colab/`.

## Licence

MIT — see [`LICENSE`](LICENSE). Runtime dependencies are permissive
(MIT / BSD / PSF); see [`docs/licenses.md`](docs/licenses.md).
