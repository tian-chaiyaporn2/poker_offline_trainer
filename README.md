# Hold'em Trainer — offline, solver-backed poker training

A poker trainer whose every grade and explanation comes from **our own MIT CFR+ solves**,
packaged into integrity-checked content packs and served from **one self-contained page**
that works fully offline. The same page is the GitHub Pages site and the web layer of the
iOS/Android app.

**Live:** https://tian-chaiyaporn2.github.io/poker_offline_trainer/

## What you can train

- **Play a hand** (default): five full hands, flop → turn → river, against an opponent playing
  the solved line (48 texture-diverse boards).
- **Street drills:** pre-flop, flop, turn, river. Flop spots cover seven matchups (BTN, CO, HJ,
  UTG, SB vs BB; BTN vs SB; a low-SPR 3-bet pot) with Fold / Call / Raise facing a bet.
  Turn/river drills come from solved hand lines, so ranges reflect the earlier action.
- **Exploit mode:** full hands against five leaky archetypes (station, nit, maniac, LAG, reg),
  graded on the best response rather than GTO.
- **Basics:** reading the board and your hand, pot odds, how often you win.
- **Explanations at four levels** (Beginner / Learning / Pro / Adaptive), a "similar hand,
  opposite play" contrast, and an optional **bring-your-own-key AI coach** that is handed the
  spot's exact solver numbers.

## How it's built

```
ranges + boards ─▶ CFR+ solve (GPU on Kaggle) ─▶ decision records ─▶ integrity-checked pack ─▶ trainer page ─▶ Pages / app
```

- **Solvers** (`src/pokertrainer/solver/`): batched CPU + GPU (CuPy, float32) CFR+, an
  independent multi-street oracle they're cross-checked against, a preflop solver, multiple bet
  sizes, raises and an all-in cap.
- **Content** (`src/pokertrainer/content_yield.py`, `content_pack.py`): convergence-gated
  records, explanations, HMAC-integrity-checked SQLite packs in `output/packs/`.
- **Trainer** (`demo/build_trainer.py` → `index.html`): embeds the packs, fonts and art into a
  single page with a strict CSP.
- **App** (`mobile/`): Capacitor shell; the coach uses native HTTP and the OS secure store.

See [`docs/runbook.md`](docs/runbook.md) for the end-to-end solve → pack → serve guide.
[`PRD.md`](PRD.md) and [`docs/feasibility_report.md`](docs/feasibility_report.md) record the
original proof of concept.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python demo/build_trainer.py   # rebuild index.html from the committed packs
open index.html                               # it runs from file:// — no server needed

python -m pytest -q                           # full suite (~10 min; solver cross-checks are slow)

cd mobile && npm install && npm run ios       # or: npm run android  (see runbook §6b)
```

Full-range solves run on a Kaggle GPU via the notebooks in `colab/`; the generated packs are
committed, so nothing needs re-solving to build the trainer.

## Licence

MIT — see [`LICENSE`](LICENSE). The solvers and evaluator are our own code; runtime
dependencies are permissive (MIT / BSD / PSF); see [`docs/licenses.md`](docs/licenses.md).
TexasSolver is a dev-only reference and is never bundled.
