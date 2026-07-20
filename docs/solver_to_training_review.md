# Solver → Recommendation → Training Review

Review date: 2026-07-20. Scope: how full-street CFR outputs become the
**recommended action**, **grades**, **explanations**, and **trainer feedback**.

---

## Pipeline (production path)

```
BatchedCFR / BatchedGPUCFR  (averaged strategy S + action utilities U)
        │
        ▼
_flop_decisions_from_cap / preferred_action()
        │  preferred = max EV; EV ties → higher frequency
        ▼
content_yield.extract_records()
        │  reach gate, mixed flag, hand/board tags, explain()
        ▼
content_pack.build_pack()
        │  EV-regret grades, dedup, signed SQLite
        ▼
trainer (pack_server / demo build_trainer)
```

Legacy flop-only path (`runner` → `export.build_questions` → `trainer.db`) still
exists with different action names and coarser grades; do not mix contracts.

---

## Contract: what “recommended” means

| Layer | Rule |
|-------|------|
| Preferred / recommended | **Max EV** under the averaged strategy |
| EV ties | Higher **solver frequency** |
| Grades | EV regret as % of pot vs best EV |
| Mixed lesson | **Every** legal action within 0.5% pot of best |
| Frequency bars | Teaching detail — **not** the recommendation selector |

Integrity checks reject `preferred` that is not max-EV before a pack is signed.

---

## Pass 1 findings (fixed)

### H1. Trainer called tied-best picks a “costly leak” — **FIXED**
- Verdict now follows the precomputed grade; mixed + good gets soft copy.

### H2. `mixed` flag dropped before the trainer UI — **FIXED**
- `_to_q` now passes `mixed` through.

### M1. EV ties preferred the first action key — **FIXED**
- `preferred_action()` tie-breaks by frequency.

### M2. Priority impact used a global pot — **FIXED**
- Uses per-record `pot_bb`.

### L1. Legacy “GTO pick” label — **FIXED**
- Relabeled “recommended”.

---

## Pass 2 findings (fixed)

### H3. OOP facing a bet labeled “You act first” — **FIXED**
- **Where:** `demo/build_trainer._to_q` `acts_first`
- **Bug:** `acts_first` was true for OOP `*_vs_bet` (already checked, now facing
  a bet). Beginner badge and Learning copy said “you act first” while the
  situation said facing a bet (~25% of shipped demo vs-bet spots).
- **Fix:** `acts_first = node.endswith("_first")` only.

### H4. Top-2 indifference mislabeled 3-action spots as mixed — **FIXED**
- **Where:** `content_yield.extract_records` `mixed`; raise-demo pack rows
- **Bug:** `mixed = (best − 2nd) < 0.5% pot` ignored a dominated third action.
  Example: raise≈call but fold −18% pot still got headline “either is
  acceptable” / reason `mixed`.
- **Fix:** `mixed` requires **all** legal actions within `CLEAR_SEP_PCT`.
  Headline: “All actions are close…”. `refresh_pack_lessons` repaired shipped
  packs in place (raise demo: 2 spots → `raise_bluff` / `bluff_catch`).

### M3. Pack trainer always said “flop” — **FIXED**
- **Where:** `pack_server._situation`, `pack_index.html`
- **Bug:** Turn/river packs (via `POKERTRAINER_PACK`) still said “on the flop”
  and showed a “Flop” board caption.
- **Fix:** Derive street from board length; thread into situation + UI caption.

### M4. Launch fullrange pack missing `pot_bb` / roles — **FIXED**
- **Where:** `flop_pack_v1_fullrange.db`
- **Fix:** `refresh_pack_lessons` backfills `pot_bb`, `oop_pos`, `ip_pos` from
  pack config and re-signs. CLI: `--refresh-lessons`.

---

## Pass 3 findings (fixed)

### H5. Displayed mix frequencies summed to 99 or 101 — **FIXED**
- **Where:** `explanations.freq_pct_ints`, demo `_to_q`, pack_server,
  legacy trainer bars
- **Bug:** Independent `round(100 * p)` broke the mix (e.g. 69+30+2).
- **Fix:** Largest-remainder ints that always sum to 100.

### H6. Learning/Pro called IP leads a “c-bet” — **FIXED**
- **Where:** demo trainer `situation()`
- **Bug:** `btn_vs_bet` / IP facing-bet spots said “facing a 66% c-bet”; those
  are opponent leads. Pack server already distinguished lead vs checked-then-bet.
- **Fix:** OOP → “you checked and face a bet”; IP → “they led into you”
  (`is_oop` on each question).

### M5. Mixed detail still said “both” for 3-action spots — **FIXED**
- Detail lists all actions; UI soft copy says “any,” not “either.”

### M6. Adaptive unlock taught “blinds act first” — **FIXED**
- Wrong for blended SB-vs-BB where BB is IP. Unlock/glossary now teach IP/OOP.

### M7. Legacy trainer called non-recommended `good` “GTO-optimal” — **FIXED**
- `trainer/server.grade_answer` mirrors pack soft-close wording.

### L2. `classify_reason` mapped raise-on-first_action to check reasons — **FIXED**
- Only `check` enters the check branch; other actions fall through safely.

---

## Design notes (not bugs)

1. **★ vs frequency bars** — Preferred ≠ modal frequency in ~6% of full-range
   rows. Bars sort by freq; ★ marks max EV.
2. **Explanations are labels** — Heuristic matrix over category × action ×
   texture; does not invent strategy.
3. **Two pipelines** — Full-street pack vs flop-only export still differ.
4. **Indifference thresholds** — 0.25% (`best`) / 0.5% (`mixed`) / 1.0%
   (cross-solver compare) remain intentionally layered.

---

## Tests / tooling

- `tests/test_solver_to_training.py` — preferred, export tie-break, priority pot,
  all-action mixed refresh, pot/role backfill, `acts_first`, freq ints, mixed
  detail, classify fallthrough.
- `tests/test_explanations.py` — 3-action near-top-2 is not `mixed`.
- `python -m pokertrainer.content_pack --refresh-lessons <pack.db>`
