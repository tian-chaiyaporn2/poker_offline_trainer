# Dependency & Licence Inventory

_Deliverable 1 (PRD §9). Updated as dependencies are added. Every entry must be
verified as commercially permissive (MIT / Apache-2.0 / BSD) before use._

## Guiding rules (PRD §5, Step 1)

- Record the licence of every dependency.
- Preserve required copyright and licence notices (see `reference/NOTICES/`).
- Reject dependencies whose commercial-use rights are unclear.
- Keep TexasSolver **completely separate** from the distributable project.

## Design choice: minimise the dependency surface

To make the licensing story airtight, the distributable POC is built almost
entirely from the Python standard library plus our own MIT code. In particular:

- **The solver is our own code** (Discounted CFR). This is the cleanest possible
  answer to the core question — there is no third-party solver licence to audit,
  because the "permissive solver" is MIT by construction.
- **The hand evaluator is our own code** — no third-party evaluator licence to
  vet.
- The only third-party runtime dependency is **NumPy** (BSD-3-Clause), used for
  CFR array math.

## Distributable runtime dependencies

| Component            | Package | Version | Licence      | Commercial use | Role                              |
|----------------------|---------|---------|--------------|----------------|-----------------------------------|
| Array math (CFR)     | numpy   | >=1.24  | BSD-3-Clause | ✅ Yes         | Vectorised regret/strategy arrays |
| Storage              | sqlite3 | stdlib  | Python (PSF) | ✅ Yes         | Training question DB              |
| Storage / interchange| json    | stdlib  | Python (PSF) | ✅ Yes         | Scenario + question files         |
| Local web server     | http.server | stdlib | Python (PSF) | ✅ Yes       | Serves the offline trainer        |
| Hand evaluator       | *(ours)*| —       | MIT          | ✅ Yes         | 5–7 card hand ranking             |
| Solver               | *(ours)*| —       | MIT          | ✅ Yes         | Postflop Discounted-CFR           |

The Python standard library is distributed under the PSF License Agreement,
which permits commercial use and redistribution.

## Development-only dependencies (not shipped)

| Package | Licence | Role                         |
|---------|---------|------------------------------|
| pytest  | MIT     | Test runner                  |

## Reference / benchmarking tools (NOT distributed, NOT imported)

| Tool        | Licence          | Status / constraints |
|-------------|------------------|----------------------|
| TexasSolver | AGPL-3.0 (verify)| Dev-only external reference. Run as a **separate process** in `reference/texassolver/` (gitignored). Never imported, never bundled, never required for the trainer to run. Its licence and the permissibility of internal commercial benchmarking **must be confirmed** before results are relied upon (PRD §3, §5 Step 1, §10). |

### Why TexasSolver must stay isolated

TexasSolver is published under a copyleft licence (AGPL-family). Linking it into
the codebase or shipping it would impose copyleft obligations on the whole
product. The PRD therefore requires it to be used only as an independent,
out-of-process comparison tool. This repo enforces that by:

1. `.gitignore` excludes `reference/texassolver/`.
2. Nothing under `src/` imports or shells out to it at runtime.
3. The benchmark adapter (Deliverable 4) reads TexasSolver's exported files
   offline; it does not embed the tool.

## Open verification items

- [ ] Confirm TexasSolver's exact licence version and whether internal
      commercial benchmarking is permitted (owner: team, PRD §3).
- [ ] Re-check NumPy transitive build deps if we ever pin a source build
      (binary wheels are BSD/permissive).
