"""TexasSolver benchmark adapter (Deliverable 4) — MIT wrapper, tool NOT bundled.

IMPORTANT (PRD §3, §5 Step 1, §10):
- TexasSolver is an independent, out-of-process reference ONLY.
- It is NOT imported, NOT bundled, and NOT required for anything to run.
- Its licence (copyleft/AGPL-family) and the permissibility of internal
  commercial benchmarking must be confirmed before its results are relied upon.

This adapter (a) converts a canonical scenario into TexasSolver's text input
format, and (b) parses a TexasSolver strategy export into our NormalizedSolve.
Running the solver is gated behind an explicit binary path AND an
acknowledgement flag so it can never execute by accident.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .normalize import CANON_ACTIONS, NormalizedSolve


class TexasSolverNotConfigured(RuntimeError):
    pass


def scenario_to_texassolver_input(raw: Dict) -> str:
    """Render a canonical scenario as a TexasSolver-style command script.

    Bet sizes, board, ranges, pot and stacks are emitted so the reference solve
    uses inputs equivalent to ours (PRD §5 Step 2 equivalence contract)."""
    board = "".join(raw["board"])
    btn = ",".join(f"{k}:{v}" for k, v in raw["ranges"]["BTN"]["combos"].items())
    bb = ",".join(f"{k}:{v}" for k, v in raw["ranges"]["BB"]["combos"].items())
    sizes = raw["actions"]["bet_sizes_pct_pot"]
    lines = [
        f"set_pot {raw['pot_bb']}",
        f"set_effective_stack {raw['effective_stack_bb']}",
        f"set_board {board}",
        f"set_range_ip {btn}",
        f"set_range_oop {bb}",
        f"set_bet_sizes oop,flop,bet,{sizes['small']},{sizes['large']}",
        f"set_bet_sizes ip,flop,bet,{sizes['small']},{sizes['large']}",
        "set_raise_limit 0",              # match no_raise_v1 abstraction
        "set_allin_threshold 0.67",
        "set_use_isomorphism 1",
        f"set_thread_num 4",
        f"set_accuracy {raw['solver']['convergence_target_exploitability_pct_pot']}",
        f"set_max_iteration {raw['solver']['iterations']}",
        "build_tree",
        "start_solve",
        "set_dump_rounds 1",
        "dump_result output_reference.json",
    ]
    return "\n".join(lines) + "\n"


def run_texassolver(raw: Dict, binary_path: str | None = None,
                    i_acknowledge_license: bool = False) -> NormalizedSolve:
    """Run TexasSolver out-of-process. Gated; never runs by default."""
    binary_path = binary_path or os.environ.get("TEXASSOLVER_BIN")
    if not i_acknowledge_license:
        raise TexasSolverNotConfigured(
            "Refusing to run TexasSolver: pass i_acknowledge_license=True only "
            "after confirming its licence permits internal commercial "
            "benchmarking (PRD §3). See docs/licenses.md."
        )
    if not binary_path or not os.path.exists(binary_path):
        raise TexasSolverNotConfigured(
            "TexasSolver binary not found. Set TEXASSOLVER_BIN or pass "
            "binary_path. The tool is intentionally not bundled (PRD §3)."
        )
    raise TexasSolverNotConfigured(
        "TexasSolver integration is a documented stub for this POC. The binary, "
        "its licence, and benchmarking permission are unverified, so the POC "
        "uses the independent reference CFR instead (see docs/feasibility_report.md "
        "and reference_solver.py). Implement parse_texassolver_output() when a "
        "licence-cleared binary is available."
    )


def parse_texassolver_output(path: str, scenario_id: str, board: List[str]) -> NormalizedSolve:
    """Parse a TexasSolver strategy dump into a NormalizedSolve (stub).

    Left unimplemented on purpose: it must map TexasSolver's hand/suit/action/EV
    conventions onto our canonical ones (PRD §5 Step 5). Implemented only once a
    licence-cleared binary and real output are available."""
    raise NotImplementedError(
        "parse_texassolver_output is a stub; implement against a real, "
        "licence-cleared TexasSolver dump. Canonical mapping required: "
        f"actions -> {CANON_ACTIONS}, hands high-card-first 'shdc', EV in bb."
    )
