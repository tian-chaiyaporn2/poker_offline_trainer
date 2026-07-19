"""GPU benchmark for the batched multi-street CFR (MIT).

Runs the GPU-ready solver (solver/batched_gpu.py) on whatever backend is present:
CuPy if installed + a GPU is visible, otherwise NumPy (CPU) as a fallback so the
script still runs and self-checks.

On a GPU box:
    pip install cupy-cuda12x          # match your CUDA version
    PYTHONPATH=src python bench/gpu_bench.py 3 80 160 250

Args: <streets> <combo sizes...>   (defaults: streets=3, sizes 40 80 160)

The point of this bench: the dominant cost is the showdown einsum, a big batched
matrix multiply. On CPU it is ~n²-bound (see docs/multistreet_spike.md); on GPU
it should drop sharply. Compare the printed ms/iter across backends.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from pokertrainer.cards import parse_cards                        # noqa: E402
from pokertrainer.presets import BB_SRP, BTN_SRP                  # noqa: E402
from pokertrainer.ranges import expand_range                     # noqa: E402
from pokertrainer.solver.batched_gpu import BatchedGPUCFR, get_backend  # noqa: E402


def subsample(lst, n):
    if n >= len(lst):
        return lst
    idx = np.linspace(0, len(lst) - 1, n).astype(int)
    return [lst[i] for i in idx]


def main():
    streets = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sizes = [int(x) for x in sys.argv[2:]] or [40, 80, 160]

    # POKER_BACKEND=cupy|numpy|auto ; POKER_DTYPE=float32|float64
    prefer = os.environ.get("POKER_BACKEND", "auto")
    dtype = os.environ.get("POKER_DTYPE", "float64")
    backend = get_backend(prefer)[4]
    print(f"backend: {backend}   dtype: {dtype}"
          + ("" if backend == "cupy" else "   (CPU — install cupy on a GPU box for GPU)"))
    print(f"streets={streets}\n")

    flop = parse_cards("As7h2d")
    oop_full = [c for c, _ in expand_range(BB_SRP, flop)]
    ip_full = [c for c, _ in expand_range(BTN_SRP, flop)]

    def free_pool(xp):
        try:                       # best-effort GPU memory release between sizes
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    print(f"  {'combos':>7} {'warmup':>9} {'ms/iter':>9}")
    for n in sizes:
        oop, ip = subsample(oop_full, n), subsample(ip_full, n)
        wo, wi = np.ones(len(oop)), np.ones(len(ip))
        try:
            s = BatchedGPUCFR(flop, oop, ip, wo, wi, 5.5, 0.66, streets=streets,
                              backend=prefer, dtype=dtype)
            # Warm up construction + one iteration on the same instance.
            t = time.time(); s.run(1); warmup = time.time() - t
            K = 10
            t = time.time(); r = s.run(K)
            steady = r["runtime_sec"] / K
            print(f"  {len(oop):>7} {warmup:>9.2f} {steady * 1000:>9.1f}", flush=True)
            xp = s.xp
            del s; free_pool(xp)
        except Exception as e:          # e.g. OOM at large n — keep earlier rows
            print(f"  {len(oop):>7}   FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)

    print("\nAt 600 iters, ms/iter × 600 / 1000 = seconds/board; × 12 / 60 = minutes for the library.")


if __name__ == "__main__":
    main()
