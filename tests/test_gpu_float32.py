"""Guards for the float32 GPU path (MIT).

The GPU solver's whole reason to exist is a fast float32 run on FP64-limited
cards (e.g. Kaggle T4). A single un-typed accumulator once silently promoted the
run to float64 (~32x slower on a T4) — invisible to the float64/streets<=2 tests.
These tests exercise the full street-3 tree (with and without the raise action)
on the NumPy backend at float32 and assert the dtype never escapes.
"""

import math

import numpy as np

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.solver.batched_gpu import BatchedGPUCFR

FLOP = parse_cards("As7h2d")
OOP = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c", "AhKh", "Ts9s", "QsQd"]]
IP = [parse_hand(h) for h in ["AhQh", "KsKh", "JsJh", "AhTh", "Tc9c", "9s8s"]]


def _solve(dtype, raise_x):
    wo, wi = np.ones(len(OOP)), np.ones(len(IP))
    s = BatchedGPUCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=3,
                      backend="numpy", dtype=dtype, raise_x=raise_x)
    s.run(6)
    return s


def _all_dtype(s, dtype):
    arrs = list(s.R.values()) + list(s.S.values())
    return arrs and all(str(a.dtype) == dtype for a in arrs)


def test_float32_stays_float32_full_tree():
    # streets=3 chance nodes must not promote regrets/strategies to float64.
    s = _solve("float32", None)
    assert _all_dtype(s, "float32"), "regret/strategy tables leaked to float64"


def test_float32_stays_float32_with_raise():
    s = _solve("float32", 3)
    assert _all_dtype(s, "float32"), "raise-tree tables leaked to float64"


def test_float64_backend_unaffected():
    s = _solve("float64", 3)
    assert _all_dtype(s, "float64")


def test_report_finite_float32():
    s = _solve("float32", 3)
    recs = s.flop_decisions_report()
    assert recs
    for r in recs:
        assert all(math.isfinite(v) for v in r["ev"].values())
        assert all(math.isfinite(v) for v in r["freq"].values())
