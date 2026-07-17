"""Batched multi-street CFR must match the naive oracle exactly (MIT).

Both solve the same full-history game; identical inputs + iterations must give
identical EVs (to machine precision). This is the correctness guarantee for the
batched public-tree solver.
"""

import numpy as np

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.solver.batched import BatchedCFR
from pokertrainer.solver.multistreet import MultiStreetSpike

FLOP = parse_cards("As7h2d")
OOP = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c", "AhKh", "Ts9s"]]
IP = [parse_hand(h) for h in ["AhQh", "KsKh", "JsJh", "AhTh", "Tc9c"]]


def _run(cls, streets, iters):
    wo, wi = np.ones(len(OOP)), np.ones(len(IP))
    return cls(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=streets).run(iters)


def test_flop_only_matches_oracle():
    a = _run(MultiStreetSpike, 1, 60)
    b = _run(BatchedCFR, 1, 60)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9


def test_two_street_matches_oracle():
    a = _run(MultiStreetSpike, 2, 80)
    b = _run(BatchedCFR, 2, 80)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9


def test_batched_converges_and_is_deterministic():
    low = _run(BatchedCFR, 2, 50)["root_ev_oop_bb"]
    high = _run(BatchedCFR, 2, 400)["root_ev_oop_bb"]
    # stabilises (small change) and is reproducible
    assert abs(high - low) < 0.5
    again = _run(BatchedCFR, 2, 400)["root_ev_oop_bb"]
    assert high == again
