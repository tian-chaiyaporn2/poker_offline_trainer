"""Raise action (FR-011): oracle==batched==GPU exact; raise is used (MIT)."""

import numpy as np

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.solver.batched import BatchedCFR
from pokertrainer.solver.batched_gpu import BatchedGPUCFR
from pokertrainer.solver.multistreet import MultiStreetSpike

FLOP = parse_cards("As7h2d")
OOP = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c", "AhKh", "Ts9s"]]
IP = [parse_hand(h) for h in ["AhQh", "KsKh", "JsJh", "AhTh", "Tc9c"]]


def _wo_wi():
    return np.ones(len(OOP)), np.ones(len(IP))


def test_batched_matches_oracle_with_raise():
    wo, wi = _wo_wi()
    for streets in (1, 2):
        a = MultiStreetSpike(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=streets, raise_x=3.0).run(120)
        b = BatchedCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=streets, raise_x=3.0).run(120)
        assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9


def test_gpu_matches_cpu_with_raise():
    wo, wi = _wo_wi()
    c = BatchedCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=2, raise_x=3.0).run(120)
    g = BatchedGPUCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=2, raise_x=3.0,
                      backend="numpy", dtype="float64").run(120)
    assert abs(c["root_ev_oop_bb"] - g["root_ev_oop_bb"]) < 1e-9


def test_no_raise_unchanged():
    wo, wi = _wo_wi()
    a = MultiStreetSpike(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=2).run(120)
    b = BatchedCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=2).run(120)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9


def test_raise_action_is_used():
    # IP with a set should raise facing a bet on a multi-street tree.
    oop = [parse_hand(h) for h in ["KsKc", "QsQc", "KhQh", "Ts9s", "6s5s", "AhKh"]]
    ip = [parse_hand(h) for h in ["AhAc", "7s7c", "AhQh", "JsJh", "Tc9c", "6h5h"]]
    s = BatchedCFR(FLOP, oop, ip, np.ones(6), np.ones(6), 5.5, 0.66, streets=2, raise_x=3.0)
    s.run(300)
    recs = s.flop_decisions_report()
    ivb = [r for r in recs if r["node"] == "btn_vs_bet"]
    assert all(r["actions"] == ["fold", "call", "raise"] for r in ivb)
    assert max(r["freq"]["raise"] for r in ivb) > 0.1     # some strong hand raises


def test_report_has_raise_action_and_cpu_gpu_match():
    wo, wi = _wo_wi()
    c = BatchedCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=2, raise_x=3.0); c.run(100)
    g = BatchedGPUCFR(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=2, raise_x=3.0,
                      backend="numpy", dtype="float64"); g.run(100)
    rc = sorted(c.flop_decisions_report(), key=lambda r: (r["node"], r["hand"]))
    rg = sorted(g.flop_decisions_report(), key=lambda r: (r["node"], r["hand"]))
    assert len(rc) == len(rg)
    for a, b in zip(rc, rg):
        for act in a["ev"]:
            assert abs(a["ev"][act] - b["ev"][act]) < 1e-9
