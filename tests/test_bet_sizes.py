"""Multiple bet sizes (bet_fracs) across the three solvers (MIT).

- bet_fracs=None / [0.66] reproduce the legacy single-size tree bit-for-bit.
- The multi-size tree is cross-checked oracle == batched == GPU(NumPy backend), to machine
  precision, with and without raises and with the eff_stack all-in cap.
- Exploitability of the averaged multi-size profile shrinks toward 0.
- Extraction emits per-size action labels (bet_33 / bet_75) and per-size response nodes.
"""

import math

import numpy as np
import pytest

from pokertrainer.cards import parse_cards, parse_hand
from pokertrainer.content_pack import build_pack, verify_pack
from pokertrainer.content_yield import (_is_finite_record, extract_records, node_role,
                                        parse_bet_sizes, validate_records)
from pokertrainer.solver.batched import BatchedCFR
from pokertrainer.solver.batched_gpu import BatchedGPUCFR
from pokertrainer.solver.betsizes import normalize_bet_fracs
from pokertrainer.solver.multistreet import MultiStreetSpike
from pokertrainer.validate_flop import _make_solver

FLOP = parse_cards("As7h2d")
OOP = [parse_hand(h) for h in ["AhAc", "KsKc", "7s7c", "AhKh", "Ts9s"]]
IP = [parse_hand(h) for h in ["AhQh", "KsKh", "JsJh", "AhTh", "Tc9c"]]
SIZES = [0.33, 0.75]


def _w():
    return np.ones(len(OOP)), np.ones(len(IP))


def _mk(cls, streets, **kw):
    wo, wi = _w()
    if cls is BatchedGPUCFR:
        kw.setdefault("backend", "numpy")
        kw.setdefault("dtype", "float64")
    return cls(FLOP, OOP, IP, wo, wi, 5.5, 0.66, streets=streets, **kw)


def _tables_equal(a, b):
    assert set(a.R) == set(b.R)
    for k in a.R:
        assert np.array_equal(a.R[k], b.R[k]) and np.array_equal(a.S[k], b.S[k]), k


# --- (a) default / single-size is the legacy tree, bit-for-bit -------------------------

@pytest.mark.parametrize("raise_x", [None, 3.0])
def test_none_and_single_size_bit_identical_batched(raise_x):
    base = _mk(BatchedCFR, 2, raise_x=raise_x)
    r0 = base.run(25)
    for fr in (None, [0.66]):
        s = _mk(BatchedCFR, 2, raise_x=raise_x, bet_fracs=fr)
        r = s.run(25)
        assert r["root_ev_oop_bb"] == r0["root_ev_oop_bb"]
        assert r["n_infosets"] == r0["n_infosets"]
        _tables_equal(base, s)
        assert s.flop_decisions_report() == base.flop_decisions_report()
        assert s.flop_root_report() == base.flop_root_report()
    assert all(r["actions"] in (["check", "bet"], ["fold", "call"], ["fold", "call", "raise"])
               for r in base.flop_decisions_report())
    assert not any("facing_bet_frac" in r for r in base.flop_decisions_report())


def test_none_and_single_size_bit_identical_gpu_and_oracle():
    g0 = _mk(BatchedGPUCFR, 2, dtype="float32")
    g0.run(10)
    g1 = _mk(BatchedGPUCFR, 2, dtype="float32", bet_fracs=[0.66])
    g1.run(10)
    _tables_equal(g0, g1)
    assert g0.flop_decisions_report() == g1.flop_decisions_report()
    o0 = _mk(MultiStreetSpike, 1).run(40)["root_ev_oop_bb"]
    o1 = _mk(MultiStreetSpike, 1, bet_fracs=[0.66]).run(40)["root_ev_oop_bb"]
    assert o0 == o1


def test_generic_tree_with_one_size_matches_legacy():
    """The multi-size code path, forced on with a single size, solves the same game as the
    legacy tree (different infoset path tokens, same values)."""
    for raise_x in (None, 3.0):
        a = _mk(BatchedCFR, 2, raise_x=raise_x).run(30)["root_ev_oop_bb"]
        s = _mk(BatchedCFR, 2, raise_x=raise_x)
        s._multi = True
        assert abs(s.run(30)["root_ev_oop_bb"] - a) < 1e-9


# --- (b) multi-size: batched == oracle (machine precision) -----------------------------

@pytest.mark.parametrize("raise_x", [None, 3.0])
def test_multi_size_matches_oracle_flop_only(raise_x):
    a = _mk(MultiStreetSpike, 1, raise_x=raise_x, bet_fracs=SIZES).run(80)
    b = _mk(BatchedCFR, 1, raise_x=raise_x, bet_fracs=SIZES).run(80)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9


@pytest.mark.parametrize("raise_x", [None, 3.0])
def test_multi_size_matches_oracle_two_streets(raise_x):
    a = _mk(MultiStreetSpike, 2, raise_x=raise_x, bet_fracs=SIZES).run(6)
    b = _mk(BatchedCFR, 2, raise_x=raise_x, bet_fracs=SIZES).run(6)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9


@pytest.mark.parametrize("raise_x,iters", [(None, 6), (3.0, 2)])
def test_multi_size_eff_stack_cap_matches_oracle(raise_x, iters):
    # eff 6bb behind a 5.5bb pot: flop 50%/100% bets are 2.75/5.5 (distinct), but after a
    # flop bet+call BOTH turn sizes cap to the same all-in — kept as two identical actions.
    # With raises the capped tree is full of EXACT action ties (raise-to == call once
    # all-in), where a 1e-16 summation-order difference flips a CFR+ regret across 0; the
    # legacy single-size raise+eff tree shows the same oracle/batched drift after a few
    # iterations. So the raise case is compared over the first iterations only.
    kw = dict(bet_fracs=[0.5, 1.0], eff_stack=6.0, raise_x=raise_x)
    a = _mk(MultiStreetSpike, 2, **kw).run(iters)
    b = _mk(BatchedCFR, 2, **kw).run(iters)
    assert abs(a["root_ev_oop_bb"] - b["root_ev_oop_bb"]) < 1e-9
    # and the cap actually binds (differs from the uncapped tree)
    c = _mk(BatchedCFR, 2, bet_fracs=[0.5, 1.0], raise_x=raise_x).run(iters)
    assert abs(c["root_ev_oop_bb"] - b["root_ev_oop_bb"]) > 1e-6


# --- (c) GPU class parity (NumPy backend == batched; float32 never promotes) -----------

@pytest.mark.parametrize("raise_x", [None, 3.0])
def test_gpu_multi_size_matches_batched(raise_x):
    c = _mk(BatchedCFR, 2, raise_x=raise_x, bet_fracs=SIZES)
    g = _mk(BatchedGPUCFR, 2, raise_x=raise_x, bet_fracs=SIZES)
    rc, rg = c.run(20), g.run(20)
    assert g.backend == "numpy"
    assert abs(rc["root_ev_oop_bb"] - rg["root_ev_oop_bb"]) < 1e-9
    a = sorted(c.flop_decisions_report(), key=lambda r: (r["node"], r["hand"]))
    b = sorted(g.flop_decisions_report(), key=lambda r: (r["node"], r["hand"]))
    assert [(r["node"], r["hand"], r["actions"]) for r in a] == \
           [(r["node"], r["hand"], r["actions"]) for r in b]
    for x, y in zip(a, b):
        for act in x["ev"]:
            assert abs(x["ev"][act] - y["ev"][act]) < 1e-9
            assert abs(x["freq"][act] - y["freq"][act]) < 1e-9


@pytest.mark.parametrize("raise_x", [None, 3.0])
def test_gpu_multi_size_float32_stays_float32(raise_x):
    s = _mk(BatchedGPUCFR, 3, dtype="float32", raise_x=raise_x, bet_fracs=SIZES,
            eff_stack=20.0)
    s.run(2)
    assert s.R and all(str(a.dtype) == "float32" for a in list(s.R.values()) + list(s.S.values()))
    for r in s.flop_decisions_report():
        assert all(math.isfinite(v) for v in r["ev"].values())
        assert all(math.isfinite(v) for v in r["freq"].values())


# --- (d) convergence on a tiny multi-size game ----------------------------------------

@pytest.mark.parametrize("raise_x", [None, 3.0])
def test_multi_size_exploitability_decreases(raise_x):
    ex = []
    for it in (10, 50, 300):
        s = _mk(MultiStreetSpike, 1, raise_x=raise_x, bet_fracs=SIZES)
        s.run(it)
        ex.append(s.exploitability())
    assert all(e >= -1e-9 for e in ex)
    assert ex[0] > ex[1] > ex[2]
    assert ex[2] < 0.02 * 5.5          # < 2% of the pot


# --- (e) extraction: per-size labels and per-size response records --------------------

def test_extraction_labels_and_per_size_response_nodes():
    s = _mk(BatchedCFR, 1, bet_fracs=[0.75, 0.33])       # unsorted input -> canonical order
    s.run(60)
    recs = s.flop_decisions_report()
    by_node = {}
    for r in recs:
        by_node.setdefault(r["node"], []).append(r)
    assert set(by_node) == {"bb_first", "btn_vs_check", "bb_vs_bet_33", "btn_vs_bet_33",
                            "bb_vs_bet_75", "btn_vs_bet_75"}
    for n in ("bb_first", "btn_vs_check"):
        assert all(r["actions"] == ["check", "bet_33", "bet_75"] for r in by_node[n])
        assert all("facing_bet_frac" not in r for r in by_node[n])
    for lab, frac in (("bet_33", 0.33), ("bet_75", 0.75)):
        for n in (f"bb_vs_{lab}", f"btn_vs_{lab}"):
            assert len(by_node[n]) == len(OOP if n.startswith("bb") else IP)
            assert all(r["actions"] == ["fold", "call"] for r in by_node[n])
            assert all(r["facing_bet"] == lab and r["facing_bet_frac"] == frac
                       for r in by_node[n])
    # facing a small vs a big bet are genuinely different spots (different EVs)
    ev33 = {r["hand"]: r["ev"]["call"] for r in by_node["btn_vs_bet_33"]}
    ev75 = {r["hand"]: r["ev"]["call"] for r in by_node["btn_vs_bet_75"]}
    assert any(abs(ev33[h] - ev75[h]) > 1e-6 for h in ev33)
    rr = s.flop_root_report()
    assert all(set(v["ev"]) == {"check", "bet_33", "bet_75"} for v in rr.values())


def test_extraction_with_raise_has_per_size_raise_nodes():
    s = _mk(BatchedCFR, 1, bet_fracs=SIZES, raise_x=3.0)
    s.run(30)
    recs = s.flop_decisions_report()
    vb = [r for r in recs if "_vs_bet_" in r["node"]]
    assert {r["node"] for r in vb} == {"bb_vs_bet_33", "btn_vs_bet_33",
                                       "bb_vs_bet_75", "btn_vs_bet_75"}
    assert all(r["actions"] == ["fold", "call", "raise"] for r in vb)


def test_content_yield_records_and_pack(tmp_path):
    make = _make_solver("cpu", "float64", bet_fracs=SIZES)
    recs = extract_records("As7h2d", OOP, IP, 20, make, 5.5, 0.66, streets=1)
    assert not validate_records(recs)
    assert all(_is_finite_record(r) for r in recs)
    nodes = {r["node"] for r in recs}
    assert {"bb_vs_bet_33", "btn_vs_bet_75", "bb_first", "btn_vs_check"} <= nodes
    for r in recs:
        if "_vs_bet_" in r["node"]:
            assert r["decision_type"] == "vs_bet" and r["facing_bet_frac"] in SIZES
        else:
            assert r["decision_type"] == "first_action"
        if r["preferred"].startswith("bet_"):
            assert r["explanation"]["reason"] in ("value", "protection", "bluff", "semi_bluff",
                                                  "mixed")
    # sized labels read naturally in the explanation text
    assert any("bet 33%" in d or "bet 75%" in d
               for r in recs for d in r["explanation"]["detail"])
    out = build_pack(recs, {"bet_sizes_pct": [33, 75]}, str(tmp_path), "betsizes_test")
    v = verify_pack(str(tmp_path / "flop_pack_betsizes_test.db"))
    assert v["hash_ok"] and v["signature_ok"]
    assert out


def test_node_role_relabels_per_size():
    assert node_role("bb_vs_bet") == ("oop", "vs_bet")
    assert node_role("btn_vs_bet_75") == ("ip", "vs_bet_75")
    with pytest.raises(KeyError):
        node_role("nonsense")


# --- validation / rejected combinations ------------------------------------------------

def test_bet_size_validation():
    assert normalize_bet_fracs(0.66, None, 5.5, None) == ([0.66], ["bet"])
    assert normalize_bet_fracs(0.66, [0.5], 5.5, None) == ([0.5], ["bet"])
    assert normalize_bet_fracs(0.66, [0.75, 0.33], 5.5, None) == ([0.33, 0.75],
                                                                  ["bet_33", "bet_75"])
    for bad in ([0.33, 0.33], [0.331, 0.334], [0.0, 0.5], [-0.5, 0.5], [float("nan"), 0.5],
                [], [0.1 * k for k in range(1, 11)]):
        with pytest.raises(ValueError):
            normalize_bet_fracs(0.66, bad, 5.5, None)
    # both sizes all-in at the root -> same action under two labels: rejected
    with pytest.raises(ValueError, match="all-in"):
        BatchedCFR(FLOP, OOP, IP, *_w(), 5.5, bet_fracs=[1.0, 1.5], eff_stack=4.0)


def test_multi_size_rejects_trajectory_extraction():
    for s in (_mk(MultiStreetSpike, 1, bet_fracs=SIZES), _mk(BatchedGPUCFR, 1, bet_fracs=SIZES)):
        with pytest.raises(ValueError, match="single-size"):
            s.eval_capture_targets({("", tuple(FLOP))})


def test_parse_bet_sizes_cli():
    assert parse_bet_sizes(None) is None
    assert parse_bet_sizes("") is None
    assert parse_bet_sizes("0.33,0.75") == [0.33, 0.75]
    with pytest.raises(SystemExit):
        parse_bet_sizes("0.33,big")
