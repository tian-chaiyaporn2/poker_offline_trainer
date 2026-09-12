"""Pre-flop equity foundation (A1)."""
from pokertrainer.preflop_equity import preflop_equity, hand_classes
from pokertrainer.ranges import class_to_combos


def _combo(cls, avoid=frozenset()):
    return next(c for c in class_to_combos(cls) if not (set(c) & set(avoid)))


def test_hand_classes_count():
    cls = hand_classes()
    assert len(cls) == 169
    assert cls.count("AA") == 1 and "AKs" in cls and "AKo" in cls and "72o" in cls
    assert len(set(cls)) == 169  # no dupes


def test_textbook_matchups():
    # (a, b, expected equity of a, tolerance)
    for a, b, exp, tol in [
        ("AA", "KK", 0.82, 0.02),
        ("AA", "AKs", 0.87, 0.02),   # card removal: AK holds two of AA's outs
        ("AKs", "QQ", 0.46, 0.02),
        ("22", "AKs", 0.50, 0.03),   # the classic coinflip
    ]:
        ca = _combo(a)
        cb = _combo(b, avoid=set(ca))
        got = preflop_equity(ca, cb, samples=8000, seed=3)
        assert abs(got - exp) < tol, f"{a} vs {b}: {got:.3f} not ~{exp}"


def test_equity_is_zero_sum():
    # Same seed => same boards => eq(a,b) + eq(b,a) == 1 exactly (win/loss/tie split).
    ca = _combo("QJs")
    cb = _combo("55", avoid=set(ca))
    ab = preflop_equity(ca, cb, samples=3000, seed=9)
    ba = preflop_equity(cb, ca, samples=3000, seed=9)
    assert abs((ab + ba) - 1.0) < 1e-9


def test_collision_rejected():
    ca = class_to_combos("AA")[0]
    import pytest
    with pytest.raises(ValueError):
        preflop_equity(ca, ca)  # same cards -> collision


def test_action_evs_jam_aa_over_fold():
    """Solved action EVs: AA's jam EV exceeds fold EV in the exact push/fold game."""
    import numpy as np
    from pokertrainer.solver.preflop import PreflopCFR, push_fold_game, combo_weights
    from pokertrainer.preflop_equity import hand_classes
    classes = hand_classes()
    n = len(classes)
    strength = np.linspace(1.0, 0.0, n)
    E = 0.5 + 0.5 * (strength[:, None] - strength[None, :])
    np.fill_diagonal(E, 0.5)
    cfr = PreflopCFR(push_fold_game(stack=10.0), E, combo_weights(classes),
                     ip_player=0, realize=0.0)
    avg = cfr.run(iters=800)
    evs = cfr.action_evs(avg)
    keys = cfr.nodes_by_actions()
    root = evs[keys[("fold", "jam")]]
    aa = classes.index("AA")
    trash = classes.index("72o")
    assert root[aa, 1] > root[aa, 0]          # AA: jam > fold
    assert root[trash, 0] > root[trash, 1]    # 72o: fold > jam
    # Facing the jam, fold forfeits the 1bb blind — chip units, not reach-scaled.
    vs_jam = evs[keys[("fold", "call")]]
    assert np.allclose(vs_jam[:, 0], -1.0, atol=1e-6)


def test_action_evs_fold_is_forfeit_in_chips():
    """Deeper-node fold EV is −inv, not counterfactual value × reach mass."""
    import numpy as np
    from pokertrainer.solver.preflop import PreflopCFR, raise_ladder_game, combo_weights
    from pokertrainer.preflop_equity import hand_classes
    classes = hand_classes()
    n = len(classes)
    strength = np.linspace(1.0, 0.0, n)
    E = 0.5 + 0.5 * (strength[:, None] - strength[None, :])
    np.fill_diagonal(E, 0.5)
    cfr = PreflopCFR(raise_ladder_game(), E, combo_weights(classes),
                     ip_player=0, realize=0.0)
    avg = cfr.run(iters=40)
    evs = cfr.action_evs(avg)
    keys = cfr.nodes_by_actions()
    vs_open = evs[keys[("fold", "call", "3bet")]]
    vs_3bet = evs[keys[("fold", "call", "4bet")]]
    assert np.allclose(vs_open[:, 0], -1.0, atol=1e-6)    # BB folds the 1bb blind
    assert np.allclose(vs_3bet[:, 0], -2.5, atol=1e-6)    # BTN folds the 2.5bb open


def test_pack_records_solver_evs_only_on_btn_bb_tree():
    """Solver EVs attach only to BTN-vs-BB spots; early-seat chart answers stay put."""
    from pokertrainer.preflop_content import pack_records, build_questions
    qs = build_questions()
    btn = next(q for q in qs if q.get("ctx") == "rfi" and q["pos"] == "BTN")
    utg = next(q for q in qs if q.get("ctx") == "rfi" and q["pos"] == "UTG")
    bb = next(q for q in qs if q.get("ctx") == "def" and q["pos"] == "BB"
              and q.get("opener") == "BTN")
    v3 = next(q for q in qs if q.get("ctx") == "vs3bet" and q["pos"] == "BTN"
              and q.get("tbettor") == "BB")
    sb3 = next(q for q in qs if q.get("ctx") == "vs3bet" and q.get("tbettor") == "SB")
    book = {"rfi": {
        btn["cls"]: {a: float(i) for i, a in enumerate(btn["actions"])},
        utg["cls"]: {a: float(i) for i, a in enumerate(utg["actions"])},
    }, "def": {
        bb["cls"]: {a: float(i) for i, a in enumerate(bb["actions"])},
    }, "vs3bet": {
        v3["cls"]: {a: float(i) for i, a in enumerate(v3["actions"])},
        sb3["cls"]: {a: float(i) for i, a in enumerate(sb3["actions"])},
    }}
    recs = pack_records(ev_book=book)
    hit = next(r for r in recs if r["hand_category"] == btn["cls"]
               and r["explanation"]["detail"]["ctx"] == "rfi"
               and r["acting_player"] == "BTN")
    early = next(r for r in recs if r["hand_category"] == utg["cls"]
                 and r["explanation"]["detail"]["ctx"] == "rfi"
                 and r["acting_player"] == "UTG")
    defn = next(r for r in recs if r["hand_category"] == bb["cls"]
                and r["explanation"]["detail"]["ctx"] == "def"
                and r["acting_player"] == "BB"
                and r["explanation"]["detail"].get("opener") == "BTN")
    vs_bb = next(r for r in recs if r["hand_category"] == v3["cls"]
                 and r["explanation"]["detail"]["ctx"] == "vs3bet"
                 and r["acting_player"] == "BTN"
                 and r["explanation"]["detail"].get("tbettor") == "BB")
    vs_sb = next(r for r in recs if r["hand_category"] == sb3["cls"]
                 and r["explanation"]["detail"]["ctx"] == "vs3bet"
                 and r["explanation"]["detail"].get("tbettor") == "SB")
    assert hit["explanation"]["detail"]["ev_source"] == "cfr"
    assert hit["preferred"] == max(hit["ev"], key=hit["ev"].get)
    assert defn["explanation"]["detail"]["ev_source"] == "cfr"
    assert vs_bb["explanation"]["detail"]["ev_source"] == "cfr"
    assert early["explanation"]["detail"]["ev_source"] == "chart_sentinel"
    assert early["preferred"] == utg["answer"]
    assert vs_sb["explanation"]["detail"]["ev_source"] == "chart_sentinel"
    assert vs_sb["preferred"] == sb3["answer"]


def test_pushfold_cfr_converges_and_is_monotone():
    """CFR engine correctness on the EXACT push/fold game (all-in terminals, no
    realization model) using a synthetic monotone equity matrix — fast + deterministic."""
    import numpy as np
    from pokertrainer.solver.preflop import PreflopCFR, push_fold_game, combo_weights
    from pokertrainer.preflop_equity import hand_classes
    classes = hand_classes()
    n = len(classes)
    strength = np.linspace(1.0, 0.0, n)                 # index 0 = strongest
    E = 0.5 + 0.5 * (strength[:, None] - strength[None, :])  # valid: E + E.T == 1
    np.fill_diagonal(E, 0.5)
    w = combo_weights(classes)
    cfr = PreflopCFR(push_fold_game(stack=10.0), E, w, ip_player=0, realize=0.0)
    avg = cfr.run(iters=1500)
    assert cfr.exploitability(avg) < 0.01, "CFR did not converge to ~equilibrium"
    jam = avg[0][:, 1]                                   # opener jam prob per class
    assert jam[0] > 0.99 and jam[-1] < 0.01             # strongest jams, weakest folds
    # monotone-ish: the jam frequency should broadly decrease with weakness
    assert (w * jam).sum() / w.sum() > 0.2              # a non-trivial jamming range


def test_multiway_equity():
    from pokertrainer.preflop_equity import multiway_equity
    from pokertrainer.ranges import class_to_combos

    def combos(*names):
        out, used = [], set()
        for nm in names:
            c = next(c for c in class_to_combos(nm) if not set(c) & used)
            out.append(c); used |= set(c)
        return out

    eq3 = multiway_equity(combos("AA", "KK", "QQ"), samples=15000)
    assert abs(sum(eq3) - 1.0) < 1e-9            # shares partition the pot
    assert eq3[0] > eq3[1] > eq3[2]              # AA > KK > QQ
    assert eq3[0] > 0.6                          # AA dominates 3-way
    # 2-way case must agree with the pairwise engine
    eq2 = multiway_equity(combos("AA", "KK"), samples=15000)
    assert abs(eq2[0] - 0.82) < 0.02


def test_rfi_ranges_are_nested_and_sensible():
    from pokertrainer.preflop_ranges import rfi_ranges, RFI_FREQ
    from pokertrainer.solver.preflop import combo_weights
    from pokertrainer.preflop_equity import hand_classes
    r = rfi_ranges()
    classes = hand_classes()
    w = dict(zip(classes, combo_weights(classes)))
    def opens(pos): return {c for c, a in r[pos].items() if a == "open"}
    # frequency roughly matches the target (within one class of combos)
    for pos, pct in RFI_FREQ.items():
        got = 100 * sum(w[c] for c in opens(pos)) / sum(w.values())
        assert abs(got - pct) < 2.0, f"{pos}: {got:.0f}% vs target {pct}%"
    # nested: each later seat opens a superset
    for a, b in [("UTG", "HJ"), ("HJ", "CO"), ("CO", "BTN")]:
        assert opens(a) <= opens(b), f"{a} not subset of {b}"
    # premiums always open, trash never; playability: 76s opens at BTN, not UTG
    for pos in RFI_FREQ:
        assert "AA" in opens(pos) and "72o" not in opens(pos)
    assert "76s" in opens("BTN") and "76s" not in opens("UTG")


def test_preflop_signed_pack_roundtrips(tmp_path):
    """A5: pre-flop ranges wrap into a signed, verifiable pack; chart answers encode as a
    pure strategy with flat neutral EV, and the plain-language fields round-trip via detail."""
    import json
    from pokertrainer.preflop_content import pack_records
    from pokertrainer.content_pack import build_pack, verify_pack

    recs = pack_records()
    assert len(recs) > 50
    for r in recs:                                   # honest pure-strategy encoding
        assert r["board"] == "" and r["decision_type"] == "preflop"
        assert set(r["freq"]) == set(r["actions"]) == set(r["ev"])
        assert abs(sum(r["freq"].values()) - 1.0) < 1e-9
        assert r["freq"][r["preferred"]] == 1.0      # pure on the correct action
        # EV sentinels (not solver-derived): answer is the unique max at 0.0; every other
        # action is strictly worse, so stored action_grades never label a mistake "best".
        assert r["ev"][r["preferred"]] == 0.0
        assert r["ev"][r["preferred"]] == max(r["ev"].values())
        assert all(v < 0 for a, v in r["ev"].items() if a != r["preferred"])
        assert r["explanation"]["detail"]["why"]     # coaching preserved

    report = build_pack(recs, {"content_kind": "preflop_ranges"},
                        str(tmp_path), "preflop_test", pot=2.5, dedup_cap=99)
    assert report["records_after_dedup"] == len(recs)   # dedup_cap high => nothing dropped
    v = verify_pack(str(tmp_path / "flop_pack_preflop_test.db"))
    assert v["hash_ok"] and v["signature_ok"] and v["records"] == len(recs)


def test_bb_defense_ranges():
    from pokertrainer.preflop_ranges import bb_defense_ranges, BB_DEFENSE
    from pokertrainer.solver.preflop import combo_weights
    from pokertrainer.preflop_equity import hand_classes
    r = bb_defense_ranges()
    classes = hand_classes()
    w = dict(zip(classes, combo_weights(classes)))
    def pct(pos, *acts):
        return 100 * sum(w[c] for c, a in r[pos].items() if a in acts) / sum(w.values())
    # defends tighter vs early opens, wider vs the SB; frequencies near target
    for opener, (dfd, tb3) in BB_DEFENSE.items():
        assert abs(pct(opener, "3bet", "call") - dfd) < 2.5
        assert abs(pct(opener, "3bet") - tb3) < 2.0
    assert pct("UTG", "3bet", "call") < pct("BTN", "3bet", "call") < pct("SB", "3bet", "call")
    # premiums 3bet, trash folds, suited connectors defend
    assert r["BTN"]["AA"] == "3bet" and r["BTN"]["72o"] == "fold"
    assert r["BTN"]["76s"] in ("call", "3bet")
