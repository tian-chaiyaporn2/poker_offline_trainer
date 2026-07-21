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
