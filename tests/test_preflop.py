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
