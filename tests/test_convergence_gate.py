"""D1: the content pipeline drops decisions whose preferred action hasn't converged.

Uses a solver stub whose `flop_decisions_report()` returns a DIFFERENT preferred action on the
mid snapshot vs the final one, so we can assert the gate deterministically (no real CFR run):
- a clear-EV spot that flips mid->final is unstable -> not accepted (must not ship as the key)
- a near-indifferent spot that flips is expected noise -> stays accepted
- a stable spot is accepted
"""
from pokertrainer.content_yield import extract_records


class _StubSolver:
    """Minimal BatchedCFR stand-in. Call #1 to flop_decisions_report() is the mid snapshot,
    call #2 the final; `flip` hands report a different preferred on the final call."""
    def __init__(self):
        self._calls = 0

    def run(self, iters):
        return {"root_ev_pct_pot": 50.0}

    def flop_decisions_report(self):
        self._calls += 1
        final = self._calls >= 2
        # clear-EV spot: check is decisively best (gap >> CLEAR_SEP_PCT). Final flips to bet.
        clear = {"node": "bb_first", "hand": "AhKs", "reach_mass": 0.9,
                 "ev": {"check": (0.0 if final else 2.0), "bet": (2.0 if final else 0.0)},
                 "freq": {"check": 0.5, "bet": 0.5},
                 "preferred": ("bet" if final else "check")}
        # near-indifferent spot: check/bet within a whisker. Final flips too, but it's a tie.
        tie = {"node": "btn_vs_check", "hand": "9d9c", "reach_mass": 0.9,
               "ev": {"check": (0.001 if final else 0.0), "bet": (0.0 if final else 0.001)},
               "freq": {"check": 0.5, "bet": 0.5},
               "preferred": ("check" if final else "bet")}
        # stable spot: same preferred both snapshots.
        stable = {"node": "bb_vs_bet", "hand": "Qd7h", "reach_mass": 0.9,
                  "ev": {"fold": 0.0, "call": 3.0}, "freq": {"fold": 0.3, "call": 0.7},
                  "preferred": "call"}
        return [clear, tie, stable]


def _make(*_a, **_k):
    return _StubSolver()


def test_unstable_preferred_is_dropped_but_ties_and_stable_survive():
    recs = extract_records("7c6d2s", ["AhKs"], ["9d9c"], iters=100, make=_make,
                           pot=5.5, bet_frac=0.66)
    by_hand = {r["hand"]: r for r in recs}
    assert by_hand["AhKs"]["stability"] == "unstable"     # clear-EV flip -> unconverged
    assert by_hand["AhKs"]["accepted"] is False           # must not ship as the graded answer
    assert by_hand["9d9c"]["stability"] == "stable"       # near-tie flip is expected noise
    assert by_hand["9d9c"]["accepted"] is True
    assert by_hand["Qd7h"]["stability"] == "stable" and by_hand["Qd7h"]["accepted"] is True
