"""Pre-flop CFR over hand classes (MIT).

A 2-player pre-flop sub-game solver (the heads-up-by-flop model: opener vs one
defender). Vector CFR — each info set is (node, hand-class) and strategies/ranges are
169-vectors — so an iteration is a handful of matrix-vector products.

Terminal EVs:
  * all-in / showdown  -> exact chip-EV from the 169x169 pre-flop equity table.
  * fold               -> the folder loses what they put in.
  * see-a-flop (call not all-in) -> equity + a position REALIZATION bonus. This is a
    modeled approximation (the honest way to do pre-flop without coupling to the full
    postflop solve); it is a tunable parameter, NOT claimed-exact GTO. All-in-only
    games (push/fold) use no realization term and are therefore exact.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

_TABLE_PATH = os.path.join("output", "preflop", "equity_169.npz")


def load_equity_table(path: str = _TABLE_PATH) -> Tuple[List[str], np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return list(d["classes"]), d["E"].astype(np.float64)


def combo_weights(classes: List[str]) -> np.ndarray:
    """Combos per class: pair=6, suited=4, offsuit=12."""
    w = np.empty(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        w[i] = 6.0 if len(c) == 2 else (4.0 if c.endswith("s") else 12.0)
    return w


# ---- game tree -------------------------------------------------------------------
# A node is a dict:
#   decision: {"actor": 0|1, "actions": [(name, child_node), ...]}
#   terminal: {"term": kind, "pot": float, "inv": (invO, invD)}  kind in
#             {"fold0","fold1","allin","flop"}. invO/invD are chips each committed.

def _fold(kind, pot, invO, invD):
    return {"term": kind, "pot": pot, "inv": (invO, invD)}


def raise_ladder_game(stack: float = 100.0, bb: float = 1.0, sb: float = 0.5,
                      open_to: float = 2.5, tbet_to: float = 10.0,
                      fbet_to: float = 24.0) -> Dict:
    """Deep-stack RFI ladder: opener (player 0) opens; defender (player 1) fold/call/3bet;
    opener fold/call/4bet; defender fold/call/jam; opener fold/call. Calls that aren't
    all-in end at a 'flop' terminal (realization model); the jam-call is an exact all-in
    showdown. Sizes in bb; dead SB stays in the pot. (open+call gives pot 5.5 — the same
    single-raised pot the postflop trainer already uses.)"""
    dead = sb

    def fold0(invO, invD):  # opener folded
        return {"term": "fold0", "pot": invO + invD + dead, "inv": (invO, invD)}

    def fold1(invO, invD):  # defender folded
        return {"term": "fold1", "pot": invO + invD + dead, "inv": (invO, invD)}

    def flop(invO, invD):
        return {"term": "flop", "pot": invO + invD + dead, "inv": (invO, invD)}

    def allin(invO, invD):
        return {"term": "allin", "pot": invO + invD + dead, "inv": (invO, invD)}

    o_vs_jam = {"actor": 0, "actions": [
        ("fold", fold0(fbet_to, stack)), ("call", allin(stack, stack))]}
    d_vs_4bet = {"actor": 1, "actions": [
        ("fold", fold1(fbet_to, tbet_to)), ("call", flop(fbet_to, fbet_to)), ("jam", o_vs_jam)]}
    o_vs_3bet = {"actor": 0, "actions": [
        ("fold", fold0(open_to, tbet_to)), ("call", flop(tbet_to, tbet_to)), ("4bet", d_vs_4bet)]}
    d_vs_open = {"actor": 1, "actions": [
        ("fold", fold1(open_to, bb)), ("call", flop(open_to, open_to)), ("3bet", o_vs_3bet)]}
    root = {"actor": 0, "actions": [
        ("fold", fold0(0.0, bb)), ("open", d_vs_open)]}
    return root


def push_fold_game(stack: float = 10.0, bb: float = 1.0, sb: float = 0.5) -> Dict:
    """Opener (player 0, the button) jams or folds; defender (player 1, BB) calls or
    folds. All non-fold ends are all-in -> exact (no realization term). Dead SB stays
    in the pot. Stacks in bb."""
    dead = sb                                   # SB folds, its blind is dead money
    # opener folds -> opener put in 0, loses nothing; defender keeps blind + dead
    t_open_fold = _fold("fold0", pot=bb + dead, invO=0.0, invD=bb)
    # opener jams, defender folds -> opener wins (bb + dead), invested = stack? no: on a
    # fold win nobody is all-in; opener risked stack but wins the pot uncontested.
    t_def_fold = _fold("fold1", pot=stack + bb + dead, invO=stack, invD=bb)
    # opener jams, defender calls -> all-in showdown, pot = 2*stack + dead
    t_showdown = {"term": "allin", "pot": 2 * stack + dead, "inv": (stack, stack)}
    vs_jam = {"actor": 1, "actions": [("fold", t_def_fold), ("call", t_showdown)]}
    jam_node = vs_jam
    root = {"actor": 0, "actions": [("fold", t_open_fold), ("jam", jam_node)]}
    return root


# ---- terminal utility matrices ---------------------------------------------------

def type_bonus(classes: List[str]) -> np.ndarray:
    """Per-class realization bonus: how much better/worse a hand plays postflop than its
    raw equity suggests. Suited + connected realize MORE (disguised straights/flushes);
    disconnected offsuit realizes LESS (dominated, hard to continue). Modeled, tunable."""
    order = "23456789TJQKA"
    ri = {r: i for i, r in enumerate(order)}
    tb = np.zeros(len(classes))
    for i, c in enumerate(classes):
        if len(c) == 2:                         # pocket pair — set mining, modest bump
            tb[i] = 0.02
            continue
        hi, lo, suit = c[0], c[1], c[2]
        gap = abs(ri[hi] - ri[lo])
        if suit == "s":                          # suited: flush + straight potential
            b = 0.05
            if gap == 1:
                b += 0.04                         # suited connector
            elif gap == 2:
                b += 0.02
            elif gap >= 5:
                b -= 0.02
        else:                                    # offsuit: plays for high-card/pair, not draws
            b = 0.0
            if gap == 1:
                b += 0.01                         # a little connectedness value only
            if gap >= 4:
                b -= 0.03                         # disconnected offsuit is hard to play
            if gap >= 6:
                b -= 0.02
        tb[i] = b
    return tb


def _terminal_util(node: Dict, E: np.ndarray, ip_player: int, realize: float,
                   tb: np.ndarray = None):
    """Return (U0, U1): 169x169 net-chip utilities (final - initial) for each (h0,h1).
    U0 + U1 == dead money everywhere, so the game stays (near) zero-sum."""
    n = E.shape[0]
    kind = node["term"]
    pot = node["pot"]
    invO, invD = node["inv"]
    if kind == "fold0":                        # opener folded
        U0 = np.full((n, n), -invO)
        U1 = np.full((n, n), pot - invD)
    elif kind == "fold1":                      # defender folded
        U0 = np.full((n, n), pot - invO)
        U1 = np.full((n, n), -invD)
    else:                                       # allin or flop -> equity-based
        share0 = E                              # player 0's realized share of the pot
        if kind == "flop":                      # not all-in: realization applies
            pos0 = realize if ip_player == 0 else -realize
            diff = 0.0 if tb is None else (tb[:, None] - tb[None, :])  # O plays better if tb[i]>tb[j]
            share0 = np.clip(E + pos0 + diff, 0.0, 1.0)
        U0 = share0 * pot - invO
        U1 = (1.0 - share0) * pot - invD        # conserves: U0+U1 == dead
    return U0, U1


# ---- vector CFR ------------------------------------------------------------------

class PreflopCFR:
    def __init__(self, root: Dict, E: np.ndarray, w: np.ndarray,
                 ip_player: int = 0, realize: float = 0.0, tb: np.ndarray = None):
        self.root, self.E, self.w = root, E, w
        self.n = E.shape[0]
        self.ip_player, self.realize, self.tb = ip_player, realize, tb
        self.regret: Dict[int, np.ndarray] = {}
        self.strat_sum: Dict[int, np.ndarray] = {}
        self._id = {}
        self._label(root)

    def _label(self, node, key=""):            # assign a stable id per decision node
        if "term" in node:
            return
        nid = len(self._id)
        self._id[id(node)] = nid
        a = len(node["actions"])
        self.regret[nid] = np.zeros((self.n, a))
        self.strat_sum[nid] = np.zeros((self.n, a))
        for name, child in node["actions"]:
            self._label(child, key + "/" + name)

    def _sigma(self, nid):
        r = np.maximum(self.regret[nid], 0.0)
        s = r.sum(axis=1, keepdims=True)
        a = r.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s > 0, r / s, 1.0 / a)

    def _walk(self, node, reach0, reach1, update=True):
        """Return (v0, v1): counterfactual value vectors (len n) for each player."""
        if "term" in node:
            U0, U1 = _terminal_util(node, self.E, self.ip_player, self.realize, self.tb)
            # v0[h] = sum_h' (w[h']*reach1[h']) * U0[h,h'] ; v1 symmetric.
            v0 = U0 @ (self.w * reach1)
            v1 = U1.T @ (self.w * reach0)
            return v0, v1
        nid = self._id[id(node)]
        actor = node["actor"]
        sigma = self._sigma(nid)               # n x A
        acts = node["actions"]
        v0 = np.zeros(self.n)
        v1 = np.zeros(self.n)
        child_v_actor = []                     # actor's value per action (for regret)
        for k, (_, child) in enumerate(acts):
            if actor == 0:
                cv0, cv1 = self._walk(child, reach0 * sigma[:, k], reach1, update)
                v0 += sigma[:, k] * cv0
                v1 += cv1
                child_v_actor.append(cv0)
            else:
                cv0, cv1 = self._walk(child, reach0, reach1 * sigma[:, k], update)
                v1 += sigma[:, k] * cv1
                v0 += cv0
                child_v_actor.append(cv1)
        if update:
            node_val = v0 if actor == 0 else v1
            own_reach = reach0 if actor == 0 else reach1
            for k in range(len(acts)):
                self.regret[nid][:, k] += child_v_actor[k] - node_val
            self.strat_sum[nid] += own_reach[:, None] * sigma
        return v0, v1

    def run(self, iters: int = 1000):
        base = np.ones(self.n)                  # uniform reach (combo weights applied at terminal)
        for _ in range(iters):
            self._walk(self.root, base.copy(), base.copy(), update=True)
        return self.average_strategy()

    def average_strategy(self) -> Dict[int, np.ndarray]:
        out = {}
        for nid, ss in self.strat_sum.items():
            s = ss.sum(axis=1, keepdims=True)
            a = ss.shape[1]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[nid] = np.where(s > 0, ss / s, 1.0 / a)
        return out

    # ---- exploitability (best response vs the average strategy) ------------------
    def _pval(self, node, reach_opp, avg, player, best):
        """Player's per-hand value vector: reach_opp is the OPPONENT's (normalized)
        reach to this node. `best`=True -> player best-responds; else plays `avg`."""
        if "term" in node:
            U0, U1 = _terminal_util(node, self.E, self.ip_player, self.realize, self.tb)
            Up = U0 if player == 0 else U1.T
            return Up @ reach_opp              # reach_opp already carries the combo prior (wn)
        nid = self._id[id(node)]
        actor = node["actor"]
        acts = node["actions"]
        if actor == player:                    # player's own decision — no reach scaling
            vals = [self._pval(c, reach_opp, avg, player, best) for _, c in acts]
            stack = np.stack(vals, axis=1)     # n x A
            if best:
                return stack.max(axis=1)
            return (avg[nid] * stack).sum(axis=1)
        v = np.zeros(self.n)                    # opponent's decision — scale their reach
        for k, (_, c) in enumerate(acts):
            v += self._pval(c, reach_opp * avg[nid][:, k], avg, player, best)
        return v

    def exploitability(self, avg) -> float:
        """(BR gain of player 0) + (BR gain of player 1) over the equilibrium value, in
        bb per normalized hand. ~0 means `avg` is a Nash equilibrium."""
        wn = self.w / self.w.sum()             # normalized hand distribution
        gain = 0.0
        for p in (0, 1):
            br = self._pval(self.root, wn, avg, p, best=True) @ wn
            gv = self._pval(self.root, wn, avg, p, best=False) @ wn
            gain += br - gv
        return gain
