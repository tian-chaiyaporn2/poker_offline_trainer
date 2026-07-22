"""Build the self-contained interactive trainer demo (MIT).

Samples real questions from a signed pack and emits a single HTML file that plays
the actual end-user loop — deal a spot, pick an action, get graded + taught why —
with no server. Writes demo/trainer_demo.html + trainer.html (Pages).

Run:  PYTHONPATH=src python demo/build_trainer.py
"""
import base64
import html
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pokertrainer.content_pack import verify_pack

# ---- Locked visual assets: embedded fonts (Rye + Space Mono) and the custom
# paper-craft folded suit SVG symbols. Emitted into the page at build time.
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def _fontface():
    out = []
    for fam, w, fn in [("Rye", 400, "Rye-Regular.ttf"),
                       ("Space Mono", 400, "SpaceMono-Regular.ttf"),
                       ("Space Mono", 700, "SpaceMono-Bold.ttf")]:
        p = os.path.join(_FONT_DIR, fn)
        if not os.path.exists(p):
            continue
        b = base64.b64encode(open(p, "rb").read()).decode()
        out.append("@font-face{font-family:'%s';font-style:normal;font-weight:%d;"
                   "font-display:swap;src:url(data:font/ttf;base64,%s) format('truetype');}" % (fam, w, b))
    return "".join(out)


# suit shapes + per-suit (light, light_hi, dark, dark_lo, lightRegion, foldCurve)
_SPD = 'M50 9 C 61 32, 90 46, 90 64 C 90 78, 79 85, 68 82.5 C 62 81, 57 77, 54.5 72 C 55 79, 58 87, 65 92 L 35 92 C 42 87, 45 79, 45.5 72 C 43 77, 38 81, 32 82.5 C 21 85, 10 78, 10 64 C 10 46, 39 32, 50 9 Z'
_HRT = 'M50 86 C 22 63, 8 48, 8 32 C 8 18, 19 10, 31 10 C 40 10, 47 15, 50 23 C 53 15, 60 10, 69 10 C 81 10, 92 18, 92 32 C 92 48, 78 63, 50 86 Z'
_DIA = 'M50 8 C 58 24, 76 42, 92 50 C 76 58, 58 76, 50 92 C 42 76, 24 58, 8 50 C 24 42, 42 24, 50 8 Z'
_CLB = '<circle cx="50" cy="31" r="19"/><circle cx="30" cy="56" r="19"/><circle cx="70" cy="56" r="19"/><path d="M50 48 C 47 68, 41 83, 30 92 L 70 92 C 59 83, 53 68, 50 48 Z"/>'
_GEOM = {'heart': f'<path d="{_HRT}"/>', 'spade': f'<path d="{_SPD}"/>',
         'diam': f'<path d="{_DIA}"/>', 'club': _CLB}
_SUIT = {
    'spade': ('#34373d', '#454951', '#141519', '#0c0d11', 'M50 11 C 39 30 12 46 12 64 C 12 79 27 85 39 79 C 51 73 58 56 57 42 C 56 30 54 19 50 11 Z', 'M50 11 C 54 19 56 30 57 42 C 58 56 51 73 39 79'),
    'heart': ('#e8232f', '#f6454f', '#a8121d', '#880c15', 'M50 22 C 45 14 38 10 30 10 C 18 10 8 18 8 32 C 8 47 24 63 50 82 C 61 62 61 36 50 22 Z', 'M50 22 C 61 36 61 62 50 82'),
    'diam': ('#f4551b', '#ff7134', '#c31f12', '#9f170c', 'M50 8 C 41 26 22 44 8 50 C 24 57 43 75 50 92 C 61 66 61 34 50 8 Z', 'M50 8 C 61 34 61 66 50 92'),
    'club': ('#334339', '#415448', '#15231c', '#0d1712', 'M48 10 C 40 14 34 22 32 30 C 20 33 10 44 12 56 C 14 70 28 80 42 75 C 52 71 58 58 57 42 C 56 28 54 18 48 10 Z', 'M48 10 C 54 18 56 28 57 42 C 58 58 52 71 42 75'),
}


def _suitdefs():
    d = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
         '<filter id="soft" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.6"/></filter>'
         '<filter id="soft2" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="1.4"/></filter>'
         '<radialGradient id="sheen" cx="0.34" cy="0.26" r="0.62"><stop offset="0" stop-color="#fff" stop-opacity="0.17"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>')
    for sid, v in _SUIT.items():
        light, lhi, dark, dlo, lr, fold = v
        d += f'<clipPath id="cl-{sid}">{_GEOM[sid]}</clipPath><clipPath id="lr-{sid}"><path d="{lr}"/></clipPath>'
        d += f'<linearGradient id="gl-{sid}" x1="0.3" y1="0" x2="0.7" y2="1"><stop offset="0" stop-color="{lhi}"/><stop offset="1" stop-color="{light}"/></linearGradient>'
        d += f'<linearGradient id="gd-{sid}" x1="0.4" y1="0" x2="0.7" y2="1"><stop offset="0" stop-color="{dark}"/><stop offset="1" stop-color="{dlo}"/></linearGradient>'
    for sid, v in _SUIT.items():
        fold = v[5]
        inner = (f'<rect width="100" height="100" fill="url(#gd-{sid})"/>'
                 f'<path d="{fold}" fill="none" stroke="#000" stroke-width="8" opacity="0.42" filter="url(#soft)" transform="translate(2.6 1)"/>'
                 f'<g clip-path="url(#lr-{sid})"><rect width="100" height="100" fill="url(#gl-{sid})"/>'
                 f'<path d="{fold}" fill="none" stroke="#000" stroke-width="4" opacity="0.16" filter="url(#soft2)" transform="translate(-0.6 0)"/></g>'
                 f'<ellipse cx="31" cy="24" rx="34" ry="26" fill="url(#sheen)"/>')
        d += f'<symbol id="sym-{sid}" viewBox="0 0 100 100"><g clip-path="url(#cl-{sid})">{inner}</g></symbol>'
        d += f'<symbol id="so-{sid}" viewBox="0 0 100 100"><g fill="currentColor">{_GEOM[sid]}</g></symbol>'
    d += '</defs></svg>'
    return d

DB = "output/packs/flop_pack_v1_fullrange.db"
RAISE_DB = "output/packs/flop_pack_v1_raise_demo.db"   # reduced-range, but HAS fold/call/raise
TR_DB = "output/packs/flop_pack_turnriver_fullrange.db"  # turn/river decisions (full range; still unconditioned)
SB_DB = "output/packs/flop_pack_sb_vs_bb.db"           # 2nd scenario: SB vs BB (full range)
BTNSB_DB = "output/packs/flop_pack_btn_vs_sb.db"       # BTN vs SB single-raised pot
CO_DB = "output/packs/flop_pack_co_vs_bb.db"           # CO vs BB single-raised pot (pending)
UTG_DB = "output/packs/flop_pack_utg_vs_bb.db"         # UTG vs BB single-raised pot (pending)
HJ_DB = "output/packs/flop_pack_hj_vs_bb.db"           # HJ vs BB single-raised pot (pending)
PER_REASON = 6          # cap questions per reason for variety
MAX_Q = 60
RAISE_Q = 12            # extra 3-action spots blended in to show the raise UX
TR_Q = 16               # turn/river spots blended in
SB_Q = 20               # SB-vs-BB spots blended in (2nd position)
BTNSB_Q = 16            # BTN-vs-SB spots blended in
CO_Q = 16               # CO-vs-BB spots blended in

STREET = {6: "flop", 8: "turn", 10: "river"}       # by board-string length


def _require_verified(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"required pack not found: {path}")
    verdict = verify_pack(path)
    if not (verdict.get("hash_ok") and verdict.get("signature_ok")):
        raise SystemExit(f"pack failed integrity check: {path} verify={verdict}")
    return verdict


def _situation(node, actor, street):
    if street == "flop":
        return SITUATION.get(node, f"You're the {actor}, on the flop.")
    # Unconditioned later-street demo — do not claim a check-through range.
    pre = ("On the turn, " if street == "turn" else "On the river, ")
    if node.endswith("_first"):
        return pre + f"you're the {actor}, first to act."
    if node.endswith("_vs_check"):
        return pre + f"it's checked to you ({actor})."
    return pre + f"you face a bet ({actor})."

SITUATION = {
    "bb_first": "You're the BB, first to act on the flop.",
    "btn_vs_check": "You're the BTN — the BB checked to you.",
    "bb_vs_bet": "You're the BB — you checked and the BTN bet 66% of the pot.",
    "btn_vs_bet": "You're the BTN — the BB led into you for 66% of the pot.",
    "sb_first": "You're the SB, first to act on the flop.",
    "bb_vs_check": "You're the BB — the SB checked to you.",
    "sb_vs_bet": "You're the SB — you checked and the BB bet 66% of the pot.",
}
ALAB = {"check": "Check", "bet": "Bet 66%", "fold": "Fold", "call": "Call", "raise": "Raise 3×"}


def _raise_label_from_pack(path: str) -> str:
    """Derive raise button label from pack config.raise_x when present."""
    try:
        meta = dict(sqlite3.connect(path).execute("SELECT key, value FROM pack_meta"))
        cfg = json.loads(meta.get("config") or "{}")
        rx = cfg.get("raise_x")
        if rx is None:
            return "Raise 3×"
        x = float(rx)
        return f"Raise {x:g}×" if x != int(x) else f"Raise {int(x)}×"
    except Exception:
        return "Raise 3×"
RLAB = {"value": "Value bet", "protection": "Protection", "bluff": "Bluff", "semi_bluff": "Semi-bluff",
        "pot_control": "Pot control", "trap": "Trap", "realization": "Give up / realize equity",
        "value_call": "Value call", "bluff_catch": "Bluff-catch", "call_odds": "Call on odds",
        "raise_value": "Value raise", "raise_bluff": "Bluff raise", "raise_semibluff": "Semi-bluff raise",
        "fold": "Fold", "mixed": "Mixed / close"}


COLS = ("id board node acting_player hand actions ev freq preferred_action "
        "action_grades reason headline detail mixed").split()


def _oop_pos(rows):
    """OOP position = the actor in the '_first' node (works for any scenario)."""
    for d in rows:
        if d["node"].endswith("_first"):
            return d["acting_player"]
    return "BB"


def _ip_pos(rows, oop):
    """IP position = the other of the two seats present in the rows."""
    for d in rows:
        if d["acting_player"] != oop:
            return d["acting_player"]
    return "BTN" if oop != "BTN" else "BB"


def _to_q(d, oop_pos="BB", ip_pos=None):
    from pokertrainer.explanations import freq_pct_ints
    acts = json.loads(d["actions"])
    board = [d["board"][i:i+2] for i in range(0, len(d["board"]), 2)]
    street = STREET.get(len(d["board"]), "flop")
    node = d["node"]
    # Act order is a ROLE (OOP acts first, IP acts last), not tied to the position
    # code — in SB-vs-BB the BB is IP (acts last), the opposite of BTN-vs-BB. (This
    # stays role-based: the situation copy below clarifies "checked, now facing a bet".)
    acts_first = node.endswith("_first") or (
        not node.endswith("_vs_check") and d["acting_player"] == oop_pos)
    freq_raw = {k: float(v) for k, v in json.loads(d["freq"]).items()}
    return {
        "board": board, "hero": [d["hand"][0:2], d["hand"][2:4]], "street": street,
        "node": node, "acting_player": d["acting_player"], "acts_first": acts_first,
        "is_oop": d["acting_player"] == oop_pos,
        # the opponent's actual seat, so the table graphic labels it correctly for
        # every matchup (not just BTN/SB-vs-BB): villain = the seat that isn't the hero.
        "villain": (ip_pos or ("BTN" if oop_pos != "BTN" else "BB"))
        if d["acting_player"] == oop_pos else oop_pos,
        "actions": acts, "labels": {a: ALAB.get(a, a) for a in acts},
        "ev": {k: round(v, 2) for k, v in json.loads(d["ev"]).items()},
        # Largest-remainder ints so the Pro frequency mix always sums to 100.
        "freq": freq_pct_ints(freq_raw, order=acts),
        "preferred": d["preferred_action"], "grades": json.loads(d["action_grades"]),
        "reason": d["reason"], "reason_label": RLAB.get(d["reason"], d["reason"]),
        "headline": d["headline"], "detail": json.loads(d["detail"]),
        # Near-indifferent spots: let feedback treat them as ties, not a punished pick.
        "mixed": bool(d.get("mixed")),
    }


def load_questions():
    verdict = _require_verified(DB)
    c = sqlite3.connect(DB)
    meta = dict(c.execute("SELECT key, value FROM pack_meta").fetchall())
    meta["record_count"] = str(verdict["records"])
    rows = c.execute(f"SELECT {','.join(COLS)} FROM flop_decision").fetchall()
    c.close()
    buckets = defaultdict(list)
    for r in rows:
        d = dict(zip(COLS, r))
        buckets[(d["node"], d["reason"])].append(d)   # balance across BOTH node and reason
    # round-robin across (node, reason) groups so every decision node and every
    # reason type is represented even after the total cap.
    from itertools import zip_longest
    groups = [g[:PER_REASON] for g in buckets.values()]
    picked = [d for tier in zip_longest(*groups) for d in tier if d is not None][:MAX_Q]
    oop = _oop_pos(picked); ip = _ip_pos(picked, oop)
    return meta, [_to_q(d, oop, ip) for d in picked]


def load_raise(n=RAISE_Q, required=True):
    """A few real fold/call/raise spots from the raise-enabled (reduced-range) pack,
    so the trainer demonstrates the 3-action UX until the full-range raise run lands."""
    if not os.path.exists(RAISE_DB):
        msg = f"optional raise pack missing ({RAISE_DB})"
        if required:
            raise SystemExit(msg + " — pass --allow-missing-demo-packs to skip")
        print(f"  warn: {msg} — skipping")
        return []
    _require_verified(RAISE_DB)
    raise_lab = _raise_label_from_pack(RAISE_DB)
    c = sqlite3.connect(RAISE_DB)
    rows = c.execute(f"SELECT {','.join(COLS)} FROM flop_decision "
                     "WHERE actions LIKE '%raise%'").fetchall()
    c.close()
    by_reason = defaultdict(list)
    for r in rows:
        d = dict(zip(COLS, r))
        by_reason[d["reason"]].append(d)
    from itertools import zip_longest
    groups = [g[:3] for g in by_reason.values()]
    picked = [d for tier in zip_longest(*groups) for d in tier if d is not None][:n]
    oop = _oop_pos(picked); ip = _ip_pos(picked, oop)
    out = []
    for q in (_to_q(d, oop, ip) for d in picked):
        q["labels"] = {**q["labels"], "raise": raise_lab}
        q["badge"] = "raise demo"     # flag so the UI can note the reduced-range source
        out.append(q)
    if required and len(out) < 3:
        raise SystemExit(f"raise pack produced only {len(out)} spots (need ≥3)")
    return out


def load_turnriver(n=TR_Q, required=True):
    """Turn + river decisions from the full-range later-street pack (still unconditioned —
    ranges are card-removal only, not filtered by a prior check/check line)."""
    if not os.path.exists(TR_DB):
        msg = f"optional turn/river pack missing ({TR_DB})"
        if required:
            raise SystemExit(msg + " — pass --allow-missing-demo-packs to skip")
        print(f"  warn: {msg} — skipping")
        return []
    _require_verified(TR_DB)
    c = sqlite3.connect(TR_DB)
    rows = c.execute(f"SELECT {','.join(COLS)} FROM flop_decision").fetchall()
    c.close()
    by_key = defaultdict(list)
    for r in rows:
        d = dict(zip(COLS, r))
        street = STREET.get(len(d["board"]), "flop")
        by_key[(street, d["node"], d["reason"])].append(d)
    from itertools import zip_longest
    # Interleave facing-a-bet (Fold/Call/Raise) groups with first-to-act/checked-to
    # (Check/Bet) groups, so the turn/river sample shows a MIX — otherwise one node type
    # fills the whole cap and the raise content (or the check/bet content) never appears.
    items = sorted(by_key.items(), key=lambda kv: kv[0])
    vb = [g[:2] for k, g in items if "vs_bet" in k[1]]
    nb = [g[:2] for k, g in items if "vs_bet" not in k[1]]
    groups = [g for pair in zip_longest(vb, nb) for g in pair if g is not None]
    picked = [d for tier in zip_longest(*groups) for d in tier if d is not None][:n]
    oop = _oop_pos(picked); ip = _ip_pos(picked, oop)
    out = []
    for q in (_to_q(d, oop, ip) for d in picked):
        q["badge"] = q["street"]
        out.append(q)
    streets = {q["street"] for q in out}
    if required and not ({"turn", "river"} <= streets):
        raise SystemExit(f"turn/river pack missing street coverage: {streets}")
    return out


def load_scenario(db, badge, n):
    """Blend in a secondary single-raised-pot scenario (SB-vs-BB, BTN-vs-SB, CO-vs-BB, ...),
    balanced across (node, reason). Positions (OOP/IP + villain seat) are read from the pack
    itself, so the table graphic is correct for any matchup. Missing pack => skipped."""
    if not os.path.exists(db):
        print(f"  note: pack not present ({db}) — skipping {badge} spots")
        return []
    _require_verified(db)
    c = sqlite3.connect(db)
    rows = c.execute(f"SELECT {','.join(COLS)} FROM flop_decision").fetchall()
    c.close()
    dicts = [dict(zip(COLS, r)) for r in rows]
    oop = _oop_pos(dicts); ip = _ip_pos(dicts, oop)
    buckets = defaultdict(list)
    for d in dicts:
        buckets[(d["node"], d["reason"])].append(d)
    from itertools import zip_longest
    groups = [g[:2] for g in buckets.values()]
    picked = [d for tier in zip_longest(*groups) for d in tier if d is not None][:n]
    out = []
    for q in (_to_q(d, oop, ip) for d in picked):
        q["badge"] = badge
        out.append(q)
    return out


def load_sb(n=SB_Q, required=False):
    return load_scenario(SB_DB, "SB vs BB", n)


PF_Q = 16   # pre-flop spots blended in ("Chapter 0")


def load_preflop(n=PF_Q):
    """Pre-flop spots (open/fold + BB defense) from the calibrated ranges — a different
    question KIND the trainer renders on its own path (no board)."""
    from pokertrainer.preflop_content import build_questions
    out = []
    for q in build_questions()[:n]:
        out.append({
            "preflop": True, "badge": "Preflop", "pos": q["pos"],
            "ctx": q.get("ctx"), "opener": q.get("opener"), "tbettor": q.get("tbettor"),
            "hand": q["hand"], "actions": q["actions"],
            "answer": q["answer"], "mixed": q.get("mixed", False), "alt": q.get("alt"),
            "read": q["read"], "why": q["why"], "rule": q["rule"],
        })
    return out


def load_contrast_pool(per_bucket=2):
    """A broader, category-diverse pool drawn from the FULL packs (not just the quiz sample),
    so the "similar hand, opposite play" contrast can find a genuine same-category twin even
    for combos the 96-spot deck doesn't happen to include (e.g. a two-pair that FOLDS on a
    wet board). These never enter the quiz rotation — they only serve as contrasts."""
    from pokertrainer.cards import parse_cards, card_rank
    from pokertrainer.evaluator import evaluate, category_name
    specs = [(DB, None), (SB_DB, "SB vs BB"), (BTNSB_DB, "BTN vs SB"),
             (CO_DB, "CO vs BB"), (UTG_DB, "UTG vs BB"), (HJ_DB, "HJ vs BB"), (TR_DB, "street")]
    buckets, seen, out = defaultdict(int), set(), []
    for db, badge in specs:
        if not os.path.exists(db):
            continue
        try:
            _require_verified(db)
        except SystemExit:
            continue
        c = sqlite3.connect(db)
        dicts = [dict(zip(COLS, r)) for r in
                 c.execute(f"SELECT {','.join(COLS)} FROM flop_decision").fetchall()]
        c.close()
        oop = _oop_pos(dicts); ip = _ip_pos(dicts, oop)
        for d in dicts:
            cs = parse_cards(d["board"]) + parse_cards(d["hand"])
            if len(cs) < 5:
                continue
            cat = category_name(evaluate(cs))
            # Match the JS handRead teaching: "two pair" only counts when the hero's OWN two
            # cards pair the board. A board pair (KK) + one matching hole card is really a single
            # pair, so it must not fill the genuine two-pair bucket used for contrasts.
            if cat == "two pair":
                hr = [card_rank(x) for x in parse_cards(d["hand"])]
                br = [card_rank(x) for x in parse_cards(d["board"])]
                from collections import Counter as _C
                allc = _C(br + hr); bc = _C(br)
                hero_made = [r for r, n in allc.items() if n == 2 and r in hr and bc.get(r, 0) < 2]
                if len(hero_made) < 2:
                    cat = "one pair"
            key = (d["reason"], cat)
            k2 = (d["board"], d["hand"], d["node"])
            if buckets[key] >= per_bucket or k2 in seen:
                continue
            seen.add(k2); buckets[key] += 1
            q = _to_q(d, oop, ip)
            q["badge"] = q["street"] if badge == "street" else badge
            out.append(q)
    return out


def build(allow_missing_demo_packs=False):
    # The full-range pack now includes Fold/Call/Raise on facing-a-bet nodes (FR-011
    # landed), so the reduced-range raise-demo blend is retired.
    meta, qs = load_questions()
    tr_qs = load_turnriver(required=not allow_missing_demo_packs)
    sb_qs = load_sb()
    btnsb_qs = load_scenario(BTNSB_DB, "BTN vs SB", BTNSB_Q)
    co_qs = load_scenario(CO_DB, "CO vs BB", CO_Q)
    utg_qs = load_scenario(UTG_DB, "UTG vs BB", CO_Q)
    hj_qs = load_scenario(HJ_DB, "HJ vs BB", CO_Q)
    pf_qs = load_preflop()
    qs = qs + tr_qs + sb_qs + btnsb_qs + co_qs + utg_qs + hj_qs + pf_qs
    cpool = load_contrast_pool()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "local"
    print(f"  ({len(tr_qs)} turn/river + {len(sb_qs)} SB-vs-BB + {len(btnsb_qs)} BTN-vs-SB + "
          f"{len(co_qs)} CO-vs-BB + {len(utg_qs)} UTG-vs-BB + {len(hj_qs)} HJ-vs-BB + "
          f"{len(pf_qs)} pre-flop spots blended in; {len(cpool)} contrast-pool spots)")
    # Escape </script> so pack strings cannot break out of the inline script.
    data = json.dumps(qs, separators=(",", ":")).replace("<", "\\u003c")
    cdata = json.dumps(cpool, separators=(",", ":")).replace("<", "\\u003c")
    body = TEMPLATE.replace("__DATA__", data).replace("__CPOOL__", cdata).replace("__VERSION__", html.escape(meta.get("version", ""))) \
                   .replace("__RECORDS__", html.escape(str(meta.get("record_count", "")))).replace("__COMMIT__", html.escape(commit)) \
                   .replace("__FONTFACE__", _fontface()).replace("__SUITDEFS__", _suitdefs())
    os.makedirs("demo", exist_ok=True)
    open("demo/trainer_demo.html", "w").write(body)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           '<title>Full-Street Flop Trainer</title>\n'
           '<meta name="description" content="Interactive GTO flop trainer — pick an action, '
           'get graded, learn why.">\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    open("index.html", "w").write(doc)     # Pages landing = the interactive trainer
    print(f"wrote demo/trainer_demo.html + index.html | {len(qs)} questions | "
          f"pack {meta.get('version')} ({meta.get('record_count')} recs) | build {commit}")


TEMPLATE = r'''<style>
__FONTFACE__
/* Locked dark visual design — one committed dark world (Rye + Space Mono + folded suits). */
:root, :root[data-theme="light"], :root[data-theme="dark"]{
  --bg:#0b0c10; --panel:#16171d; --panel2:#1d1f27; --ink:#f2f1ea; --muted:#888e9b; --line:#2a2c35;
  --brass:#8aa0ff; --brass-soft:#5b74ff;
  --best:#2fd08a; --good:#5ee7a8; --accept:#ffc24d; --costly:#ff8a6e; --major:#e0341a;
  --pc-bg:#f5f0e7; --pc-ink:#15171e; --pc-red:#cf1a2c; --pc-line:#e3ddcf;
  --disp:"Rye","Iowan Old Style",Georgia,serif;
  --sans:"Avenir Next","Avenir",system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
  --mono:"Space Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --label:"Space Mono",ui-monospace,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;overflow-x:hidden;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:640px;margin:0 auto;padding:20px 16px 56px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.brand{font-family:var(--disp);font-weight:600;font-size:19px}
.brand .sp{color:var(--brass)}
.score{display:flex;gap:10px;align-items:baseline}
.score .acc{font-family:var(--mono);font-size:18px;color:var(--brass);font-weight:700;font-variant-numeric:tabular-nums}
.score .sbits{font-size:11.5px;color:var(--muted)}
.score .sbits b{font-variant-numeric:tabular-nums}
.cSolid{color:var(--best)}.cOk{color:var(--accept)}.cLeak{color:var(--costly)}
.controls{margin-bottom:14px}
.level{display:flex;align-items:center;gap:11px;flex-wrap:wrap}
.lvl-cap{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700}
.lvl-hint{margin:9px 0 0;font-size:12px;color:var(--muted);line-height:1.45}
.bar-top{height:4px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:9px}
.bar-top>i{display:block;height:100%;background:var(--brass);transition:width .3s}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.sit{padding:7px 15px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:9px;font-family:var(--sans);font-size:13.5px;line-height:1.3}
.pos{font-family:var(--sans);font-size:11px;font-weight:700;letter-spacing:.05em;padding:2px 8px;border-radius:6px;flex:none}
.pos.BB,.pos.SB,.pos.UTG,.pos.HJ,.pos.CO{background:color-mix(in srgb,var(--brass) 20%,transparent);color:var(--brass)}
.pos.BTN{background:color-mix(in srgb,var(--best) 20%,transparent);color:var(--best)}
.demo{margin-left:auto;font-size:9.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--accept);border:1px solid color-mix(in srgb,var(--accept) 45%,var(--line));border-radius:6px;padding:1px 6px}
.felt{background:radial-gradient(120% 130% at 50% -10%,color-mix(in srgb,var(--best) 20%,var(--panel)),var(--panel));padding:8px 16px 8px;text-align:center}
.cap{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.cards{display:flex;gap:8px;justify-content:center}
/* folded-suit playing cards: corner rank (Space Mono) + centered paper-craft suit */
.pc{position:relative;background:linear-gradient(160deg,#f5f0e7,#e7e1d3);border-radius:7px;width:40px;height:54px;
  box-shadow:0 7px 16px -5px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.7);animation:dealIn .5s cubic-bezier(.2,.9,.3,1.25) both}
.pc::after{content:"";position:absolute;inset:3px;border:1px solid rgba(20,25,40,.09);border-radius:5px;pointer-events:none}
.pc .ix{position:absolute;top:3px;left:4px;display:flex;flex-direction:column;align-items:center;line-height:.82;z-index:2}
.pc .ix b{font-family:var(--mono);font-weight:700;font-size:12px}
.pc .ix .mini{width:8px;height:8px;margin-top:1px}
.pc .center{position:absolute;inset:0;display:grid;place-items:center;z-index:1}
.pc .center .psuit{width:21px;height:21px;filter:drop-shadow(0 2px 3px rgba(20,15,10,.4))}
.pc.pc-heart .ix{color:#cf1a2c}.pc.pc-diam .ix{color:#d84a17}.pc.pc-club .ix{color:#2f3d35}.pc.pc-spade .ix{color:#26282e}
.cards .pc:nth-child(2){animation-delay:.08s}.cards .pc:nth-child(3){animation-delay:.16s}.cards .pc:nth-child(4){animation-delay:.24s}.cards .pc:nth-child(5){animation-delay:.32s}
@keyframes dealIn{from{opacity:0;transform:translateY(16px) rotate(-8deg) scale(.9)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.pc{animation:none}}
.hero{margin-top:5px}
.hero .cap{color:var(--brass);font-weight:600}
.acts{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;padding:10px 16px}
.act{appearance:none;font-family:var(--sans);font-size:15px;font-weight:600;color:var(--ink);background:var(--panel2);
  border:1px solid var(--line);border-radius:11px;padding:10px 10px;cursor:pointer;transition:.12s;display:flex;flex-direction:column;gap:2px;align-items:center}
.act .k{font-family:var(--mono);font-size:10px;color:var(--muted);font-weight:400}
.act:hover:not(:disabled){border-color:var(--brass);background:color-mix(in srgb,var(--brass) 8%,var(--panel2));transform:translateY(-1px)}
.act:focus-visible{outline:2px solid var(--brass);outline-offset:2px}
.act:disabled{cursor:default;opacity:.9}
.act.chosen{box-shadow:inset 0 0 0 2px var(--gc,var(--brass))}
.act.g-best{--gc:var(--best)}.act.g-good{--gc:var(--good)}.act.g-acceptable{--gc:var(--accept)}
.act.g-costly{--gc:var(--costly)}.act.g-major_error{--gc:var(--major)}
/* feedback */
.fb{display:none}
/* result = slide-up bottom sheet over the hand */
.fb.on{display:block;position:fixed;left:50%;bottom:0;transform:translateX(-50%);width:100%;max-width:440px;z-index:50;
  background:var(--panel);border:1px solid var(--line);border-bottom:none;border-radius:22px 22px 0 0;
  box-shadow:0 -16px 44px -10px rgba(0,0,0,.75);max-height:86vh;overflow-y:auto;overscroll-behavior:contain;
  animation:sheetup .3s cubic-bezier(.2,.85,.25,1);padding-bottom:10px}
.fb .grab{width:38px;height:4px;border-radius:3px;background:var(--line);margin:9px auto 2px}
/* decision breakdown: the factors behind the play, each with a plain why */
.factors{margin:11px 0 4px;border:1px solid var(--line);border-radius:12px;background:color-mix(in srgb,var(--panel2) 45%,transparent);padding:2px 14px}
.fac{padding:10px 0;border-top:1px solid var(--line)}
.fac:first-child{border-top:none}
.fac-top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.fac-top>span:first-child{min-width:0}
.fac-l{font-family:var(--label);font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:700}
.fac-read{font-weight:700;font-size:14px}
.meter{display:inline-flex;gap:3px;flex:none}
.meter i{width:7px;height:7px;border-radius:50%;background:var(--line)}
.meter i.on{background:var(--best)}
.meter i.on.mid{background:var(--accept)}
.meter i.on.low{background:var(--costly)}
.fac-why{font-size:12.5px;color:var(--muted);line-height:1.45;margin-top:3px}
/* "similar hand, opposite rule" — teach the deciding factor by contrast */
.compare{margin:8px 18px 2px;border:1px solid color-mix(in srgb,var(--brass) 30%,var(--line));border-radius:12px;background:color-mix(in srgb,var(--brass) 6%,transparent);overflow:hidden}
.compare>summary{cursor:pointer;padding:11px 14px;font-family:var(--label);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--brass);list-style:none;display:flex;align-items:center;gap:7px}
.compare>summary::-webkit-details-marker{display:none}
.compare>summary span{font-size:14px}
.compare[open]>summary{border-bottom:1px solid color-mix(in srgb,var(--brass) 22%,var(--line))}
.compare-body{padding:13px 14px;display:flex;flex-direction:column;gap:11px}
.cmp-line{font-size:13.5px;line-height:1.4}
.cmp-line .cw{display:block;font-family:var(--label);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px}
.cmp-play{font-weight:700}
.cmp-play.a-bet,.cmp-play.a-raise{color:var(--best)}
.cmp-play.a-check,.cmp-play.a-call{color:var(--brass)}
.cmp-play.a-fold{color:var(--muted)}
.cmp-hand{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.cmp-cards{display:flex;gap:4px;align-items:center}
.cmp-cards .pc{width:29px;height:40px;border-radius:5px;animation:none;box-shadow:0 3px 7px -3px rgba(0,0,0,.6)}
.cmp-cards .pc::after{inset:2px;border-radius:3px}
.cmp-cards .pc .ix{top:2px;left:2px}
.cmp-cards .pc .ix b{font-size:9px}
.cmp-cards .pc .ix .mini{width:6px;height:6px}
.cmp-cards .pc .center .psuit{width:14px;height:14px}
.cmp-cards .plus{color:var(--muted);font-size:11px}
.cmp-why{font-size:13px;line-height:1.45;background:color-mix(in srgb,var(--panel2) 60%,transparent);border-radius:9px;padding:10px 12px}
.cmp-why b{color:var(--ink)}
.cmp-go{align-self:flex-start;appearance:none;font-family:var(--label);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:var(--brass);background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:8px 14px;cursor:pointer}
.cmp-go:hover{border-color:var(--brass)}
.fb-scrim{position:fixed;inset:0;z-index:40;background:rgba(0,0,0,.55);backdrop-filter:blur(1.5px);animation:rise .25s ease}
@keyframes sheetup{from{transform:translate(-50%,100%)}to{transform:translate(-50%,0)}}
@keyframes rise{from{opacity:0}to{opacity:1}}
@media (prefers-reduced-motion:reduce){.fb.on,.fb-scrim{animation:none}.act:hover:not(:disabled){transform:none}}
.verdict{padding:13px 18px;font-weight:700;font-size:15px;display:flex;align-items:center;gap:9px}
.verdict .dot{width:10px;height:10px;border-radius:50%;background:var(--vc)}
.v-best{--vc:var(--best)}.v-good{--vc:var(--good)}.v-acceptable{--vc:var(--accept)}.v-costly{--vc:var(--costly)}.v-major_error{--vc:var(--major)}
.verdict{color:var(--vc)}
.why{padding:2px 18px 4px}
.reason{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.02em;color:var(--brass);border:1px solid color-mix(in srgb,var(--brass) 40%,var(--line));border-radius:999px;padding:2px 10px;margin-bottom:8px}
.head{margin:0 0 6px;font-size:15px;font-weight:600}
.det{margin:0 0 4px;padding-left:16px;color:var(--muted);font-size:12.5px;display:flex;flex-direction:column;gap:2px}
.det li{font-variant-numeric:tabular-nums}
.mix{padding:6px 18px 14px}
.mix h4{margin:8px 0 8px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}
.row{margin:9px 0}
.rlab{display:flex;justify-content:space-between;align-items:baseline;font-size:13px;margin-bottom:4px;gap:8px}
.rlab .nm{font-weight:600}
.rlab .you{font-size:10px;color:var(--brass);font-weight:700;letter-spacing:.04em}
.rlab .star{color:var(--rc)}
.rlab .num{font-family:var(--mono);font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.track{height:8px;background:var(--line);border-radius:5px;overflow:hidden}
.track>i{display:block;height:100%;border-radius:5px;background:var(--rc,var(--muted))}
.tag{font-size:10px;font-weight:700;padding:1px 6px;border-radius:5px;color:#fff;background:var(--rc)}
.g-best{--rc:var(--best)}.g-good{--rc:var(--good)}.g-acceptable{--rc:var(--accept)}.g-costly{--rc:var(--costly)}.g-major_error{--rc:var(--major)}
.read{margin:0 0 5px;font-size:14.5px;font-weight:600;color:var(--ink)}
.read b{color:var(--brass)}
.stand{margin:0 0 10px;font-size:12.5px;color:var(--muted);line-height:1.45}
.more{appearance:none;background:none;border:none;color:var(--brass);font-family:var(--sans);font-size:12px;font-weight:600;cursor:pointer;padding:2px 0;margin:2px 0 8px}
.more:hover{text-decoration:underline}
.more:focus-visible{outline:2px solid var(--brass);outline-offset:2px;border-radius:3px}
.morebody{margin:0 0 4px}
.rule{margin:8px 0 4px;font-size:12.5px;color:var(--ink);line-height:1.5;padding:8px 11px;border-radius:8px;background:color-mix(in srgb,var(--best) 9%,transparent)}
.rule b{display:block;color:var(--best);text-transform:uppercase;font-size:10px;letter-spacing:.1em;margin-bottom:2px;font-weight:700}
.cost{margin:-2px 0 8px;font-size:12.5px;color:var(--ink);background:color-mix(in srgb,var(--brass) 10%,transparent);border-left:3px solid var(--brass);padding:7px 11px;border-radius:0 7px 7px 0;font-variant-numeric:tabular-nums}
.row.best-row,.row.you-row{padding:6px 9px;border-radius:9px;margin:6px -9px}
.row.best-row{background:color-mix(in srgb,var(--rc) 12%,transparent)}
.row.you-row{box-shadow:inset 0 0 0 1.5px color-mix(in srgb,var(--rc) 55%,var(--line))}
.next{margin:8px 18px 18px;width:calc(100% - 36px);padding:14px;border:none;border-radius:11px;background:var(--brass);color:#fff;font-family:var(--sans);font-size:15px;font-weight:700;cursor:pointer}
.next:hover{filter:brightness(1.06)}.next:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.foot{margin-top:18px;text-align:center;color:var(--muted);font-size:11.5px;line-height:1.6}
.foot code{font-family:var(--mono)}.foot a{color:var(--brass)}
.hint{font-size:11px;color:var(--muted);text-align:center;margin-top:10px}
kbd{font-family:var(--mono);font-size:10.5px;background:color-mix(in srgb,var(--ink) 8%,transparent);border:1px solid var(--line);border-radius:4px;padding:0 4px}
.lang{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;font-size:11.5px;flex:none}
.lang button{appearance:none;border:none;background:transparent;color:var(--muted);padding:5px 11px;cursor:pointer;font-family:var(--sans);font-weight:600;white-space:nowrap}
.lang button.on{background:var(--brass);color:#fff}
.lang button:focus-visible{outline:2px solid var(--brass);outline-offset:2px}
.vocab{font-family:var(--mono);font-size:11.5px;color:var(--brass);font-weight:700;flex:none}
.unlock{margin:0 18px 4px;padding:10px 12px;border-radius:10px;background:color-mix(in srgb,var(--brass) 12%,var(--panel));border:1px solid var(--brass-soft)}
.unlock .ul-row{font-size:12.5px;color:var(--ink);line-height:1.45}
.unlock .ul-row+.ul-row{margin-top:6px}
.glossary{margin-top:16px;border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:0 16px}
.glossary summary{cursor:pointer;font-weight:600;padding:13px 0;font-size:13px;color:var(--brass);list-style:none}
.glossary summary::-webkit-details-marker{display:none}
.glossary summary::before{content:"＋ ";font-family:var(--mono)}
.glossary[open] summary::before{content:"－ "}
.glossary dl{margin:0 0 14px;font-size:12.5px;line-height:1.5}
.glossary dt{font-weight:700;margin-top:9px}
.glossary dd{margin:1px 0 0;color:var(--muted)}
.intro{margin:0 0 16px;border:1px solid var(--brass-soft);border-radius:12px;background:color-mix(in srgb,var(--brass) 6%,var(--panel));padding:0 16px}
.intro summary{cursor:pointer;font-weight:700;padding:12px 0;font-size:13.5px;color:var(--brass);list-style:none;position:relative;padding-right:22px}
.intro summary::-webkit-details-marker{display:none}
.intro summary::after{content:"▸";position:absolute;right:2px;top:12px;color:var(--brass);transition:transform .15s}
.intro[open] summary::after{transform:rotate(90deg)}
.intro p{margin:0 0 10px;font-size:13px;color:var(--ink);line-height:1.6}
.intro p:first-of-type{margin-top:2px}
.intro b{color:var(--ink)}
/* coach (bring-your-own-key) */
.coach{margin-top:14px;border:1px solid var(--line);border-radius:14px;background:var(--panel);overflow:hidden}
.coach>summary{cursor:pointer;font-weight:600;padding:13px 16px;font-size:13px;color:var(--brass);list-style:none;display:flex;align-items:center;gap:8px}
.coach>summary::-webkit-details-marker{display:none}
.coach>summary .byok{margin-left:auto;font-size:9.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:1px 6px}
.coach-body{padding:0 16px 16px}
.coach-note{font-size:11.5px;color:var(--muted);line-height:1.55;margin:0 0 12px}
.coach-note b{color:var(--ink)}
.coach-set{display:flex;flex-direction:column;gap:9px;margin-bottom:12px;padding:12px;background:var(--panel2);border:1px solid var(--line);border-radius:11px}
.coach-set label{display:block;font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.coach-set input,.coach-set select{font-family:var(--sans);font-size:13px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 10px;width:100%}
.coach-set input:focus,.coach-set select:focus{outline:2px solid var(--brass);outline-offset:1px}
.coach-row2{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.coach-save{appearance:none;border:none;background:var(--brass);color:#fff;font-family:var(--sans);font-weight:700;font-size:13px;padding:10px;border-radius:9px;cursor:pointer;margin-top:2px}
.coach-save:hover{filter:brightness(1.06)}
.coach-conn{font-size:11.5px;color:var(--muted);margin:0 0 11px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.coach-conn b{color:var(--ink)}
.coach-conn button{appearance:none;background:none;border:none;color:var(--brass);font-family:var(--sans);font-size:11.5px;font-weight:600;cursor:pointer;padding:0}
.coach-conn button:hover{text-decoration:underline}
.coach-chips{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:11px}
.coach-chips button{appearance:none;font-family:var(--sans);font-size:11.5px;color:var(--ink);background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:6px 11px;cursor:pointer}
.coach-chips button:hover{border-color:var(--brass)}
.coach-log{display:flex;flex-direction:column;gap:9px;margin-bottom:11px;max-height:340px;overflow-y:auto}
.coach-log:empty{display:none}
.cmsg{font-size:13px;line-height:1.5;padding:9px 12px;border-radius:12px;max-width:88%;white-space:pre-wrap;overflow-wrap:anywhere}
.cmsg.user{align-self:flex-end;background:color-mix(in srgb,var(--brass) 16%,var(--panel2));border:1px solid color-mix(in srgb,var(--brass) 30%,var(--line))}
.cmsg.bot{align-self:flex-start;background:var(--panel2);border:1px solid var(--line)}
.cmsg.think{color:var(--muted)}
.cmsg.err{align-self:stretch;max-width:100%;background:color-mix(in srgb,var(--costly) 12%,var(--panel2));border:1px solid color-mix(in srgb,var(--costly) 40%,var(--line));color:var(--ink)}
.coach-ask{display:flex;gap:8px}
.coach-ask input{flex:1;min-width:0;font-family:var(--sans);font-size:13px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px 12px}
.coach-ask input:focus{outline:2px solid var(--brass);outline-offset:1px}
.coach-ask button{appearance:none;border:none;background:var(--brass);color:#fff;font-family:var(--sans);font-weight:700;font-size:13px;padding:0 16px;border-radius:9px;cursor:pointer;flex:none}
.coach-ask button:disabled{opacity:.5;cursor:default}
/* ===== mobile app shell ===== */
.app{max-width:440px;margin:0 auto;min-height:100vh;display:flex;flex-direction:column;position:relative;background:var(--bg)}
.appbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:flex-start;gap:11px;padding:12px 16px;background:color-mix(in srgb,var(--bg) 86%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.pager{display:flex;gap:5px;flex:none}
.revbtn{appearance:none;flex:none;font-family:var(--label);font-size:16px;line-height:1;font-weight:700;color:var(--brass);background:var(--panel2);border:1px solid var(--line);border-radius:9px;width:30px;height:30px;display:grid;place-items:center;cursor:pointer;transition:.12s;padding:0}
.revbtn:hover:not(:disabled){border-color:var(--brass);background:color-mix(in srgb,var(--brass) 10%,var(--panel2))}
.revbtn:disabled{opacity:.3;cursor:default}
.appbar .brand{white-space:nowrap}
.appbar .vocab{margin-left:auto}
.appbar .brand{font-size:18px}
.views{flex:1;padding-bottom:72px}
.view{display:none;padding:14px 15px 8px}
.view.on{display:block;animation:viewin .24s ease}
@keyframes viewin{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.no-motion *{animation:none!important;transition:none!important}
.tabbar{position:fixed;bottom:0;left:50%;transform:translateX(-50%);width:100%;max-width:440px;z-index:30;display:flex;background:color-mix(in srgb,var(--panel) 94%,transparent);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-top:1px solid var(--line);padding-bottom:env(safe-area-inset-bottom)}
.tabbar button{flex:1;appearance:none;border:none;background:none;color:var(--muted);font-family:var(--label);font-size:9px;letter-spacing:.05em;text-transform:uppercase;font-weight:700;padding:9px 4px 11px;display:flex;flex-direction:column;align-items:center;gap:4px;cursor:pointer}
.tabbar button .ti{width:19px;height:19px;font-size:16px;line-height:1;display:grid;place-items:center;color:currentColor}
.tabbar button.on{color:var(--brass)}
.street-seg{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;margin-bottom:6px;scrollbar-width:none}
.street-seg::-webkit-scrollbar{display:none}
.street-seg button{appearance:none;flex:none;font-family:var(--label);font-size:11px;text-transform:uppercase;letter-spacing:.03em;font-weight:700;color:var(--muted);background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:7px 13px;cursor:pointer}
.street-seg button.on{background:var(--brass);color:#0b0c10;border-color:var(--brass)}
.street-seg button:disabled{opacity:.4}
.catcount{display:block;font-family:var(--mono);font-size:10.5px;color:var(--muted);margin:0 2px 5px}
/* progress view */
.phead{margin-bottom:12px}.ptitle{font-family:var(--disp);font-size:26px}
.big-stat{display:flex;align-items:baseline;gap:9px;margin-bottom:14px}
.big-stat .acc{font-family:var(--mono);font-size:40px;color:var(--best);font-weight:700;font-variant-numeric:tabular-nums}
.big-stat span{font-family:var(--label);font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.statrow{display:flex;gap:8px;margin-bottom:22px}
.statrow .stat{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 6px;text-align:center}
.statrow .stat b{display:block;font-family:var(--mono);font-size:18px;font-variant-numeric:tabular-nums}
.statrow .stat span{font-family:var(--label);font-size:8px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.prow-h{font-family:var(--label);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:700;margin-bottom:12px}
.prow{display:flex;align-items:center;gap:9px;margin-bottom:11px}
.prow .pn{font-family:var(--label);font-size:11px;text-transform:uppercase;width:62px;color:var(--ink);font-weight:700}
.prow .pbar{flex:1;height:9px;background:var(--panel2);border-radius:5px;overflow:hidden}
.prow .pbar>i{display:block;height:100%;border-radius:5px;transition:width .5s ease}
.c-pre{background:linear-gradient(90deg,#8aa0ff,#5b74ff)}.c-flop{background:linear-gradient(90deg,#5ee7a8,#2fd08a)}.c-turn{background:linear-gradient(90deg,#ffd67a,#ffb020)}.c-river{background:linear-gradient(90deg,#ff8a6e,#ff6a4d)}
.prow .pv{font-family:var(--mono);font-size:11px;color:var(--muted);width:36px;text-align:right}
/* settings view */
.s-sec{font-family:var(--label);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:700;margin:20px 0 8px}
.seg{display:flex;background:var(--panel2);border-radius:11px;padding:3px;gap:2px;border:1px solid var(--line)}
.seg button{flex:1;appearance:none;border:none;text-align:center;font-family:var(--label);font-size:9px;text-transform:uppercase;letter-spacing:.02em;padding:9px 2px;border-radius:8px;color:var(--muted);font-weight:700;cursor:pointer;background:none}
.seg button.on{background:var(--brass);color:#0b0c10}
.s-row{display:flex;align-items:center;justify-content:space-between;padding:12px 2px;border-bottom:1px solid var(--line);font-size:13px}
.tog{appearance:none;width:40px;height:23px;border-radius:12px;background:var(--line);position:relative;border:none;cursor:pointer;flex:none}
.tog::after{content:"";position:absolute;top:2px;left:2px;width:19px;height:19px;border-radius:50%;background:#fff;transition:.18s}
.tog.on{background:var(--best)}.tog.on::after{left:19px}
.s-reset{appearance:none;width:100%;background:none;border:1px solid var(--line);color:var(--costly);font-family:var(--label);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:13px;border-radius:12px;cursor:pointer;margin-top:16px}
.view .foot{text-align:left;margin-top:6px}
.view .intro,.view .glossary{margin-top:8px}
/* ===== position / situation graphic ===== */
.seats{padding:6px 14px 2px}
/* 6-max ring seen from above: all seats, folded ones dimmed, button on its seat */
.tv{position:relative;height:146px;max-width:320px;margin:0 auto;
  background:radial-gradient(120% 130% at 50% 46%, color-mix(in srgb,var(--best) 15%,var(--panel2)) 0%, color-mix(in srgb,var(--best) 4%,var(--panel)) 60%, var(--panel) 100%);
  border:1px solid color-mix(in srgb,var(--best) 22%,var(--line));border-radius:118px/56px;
  box-shadow:inset 0 2px 16px rgba(0,0,0,.35), inset 0 0 0 6px color-mix(in srgb,var(--best) 6%,transparent)}
.tvs{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:1px;width:72px;text-align:center;line-height:1.12}
.tvs.ab{flex-direction:column-reverse}
.av{width:21px;height:21px;border-radius:50%;position:relative;
  background:radial-gradient(70% 70% at 50% 35%, var(--panel2), var(--panel));border:1.5px solid var(--line)}
.av::after{content:"";position:absolute;left:50%;top:52%;transform:translate(-50%,-50%);width:9px;height:9px;border-radius:50%;
  background:color-mix(in srgb,var(--muted) 50%,transparent)}
.tvs.fold{opacity:.4}
.tvs.fold .av{width:17px;height:17px}
.tvs.opp .av{border-color:var(--brass);box-shadow:0 0 0 3px color-mix(in srgb,var(--brass) 20%,transparent)}
.tvs.opp .av::after{background:color-mix(in srgb,var(--brass) 75%,transparent)}
.tvs.you .av{width:26px;height:26px;border-color:var(--best);box-shadow:0 0 0 3px color-mix(in srgb,var(--best) 20%,transparent)}
.tvs.you .av::after{width:11px;height:11px;background:color-mix(in srgb,var(--best) 78%,transparent)}
.dbtn{position:absolute;right:-5px;bottom:-3px;width:14px;height:14px;border-radius:50%;
  background:#f5f0e7;color:#15171e;font-family:var(--mono);font-weight:700;font-size:9px;
  display:grid;place-items:center;border:1px solid rgba(0,0,0,.4);box-shadow:0 1px 3px rgba(0,0,0,.4);z-index:3}
.nm{font-size:9.5px;font-weight:700;color:var(--ink)}
.tvs.fold .nm{font-size:8.5px;color:var(--muted);font-family:var(--label);letter-spacing:.04em}
.tvs.you .nm{color:var(--best)}
.tvs.opp .nm{color:var(--brass)}
.ps{font-size:8.5px;color:var(--muted)}
.turn{font-family:var(--label);font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--best)}
.tvmid{position:absolute;top:46%;left:50%;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:6px}
.potlab{font-family:var(--label);font-size:8.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);
  border:1px dashed color-mix(in srgb,var(--muted) 40%,transparent);border-radius:999px;padding:3px 13px}
.oppchip{font-family:var(--label);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;
  padding:4px 11px;border-radius:999px;border:1px solid transparent}
.oppchip.a-check{background:color-mix(in srgb,var(--accept) 16%,transparent);color:var(--accept);border-color:color-mix(in srgb,var(--accept) 40%,var(--line))}
.oppchip.a-bet{background:color-mix(in srgb,var(--costly) 16%,transparent);color:var(--costly);border-color:color-mix(in srgb,var(--costly) 40%,var(--line))}
</style>
__SUITDEFS__
<div class="app">
  <div class="appbar">
    <div class="pager"><button class="revbtn" id="prev" type="button" disabled aria-label="Back to the previous hand" title="Previous hand">&#8249;</button><button class="revbtn" id="fwd" type="button" disabled aria-label="Forward to the next hand" title="Next hand">&#8250;</button></div>
    <div class="brand"><span class="sp">&spades;</span> Hold'em Trainer</div>
    <span class="vocab" id="vocab" hidden></span>
  </div>
  <div class="views">
  <section class="view on" id="v-train">
    <div class="street-seg" id="cats" role="group" aria-label="Which part to train">
      <button data-c="all" type="button">All</button><button data-c="preflop" type="button">Pre-flop</button><button data-c="flop" type="button">Flop</button><button data-c="turn" type="button">Turn</button><button data-c="river" type="button">River</button>
    </div>
    <span class="catcount" id="catcount"></span>
    <div class="bar-top"><i id="prog" style="width:0"></i></div>

  <div class="card">
    <div class="sit"><span class="pos" id="pos"></span><span id="sit"></span><span class="demo" id="demotag" hidden>raise demo</span></div>
    <div class="seats" id="seats" hidden></div>
    <div class="felt">
      <div class="cap" id="boardcap">Flop</div>
      <div class="cards" id="board"></div>
      <div class="hero"><div class="cap" id="herocap">Your hand</div><div class="cards" id="hero"></div></div>
    </div>
    <div class="acts" id="acts"></div>
    <div class="fb" id="fb">
      <div class="grab" aria-hidden="true"></div>
      <div class="verdict" id="verdict"></div>
      <div class="unlock" id="unlock" hidden></div>
      <div class="why">
        <p class="read" id="read"></p>
        <span class="reason" id="reason"></span>
        <p class="head" id="head"></p>
        <div class="factors" id="factors" hidden></div>
        <button class="more" id="moretoggle" type="button" hidden>Explain more ▸</button>
        <div class="morebody" id="morebody" hidden>
          <p class="stand" id="stand"></p>
          <p class="rule" id="rule"></p>
        </div>
        <p class="cost" id="cost" hidden></p>
        <ul class="det" id="det"></ul>
      </div>
      <div class="mix"><h4 id="mixhead"></h4><div id="bars"></div></div>
      <details class="compare" id="compare" hidden>
        <summary><span>&#8646;</span> A similar hand plays the opposite way &mdash; see why</summary>
        <div class="compare-body" id="compare-body"></div>
      </details>
      <button class="next" id="next">Next hand &nbsp;&#8629;</button>
    </div>
  </div>

  <details class="coach" id="coach">
    <summary>💬 Ask a coach about this hand<span class="byok">your API key</span></summary>
    <div class="coach-body">
      <p class="coach-note">Ask anything about this spot in plain language. The coach is handed this hand's exact solver
        numbers, so it explains the real decision — not generic tips. It runs on <b>your own API key</b>, kept only on
        this device and sent straight to your provider (we never see it). Use a spend-limited key. <span id="coach-envnote"></span></p>
      <div class="coach-set" id="coach-set" hidden>
        <div><label for="coach-prov">Provider</label><select id="coach-prov"></select></div>
        <div class="coach-row2">
          <div><label for="coach-model">Model</label>
            <input id="coach-model" list="coach-models" autocomplete="off" spellcheck="false"></div>
          <div><label for="coach-key">API key</label>
            <input id="coach-key" type="password" autocomplete="off" spellcheck="false"></div>
        </div>
        <datalist id="coach-models"></datalist>
        <button class="coach-save" id="coach-save" type="button">Save &amp; connect</button>
      </div>
      <p class="coach-conn" id="coach-conn" hidden></p>
      <div class="coach-chips" id="coach-chips"></div>
      <div class="coach-log" id="coach-log"></div>
      <div class="coach-ask">
        <input id="coach-input" placeholder="Ask about this hand…" autocomplete="off">
        <button id="coach-send" type="button">Send</button>
      </div>
    </div>
  </details>

  <div class="hint" id="hint">Pick with <kbd>1</kbd><kbd>2</kbd> · next <kbd>Enter</kbd> · back <kbd>←</kbd></div>
  </section>

  <section class="view" id="v-progress">
    <div class="phead"><div class="ptitle">Progress</div></div>
    <div class="big-stat"><b class="acc" id="acc">—</b><span>overall accuracy</span></div>
    <div class="statrow">
      <div class="stat"><b id="n">0</b><span>Played</span></div>
      <div class="stat"><b class="cSolid" id="solid">0</b><span>Solid</span></div>
      <div class="stat"><b class="cOk" id="ok">0</b><span>OK</span></div>
      <div class="stat"><b class="cLeak" id="leak">0</b><span>Leak</span></div>
    </div>
    <div class="prow-h">Mastery by street</div>
    <div id="mastery"></div>
  </section>

  <section class="view" id="v-settings">
    <div class="phead"><div class="ptitle">Settings</div></div>
    <div class="s-sec">Language</div>
    <div class="seg" id="lang" role="group" aria-label="Language level">
      <button data-m="progressive" type="button">Adaptive</button><button data-m="plain" type="button">Beginner</button><button data-m="learning" type="button">Learning</button><button data-m="poker" type="button">Pro</button>
    </div>
    <p class="lvl-hint" id="levelhint"></p>
    <div class="s-sec">App</div>
    <div class="s-row"><span>Reduce motion</span><button class="tog" id="tog-motion" type="button" aria-label="Reduce motion"></button></div>
    <div class="s-sec">New to poker?</div>
    <details class="intro" id="intro">
      <summary>🔰 The 30-second primer</summary>
      <p>You and one opponent each get <b>2 secret cards</b> (only you see yours). Then <b>5 shared cards</b>
        everyone can use are dealt in stages: <b>3 at once</b>, then a <b>4th</b>, then a <b>5th</b>. You make your
        best five-card hand from your 2 cards plus the shared ones.</p>
      <p>At each stage you choose: <b>Check</b> (pass, bet nothing), <b>Bet</b> (put chips in), <b>Call</b>
        (match a bet), <b>Raise</b> (bet even more), or <b>Fold</b> (give up the hand). The chips already in the
        middle are the <b>pot</b> — that's what you're playing for.</p>
      <p>One player <b>acts first</b>, the other <b>acts last</b>. Acting last is an advantage — you see what your
        opponent does before deciding. The trainer tells you which you are each hand.</p>
      <p>Your job: pick the action a strong player would — graded instantly, told why. The <b>Adaptive</b> setting
        starts in plain words and <b>teaches you the poker terms as you play them well</b>.</p>
    </details>
    <div class="s-sec">About</div>
    <div class="foot">
    Real solver output — pack <code>__VERSION__</code>, <b>__RECORDS__</b> signed records, build <code>__COMMIT__</code>.
    Every grade &amp; explanation is computed from a full flop&rarr;turn&rarr;river solve; nothing is hand-written.<br>
    Flop spots — including Fold/Call/Raise when you face a bet — come from the full-range pack.
    <span class="demo">Turn / river</span> spots are now full-range solver output too, but
    <b>unconditioned</b> (ranges are card-removal only, not filtered by a prior check/check line),
    and facing-a-bet there is Fold/Call — the raise pass on those streets is the next depth work.
    Pre-flop spots are calibrated ranges (solver-approximate, tuned to standard frequencies).<br>
    Prefer to review the answers at a glance? See the <a href="preview.html">content gallery</a>.
  </div>
  <div class="s-sec">Poker terms</div>
  <details class="glossary">
    <summary>Poker terms — tap to learn the lingo</summary>
    <dl>
      <dt>The board / shared cards</dt>
      <dd>The cards in the middle everyone can use — up to 5 of them.</dd>
      <dt>Flop, turn, river</dt>
      <dd>The stages the shared cards arrive in: the flop is the first 3, the turn is the 4th, the river is the 5th (last).</dd>
      <dt>Your hand (hole cards)</dt>
      <dd>Your 2 secret cards that only you can see.</dd>
      <dt>Pot</dt>
      <dd>The chips already in the middle — what you're playing to win.</dd>
      <dt>Blinds (small / big)</dt>
      <dd>Forced bets two players post before the cards, so there's always something to play for.</dd>
      <dt>Check · Bet · Call · Raise · Fold</dt>
      <dd>Pass (no bet) · put chips in · match a bet · bet even more · give up the hand.</dd>
      <dt>Position — Button (BTN), Big Blind (BB), Small Blind (SB)</dt>
      <dd>Where you sit. After the flop, the player in position (usually the Button) acts last — an advantage; out of position acts first. In blind-vs-blind, the BB is in position.</dd>
      <dt>The seats, first to last — UTG, Hijack (HJ), Cutoff (CO), Button (BTN), then the blinds (SB, BB)</dt>
      <dd>UTG acts first pre-flop (tightest, most players still behind); the Button acts last (widest). "The Hijack" and "the Cutoff" are the two seats just before the Button.</dd>
      <dt>In / out of position</dt>
      <dd>Whether you act last (in position) or first (out of position) on each street.</dd>
      <dt>Open (raise first-in / RFI)</dt>
      <dd>The first player to enter the pot by raising, pre-flop, when everyone before folded.</dd>
      <dt>3-bet / 4-bet</dt>
      <dd>A re-raise. The blind's raise is the "1st bet"; the open is the 2nd; re-raising the open is the <b>3-bet</b>; re-raising that is the <b>4-bet</b>.</dd>
      <dt>C-bet (continuation bet)</dt>
      <dd>Betting the flop after you were the one who raised before it.</dd>
      <dt>Value bet</dt>
      <dd>Betting a strong hand to get called by weaker ones.</dd>
      <dt>Bluff / semi-bluff</dt>
      <dd>Betting a weak hand to make better hands fold. A semi-bluff is a draw that can still improve.</dd>
      <dt>Bluff-catch</dt>
      <dd>Calling with a medium hand mainly to beat the times they're bluffing.</dd>
      <dt>Pot control</dt>
      <dd>Checking a decent-but-not-great hand to keep the pot small.</dd>
      <dt>Trap</dt>
      <dd>Checking a very strong hand to let opponents catch up or bluff into you.</dd>
      <dt>EV (expected value)</dt>
      <dd>Your average profit from a play, measured in big blinds (bb).</dd>
      <dt>Range</dt>
      <dd>All the different hands a player could have in this exact spot.</dd>
      <dt>Pot odds</dt>
      <dd>The price you're getting to call, compared with the size of the pot.</dd>
    </dl>
  </details>
    <button class="s-reset" id="reset-btn" type="button">Reset progress</button>
  </section>
  </div>
  <div class="fb-scrim" id="fb-scrim" hidden></div>
  <nav class="tabbar" id="tabbar">
    <button data-v="train" class="on"><svg class="ti"><use href="#so-spade"/></svg>Train</button>
    <button data-v="progress"><span class="ti">&#9636;</span>Progress</button>
    <button data-v="settings"><span class="ti">&#9776;</span>Settings</button>
  </nav>
</div>
<script>
const Q = __DATA__;
// Extra spots (from the full packs) used ONLY as "similar hand" contrasts, never in the quiz.
const CPOOL = __CPOOL__;
const ALLSPOTS = Q.concat(CPOOL);
const SUIT = {s:["♠",0],h:["♥",1],d:["♦",1],c:["♣",0]};

// Plain-English hand reader — tells the player WHAT they hold and where they stand
// (top pair / overpair / a set / just a draw). This is the piece beginners lack:
// they can't yet read their own hand, so every "why" falls flat. Computed live from
// the hero cards + board; validated against the pack's real hands.
const RV={2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,T:10,J:11,Q:12,K:13,A:14};
const ONE={2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine",10:"Ten",11:"Jack",12:"Queen",13:"King",14:"Ace"};
const MANY={2:"Twos",3:"Threes",4:"Fours",5:"Fives",6:"Sixes",7:"Sevens",8:"Eights",9:"Nines",10:"Tens",11:"Jacks",12:"Queens",13:"Kings",14:"Aces"};
function hasStraight(vals){const s=new Set(vals);if(s.has(14))s.add(1);
  for(let lo=1;lo<=10;lo++){let ok=true;for(let k=0;k<5;k++)if(!s.has(lo+k)){ok=false;break;}if(ok)return true;}return false;}
function straightDraw(vals){if(hasStraight(vals))return null;const base=new Set(vals);const comp=[];
  for(let r=2;r<=14;r++){if(base.has(r))continue;if(hasStraight([...base,r]))comp.push(r);}
  if(!comp.length)return null;return comp.length>=2?"an open-ended straight draw (four to a straight, open both ends)":"a gutshot straight draw (four to a straight, needs one card in the middle)";}
function handRead(hero,board){
  const hs=hero.map(c=>c[1]),bs=board.map(c=>c[1]);
  const hv=hero.map(c=>RV[c[0]]),bv=board.map(c=>RV[c[0]]);
  const allV=[...hv,...bv],allS=[...hs,...bs],river=board.length>=5;
  const cnt={};allV.forEach(v=>cnt[v]=(cnt[v]||0)+1);
  const groups=Object.keys(cnt).map(Number).sort((a,b)=>cnt[b]-cnt[a]||b-a);
  const suitCnt={};allS.forEach(s=>suitCnt[s]=(suitCnt[s]||0)+1);
  const flushSuit=Object.keys(suitCnt).find(s=>suitCnt[s]>=5),flush=!!flushSuit;
  const straight=hasStraight(allV),maxB=Math.max(...bv),sortB=[...new Set(bv)].sort((a,b)=>b-a),pocket=hv[0]===hv[1];
  // A straight flush needs 5 consecutive cards OF THE SAME SUIT — a flush and a
  // straight made from different cards is not one.
  const sflush=flush&&hasStraight(allV.filter((_,i)=>allS[i]===flushSuit));
  const bCnt={};bv.forEach(v=>bCnt[v]=(bCnt[v]||0)+1);
  const boardPaired=Object.keys(bCnt).some(v=>bCnt[v]>=2);
  // Board coordination: how likely the SHARED cards make a straight/flush for anyone.
  // 0 = dry, 1 = possible, 2 = very live (four to a straight / four-flush board).
  const bvu=[...new Set(bv)];if(bvu.includes(14))bvu.push(1);
  let strun=0;for(let lo=1;lo<=10;lo++){let k=0;for(const v of bvu)if(v>=lo&&v<lo+5)k++;if(k>strun)strun=k;}
  const boardStraighty=strun>=4?2:(strun>=3?1:0);
  const bsuit={};bs.forEach(s=>bsuit[s]=(bsuit[s]||0)+1);
  const bmax=Math.max(0,...Object.values(bsuit));
  const boardFlushy=bmax>=4?2:(bmax>=3?1:0);
  // Board alone already makes the hand (river): everyone shares it unless hole cards beat it.
  const boardFlushAlone=river&&bmax>=5;
  const boardStraightAlone=river&&hasStraight(bv);
  const top=cnt[groups[0]],second=cnt[groups[1]]||0;
  let made,strength=null,cat="high",pairKind=null,overs=[];
  // A pair/trips the hero doesn't contribute to is the shared board, not their hand —
  // teach it by what the hero actually holds, not the board's rank.
  const air=()=>{made=ONE[Math.max(...hv)]+" high (no pair)";cat="high";};
  function pairOf(pr){
    made="a pair of "+MANY[pr];cat="pair";
    overs=[...new Set(bv.filter(v=>v>pr))].sort((a,b)=>b-a);
    if(pocket&&hv[0]===pr){if(pr>maxB){pairKind="over";strength="an overpair (higher than every shared card)";}
      else{pairKind="under";strength="the "+ONE[maxB]+" among the shared cards outranks it";}}
    else if(pr===sortB[0]){pairKind="top";strength="top pair (you matched the highest shared card)";}
    else if(pr===sortB[1]){pairKind="mid";strength="middle pair";}
    else{pairKind="low";strength="a low pair";}
  }
  if(sflush){made="a straight flush";cat="sflush";}
  else if(top===4){made="four of a kind ("+MANY[groups[0]]+")";cat="quads";}
  else if(top===3&&second>=2){made="a full house";cat="full";}
  else if(flush){made=boardFlushAlone?"a flush (the board is already a flush)":"a flush";cat="flush";}
  else if(straight){made=boardStraightAlone?"a straight (the board is already a straight)":"a straight";cat="straight";}
  else if(top===3){made=(pocket&&hv[0]===groups[0])?"a set of "+MANY[groups[0]]+" (three of a kind)":"three "+MANY[groups[0]]+" (three of a kind)";cat="trips";}
  else if(top===2&&second===2){
    // GENUINE two pair = the hero's OWN two cards pair two board cards. If one of the pairs is
    // really a board pair (KK on the board) and the hero only holds one matching card, they
    // have a single pair, not two pair — teach that (same idea as a pocket pair on a paired
    // board). heroMade = paired ranks the hero holds that aren't already a board pair.
    const prRanks=groups.filter(function(g){return cnt[g]===2;});
    const heroMade=prRanks.filter(function(r){return hv.indexOf(r)>=0&&(bCnt[r]||0)<2;});
    if(heroMade.length>=2){made="two pair";cat="twopair";}       // both hole cards work
    else if(heroMade.length===1)pairOf(heroMade[0]);             // one real pair alongside a board pair
    else{made="two pair";cat="twopair";}                         // double-paired board — hero rides it
  }
  else if(top===2){const pr=groups[0];const heroInPair=pocket?hv[0]===pr:hv.includes(pr);
    if(!heroInPair)air();    // board-pair-only — hero holds none of that rank
    else pairOf(pr);}
  else air();
  let draw=null;
  if(!river&&!flush&&!straight){const parts=[];
    if(Object.keys(suitCnt).some(s=>suitCnt[s]===4&&hs.includes(s)))parts.push("a flush draw (four to a flush)");
    // Only the hero's draw — if the board alone already forms the straight draw, it's shared, not yours.
    const sd=straightDraw(allV);if(sd&&!straightDraw(bv))parts.push(sd);
    if(parts.length)draw=parts.join(" and ");}
  return {made,strength,draw,cat,pairKind,overs,boardStraighty,boardFlushy,boardFlushAlone,boardStraightAlone};
}
// "Where you stand" — plain relative strength: what you beat and what beats you.
// The single most important read for a beginner; hedged so it stays true regardless
// of the exact board (a set only loses to a straight/flush "if the board allows it").
function nm(v){const w=ONE[v];return (/^[AE]/.test(w)?"an ":"a ")+w;}   // an Ace, an Eight
function orList(vals){const a=vals.map(nm);
  if(a.length<=1)return a[0]||"";
  if(a.length===2)return a[0]+" or "+a[1];
  return a.slice(0,-1).join(", ")+", or "+a[a.length-1];}
function standingText(rd){
  // What the coordinated shared cards let opponents have (straights/flushes).
  const dg=[];
  if(rd.boardStraighty>=2)dg.push("the shared cards are four to a straight, so a straight is very much in play");
  else if(rd.boardStraighty===1)dg.push("a straight is possible");
  if(rd.boardFlushy>=2)dg.push("four to a flush is out there");
  else if(rd.boardFlushy===1)dg.push("a flush is possible");
  const danger=dg.length===1?dg[0]:dg.length?dg.slice(0,-1).join(", ")+" and "+dg[dg.length-1]:"";
  const high=rd.boardStraighty>=2||rd.boardFlushy>=2;
  // Board alone is already the made hand — do not claim nut-level strength.
  if(rd.boardFlushAlone&&(rd.cat==="flush"||rd.cat==="high"||rd.cat==="pair")){
    return "The board itself is a flush — everyone shares it. Your hole cards only help if they make a better flush (or a full house). Any higher flush card still beats you.";
  }
  if(rd.boardStraightAlone&&(rd.cat==="straight"||rd.cat==="high"||rd.cat==="pair")){
    return "The board itself is a straight — everyone shares it unless your hole cards make a better straight (or a flush/full house).";
  }
  // A very coordinated board turns even a big made hand vulnerable — state the risk
  // without prescribing "don't value-bet" (the solver may still prefer a bet/raise).
  if(high&&(rd.cat==="pair"||rd.cat==="twopair"||rd.cat==="trips")){
    const nm=rd.cat==="trips"?"three of a kind":rd.cat==="twopair"?"two pair":"a pair";
    return "Careful — even with "+nm+", "+danger+", so made hands are vulnerable here.";
  }
  let base;
  switch(rd.cat){
    case "pair":
      base=rd.pairKind==="over"?"You're ahead of every worse pair and all the bluffs — mostly only two pair or three of a kind beats you now."
        :rd.pairKind==="top"?"You beat worse pairs and the hands still chasing — a bigger side card, two pair, or three of a kind has you beat."
        :"You beat high cards and bluffs, but "+orList(rd.overs)+" makes a better pair, and two pair or three of a kind is ahead too.";
      break;
    case "twopair":base="You're ahead of every one-pair hand — mainly three of a kind or better beats you.";break;
    case "trips":base="Very strong — only a straight, flush, or full house could beat you, and only if the shared cards line up for it.";break;
    case "straight":return "A big hand — only a flush or a full house beats you here.";
    case "flush":return "A big hand — only a full house or better beats you.";
    case "full":case "quads":case "sflush":return "You've got a monster — just about nothing beats this.";
    default:return rd.draw
      ? "Nothing made yet, but your hand can still improve — for now you're behind any pair."
      : "No pair yet — you're behind almost any hand that's already made a pair or better; you'd need to improve or get them to fold.";
  }
  if(danger)base+=" Note "+danger+", so play it a touch cautiously.";   // moderate board danger
  return base;
}

// Plain (no jargon) vs Poker (real terminology) — the same data, two vocabularies.
const TERMS = {
  poker:{
    pos:{BTN:"BTN",BB:"BB",SB:"SB"},
    act:{check:"Check",bet:"Bet 66%",fold:"Fold",call:"Call",raise:"Raise 3×"},
    reason:{value:"Value bet",protection:"Protection",bluff:"Bluff",semi_bluff:"Semi-bluff",
      pot_control:"Pot control",trap:"Trap",realization:"Give up / realize equity",value_call:"Value call",
      bluff_catch:"Bluff-catch",call_odds:"Call on odds",raise_value:"Value raise",raise_bluff:"Bluff raise",
      raise_semibluff:"Semi-bluff raise",fold:"Fold",mixed:"Mixed / close"},
    ev:"EV",boardcap:{flop:"Flop",turn:"Turn",river:"River"},herocap:"Your hand"},
  plain:{
    pos:{BTN:"You act last",BB:"You act first",SB:"You act first"},
    act:{check:"Check (pass, no bet)",bet:"Bet (put chips in)",fold:"Fold (give up the hand)",
      call:"Call (match their bet)",raise:"Raise (bet even more)"},
    reason:{value:"Bet a strong hand to get paid",protection:"Bet so hands hoping to improve have to pay",
      bluff:"Bet a weak hand to make them give up",semi_bluff:"Bet a hand that can still improve",
      pot_control:"Just check to keep the pot small",trap:"Check a very strong hand to trap them",
      realization:"Check a weak hand and see the next card free",value_call:"Call — you're probably ahead",
      bluff_catch:"Call — you beat the hands they'd bluff with",call_odds:"Call — cheap enough to keep going",
      raise_value:"Raise a strong hand to build the pot",raise_bluff:"Raise as a bluff to make them fold",
      raise_semibluff:"Raise a hand that can improve",fold:"Fold — not strong enough to continue",
      mixed:"It's close — any choice is fine"},
    ev:"profit",
    boardcap:{flop:"The 3 shared cards",turn:"The 4th shared card is out",river:"The last (5th) shared card"},
    herocap:"Your 2 cards (only you can see these)"},
  learning:{
    pos:{BTN:"BTN",BB:"BB",SB:"SB"},
    act:{check:"Check",bet:"Bet 66%",fold:"Fold",call:"Call",raise:"Raise 3×"},
    reason:{value:"Value bet — get paid by worse hands",protection:"Protection — charge the hands still chasing a card",
      bluff:"Bluff — make better hands fold",semi_bluff:"Semi-bluff — bet a hand that can still improve",
      pot_control:"Pot control — keep the pot small",trap:"Trap — check a very strong hand to hide it",
      realization:"Realize equity — check and take a free card to improve",value_call:"Value call — you're ahead",
      bluff_catch:"Bluff-catch — you beat the hands they'd bluff with",call_odds:"Call on odds — the price is right to chase",
      raise_value:"Value raise — build the pot",raise_bluff:"Bluff raise — make them fold",
      raise_semibluff:"Semi-bluff raise — raise a hand that can improve",fold:"Fold — not strong enough",mixed:"Mixed — any is fine"},
    ev:"EV",boardcap:{flop:"Flop (first 3 shared cards)",turn:"Turn (4th card)",river:"River (5th card)"},
    herocap:"Your hand (your 2 private cards)"}
};
let order=[], pos=0, answered=false, cur=null, chosen=null, stats={n:0,solid:0,ok:0,leak:0,street:{}};
// Hand history so you can step BACK and re-read a hand you already answered (with its
// result), then step forward again. hist = [{qi, pick}] in the order shown; hidx = cursor.
let hist=[], hidx=-1;
function trackStreet(hit){const k=qcat(cur);const s=stats.street[k]||(stats.street[k]={n:0,hit:0});s.n++;if(hit)s.hit++;}
let mode=(function(){try{const m=localStorage.getItem("lang");return (m==="poker"||m==="learning"||m==="plain"||m==="progressive")?m:"progressive";}catch(e){return "progressive";}})();
// Which part to train (pre-flop / flop / turn / river / all).
let cat=(function(){try{const c=localStorage.getItem("cat");return ["all","preflop","flop","turn","river"].includes(c)?c:"all";}catch(e){return "all";}})();
// Adaptive mode: each concept shows in plain words until you've EARNED it (played a
// spot that uses it well); then it graduates to the poker term + its meaning.
let learned=(function(){try{return new Set(JSON.parse(localStorage.getItem("learned")||"[]"));}catch(e){return new Set();}})();
// "Explain more" (where-you-stand + rule of thumb) — default closed for a calm view,
// remembered once a reader opens it so they don't re-tap every hand.
let moreOpen=(function(){try{return localStorage.getItem("moreOpen")==="1";}catch(e){return false;}})();
const VOCAB_TOTAL=2+Object.keys(TERMS.poker.reason).length;
function eff(term){return mode!=="progressive"?mode:(learned.has(term)?"learning":"plain");}
function T(){return TERMS[eff("streets")];}
function posLabel(q){const m=eff("positions");
  if(m==="plain")return q.acts_first?"You act first":"You act last";
  return (TERMS[m].pos[q.acting_player]||q.acting_player);}
function actLabel(a){const m=eff("positions");
  if(m!=="plain"&&cur&&cur.labels&&cur.labels[a])return cur.labels[a];  // per-pack bet/raise sizing
  return (TERMS[m].act[a]||a);}
// Short, jargon-free action names for the verdict/cost sentences (the verbose plain
// labels like "Check (pass, no bet)" are for the buttons, not for prose).
const ACT_SHORT={check:"Check",bet:"Bet",fold:"Fold",call:"Call",raise:"Raise"};
function shortAct(a){return ACT_SHORT[a]||a;}
function fmtEv(v){return (v>=0?"+":"")+v;}
function reasonLabel(r){return (TERMS[eff("reason:"+r)].reason[r]||r);}
// Beginner "why": explains the LOGIC of the play in plain words, not just names the
// action. Replaces the generic per-reason phrase for plain mode; Learning keeps the
// term-tagged phrase, Pro keeps the solver's baked headline.
const PLAIN_HEAD={
  value:"You're ahead of the hands that would call — bet so the weaker ones pay you off.",
  protection:"You're often ahead now, but cards could still come that beat you — bet so the chasing hands have to pay.",
  bluff:"Your hand probably can't win if you both check to the end — so bet to make better hands give up.",
  semi_bluff:"Betting can make better hands fold now — and if you're called, your hand can still improve to the best.",
  pot_control:"A decent hand, but not strong enough to build a big pot — check to keep it small and cheap.",
  trap:"You're very strong here — checking hides it and lets your opponent bluff or catch up before you pounce.",
  realization:"Not much yet — check to see the next card for free instead of throwing chips in.",
  value_call:"You're ahead of enough of their betting hands — call to keep collecting from the worse ones.",
  bluff_catch:"Your hand beats the ones they'd bluff with — call to catch those bluffs.",
  call_odds:"Your hand isn't made yet but could improve — it's cheap enough to call and try to get there.",
  fold:"There isn't enough here to keep going — fold and save your chips for a better spot.",
  raise_value:"You're strong — raise to build a bigger pot while the worse hands pay.",
  raise_bluff:"Raising tells the story of a big hand — do it to pressure them into folding.",
  raise_semibluff:"Raise: you can fold out better hands now, and still improve if they call.",
  mixed:"This one's genuinely close — any play is fine here."
};
// River-specific "why": no more cards, so nothing about drawing / free cards / improving.
const RIVER_PLAIN={
  realization:"Nothing worth betting, and there are no more cards — just check and see who wins.",
  semi_bluff:"You can't improve anymore — but you can still bet to make better hands fold.",
  call_odds:"It's the last card — a cheap call to see who wins.",
  raise_semibluff:"Nothing left to draw to — this raise is a bluff to pressure them into folding.",
  protection:"The board is complete — bet to get paid by worse hands; nothing can catch up now.",
  // no more cards on the river, so "let them catch up" doesn't apply — checking induces a bet/bluff.
  trap:"You're very strong — checking hides it so they bet into you (or bluff) instead of folding to a bet; then you take their chips."
};
// Learning-mode river tags (short term + meaning; flop learning copy still says "improve").
const RIVER_LEARNING={
  realization:"Realize equity — check; there are no more cards to come",
  semi_bluff:"Bluff — no more cards, so this bet only works if they fold",
  call_odds:"Call — the price is right to see the showdown",
  raise_semibluff:"Bluff raise — nothing left to draw to; pressure them to fold",
  protection:"Thin value / denial — charge worse hands on a completed board",
  trap:"Trap — check a monster to induce a bet; nothing can catch up"
};
// A "mixed" (near-tie) spot only says "any play is fine" by default — but when one of the
// tied plays is an aggressive line with a weak hand, that reads as a contradiction ("why is
// raising with Queen-high as good as folding?"). This spells out the missing half: the
// aggressive line is a bluff/semi-bluff, and its fold-equity is what makes it break even with
// giving up. Keyed off the actual co-best actions + the real holding.
// Infer the *logic* of an action from the hand + the action itself, so a "mixed" spot (where
// the solver only tags it "close") can still say WHY the best play edges ahead.
function inferReason(q,rd){
  const p=q.preferred,made=rd.cat!=="high",strong=handTier(rd)>=4;
  if(p==="raise")return !made?(rd.draw?"raise_semibluff":"raise_bluff"):(strong?"raise_value":"raise_semibluff");
  if(p==="bet")return !made?(rd.draw?"semi_bluff":"bluff"):(strong?"value":"protection");
  if(p==="check")return strong?"trap":(made?"pot_control":"realization");
  // No-pair calls are bluff-catches, not "odds to chase" — only draws chase.
  if(p==="call")return made?"value_call":(rd.draw?"call_odds":"bluff_catch");
  if(p==="fold")return "fold";
  return null;
}
function closeExplain(q,rd){
  const pa=shortAct(q.preferred),r=inferReason(q,rd);
  // why the best play edges ahead — reuse the plain per-reason logic (river-aware).
  const why=(q.street==="river"&&RIVER_PLAIN[r])?RIVER_PLAIN[r]:(r&&PLAIN_HEAD[r])?PLAIN_HEAD[r]:"the options are all near break-even here.";
  const alt=q.actions.filter(function(a){return a!==q.preferred;})
    .sort(function(a,b){return (q.ev[b]||0)-(q.ev[a]||0);});
  const altTxt=alt.length?" "+cap1(shortAct(alt[0]))+" is fine too — it's nearly a coin-flip, so don't sweat which you pick.":"";
  return "It's genuinely close, but "+pa+" edges it out. "+why+altTxt;
}
// True when a "trap"/"value" made hand should be reframed as a bluff-catcher: a very
// coordinated board AND this is actually a check-or-bet decision (so "check to keep the pot
// small" is a real option — never fire on a facing-a-bet fold/call/raise node).
function bcReframe(q,rd){
  return (q.reason==="trap"||q.reason==="value")
    &&(rd.boardStraighty>=2||rd.boardFlushy>=2)
    &&(rd.cat==="pair"||rd.cat==="twopair"||rd.cat==="trips")
    &&q.actions.indexOf("check")>=0&&q.actions.indexOf("bet")>=0;
}
function plainHead(q){
  // A "trap"/"value" framing overclaims on a very coordinated board — the made hand is
  // really a bluff-catcher there, so betting bloats the pot into likely straights/flushes.
  const rd=handRead(q.hero,q.board);
  if(bcReframe(q,rd)){
    let s="The board is coordinated — a straight or flush is very possible — so your hand is more of a bluff-catcher than a monster. Check to keep the pot small and take a cheap showdown rather than bet into the hands that beat you.";
    // If it was checked to you, name the trap: a check does NOT rule out the straight here.
    if(!q.acts_first)s+=" It was checked to you, but on a board this connected a check doesn't rule out a straight — strong hands often check to trap — so betting mostly folds out the hands you beat and gets called by the ones that beat you.";
    return s;}
  if(q.street==="river"&&RIVER_PLAIN[q.reason])return RIVER_PLAIN[q.reason];
  return PLAIN_HEAD[q.reason]||TERMS.plain.reason[q.reason]||q.headline;
}
// The generalizable lesson — a one-line pattern per spot type, so the takeaway
// transfers to the next hand instead of staying stuck to this one. Jargon-free.
const RULES={
  value:"When you're ahead of the hands that would call, bet to get paid.",
  protection:"Ahead, but dangerous cards could still come? Bet so the hands chasing a card have to pay.",
  bluff:"No way to win at the end? Bet to make better hands fold.",
  semi_bluff:"Bet hands that can still improve — you win now if they fold, later if your hand improves.",
  pot_control:"A medium-strength hand plays best in a small pot — check.",
  trap:"A huge hand can check to let a weaker one bluff or catch up.",
  realization:"A weak hand wants to reach the end cheaply — check and take a free card.",
  value_call:"Call when you're ahead of the hands doing the betting.",
  bluff_catch:"Call when your hand beats the ones they'd bluff with.",
  call_odds:"Chase an unfinished hand when it's cheap enough to be worth your chance of completing it.",
  fold:"Not enough to keep going? Folding to save chips is the winning play.",
  raise_value:"A strong hand raises to build a bigger pot.",
  raise_bluff:"Raise as a bluff when a big hand is believable and they can fold.",
  raise_semibluff:"Raise hands that can improve — pressure them now, and you can still hit.",
  mixed:"When two plays are this close, either is fine — just pick one."
};
// River rules: no "cards to come" / "improve" / "catch up" wording (board is final).
const RIVER_RULES={
  realization:"A weak hand on the river just checks and goes to showdown cheaply.",
  semi_bluff:"No more cards — this bet is a pure bluff to make better hands fold.",
  call_odds:"It's the last card — call only when the price is right to see who wins.",
  raise_semibluff:"Nothing left to draw to — this raise is a bluff to pressure them into folding.",
  protection:"The board is final — bet thin value / denial against hands that still call worse.",
  trap:"A huge hand can check to induce a bet or bluff; nothing can catch up anymore."
};
function ruleFor(q){
  if(q.street==="river"&&RIVER_RULES[q.reason])return RIVER_RULES[q.reason];
  return RULES[q.reason];
}
function situation(q){
  const first=q.node.endsWith("_first"), vscheck=q.node.endsWith("_vs_check");
  const sm=eff("positions");
  if(sm==="plain"){
    if(first) return "It's your turn, and you go first — you decide before your opponent does.";
    if(vscheck) return "Your opponent passed (checked) to you. It's your turn.";
    return q.actions.indexOf("raise")>=0
      ? "Your opponent just put chips in (bet). It's on you — match it, put in even more, or give up?"
      : "Your opponent just put chips in (bet). It's on you — match it, or give up?";
  }
  const pre=q.street==="turn"?"On the turn, ":q.street==="river"?"On the river, ":"On the flop, ";
  // vs_bet: an OOP player checked and now faces a bet; an IP player faces an opponent
  // lead — not their own c-bet. (Same distinction the pack server already makes.)
  const betRole=q.is_oop?" — you checked and now face a bet.":" — they led into you.";
  if(sm==="learning"){
    const who="you're the "+q.acting_player+" (you act "+(q.acts_first?"first":"last")+")";
    const role=first?", first to act.":vscheck?" — it's checked to you.":betRole;
    return pre+who+role;
  }
  const role=first?", first to act.":vscheck?" and it's checked to you.":betRole;
  return pre+"you're the "+q.acting_player+role;
}

function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
const SYM={h:"heart",d:"diam",c:"club",s:"spade"};
function svgUse(cls,id){const NS="http://www.w3.org/2000/svg";
  const svg=document.createElementNS(NS,"svg");svg.setAttribute("class",cls);
  const u=document.createElementNS(NS,"use");u.setAttribute("href","#"+id);svg.appendChild(u);return svg;}
function card(t){const r=t[0],s=(t[1]||"").toLowerCase(),sy=SYM[s]||"spade";
  const e=document.createElement("div");e.className="pc pc-"+sy;
  const ix=document.createElement("span");ix.className="ix";
  const b=document.createElement("b");b.textContent=(r==="T"?"10":r);ix.appendChild(b);
  ix.appendChild(svgUse("mini","so-"+sy));
  const ct=document.createElement("span");ct.className="center";ct.appendChild(svgUse("psuit","sym-"+sy));
  e.appendChild(ix);e.appendChild(ct);return e;}
function render(cs,el){el.innerHTML="";cs.forEach(c=>el.appendChild(card(c)));}

// --- pre-flop ("Chapter 0"): a distinct question kind, its own render/feedback path ---
const PF_ACT={
  plain:{fold:"Fold",open:"Raise (open)",call:"Call","3bet":"Re-raise","4bet":"4-bet"},
  learning:{fold:"Fold",open:"Open (raise first-in)",call:"Call","3bet":"3-bet (re-raise)","4bet":"4-bet (re-raise a 3-bet)"},
  poker:{fold:"Fold",open:"Open 2.5bb",call:"Call","3bet":"3-bet","4bet":"4-bet"}};
function pfActLabel(a){return (PF_ACT[eff("positions")]||PF_ACT.poker)[a]||a;}
// Position terms ladder: full names in Beginner/Learning; abbreviations in Pro.
const PF_POS_FULL={UTG:"UTG (first to act)",HJ:"the Hijack",CO:"the Cutoff",BTN:"the Button",SB:"the Small Blind",BB:"the Big Blind"};
const PF_POS_ABBR={UTG:"UTG",HJ:"the HJ",CO:"the CO",BTN:"the BTN",SB:"the SB",BB:"the BB"};
function pfPos(p){return (eff("positions")==="poker"?PF_POS_ABBR:PF_POS_FULL)[p]||p;}
function pfSituation(q){
  if(q.ctx==="def")return "You're in "+pfPos(q.pos)+", and "+pfPos(q.opener)+" opens. It's on you.";
  if(q.ctx==="vs3bet")return "You opened from "+pfPos(q.pos)+", and "+pfPos(q.tbettor)+" 3-bets. Back on you.";
  return "You're on "+pfPos(q.pos)+". It folds to you.";   // rfi
}
function renderPreflop(q){
  const posEl=document.getElementById("pos");posEl.textContent=q.pos;posEl.className="pos "+q.pos;
  document.getElementById("sit").textContent=pfSituation(q);
  const bd=document.getElementById("demotag");bd.hidden=false;bd.textContent="Preflop";
  document.getElementById("boardcap").style.display="none";document.getElementById("board").style.display="none";
  document.getElementById("herocap").textContent="Your hand";
  render(q.hand,document.getElementById("hero"));
  const box=document.getElementById("acts");box.innerHTML="";
  q.actions.forEach((a,i)=>{const b=document.createElement("button");b.className="act";b.dataset.a=a;
    const l=document.createElement("span");l.textContent=pfActLabel(a);const k=document.createElement("span");k.className="k";k.textContent=String(i+1);
    b.appendChild(l);b.appendChild(k);b.onclick=()=>answer(a);box.appendChild(b);});
  setHint(q.actions.length);renderSeats(q);
  document.getElementById("prog").style.width=(100*pos/Math.max(1,order.length))+"%";
}
function renderQuestion(q){
  if(q.preflop)return renderPreflop(q);
  document.getElementById("boardcap").style.display="";document.getElementById("board").style.display="";
  const posEl=document.getElementById("pos");posEl.textContent=posLabel(q);posEl.className="pos "+q.acting_player;
  document.getElementById("sit").textContent=situation(q);
  const bd=document.getElementById("demotag");bd.hidden=!q.badge;bd.textContent=q.badge||"";
  document.getElementById("boardcap").textContent=(T().boardcap&&T().boardcap[q.street])||"Flop";
  document.getElementById("herocap").textContent=T().herocap;
  render(q.board,document.getElementById("board"));
  render(q.hero,document.getElementById("hero"));
  const box=document.getElementById("acts");box.innerHTML="";
  q.actions.forEach((a,i)=>{
    const b=document.createElement("button");b.className="act";b.dataset.a=a;
    const lab=document.createElement("span");lab.textContent=actLabel(a);
    const k=document.createElement("span");k.className="k";k.textContent=String(i+1);
    b.appendChild(lab);b.appendChild(k);
    b.onclick=()=>answer(a);box.appendChild(b);
  });
  setHint(q.actions.length);renderSeats(q);
  document.getElementById("prog").style.width=(100*pos/Math.max(1,order.length))+"%";
}
function setHint(n){const el=document.getElementById("hint");if(!el)return;
  let ks="";for(let i=1;i<=n;i++)ks+="<kbd>"+i+"</kbd>";
  el.innerHTML="Pick with "+ks+" &middot; next <kbd>Enter</kbd> &middot; back <kbd>&larr;</kbd>";}
// ---- position / situation graphic: a small table diagram so who-you-are, where the
// button is, and what just happened read at a glance (not only from text). ----
function renderSeats(q){
  const el=document.getElementById("seats");if(!el)return;
  el.innerHTML="";el.hidden=false;
  el.appendChild(q.preflop?preflopRing(q):ringTable(q));
}
const POS_PLAIN={BTN:"on the button",BB:"in the big blind",SB:"in the small blind",CO:"in the cutoff",HJ:"in the hijack",UTG:"first to act"};
// full 6-max ring so position is spatial: clockwise seating + fixed screen slots.
const RING_ORDER=["BTN","SB","BB","UTG","HJ","CO"];
const RING_SLOTS=[[50,85,1],[15,67,1],[15,33,0],[50,15,0],[85,33,0],[85,67,1]]; // x%,y%,labelAbove
const RING_SHORT={BTN:"the button",BB:"big blind",SB:"small blind",CO:"cutoff",HJ:"hijack",UTG:"under the gun"};
function ringTable(q){
  const hero=q.acting_player||"BB";
  const villain=q.villain||(hero==="BTN"?"BB":"BTN");
  const node=q.node||"";
  let oppAct="",ok="wait";
  if(node.endsWith("_vs_check")){oppAct="checked";ok="check";}
  else if(node.endsWith("_vs_bet")){oppAct="bet";ok="bet";}
  const start=Math.max(0,RING_ORDER.indexOf(hero));
  const seats=[];for(var i=0;i<6;i++)seats.push(RING_ORDER[(start+i)%6]);
  const D='<span class="dbtn" title="dealer button">D</span>';
  let html="";
  seats.forEach(function(pos,i){
    const x=RING_SLOTS[i][0],y=RING_SLOTS[i][1],above=RING_SLOTS[i][2];
    const role=pos===hero?"you":pos===villain?"opp":"fold";
    const btn=(pos==="BTN")?D:"";
    let inner='<div class="av">'+btn+'</div>';
    if(role==="you")inner+='<div class="nm">You</div><div class="ps">'+(RING_SHORT[pos]||pos)+'</div><div class="turn">&#9679; your move</div>';
    else if(role==="opp")inner+='<div class="nm">Opponent</div><div class="ps">'+(RING_SHORT[pos]||pos)+'</div>';
    else inner+='<div class="nm">'+pos+'</div>';
    html+='<div class="tvs '+role+(above?" ab":"")+'" style="left:'+x+'%;top:'+y+'%">'+inner+'</div>';
  });
  html+='<div class="tvmid">'+(ok!=="wait"?'<span class="oppchip a-'+ok+'">'+oppAct+'</span>':'')+'<span class="potlab">pot</span></div>';
  const w=document.createElement("div");w.className="tv";w.innerHTML=html;return w;
}
function preflopRing(q){
  // same 6-max ring as postflop, so preflop position is spatial too: you + the raiser
  // highlighted, the button on its seat, everyone else dimmed. The situation line above
  // spells out the action ("the button opens, it's on you"); the chip echoes it.
  const hero=q.pos, opener=q.opener||null, tb=q.tbettor||null;
  const start=Math.max(0,RING_ORDER.indexOf(hero));
  const seats=[];for(var i=0;i<6;i++)seats.push(RING_ORDER[(start+i)%6]);
  const D='<span class="dbtn" title="dealer button">D</span>';
  let html="";
  seats.forEach(function(pos,i){
    const x=RING_SLOTS[i][0],y=RING_SLOTS[i][1],above=RING_SLOTS[i][2];
    const isHero=pos===hero, isTb=pos===tb&&!isHero, isOpener=pos===opener&&!isHero&&!isTb;
    const role=isHero?"you":(isTb||isOpener)?"opp":"fold";
    const btn=(pos==="BTN")?D:"";
    let inner='<div class="av">'+btn+'</div>';
    if(isHero)inner+='<div class="nm">You</div><div class="ps">'+(RING_SHORT[pos]||pos)+'</div><div class="turn">&#9679; your move</div>';
    else if(isTb)inner+='<div class="nm">Opponent</div><div class="ps">'+(RING_SHORT[pos]||pos)+'</div><div class="oppchip a-bet">3-bets</div>';
    else if(isOpener)inner+='<div class="nm">Opponent</div><div class="ps">'+(RING_SHORT[pos]||pos)+'</div><div class="oppchip a-bet">opens</div>';
    else inner+='<div class="nm">'+pos+'</div>';
    html+='<div class="tvs '+role+(above?" ab":"")+'" style="left:'+x+'%;top:'+y+'%">'+inner+'</div>';
  });
  html+='<div class="tvmid"><span class="potlab">blinds</span></div>';
  const w=document.createElement("div");w.className="tv";w.innerHTML=html;return w;
}
function renderHand(){                                  // draw the current history entry
  const e=hist[hidx];answered=false;chosen=null;cur=e.q;
  document.getElementById("fb").className="fb";sheetOpen(false);
  renderQuestion(cur);coachReset();
  if(e.pick!=null)replayAnswer(e.pick);                 // already answered -> show its result again
  updateNav();
}
function newHand(){hist.push({q:Q[order[pos]],pick:null});hidx=hist.length-1;renderHand();}
function replayAnswer(a){answered=true;chosen=a;renderFeedback(cur,a,[]);}  // review: no stats change
function prev(){if(hidx>0){hidx--;renderHand();}}
// Forward is allowed when reviewing an earlier hand, or on the newest hand once it's answered
// (so you can't skip a hand without answering it).
function canFwd(){return hidx<hist.length-1||(hidx===hist.length-1&&answered);}
function updateNav(){const p=document.getElementById("prev"),f=document.getElementById("fwd");
  if(p)p.disabled=hidx<=0;if(f)f.disabled=!canFwd();}

function renderPreflopFeedback(q,a){
  const correct=a===q.answer, closeOk=q.mixed&&a===q.alt;
  document.querySelectorAll("#acts .act").forEach(b=>{b.disabled=true;b.className="act";
    if(b.dataset.a===q.answer)b.classList.add("g-best","chosen");
    else if(q.mixed&&b.dataset.a===q.alt)b.classList.add("g-good","chosen");
    else if(b.dataset.a===a)b.classList.add("g-major_error","chosen");});
  const v=document.getElementById("verdict");
  if(correct){v.className="verdict v-best";
    v.textContent=q.mixed?("✓ "+pfActLabel(a)+" — right, and it's close ("+pfActLabel(q.alt).toLowerCase()+" is fine too).")
      :("✓ Correct — "+pfActLabel(a).toLowerCase()+".");}
  else if(closeOk){v.className="verdict v-good";
    v.textContent="≈ Close — "+pfActLabel(a).toLowerCase()+" is fine; "+pfActLabel(q.answer).toLowerCase()+" is the small favourite.";}
  else{v.className="verdict v-major_error";v.textContent="✗ Not quite — the play is "+pfActLabel(q.answer)+".";}
  const rd=document.getElementById("read");rd.innerHTML="";rd.appendChild(document.createTextNode("You held "));
  const bc=document.createElement("b");bc.textContent=q.read;rd.appendChild(bc);rd.appendChild(document.createTextNode("."));
  document.getElementById("reason").style.display="none";
  document.getElementById("head").textContent=q.why;
  document.getElementById("cost").hidden=true;document.getElementById("det").innerHTML="";
  document.getElementById("unlock").hidden=true;document.getElementById("moretoggle").hidden=true;
  document.getElementById("stand").hidden=true;document.getElementById("morebody").hidden=false;
  const ruleEl=document.getElementById("rule");ruleEl.hidden=false;ruleEl.innerHTML="";
  const lb=document.createElement("b");lb.textContent="Rule of thumb";ruleEl.appendChild(lb);ruleEl.appendChild(document.createTextNode(q.rule));
  document.querySelector(".mix").style.display="none";
  document.getElementById("fb").className="fb on";sheetOpen(true);
}
// ===== Decision breakdown: the handful of factors behind the play, each with a plain why.
// Only HAND STRENGTH gets a 1-5 meter (it is genuinely an ordinal scale); position / board /
// their line are categorical, so they get a clear read + why instead of a fake number. The
// verdict line above the panel is the synthesis — these are the pieces that add up to it. =====
function handTier(rd){
  const c=rd.cat;
  if(["straight","flush","full","quads","sflush","trips","twopair"].indexOf(c)>=0)return 5;
  if(c==="pair")return (rd.pairKind==="over"||rd.pairKind==="top")?4:(rd.pairKind==="mid"?3:2);
  return rd.draw?2:1;   // air (with/without a draw)
}
function decisionFactors(q,rd){
  const items=[];
  // relative-strength read (what you beat / what beats you) — the real "how strong is this".
  items.push({label:"Your hand",meter:handTier(rd),read:cap1(rd.made),why:standingText(rd)});
  const bd=rd.boardStraighty>=2||rd.boardFlushy>=2?2:(rd.boardStraighty>=1||rd.boardFlushy>=1?1:0);
  const river=q.street==="river";
  items.push({label:"Board",meter:null,
    read:rd.boardFlushAlone?"Board flush":rd.boardStraightAlone?"Board straight"
      :(bd===0?"Dry & safe":bd===1?"A few draws out there":"Wet — straights/flushes live"),
    why:rd.boardFlushAlone?"The five shared cards are already a flush — hole cards only matter if they beat that flush."
      :rd.boardStraightAlone?"The five shared cards are already a straight — hole cards only matter if they beat that straight."
      :bd===0?(river?"No straights or flushes on this board, so the ranking is unlikely to surprise you."
        :"No straights or flushes are possible, so the board is unlikely to change who's ahead.")
      :bd===1?(river?"Some straight/flush possibilities are already on the board — keep that in mind."
        :"Some cards can still come that shift who's ahead — worth keeping in mind.")
      :(river?"Straights and flushes are live on this board, so big made hands are vulnerable."
        :"Straights and flushes are live, so big made hands and big draws are both in play.")});
  items.push({label:"Position",meter:null,
    read:q.is_oop?"Out of position":"In position",
    why:q.is_oop?"You act first — you have to decide before seeing what they do, which is harder."
      :"You act last — you decide with the most information, which is the easier seat."});
  const node=q.node||""; let lr,lw;
  if(node.indexOf("_vs_check")>=0){lr="They checked to you";lw="Their range is capped toward weak — most strong hands would have bet, so a monster is unlikely.";}
  else if(node.indexOf("_vs_bet")>=0){lr="They bet into you";lw="They're representing strength — but a betting range still holds bluffs and worse hands you beat.";}
  else{lr="You're first to act";lw="No information yet — you set the price and take the lead.";}
  items.push({label:"Their move",meter:null,read:lr,why:lw});
  return items;
}
function renderFactors(q){
  const el=document.getElementById("factors");if(!el)return;
  if(q.preflop||eff("reason:"+q.reason)==="poker"){el.hidden=true;el.innerHTML="";return;}
  el.hidden=false;el.innerHTML="";
  decisionFactors(q,handRead(q.hero,q.board)).forEach(function(f){
    const row=document.createElement("div");row.className="fac";
    const top=document.createElement("div");top.className="fac-top";
    top.innerHTML='<span><span class="fac-l">'+f.label+'</span> &middot; <b class="fac-read">'+f.read+'</b></span>';
    if(f.meter!=null){const m=document.createElement("span");m.className="meter";
      const cls=f.meter>=4?"":f.meter===3?" mid":" low";
      for(let i=1;i<=5;i++){const d=document.createElement("i");if(i<=f.meter)d.className="on"+cls;m.appendChild(d);}
      top.appendChild(m);}
    const why=document.createElement("div");why.className="fac-why";why.textContent=f.why;
    row.appendChild(top);row.appendChild(why);el.appendChild(row);
  });
}
// ===== "A similar hand plays the opposite way" — the same hand strength can call for
// opposite plays (bet-for-value vs check-to-trap, bluff vs give-up, call vs fold, ...). Each
// reason IS the rule of thumb, so a confusing twin = same hand tier + the paired opposite
// reason. We surface the closest twin and name the deciding factor that flips it. =====
const CONTRAST={
  value:{vs:["trap","pot_control"],axis:"strong"},
  trap:{vs:["value","protection"],axis:"strong"},
  protection:{vs:["pot_control","trap"],axis:"medium"},
  pot_control:{vs:["protection","value"],axis:"medium"},
  bluff:{vs:["realization"],axis:"weak"},
  semi_bluff:{vs:["realization"],axis:"draw"},
  realization:{vs:["bluff","semi_bluff"],axis:"weak"},
  bluff_catch:{vs:["fold"],axis:"face"},
  value_call:{vs:["fold"],axis:"face"},
  call_odds:{vs:["fold"],axis:"draw2"},
  fold:{vs:["bluff_catch","value_call"],axis:"face"},
  raise_value:{vs:["value_call","call_odds"],axis:"raise"},
  raise_bluff:{vs:["fold","call_odds"],axis:"raise"},
  raise_semibluff:{vs:["call_odds"],axis:"raise"}
};
const SHORT_RULE={value:"bet for value",protection:"bet to protect",trap:"check to trap",
  pot_control:"check for pot control",bluff:"bet as a bluff",semi_bluff:"bet as a semi-bluff",
  realization:"check and give up",value_call:"call — you're ahead",bluff_catch:"call to catch a bluff",
  call_odds:"call on the odds",fold:"fold",raise_value:"raise for value",raise_bluff:"raise as a bluff",
  raise_semibluff:"raise as a semi-bluff"};
const AXIS_WHY={
  strong:"whether a bet gets <b>called by worse hands</b>. Bet to build the pot when weaker hands will pay you off; check to trap when a bet would fold out everything worse — so you keep their bluffs and weak hands in.",
  medium:"how exposed your hand is and how many worse hands call. Bet to charge draws and get value when a later card could beat you; check to keep the pot small when the board is safe and betting only folds out worse.",
  weak:"whether betting can win a pot you'd otherwise lose. Bet as a bluff when you can't win by checking but can make better hands fold; check (give up) when a free card or keeping their bluffs in is worth more.",
  draw:"whether the draw is worth betting now. Bet it as a semi-bluff to fold out better hands and build a pot you'll often win; check to take a free card and keep the pot small.",
  draw2:"the price versus your chance to improve. Call when the pot lays you enough to chase; fold when it's too expensive for how often you get there.",
  face:"how many <b>bluffs</b> are in their betting range versus real hands. Call when they'd bet worse (or bluff) often enough that you beat those; fold when their bet is mostly hands that already beat you.",
  raise:"whether raising wins more than flat-calling. Raise to build the pot / deny equity when worse hands pay or draws must fold; just call to keep their bluffs and weaker hands in."
};
function cap1(s){return s?s.charAt(0).toUpperCase()+s.slice(1):s;}
function findContrast(q){
  if(q.preflop||!CONTRAST[q.reason])return null;
  const rd=handRead(q.hero,q.board),vs=CONTRAST[q.reason].vs;
  // A twin is only instructive if the HAND is genuinely similar — otherwise the strength gap
  // IS the reason it plays differently (two pair calls / one pair folds is trivial, not a
  // "same hand, opposite play"). Require the SAME made-hand category; if there's no twin of
  // the same category with the opposite play, hide the block rather than show a mismatch.
  const fam=function(x){var a=x.actions||[];return (a.indexOf("bet")>=0&&a.indexOf("check")>=0)?"line":(a.indexOf("fold")>=0&&a.indexOf("call")>=0)?"facing":"?";};
  const myFam=fam(q);
  let best=null,bs=-1;
  for(let i=0;i<ALLSPOTS.length;i++){const o=ALLSPOTS[i];   // deck + contrast-only pool
    if(o===q||o.preflop||vs.indexOf(o.reason)<0)continue;
    const ord=handRead(o.hero,o.board);
    if(ord.cat!==rd.cat)continue;                                    // same made-hand category only
    // strongly prefer the SAME situation (both check/bet, or both facing-a-bet), then same
    // street, then a position flip — so the twin differs in the deciding factor, not the setup.
    const sc=(fam(o)===myFam?4:0)+(o.street===q.street?2:0)+(o.is_oop!==q.is_oop?1:0);
    if(sc>bs){bs=sc;best={q:o,rd:ord};}
  }
  return best;
}
function contrastWhy(q,rd,o,ord){
  let s=AXIS_WHY[CONTRAST[q.reason].axis]||AXIS_WHY.strong;
  const d=[];
  if(q.is_oop!==o.is_oop)d.push("here you "+(q.is_oop?"act first (out of position)":"act last (in position)")+", there you "+(o.is_oop?"act first":"act last"));
  const wa=rd.boardStraighty>=2||rd.boardFlushy>=2, wb=ord.boardStraighty>=2||ord.boardFlushy>=2;
  if(wa!==wb)d.push("this board is "+(wa?"wet — straights/flushes live":"dry and safe")+", the other the reverse");
  if(d.length)s+=" <b>In this pair it comes down to:</b> "+d.join("; ")+".";
  return s;
}
function renderContrast(q){
  const wrap=document.getElementById("compare");if(!wrap)return;
  const body=document.getElementById("compare-body");
  const c=findContrast(q);
  if(!c){wrap.hidden=true;wrap.open=false;return;}
  wrap.hidden=false;wrap.open=false;body.innerHTML="";
  const rd=handRead(q.hero,q.board);
  const mine=SHORT_RULE[q.reason]||shortAct(q.preferred), theirs=SHORT_RULE[c.q.reason]||shortAct(c.q.preferred);
  const l1=document.createElement("div");l1.className="cmp-line";
  l1.innerHTML='<span class="cw">This hand</span><b>'+cap1(rd.made)+'</b> &rarr; the play is <span class="cmp-play a-'+q.preferred+'">'+mine+'</span>.';
  const l2=document.createElement("div");l2.className="cmp-line";
  l2.innerHTML='<span class="cw">A look-alike hand</span>';
  const hand=document.createElement("div");hand.className="cmp-hand";
  const bc=document.createElement("div");bc.className="cmp-cards";c.q.board.forEach(function(x){bc.appendChild(card(x));});
  const plus=document.createElement("span");plus.className="plus";plus.textContent="+ you";
  const hc=document.createElement("div");hc.className="cmp-cards";c.q.hero.forEach(function(x){hc.appendChild(card(x));});
  hand.appendChild(bc);hand.appendChild(plus);hand.appendChild(hc);
  const play=document.createElement("div");play.className="cmp-line";play.style.marginTop="8px";
  play.innerHTML='<b>'+cap1(c.rd.made)+'</b> &rarr; the play is <span class="cmp-play a-'+c.q.preferred+'">'+theirs+'</span>.';
  l2.appendChild(hand);l2.appendChild(play);
  const why=document.createElement("div");why.className="cmp-why";
  why.innerHTML="<b>What flips it:</b> "+contrastWhy(q,rd,c.q,c.rd);
  const go=document.createElement("button");go.type="button";go.className="cmp-go";go.textContent="Play this hand →";
  // insert right AFTER the current hand (not at the end) so Back returns to this one even
  // when the contrast is opened while reviewing an earlier hand.
  go.onclick=function(){hist.splice(hidx+1,0,{q:c.q,pick:null});hidx++;renderHand();};
  body.appendChild(l1);body.appendChild(l2);body.appendChild(why);body.appendChild(go);
}
function renderFeedback(q,a,gained){
  renderContrast(q);renderFactors(q);
  if(q.preflop)return renderPreflopFeedback(q,a);
  document.querySelector(".mix").style.display="";document.getElementById("stand").hidden=false;
  document.querySelectorAll("#acts .act").forEach(b=>{
    b.disabled=true;const ga=q.grades[b.dataset.a];b.className="act g-"+ga;
    if(b.dataset.a===a)b.classList.add("chosen");
  });
  const g=q.grades[a],pref=q.preferred;
  const you=shortAct(a),best=shortAct(pref);
  const v=document.getElementById("verdict");v.className="verdict v-"+g;
  v.textContent="";const dot=document.createElement("span");dot.className="dot";v.appendChild(dot);
  // Verdict names YOUR pick AND the better action — so a wrong answer immediately
  // shows what you should have done, not just an abstract grade.
  // Key off the GRADE, not a===pref: a co-best action (graded "best" but not the
  // single top-EV one) must still read as correct, not as a leak.
  let vmsg;
  if(g==="best")vmsg=(a===pref)?"✓ "+you+" — the best play here.":"✓ "+you+" — also a top play here.";
  else if(q.mixed&&(g==="good"||g==="acceptable"))vmsg="✓ "+you+" — close enough; any play is fine here.";
  else if(g==="good")vmsg="✓ "+you+" works — "+best+" is only a touch better.";
  else if(g==="acceptable")vmsg="~ "+you+" is OK, but "+best+" is the better play.";
  else vmsg="✗ You picked "+you+" — "+(g==="major_error"?"a big mistake":"a costly leak")+". The play is "+best+".";
  v.appendChild(document.createTextNode(vmsg));
  // Ground the explanation in the actual holding: "You held a pair of Jacks — the
  // Ace on the board outranks it." Beginners can't read their own hand yet, so this
  // is what makes the 'why' land.
  const rd=handRead(q.hero,q.board);
  const readEl=document.getElementById("read");readEl.innerHTML="";
  readEl.appendChild(document.createTextNode("You held "));
  const mb=document.createElement("b");mb.textContent=rd.made;readEl.appendChild(mb);
  if(rd.strength)readEl.appendChild(document.createTextNode(" — "+rd.strength));
  if(rd.draw)readEl.appendChild(document.createTextNode(", plus "+rd.draw));
  readEl.appendChild(document.createTextNode("."));
  // "Where you stand" is populated here but lives inside the collapsible "more" block
  // (set up further down), so the default view stays calm.
  document.getElementById("stand").textContent=standingText(rd);
  // explanation adapts to the level: Beginner = plain 'why' only; Learning = term
  // tag + explaining headline; Pro = term tag + richer baked headline + bullets.
  const rm=eff("reason:"+q.reason);
  const unit=(rm==="plain")?"chips":"bb";     // plain mode avoids the "bb" jargon
  const rp=document.getElementById("reason");
  if(rm==="plain"){rp.style.display="none";}
  else{rp.style.display="";rp.textContent=TERMS.poker.reason[q.reason]||q.reason;}
  document.getElementById("head").textContent=(rm==="poker")?q.headline:(rm==="plain")?plainHead(q):(TERMS[rm].reason[q.reason]||q.headline);
  // Learning mode: flop TERMS still say "improve" / "free card" — use river tags.
  if(rm==="learning"&&q.street==="river"&&RIVER_LEARNING[q.reason]){
    document.getElementById("head").textContent=RIVER_LEARNING[q.reason];
  }
  // For near-tie spots, replace the generic "any play is fine" with a reason the tie
  // exists — especially why an aggressive line with a weak hand is a co-best play (a bluff).
  if(q.reason==="mixed"&&rm!=="poker")document.getElementById("head").textContent=closeExplain(q,rd);
  // The portable takeaway, also inside the collapsible block.
  const ruleEl=document.getElementById("rule");
  ruleEl.innerHTML="";
  // On a coordinated board a one-pair-ish hand is a bluff-catcher, so the trap/value
  // takeaway is replaced by a pot-control one.
  const bcRule=bcReframe(q,rd);
  const ruleText=bcRule
    ? "On a coordinated board, a one-pair-type hand is a bluff-catcher, not a monster — keep the pot small and don't bet into the likely straights and flushes."
    : ruleFor(q);
  if(ruleText){ruleEl.hidden=false;
    const lab=document.createElement("b");lab.textContent="Rule of thumb";ruleEl.appendChild(lab);
    ruleEl.appendChild(document.createTextNode(ruleText));}
  else{ruleEl.hidden=true;}
  // Depth (where-you-stand + rule) lives behind "Explain more" so the default view is
  // just verdict -> held -> why -> cost -> payoffs. Hidden entirely in Pro (which
  // shows the solver detail bullets instead). Remembers the reader's open/closed choice.
  const depth=(mode!=="poker");
  const moreT=document.getElementById("moretoggle"),moreB=document.getElementById("morebody");
  if(depth){moreT.hidden=false;moreB.hidden=!moreOpen;moreT.textContent=moreOpen?"Show less ▾":"Explain more ▸";}
  else{moreT.hidden=true;moreB.hidden=true;}
  // Concrete cost: when the pick isn't the top play, show how much it gives up so
  // "why is this wrong?" has a number behind it, not just a color.
  const cost=document.getElementById("cost");
  const dEv=Math.round((q.ev[pref]-q.ev[a])*100)/100;
  if(g!=="best"&&dEv>=0.05){cost.hidden=false;
    cost.textContent=best+" averages "+fmtEv(q.ev[pref])+" "+unit+" here vs your "+you+" at "+fmtEv(q.ev[a])+" "+unit+" — about "+dEv+" "+unit+" per hand left behind.";
  }else{cost.hidden=true;}
  const dl=document.getElementById("det");dl.innerHTML="";
  // Pro detail: keep the baked bullets but tame the false precision ("~7.951%") and
  // the meaningless >100% figures that pop out on tiny pots.
  if(rm==="poker"){q.detail.forEach(d=>{
    const clean=d.replace(/~?(\d+(?:\.\d+)?)%/g,(m,n)=>{const v=Math.round(parseFloat(n));return v>100?"a large amount":v+"%";});
    const li=document.createElement("li");li.textContent=clean;dl.appendChild(li);});}
  const ut=document.getElementById("unlock");
  if(gained&&gained.length){ut.hidden=false;ut.innerHTML="";
    gained.forEach(t=>{const d=document.createElement("div");d.className="ul-row";d.textContent="🔓 New term learned — "+unlockText(t);ut.appendChild(d);});
  }else{ut.hidden=true;}
  // The solver's mixed-strategy FREQUENCY (e.g. "Raise 14%") is a game-theory
  // balancing concept, and it can flatly contradict the grade ("14% · major error")
  // — which just erodes trust. So show it only in Pro. Everyone else gets a payoff
  // view: what each choice is worth, sorted best-first, with a plain verdict and no
  // confusing percentages. The bar length encodes the grade (how good), which is the
  // question a learner is actually asking.
  const payoffView=(mode!=="poker");
  document.getElementById("mixhead").textContent=payoffView
    ?"What each choice is worth (best first)"
    :"Solver mix — how often the solver plays each action (★ = best EV), and its EV";
  const bars=document.getElementById("bars");bars.innerHTML="";
  const maxf=Math.max(1,...q.actions.map(x=>q.freq[x]));
  const GW={best:100,good:82,acceptable:58,costly:32,major_error:12};   // bar = how good
  const VW={best:"Best",good:"Fine",acceptable:"Playable",costly:"Loses money",major_error:"Big mistake"};
  const ordered=payoffView?q.actions.slice().sort((x,y)=>q.ev[y]-q.ev[x])
                          :q.actions.slice().sort((x,y)=>q.freq[y]-q.freq[x]);
  ordered.forEach(x=>{
    const ga=q.grades[x],rec=x===q.preferred,you=x===a;
    const row=document.createElement("div");row.className="row g-"+ga;
    if(rec)row.classList.add("best-row");
    if(you)row.classList.add("you-row");
    const rlab=document.createElement("div");rlab.className="rlab";
    const nm=document.createElement("span");nm.className="nm";nm.textContent=actLabel(x)+" ";
    if(rec){const st=document.createElement("span");st.className="star";st.textContent="★";st.title="Best EV (recommended)";nm.appendChild(st);nm.appendChild(document.createTextNode(" "));}
    if(you){const yp=document.createElement("span");yp.className="you";yp.textContent="YOUR PICK";nm.appendChild(yp);}
    const num=document.createElement("span");num.className="num";
    const ev=q.ev[x];
    if(payoffView){
      // Best row shows what it wins; the rest show what they GIVE UP vs best — so the
      // number agrees with the verdict (a "mistake" reads as a loss, never a + payoff).
      const txt=(ga==="best")?fmtEv(ev)+" "+unit
        :"−"+(Math.round((q.ev[pref]-ev)*100)/100)+" "+unit+" vs best";
      num.appendChild(document.createTextNode(txt+" "));
    }else{
      num.appendChild(document.createTextNode(q.freq[x]+"% · "+(ev>=0?"+":"")+ev+" "+unit+" "));
    }
    const tag=document.createElement("span");tag.className="tag";
    tag.textContent=payoffView?(VW[ga]||ga):ga.replace("_"," ");
    num.appendChild(tag);
    rlab.appendChild(nm);rlab.appendChild(num);
    const track=document.createElement("div");track.className="track";
    const i=document.createElement("i");
    i.style.width=Math.max(3,payoffView?(GW[ga]||50):Math.round(100*q.freq[x]/maxf))+"%";
    track.appendChild(i);row.appendChild(rlab);row.appendChild(track);bars.appendChild(row);
  });
  document.getElementById("fb").className="fb on";sheetOpen(true);
}
function answer(a){
  if(answered)return;answered=true;chosen=a;if(hist[hidx])hist[hidx].pick=a;updateNav();
  if(cur.preflop){
    const correct=a===cur.answer, closeOk=!correct&&cur.mixed&&a===cur.alt;
    stats.n++; if(correct||closeOk)stats.solid++; else stats.leak++;
    trackStreet(correct||closeOk);
    document.getElementById("n").textContent=stats.n;document.getElementById("solid").textContent=stats.solid;
    document.getElementById("ok").textContent=stats.ok;document.getElementById("leak").textContent=stats.leak;
    document.getElementById("acc").textContent=Math.round(100*(stats.solid+stats.ok)/stats.n)+"%";
    renderFeedback(cur,a,[]);
    document.getElementById("next").focus({preventScroll:true});
    return;
  }
  const g=cur.grades[a];
  stats.n++;
  if(g==="best"||g==="good")stats.solid++;else if(g==="acceptable")stats.ok++;else stats.leak++;
  trackStreet(g!=="costly"&&g!=="major_error");
  document.getElementById("n").textContent=stats.n;document.getElementById("solid").textContent=stats.solid;
  document.getElementById("ok").textContent=stats.ok;document.getElementById("leak").textContent=stats.leak;
  document.getElementById("acc").textContent=Math.round(100*(stats.solid+stats.ok)/stats.n)+"%";
  const gained=tryUnlock(cur,g);
  // Don't re-render the question on unlock — swapping the button labels mid-answer
  // shifts the layout under the user's cursor. The feedback already reflects the new
  // vocabulary, and the next hand deals in the upgraded language.
  renderFeedback(cur,a,gained);
  // preventScroll: focusing Next must not yank the viewport to the bottom of the card.
  document.getElementById("next").focus({preventScroll:true});
}
function next(){
  if(hidx<hist.length-1){hidx++;renderHand();}          // stepping forward through review
  else{pos=(pos+1)%order.length;if(pos===0)order=shuffle(order.slice());newHand();}
}
// ---- result as a slide-up bottom sheet: verdict + why rise over the hand; swipe down,
// tap the dimmer, or hit Next to deal the next one. ----
function sheetOpen(b){const s=document.getElementById("fb-scrim");if(s)s.hidden=!b;
  document.documentElement.classList.toggle("sheet-open",b);}
// Dismissing the sheet (scrim tap / swipe down) only CLOSES it — it never advances the hand.
// Advancing is always an explicit Next (the sheet button, the app-bar Next, or Enter/->), so
// dismissing a hand you stepped Back to review no longer skips you forward.
function closeSheet(){document.getElementById("fb").className="fb";sheetOpen(false);}
function reopenSheet(){if(answered){document.getElementById("fb").className="fb on";sheetOpen(true);}}
(function(){const scrim=document.getElementById("fb-scrim"),fb=document.getElementById("fb");
  if(scrim)scrim.onclick=function(){closeSheet();};
  // tap the hand (once dismissed) to bring the result back up
  const card=document.querySelector(".card");
  if(card)card.addEventListener("click",function(e){
    if(!answered||e.target.closest(".fb"))return;
    if(!document.getElementById("fb").classList.contains("on"))reopenSheet();});
  if(!fb)return;let y0=null;
  fb.addEventListener("touchstart",function(e){y0=fb.scrollTop<=0?e.touches[0].clientY:null;},{passive:true});
  fb.addEventListener("touchmove",function(e){if(y0==null)return;const dy=e.touches[0].clientY-y0;
    if(dy>0)fb.style.transform="translate(-50%,"+Math.min(dy,140)+"px)";},{passive:true});
  fb.addEventListener("touchend",function(e){if(y0==null)return;const dy=e.changedTouches[0].clientY-y0;y0=null;
    fb.style.transform="";if(dy>90)closeSheet();},{passive:true});
})();

// Adaptive unlock: play a spot well (best/good) and its concept graduates into
// your vocabulary. The 'basics' (positions + streets) unlock on your first good
// answer; each strategy concept unlocks the first time you nail that spot type.
function tryUnlock(q,g){
  if(mode!=="progressive"||!(g==="best"||g==="good"))return [];
  const gained=[];
  ["positions","streets"].forEach(t=>{if(!learned.has(t)){learned.add(t);gained.push(t);}});
  const rt="reason:"+q.reason;
  if(!learned.has(rt)){learned.add(rt);gained.push(rt);}
  if(gained.length){try{localStorage.setItem("learned",JSON.stringify([...learned]));}catch(e){}updateVocab();}
  return gained;
}
function unlockText(t){
  if(t==="positions")return "Positions — who acts when. In position (IP) acts last (usually the Button); out of position (OOP) acts first. In blind-vs-blind, the BB is IP.";
  if(t==="streets")return "Flop, turn, river — the shared cards come in stages (3, then a 4th, then a 5th).";
  return (TERMS.poker.reason[t.slice(7)]||"")+" — "+(TERMS.learning.reason[t.slice(7)]||"").replace(/^[^—]*— /,"");
}
function updateVocab(){const v=document.getElementById("vocab");
  v.hidden=(mode!=="progressive");
  v.textContent="🔓 "+learned.size+" / "+VOCAB_TOTAL+" terms";}
const LEVEL_HINT={
  progressive:"Starts in plain words; poker terms unlock as you play them well.",
  plain:"Plain English — no poker jargon.",
  learning:"Real poker terms, each with a short explanation.",
  poker:"Full terminology and solver detail."};
function updateLevelHint(){const el=document.getElementById("levelhint");if(el)el.textContent=LEVEL_HINT[mode]||"";}

function applyModeUI(){document.querySelectorAll("#lang button").forEach(b=>b.classList.toggle("on",b.dataset.m===mode));}
function setMode(m){mode=m;try{localStorage.setItem("lang",m);}catch(e){}applyModeUI();updateVocab();updateLevelHint();
  if(cur){renderQuestion(cur);if(answered)renderFeedback(cur,chosen,[]);}}
document.querySelectorAll("#lang button").forEach(b=>b.onclick=()=>setMode(b.dataset.m));
// --- train-category selector: filter the deck to one street (or all) ---
function qcat(q){return q.preflop?"preflop":(q.street||"flop");}
function catCounts(){const c={all:Q.length,preflop:0,flop:0,turn:0,river:0};Q.forEach(q=>{const k=qcat(q);c[k]=(c[k]||0)+1;});return c;}
function buildOrder(){order=shuffle([...Q.keys()].filter(i=>cat==="all"||qcat(Q[i])===cat));pos=0;hist=[];hidx=-1;
  const cc=catCounts();document.getElementById("catcount").textContent=(cat==="all"?Q.length:(cc[cat]||0))+" spots";}
function applyCatUI(){const cc=catCounts();
  document.querySelectorAll("#cats button").forEach(b=>{b.classList.toggle("on",b.dataset.c===cat);
    const n=b.dataset.c==="all"?Q.length:(cc[b.dataset.c]||0);b.disabled=n===0;b.style.opacity=n===0?"0.4":"";});}
function setCat(c){cat=c;try{localStorage.setItem("cat",c);}catch(e){}applyCatUI();buildOrder();newHand();}
document.querySelectorAll("#cats button").forEach(b=>b.onclick=()=>setCat(b.dataset.c));
// intro: open by default, remember if the reader dismisses it
const intro=document.getElementById("intro");
// Collapsed by default so the game sits at the top; remember if the reader opens it.
try{intro.open=localStorage.getItem("introOpen")==="1";}catch(e){intro.open=false;}
intro.addEventListener("toggle",()=>{try{localStorage.setItem("introOpen",intro.open?"1":"0");}catch(e){}});

document.getElementById("next").onclick=next;
document.getElementById("prev").onclick=prev;
document.getElementById("fwd").onclick=next;
document.getElementById("moretoggle").onclick=function(){
  moreOpen=!moreOpen;try{localStorage.setItem("moreOpen",moreOpen?"1":"0");}catch(e){}
  document.getElementById("morebody").hidden=!moreOpen;
  this.textContent=moreOpen?"Show less ▾":"Explain more ▸";};
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="SUMMARY"||e.target.id==="moretoggle")return;   // let the toggle handle its own Enter/Space
  if(/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName))return;         // don't hijack typing in inputs
  if(e.target.closest&&e.target.closest("#coach"))return;              // coach panel owns its own keys (chips/send/input)
  if(e.target.closest&&e.target.closest("#compare"))return;            // let the compare summary / "Play this hand" handle Enter/Space themselves
  const tv=document.getElementById("v-train");if(tv&&!tv.classList.contains("on"))return;  // only the Train view takes hotkeys
  if(e.key==="ArrowLeft"){e.preventDefault();prev();return;}   // step back to review
  if(!answered){const i=parseInt(e.key);if(cur&&i>=1&&i<=cur.actions.length)answer(cur.actions[i-1]);}
  else if(e.key==="Enter"||e.key===" "||e.key==="ArrowRight"){e.preventDefault();next();}
});
// ===== Ask-a-coach — bring-your-own-key. Transport seam: web fetch today, native
// CapacitorHttp when this same page is wrapped as the mobile app (no CORS/CSP there). =====
const SUIT_T={h:"♥",d:"♦",c:"♣",s:"♠"};
function cardsText(a){return (a||[]).map(c=>c[0]+(SUIT_T[c[1]]||c[1])).join(" ");}
// Providers are config-driven so adding one (or a custom base URL for a local model) is a
// data change, not new plumbing. Claude is the default; each entry builds its own request.
const PROVIDERS={
  claude:{label:"Claude (Anthropic)",dflt:"claude-sonnet-5",
    models:["claude-sonnet-5","claude-opus-4-8","claude-haiku-4-5-20251001"],keyhint:"sk-ant-…",
    req(k,sys,ms,model){return{url:"https://api.anthropic.com/v1/messages",
      headers:{"content-type":"application/json","x-api-key":k,"anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
      body:{model:model,max_tokens:1024,system:sys,messages:ms}};},
    parse(j){return ((j&&j.content)||[]).map(b=>b.text||"").join("").trim();},
    emsg(j){return j&&j.error&&j.error.message;}},
  openai:{label:"OpenAI",dflt:"gpt-4o-mini",
    models:["gpt-4o-mini","gpt-4o"],keyhint:"sk-…",
    req(k,sys,ms,model){return{url:"https://api.openai.com/v1/chat/completions",
      headers:{"content-type":"application/json","authorization":"Bearer "+k},
      body:{model:model,messages:[{role:"system",content:sys}].concat(ms)}};},
    parse(j){return ((j&&j.choices&&j.choices[0]&&j.choices[0].message&&j.choices[0].message.content)||"").trim();},
    emsg(j){return j&&j.error&&j.error.message;}},
};
function coachCfg(){try{return JSON.parse(localStorage.getItem("coach")||"{}");}catch(e){return {};}}
function coachSaveCfg(c){try{localStorage.setItem("coach",JSON.stringify(c));}catch(e){}}
let coachMsgs=[],coachBusy=false,coachErr=null,coachGen=0;

// Serialize the current spot's real solver data so the model explains THIS hand, not
// generic theory. Everything here already drives the on-screen feedback.
function coachSpot(q){
  const L=[];
  if(q.preflop){
    L.push("Street: PRE-FLOP.");
    L.push("Your position: "+(q.pos||"?")+".");
    L.push("Your hand: "+cardsText(q.hand)+".");
    try{L.push("Situation: "+pfSituation(q)+".");}catch(e){}
    if(q.answer){let s="Solver's recommended action: "+coachPfLabel(q.answer);
      if(q.mixed&&q.alt)s+=" (also acceptable: "+coachPfLabel(q.alt)+")";L.push(s+".");}
    if(q.why)L.push("Short reason: "+q.why);
    if(q.rule)L.push("Rule of thumb: "+q.rule);
    return L.join("\n");
  }
  L.push("Street: "+(q.street||"flop")+".");
  L.push("Board (shared cards): "+cardsText(q.board)+".");
  L.push("Your hand: "+cardsText(q.hero)+".");
  L.push("You "+(q.acts_first?"act first":"act last")+" ("+(q.is_oop?"out of position":"in position")+").");
  try{L.push("Situation: "+situation(q)+".");}catch(e){}
  L.push("Your options — solver EV (in pot units, higher is better), how often the solver takes each, and its grade:");
  q.actions.forEach(a=>{const lab=(q.labels&&q.labels[a])||a;
    L.push("  • "+lab+": EV "+q.ev[a]+", played "+q.freq[a]+"% of the time, grade "+String(q.grades[a]).replace("_"," ")+(a===q.preferred?"  <- solver's preferred play":""));});
  try{L.push("Plain explanation already shown to the learner: "+plainHead(q));}catch(e){}
  return L.join("\n");
}
function coachPfLabel(a){try{return pfActLabel(a);}catch(e){return a;}}
function coachSystem(q){
  return "You are a friendly, concise poker coach inside a beginner training app. Answer ONLY about the "+
  "current spot below, grounded in its exact numbers. Never contradict the solver's preferred play — if the "+
  "learner proposes a different line, explain what the numbers say about it. If a question needs information not "+
  "shown here, say what you'd need. Keep answers to 2–5 short sentences, plain English, focused on WHY. This is "+
  "a study tool using play chips — do not give real-money gambling or betting advice.\n\n--- CURRENT SPOT ---\n"+coachSpot(q);
}

function coachRender(){
  const log=document.getElementById("coach-log");if(!log)return;log.innerHTML="";
  coachMsgs.forEach(m=>{const d=document.createElement("div");d.className="cmsg "+(m.role==="user"?"user":"bot");d.textContent=m.content;log.appendChild(d);});
  if(coachBusy){const d=document.createElement("div");d.className="cmsg bot think";d.textContent="Thinking…";log.appendChild(d);}
  if(coachErr){const d=document.createElement("div");d.className="cmsg err";d.textContent=coachErr;log.appendChild(d);}
  log.scrollTop=log.scrollHeight;
}
function coachToggleSend(){const b=document.getElementById("coach-send"),i=document.getElementById("coach-input");
  if(b)b.disabled=coachBusy;if(i)i.disabled=coachBusy;}

async function coachTransport(url,headers,body){
  const cap=window.Capacitor;
  if(cap&&cap.isNativePlatform&&cap.isNativePlatform()){       // native app: no CORS/CSP, key from secure store
    const http=(cap.Plugins&&cap.Plugins.CapacitorHttp)||window.CapacitorHttp;
    if(http){const r=await http.request({url:url,method:"POST",headers:headers,data:body});
      const d=typeof r.data==="string"?JSON.parse(r.data||"{}"):r.data;
      return {ok:r.status>=200&&r.status<300,status:r.status,json:d};}
  }
  const resp=await fetch(url,{method:"POST",headers:headers,body:JSON.stringify(body)});
  let j=null;try{j=await resp.json();}catch(e){}
  return {ok:resp.ok,status:resp.status,json:j};
}
async function coachAsk(text){
  if(coachBusy||!text)return;
  const cfg=coachCfg();
  if(!cfg.key){coachSettings(true);coachErr="Add your API key above to start.";coachRender();return;}
  const P=PROVIDERS[cfg.provider]||PROVIDERS.claude;
  const gen=coachGen;   // if the hand changes mid-flight, drop the stale reply (see coachReset)
  coachErr=null;coachMsgs.push({role:"user",content:text});coachBusy=true;coachRender();coachToggleSend();
  try{
    const r0=P.req(cfg.key,coachSystem(cur),coachMsgs,cfg.model||P.dflt);
    const r=await coachTransport(r0.url,r0.headers,r0.body);
    if(gen!==coachGen)return;   // user moved to a new hand while waiting — this reply is stale
    coachBusy=false;
    if(!r.ok){coachMsgs.pop();coachErr=coachHttpHelp(r.status,r.json&&P.emsg(r.json));}
    else{const t=P.parse(r.json);
      if(t)coachMsgs.push({role:"assistant",content:t});
      else{coachMsgs.pop();coachErr="The provider returned an empty reply — try again.";}}
  }catch(e){if(gen!==coachGen)return;coachBusy=false;coachMsgs.pop();coachErr=coachNetHelp();}
  coachRender();coachToggleSend();
}
function coachHttpHelp(status,msg){
  if(status===401||status===403)return "Your API key was rejected ("+status+"). Check it in the box above.";
  if(status===429)return "Rate-limited or out of credit (429). Check your provider account.";
  return "Request failed"+(status?(" ("+status+")"):"")+(msg?": "+msg:".");
}
function coachNetHelp(){
  return "Couldn't reach the provider. If you're viewing this inside the Claude artifact preview, external calls are "+
  "blocked here — open the GitHub Pages site or the app to chat. Otherwise check your connection and key.";
}

function coachSettings(show){
  const set=document.getElementById("coach-set"),conn=document.getElementById("coach-conn");
  const cfg=coachCfg(),connected=!!cfg.key;
  set.hidden=connected&&!show;
  if(connected){conn.hidden=false;conn.innerHTML="";
    const P=PROVIDERS[cfg.provider]||PROVIDERS.claude;
    const s=document.createElement("span");s.textContent="Connected — ";
    const b=document.createElement("b");b.textContent=P.label+" · "+(cfg.model||P.dflt);s.appendChild(b);
    const btn=document.createElement("button");btn.type="button";btn.textContent=set.hidden?"Change key / model":"Hide";
    btn.onclick=()=>coachSettings(set.hidden);
    conn.appendChild(s);conn.appendChild(btn);
  }else{conn.hidden=true;set.hidden=false;}
}
function coachInit(){
  const prov=document.getElementById("coach-prov");if(!prov)return;
  Object.keys(PROVIDERS).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=PROVIDERS[k].label;prov.appendChild(o);});
  const cfg=coachCfg();prov.value=cfg.provider||"claude";
  const applyProv=resetModel=>{const P=PROVIDERS[prov.value]||PROVIDERS.claude;
    const dl=document.getElementById("coach-models");dl.innerHTML="";
    P.models.forEach(m=>{const o=document.createElement("option");o.value=m;dl.appendChild(o);});
    document.getElementById("coach-key").placeholder=P.keyhint;
    if(resetModel)document.getElementById("coach-model").value=P.dflt;};
  applyProv(false);
  document.getElementById("coach-model").value=cfg.model||(PROVIDERS[prov.value]||PROVIDERS.claude).dflt;
  if(cfg.key)document.getElementById("coach-key").value=cfg.key;
  prov.onchange=()=>applyProv(true);
  document.getElementById("coach-save").onclick=()=>{
    const key=document.getElementById("coach-key").value.trim();
    if(!key){coachErr="Please paste an API key first.";coachRender();return;}
    coachSaveCfg({provider:prov.value,model:document.getElementById("coach-model").value.trim(),key:key});
    coachErr=null;coachSettings(false);coachRender();
  };
  // Best-effort hint that an embedded preview will block the network call.
  try{if(window.self!==window.top)document.getElementById("coach-envnote").textContent="Note: embedded previews block external calls — if chat fails, open the Pages site or app.";}
  catch(e){document.getElementById("coach-envnote").textContent="Note: embedded previews block external calls — if chat fails, open the Pages site or app.";}
  const chips=["Why is that the best play?","What hands beat me here?","When would the other option be right?"];
  const cw=document.getElementById("coach-chips");
  chips.forEach(c=>{const b=document.createElement("button");b.type="button";b.textContent=c;b.onclick=()=>coachAsk(c);cw.appendChild(b);});
  const send=document.getElementById("coach-send"),inp=document.getElementById("coach-input");
  const fire=()=>{const t=inp.value.trim();if(t){inp.value="";coachAsk(t);}};
  send.onclick=fire;
  inp.addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();fire();}});
  coachSettings(false);
}
function coachReset(){coachGen++;coachMsgs=[];coachErr=null;coachBusy=false;coachRender();coachToggleSend();}

// ===== mobile-app shell: view switching, progress, settings =====
function renderProgress(){
  const el=document.getElementById("mastery");if(!el)return;el.innerHTML="";
  [["preflop","Preflop","c-pre"],["flop","Flop","c-flop"],["turn","Turn","c-turn"],["river","River","c-river"]].forEach(function(r){
    const s=stats.street[r[0]]||{n:0,hit:0};const pct=s.n?Math.round(100*s.hit/s.n):0;
    const row=document.createElement("div");row.className="prow";
    const nm=document.createElement("span");nm.className="pn";nm.textContent=r[1];
    const bar=document.createElement("span");bar.className="pbar";
    const i=document.createElement("i");i.className=r[2];i.style.width=pct+"%";bar.appendChild(i);
    const v=document.createElement("span");v.className="pv";v.textContent=s.n?pct+"%":"—";
    row.appendChild(nm);row.appendChild(bar);row.appendChild(v);el.appendChild(row);
  });
}
function setView(v){
  document.querySelectorAll(".view").forEach(function(s){s.classList.toggle("on",s.id==="v-"+v);});
  document.querySelectorAll("#tabbar button").forEach(function(b){b.classList.toggle("on",b.dataset.v===v);});
  if(v==="progress")renderProgress();
  window.scrollTo(0,0);
  try{localStorage.setItem("view",v);}catch(e){}
}
document.querySelectorAll("#tabbar button").forEach(function(b){b.onclick=function(){setView(b.dataset.v);};});
// reduce-motion toggle
let motionOff=false;try{motionOff=localStorage.getItem("motion")==="off";}catch(e){}
function applyMotion(){document.documentElement.classList.toggle("no-motion",motionOff);
  const t=document.getElementById("tog-motion");if(t)t.classList.toggle("on",motionOff);}
document.getElementById("tog-motion").onclick=function(){motionOff=!motionOff;try{localStorage.setItem("motion",motionOff?"off":"on");}catch(e){}applyMotion();};
document.getElementById("reset-btn").onclick=function(){
  if(!confirm("Reset your progress and learned terms?"))return;
  try{localStorage.removeItem("learned");localStorage.removeItem("cat");}catch(e){}location.reload();};
applyMotion();
try{setView(localStorage.getItem("view")||"train");}catch(e){setView("train");}

coachInit();applyModeUI();updateVocab();updateLevelHint();applyCatUI();buildOrder();newHand();
</script>'''

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-demo-packs", action="store_true",
                    help="skip raise/turn-river packs if absent (default: require them)")
    a = ap.parse_args()
    build(allow_missing_demo_packs=a.allow_missing_demo_packs)
