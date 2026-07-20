"""Build the self-contained interactive trainer demo (MIT).

Samples real questions from a signed pack and emits a single HTML file that plays
the actual end-user loop — deal a spot, pick an action, get graded + taught why —
with no server. Writes demo/trainer_demo.html + trainer.html (Pages).

Run:  PYTHONPATH=src python demo/build_trainer.py
"""
import html
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pokertrainer.content_pack import verify_pack

DB = "output/packs/flop_pack_v1_fullrange.db"
RAISE_DB = "output/packs/flop_pack_v1_raise_demo.db"   # reduced-range, but HAS fold/call/raise
TR_DB = "output/packs/flop_pack_turnriver_demo.db"     # turn/river decisions (later-street demo)
SB_DB = "output/packs/flop_pack_sb_vs_bb.db"           # 2nd scenario: SB vs BB (full range)
PER_REASON = 6          # cap questions per reason for variety
MAX_Q = 60
RAISE_Q = 12            # extra 3-action spots blended in to show the raise UX
TR_Q = 16               # turn/river spots blended in
SB_Q = 20               # SB-vs-BB spots blended in (2nd position)

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


def _to_q(d, oop_pos="BB"):
    from pokertrainer.explanations import freq_pct_ints
    acts = json.loads(d["actions"])
    board = [d["board"][i:i+2] for i in range(0, len(d["board"]), 2)]
    street = STREET.get(len(d["board"]), "flop")
    node = d["node"]
    # First-to-act on this street only — facing a check/bet is never "act first",
    # even when hero is OOP (they already checked and now face a bet).
    acts_first = node.endswith("_first")
    freq_raw = {k: float(v) for k, v in json.loads(d["freq"]).items()}
    return {
        "board": board, "hero": [d["hand"][0:2], d["hand"][2:4]], "street": street,
        "node": node, "acting_player": d["acting_player"], "acts_first": acts_first,
        "is_oop": d["acting_player"] == oop_pos,
        "actions": acts, "labels": {a: ALAB.get(a, a) for a in acts},
        "ev": {k: round(v, 2) for k, v in json.loads(d["ev"]).items()},
        "freq": freq_pct_ints(freq_raw, order=acts),
        "preferred": d["preferred_action"], "grades": json.loads(d["action_grades"]),
        "reason": d["reason"], "reason_label": RLAB.get(d["reason"], d["reason"]),
        "headline": d["headline"], "detail": json.loads(d["detail"]),
        # Keep mixed so feedback can treat near-indifferent spots as ties rather
        # than punishing the non-starred of two "best" actions.
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
    oop = _oop_pos(picked)
    return meta, [_to_q(d, oop) for d in picked]


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
    oop = _oop_pos(picked)
    out = []
    for q in (_to_q(d, oop) for d in picked):
        q["labels"] = {**q["labels"], "raise": raise_lab}
        q["badge"] = "raise demo"     # flag so the UI can note the reduced-range source
        out.append(q)
    if required and len(out) < 3:
        raise SystemExit(f"raise pack produced only {len(out)} spots (need ≥3)")
    return out


def load_turnriver(n=TR_Q, required=True):
    """Turn + river decisions from the reduced-range later-street demo pack."""
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
    groups = [g[:2] for g in by_key.values()]
    picked = [d for tier in zip_longest(*groups) for d in tier if d is not None][:n]
    oop = _oop_pos(picked)
    out = []
    for q in (_to_q(d, oop) for d in picked):
        q["badge"] = q["street"] + " · demo"
        out.append(q)
    streets = {q["street"] for q in out}
    if required and not ({"turn", "river"} <= streets):
        raise SystemExit(f"turn/river pack missing street coverage: {streets}")
    return out


def load_sb(n=SB_Q, required=False):
    """SB-vs-BB spots (2nd position, full range). Here the SB is out of position
    (acts first) and the BB is in position (acts last) — inverse of BTN-vs-BB."""
    if not os.path.exists(SB_DB):
        if required:
            raise SystemExit(f"SB pack missing ({SB_DB})")
        print(f"  note: SB pack not present ({SB_DB}) — skipping SB spots")
        return []
    _require_verified(SB_DB)
    c = sqlite3.connect(SB_DB)
    rows = c.execute(f"SELECT {','.join(COLS)} FROM flop_decision").fetchall()
    c.close()
    dicts = [dict(zip(COLS, r)) for r in rows]
    oop = _oop_pos(dicts)
    buckets = defaultdict(list)
    for d in dicts:
        buckets[(d["node"], d["reason"])].append(d)
    from itertools import zip_longest
    groups = [g[:2] for g in buckets.values()]
    picked = [d for tier in zip_longest(*groups) for d in tier if d is not None][:n]
    out = []
    for q in (_to_q(d, oop) for d in picked):
        q["badge"] = "SB vs BB"
        out.append(q)
    return out


def build(allow_missing_demo_packs=False):
    meta, qs = load_questions()
    raise_qs = load_raise(required=not allow_missing_demo_packs)
    tr_qs = load_turnriver(required=not allow_missing_demo_packs)
    sb_qs = load_sb()
    qs = qs + raise_qs + tr_qs + sb_qs
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "local"
    print(f"  ({len(raise_qs)} raise + {len(tr_qs)} turn/river + {len(sb_qs)} SB-vs-BB spots blended in)")
    # Escape </script> so pack strings cannot break out of the inline script.
    data = json.dumps(qs, separators=(",", ":")).replace("<", "\\u003c")
    body = TEMPLATE.replace("__DATA__", data).replace("__VERSION__", html.escape(meta.get("version", ""))) \
                   .replace("__RECORDS__", html.escape(str(meta.get("record_count", "")))).replace("__COMMIT__", html.escape(commit))
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
:root{
  --bg:#e9ece6; --panel:#ffffff; --panel2:#f4f5f0; --ink:#171d19; --muted:#59635c; --line:#dbe0d8;
  --brass:#9a7c41; --brass-soft:#b8975a;
  --best:#2f7d54; --good:#4f8f66; --accept:#b07f2a; --costly:#bf5330; --major:#9c3320;
  --pc-bg:#fcfbf7; --pc-ink:#181818; --pc-red:#bf1d2c; --pc-line:#d9d7cd;
  --disp:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0e1512; --panel:#151d18; --panel2:#1b241f; --ink:#e6ece7; --muted:#8f9d94; --line:#25302a;
  --brass:#cba066; --brass-soft:#d8b478;
  --best:#4bb57e; --good:#66b784; --accept:#d1a048; --costly:#e0714e; --major:#cf5138;
  --pc-bg:#f6f4ee; --pc-ink:#181818; --pc-red:#c02636; --pc-line:#cbc9bf;
}}
:root[data-theme="light"]{--bg:#e9ece6;--panel:#ffffff;--panel2:#f4f5f0;--ink:#171d19;--muted:#59635c;--line:#dbe0d8;--brass:#9a7c41;--best:#2f7d54;--good:#4f8f66;--accept:#b07f2a;--costly:#bf5330;--major:#9c3320;--pc-bg:#fcfbf7;--pc-ink:#181818;--pc-red:#bf1d2c;--pc-line:#d9d7cd;}
:root[data-theme="dark"]{--bg:#0e1512;--panel:#151d18;--panel2:#1b241f;--ink:#e6ece7;--muted:#8f9d94;--line:#25302a;--brass:#cba066;--best:#4bb57e;--good:#66b784;--accept:#d1a048;--costly:#e0714e;--major:#cf5138;--pc-bg:#f6f4ee;--pc-ink:#181818;--pc-red:#c02636;--pc-line:#cbc9bf;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
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
.bar-top{height:4px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:16px}
.bar-top>i{display:block;height:100%;background:var(--brass);transition:width .3s}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.sit{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:9px;font-family:var(--disp);font-size:16px}
.pos{font-family:var(--sans);font-size:11px;font-weight:700;letter-spacing:.05em;padding:2px 8px;border-radius:6px;flex:none}
.pos.BB,.pos.SB{background:color-mix(in srgb,var(--brass) 20%,transparent);color:var(--brass)}
.pos.BTN{background:color-mix(in srgb,var(--best) 20%,transparent);color:var(--best)}
.demo{margin-left:auto;font-size:9.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--accept);border:1px solid color-mix(in srgb,var(--accept) 45%,var(--line));border-radius:6px;padding:1px 6px}
.felt{background:radial-gradient(120% 130% at 50% -10%,color-mix(in srgb,var(--best) 20%,var(--panel)),var(--panel));padding:20px 18px 18px;text-align:center}
.cap{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.cards{display:flex;gap:8px;justify-content:center}
.pc{background:var(--pc-bg);color:var(--pc-ink);border:1px solid var(--pc-line);border-radius:7px;width:46px;height:62px;display:inline-flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:0 2px 5px rgba(0,0,0,.22);line-height:1}
.pc b{font-size:22px;font-weight:700}.pc i{font-size:19px;font-style:normal;margin-top:1px}
.pc.red{color:var(--pc-red)}
.hero{margin-top:16px}
.hero .cap{color:var(--brass);font-weight:600}
.acts{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;padding:16px 18px}
.act{appearance:none;font-family:var(--sans);font-size:15px;font-weight:600;color:var(--ink);background:var(--panel2);
  border:1px solid var(--line);border-radius:11px;padding:15px 10px;cursor:pointer;transition:.12s;display:flex;flex-direction:column;gap:2px;align-items:center}
.act .k{font-family:var(--mono);font-size:10px;color:var(--muted);font-weight:400}
.act:hover:not(:disabled){border-color:var(--brass);background:color-mix(in srgb,var(--brass) 8%,var(--panel2));transform:translateY(-1px)}
.act:focus-visible{outline:2px solid var(--brass);outline-offset:2px}
.act:disabled{cursor:default;opacity:.9}
.act.chosen{box-shadow:inset 0 0 0 2px var(--gc,var(--brass))}
.act.g-best{--gc:var(--best)}.act.g-good{--gc:var(--good)}.act.g-acceptable{--gc:var(--accept)}
.act.g-costly{--gc:var(--costly)}.act.g-major_error{--gc:var(--major)}
/* feedback */
.fb{display:none;border-top:1px solid var(--line)}
.fb.on{display:block;animation:rise .25s ease}
@keyframes rise{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.fb.on{animation:none}.act:hover:not(:disabled){transform:none}}
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
.intro summary{cursor:pointer;font-weight:700;padding:13px 0;font-size:13.5px;color:var(--brass);list-style:none}
.intro summary::-webkit-details-marker{display:none}
.intro p{margin:0 0 10px;font-size:13px;color:var(--ink);line-height:1.6}
.intro p:first-of-type{margin-top:2px}
.intro b{color:var(--ink)}
</style>
<div class="wrap">
  <header>
    <div class="brand"><span class="sp">&spades;</span> Full-Street Flop Trainer</div>
    <div class="score" id="score" hidden>
      <span class="acc" id="acc">—</span>
      <span class="sbits"><b id="n">0</b> played · <b class="cSolid" id="solid">0</b> solid ·
        <b class="cOk" id="ok">0</b> ok · <b class="cLeak" id="leak">0</b> leak</span>
    </div>
  </header>
  <div class="controls">
    <div class="level">
      <span class="lvl-cap">Language</span>
      <div class="lang" id="lang" role="group" aria-label="Language level">
        <button data-m="progressive" type="button">Adaptive</button><button data-m="plain" type="button">Beginner</button><button data-m="learning" type="button">Learning</button><button data-m="poker" type="button">Pro</button>
      </div>
      <span class="vocab" id="vocab" hidden></span>
    </div>
    <p class="lvl-hint" id="levelhint"></p>
  </div>
  <div class="bar-top"><i id="prog" style="width:0"></i></div>

  <details class="intro" id="intro">
    <summary>🔰 New to poker? Start here (30 seconds)</summary>
    <p>You and one opponent each get <b>2 secret cards</b> (only you see yours). Then <b>5 shared cards</b>
      everyone can use are dealt in stages: <b>3 at once</b>, then a <b>4th</b>, then a <b>5th</b>. You make your
      best five-card hand from your 2 cards plus the shared ones.</p>
    <p>At each stage you choose: <b>Check</b> (pass, bet nothing), <b>Bet</b> (put chips in), <b>Call</b>
      (match a bet), <b>Raise</b> (bet even more), or <b>Fold</b> (give up the hand). The chips already in the
      middle are the <b>pot</b> — that's what you're playing for.</p>
    <p>One player <b>acts first</b>, the other <b>acts last</b>. Acting last is an advantage — you see what your
      opponent does before deciding. The trainer tells you which you are each hand.</p>
    <p>Your job: pick the action a strong player would — graded instantly, told why. The <b>Adaptive</b> setting
      (top) starts in plain words and <b>teaches you the poker terms as you play them well</b> — each one you earn
      is added to your vocabulary (🔓 counter). Prefer a fixed level? Switch to <b>Beginner</b>, <b>Learning</b>,
      or <b>Pro</b> any time.</p>
  </details>

  <div class="card">
    <div class="sit"><span class="pos" id="pos"></span><span id="sit"></span><span class="demo" id="demotag" hidden>raise demo</span></div>
    <div class="felt">
      <div class="cap" id="boardcap">Flop</div>
      <div class="cards" id="board"></div>
      <div class="hero"><div class="cap" id="herocap">Your hand</div><div class="cards" id="hero"></div></div>
    </div>
    <div class="acts" id="acts"></div>
    <div class="fb" id="fb">
      <div class="verdict" id="verdict"></div>
      <div class="unlock" id="unlock" hidden></div>
      <div class="why">
        <p class="read" id="read"></p>
        <p class="stand" id="stand" hidden></p>
        <span class="reason" id="reason"></span>
        <p class="head" id="head"></p>
        <p class="cost" id="cost" hidden></p>
        <ul class="det" id="det"></ul>
      </div>
      <div class="mix"><h4 id="mixhead"></h4><div id="bars"></div></div>
      <button class="next" id="next">Next hand &nbsp;&#8629;</button>
    </div>
  </div>

  <div class="hint">Pick with <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> · next hand with <kbd>Enter</kbd></div>
  <div class="foot">
    Real solver output — pack <code>__VERSION__</code>, <b>__RECORDS__</b> signed records, build <code>__COMMIT__</code>.
    Every grade &amp; explanation is computed from a full flop&rarr;turn&rarr;river solve; nothing is hand-written.<br>
    Flop spots come from the full-range launch pack. Spots marked <span class="demo">raise demo</span>
    (Fold/Call/Raise) and <span class="demo">turn / river</span> (later-street boards, reduced range,
    unconditioned — not check-check filtered) are real solver output from demo packs — the full-range
    raise + turn/river passes are the next depth work.<br>
    Prefer to review the answers at a glance? See the <a href="preview.html">content gallery</a>.
  </div>
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
      <dd>Where you sit relative to the dealer. Postflop, the player in position (IP) acts last — usually the Button. Out of position (OOP) acts first. In blind-vs-blind, the BB is IP.</dd>
      <dt>In / out of position</dt>
      <dd>Whether you act last (in position) or first (out of position) on each street.</dd>
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
</div>
<script>
const Q = __DATA__;
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
  if(!comp.length)return null;return comp.length>=2?"an open-ended straight draw":"a gutshot straight draw";}
function handRead(hero,board){
  const hs=hero.map(c=>c[1]),bs=board.map(c=>c[1]);
  const hv=hero.map(c=>RV[c[0]]),bv=board.map(c=>RV[c[0]]);
  const allV=[...hv,...bv],allS=[...hs,...bs],river=board.length>=5;
  const cnt={};allV.forEach(v=>cnt[v]=(cnt[v]||0)+1);
  const groups=Object.keys(cnt).map(Number).sort((a,b)=>cnt[b]-cnt[a]||b-a);
  const suitCnt={};allS.forEach(s=>suitCnt[s]=(suitCnt[s]||0)+1);
  const flush=Object.keys(suitCnt).some(s=>suitCnt[s]>=5);
  const straight=hasStraight(allV),maxB=Math.max(...bv),sortB=[...new Set(bv)].sort((a,b)=>b-a),pocket=hv[0]===hv[1];
  const top=cnt[groups[0]],second=cnt[groups[1]]||0;
  let made,strength=null,cat="high",pairKind=null,overs=[];
  if(flush&&straight){made="a straight flush";cat="sflush";}
  else if(top===4){made="four of a kind ("+MANY[groups[0]]+")";cat="quads";}
  else if(top===3&&second>=2){made="a full house";cat="full";}
  else if(flush){made="a flush";cat="flush";}
  else if(straight){made="a straight";cat="straight";}
  else if(top===3){made=(pocket&&hv[0]===groups[0])?"a set of "+MANY[groups[0]]:"three "+MANY[groups[0]];cat="trips";}
  else if(top===2&&second===2){made="two pair";cat="twopair";}
  else if(top===2){const pr=groups[0];made="a pair of "+MANY[pr];cat="pair";
    overs=[...new Set(bv.filter(v=>v>pr))].sort((a,b)=>b-a);
    if(pocket&&hv[0]===pr){if(pr>maxB){pairKind="over";strength="an overpair (higher than every board card)";}
      else{pairKind="under";strength="the "+ONE[maxB]+" on the board outranks it";}}
    else if(pr===sortB[0]){pairKind="top";strength="top pair (you matched the highest board card)";}
    else if(pr===sortB[1]){pairKind="mid";strength="middle pair";}
    else{pairKind="low";strength="a low pair";}}
  else made=ONE[Math.max(...hv)]+" high (no pair)";
  let draw=null;
  if(!river&&!flush&&!straight){const parts=[];
    if(Object.keys(suitCnt).some(s=>suitCnt[s]===4&&hs.includes(s)))parts.push("a flush draw (four to a flush)");
    const sd=straightDraw(allV);if(sd)parts.push(sd);
    if(parts.length)draw=parts.join(" and ");}
  return {made,strength,draw,cat,pairKind,overs};
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
  switch(rd.cat){
    case "pair":
      if(rd.pairKind==="over")return "You're ahead of every worse pair and all the bluffs — mostly just two pair or a set beats you now.";
      if(rd.pairKind==="top")return "You beat worse pairs and the draws — a better kicker, two pair, or a set has you beat.";
      return "You beat high cards and bluffs, but "+orList(rd.overs)+" makes a better pair, and two pair or a set is ahead too.";
    case "twopair":return "You're ahead of every one-pair hand — mainly a set or better beats you.";
    case "trips":return "Very strong — only a straight, flush, or full house could beat you, and only if the board allows it.";
    case "straight":return "A big hand — only a flush or a full house beats you here.";
    case "flush":return "A big hand — only a full house or better beats you.";
    case "full":case "quads":case "sflush":return "You've got a monster — just about nothing beats this.";
    default:return rd.draw
      ? "Nothing made yet, but your draw can still get there — for now you're behind any pair."
      : "No pair yet — you're behind almost any made hand; you'd need to improve or get them to fold.";
  }
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
    reason:{value:"Bet a strong hand to get paid",protection:"Bet so drawing hands pay to chase",
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
    reason:{value:"Value bet — get paid by worse",protection:"Protection — charge the draws",
      bluff:"Bluff — make better hands fold",semi_bluff:"Semi-bluff — bet a hand that can improve",
      pot_control:"Pot control — keep it small",trap:"Trap — check a monster",
      realization:"Realize equity — take a free card",value_call:"Value call — you're ahead",
      bluff_catch:"Bluff-catch — you beat their bluffs",call_odds:"Call on odds — right price to draw",
      raise_value:"Value raise — build the pot",raise_bluff:"Bluff raise — make them fold",
      raise_semibluff:"Semi-bluff raise — a draw",fold:"Fold — not strong enough",mixed:"Mixed — any is fine"},
    ev:"EV",boardcap:{flop:"Flop (first 3 shared cards)",turn:"Turn (4th card)",river:"River (5th card)"},
    herocap:"Your hand (2 hole cards)"}
};
let order=[], pos=0, answered=false, cur=null, chosen=null, stats={n:0,solid:0,ok:0,leak:0};
let mode=(function(){try{const m=localStorage.getItem("lang");return (m==="poker"||m==="learning"||m==="plain"||m==="progressive")?m:"progressive";}catch(e){return "progressive";}})();
// Adaptive mode: each concept shows in plain words until you've EARNED it (played a
// spot that uses it well); then it graduates to the poker term + its meaning.
let learned=(function(){try{return new Set(JSON.parse(localStorage.getItem("learned")||"[]"));}catch(e){return new Set();}})();
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
  protection:"You're probably best, but cards could come that beat you — bet so the chasing hands have to pay.",
  bluff:"You won't win if you just show it down, so bet to push better hands into folding.",
  semi_bluff:"Betting can make better hands fold now — and if you're called, your hand can still improve to the best.",
  pot_control:"A decent hand, but not strong enough to build a big pot — check to keep it small and cheap.",
  trap:"You're very strong here — checking hides it and lets your opponent bluff or catch up before you pounce.",
  realization:"Not much yet — check to see the next card for free instead of throwing chips in.",
  value_call:"You're ahead of enough of their betting hands — call to keep collecting from the worse ones.",
  bluff_catch:"Your hand beats the ones they'd bluff with — call to catch those bluffs.",
  call_odds:"Your draw is cheap enough to chase here — call and try to complete it.",
  fold:"There isn't enough here to keep going — fold and save your chips for a better spot.",
  raise_value:"You're strong — raise to build a bigger pot while the worse hands pay.",
  raise_bluff:"Raising tells the story of a big hand — do it to pressure them into folding.",
  raise_semibluff:"Raise: you can fold out better hands now, and still improve if they call.",
  mixed:"This one's genuinely close — either play is fine here."
};
function plainHead(q){return PLAIN_HEAD[q.reason]||TERMS.plain.reason[q.reason]||q.headline;}
function situation(q){
  const first=q.node.endsWith("_first"), vscheck=q.node.endsWith("_vs_check");
  const sm=eff("positions");
  if(sm==="plain"){
    if(first) return "It's your turn, and you go first — you decide before your opponent does.";
    if(vscheck) return "Your opponent passed (checked) to you. It's your turn.";
    return "Your opponent just put chips in (bet). It's on you — match it, put in even more, or give up?";
  }
  const pre=q.street==="turn"?"On the turn, ":q.street==="river"?"On the river, ":"On the flop, ";
  // vs_bet: OOP checked then faces a bet; IP faces an opponent lead (not a c-bet).
  const betRole=q.is_oop
    ? " — you checked and face a bet."
    : " — they led into you.";
  if(sm==="learning"){
    const who="you're the "+q.acting_player+" (you act "+(q.acts_first?"first":"last")+")";
    const role=first?", first to act.":vscheck?" — it's checked to you.":betRole;
    return pre+who+role;
  }
  const role=first?", first to act.":vscheck?" and it's checked to you.":betRole;
  return pre+"you're the "+q.acting_player+role;
}

function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function card(t){const r=t[0],s=(t[1]||"").toLowerCase(),su=SUIT[s]||[s,0];
  const e=document.createElement("div");e.className="pc"+(su[1]?" red":"");
  const b=document.createElement("b");b.textContent=(r==="T"?"10":r);
  const i=document.createElement("i");i.textContent=su[0];
  e.appendChild(b);e.appendChild(i);return e;}
function render(cs,el){el.innerHTML="";cs.forEach(c=>el.appendChild(card(c)));}

function renderQuestion(q){
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
  document.getElementById("prog").style.width=(100*pos/Q.length)+"%";
}
function deal(){answered=false;chosen=null;cur=Q[order[pos]];document.getElementById("fb").className="fb";renderQuestion(cur);}

function renderFeedback(q,a,gained){
  document.querySelectorAll("#acts .act").forEach(b=>{
    b.disabled=true;const ga=q.grades[b.dataset.a];b.className="act g-"+ga;
    if(b.dataset.a===a)b.classList.add("chosen");
  });
  const g=q.grades[a],pref=q.preferred;
  const you=shortAct(a),best=shortAct(pref);
  const v=document.getElementById("verdict");v.className="verdict v-"+g;
  v.textContent="";const dot=document.createElement("span");dot.className="dot";v.appendChild(dot);
  // Key off the GRADE, not a===pref: a co-best action (graded "best" but not the
  // single top-EV one) must still read as correct, not as a leak. Preferred is
  // max-EV (freq tie-break); mixed spots get soft copy when the pick is close.
  let vmsg;
  if(g==="best")vmsg=(a===pref)?"✓ "+you+" — the best play here.":"✓ "+you+" — also a top play here.";
  else if(q.mixed&&(g==="good"||g==="acceptable"))vmsg="✓ "+you+" — close enough; any play is fine here (listed preferred: "+best+").";
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
  // "Where you stand" — plain relative strength. Beginner-oriented, so hide it in Pro.
  const standEl=document.getElementById("stand");
  if(mode==="poker"){standEl.hidden=true;}
  else{standEl.hidden=false;standEl.textContent=standingText(rd);}
  // explanation adapts to the level: Beginner = plain 'why' only; Learning = term
  // tag + explaining headline; Pro = term tag + richer baked headline + bullets.
  const rm=eff("reason:"+q.reason);
  const unit=(rm==="plain")?"chips":"bb";     // plain mode avoids the "bb" jargon
  const rp=document.getElementById("reason");
  if(rm==="plain"){rp.style.display="none";}
  else{rp.style.display="";rp.textContent=TERMS.poker.reason[q.reason]||q.reason;}
  document.getElementById("head").textContent=(rm==="poker")?q.headline:(rm==="plain")?plainHead(q):(TERMS[rm].reason[q.reason]||q.headline);
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
  document.getElementById("mixhead").textContent=(unit==="chips")
    ?"How the solver plays it — how often each action is right, and its average payoff"
    :"Solver mix — how often each action is right, and its EV";
  const bars=document.getElementById("bars");bars.innerHTML="";
  const maxf=Math.max(1,...q.actions.map(x=>q.freq[x]));
  q.actions.slice().sort((x,y)=>q.freq[y]-q.freq[x]).forEach(x=>{
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
    num.appendChild(document.createTextNode(q.freq[x]+"% · "+(ev>=0?"+":"")+ev+" "+unit+" "));
    const tag=document.createElement("span");tag.className="tag";tag.textContent=ga.replace("_"," ");
    num.appendChild(tag);
    rlab.appendChild(nm);rlab.appendChild(num);
    const track=document.createElement("div");track.className="track";
    const i=document.createElement("i");i.style.width=Math.max(3,Math.round(100*q.freq[x]/maxf))+"%";
    track.appendChild(i);row.appendChild(rlab);row.appendChild(track);bars.appendChild(row);
  });
  document.getElementById("fb").className="fb on";
}
function answer(a){
  if(answered)return;answered=true;chosen=a;
  const g=cur.grades[a];
  stats.n++;
  if(g==="best"||g==="good")stats.solid++;else if(g==="acceptable")stats.ok++;else stats.leak++;
  document.getElementById("score").hidden=false;
  document.getElementById("n").textContent=stats.n;document.getElementById("solid").textContent=stats.solid;
  document.getElementById("ok").textContent=stats.ok;document.getElementById("leak").textContent=stats.leak;
  document.getElementById("acc").textContent=Math.round(100*(stats.solid+stats.ok)/stats.n)+"%";
  const gained=tryUnlock(cur,g);
  if(gained.length)renderQuestion(cur);  // this hand's buttons/situation graduate too, in sync with the unlock
  renderFeedback(cur,a,gained);
  document.getElementById("next").focus();
}
function next(){pos=(pos+1)%Q.length;if(pos===0)order=shuffle(order.slice());deal();}

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
  if(t==="positions")return "Positions — who acts when. In position (IP) acts last (usually the Button). Out of position (OOP) acts first. In blind-vs-blind, the BB is IP.";
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
// intro: open by default, remember if the reader dismisses it
const intro=document.getElementById("intro");
try{intro.open=localStorage.getItem("introClosed")!=="1";}catch(e){intro.open=true;}
intro.addEventListener("toggle",()=>{try{localStorage.setItem("introClosed",intro.open?"0":"1");}catch(e){}});

document.getElementById("next").onclick=next;
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="SUMMARY")return;
  if(!answered){const i=parseInt(e.key);if(cur&&i>=1&&i<=cur.actions.length)answer(cur.actions[i-1]);}
  else if(e.key==="Enter"||e.key===" "){e.preventDefault();next();}
});
applyModeUI();updateVocab();updateLevelHint();order=shuffle([...Q.keys()]);deal();
</script>'''

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-demo-packs", action="store_true",
                    help="skip raise/turn-river packs if absent (default: require them)")
    a = ap.parse_args()
    build(allow_missing_demo_packs=a.allow_missing_demo_packs)
