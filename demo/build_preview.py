"""Regenerate the shareable content-preview page from a signed pack (MIT).

Extracts one representative question per reason type from the pack and renders
demo/content_preview.html (Artifact fragment) + index.html (standalone, GitHub
Pages). Run:  PYTHONPATH=src python demo/build_preview.py
"""
import html
import json
import os
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pokertrainer.content_pack import verify_pack

DB = "output/packs/flop_pack_v1_fullrange.db"
DISPLAY_ORDER = ["value", "protection", "bluff", "trap", "pot_control", "realization",
                 "value_call", "bluff_catch", "fold", "mixed", "raise_value", "raise_bluff"]


def _require_verified(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"pack not found: {path}")
    verdict = verify_pack(path)
    if not (verdict.get("hash_ok") and verdict.get("signature_ok")):
        raise SystemExit(f"pack failed integrity check: {path} verify={verdict}")
    return verdict


def load_samples():
    verdict = _require_verified(DB)
    c = sqlite3.connect(DB)
    meta = dict(c.execute("SELECT key, value FROM pack_meta").fetchall())
    # Prefer verified row count over unsigned pack_meta.record_count
    meta["record_count"] = str(verdict["records"])
    cols = ("board node acting_player hand hand_category actions ev freq "
            "preferred_action action_grades ev_sep_pct mixed reason headline detail").split()
    rows = c.execute(f"SELECT {','.join(cols)} FROM flop_decision").fetchall()
    c.close()
    recs = [dict(zip(cols, r)) for r in rows]
    for r in recs:
        for k in ("actions", "ev", "freq", "action_grades", "detail"):
            r[k] = json.loads(r[k])
    # one representative per reason: the clearest teaching example (highest
    # frequency on the preferred action), mixed picks an actual mixed spot.
    picked = {}
    for reason in DISPLAY_ORDER:
        cand = [r for r in recs if r["reason"] == reason]
        if reason == "mixed":
            cand = [r for r in cand if r["mixed"]] or cand
        if not cand:
            continue
        picked[reason] = max(cand, key=lambda r: r["freq"].get(r["preferred_action"], 0))
    samples = [picked[r] for r in DISPLAY_ORDER if r in picked]
    return meta, samples


SIT = {"bb_first": "first to act", "btn_vs_check": "checked to",
       "bb_vs_bet": "facing a 66% bet", "btn_vs_bet": "facing a 66% bet"}
ALAB = {"check": "Check", "bet": "Bet 66%", "fold": "Fold", "call": "Call", "raise": "Raise 3×"}
GLAB = {"best": "Best", "good": "Good", "acceptable": "OK", "costly": "Costly", "major_error": "Major error"}
RLAB = {"value": "Value bet", "protection": "Protection", "bluff": "Bluff", "semi_bluff": "Semi-bluff",
        "pot_control": "Pot control", "trap": "Trap", "realization": "Realization", "value_call": "Value call",
        "bluff_catch": "Bluff-catch", "call_odds": "Call · odds", "raise_value": "Value raise",
        "raise_bluff": "Bluff raise", "raise_semibluff": "Semi-bluff raise", "fold": "Fold", "mixed": "Mixed"}
SUIT = {"s": ("♠", 0), "h": ("♥", 1), "d": ("♦", 1), "c": ("♣", 0)}


def mini(card):
    r, s = card[0], card[1].lower()
    sym, red = SUIT.get(s, (s, 0))
    r = "10" if r == "T" else r
    return f'<span class="pc{" red" if red else ""}"><b>{r}</b><i>{sym}</i></span>'


def cards(cs):
    return "".join(mini(c) for c in cs)


def render(meta, samples):
    cards_html = []
    for s in samples:
        board = [s["board"][i:i + 2] for i in range(0, len(s["board"]), 2)]
        hero = [s["hand"][0:2], s["hand"][2:4]]
        acts = []
        for a in s["actions"]:
            g = s["action_grades"][a]
            rec = " rec" if a == s["preferred_action"] else ""
            fr = round(100 * s["freq"][a])
            ev = s["ev"][a]
            acts.append(f'<div class="act g-{html.escape(g)}{rec}"><span class="al">{html.escape(ALAB.get(a, a))}</span>'
                        f'<span class="ameta"><span class="af">{fr}%</span><span class="ae">{ev:+.2f} bb</span></span>'
                        f'<span class="ag">{html.escape(GLAB.get(g, g))}</span></div>')
        det = "".join(f"<li>{html.escape(x)}</li>" for x in s["detail"])
        reason = html.escape(RLAB.get(s["reason"], s["reason"]))
        actor = html.escape(s["acting_player"])
        node = s["node"]
        sit = html.escape(SIT.get(node, node))
        hc = html.escape(s["hand_category"].replace("_", " "))
        cards_html.append(f'''<article class="q">
  <header class="qh">
    <div class="sit"><span class="pos {actor}">{actor}</span> {sit}</div>
    <span class="hc">{hc}</span>
  </header>
  <div class="felt">
    <div class="crow"><span class="cap">Flop</span><span class="pcs">{cards(board)}</span></div>
    <div class="crow"><span class="cap">Hand</span><span class="pcs">{cards(hero)}</span></div>
  </div>
  <div class="acts">{"".join(acts)}</div>
  <div class="expl">
    <span class="reason">{reason}</span>
    <p class="head">{html.escape(s['headline'])}</p>
    <ul class="det">{det}</ul>
  </div>
</article>''')

    scale = "".join(f'<span class="chip"><span class="dot" style="background:var(--{c})"></span>{l}</span>'
                    for c, l in [("best", "Best"), ("good", "Good"), ("accept", "OK"),
                                 ("costly", "Costly"), ("major", "Major error")])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "local"
    body = f'''<style>{CSS}</style>
<div class="wrap">
  <div class="eyebrow">Solver content preview</div>
  <h1>Full-street flop decisions, in plain language</h1>
  <p class="thesis">A sample of what the trainer ships: every recommendation comes from a complete
  flop-to-turn-to-river solve, graded by EV, and explained in one sentence a casual player understands.</p>
  <hr class="rule">
  <div class="meta">
    <span>Pack <b>{html.escape(str(meta.get('version')))}</b></span>
    <span><b>{html.escape(str(meta.get('record_count')))}</b> decision records</span>
    <span>🔏 <b>Signed</b> &amp; integrity-verified</span>
    <span>Solver <b>full-street CFR+</b></span>
    <span>Build <b>{html.escape(commit)}</b></span>
  </div>

  <div class="key">
    <h2>How to read a card</h2>
    <p>Each card is one real question. The player picks an action; it's graded instantly from the pack's
    precomputed grades, then the reason is shown. The action marked ★ is the solver's pick. Numbers are the
    solver's frequency and expected value (in big blinds) for each action.</p>
    <div class="scale">{scale}</div>
  </div>

  <div class="grid">
    {"".join(cards_html)}
  </div>

  <footer>
    These cover the reasons the trainer teaches — value, protection, bluff, trap, pot control,
    realization, bluff-catch, folds, and mixed spots. Grading and explanations are generated
    deterministically from the solve; nothing here is hand-written strategy.
    <br><br>
    <b>This is the real launch pack:</b> a full-range flop solve across 12 boards
    (BTN vs BB, single-raised pot, 100&nbsp;bb, 66% c-bet) — the same {html.escape(str(meta.get('record_count')))} decisions
    the trainer ships, on a GPU-solved flop→turn→river tree. EVs and frequencies are the actual solver
    output. Raise decisions (FR-011) are a separate run and aren't shown here. Reason labels come from
    <code>explanations.py</code>; grades from EV-regret thresholds stored in the pack.
  </footer>
</div>'''
    os.makedirs("demo", exist_ok=True)
    open("demo/content_preview.html", "w").write(body)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n'
           '<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           '<title>Full-Street Flop Trainer — Content Preview</title>\n'
           '<meta name="description" content="A preview of full-street flop training questions '
           'with solver grades and plain-language explanations.">\n'
           '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    open("preview.html", "w").write(doc)   # gallery served at /preview.html (trainer is the landing)
    return len(cards_html), commit


CSS = '''
:root{
  --bg:#e9ece6; --panel:#ffffff; --ink:#171d19; --muted:#59635c; --line:#dbe0d8;
  --brass:#9a7c41; --brass-soft:#b8975a;
  --best:#2f7d54; --good:#4f8f66; --accept:#b07f2a; --costly:#bf5330; --major:#9c3320;
  --pc-bg:#fcfbf7; --pc-ink:#181818; --pc-red:#bf1d2c; --pc-line:#d9d7cd;
  --disp:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0e1512; --panel:#151d18; --ink:#e6ece7; --muted:#8f9d94; --line:#25302a;
  --brass:#cBa066; --brass-soft:#d8b478;
  --best:#4bb57e; --good:#66b784; --accept:#d1a048; --costly:#e0714e; --major:#cf5138;
  --pc-bg:#f6f4ee; --pc-ink:#181818; --pc-red:#c02636; --pc-line:#cbc9bf;
}}
:root[data-theme="light"]{--bg:#e9ece6;--panel:#ffffff;--ink:#171d19;--muted:#59635c;--line:#dbe0d8;--brass:#9a7c41;--brass-soft:#b8975a;--best:#2f7d54;--good:#4f8f66;--accept:#b07f2a;--costly:#bf5330;--major:#9c3320;--pc-bg:#fcfbf7;--pc-ink:#181818;--pc-red:#bf1d2c;--pc-line:#d9d7cd;}
:root[data-theme="dark"]{--bg:#0e1512;--panel:#151d18;--ink:#e6ece7;--muted:#8f9d94;--line:#25302a;--brass:#cBa066;--brass-soft:#d8b478;--best:#4bb57e;--good:#66b784;--accept:#d1a048;--costly:#e0714e;--major:#cf5138;--pc-bg:#f6f4ee;--pc-ink:#181818;--pc-red:#c02636;--pc-line:#cbc9bf;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:40px 22px 64px}
.eyebrow{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--brass);font-weight:600}
h1{font-family:var(--disp);font-weight:600;font-size:clamp(28px,4.4vw,46px);line-height:1.05;
  margin:.28em 0 .18em;text-wrap:balance;letter-spacing:-.01em}
.thesis{color:var(--muted);font-size:clamp(15px,1.7vw,17.5px);max-width:60ch;margin:0}
.rule{height:1px;background:linear-gradient(90deg,var(--brass),transparent);margin:22px 0;border:0}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;font-family:var(--mono);font-size:12.5px;color:var(--muted)}
.meta b{color:var(--ink);font-weight:600}
.key{display:grid;grid-template-columns:1fr;gap:14px;background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:18px 20px;margin:26px 0 30px}
.key h2{font-family:var(--disp);font-size:16px;margin:0 0 2px;font-weight:600}
.key p{margin:0;color:var(--muted);font-size:13.5px;max-width:70ch}
.scale{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.chip{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:3px 10px}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block}
.grid{display:grid;grid-template-columns:1fr;gap:18px}
@media(min-width:760px){.grid{grid-template-columns:1fr 1fr}}
.q{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 18px 16px;
  display:flex;flex-direction:column;gap:13px;overflow:hidden}
.qh{display:flex;align-items:center;justify-content:space-between;gap:10px}
.sit{font-family:var(--disp);font-size:16.5px;font-weight:600}
.pos{font-family:var(--sans);font-size:11px;font-weight:700;letter-spacing:.05em;padding:2px 7px;border-radius:6px;
  vertical-align:middle;margin-right:5px}
.pos.BB{background:color-mix(in srgb,var(--brass) 20%,transparent);color:var(--brass)}
.pos.BTN{background:color-mix(in srgb,var(--best) 20%,transparent);color:var(--best)}
.hc{font-size:11.5px;color:var(--muted);text-transform:capitalize;font-variant:small-caps;letter-spacing:.03em}
.felt{background:radial-gradient(120% 120% at 50% 0,color-mix(in srgb,var(--best) 22%,var(--panel)),var(--panel));
  border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:flex;flex-direction:column;gap:9px}
.crow{display:flex;align-items:center;gap:12px}
.cap{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);width:38px;flex:none}
.pcs{display:flex;gap:6px}
.pc{background:var(--pc-bg);color:var(--pc-ink);border:1px solid var(--pc-line);border-radius:6px;
  width:34px;height:46px;display:inline-flex;flex-direction:column;align-items:center;justify-content:center;
  box-shadow:0 1px 2px rgba(0,0,0,.18);line-height:1}
.pc b{font-size:16px;font-weight:700;font-family:var(--sans)}
.pc i{font-size:15px;font-style:normal;margin-top:1px}
.pc.red{color:var(--pc-red)}
.acts{display:flex;flex-direction:column;gap:6px}
.act{display:grid;grid-template-columns:1fr auto auto;align-items:center;gap:10px;
  border:1px solid var(--line);border-left:3px solid var(--gc,var(--line));border-radius:8px;padding:8px 11px;position:relative}
.act.g-best{--gc:var(--best)} .act.g-good{--gc:var(--good)} .act.g-acceptable{--gc:var(--accept)}
.act.g-costly{--gc:var(--costly)} .act.g-major_error{--gc:var(--major)}
.act.rec{background:color-mix(in srgb,var(--gc) 9%,var(--panel))}
.al{font-weight:600;font-size:14px}
.act.rec .al::after{content:"★";color:var(--gc);font-size:11px;margin-left:7px;vertical-align:1px}
.ameta{font-family:var(--mono);font-size:12px;color:var(--muted);display:flex;gap:12px;font-variant-numeric:tabular-nums}
.ae{color:var(--ink)}
.ag{font-size:11px;font-weight:700;color:var(--gc);text-align:right;min-width:74px}
.expl{border-top:1px solid var(--line);padding-top:12px;display:flex;flex-direction:column;gap:7px}
.reason{align-self:flex-start;font-size:11px;font-weight:600;letter-spacing:.03em;color:var(--brass);
  border:1px solid color-mix(in srgb,var(--brass) 40%,var(--line));border-radius:999px;padding:2px 10px}
.head{margin:0;font-size:14.5px;font-weight:600}
.det{margin:0;padding-left:17px;color:var(--muted);font-size:12.5px;display:flex;flex-direction:column;gap:2px}
.det li{font-variant-numeric:tabular-nums}
footer{margin-top:34px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:16px;max-width:75ch}
footer code{font-family:var(--mono);font-size:11.5px;background:color-mix(in srgb,var(--ink) 7%,transparent);padding:1px 5px;border-radius:4px}
'''

if __name__ == "__main__":
    meta, samples = load_samples()
    n, commit = render(meta, samples)
    print(f"wrote demo/content_preview.html + preview.html | {n} cards | pack {meta.get('version')} "
          f"({meta.get('record_count')} recs) | build {commit}")
    print("reasons:", [s["reason"] for s in samples])
