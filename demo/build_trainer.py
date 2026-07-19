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
from collections import defaultdict

DB = "output/packs/flop_pack_v1_fullrange.db"
RAISE_DB = "output/packs/flop_pack_v1_raise_demo.db"   # reduced-range, but HAS fold/call/raise
PER_REASON = 6          # cap questions per reason for variety
MAX_Q = 60
RAISE_Q = 12            # extra 3-action spots blended in to show the raise UX

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
RLAB = {"value": "Value bet", "protection": "Protection", "bluff": "Bluff", "semi_bluff": "Semi-bluff",
        "pot_control": "Pot control", "trap": "Trap", "realization": "Give up / realize equity",
        "value_call": "Value call", "bluff_catch": "Bluff-catch", "call_odds": "Call on odds",
        "raise_value": "Value raise", "raise_bluff": "Bluff raise", "raise_semibluff": "Semi-bluff raise",
        "fold": "Fold", "mixed": "Mixed / close"}


COLS = ("id board node acting_player hand actions ev freq preferred_action "
        "action_grades reason headline detail mixed").split()


def _to_q(d):
    acts = json.loads(d["actions"])
    board = [d["board"][i:i+2] for i in range(0, len(d["board"]), 2)]
    return {
        "board": board, "hero": [d["hand"][0:2], d["hand"][2:4]],
        "node": d["node"], "acting_player": d["acting_player"],
        "situation": SITUATION.get(d["node"], f"You're the {d['acting_player']}."),
        "actions": acts, "labels": {a: ALAB.get(a, a) for a in acts},
        "ev": {k: round(v, 2) for k, v in json.loads(d["ev"]).items()},
        "freq": {k: round(100 * v) for k, v in json.loads(d["freq"]).items()},
        "preferred": d["preferred_action"], "grades": json.loads(d["action_grades"]),
        "reason": d["reason"], "reason_label": RLAB.get(d["reason"], d["reason"]),
        "headline": d["headline"], "detail": json.loads(d["detail"]),
    }


def load_questions():
    c = sqlite3.connect(DB)
    meta = dict(c.execute("SELECT key, value FROM pack_meta").fetchall())
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
    return meta, [_to_q(d) for d in picked]


def load_raise(n=RAISE_Q):
    """A few real fold/call/raise spots from the raise-enabled (reduced-range) pack,
    so the trainer demonstrates the 3-action UX until the full-range raise run lands."""
    if not os.path.exists(RAISE_DB):
        return []
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
    out = []
    for q in (_to_q(d) for d in picked):
        q["demo_raise"] = True     # flag so the UI can note the reduced-range source
        out.append(q)
    return out


def build():
    meta, qs = load_questions()
    raise_qs = load_raise()
    qs = qs + raise_qs
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "local"
    print(f"  ({len(raise_qs)} raise spots blended from the reduced-range raise pack)")
    data = json.dumps(qs, separators=(",", ":"))
    body = TEMPLATE.replace("__DATA__", data).replace("__VERSION__", meta.get("version", "")) \
                   .replace("__RECORDS__", str(meta.get("record_count", ""))).replace("__COMMIT__", commit)
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
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.brand{font-family:var(--disp);font-weight:600;font-size:19px}
.brand .sp{color:var(--brass)}
.score{display:flex;gap:12px;font-family:var(--mono);font-size:12px;color:var(--muted);align-items:baseline}
.score b{color:var(--ink)}
.score .acc{font-size:15px;color:var(--brass);font-weight:700}
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
.next{margin:8px 18px 18px;width:calc(100% - 36px);padding:14px;border:none;border-radius:11px;background:var(--brass);color:#fff;font-family:var(--sans);font-size:15px;font-weight:700;cursor:pointer}
.next:hover{filter:brightness(1.06)}.next:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.foot{margin-top:18px;text-align:center;color:var(--muted);font-size:11.5px;line-height:1.6}
.foot code{font-family:var(--mono)}.foot a{color:var(--brass)}
.hint{font-size:11px;color:var(--muted);text-align:center;margin-top:10px}
kbd{font-family:var(--mono);font-size:10.5px;background:color-mix(in srgb,var(--ink) 8%,transparent);border:1px solid var(--line);border-radius:4px;padding:0 4px}
</style>
<div class="wrap">
  <header>
    <div class="brand"><span class="sp">&spades;</span> Full-Street Flop Trainer</div>
    <div class="score" id="score" hidden>
      <span class="acc" id="acc">—</span>
      <span>played <b id="n">0</b></span>
      <span style="color:var(--best)">solid <b id="solid">0</b></span>
      <span style="color:var(--accept)">ok <b id="ok">0</b></span>
      <span style="color:var(--costly)">leak <b id="leak">0</b></span>
    </div>
  </header>
  <div class="bar-top"><i id="prog" style="width:0"></i></div>

  <div class="card">
    <div class="sit"><span class="pos" id="pos"></span><span id="sit"></span><span class="demo" id="demotag" hidden>raise demo</span></div>
    <div class="felt">
      <div class="cap">Flop</div>
      <div class="cards" id="board"></div>
      <div class="hero"><div class="cap">Your hand</div><div class="cards" id="hero"></div></div>
    </div>
    <div class="acts" id="acts"></div>
    <div class="fb" id="fb">
      <div class="verdict" id="verdict"></div>
      <div class="why">
        <span class="reason" id="reason"></span>
        <p class="head" id="head"></p>
        <ul class="det" id="det"></ul>
      </div>
      <div class="mix"><h4>Solver mix — how often each action is right, and its EV</h4><div id="bars"></div></div>
      <button class="next" id="next">Next hand &nbsp;&#8629;</button>
    </div>
  </div>

  <div class="hint">Pick with <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> · next hand with <kbd>Enter</kbd></div>
  <div class="foot">
    Real solver output — pack <code>__VERSION__</code>, <b>__RECORDS__</b> signed records, build <code>__COMMIT__</code>.
    Every grade &amp; explanation is computed from a full flop&rarr;turn&rarr;river solve; nothing is hand-written.<br>
    Most spots are Check/Bet or Fold/Call (the launch pack). The few <b>Fold/Call/Raise</b> spots
    (marked <span class="demo">raise demo</span>) come from a reduced-range pack — the full-range raise
    run (FR-011) is the next depth pass.<br>
    Prefer to review the answers at a glance? See the <a href="preview.html">content gallery</a>.
  </div>
</div>
<script>
const Q = __DATA__;
const SUIT = {s:["♠",0],h:["♥",1],d:["♦",1],c:["♣",0]};
const VERD = {best:"Best — the top play.",good:"Good — barely gives anything up.",
  acceptable:"OK — playable, not ideal.",costly:"Costly — a recurring leak.",major_error:"Major error — clearly dominated."};
let order=[], pos=0, answered=false, stats={n:0,solid:0,ok:0,leak:0};

function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function card(t){const r=t[0],s=(t[1]||"").toLowerCase(),su=SUIT[s]||[s,0];
  const e=document.createElement("div");e.className="pc"+(su[1]?" red":"");
  e.innerHTML="<b>"+(r==="T"?"10":r)+"</b><i>"+su[0]+"</i>";return e;}
function render(cs,el){el.innerHTML="";cs.forEach(c=>el.appendChild(card(c)));}

function deal(){
  answered=false;
  const q=Q[order[pos]];
  document.getElementById("fb").className="fb";
  const posEl=document.getElementById("pos");posEl.textContent=q.acting_player;posEl.className="pos "+q.acting_player;
  document.getElementById("sit").textContent=q.situation;
  document.getElementById("demotag").hidden=!q.demo_raise;
  render(q.board,document.getElementById("board"));
  render(q.hero,document.getElementById("hero"));
  const box=document.getElementById("acts");box.innerHTML="";
  q.actions.forEach((a,i)=>{
    const b=document.createElement("button");b.className="act";b.dataset.a=a;
    b.innerHTML="<span>"+q.labels[a]+"</span><span class='k'>"+(i+1)+"</span>";
    b.onclick=()=>answer(a);box.appendChild(b);
  });
  document.getElementById("prog").style.width=(100*pos/Q.length)+"%";
}
function answer(a){
  if(answered)return;answered=true;
  const q=Q[order[pos]], g=q.grades[a];
  document.querySelectorAll("#acts .act").forEach(b=>{
    b.disabled=true;const ga=q.grades[b.dataset.a];b.classList.add("g-"+ga);
    if(b.dataset.a===a)b.classList.add("chosen");
  });
  // score
  stats.n++;
  if(g==="best"||g==="good")stats.solid++;else if(g==="acceptable")stats.ok++;else stats.leak++;
  const sc=document.getElementById("score");sc.hidden=false;
  n.textContent=stats.n;solid.textContent=stats.solid;ok.textContent=stats.ok;leak.textContent=stats.leak;
  document.getElementById("acc").textContent=Math.round(100*(stats.solid+stats.ok)/stats.n)+"%";
  // verdict + why
  const v=document.getElementById("verdict");v.className="verdict v-"+g;
  v.innerHTML="<span class='dot'></span>"+VERD[g];
  document.getElementById("reason").textContent=q.reason_label;
  document.getElementById("head").textContent=q.headline;
  const dl=document.getElementById("det");dl.innerHTML="";q.detail.forEach(d=>{const li=document.createElement("li");li.textContent=d;dl.appendChild(li);});
  // solver mix bars
  const bars=document.getElementById("bars");bars.innerHTML="";
  const maxf=Math.max(1,...q.actions.map(x=>q.freq[x]));
  q.actions.slice().sort((x,y)=>q.freq[y]-q.freq[x]).forEach(x=>{
    const ga=q.grades[x],rec=x===q.preferred,you=x===a;
    const row=document.createElement("div");row.className="row g-"+ga;
    row.innerHTML="<div class='rlab'><span class='nm'>"+q.labels[x]+
      (rec?" <span class='star'>&#9733;</span>":"")+(you?" <span class='you'>YOUR PICK</span>":"")+
      "</span><span class='num'>"+q.freq[x]+"%&nbsp;&middot;&nbsp;"+(q.ev[x]>=0?"+":"")+q.ev[x]+" bb"+
      " <span class='tag'>"+ga.replace("_"," ")+"</span></span></div>"+
      "<div class='track'><i style='width:"+Math.max(3,Math.round(100*q.freq[x]/maxf))+"%'></i></div>";
    bars.appendChild(row);
  });
  document.getElementById("fb").className="fb on";
  document.getElementById("next").focus();
}
function next(){pos=(pos+1)%Q.length;if(pos===0)order=shuffle(order.slice());deal();}
document.getElementById("next").onclick=next;
document.addEventListener("keydown",e=>{
  if(!answered){const i=parseInt(e.key);const q=Q[order[pos]];if(i>=1&&i<=q.actions.length)answer(q.actions[i-1]);}
  else if(e.key==="Enter"||e.key===" "){e.preventDefault();next();}
});
order=shuffle([...Q.keys()]);deal();
</script>'''

if __name__ == "__main__":
    build()
