"""Build the self-contained pre-flop trainer (MIT).

Generates a single HTML file that drills the two core pre-flop decisions — opening
(raise-first-in) and defending the Big Blind vs an open — from the calibrated 6-max
ranges. Plain-language explanations in the same voice as the postflop trainer.

Run:  PYTHONPATH=src python demo/build_preflop_trainer.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pokertrainer.preflop_content import build_questions  # noqa: E402



TEMPLATE = r'''<style>
:root{--bg:#e9ece6;--panel:#fff;--panel2:#f4f5f0;--ink:#171d19;--muted:#59635c;--line:#dbe0d8;
 --brass:#9a7c41;--best:#2f7d54;--good:#4f8f66;--costly:#bf5330;--major:#9c3320;
 --disp:"Iowan Old Style",Palatino,Georgia,serif;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace;
 --pc-bg:#fcfbf7;--pc-ink:#181818;--pc-red:#bf1d2c;--pc-line:#d9d7cd;}
@media (prefers-color-scheme:dark){:root{--bg:#0e1512;--panel:#151d18;--panel2:#1b241f;--ink:#e6ece7;--muted:#8f9d94;--line:#25302a;
 --brass:#cba066;--best:#4bb57e;--good:#66b784;--costly:#e0714e;--major:#cf5138;--pc-bg:#f6f4ee;--pc-red:#c02636;--pc-line:#cbc9bf;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:600px;margin:0 auto;padding:20px 16px 56px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.brand{font-family:var(--disp);font-weight:600;font-size:19px}.brand .sp{color:var(--brass)}
.score{display:flex;gap:10px;align-items:baseline}.score .acc{font-family:var(--mono);font-size:18px;color:var(--brass);font-weight:700}
.score .sbits{font-size:11.5px;color:var(--muted)}
.tag{font-size:11.5px;color:var(--muted);margin-bottom:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.sit{padding:15px 18px;border-bottom:1px solid var(--line);font-family:var(--disp);font-size:16px;display:flex;align-items:center;gap:9px}
.pos{font-family:var(--sans);font-size:11px;font-weight:700;letter-spacing:.05em;padding:2px 8px;border-radius:6px;background:color-mix(in srgb,var(--brass) 20%,transparent);color:var(--brass);flex:none}
.felt{background:radial-gradient(120% 130% at 50% -10%,color-mix(in srgb,var(--best) 20%,var(--panel)),var(--panel));padding:24px 18px;text-align:center}
.cap{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--brass);font-weight:600;margin-bottom:10px}
.cards{display:flex;gap:8px;justify-content:center}
.pc{background:var(--pc-bg);color:var(--pc-ink);border:1px solid var(--pc-line);border-radius:7px;width:52px;height:70px;display:inline-flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:0 2px 5px rgba(0,0,0,.22);line-height:1}
.pc b{font-size:25px;font-weight:700}.pc i{font-size:21px;font-style:normal;margin-top:2px}.pc.red{color:var(--pc-red)}
.acts{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;padding:16px 18px}
.act{appearance:none;font-family:var(--sans);font-size:15px;font-weight:600;color:var(--ink);background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:15px 10px;cursor:pointer;transition:.12s;display:flex;flex-direction:column;gap:2px;align-items:center}
.act .k{font-family:var(--mono);font-size:10px;color:var(--muted);font-weight:400}
.act:hover:not(:disabled){border-color:var(--brass);background:color-mix(in srgb,var(--brass) 8%,var(--panel2));transform:translateY(-1px)}
.act:focus-visible{outline:2px solid var(--brass);outline-offset:2px}.act:disabled{cursor:default;opacity:.95}
.act.right{box-shadow:inset 0 0 0 2px var(--best)}.act.wrong{box-shadow:inset 0 0 0 2px var(--costly)}.act.ok{box-shadow:inset 0 0 0 2px var(--good)}
.fb{display:none;border-top:1px solid var(--line)}.fb.on{display:block;animation:f .2s ease}@keyframes f{from{opacity:0}to{opacity:1}}
@media(prefers-reduced-motion:reduce){.fb.on{animation:none}.act:hover:not(:disabled){transform:none}}
.verdict{padding:13px 18px;font-weight:700;font-size:15px;color:var(--vc)}
.v-ok{--vc:var(--best)}.v-no{--vc:var(--costly)}
.why{padding:2px 18px 6px}.read{margin:0 0 6px;font-size:14.5px;font-weight:600}.read b{color:var(--brass)}
.head{margin:0 0 8px;font-size:14.5px}
.rule{margin:6px 0 4px;font-size:12.5px;color:var(--ink);padding:8px 11px;border-radius:8px;background:color-mix(in srgb,var(--best) 9%,transparent)}
.rule b{display:block;color:var(--best);text-transform:uppercase;font-size:10px;letter-spacing:.1em;margin-bottom:2px}
.next{margin:10px 18px 18px;width:calc(100% - 36px);padding:14px;border:none;border-radius:11px;background:var(--brass);color:#fff;font-size:15px;font-weight:700;cursor:pointer}
.next:hover{filter:brightness(1.06)}.next:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.hint{font-size:11px;color:var(--muted);text-align:center;margin-top:10px}
kbd{font-family:var(--mono);font-size:10.5px;background:color-mix(in srgb,var(--ink) 8%,transparent);border:1px solid var(--line);border-radius:4px;padding:0 4px}
.foot{margin-top:16px;text-align:center;color:var(--muted);font-size:11.5px;line-height:1.6}.foot a{color:var(--brass)}
</style>
<div class="wrap">
 <header>
  <div class="brand"><span class="sp">&spades;</span> Pre-flop Trainer</div>
  <div class="score" id="score"><span class="acc" id="acc">&mdash;</span>
   <span class="sbits"><b id="n">0</b> played &middot; <b id="ok">0</b> right</span></div>
 </header>
 <div class="tag">Should you play this hand before the flop? Open, defend the blinds, or handle a 3-bet — the core pre-flop decisions. Calibrated 6-max ranges.</div>
 <div class="card">
  <div class="sit"><span class="pos" id="pos"></span><span id="sit"></span></div>
  <div class="felt"><div class="cap">Your hand</div><div class="cards" id="hand"></div></div>
  <div class="acts" id="acts"></div>
  <div class="fb" id="fb">
   <div class="verdict" id="verdict"></div>
   <div class="why">
    <p class="read" id="read"></p>
    <p class="head" id="head"></p>
    <p class="rule" id="rule"></p>
   </div>
   <button class="next" id="next">Next hand &nbsp;&#8629;</button>
  </div>
 </div>
 <div class="hint">Pick with <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> &middot; next with <kbd>Enter</kbd></div>
 <div class="foot">Solver-approximate ranges, tuned to standard 6-max opening frequencies &mdash; our own strength/playability ordering, not copied charts. 100bb, no rake.</div>
</div>
<script>
const Q=__DATA__;
const SUIT={s:["♠",0],h:["♥",1],d:["♦",1],c:["♣",0]};
const ALAB={fold:"Fold",open:"Raise (open)",call:"Call","3bet":"Re-raise (3-bet)","4bet":"4-bet"};
let order=[],pos=0,answered=false,cur=null,stats={n:0,ok:0};
function card(t){const r=t[0],s=(t[1]||"").toLowerCase(),su=SUIT[s]||[s,0];
 const e=document.createElement("div");e.className="pc"+(su[1]?" red":"");
 const b=document.createElement("b");b.textContent=(r==="T"?"10":r);const i=document.createElement("i");i.textContent=su[0];
 e.appendChild(b);e.appendChild(i);return e;}
function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function render(q){
 document.getElementById("pos").textContent=q.pos;
 document.getElementById("sit").textContent=q.situation;
 const h=document.getElementById("hand");h.innerHTML="";q.hand.forEach(c=>h.appendChild(card(c)));
 const box=document.getElementById("acts");box.innerHTML="";
 q.actions.forEach((a,i)=>{const b=document.createElement("button");b.className="act";b.dataset.a=a;
  const l=document.createElement("span");l.textContent=ALAB[a]||a;const k=document.createElement("span");k.className="k";k.textContent=String(i+1);
  b.appendChild(l);b.appendChild(k);b.onclick=()=>answer(a);box.appendChild(b);});
}
function deal(){answered=false;cur=Q[order[pos]];document.getElementById("fb").className="fb";render(cur);}
function answer(a){
 if(answered)return;answered=true;
 const correct=a===cur.answer;
 const closeOk=!correct&&cur.mixed&&a===cur.alt;   // picked the acceptable close alternative
 stats.n++;if(correct||closeOk)stats.ok++;
 document.getElementById("n").textContent=stats.n;document.getElementById("ok").textContent=stats.ok;
 document.getElementById("acc").textContent=Math.round(100*stats.ok/stats.n)+"%";
 document.querySelectorAll("#acts .act").forEach(b=>{b.disabled=true;
  if(b.dataset.a===cur.answer)b.classList.add("right");
  else if(cur.mixed&&b.dataset.a===cur.alt)b.classList.add("ok");   // also acceptable on a close spot
  else if(b.dataset.a===a)b.classList.add("wrong");});
 const v=document.getElementById("verdict");
 if(correct){v.className="verdict v-ok";
  v.textContent=cur.mixed?("✓ "+ALAB[a]+" — right, and it's close ("+ALAB[cur.alt].toLowerCase()+" is fine too).")
   :("✓ Correct — "+ALAB[a].toLowerCase()+".");}
 else if(closeOk){v.className="verdict v-ok";
  v.textContent="≈ Close — "+ALAB[a].toLowerCase()+" is fine here; "+ALAB[cur.answer].toLowerCase()+" is the small favourite.";}
 else{v.className="verdict v-no";v.textContent="✗ Not quite — the play is "+ALAB[cur.answer]+".";}
 const rd=document.getElementById("read");rd.innerHTML="";rd.appendChild(document.createTextNode("You held "));
 const bcls=document.createElement("b");bcls.textContent=cur.read;rd.appendChild(bcls);rd.appendChild(document.createTextNode("."));
 document.getElementById("head").textContent=cur.why;
 const ru=document.getElementById("rule");ru.innerHTML="";const lb=document.createElement("b");lb.textContent="Rule of thumb";ru.appendChild(lb);ru.appendChild(document.createTextNode(cur.rule));
 document.getElementById("fb").className="fb on";document.getElementById("next").focus({preventScroll:true});
}
function next(){pos=(pos+1)%Q.length;if(pos===0)order=shuffle(order.slice());deal();}
document.getElementById("next").onclick=next;
document.addEventListener("keydown",e=>{if(!answered){const i=parseInt(e.key);if(cur&&i>=1&&i<=cur.actions.length)answer(cur.actions[i-1]);}
 else if(e.key==="Enter"||e.key===" "){e.preventDefault();next();}});
order=shuffle([...Q.keys()]);deal();
</script>'''


def build():
    qs = build_questions()
    body = TEMPLATE.replace("__DATA__", json.dumps(qs, separators=(",", ":")))
    os.makedirs("demo", exist_ok=True)
    open("demo/preflop_trainer.html", "w").write(body)
    doc = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           '<title>Pre-flop Trainer</title></head><body>\n' + body + '\n</body></html>\n')
    open("preflop.html", "w").write(doc)
    print(f"wrote demo/preflop_trainer.html + preflop.html | {len(qs)} questions")


if __name__ == "__main__":
    build()
