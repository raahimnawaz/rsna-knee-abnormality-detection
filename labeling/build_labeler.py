"""Generate a self-contained offline labeling UI from labeling_sample.csv.

Everything is inlined into one HTML file (no server, no fetch, works from file://).
Progress autosaves to localStorage; export to CSV when done or to back up.
"""
import pandas as pd, json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

ROOT = PROJ / "labeling"
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
KEYS = ["1","2","3","4","5","6","7","8","9","0","-","="]

df = pd.read_csv(ROOT / "labeling_sample.csv")
gloss = json.loads((ROOT / "glossary.json").read_text(encoding="utf-8"))

items = [{"id": r.item_id, "uid": r.StudyInstanceUID, "lang": r.lang,
          "text": "" if pd.isna(r.Report) else str(r.Report)}
         for r in df.itertuples()]

payload = {"items": items, "labels": LABELS, "keys": KEYS,
           "findings": gloss["findings"], "cues": gloss["cues"]}
blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSNA Knee - Report Labeler</title>
<style>
:root{--bg:#fbfbfa;--fg:#1c1b1a;--mut:#6b6a67;--line:#e3e1dd;--card:#fff;--acc:#2f6f4e}
@media(prefers-color-scheme:dark){:root{--bg:#191817;--fg:#eceae6;--mut:#9b9894;--line:#332f2c;--card:#211f1d;--acc:#7fc9a0}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{border-bottom:1px solid var(--line);padding:8px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:var(--card)}
h1{font-size:14px;margin:0;font-weight:650;letter-spacing:.01em}
.bar{flex:1;min-width:160px;height:7px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
button{font:inherit;font-size:13px;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--acc)}
.mut{color:var(--mut);font-size:12.5px}
main{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:0;overflow:hidden}
@media(max-width:900px){main{grid-template-columns:1fr;overflow:auto}}
#rep{padding:16px 20px;overflow-y:auto;white-space:pre-wrap;word-wrap:break-word;font-size:15px;line-height:1.7}
#side{border-left:1px solid var(--line);padding:12px;overflow-y:auto;background:var(--card)}
.row{display:flex;align-items:center;gap:8px;padding:3px 5px;border-radius:6px;cursor:pointer;user-select:none}
.row:hover{background:var(--bg)}
.k{font:12px ui-monospace,monospace;color:var(--mut);width:16px;text-align:center;flex:none}
.nm{flex:1;font-size:13.5px}
.hit{font:11px ui-monospace,monospace;color:var(--acc);flex:none}
.st{width:26px;height:24px;line-height:22px;text-align:center;border:1px solid var(--line);border-radius:5px;font-weight:700;font-size:14px;flex:none}
.s0{color:var(--mut)}
.s1{background:#e8590c22;border-color:#e8590c;color:#e8590c}
.s2{background:#7048e822;border-color:#7048e8;color:#8b6cf0}
.s3{background:#2f9e4422;border-color:#2f9e44;color:#2f9e44}
mark{background:#ffe06655;border-bottom:2px solid #f0a90080;padding:0 1px;border-radius:2px}
mark.neg{background:#2f9e4422;border-bottom-color:#2f9e4480}
mark.unc{background:#7048e822;border-bottom-color:#7048e880}
textarea{width:100%;font:inherit;font-size:13px;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);resize:vertical}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:1px 4px;color:var(--mut)}
footer{border-top:1px solid var(--line);padding:6px 14px;background:var(--card)}
.flag{color:#e8590c;font-weight:700}
</style></head><body>
<header>
  <h1>Report Labeler</h1>
  <div class="bar"><i id="pi"></i></div>
  <span class="mut" id="prog"></span>
  <button id="bprev">&larr;</button><button id="bnext">&rarr;</button>
  <button id="bnu">next unlabeled</button>
  <button id="bhl">highlight: on</button>
  <button id="bexp">export CSV</button>
  <label class="mut" style="cursor:pointer">import<input id="bimp" type="file" accept=".csv" hidden></label>
</header>
<main>
  <div id="rep"></div>
  <div id="side">
    <div class="mut" id="meta" style="margin-bottom:8px"></div>
    <div id="rows"></div>
    <div style="margin-top:10px">
      <div class="mut" style="margin-bottom:4px">notes <kbd>t</kbd></div>
      <textarea id="note" rows="3" placeholder="ambiguities, disagreements, template quirks..."></textarea>
    </div>
    <div style="margin-top:8px"><button id="bflag">flag for review <kbd>f</kbd></button></div>
  </div>
</main>
<footer class="mut">
  <kbd>1</kbd>..<kbd>=</kbd> cycle label &nbsp; <kbd>&rarr;</kbd>/<kbd>&larr;</kbd> browse &nbsp;
  <kbd>Enter</kbd> mark done + next unlabeled &nbsp; <kbd>f</kbd> flag &nbsp; <kbd>t</kbd> notes &nbsp;
  <kbd>h</kbd> highlight &nbsp; states: <b style="color:#e8590c">+</b> present
  <b style="color:#8b6cf0">?</b> uncertain <b style="color:#2f9e44">&minus;</b> explicitly negated
  <b>&mdash;</b> not mentioned
</footer>
<script>
const DATA = __PAYLOAD__;
const {items, labels, keys, findings, cues} = DATA;
const SKEY = "rsna_knee_labeling_v1";
const SYM = ["\\u2014","+","?","\\u2212"];
let st = JSON.parse(localStorage.getItem(SKEY) || "{}");
let i = 0, hl = true, t0 = Date.now();

const rec = id => st[id] || (st[id] = {l:Array(12).fill(0), flag:0, note:"", done:0, secs:0});
const save = () => localStorage.setItem(SKEY, JSON.stringify(st));
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

// ---- glossary matching -------------------------------------------------------
function terms(lang){
  const out = [];
  for(const [fk, byLang] of Object.entries(findings))
    for(const t of (byLang[lang]||[])) out.push([t, "f", fk]);
  for(const t of (cues.negation[lang]||[]))    out.push([t, "neg", "neg"]);
  for(const t of (cues.uncertainty[lang]||[])) out.push([t, "unc", "unc"]);
  return out;
}
// Index-safe normalisation: output length ALWAYS equals input length, so match offsets
// map straight back onto the original string for highlighting.
//   * NFKD folds the MICRO SIGN (U+00B5) onto GREEK MU (U+03BC) -- these reports come from
//     Greek Windows and use the micro sign throughout ('µηνίσκου'), which silently broke
//     every Greek match.
//   * strips combining accents, so uppercase Greek (accent-less) matches accented terms
//   * final sigma to medial sigma
function normChar(ch){
  let c = ch.toLowerCase().normalize("NFKD").replace(/[\\u0300-\\u036f]/g, "");
  if(c.length === 0) return ch;
  c = c[0];
  return c === "\\u03c2" ? "\\u03c3" : c;
}
function norm(s){ let o = ""; for(let k = 0; k < s.length; k++) o += normChar(s[k]); return o; }

function spans(text, lang){
  const low = norm(text), hits = {}, iv = [];
  for(const [t, cls, fk] of terms(lang)){
    const q = norm(t); let p = low.indexOf(q);
    while(p !== -1){ iv.push([p, p+q.length, cls]); hits[fk]=(hits[fk]||0)+1; p = low.indexOf(q, p+1); }
  }
  iv.sort((a,b) => a[0]-b[0] || (b[1]-b[0])-(a[1]-a[0]));
  const keep = []; let end = -1;
  for(const s of iv) if(s[0] >= end){ keep.push(s); end = s[1]; }
  return {keep, hits};
}
function render(text, lang){
  if(!hl) return esc(text);
  const {keep} = spans(text, lang); let out = "", p = 0;
  for(const [a,b,c] of keep){
    out += esc(text.slice(p,a)) + `<mark class="${c==="f"?"":c}">` + esc(text.slice(a,b)) + "</mark>";
    p = b;
  }
  return out + esc(text.slice(p));
}

// ---- UI ----------------------------------------------------------------------
function draw(){
  const it = items[i], r = rec(it.id);
  document.getElementById("rep").innerHTML = render(it.text, it.lang);
  document.getElementById("rep").scrollTop = 0;
  const {hits} = spans(it.text, it.lang);
  document.getElementById("meta").innerHTML =
    `<b>${i+1} / ${items.length}</b> &nbsp; ${it.lang} &nbsp; ${it.text.length} chars` +
    (r.flag ? ' <span class="flag">FLAGGED</span>' : "") + (r.done ? " &nbsp;done" : "");
  document.getElementById("rows").innerHTML = labels.map((L,j) => {
    const h = hits[L] || 0;
    const extra = L.endsWith("OA") ? (hits["_OA_generic"]||0) : (L.includes("Meniscus")||L==="ACL"||L==="MCL") ? (hits["_tear"]||0) : 0;
    return `<div class="row" data-j="${j}"><span class="k">${keys[j]}</span>`
      + `<span class="nm">${L}</span>`
      + `<span class="hit">${h?h:""}${extra?"+"+extra:""}</span>`
      + `<span class="st s${r.l[j]}">${SYM[r.l[j]]}</span></div>`;
  }).join("");
  document.querySelectorAll(".row").forEach(el =>
    el.onclick = () => cycle(+el.dataset.j));
  document.getElementById("note").value = r.note;
  const done = Object.values(st).filter(x => x.done).length;
  document.getElementById("prog").textContent = `${done} done`;
  document.getElementById("pi").style.width = (100*done/items.length) + "%";
  t0 = Date.now();
}
function cycle(j){ const r = rec(items[i].id); r.l[j] = (r.l[j]+1) % 4; save(); draw(); }
function go(d){
  const r = rec(items[i].id); r.secs += Math.round((Date.now()-t0)/1000); save();
  i = Math.max(0, Math.min(items.length-1, i+d)); draw();
}
function nextUnlabeled(){
  for(let k=1; k<=items.length; k++){
    const j = (i+k) % items.length;
    if(!rec(items[j].id).done){ const r=rec(items[i].id); r.secs+=Math.round((Date.now()-t0)/1000); i=j; save(); draw(); return; }
  }
  alert("All items marked done.");
}
document.getElementById("bprev").onclick = () => go(-1);
document.getElementById("bnext").onclick = () => go(1);
document.getElementById("bnu").onclick = nextUnlabeled;
document.getElementById("bflag").onclick = () => { const r=rec(items[i].id); r.flag = r.flag?0:1; save(); draw(); };
document.getElementById("bhl").onclick = e => { hl = !hl; e.target.textContent = "highlight: " + (hl?"on":"off"); draw(); };
document.getElementById("note").oninput = e => { rec(items[i].id).note = e.target.value; save(); };

document.addEventListener("keydown", e => {
  if(e.target.tagName === "TEXTAREA"){ if(e.key === "Escape") e.target.blur(); return; }
  if(e.ctrlKey || e.altKey || e.metaKey) return;
  const k = keys.indexOf(e.key);
  if(k !== -1){ e.preventDefault(); cycle(k); return; }
  if(e.key === "ArrowRight"){ e.preventDefault(); go(1); }
  else if(e.key === "ArrowLeft"){ e.preventDefault(); go(-1); }
  else if(e.key === "Enter"){ e.preventDefault(); rec(items[i].id).done = 1; save(); nextUnlabeled(); }
  else if(e.key === "f"){ e.preventDefault(); const r=rec(items[i].id); r.flag=r.flag?0:1; save(); draw(); }
  else if(e.key === "t"){ e.preventDefault(); document.getElementById("note").focus(); }
  else if(e.key === "h"){ e.preventDefault(); document.getElementById("bhl").click(); }
});

// ---- CSV in/out --------------------------------------------------------------
const q = s => '"' + String(s).replace(/"/g,'""') + '"';
document.getElementById("bexp").onclick = () => {
  const head = ["item_id","StudyInstanceUID","lang"].concat(labels.map(l=>'"'+l+'"'))
               .concat(["flag","note","done","secs"]).join(",");
  const rows = items.map(it => { const r = rec(it.id);
    return [it.id, it.uid, it.lang].concat(r.l).concat([r.flag, q(r.note), r.done, r.secs]).join(","); });
  const blob = new Blob([head+"\\n"+rows.join("\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "hand_labels_" + new Date().toISOString().slice(0,10) + ".csv";
  a.click();
};
document.getElementById("bimp").onchange = ev => {
  const f = ev.target.files[0]; if(!f) return;
  f.text().then(txt => {
    const lines = txt.split(/\\r?\\n/).filter(Boolean); let n = 0;
    for(const ln of lines.slice(1)){
      const c = ln.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(s => s.replace(/,$/,"").replace(/^"|"$/g,"").replace(/""/g,'"'));
      if(!c[0] || !c[0].startsWith("it")) continue;
      st[c[0]] = {l: c.slice(3,15).map(Number), flag:+c[15]||0, note:c[16]||"", done:+c[17]||0, secs:+c[18]||0};
      n++;
    }
    save(); draw(); alert("imported " + n + " rows");
  });
};
draw();
</script></body></html>
"""

out = ROOT / "labeler.html"
out.write_text(HTML.replace("__PAYLOAD__", blob), encoding="utf-8")
kb = out.stat().st_size / 1024
print(f"wrote {out}  ({kb:.0f} KB, {len(items)} items)")
print(f"open it with:  start \"\" \"{out}\"")
