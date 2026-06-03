"""Ariadne Discovery Console -- a local mission-control dashboard over the engines.

Run:  python scripts/dashboard.py   then open http://127.0.0.1:8765

A single clean page:
  * Live sky map of the survey grid, swept tiles, and discovery candidates
  * Survey progress (tiles swept, runs, recovery, scheduled-task next run)
  * Classify a mover or a variable star through the coherence engines (instant)
  * Browse the candidate ledger; above-floor candidates flagged
  * Kick off a field run or an NSC catalog sweep in the background

Self-contained: Flask + a vanilla-JS canvas (no CDN, works offline). Everything
talks to the same coherence engines the automated survey uses.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data"

from ariadne.discovery.imaging.coherence_classifier import (  # noqa: E402
    classify_mover, classify_variable)
try:
    from run_catalog_survey import ecliptic_tiles
    TILES = ecliptic_tiles(step_deg=2.0)            # the survey grid for the map
except Exception:
    TILES = []

app = Flask(__name__)


def _ledger():
    p = DATA / "discovery_ledger.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def _state(name):
    p = DATA / name
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _next_scheduled():
    try:
        out = subprocess.run(["schtasks", "/Query", "/TN", "AriadneAutoDiscovery", "/FO", "LIST", "/V"],
                             capture_output=True, text=True, timeout=8).stdout
        m = re.search(r"Next Run Time:\s*(.+)", out)
        return m.group(1).strip() if m else None
    except Exception:
        return None


PAGE = r"""
<!doctype html><html><head><meta charset="utf-8"><title>Ariadne Discovery Console</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#070b12;--panel:#121a28;--pa2:#0e1521;--ink:#e8eef7;--mut:#8298b8;--acc:#6ea8fe;--ok:#3fb950;--af:#f0b429;--line:#1c2738}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#10203a 0,var(--bg) 60%);color:var(--ink);font:14.5px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:14px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;background:#070b12ee;backdrop-filter:blur(6px);z-index:5}
h1{font-size:18px;margin:0;letter-spacing:.4px} h1 b{color:var(--acc)}
.spacer{flex:1}
.stat{display:flex;flex-direction:column;align-items:flex-end;line-height:1.15} .stat b{font-size:17px} .stat span{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.6px}
.pill{font-size:11.5px;padding:4px 11px;border-radius:999px;background:#16202f;color:var(--mut);border:1px solid var(--line)}
.pill.ok{color:var(--ok);border-color:#1d3a23} .pill.warn{color:var(--af);border-color:#3a3216}
main{display:grid;grid-template-columns:1.45fr 1fr;gap:16px;padding:18px;max-width:1320px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h2{margin:0 0 4px;font-size:13px;color:var(--acc);text-transform:uppercase;letter-spacing:.7px}
.card .hint{color:var(--mut);font-size:12px;margin-bottom:10px}
.full{grid-column:1/3}
canvas{width:100%;height:auto;display:block;border-radius:10px;background:#070d17;border:1px solid var(--line)}
.legend{display:flex;gap:16px;margin-top:8px;font-size:12px;color:var(--mut);flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
label{display:block;font-size:11px;color:var(--mut);margin:9px 0 3px;text-transform:uppercase;letter-spacing:.5px}
input,select{width:100%;padding:9px 11px;background:var(--pa2);border:1px solid #243149;border-radius:9px;color:var(--ink);font-size:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{margin-top:12px;padding:10px 15px;background:var(--acc);color:#06101f;border:0;border-radius:9px;font-weight:650;cursor:pointer;font-size:13.5px}
button.ghost{background:#1b2536;color:var(--ink);border:1px solid var(--line)}
.res{margin-top:13px;font-size:13.5px}
.cls{display:flex;justify-content:space-between;font-size:13px;margin:7px 0 2px} .cls b{color:var(--ink)}
.bar{height:8px;background:var(--pa2);border-radius:6px;overflow:hidden} .bar>i{display:block;height:100%;background:linear-gradient(90deg,#6ea8fe,#3fb950)}
.top{outline:1px solid #2b3f63;border-radius:8px;padding:6px 8px;background:#0f1a2c}
table{width:100%;border-collapse:collapse;font-size:12.5px} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #18222f}
th{color:var(--mut);font-weight:500;text-transform:uppercase;font-size:10.5px;letter-spacing:.5px} tr.afrow td{background:#1c170a}
td.af{color:var(--af);font-weight:700}
.prog{height:10px;background:var(--pa2);border-radius:6px;overflow:hidden;margin:6px 0} .prog>i{display:block;height:100%;background:linear-gradient(90deg,#6ea8fe,#3fb950)}
.runs{font-size:12px;color:var(--mut);max-height:150px;overflow:auto} .runs div{padding:3px 0;border-bottom:1px solid #16202f}
.muted{color:var(--mut);font-size:12.5px}
</style></head><body>
<header>
  <h1>ARIADNE <b>·</b> Discovery Console</h1>
  <span id="p_dl" class="pill">Data Lab …</span>
  <span id="p_task" class="pill">scheduler …</span>
  <div class="spacer"></div>
  <div class="stat"><b id="s_led">-</b><span>candidates</span></div>
  <div class="stat"><b id="s_af" style="color:var(--af)">-</b><span>above floor</span></div>
  <div class="stat"><b id="s_rec">-</b><span>knowns recovered</span></div>
  <div class="stat"><b id="s_tiles">-</b><span>tiles swept</span></div>
</header>
<main>
  <div class="card">
    <h2>Sky map · survey grid &amp; candidates</h2>
    <div class="hint">Ecliptic survey grid, tiles already swept, and every discovery candidate. Above-floor candidates are flagged.</div>
    <canvas id="sky" width="900" height="360"></canvas>
    <div class="legend">
      <span><i class="dot" style="background:#26344a"></i>survey grid</span>
      <span><i class="dot" style="background:#6ea8fe"></i>swept tile</span>
      <span><i class="dot" style="background:#3fb950"></i>candidate</span>
      <span><i class="dot" style="background:#f0b429"></i>above floor</span>
      <span style="color:#3a4a63">- ecliptic</span>
    </div>
  </div>
  <div class="card">
    <h2>Survey progress</h2>
    <div class="hint">Catalog (NSC) tile sweep - no image downloads.</div>
    <div id="prog_txt" class="muted">…</div>
    <div class="prog"><i id="prog_bar" style="width:0%"></i></div>
    <div style="margin-top:12px"><label>recent runs</label><div id="runs" class="runs"></div></div>
  </div>

  <div class="card">
    <h2>Classify a mover</h2>
    <div class="hint">Heliocentric distance (from the observed rate); eccentricity if an orbit is known.</div>
    <div class="row"><div><label>distance (AU)</label><input id="m_r" type="number" step="0.1" value="2.7"></div>
      <div><label>eccentricity (opt)</label><input id="m_e" type="number" step="0.05" placeholder="-"></div></div>
    <button onclick="cmover()">Classify mover</button>
    <div id="m_res" class="res"></div>
  </div>
  <div class="card">
    <h2>Classify a variable star</h2>
    <div class="hint">Light-curve features + color (g-r breaks the look-alike classes).</div>
    <div class="row"><div><label>period (d)</label><input id="v_p" type="number" step="0.01" value="0.55"></div>
      <div><label>amplitude (mag)</label><input id="v_a" type="number" step="0.05" value="0.7"></div></div>
    <div class="row"><div><label>R21</label><input id="v_r" type="number" step="0.05" value="0.45"></div>
      <div><label>color g-r</label><input id="v_g" type="number" step="0.05" value="0.3"></div></div>
    <button onclick="cvar()">Classify variable</button>
    <div id="v_res" class="res"></div>
  </div>

  <div class="card full">
    <h2>Run the pipeline</h2>
    <div class="row" style="grid-template-columns:2fr 1fr 2fr 1fr;align-items:end;gap:12px">
      <div><label>local field (image-based)</label><select id="field"></select></div>
      <div><button onclick="runField()">Run field</button></div>
      <div><label>NSC catalog tiles (no downloads)</label><input id="tiles" type="number" value="10"></div>
      <div><button class="ghost" onclick="runCatalog()">Sweep NSC</button></div>
    </div>
    <div id="run_res" class="res muted"></div>
  </div>

  <div class="card full">
    <h2>Discovery ledger <span id="led_n" class="muted"></span></h2>
    <div id="ledger"></div>
  </div>
</main>
<script>
const SKY=document.getElementById('sky'),CX=SKY.getContext('2d');
async function j(u,o){const r=await fetch(u,o);return r.json()}
function clsbars(post){const e=Object.entries(post).sort((a,b)=>b[1]-a[1]).slice(0,5);
  return e.map(([k,v],i)=>`<div class="${i==0?'top':''}"><div class="cls"><span>${k}</span><b>${(v*100).toFixed(0)}%</b></div><div class="bar"><i style="width:${(v*100).toFixed(0)}%"></i></div></div>`).join('')}
async function cmover(){const r=document.getElementById('m_r').value,e=document.getElementById('m_e').value;
  document.getElementById('m_res').innerHTML=clsbars((await j(`/api/classify/mover?r=${r}&e=${e}`)).post)}
async function cvar(){const p=document.getElementById('v_p').value,a=document.getElementById('v_a').value,r=document.getElementById('v_r').value,g=document.getElementById('v_g').value;
  document.getElementById('v_res').innerHTML=clsbars((await j(`/api/classify/variable?p=${p}&a=${a}&r21=${r}&gr=${g}`)).post)}
async function runField(){const f=document.getElementById('field').value;document.getElementById('run_res').textContent='starting field run on '+f+' …';
  document.getElementById('run_res').textContent=(await j('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'field',field:f})})).msg}
async function runCatalog(){const t=document.getElementById('tiles').value;document.getElementById('run_res').textContent='starting NSC sweep …';
  document.getElementById('run_res').textContent=(await j('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'catalog',tiles:t})})).msg}
function X(ra){return 40+(ra/360)*(SKY.width-55)} function Y(dec){return SKY.height/2-(dec/32)*(SKY.height/2-22)}
function drawSky(d){const W=SKY.width,H=SKY.height;CX.clearRect(0,0,W,H);
  CX.strokeStyle='#16202f';CX.lineWidth=1;
  for(let ra=0;ra<=360;ra+=60){CX.beginPath();CX.moveTo(X(ra),12);CX.lineTo(X(ra),H-16);CX.stroke();
    CX.fillStyle='#3a4a63';CX.font='10px system-ui';CX.fillText(ra+'°',X(ra)-7,H-4)}
  for(let dec=-30;dec<=30;dec+=15){CX.beginPath();CX.moveTo(40,Y(dec));CX.lineTo(W-15,Y(dec));CX.stroke();
    CX.fillStyle='#3a4a63';CX.fillText((dec>0?'+':'')+dec,4,Y(dec)+3)}
  CX.strokeStyle='#2a3a57';CX.beginPath();for(let ra=0;ra<=360;ra+=2){const dec=23.44*Math.sin(ra*Math.PI/180);const x=X(ra),y=Y(dec);ra==0?CX.moveTo(x,y):CX.lineTo(x,y)}CX.stroke();
  (d.grid||[]).forEach(t=>{CX.fillStyle='#26344a';CX.fillRect(X(t[0])-1,Y(t[1])-1,2,2)});
  (d.tiles||[]).forEach(t=>{CX.fillStyle='#6ea8fe';CX.beginPath();CX.arc(X(t[0]),Y(t[1]),2.4,0,7);CX.fill()});
  (d.candidates||[]).forEach(c=>{const x=X(c.ra),y=Y(c.dec);
    if(c.above){CX.fillStyle='#f0b429';CX.shadowColor='#f0b429';CX.shadowBlur=8;CX.beginPath();CX.arc(x,y,5,0,7);CX.fill();CX.shadowBlur=0}
    else{CX.fillStyle='#3fb950';CX.beginPath();CX.arc(x,y,3.4,0,7);CX.fill()}})}
async function refresh(){
  const s=await j('/api/status');
  document.getElementById('p_dl').textContent='Data Lab: '+(s.datalab?'authenticated':'not set');document.getElementById('p_dl').className='pill '+(s.datalab?'ok':'warn');
  document.getElementById('p_task').textContent='next run: '+(s.next_run||'not scheduled');document.getElementById('p_task').className='pill '+(s.next_run?'ok':'warn');
  document.getElementById('s_led').textContent=s.ledger;document.getElementById('s_af').textContent=s.above_floor;
  document.getElementById('s_rec').textContent=s.recovered+'/'+s.recoverable;document.getElementById('s_tiles').textContent=s.tiles_done+'/'+s.tiles_total;
  document.getElementById('prog_txt').textContent=`${s.tiles_done} of ${s.tiles_total} ecliptic tiles swept · ${s.runs} runs logged`;
  document.getElementById('prog_bar').style.width=(100*s.tiles_done/Math.max(s.tiles_total,1)).toFixed(1)+'%';
  document.getElementById('field').innerHTML=s.fields.map(f=>`<option>${f}</option>`).join('')||'<option>(no local fields)</option>';
  document.getElementById('runs').innerHTML=(s.recent||[]).map(r=>`<div>${(r.utc||'').slice(0,16).replace('T',' ')} · ${r.field} · rec ${r.recovered}/${r.recoverable} · cand ${r.candidates}${r.above_floor?' · <span style="color:#f0b429">ABOVE</span>':''}</div>`).join('')||'<div>no runs yet</div>';
  drawSky(await j('/api/skymap'));
  const led=await j('/api/ledger');document.getElementById('led_n').textContent='· '+led.rows.length;
  document.getElementById('ledger').innerHTML=led.rows.length?(
    '<table><tr><th>field</th><th>source</th><th>RA</th><th>Dec</th><th>rate "/hr</th><th>nights</th><th>coh</th><th>floor</th></tr>'+
    led.rows.slice().reverse().map(r=>`<tr class="${r.above_floor?'afrow':''}"><td>${r.field_id||''}</td><td>${r.source||'image'}</td><td>${(+r.ra_deg).toFixed(4)}</td><td>${(+r.dec_deg).toFixed(4)}</td><td>${(+r.rate_arcsec_hr).toFixed(0)}</td><td>${(r.nights||[]).length}</td><td>${(+(r.coherence||0)).toFixed(3)}</td><td class="${r.above_floor?'af':''}">${r.above_floor?'ABOVE':'-'}</td></tr>`).join('')+'</table>'
  ):'<div class="muted">No candidates yet. Run a local field or sweep NSC tiles above.</div>';
}
refresh();setInterval(refresh,8000);
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/classify/mover")
def api_mover():
    r = float(request.args.get("r", 2.7)); e = request.args.get("e", "")
    ecc = float(e) if e not in ("", "None", None) else None
    return jsonify(post=classify_mover(r, ecc))


@app.route("/api/classify/variable")
def api_var():
    p = float(request.args.get("p", 0.55)); a = float(request.args.get("a", 0.7))
    r21 = float(request.args.get("r21", 0.45)); gr = request.args.get("gr", "")
    g = float(gr) if gr not in ("", "None", None) else None
    return jsonify(post=classify_variable(p, r21, a, g_r=g))


@app.route("/api/ledger")
def api_ledger():
    return jsonify(rows=_ledger()[-300:])


@app.route("/api/skymap")
def api_skymap():
    led = _ledger()
    cat = _state("catalog_survey_state.json")
    return jsonify(
        grid=[[ra, dec] for (ra, dec) in TILES],
        tiles=[[t[0], t[1]] for t in cat.get("done", [])],
        candidates=[{"ra": c["ra_deg"], "dec": c["dec_deg"], "above": bool(c.get("above_floor"))}
                    for c in led])


@app.route("/api/status")
def api_status():
    led = _ledger()
    cat = _state("catalog_survey_state.json"); auto = _state("auto_discovery_state.json")
    runs = auto.get("runs", [])
    fields = sorted([d.name for d in DATA.iterdir() if d.is_dir() and any(d.glob("*.fits.fz"))]) if DATA.exists() else []
    datalab = bool(os.environ.get("COH_DATALAB_TOKEN") or (os.environ.get("DATALAB_USER") and os.environ.get("DATALAB_PASS")))
    rec = sum(r.get("recovered", 0) for r in runs if isinstance(r.get("recovered"), int))
    recov = sum(r.get("recoverable", 0) for r in runs if isinstance(r.get("recoverable"), int))
    return jsonify(ledger=len(led), above_floor=sum(1 for r in led if r.get("above_floor")),
                   fields=fields, datalab=datalab, tiles_done=len(cat.get("done", [])),
                   tiles_total=len(TILES) or 1, runs=len(runs), recovered=rec, recoverable=recov,
                   next_run=_next_scheduled(), recent=runs[-12:][::-1])


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    if body.get("mode") == "field":
        cmd = [sys.executable, str(ROOT / "scripts" / "run_auto_discovery.py"), "--data-dir", str(DATA / body["field"])]
        msg = f"field run started on {body['field']} - watch the map/ledger (auto-refresh 8s)"
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_catalog_survey.py"), "--tiles-per-run", str(int(body.get("tiles", 10)))]
        msg = f"NSC catalog sweep started ({body.get('tiles', 10)} tiles)"
    subprocess.Popen(cmd, cwd=str(ROOT))
    return jsonify(msg=msg)


if __name__ == "__main__":
    print("Ariadne Discovery Console -> http://127.0.0.1:8765")
    app.run(host="127.0.0.1", port=8765, debug=False)
