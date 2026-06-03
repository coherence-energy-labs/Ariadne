"""Ariadne Command Cockpit -- a local console over every engine.

Run:  python scripts/dashboard.py   (or double-click start_dashboard.bat)
It opens http://127.0.0.1:8765 in your browser automatically.

Three tabs:
  MISSION DESIGN -- CR3BP systems (instant constants), Gateway NRHO, Lyapunov &
                    Halo orbit families. The slow continuations (6-40s) run as
                    background jobs so the cockpit never freezes.
  DISCOVERY      -- classify a mover or variable star (instant), fit a TNO orbit
                    from MPC astrometry, sky map of survey candidates, run a field
                    or NSC catalog sweep, and the candidate ledger.
  JOBS           -- every background engine run with status, timing, and results.

Self-contained: Flask + vanilla JS canvas (no CDN). Talks to the same engines the
library and automated survey use.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data"
JOBDIR = DATA / "cockpit_jobs"
JOBDIR.mkdir(parents=True, exist_ok=True)

from ariadne.discovery.imaging.coherence_classifier import (  # noqa: E402
    classify_mover, classify_variable)
try:
    from run_catalog_survey import ecliptic_tiles
    TILES = ecliptic_tiles(step_deg=2.0)
except Exception:
    TILES = []

app = Flask(__name__)
SYSTEMS = ["EARTH_MOON", "SUN_EARTH", "SUN_MARS", "JUPITER_EUROPA", "JUPITER_GANYMEDE", "JUPITER_CALLISTO"]


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
<!doctype html><html><head><meta charset="utf-8"><title>Ariadne Command Cockpit</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#070b12;--panel:#121a28;--pa2:#0e1521;--ink:#e8eef7;--mut:#8298b8;--acc:#6ea8fe;--ok:#3fb950;--af:#f0b429;--line:#1c2738}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#10203a 0,var(--bg) 60%);color:var(--ink);font:14.5px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:13px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;background:#070b12ee;backdrop-filter:blur(6px);z-index:5}
h1{font-size:17px;margin:0;letter-spacing:.5px} h1 b{color:var(--acc)}
.tabs{display:flex;gap:6px;margin-left:8px}
.tab{padding:7px 15px;border-radius:9px;cursor:pointer;font-size:13px;color:var(--mut);border:1px solid transparent}
.tab.on{background:#16202f;color:var(--ink);border-color:var(--line)}
.spacer{flex:1}
.pill{font-size:11.5px;padding:4px 11px;border-radius:999px;background:#16202f;color:var(--mut);border:1px solid var(--line)}
.pill.ok{color:var(--ok);border-color:#1d3a23} .pill.warn{color:var(--af);border-color:#3a3216}
main{max-width:1320px;margin:0 auto;padding:18px}
.grid{display:grid;grid-template-columns:1.45fr 1fr;gap:16px} .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
.card h2{margin:0 0 4px;font-size:13px;color:var(--acc);text-transform:uppercase;letter-spacing:.7px}
.card .hint{color:var(--mut);font-size:12px;margin-bottom:10px}
.full{grid-column:1/3}
canvas{width:100%;height:auto;display:block;border-radius:10px;background:#070d17;border:1px solid var(--line)}
.legend{display:flex;gap:16px;margin-top:8px;font-size:12px;color:var(--mut);flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
label{display:block;font-size:11px;color:var(--mut);margin:9px 0 3px;text-transform:uppercase;letter-spacing:.5px}
input,select{width:100%;padding:9px 11px;background:var(--pa2);border:1px solid #243149;border-radius:9px;color:var(--ink);font-size:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px} .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
button{margin-top:12px;padding:10px 15px;background:var(--acc);color:#06101f;border:0;border-radius:9px;font-weight:650;cursor:pointer;font-size:13.5px}
button.ghost{background:#1b2536;color:var(--ink);border:1px solid var(--line)}
.res{margin-top:13px;font-size:13.5px}
.cls{display:flex;justify-content:space-between;font-size:13px;margin:7px 0 2px}
.bar{height:8px;background:var(--pa2);border-radius:6px;overflow:hidden} .bar>i{display:block;height:100%;background:linear-gradient(90deg,#6ea8fe,#3fb950)}
.top{outline:1px solid #2b3f63;border-radius:8px;padding:6px 8px;background:#0f1a2c}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:13px} .kv b{color:var(--mut);font-weight:500} .kv span{text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:12.5px} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #18222f}
th{color:var(--mut);font-weight:500;text-transform:uppercase;font-size:10.5px;letter-spacing:.5px} tr.afrow td{background:#1c170a} td.af{color:var(--af);font-weight:700}
.prog{height:10px;background:var(--pa2);border-radius:6px;overflow:hidden;margin:6px 0} .prog>i{display:block;height:100%;background:linear-gradient(90deg,#6ea8fe,#3fb950)}
.runs{font-size:12px;color:var(--mut);max-height:140px;overflow:auto} .runs div{padding:3px 0;border-bottom:1px solid #16202f}
.muted{color:var(--mut);font-size:12.5px}
.capgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}
.cap{background:var(--pa2);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.cap img{width:100%;height:160px;object-fit:cover;background:#0a1019;border-bottom:1px solid var(--line)}
.cap .ph{height:160px;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:12px;background:#0a1019;border-bottom:1px solid var(--line)}
.cap .body{padding:11px 13px;flex:1;display:flex;flex-direction:column}
.cap h3{margin:0 0 4px;font-size:13.5px} .cap p{margin:0 0 10px;color:var(--mut);font-size:12px;flex:1}
.cap button{margin:0;width:100%;padding:8px}
.stat{display:flex;flex-direction:column;align-items:flex-end;line-height:1.1} .stat b{font-size:16px} .stat span{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.badge{font-size:11px;padding:2px 8px;border-radius:6px} .b-run{background:#13314f;color:#6ea8fe} .b-done{background:#12321d;color:#3fb950} .b-err{background:#3a1c1c;color:#f08a8a} .b-q{background:#222c3d;color:var(--mut)}
section{display:none} section.on{display:block}
</style></head><body>
<header>
  <h1>ARIADNE <b>//</b> Command Cockpit</h1>
  <div class="tabs">
    <div class="tab on" data-t="md" onclick="tab('md')">Mission Design</div>
    <div class="tab" data-t="caps" onclick="tab('caps')">Capabilities</div>
    <div class="tab" data-t="disc" onclick="tab('disc')">Discovery</div>
    <div class="tab" data-t="jobs" onclick="tab('jobs')">Jobs <span id="jobn" class="muted"></span></div>
  </div>
  <div class="spacer"></div>
  <span id="p_dl" class="pill">Data Lab ...</span>
  <span id="p_task" class="pill">scheduler ...</span>
</header>
<main>

<section id="md" class="on">
  <div class="grid3">
    <div class="card">
      <h2>CR3BP system</h2>
      <div class="hint">Characteristic constants of a circular restricted 3-body system (instant).</div>
      <label>system</label><select id="sysName"></select>
      <button onclick="job('system',{name:sysv()})">Show constants</button>
      <div id="sys_res" class="res"></div>
    </div>
    <div class="card">
      <h2>Gateway NRHO</h2>
      <div class="hint">NASA Gateway-class 9:2 near-rectilinear halo orbit (pseudo-arclength continuation, ~40s).</div>
      <button onclick="job('nrho',{})">Construct NRHO</button>
      <div class="muted" style="margin-top:10px">Returns period (days), Jacobi constant, z-amplitude. Watch the Jobs tab.</div>
    </div>
    <div class="card">
      <h2>Orbit family</h2>
      <div class="hint">Lyapunov (planar) or Halo (3D) family at a libration point.</div>
      <div class="row"><div><label>type</label><select id="fam"><option value="lyapunov">Lyapunov</option><option value="halo">Halo</option></select></div>
        <div><label>point</label><select id="fpt"><option>L1</option><option>L2</option></select></div></div>
      <label>number of orbits</label><input id="fn" type="number" value="12">
      <button onclick="job(famv(),{point:fptv(),n:fnv()})">Build family</button>
    </div>
  </div>
  <div class="card">
    <h2>Recent mission-design results</h2>
    <div id="md_jobs"></div>
  </div>
  <div class="card muted">
    <h2>Advanced (CLI)</h2>
    Cislunar round-trip, solar-system navigation, and the transfer atlas take rich mission parameters,
    so run them from the command line: <code>ariadne --help</code> · <code>python -c "import ariadne; ariadne.architect_cislunar_round_trip(...)"</code>.
  </div>
</section>

<section id="disc">
  <div class="grid">
    <div class="card">
      <h2>Sky map // survey candidates</h2>
      <div class="hint">Ecliptic survey grid, swept tiles, and every discovery candidate. Above-floor flagged.</div>
      <canvas id="sky" width="900" height="340"></canvas>
      <div class="legend">
        <span><i class="dot" style="background:#26344a"></i>grid</span>
        <span><i class="dot" style="background:#6ea8fe"></i>swept</span>
        <span><i class="dot" style="background:#3fb950"></i>candidate</span>
        <span><i class="dot" style="background:#f0b429"></i>above floor</span>
      </div>
    </div>
    <div>
      <div class="card">
        <h2>Classify a mover</h2>
        <div class="row"><div><label>distance (AU)</label><input id="m_r" type="number" step="0.1" value="2.7"></div>
          <div><label>eccentricity (opt)</label><input id="m_e" type="number" step="0.05" placeholder="-"></div></div>
        <button onclick="cmover()">Classify</button><div id="m_res" class="res"></div>
      </div>
      <div class="card">
        <h2>Classify a variable star</h2>
        <div class="row"><div><label>period (d)</label><input id="v_p" type="number" step="0.01" value="0.55"></div>
          <div><label>amplitude</label><input id="v_a" type="number" step="0.05" value="0.7"></div></div>
        <div class="row"><div><label>R21</label><input id="v_r" type="number" step="0.05" value="0.45"></div>
          <div><label>color g-r</label><input id="v_g" type="number" step="0.05" value="0.3"></div></div>
        <button onclick="cvar()">Classify</button><div id="v_res" class="res"></div>
      </div>
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Fit a TNO orbit (MPC)</h2>
      <div class="hint">Pull real MPC astrometry and fit a heliocentric orbit (network, a few seconds).</div>
      <div class="row"><div><label>designation</label><input id="tno" value="90377"></div>
        <div><label>window (days)</label><input id="tnow" type="number" value="720"></div></div>
      <button onclick="job('discover_tno',{designation:document.getElementById('tno').value,window_days:+document.getElementById('tnow').value})">Fit orbit</button>
      <div class="muted" style="margin-top:8px">e.g. 90377 (Sedna), 136199 (Eris), 50000 (Quaoar). Result on the Jobs tab.</div>
    </div>
    <div class="card">
      <h2>Run discovery</h2>
      <div class="row" style="grid-template-columns:2fr 1fr;align-items:end"><div><label>local field</label><select id="field"></select></div><div><button onclick="runField()">Run field</button></div></div>
      <div class="row" style="grid-template-columns:2fr 1fr;align-items:end"><div><label>NSC tiles (no downloads)</label><input id="tiles" type="number" value="10"></div><div><button class="ghost" onclick="runCatalog()">Sweep NSC</button></div></div>
      <div id="run_res" class="res muted"></div>
    </div>
  </div>
  <div class="card">
    <h2>Survey progress</h2>
    <div id="prog_txt" class="muted">...</div><div class="prog"><i id="prog_bar" style="width:0%"></i></div>
    <div style="margin-top:10px"><label>recent runs</label><div id="runs" class="runs"></div></div>
  </div>
  <div class="card">
    <h2>Discovery ledger <span id="led_n" class="muted"></span></h2><div id="ledger"></div>
  </div>
</section>

<section id="caps">
  <div class="card"><h2>Engine capabilities</h2>
    <div class="hint">Every headline engine, with the figure it produces. Click Run to execute the engine live (background job; figures regenerate and the result lands on the Jobs tab).</div>
    <div id="capgrid" class="capgrid"></div>
  </div>
</section>

<section id="jobs">
  <div class="card"><h2>Engine jobs</h2><div class="hint">Every background engine run (mission design + TNO fit). Newest first.</div><div id="jobtbl"></div></div>
</section>

</main>
<script>
const SKY=document.getElementById('sky'),CX=SKY.getContext('2d');
function tab(t){document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('on',e.dataset.t==t));document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id==t));if(t=='caps')loadCaps();if(t=='jobs')refreshJobs()}
async function loadCaps(){const d=await j('/api/capabilities');document.getElementById('capgrid').innerHTML=d.caps.map(c=>`<div class="cap">${c.figure?`<img src="/api/figure/${c.figure}?t=${Date.now()}" onerror="this.outerHTML='<div class=ph>figure not generated yet</div>'">`:'<div class="ph">pipeline engine (no figure)</div>'}<div class="body"><h3>${c.n}. ${c.title}</h3><p>${c.desc}</p><button onclick="runCap(${c.n})">Run engine</button></div></div>`).join('')}
async function runCap(n){await j('/api/job',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({engine:'capability',params:{n}})});tab('jobs')}
async function j(u,o){const r=await fetch(u,o);return r.json()}
function sysv(){return document.getElementById('sysName').value} function famv(){return document.getElementById('fam').value}
function fptv(){return document.getElementById('fpt').value} function fnv(){return +document.getElementById('fn').value}
function clsbars(post){return Object.entries(post).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,v],i)=>`<div class="${i==0?'top':''}"><div class="cls"><span>${k}</span><b>${(v*100).toFixed(0)}%</b></div><div class="bar"><i style="width:${(v*100).toFixed(0)}%"></i></div></div>`).join('')}
async function cmover(){const r=document.getElementById('m_r').value,e=document.getElementById('m_e').value;document.getElementById('m_res').innerHTML=clsbars((await j(`/api/classify/mover?r=${r}&e=${e}`)).post)}
async function cvar(){const p=document.getElementById('v_p').value,a=document.getElementById('v_a').value,r=document.getElementById('v_r').value,g=document.getElementById('v_g').value;document.getElementById('v_res').innerHTML=clsbars((await j(`/api/classify/variable?p=${p}&a=${a}&r21=${r}&gr=${g}`)).post)}
async function job(engine,params){await j('/api/job',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({engine,params})});tab('jobs');refreshJobs()}
async function runField(){const f=document.getElementById('field').value;document.getElementById('run_res').textContent='starting field run on '+f+' ...';document.getElementById('run_res').textContent=(await j('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'field',field:f})})).msg}
async function runCatalog(){const t=document.getElementById('tiles').value;document.getElementById('run_res').textContent='starting NSC sweep ...';document.getElementById('run_res').textContent=(await j('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'catalog',tiles:t})})).msg}
function kv(o){return '<div class="kv">'+Object.entries(o).map(([k,v])=>`<b>${k}</b><span>${typeof v=='number'?v:String(v)}</span>`).join('')+'</div>'}
function badge(s){return {running:'b-run',done:'b-done',error:'b-err',queued:'b-q'}[s]||'b-q'}
async function refreshJobs(){const d=await j('/api/jobs');document.getElementById('jobn').textContent=d.jobs.length?('('+d.jobs.length+')'):'';
  const rows=d.jobs.map(jb=>{const el=jb.finished&&jb.started?((jb.finished-jb.started).toFixed(1)+'s'):(jb.started?'...':'');
    const body=jb.status=='done'?kv(jb.result||{}):(jb.status=='error'?('<span class="muted">'+(jb.error||'')+'</span>'):'<span class="muted">computing...</span>');
    return `<tr><td><span class="badge ${badge(jb.status)}">${jb.status}</span></td><td>${jb.engine}</td><td class="muted">${JSON.stringify(jb.params||{})}</td><td>${el}</td><td>${body}</td></tr>`}).join('');
  document.getElementById('jobtbl').innerHTML=d.jobs.length?('<table><tr><th></th><th>engine</th><th>params</th><th>t</th><th>result</th></tr>'+rows+'</table>'):'<div class="muted">No jobs yet. Launch one from Mission Design or fit a TNO.</div>';
  // mirror mission-design results compactly on the MD tab
  const md=d.jobs.filter(x=>['nrho','lyapunov','halo','system'].includes(x.engine)).slice(0,6);
  document.getElementById('md_jobs').innerHTML=md.length?md.map(jb=>`<div style="margin-bottom:10px"><span class="badge ${badge(jb.status)}">${jb.status}</span> <b>${jb.engine}</b> ${JSON.stringify(jb.params||{})}`+(jb.status=='done'?kv(jb.result||{}):'')+`</div>`).join(''):'<div class="muted">No mission-design runs yet.</div>';
}
function X(ra){return 40+(ra/360)*(SKY.width-55)} function Y(dec){return SKY.height/2-(dec/32)*(SKY.height/2-22)}
function drawSky(d){const W=SKY.width,H=SKY.height;CX.clearRect(0,0,W,H);CX.strokeStyle='#16202f';CX.lineWidth=1;
  for(let ra=0;ra<=360;ra+=60){CX.beginPath();CX.moveTo(X(ra),12);CX.lineTo(X(ra),H-16);CX.stroke();CX.fillStyle='#3a4a63';CX.font='10px system-ui';CX.fillText(ra+'°',X(ra)-7,H-4)}
  for(let dec=-30;dec<=30;dec+=15){CX.beginPath();CX.moveTo(40,Y(dec));CX.lineTo(W-15,Y(dec));CX.stroke();CX.fillStyle='#3a4a63';CX.fillText((dec>0?'+':'')+dec,4,Y(dec)+3)}
  CX.strokeStyle='#2a3a57';CX.beginPath();for(let ra=0;ra<=360;ra+=2){const dec=23.44*Math.sin(ra*Math.PI/180);ra==0?CX.moveTo(X(ra),Y(dec)):CX.lineTo(X(ra),Y(dec))}CX.stroke();
  (d.grid||[]).forEach(t=>{CX.fillStyle='#26344a';CX.fillRect(X(t[0])-1,Y(t[1])-1,2,2)});
  (d.tiles||[]).forEach(t=>{CX.fillStyle='#6ea8fe';CX.beginPath();CX.arc(X(t[0]),Y(t[1]),2.4,0,7);CX.fill()});
  (d.candidates||[]).forEach(c=>{const x=X(c.ra),y=Y(c.dec);if(c.above){CX.fillStyle='#f0b429';CX.shadowColor='#f0b429';CX.shadowBlur=8;CX.beginPath();CX.arc(x,y,5,0,7);CX.fill();CX.shadowBlur=0}else{CX.fillStyle='#3fb950';CX.beginPath();CX.arc(x,y,3.4,0,7);CX.fill()}})}
async function refresh(){const s=await j('/api/status');
  document.getElementById('p_dl').textContent='Data Lab: '+(s.datalab?'authenticated':'not set');document.getElementById('p_dl').className='pill '+(s.datalab?'ok':'warn');
  document.getElementById('p_task').textContent='next run: '+(s.next_run||'not scheduled');document.getElementById('p_task').className='pill '+(s.next_run?'ok':'warn');
  document.getElementById('sysName').innerHTML=s.systems.map(x=>`<option>${x}</option>`).join('');
  document.getElementById('field').innerHTML=s.fields.map(f=>`<option>${f}</option>`).join('')||'<option>(no local fields)</option>';
  document.getElementById('prog_txt').textContent=`${s.tiles_done} of ${s.tiles_total} ecliptic tiles swept - ${s.runs} runs`;
  document.getElementById('prog_bar').style.width=(100*s.tiles_done/Math.max(s.tiles_total,1)).toFixed(1)+'%';
  document.getElementById('runs').innerHTML=(s.recent||[]).map(r=>`<div>${(r.utc||'').slice(0,16).replace('T',' ')} - ${r.field} - rec ${r.recovered}/${r.recoverable} - cand ${r.candidates}${r.above_floor?' - <span style="color:#f0b429">ABOVE</span>':''}</div>`).join('')||'<div>no runs yet</div>';
  drawSky(await j('/api/skymap'));
  const led=await j('/api/ledger');document.getElementById('led_n').textContent='- '+led.rows.length;
  document.getElementById('ledger').innerHTML=led.rows.length?('<table><tr><th>field</th><th>src</th><th>RA</th><th>Dec</th><th>rate</th><th>nights</th><th>coh</th><th>floor</th></tr>'+led.rows.slice().reverse().map(r=>`<tr class="${r.above_floor?'afrow':''}"><td>${r.field_id||''}</td><td>${r.source||'image'}</td><td>${(+r.ra_deg).toFixed(4)}</td><td>${(+r.dec_deg).toFixed(4)}</td><td>${(+r.rate_arcsec_hr).toFixed(0)}</td><td>${(r.nights||[]).length}</td><td>${(+(r.coherence||0)).toFixed(3)}</td><td class="${r.above_floor?'af':''}">${r.above_floor?'ABOVE':'-'}</td></tr>`).join('')+'</table>'):'<div class="muted">No candidates yet.</div>';
  refreshJobs();
}
refresh();setInterval(refresh,6000);
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/classify/mover")
def api_mover():
    r = float(request.args.get("r", 2.7)); e = request.args.get("e", "")
    return jsonify(post=classify_mover(r, float(e) if e not in ("", "None", None) else None))


@app.route("/api/classify/variable")
def api_var():
    p = float(request.args.get("p", 0.55)); a = float(request.args.get("a", 0.7))
    r21 = float(request.args.get("r21", 0.45)); gr = request.args.get("gr", "")
    return jsonify(post=classify_variable(p, r21, a, g_r=float(gr) if gr not in ("", "None", None) else None))


@app.route("/api/ledger")
def api_ledger():
    return jsonify(rows=_ledger()[-300:])


@app.route("/api/skymap")
def api_skymap():
    led = _ledger(); cat = _state("catalog_survey_state.json")
    return jsonify(grid=[[ra, dec] for (ra, dec) in TILES],
                   tiles=[[t[0], t[1]] for t in cat.get("done", [])],
                   candidates=[{"ra": c["ra_deg"], "dec": c["dec_deg"], "above": bool(c.get("above_floor"))} for c in led])


@app.route("/api/status")
def api_status():
    led = _ledger(); cat = _state("catalog_survey_state.json"); auto = _state("auto_discovery_state.json")
    runs = auto.get("runs", [])
    fields = sorted([d.name for d in DATA.iterdir() if d.is_dir() and any(d.glob("*.fits.fz"))]) if DATA.exists() else []
    datalab = bool(os.environ.get("COH_DATALAB_TOKEN") or (os.environ.get("DATALAB_USER") and os.environ.get("DATALAB_PASS")))
    return jsonify(ledger=len(led), above_floor=sum(1 for r in led if r.get("above_floor")),
                   fields=fields, systems=SYSTEMS, datalab=datalab, tiles_done=len(cat.get("done", [])),
                   tiles_total=len(TILES) or 1, runs=len(runs), next_run=_next_scheduled(), recent=runs[-12:][::-1])


@app.route("/api/job", methods=["POST"])
def api_job():
    body = request.get_json(force=True)
    jid = uuid.uuid4().hex[:10]
    jf = JOBDIR / f"{jid}.json"
    jf.write_text(json.dumps({"id": jid, "engine": body["engine"], "params": body.get("params", {}),
                              "status": "queued", "queued": time.time()}))
    subprocess.Popen([sys.executable, str(ROOT / "scripts" / "cockpit_worker.py"), str(jf)], cwd=str(ROOT))
    return jsonify(id=jid)


@app.route("/api/capabilities")
def api_caps():
    from cockpit_worker import CAPS
    out = []
    for n, (title, desc, fig) in sorted(CAPS.items()):
        figok = fig if (fig and (ROOT / "examples_out" / fig).exists()) else None
        out.append({"n": n, "title": title, "desc": desc, "figure": figok})
    return jsonify(caps=out)


@app.route("/api/figure/<name>")
def api_figure(name):
    return send_from_directory(str(ROOT / "examples_out"), name)


@app.route("/api/jobs")
def api_jobs():
    jobs = []
    for f in JOBDIR.glob("*.json"):
        try:
            jobs.append(json.loads(f.read_text()))
        except Exception:
            pass
    jobs.sort(key=lambda x: x.get("queued", 0), reverse=True)
    return jsonify(jobs=jobs[:40])


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    if body.get("mode") == "field":
        cmd = [sys.executable, str(ROOT / "scripts" / "run_auto_discovery.py"), "--data-dir", str(DATA / body["field"])]
        msg = f"field run started on {body['field']} (watch the map/ledger)"
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_catalog_survey.py"), "--tiles-per-run", str(int(body.get("tiles", 10)))]
        msg = f"NSC catalog sweep started ({body.get('tiles', 10)} tiles)"
    subprocess.Popen(cmd, cwd=str(ROOT))
    return jsonify(msg=msg)


if __name__ == "__main__":
    import logging
    import threading
    import webbrowser
    URL = "http://127.0.0.1:8765"
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    try:
        from flask import cli
        cli.show_server_banner = lambda *a, **k: None
    except Exception:
        pass
    print("\n  ARIADNE Command Cockpit")
    print(f"  Opening {URL} in your browser ...")
    print("  (keep this window open while you use it; close it or press Ctrl+C to stop)\n")
    threading.Timer(1.3, lambda: webbrowser.open(URL)).start()
    app.run(host="127.0.0.1", port=8765, debug=False)
