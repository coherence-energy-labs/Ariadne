"""Live web dashboard -- read-only view of the candidate store + alert log.

A single-file Flask app that serves:

  /              candidate table + Aitoff sky map (matplotlib -> SVG inline)
  /candidate/<key> per-candidate detail page (RMS history, fitted orbit,
                                              next-night follow-up prediction)
  /alerts        tail of the alert JSONL audit log
  /api/store     raw JSON of the candidate store (machine-readable)
  /api/alerts    raw JSONL of the alert log
  /api/score     ranked scores for every candidate (api consumers)

Designed for `python -m ariadne.discovery.dashboard --store path --alerts path
--port 5050`. NOT a long-lived service; bind to 127.0.0.1 by default; use a
reverse proxy + auth for any public deployment.

This is the operator's window into the engine: which candidates are accumulating
arc, which are stale, when the next observation should happen, and what the
ranked follow-up queue looks like for tonight.
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from string import Template

from .operations.candidate_store import CandidateStore
from .scoring import rank_candidates


# ----- HTML templates (kept inline to avoid a templates/ dir for one app) -----

_HOME_HTML = Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>Ariadne dashboard</title>
<style>
  body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }
  h1, h2 { color: #58a6ff; font-weight: 600; }
  table { border-collapse: collapse; margin: 10px 0; width: 100%; font-size: 14px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #30363d; text-align: left; }
  th { background: #161b22; color: #79c0ff; font-weight: 600; }
  tr:hover { background: #161b22; }
  .grade-A { color: #56d364; font-weight: 700; }
  .grade-B { color: #d29922; }
  .grade-C { color: #f0883e; }
  .grade-D { color: #f85149; }
  .stale { opacity: 0.4; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .stats { display: flex; gap: 24px; margin: 12px 0 24px 0; }
  .stat-box { background: #161b22; padding: 12px 18px; border-radius: 6px;
              min-width: 90px; border: 1px solid #30363d; }
  .stat-num { font-size: 26px; font-weight: 700; color: #58a6ff; }
  .stat-lbl { font-size: 12px; color: #8b949e; text-transform: uppercase; }
  img.skymap { background: #161b22; border-radius: 6px; padding: 10px;
               max-width: 100%; }
  pre { background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; }
</style>
</head><body>
<h1>Ariadne discovery dashboard</h1>
<p>Store: <code>$store_path</code> &middot; last saved $last_saved UTC</p>

<div class="stats">
  <div class="stat-box"><div class="stat-num">$n_total</div><div class="stat-lbl">total</div></div>
  <div class="stat-box"><div class="stat-num">$n_active</div><div class="stat-lbl">active leads</div></div>
  <div class="stat-box"><div class="stat-num">$n_stale</div><div class="stat-lbl">stale</div></div>
  <div class="stat-box"><div class="stat-num">$n_grade_A</div><div class="stat-lbl">grade A</div></div>
  <div class="stat-box"><div class="stat-num">$n_alerts</div><div class="stat-lbl">alerts fired</div></div>
</div>

<h2>Sky map (RA / Dec)</h2>
<img class="skymap" src="/skymap.svg" alt="sky map of all candidates" />

<h2>Candidates (ranked by quality score)</h2>
<table><thead><tr>
  <th>Grade</th><th>Score</th><th>Key</th><th>RA</th><th>Dec</th>
  <th>rate &quot;/hr</th><th>RMS &quot;</th><th>runs</th><th>arc days</th>
  <th>status</th><th>SkyBoT</th>
</tr></thead><tbody>
$rows
</tbody></table>

<p style="margin-top:24px; font-size:12px; color:#8b949e;">
  <a href="/alerts">View alert log</a> &middot;
  <a href="/api/store">API: store</a> &middot;
  <a href="/api/alerts">API: alerts</a> &middot;
  <a href="/api/score">API: score</a> &middot;
  <a href="/api/ops">API: ops</a>
</p>
</body></html>""")

_DETAIL_HTML = Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>Ariadne: $key</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0;
         padding: 20px; background: #0d1117; color: #c9d1d9; }
  h1, h2 { color: #58a6ff; }
  a { color: #58a6ff; }
  pre { background: #161b22; padding: 12px; border-radius: 6px;
        overflow-x: auto; font-size: 13px; }
  .field { margin: 8px 0; }
  .label { color: #8b949e; display: inline-block; min-width: 200px; }
</style></head><body>
<p><a href="/">&larr; back</a></p>
<h1>$key</h1>
<div class="field"><span class="label">grade</span> <b>$grade</b> (score $score)</div>
<div class="field"><span class="label">position</span> $ra, $dec deg</div>
<div class="field"><span class="label">on-sky rate</span> $rate arcsec/hr</div>
<div class="field"><span class="label">first seen</span> MJD $first</div>
<div class="field"><span class="label">last seen</span> MJD $last ($n_runs runs, $arc d arc)</div>
<div class="field"><span class="label">status</span> $status</div>
<div class="field"><span class="label">SkyBoT names</span> $skybot</div>
<div class="field"><span class="label">stored orbit state</span> $orbit_state</div>

<h2>RMS history</h2>
<pre>$rms_history</pre>

<h2>Predicted next-night position (MJD + 1)</h2>
<pre>$prediction</pre>

<h2>Raw metadata</h2>
<pre>$meta_json</pre>
</body></html>""")


def _render_skymap_svg(candidates) -> str:
    """Render a rectangular RA/Dec sky map as inline SVG (no matplotlib).

    Avoids matplotlib's `Path.__deepcopy__` infinite recursion on Python 3.14.
    Output: SVG document drawing each candidate as a coloured circle on a 360x180
    RA/Dec grid with light tick labels and a legend.
    """
    W, H = 1080, 540                            # pixel dimensions
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 30, 50
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    def x(ra_deg):    # RA 0..360 -> screen x (left of plot is RA=0)
        return PAD_L + (ra_deg % 360.0) / 360.0 * plot_w

    def y(dec_deg):   # Dec -90..+90 -> screen y (y=0 at top)
        return PAD_T + (90.0 - dec_deg) / 180.0 * plot_h

    colors = {"new": "#56d364", "active": "#58a6ff",
              "stale": "#6e7681", "rejected": "#f85149"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#161b22"/>',
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{plot_w}" height="{plot_h}" fill="#0d1117" stroke="#30363d"/>',
    ]
    # gridlines: 6 vertical (60 deg), 6 horizontal (30 deg)
    for r in range(0, 361, 60):
        gx = x(r)
        parts.append(f'<line x1="{gx}" y1="{PAD_T}" x2="{gx}" y2="{H - PAD_B}" '
                     f'stroke="#30363d" stroke-width="0.5" opacity="0.5"/>')
        parts.append(f'<text x="{gx}" y="{H - PAD_B + 18}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">{r}°</text>')
    for d in range(-90, 91, 30):
        gy = y(d)
        parts.append(f'<line x1="{PAD_L}" y1="{gy}" x2="{W - PAD_R}" y2="{gy}" '
                     f'stroke="#30363d" stroke-width="0.5" opacity="0.5"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{gy + 4}" fill="#8b949e" '
                     f'font-size="11" text-anchor="end">{d:+d}°</text>')

    parts.append(f'<text x="{PAD_L + plot_w / 2}" y="{H - 8}" '
                 f'fill="#8b949e" font-size="12" text-anchor="middle">RA (deg)</text>')
    parts.append(f'<text x="14" y="{PAD_T + plot_h / 2}" fill="#8b949e" '
                 f'font-size="12" text-anchor="middle" '
                 f'transform="rotate(-90, 14, {PAD_T + plot_h / 2})">Dec (deg)</text>')

    # candidate points
    counts = {}
    for status in ("stale", "rejected", "active", "new"):
        for c in candidates:
            if c.status != status:
                continue
            cx, cy = x(c.ra), y(c.dec)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                         f'fill="{colors[status]}" opacity="0.85"/>')
            counts[status] = counts.get(status, 0) + 1

    # legend (top-right)
    lx = W - PAD_R - 130
    ly = PAD_T + 8
    parts.append(f'<rect x="{lx}" y="{ly}" width="125" height="{14 * len(counts) + 10}" '
                 f'fill="#0d1117" stroke="#30363d" opacity="0.9"/>')
    for i, (status, n) in enumerate(counts.items()):
        parts.append(f'<circle cx="{lx + 12}" cy="{ly + 14 + i * 14}" r="5" '
                     f'fill="{colors[status]}"/>')
        parts.append(f'<text x="{lx + 24}" y="{ly + 18 + i * 14}" fill="#c9d1d9" '
                     f'font-size="11">{status} ({n})</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def create_app(store_path: str | Path, alerts_path: str | Path | None = None):
    """Build the Flask app. Pass store_path = where the JSON candidate store lives."""
    try:
        from flask import Flask, Response, abort, jsonify
    except ImportError as e:
        raise RuntimeError(
            "Flask is required for the dashboard. `pip install flask`") from e

    app = Flask(__name__)
    store_path = Path(store_path)
    alerts_path = Path(alerts_path) if alerts_path else None

    def _store():
        return CandidateStore(store_path) if store_path.exists() else CandidateStore(store_path)

    @app.get("/")
    def home():
        store = _store()
        cands = list(store)
        ranked = rank_candidates(cands)
        n_grade_A = sum(1 for _, s in ranked if s.grade() == "A")

        rows = []
        for cand, score in ranked:
            css = f"grade-{score.grade()}" + (" stale" if cand.status == "stale" else "")
            rms = cand.rms_history[-1][1] if cand.rms_history else float("nan")
            arc = cand.last_seen_mjd - cand.first_seen_mjd
            rows.append(
                f'<tr class="{css}"><td>{score.grade()}</td>'
                f'<td>{score.total:.2f}</td>'
                f'<td><a href="/candidate/{cand.key}">{cand.key}</a></td>'
                f'<td>{cand.ra:.4f}</td><td>{cand.dec:+.4f}</td>'
                f'<td>{cand.rate_arcsec_hr:.2f}</td>'
                f'<td>{rms:.2f}</td>'
                f'<td>{cand.n_runs}</td>'
                f'<td>{arc:.1f}</td>'
                f'<td>{cand.status}</td>'
                f'<td>{", ".join(cand.skybot_names) or "&mdash;"}</td></tr>'
            )

        n_alerts = 0
        if alerts_path and alerts_path.exists():
            with open(alerts_path, encoding="utf-8") as f:
                n_alerts = sum(1 for _ in f)

        n_total = len(cands)
        n_active = sum(1 for c in cands if c.status in ("new", "active"))
        n_stale = sum(1 for c in cands if c.status == "stale")
        last_saved = "(never)"
        if store_path.exists():
            with open(store_path, encoding="utf-8") as f:
                raw = json.load(f)
            ts = raw.get("saved_at_unix", 0)
            if ts:
                last_saved = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        return _HOME_HTML.substitute(
            store_path=str(store_path),
            last_saved=last_saved,
            n_total=n_total, n_active=n_active, n_stale=n_stale,
            n_grade_A=n_grade_A, n_alerts=n_alerts,
            rows="\n".join(rows) or '<tr><td colspan="11" style="text-align:center; padding:30px; color:#8b949e">no candidates yet</td></tr>',
        )

    @app.get("/skymap.svg")
    def skymap():
        store = _store()
        cands = list(store)
        return Response(_render_skymap_svg(cands), mimetype="image/svg+xml")

    @app.get("/candidate/<key>")
    def detail(key):
        store = _store()
        c = store.get(key)
        if c is None:
            abort(404)
        from .scoring import score_candidate
        s = score_candidate(c)

        rms_lines = "\n".join(f"  MJD {m:.3f}: {r:.2f}\"" for m, r in c.rms_history) or "  (none)"

        from .followup import predict_with_uncertainty
        next_mjd = c.last_seen_mjd + 1.0
        eph = predict_with_uncertainty(c, next_mjd, n_samples=20)
        if eph is None:
            pred_text = "  (no fitted orbit -- cannot predict)"
        else:
            pred_text = (f"  MJD {eph.mjd:.3f}:\n"
                         f"  RA {eph.ra_deg:.4f}, Dec {eph.dec_deg:+.4f}\n"
                         f"  search radius (4-sigma): {4.0 * eph.sigma_arcsec:.1f} arcsec\n"
                         f"  range: {eph.earth_distance_au:.2f} AU\n"
                         f"  Sun distance: {eph.sun_distance_au:.2f} AU\n"
                         f"  on-sky rate: {eph.v_sky_arcsec_hr:.2f} arcsec/hr")

        orbit = "(none)" if c.orbit_state is None else (
            f"x = [{c.orbit_state[0]:.3e}, {c.orbit_state[1]:.3e}, {c.orbit_state[2]:.3e}] km<br>"
            f"v = [{c.orbit_state[3]:.4f}, {c.orbit_state[4]:.4f}, {c.orbit_state[5]:.4f}] km/s")

        return _DETAIL_HTML.substitute(
            key=c.key, grade=s.grade(), score=f"{s.total:.2f}",
            ra=f"{c.ra:.4f}", dec=f"{c.dec:+.4f}",
            rate=f"{c.rate_arcsec_hr:.2f}",
            first=f"{c.first_seen_mjd:.3f}",
            last=f"{c.last_seen_mjd:.3f}",
            n_runs=c.n_runs,
            arc=f"{max(0.0, c.last_seen_mjd - c.first_seen_mjd):.1f}",
            status=c.status, skybot=", ".join(c.skybot_names) or "&mdash;",
            orbit_state=orbit,
            rms_history=rms_lines,
            prediction=pred_text,
            meta_json=json.dumps(c.meta, indent=2),
        )

    @app.get("/alerts")
    def alerts():
        if not alerts_path or not alerts_path.exists():
            return "<p>No alert log configured.</p>"
        with open(alerts_path, encoding="utf-8") as f:
            lines = f.readlines()[-200:]                # tail 200
        body = "<pre>" + "".join(lines) + "</pre>"
        return ("<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Ariadne alerts</title><style>body{background:#0d1117;color:#c9d1d9;"
                "font-family:ui-monospace,monospace;padding:20px}"
                "pre{background:#161b22;padding:14px;border-radius:6px;overflow-x:auto}</style>"
                f"</head><body><p><a href='/' style='color:#58a6ff'>&larr; back</a></p>"
                f"<h2 style='color:#58a6ff'>Alert log (last 200 lines)</h2>{body}</body></html>")

    @app.get("/api/store")
    def api_store():
        if not store_path.exists():
            return jsonify({"candidates": []})
        with open(store_path, encoding="utf-8") as f:
            return jsonify(json.load(f))

    @app.get("/api/alerts")
    def api_alerts():
        if not alerts_path or not alerts_path.exists():
            return jsonify([])
        with open(alerts_path, encoding="utf-8") as f:
            return jsonify([json.loads(line) for line in f if line.strip()])

    @app.get("/api/score")
    def api_score():
        store = _store()
        ranked = rank_candidates(list(store))
        return jsonify([{
            "key": c.key, "grade": s.grade(), "total": s.total,
            "rms": s.rms, "arc": s.arc, "runs": s.runs,
            "skybot": s.skybot, "obs": s.obs,
        } for c, s in ranked])

    @app.get("/api/ops")
    def api_ops():
        store = _store()
        cands = list(store)
        ranked = rank_candidates(cands)
        return jsonify({
            "n_candidates": len(cands),
            "n_ranked": len(ranked),
            "store_path": str(store_path),
            "alerts_path": str(alerts_path) if alerts_path else None,
            "top": [{
                "key": c.key,
                "grade": s.grade(),
                "score": s.total,
                "status": c.status,
                "meta": c.meta,
            } for c, s in ranked[:20]],
        })

    return app


def main(argv=None):
    """`python -m ariadne.discovery.dashboard --store path --alerts path --port 5050`"""
    import argparse
    p = argparse.ArgumentParser(prog="ariadne-dashboard")
    p.add_argument("--store", required=True, help="path to candidate JSON store")
    p.add_argument("--alerts", default=None, help="path to alert JSONL log (optional)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default localhost)")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    app = create_app(args.store, args.alerts)
    print(f"Ariadne dashboard: http://{args.host}:{args.port}/")
    print(f"  store : {args.store}")
    print(f"  alerts: {args.alerts}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
