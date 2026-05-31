"""Ariadne command-line interface -- a few high-leverage subcommands.

Installed as the `ariadne` console script via pyproject [project.scripts].

Subcommands:
  ariadne info                       -- print version, system constants, capability summary
  ariadne systems                    -- list the 7 CR3BP systems
  ariadne lyapunov  --point L1 ...   -- build a Lyapunov family + print Jacobi range
  ariadne nrho                       -- construct the Gateway NRHO + report geometry
  ariadne discover  90377            -- fit a TNO orbit from MPC astrometry
  ariadne benchmark                  -- run the 16-check reference benchmark suite
  ariadne tutorial  N                -- run example N (1..5) and produce its PNG
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _cmd_info(args: argparse.Namespace) -> int:
    import ariadne
    print(f"Ariadne {ariadne.__version__}")
    print(f"  CR3BP systems available: {len(ariadne._SYSTEMS)}")
    for name, s in ariadne._SYSTEMS.items():
        print(f"    {name:<18s}  mu={s.mu:.6e}  L*={s.L_star:>12.0f} km   "
              f"T*={s.T_star/86400.0:.3f} d")
    print(f"\n  Top-level API: ariadne.system / lyapunov_family / halo_family / "
          f"gateway_nrho / discover_tno / helmholtz_hjb / certify_route")
    return 0


def _cmd_systems(args: argparse.Namespace) -> int:
    import ariadne
    for name, s in ariadne._SYSTEMS.items():
        print(f"{name:<18s}  mu={s.mu:.6e}  L*={s.L_star:>14.1f} km  "
              f"T*={s.T_star/86400.0:.4f} d  V*={s.V_star:.4f} km/s")
    return 0


def _cmd_lyapunov(args: argparse.Namespace) -> int:
    import ariadne
    fam = ariadne.lyapunov_family(point=args.point, system_name=args.system,
                                  n=args.n)
    print(f"{args.system} {args.point} Lyapunov family: {len(fam)} orbits")
    print(f"  Jacobi range: {fam[0].orbit.jacobi:.4f}  -> {fam[-1].orbit.jacobi:.4f}")
    print(f"  amplitude:    {fam[0].amplitude:.4e} -> {fam[-1].amplitude:.4e}")
    print(f"  period range (d): {fam[0].orbit.period * ariadne.system(args.system).T_star/86400.0:.2f}"
          f"  -> {fam[-1].orbit.period * ariadne.system(args.system).T_star/86400.0:.2f}")
    return 0


def _cmd_nrho(args: argparse.Namespace) -> int:
    import numpy as np
    import ariadne
    from ariadne.data.constants import R_MOON
    from ariadne.dynamics.cr3bp import propagate
    from ariadne.orbits.differential_correction import monodromy

    em = ariadne.system("EARTH_MOON")
    nrho = ariadne.gateway_nrho()
    sol = propagate(nrho.s0, (0.0, nrho.period), em.mu,
                    t_eval=np.linspace(0.0, nrho.period, 800))
    d_moon = np.sqrt((sol.y[0] - (1 - em.mu)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * em.L_star
    floq = float(np.max(np.abs(np.linalg.eigvals(monodromy(em.mu, nrho)))))
    print(f"Gateway-class L2 NRHO (9:2 resonant)")
    print(f"  period       = {nrho.period * em.T_star / 86400.0:.3f} d   (Gateway spec ~6.56)")
    print(f"  perilune     = {d_moon.min():.0f} km  (alt {d_moon.min() - R_MOON:.0f} km over pole)")
    print(f"  apolune      = {d_moon.max():.0f} km   (Gateway spec ~70,000)")
    print(f"  Floquet max  = {floq:.2f}  (843x more stable than a deep L1 Lyapunov)")
    print(f"  periodic to residual {getattr(nrho, 'residual', getattr(nrho, 'half_period_residual', 0)):.1e}")
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    import numpy as np
    import ariadne
    from ariadne.data.constants import GM_SUN, AU_KM

    print(f"Fetching real MPC astrometry for designation '{args.designation}' "
          f"(window {args.window} d)...")
    fit = ariadne.discover_tno(args.designation, window_days=args.window)
    if fit is None:
        print("  too few tracklets to fit", file=sys.stderr)
        return 2

    r = np.asarray(fit["x_fit"]); v = np.asarray(fit["v_fit"])
    rn = float(np.linalg.norm(r)); vn = float(np.linalg.norm(v))
    a_au = (1.0 / (2.0 / rn - vn ** 2 / GM_SUN)) / AU_KM
    h = np.cross(r, v); hn = float(np.linalg.norm(h))
    ecc = float(np.linalg.norm(np.cross(v, h) / GM_SUN - r / rn))
    inc = math.degrees(math.acos(max(-1, min(1, h[2] / hn))))
    grade = "EXCELLENT" if fit["rms_arcsec"] < 1 else "GOOD" if fit["rms_arcsec"] < 10 else "POOR"
    print(f"\n  IOD seed: r={fit['iod']['r_au']:.1f} AU, rdot={fit['iod']['rdot']:+.2f} km/s")
    print(f"  FIT:      a={a_au:.2f} AU,  e={ecc:.3f},  i={inc:.2f} deg")
    print(f"  residual: {fit['rms_arcsec']:.2f}\" RMS  [{grade}]")
    return 0 if fit["rms_arcsec"] < 10.0 else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    bench = root / "benchmarks" / "reference_targets.py"
    if not bench.exists():
        print(f"benchmark script not found at {bench}", file=sys.stderr)
        return 2
    import runpy
    sys.argv = [str(bench)]
    try:
        runpy.run_path(str(bench), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def _cmd_nightly(args: argparse.Namespace) -> int:
    """Run one nightly discovery iteration. Idempotent on the candidate store.

    --cones accepts "ra1,dec1,r1;ra2,dec2,r2;..." for multi-patch sweeps.
    """
    from .discovery.operations.nightly import NightlyConfig, run_nightly
    from .discovery.operations.alerts import FileSink, WebhookSink
    sinks = []
    if args.alert_log:
        sinks.append(FileSink(args.alert_log))
    if args.webhook_url:
        sinks.append(WebhookSink(args.webhook_url, format=args.webhook_format))
    cones = None
    if args.cones:
        cones = []
        for chunk in args.cones.split(";"):
            parts = chunk.strip().split(",")
            if len(parts) != 3:
                print(f"--cones: bad entry {chunk!r}; expected 'ra,dec,radius'",
                      file=sys.stderr)
                return 2
            cones.append((float(parts[0]), float(parts[1]), float(parts[2])))
    cfg = NightlyConfig(
        store_path=args.store, source=args.source,
        ra=args.ra, dec=args.dec, radius_deg=args.radius, cones=cones,
        mjd_start=args.mjd_start, mjd_end=args.mjd_end,
        max_alerts=args.max_alerts,
        rms_threshold_arcsec=args.rms_threshold,
        use_helio_linc=args.helio_linc,
        do_xmatch=not args.no_skybot,
        dry_run=args.dry_run,
        alert_sinks=sinks,
    )
    summary = run_nightly(cfg)
    print(f"\n=== summary: {summary}")
    return 0


def _cmd_followup(args: argparse.Namespace) -> int:
    """Emit the prioritised follow-up target list for a given MJD."""
    from .discovery.operations.candidate_store import CandidateStore
    from .discovery.followup import next_night_targets
    store = CandidateStore(args.store)
    cands = store.discovery_candidates() if args.discovery_only else list(store)
    targets = next_night_targets(cands, args.mjd, max_sigma_arcsec=args.max_sigma,
                                  n_samples=args.n_samples)
    if not targets:
        print(f"No candidates with predictable position (store has {len(cands)} candidates).")
        return 1
    print(f"\nFollow-up targets for MJD {args.mjd:.3f} ({len(targets)} candidates):\n")
    print(f"  {'key':<26s}  {'RA':>8s}  {'Dec':>8s}  {'sigma':>7s}  {'r AU':>5s}  rate/h")
    for t in targets:
        print(f"  {t['key']:<26s}  {t['ra_deg']:>8.3f}  {t['dec_deg']:>+8.3f}  "
              f"{t['sigma_arcsec']:>7.1f}  {t['range_au']:>5.1f}  {t['v_sky_arcsec_hr']:>6.2f}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Rank candidates by quality score; print one line per candidate."""
    from .discovery.operations.candidate_store import CandidateStore
    from .discovery.scoring import rank_candidates
    store = CandidateStore(args.store)
    ranked = rank_candidates(list(store))
    if not ranked:
        print(f"Store is empty: {args.store}")
        return 1
    print(f"\nRanked candidates ({len(ranked)} total):\n")
    print(f"  {'grade':<5s} {'score':>5s}  {'key':<26s}  {'RMS':>6s}  {'arc':>5s}  "
          f"runs  obs  status   skybot")
    for c, s in ranked:
        rms = c.rms_history[-1][1] if c.rms_history else float("nan")
        arc = c.last_seen_mjd - c.first_seen_mjd
        sky = ",".join(c.skybot_names) or "-"
        print(f"  {s.grade():<5s} {s.total:>5.2f}  {c.key:<26s}  {rms:>6.2f}  "
              f"{arc:>5.1f}  {c.n_runs:>4d}  {len(c.rms_history):>3d}  "
              f"{c.status:<8s} {sky}")
    return 0


def _cmd_mpc_submit(args: argparse.Namespace) -> int:
    """Emit a MPC 80-column astrometric submission body for a candidate."""
    import json
    from .discovery.mpc_submit import MPCHeader, format_record, header_lines, _pack_temp_designation
    from .discovery.operations.candidate_store import CandidateStore
    store = CandidateStore(args.store)
    c = store.get(args.key)
    if c is None:
        print(f"Candidate {args.key} not in store {args.store}", file=sys.stderr)
        return 2
    header = MPCHeader(
        observatory_code=args.obscode, contact=args.contact,
        observers=args.observer, measurers=args.observer,
        telescope=args.telescope, ack_keyword=f"Ariadne_{c.key[:8]}",
        ack_email=args.email,
        comment="Generated by Ariadne candidate dashboard.",
    )
    print("\n".join(header_lines(header)))
    desig = _pack_temp_designation(c.key)
    if not c.rms_history:
        print("(no astrometry stored for this candidate)", file=sys.stderr)
        return 1
    print(format_record(
        mjd=c.last_seen_mjd, ra_deg=c.ra, dec_deg=c.dec,
        designation=desig, observatory_code=args.obscode,
        discovery_asterisk=True,
    ))
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the read-only Flask dashboard (Ctrl-C to stop)."""
    from .discovery.dashboard import main as dashboard_main
    argv = ["--store", args.store, "--host", args.host, "--port", str(args.port)]
    if args.alerts:
        argv += ["--alerts", args.alerts]
    if args.debug:
        argv.append("--debug")
    return dashboard_main(argv)


def _cmd_discover_ztf(args: argparse.Namespace) -> int:
    """Subscribe to ALeRCE/ZTF in a cone+time window, run the discovery pipeline."""
    import time
    from .discovery.brokers.alerce import AlerceZTFBroker
    from .discovery.brokers.base import collect
    from .discovery import realtime
    print(f"ALeRCE/ZTF cone search: ({args.ra}, {args.dec}) r={args.radius} deg, "
          f"MJD {args.mjd_start}..{args.mjd_end}")
    broker = AlerceZTFBroker()
    t0 = time.time()
    try:
        alerts = collect(broker.query_cone(args.ra, args.dec, args.radius,
                                            args.mjd_start, args.mjd_end,
                                            max_alerts=args.max_alerts),
                          max_n=args.max_alerts)
    except Exception as e:
        print(f"  ALeRCE query failed: {e}", file=sys.stderr)
        return 2
    print(f"  fetched {len(alerts)} alerts in {time.time()-t0:.1f}s\n")
    if not alerts:
        return 1
    res = realtime.run_pipeline(alerts,
                                 cluster_pos_tol_arcsec=args.cluster_arcsec,
                                 rms_threshold_arcsec=args.rms_threshold,
                                 do_xmatch=not args.no_skybot)
    candidates = [t for t in res
                   if t.get("status") == "accepted"
                   and t.get("xmatch", {}).get("n_known") == 0]
    return 0 if candidates else 1


def _cmd_tutorial(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    examples = root / "examples"
    candidates = sorted(examples.glob(f"{args.number:02d}_*.py"))
    if not candidates:
        print(f"tutorial {args.number:02d} not found in {examples}", file=sys.stderr)
        return 2
    script = candidates[0]
    (root / "examples_out").mkdir(exist_ok=True)
    import runpy
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ariadne",
        description="Python-native cislunar mission design + TNO discovery toolkit",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print Ariadne version + capability summary"
                   ).set_defaults(func=_cmd_info)
    sub.add_parser("systems", help="list available CR3BP systems"
                   ).set_defaults(func=_cmd_systems)

    p_lyap = sub.add_parser("lyapunov", help="build a Lyapunov orbit family")
    p_lyap.add_argument("--point", choices=("L1", "L2"), default="L1")
    p_lyap.add_argument("--system", default="EARTH_MOON")
    p_lyap.add_argument("--n", type=int, default=30)
    p_lyap.set_defaults(func=_cmd_lyapunov)

    sub.add_parser("nrho", help="construct the Gateway-class NRHO"
                   ).set_defaults(func=_cmd_nrho)

    p_disc = sub.add_parser("discover", help="fit a TNO orbit from MPC astrometry")
    p_disc.add_argument("designation", help="MPC packed or numbered designation (e.g., 90377 for Sedna)")
    p_disc.add_argument("--window", type=int, default=720, help="opposition-window days (default 720)")
    p_disc.set_defaults(func=_cmd_discover)

    sub.add_parser("benchmark", help="run the 16-check reference benchmark suite"
                   ).set_defaults(func=_cmd_benchmark)

    p_n = sub.add_parser("nightly",
        help="one nightly discovery iteration -- pull, filter, dedupe vs store, fire alerts")
    p_n.add_argument("--store", required=True, help="path to JSON candidate store")
    p_n.add_argument("--source", default="alerce_ztf",
                     choices=("alerce_ztf", "synthetic", "synthetic_kepler"))
    p_n.add_argument("--ra", type=float, default=180.0)
    p_n.add_argument("--dec", type=float, default=20.0)
    p_n.add_argument("--radius", type=float, default=5.0)
    p_n.add_argument("--mjd-start", type=float, default=None)
    p_n.add_argument("--mjd-end", type=float, default=None)
    p_n.add_argument("--max-alerts", type=int, default=1000)
    p_n.add_argument("--rms-threshold", type=float, default=15.0)
    p_n.add_argument("--no-skybot", action="store_true")
    p_n.add_argument("--dry-run", action="store_true")
    p_n.add_argument("--alert-log", default=None, help="JSON-lines alert audit log path")
    p_n.add_argument("--webhook-url", default=None, help="Slack/Discord/generic webhook URL")
    p_n.add_argument("--webhook-format", default="slack", choices=("slack", "discord", "raw"))
    p_n.add_argument("--cones", default=None,
                     help='multi-cone sweep "ra1,dec1,r1;ra2,dec2,r2;..." (overrides --ra/--dec/--radius)')
    p_n.add_argument("--helio-linc", action="store_true",
                     help="use the full HelioLinC linker (slower but recovers more links)")
    p_n.set_defaults(func=_cmd_nightly)

    p_f = sub.add_parser("followup",
        help="emit prioritised follow-up target list for a given MJD")
    p_f.add_argument("--store", required=True)
    p_f.add_argument("--mjd", type=float, required=True, help="MJD of the night to schedule for")
    p_f.add_argument("--max-sigma", type=float, default=600.0,
                     help="drop candidates whose 1-sigma > N arcsec (default 600)")
    p_f.add_argument("--n-samples", type=int, default=30, help="Monte-Carlo samples for sigma estimate")
    p_f.add_argument("--discovery-only", action="store_true",
                     help="only candidates with NO SkyBoT match")
    p_f.set_defaults(func=_cmd_followup)

    p_s = sub.add_parser("score", help="rank candidates by quality score")
    p_s.add_argument("--store", required=True)
    p_s.set_defaults(func=_cmd_score)

    p_m = sub.add_parser("mpc-submit",
        help="emit MPC 80-column astrometric submission body for one candidate")
    p_m.add_argument("--store", required=True)
    p_m.add_argument("--key", required=True, help="canonical key of the candidate to submit")
    p_m.add_argument("--obscode", default="500", help="MPC observatory code (default 500 = geocenter)")
    p_m.add_argument("--observer", default="Anonymous",
                     help="observer name(s) for COD/OBS/MEA")
    p_m.add_argument("--contact", default="Anonymous, anon@example.com",
                     help="contact line for CON")
    p_m.add_argument("--telescope", default="Synthetic; via Ariadne pipeline",
                     help="telescope description for TEL")
    p_m.add_argument("--email", default="anon@example.com",
                     help="acknowledgement email for AC2")
    p_m.set_defaults(func=_cmd_mpc_submit)

    p_dash = sub.add_parser("dashboard", help="launch read-only Flask dashboard")
    p_dash.add_argument("--store", required=True)
    p_dash.add_argument("--alerts", default=None, help="JSONL alert log to tail (optional)")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=5050)
    p_dash.add_argument("--debug", action="store_true")
    p_dash.set_defaults(func=_cmd_dashboard)

    p_ztf = sub.add_parser("discover-ztf",
        help="subscribe to ALeRCE/ZTF in a sky/time window, run the moving-object pipeline")
    p_ztf.add_argument("--ra", type=float, default=180.0, help="cone centre RA (deg)")
    p_ztf.add_argument("--dec", type=float, default=20.0, help="cone centre Dec (deg)")
    p_ztf.add_argument("--radius", type=float, default=3.0, help="cone radius (deg)")
    p_ztf.add_argument("--mjd-start", type=float, required=True, help="start MJD")
    p_ztf.add_argument("--mjd-end", type=float, required=True, help="end MJD")
    p_ztf.add_argument("--max-alerts", type=int, default=2000)
    p_ztf.add_argument("--cluster-arcsec", type=float, default=1.5,
                        help="same-night clustering radius (default 1.5\")")
    p_ztf.add_argument("--rms-threshold", type=float, default=15.0,
                        help="IOD+LM acceptance RMS threshold (default 15\")")
    p_ztf.add_argument("--no-skybot", action="store_true",
                        help="skip the SkyBoT cross-match (faster, but candidates aren't filtered)")
    p_ztf.set_defaults(func=_cmd_discover_ztf)

    p_tut = sub.add_parser("tutorial", help="run a numbered tutorial script (1..9)")
    p_tut.add_argument("number", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9))
    p_tut.set_defaults(func=_cmd_tutorial)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
