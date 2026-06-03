"""Automated discovery orchestrator.

Runs the validated discovery pipeline on a field UNATTENDED, flags genuine unknown
candidates that pass coherence vetting AND exceed the chance floor, DEDUPS them
against a persistent ledger so the same object is never re-flagged, records every
run's stats, and surfaces findings in a human-readable watch report.

Designed to be run repeatedly (cron / scheduler) over a queue of fields so the
system keeps checking and accumulating candidates. One invocation = one field =
one cycle. New fields can be auto-fetched from the NOIRLab archive (--fetch).

  python scripts/run_auto_discovery.py --data-dir data/decam_deep_field
  python scripts/run_auto_discovery.py --fetch --ra 180 --dec 0 --nights 4   # pull then run

State lives in data/: discovery_ledger.jsonl (candidates), auto_discovery_state.json
(run history), DISCOVERY_WATCH.md (the report you read).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LEDGER = DATA / "discovery_ledger.jsonl"
STATE = DATA / "auto_discovery_state.json"
WATCH = DATA / "DISCOVERY_WATCH.md"


def load_ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(ln) for ln in LEDGER.read_text().splitlines() if ln.strip()]


def is_dup(cand, ledger, pos_tol_deg=0.02, rate_tol_frac=0.25):
    """A candidate already in the ledger if a prior entry sits within pos_tol on
    sky AND has a consistent rate -- same object seen again, do not re-flag."""
    for e in ledger:
        dra = (cand["ra_deg"] - e["ra_deg"]) * math.cos(math.radians(cand["dec_deg"]))
        dd = cand["dec_deg"] - e["dec_deg"]
        if math.hypot(dra, dd) < pos_tol_deg and \
           abs(cand["rate_arcsec_hr"] - e["rate_arcsec_hr"]) <= rate_tol_frac * max(cand["rate_arcsec_hr"], 1.0):
            return True
    return False


def run_field(data_dir, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_discovery_benchmark.py"),
           "--data-dir", str(data_dir), "--vet-mode", "coherence",
           "--linker-mode", "rate", "--truth-mode", "nbody",
           "--max-detections-per-exposure", str(args.max_det),
           "--max-sources-per-exposure", str(args.max_src),
           "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=False, cwd=str(ROOT))
    summ = out_dir / "discovery_benchmark_summary.json"
    return json.loads(summ.read_text()) if summ.exists() else None


def fetch_field(ra, dec, nights, dest, args):
    """Pull a multi-night DECam field from the NOIRLab archive into `dest`."""
    sys.path.insert(0, str(ROOT / "src"))
    from ariadne.discovery.imaging.noirlab_sia2 import (
        query_decam_exposures, download_decam_exposure)
    dest.mkdir(parents=True, exist_ok=True)
    # instcal 'ooi' science exposures (what the moving-object pipeline needs)
    recs = query_decam_exposures(ra, dec, radius_deg=args.radius, band="r",
                                 proc_type="instcal", include_kinds=("ooi",),
                                 max_results=args.max_fetch)
    # require multi-night coverage BEFORE downloading (asteroid linking needs >=2
    # nights; arbitrary fields often have only 1 -> skip without wasting bandwidth).
    by_night = {}
    for r in recs:
        by_night.setdefault(int(round(r.obs_mjd)), []).append(r)
    if len(by_night) < nights and len(by_night) < 2:
        print(f"  only {len(by_night)} night(s) of coverage here -> skip (need >=2)")
        return 0
    got = 0
    for r in recs:
        try:
            if download_decam_exposure(r, dest) is not None:
                got += 1
        except Exception as ex:
            print(f"  fetch skip: {type(ex).__name__} {ex}")
    print(f"  fetched {got} exposures across {len(by_night)} nights to {dest}")
    return got


def write_watch(state, ledger):
    above = [c for c in ledger if c.get("above_floor")]
    lines = ["# Discovery Watch", "",
             f"_updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_", "",
             f"- runs: {len(state['runs'])}",
             f"- candidates in ledger: {len(ledger)}",
             f"- **candidates above chance floor: {len(above)}**", ""]
    if above:
        lines += ["## Candidates above the chance floor (review these)", "",
                  "| field | RA | Dec | rate \"/hr | nights | coherence | first seen |",
                  "|---|---|---|---|---|---|---|"]
        for c in sorted(above, key=lambda c: -c.get("coherence", 0)):
            lines.append(f"| {c.get('field_id','?')} | {c['ra_deg']:.4f} | {c['dec_deg']:.4f} | "
                         f"{c['rate_arcsec_hr']:.0f} | {len(c.get('nights',[]))} | "
                         f"{c.get('coherence',0):.3f} | {c.get('first_seen_utc','')[:19]} |")
        lines.append("")
    lines += ["## Recent runs", "",
              "| utc | field | recovered | candidates | floor | above floor | new |",
              "|---|---|---|---|---|---|---|"]
    for r in state["runs"][-15:][::-1]:
        lines.append(f"| {r['utc'][:19]} | {r['field']} | {r['recovered']}/{r['recoverable']} | "
                     f"{r['candidates']} | {r['floor']} | {'YES' if r['above_floor'] else 'no'} | "
                     f"{r['new_in_ledger']} |")
    WATCH.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="local field to process")
    ap.add_argument("--field-id", default=None)
    ap.add_argument("--fetch", action="store_true", help="pull a fresh field from NOIRLab first")
    ap.add_argument("--ra", type=float); ap.add_argument("--dec", type=float)
    ap.add_argument("--nights", type=int, default=4)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--max-fetch", type=int, default=24)
    ap.add_argument("--max-det", type=int, default=150000)
    ap.add_argument("--max-src", type=int, default=8000)
    ap.add_argument("--cleanup", action="store_true",
                    help="delete fetched FITS after processing (storage management for the survey loop)")
    args = ap.parse_args()

    if args.fetch:
        if args.ra is None or args.dec is None:
            print("--fetch needs --ra and --dec"); return 2
        field_id = args.field_id or f"auto_ra{args.ra:.1f}_dec{args.dec:+.1f}"
        data_dir = DATA / "auto_fields" / field_id
        if fetch_field(args.ra, args.dec, args.nights, data_dir, args) < 2:
            print("too few exposures fetched; aborting"); return 1
    else:
        if not args.data_dir:
            print("need --data-dir or --fetch"); return 2
        data_dir = Path(args.data_dir); field_id = args.field_id or data_dir.name

    out_dir = DATA / "auto_runs" / field_id
    s = run_field(data_dir, out_dir, args)
    if not s or s.get("status") != "pass":
        print(f"[auto] field={field_id} run did not pass "
              f"({s.get('reason') if s else 'no summary'})")
        return 1

    ledger = load_ledger()
    now = datetime.now(timezone.utc).isoformat()
    new_cands = []
    for c in s.get("candidates", []):
        if not is_dup(c, ledger):
            c2 = dict(c); c2["field_id"] = field_id; c2["first_seen_utc"] = now
            c2["above_floor"] = bool(s.get("above_floor"))
            new_cands.append(c2); ledger.append(c2)
    if new_cands:
        with open(LEDGER, "a") as f:
            for c in new_cands:
                f.write(json.dumps(c) + "\n")

    state = json.loads(STATE.read_text()) if STATE.exists() else {"runs": []}
    state["runs"].append({
        "utc": now, "field": field_id,
        "recovered": s["n_recovered_knowns"], "recoverable": s["n_recoverable_knowns"],
        "candidates": s["n_vetted_unknown_candidates"],
        "floor": s["n_scrambled_unknown_multinight_chains"],
        "above_floor": bool(s.get("above_floor")), "new_in_ledger": len(new_cands)})
    STATE.write_text(json.dumps(state, indent=2))
    write_watch(state, ledger)

    print(f"[auto] field={field_id} recovered={s['n_recovered_knowns']}/{s['n_recoverable_knowns']} "
          f"candidates={s['n_vetted_unknown_candidates']} floor={s['n_scrambled_unknown_multinight_chains']} "
          f"new={len(new_cands)} above_floor={s.get('above_floor')}")
    if s.get("above_floor") and new_cands:
        print(f"  *** {len(new_cands)} NEW candidate(s) ABOVE the chance floor -> review {WATCH} ***")

    if args.cleanup and args.fetch:
        import shutil
        try:
            shutil.rmtree(data_dir); print(f"  cleaned up fetched FITS at {data_dir}")
        except Exception as ex:
            print(f"  cleanup failed: {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
