"""Catalog-mode discovery survey -- sweep the sky by QUERYING the NSC detection
catalog (68B pre-extracted DECam detections), NEVER downloading images.

Each tile is a small NSC cone query (kilobytes) -> build within-night rate tracks
-> cross-night link -> COHERENCE vet -> scrambled-control floor -> SkyBoT
known-SSO arbitration -> any vetted unknown above the floor is persisted to the
shared discovery ledger. Tiles a band around the ecliptic, tracks which tiles are
done in state, and processes a budgeted number per invocation so a scheduler can
sweep the queryable archive over days. No FITS, no petabytes.

  python scripts/run_catalog_survey.py --tiles-per-run 20 --radius 0.12

Needs the NOIRLab Data Lab client (`dl`). Small cones use synchronous queries
(fast); for very dense cones a free Data Lab account enables async. Best of all,
run this ON the Data Lab Jupyter platform -- compute next to the data, zero egress.
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_auto_discovery import load_ledger, is_dup, LEDGER, WATCH, DATA  # noqa: E402

STATE = DATA / "catalog_survey_state.json"


def ecliptic_tiles(step_deg=1.0, dec_band=22.0, gal_plane_avoid=15.0):
    """Tiles along the ecliptic band (asteroid-dense), skipping the galactic plane."""
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        have_astropy = True
    except Exception:
        have_astropy = False
    tiles = []
    ra = 0.0
    while ra < 360.0:
        # ecliptic dec for this RA (obliquity 23.44; approx along the ecliptic)
        lam = math.radians(ra)
        dec = math.degrees(math.asin(math.sin(math.radians(23.44)) * math.sin(lam)))
        if abs(dec) <= dec_band:
            skip = False
            if have_astropy:
                b = SkyCoord(ra, dec, unit="deg").galactic.b.deg
                skip = abs(b) < gal_plane_avoid
            if not skip:
                tiles.append((round(ra, 3), round(dec, 3)))
        ra += step_deg
    return tiles


def datalab_login():
    """Authenticate to NOIRLab Data Lab if credentials are present (FREE account at
    https://datalab.noirlab.edu). Enables async server-side queries on the 68B-row
    nsc_dr2.meas table -- anonymous SYNC queries time out. Returns True if authed.
    Set COH_DATALAB_TOKEN, or DATALAB_USER + DATALAB_PASS, in the environment."""
    import os
    try:
        from dl import authClient as ac
    except Exception:
        return False
    tok = os.environ.get("COH_DATALAB_TOKEN")
    if tok:
        try:
            ac.set_token(tok); return True
        except Exception:
            pass
    user, pw = os.environ.get("DATALAB_USER"), os.environ.get("DATALAB_PASS")
    if user and pw:
        try:
            ac.login(user, pw); return True
        except Exception:
            pass
    return False


def nsc_query(ra, dec, radius, top=40000, authed=False, timeout_s=60):
    """NSC cone query -> structured detections (ra,dec,mjd,mag,fwhm). When authed,
    uses an ASYNC server-side job (robust on the huge table); otherwise a short
    sync attempt (usually times out anonymously). Returns None on failure/empty."""
    from dl import queryClient as qc
    sql = (f"SELECT ra,dec,mjd,mag_auto,fwhm FROM nsc_dr2.meas "
           f"WHERE q3c_radial_query(ra,dec,{ra},{dec},{radius}) AND mjd>0 LIMIT {top}")
    try:
        if authed:
            jobid = qc.query(sql=sql, async_=True)
            for _ in range(150):                 # poll up to ~5 min
                st = qc.status(jobid)
                if st == "COMPLETED":
                    csv = qc.results(jobid); break
                if st in ("ERROR", "ABORTED"):
                    return None
                time.sleep(2)
            else:
                return None
        else:
            try:
                csv = qc.query(sql=sql, fmt="csv", timeout=timeout_s)
            except TypeError:
                csv = qc.query(sql=sql, fmt="csv")
    except Exception:
        return None
    lines = csv.strip().splitlines()
    if len(lines) < 3:
        return None
    return np.genfromtxt(io.StringIO(csv), delimiter=",", names=True)


def discover(arr, *, min_rate=0.3, max_rate=150.0, min_points=3, min_nights=3,
             pos_tol=2.5, vet_tau=2.0):
    """Catalog detections -> vetted multi-night candidate chains (coherence vet)
    + the scrambled-control chance floor. Returns (candidates, floor_count)."""
    from ariadne.discovery.imaging.source_extraction import Source
    from ariadne.discovery.imaging.rate_constrained_linker import (
        build_within_night_tracks, link_rate_constrained)
    from ariadne.discovery.imaging.coherence_vet import track_energy
    by_night = defaultdict(list)
    for r in arr:
        s = Source(ra=float(r["ra"]), dec=float(r["dec"]),
                   flux=float(10 ** (-0.4 * float(r["mag_auto"]))), mag=float(r["mag_auto"]),
                   fwhm_px=float(r["fwhm"]) if "fwhm" in arr.dtype.names else 4.0,
                   mjd=float(r["mjd"]), image_id=f"nsc{float(r['mjd']):.5f}", x=0.0, y=0.0)
        by_night[math.floor(s.mjd - 0.5)].append(s)
    nights = sorted(by_night)
    if len(nights) < min_nights:
        return [], 0, len(nights)
    tracks = []
    for n in nights:
        tracks += build_within_night_tracks(
            by_night[n], min_rate_arcsec_hr=min_rate, max_rate_arcsec_hr=max_rate,
            pos_tol_arcsec=pos_tol, min_points=min_points)
    chains = link_rate_constrained(tracks, min_nights=min_nights)

    def coh(ch):
        ra = [t.ra_mid for t in ch]; dec = [t.dec_mid for t in ch]
        mjd = [t.jd_mid - 2400000.5 for t in ch]
        return track_energy(ra, dec, mjd)
    vetted = [c for c in chains if coh(c) <= vet_tau]
    # scrambled floor
    import copy
    off = nights[len(nights) // 2]
    scr = []
    for t in tracks:
        t2 = copy.copy(t)
        if t2.night == off:
            t2.ra_mid += 0.07; t2.dec_mid += 0.05
        scr.append(t2)
    scr_v = [c for c in link_rate_constrained(scr, min_nights=min_nights) if coh(c) <= vet_tau]
    return vetted, len(scr_v), len(nights)


def skybot_unknown(ch, radius_arcsec=20.0):
    """True if NO known solar-system object is within radius of the chain midpoint
    at its epoch (i.e. a genuine unknown candidate)."""
    try:
        from astroquery.imcce import Skybot
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u
        cc = ch[len(ch) // 2]
        tb = Skybot.cone_search(SkyCoord(cc.ra_mid, cc.dec_mid, unit="deg"),
                                radius_arcsec * u.arcsec, Time(cc.jd_mid, format="jd"))
        return tb is None or len(tb) == 0
    except Exception:
        return True   # SkyBoT raises when nothing found -> treat as unknown


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-per-run", type=int, default=15)
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--step-deg", type=float, default=1.0)
    ap.add_argument("--min-nights", type=int, default=3)
    ap.add_argument("--top", type=int, default=40000)
    ap.add_argument("--no-skybot", action="store_true")
    ap.add_argument("--allow-sync", action="store_true",
                    help="attempt unauthenticated sync queries anyway (usually time out)")
    args = ap.parse_args()

    authed = datalab_login()
    if not authed and not args.allow_sync:
        print("[catalog] NOT authenticated -> skipping (no tiles consumed). The NSC table\n"
              "          times out for anonymous sync queries. Get a FREE account at\n"
              "          https://datalab.noirlab.edu and set COH_DATALAB_TOKEN (or\n"
              "          DATALAB_USER + DATALAB_PASS); then this sweeps the archive for real.\n"
              "          (Use --allow-sync to attempt anyway.)", flush=True)
        return 0

    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "runs": 0, "stats": {}}
    done = set(tuple(t) for t in state["done"])
    tiles = [t for t in ecliptic_tiles(args.step_deg) if t not in done]
    if not tiles:
        print("[catalog] all tiles swept; resetting queue"); done = set(); tiles = ecliptic_tiles(args.step_deg)
    batch = tiles[:args.tiles_per_run]
    print(f"[catalog] {'authenticated (async)' if authed else 'sync (forced)'}; "
          f"sweeping {len(batch)} tiles (of {len(tiles)} remaining), radius {args.radius} deg", flush=True)

    ledger = load_ledger()
    now = datetime.now(timezone.utc).isoformat()
    total_new = total_cov = total_cand = total_unknown = 0
    for (ra, dec) in batch:
        arr = nsc_query(ra, dec, args.radius, top=args.top, authed=authed)
        done.add((ra, dec))
        if arr is None or arr.size < args.min_nights:
            continue
        cand, floor, n_nights = discover(arr, min_nights=args.min_nights)
        if n_nights < args.min_nights:
            continue
        total_cov += 1
        total_cand += len(cand)
        for c in cand:
            unknown = True if args.no_skybot else skybot_unknown(c)
            if not unknown:
                continue
            total_unknown += 1
            entry = {
                "ra_deg": float(c[0].ra_mid), "dec_deg": float(c[0].dec_mid),
                "rate_arcsec_hr": float(np.mean([t.rate_arcsec_hr for t in c])),
                "nights": sorted({int(t.night) for t in c}),
                "n_tracklets": len(c),
                "mag": float(np.median([t.mag for t in c])),
                "source": "nsc_catalog", "tile": [ra, dec],
                "above_floor": len(cand) > floor,
                "field_id": f"nsc_ra{ra:.1f}_dec{dec:+.1f}", "first_seen_utc": now,
            }
            if not is_dup(entry, ledger):
                with open(LEDGER, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                ledger.append(entry); total_new += 1
        print(f"  tile ({ra:.1f},{dec:+.1f}): {arr.size} det, {n_nights} nights, "
              f"{len(cand)} cand, floor {floor}", flush=True)

    state["done"] = [list(t) for t in done]
    state["runs"] += 1
    state["stats"] = {"last_run_utc": now, "tiles_with_coverage": total_cov,
                      "candidates": total_cand, "unknown": total_unknown, "new_in_ledger": total_new}
    STATE.write_text(json.dumps(state, indent=2))

    # refresh the shared watch report
    from run_auto_discovery import write_watch
    runstate = json.loads((DATA / "auto_discovery_state.json").read_text()) \
        if (DATA / "auto_discovery_state.json").exists() else {"runs": []}
    runstate["runs"].append({"utc": now, "field": f"NSC catalog x{len(batch)} tiles",
                             "recovered": total_cand - total_unknown, "recoverable": total_cand,
                             "candidates": total_unknown, "floor": "-",
                             "above_floor": False, "new_in_ledger": total_new})
    (DATA / "auto_discovery_state.json").write_text(json.dumps(runstate, indent=2))
    write_watch(runstate, load_ledger())

    print(f"[catalog] swept {len(batch)} tiles | {total_cov} had multi-night coverage | "
          f"{total_cand} candidates | {total_unknown} SkyBoT-unknown | {total_new} new in ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
