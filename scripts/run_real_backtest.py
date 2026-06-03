"""Multi-night real-data backtest: known-asteroid recovery vs magnitude.

For each real DECam exposure: extract sources from every CCD (instcal WCS),
predict every catalogue asteroid in the covered footprint at the exposure
epoch with the accurate ephemeris (N-body + ecliptic->equatorial frame +
per-object light-time + CTIO topocentric), and measure how many are
recovered (a real detection within the match radius), binned by magnitude.

This is the real-pixel analogue of synthetic injection-recovery -- the
honest "how good are we" metric, with the ephemeris validated against
JPL Horizons to sub-arcsec.

Detections are cached per exposure (npz) so re-runs skip extraction.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OBS_CODE = "807"   # CTIO / Blanco (DECam)


def extract_or_load(fits_path: str, cache_path: str, detect_sigma: float):
    if os.path.exists(cache_path):
        c = np.load(cache_path)
        return c["ra"], c["dec"], c["mag"], float(c["mjd"])
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    print(f"  extracting {Path(fits_path).name} ...", flush=True)
    t0 = time.time()
    inst = load_decam_instcal(fits_path, read_dqm=False)
    ra, dec, mag = [], [], []
    for ccd in inst.ccds:
        if ccd.wcs is None:
            continue
        try:
            srcs = detect_sources_in_image(
                ccd.science, ccd.wcs, mjd=inst.mjd, image_id=f"c{ccd.ccdnum}",
                fwhm_px=3.5, threshold_sigma=detect_sigma,
                zeropoint_mag=(ccd.magzero if ccd.magzero > 0 else None))
        except Exception:
            continue
        for s in srcs:
            ra.append(s.ra); dec.append(s.dec); mag.append(s.mag)
    ra = np.array(ra); dec = np.array(dec); mag = np.array(mag)
    np.savez(cache_path, ra=ra, dec=dec, mag=mag, mjd=inst.mjd)
    print(f"    {len(ra)} sources in {time.time()-t0:.0f}s", flush=True)
    return ra, dec, mag, inst.mjd


def backtest_night(db, fits_path, cache_path, *, match_arcsec, detect_sigma):
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    det_ra, det_dec, det_mag, mjd = extract_or_load(fits_path, cache_path, detect_sigma)
    if len(det_ra) == 0:
        return None
    # Coarse field select (2-body), then accurate refine
    in_field = pick_orbits_in_field(
        db, mjd, (det_ra.min(), det_ra.max()), (det_dec.min(), det_dec.max()),
        max_mag=22.5, limit_candidates=1600000)
    recs = [x[0] for x in in_field]
    obs = observatory_geo_km(OBS_CODE, mjd)
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=obs)
    # Match each predicted asteroid to nearest detection
    rows = []
    for i, rec in enumerate(recs):
        if np.isnan(eph[i, 0]):
            continue
        cd = math.cos(math.radians(eph[i, 1]))
        sep = np.hypot((det_ra - eph[i, 0]) * cd, det_dec - eph[i, 1]) * 3600
        rows.append((rec.designation, float(eph[i, 2]), float(sep.min())))
    return {"mjd": mjd, "n_det": len(det_ra), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real_decam_recovery")
    ap.add_argument("--db", default=os.environ.get("ARIADNE_DB", "data/recovery_clean.db"))
    ap.add_argument("--match-arcsec", type=float, default=3.0)
    ap.add_argument("--detect-sigma", type=float, default=5.0)
    args = ap.parse_args()

    from ariadne.discovery.imaging.detection_db import open_db
    db = open_db(args.db)

    fits_files = sorted(Path(args.data_dir).glob("c4d_*_ooi_r_v1.fits.fz"))
    print(f"Backtest over {len(fits_files)} real DECam exposures "
          f"(match radius {args.match_arcsec}\"):\n")

    bins = [(0, 18), (18, 19), (19, 20), (20, 21), (21, 22.5)]
    grand = {b: [0, 0] for b in bins}   # [predicted, recovered]
    for f in fits_files:
        cache = str(f).replace(".fits.fz", "_dets.npz")
        res = backtest_night(db, str(f), cache,
                               match_arcsec=args.match_arcsec,
                               detect_sigma=args.detect_sigma)
        if res is None:
            continue
        from astropy.time import Time
        iso = Time(res["mjd"], format="mjd").isot[:10]
        rows = res["rows"]
        rec_all = sum(1 for _, _, s in rows if s <= args.match_arcsec)
        print(f"  {iso} (mjd {res['mjd']:.3f}): {res['n_det']} sources, "
              f"{len(rows)} catalog asteroids predicted in footprint, "
              f"{rec_all} recovered @{args.match_arcsec}\"")
        for lo, hi in bins:
            pred = [r for r in rows if lo <= r[1] < hi]
            rec = [r for r in pred if r[2] <= args.match_arcsec]
            grand[(lo, hi)][0] += len(pred)
            grand[(lo, hi)][1] += len(rec)

    print("\n=== COMBINED real known-asteroid recovery vs magnitude ===")
    print(f"  {'mag bin':>10s}  {'predicted':>9s}  {'recovered':>9s}  {'recall':>7s}")
    tot_p = tot_r = 0
    for (lo, hi), (p, r) in grand.items():
        tot_p += p; tot_r += r
        if p:
            print(f"  {lo:.0f}-{hi:<5.1f}  {p:9d}  {r:9d}  {r/p*100:6.0f}%")
    print(f"  {'TOTAL':>10s}  {tot_p:9d}  {tot_r:9d}  "
          f"{(tot_r/max(tot_p,1))*100:6.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
