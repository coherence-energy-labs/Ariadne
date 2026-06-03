"""WHY are detectable known asteroids being missed? Detection vs matching.

recall-vs-mag showed ~0%% recall at r=20-21 even though static point-source
completeness is ~90%% to r=21.9. Two very different root causes:

  (a) DETECTED but NOT MATCHED: a real detection exists near the predicted
      position but beyond the 2.5" match radius. Cause: ephemeris/orbit
      uncertainty (faint asteroids have short arcs -> arcsec-to-arcmin
      position errors). Fix: per-object ephemeris uncertainty + adaptive
      match radius, NOT more detection power.

  (b) NOT DETECTED: no source within even 30". Cause: the asteroid TRAILS
      (a moving mag-20.5 object spreads its flux over a ~1" trail -> lower
      peak SNR than a static mag-20.5 star) or is simply below limit. Fix:
      trail-aware / matched-filter detection + shift-and-stack depth.

For every predicted known on the covered footprint, this records the nearest
detection separation and bins it. The split tells us which lever matters.
"""
from __future__ import annotations

import os
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/real_decam_recovery")
SCI = DATA / "c4d_240824_013234_ooi_r_v1.fits.fz"
DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ccd", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=5.0)
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    print("loading science exposure ...", flush=True)
    t0 = time.time()
    sci_inst = load_decam_instcal(str(SCI), read_dqm=False)
    mjd = sci_inst.mjd
    ccds = [c for c in sci_inst.ccds if c.wcs is not None and c.magzero > 0][:args.n_ccd]
    det_ra, det_dec = [], []
    # also record per-CCD footprint centers to define "covered" precisely
    for ccd in ccds:
        try:
            srcs = detect_sources_in_image(
                np.asarray(ccd.science, float), ccd.wcs, mjd=mjd,
                image_id=f"p{ccd.ccdnum}", fwhm_px=3.5,
                threshold_sigma=args.sigma, zeropoint_mag=ccd.magzero)
        except Exception:
            continue
        for s in srcs:
            det_ra.append(s.ra); det_dec.append(s.dec)
    det_ra = np.array(det_ra); det_dec = np.array(det_dec)
    print(f"  {len(ccds)} CCDs, {len(det_ra)} detections ({time.time()-t0:.0f}s)",
          flush=True)

    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (det_ra.min(), det_ra.max()),
                                 (det_dec.min(), det_dec.max()),
                                 max_mag=24.0, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd,
                                        observer_geo_km=observatory_geo_km("807", mjd))

    # nearest-detection separation for every predicted known on footprint
    rings = [(0, 2.5), (2.5, 5), (5, 10), (10, 30), (30, 120)]
    mag_groups = [("bright r<20", -99, 20.0),
                  ("mid 20-21.5", 20.0, 21.5),
                  ("faint 21.5-23", 21.5, 23.0)]
    counts = {g[0]: {r: 0 for r in rings} for g in mag_groups}
    counts_none = {g[0]: 0 for g in mag_groups}
    totals = {g[0]: 0 for g in mag_groups}
    seps_recovered = []
    for i in range(len(recs)):
        if np.isnan(eph[i, 0]) or np.isnan(eph[i, 2]):
            continue
        cd = math.cos(math.radians(eph[i, 1]))
        sep = np.hypot((det_ra - eph[i, 0]) * cd, det_dec - eph[i, 1]) * 3600
        if sep.size == 0:
            continue
        smin = float(sep.min())
        if smin > 120:                      # not on a covered CCD
            continue
        m = float(eph[i, 2])
        grp = None
        for name, lo, hi in mag_groups:
            if lo <= m < hi:
                grp = name; break
        if grp is None:
            continue
        totals[grp] += 1
        if smin <= 2.5:
            seps_recovered.append(smin)
        placed = False
        for r in rings:
            if r[0] <= smin < r[1]:
                counts[grp][r] += 1; placed = True; break
        if not placed:
            counts_none[grp] += 1

    print(f"\n=== NEAREST-DETECTION SEPARATION for predicted knowns "
          f"({len(ccds)} CCDs) ===")
    print(f"  (a detection within 2.5\" = matched; 2.5-30\" = detected but "
          f"position-off; >30\"/none = not detected)\n")
    hdr = f"  {'mag group':<16}{'N':>5}" + "".join(f"{f'{r[0]:g}-{r[1]:g}\"':>10}" for r in rings) + f"{'none>120':>10}"
    print(hdr)
    for name, lo, hi in mag_groups:
        row = f"  {name:<16}{totals[name]:>5}"
        for r in rings:
            row += f"{counts[name][r]:>10}"
        row += f"{counts_none[name]:>10}"
        print(row)
    if seps_recovered:
        sr = np.array(seps_recovered)
        print(f"\n  matched-object astrometric residual: median {np.median(sr):.2f}\" "
              f"mean {sr.mean():.2f}\" max {sr.max():.2f}\" (n={len(sr)})")
    # interpretation hint
    det_off = sum(counts[g][r] for g in counts for r in [(2.5,5),(5,10),(10,30)])
    not_det = sum(counts[g][(30,120)] for g in counts) + sum(counts_none.values())
    matched = sum(counts[g][(0,2.5)] for g in counts)
    print(f"\n  SUMMARY: matched(<2.5\")={matched}  "
          f"detected-but-off(2.5-30\")={det_off}  "
          f"not-detected(>30\")={not_det}")
    print(f"  -> if detected-but-off dominates: fix EPHEMERIS/match radius.")
    print(f"  -> if not-detected dominates: fix DETECTION (trail-aware + shift-stack).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
