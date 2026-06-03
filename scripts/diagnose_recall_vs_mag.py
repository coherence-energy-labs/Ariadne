"""Is the 12-15%% recall a real failure, or a denominator full of invisibles?

The discovery-recall headline counts EVERY predicted known asteroid on the
footprint, regardless of brightness -- including objects fainter than the
detection limit, which NO instrument could see. This bins recall by the
ephemeris-PREDICTED apparent magnitude (eph[:,2]) so we see the true
completeness curve for real known asteroids:

  - if recall is ~high for bright objects and falls off near ~21.9 (the known
    point-source 50%% limit), the front-end is healthy and the low headline is
    a metric artifact (we were scoring ourselves on invisible objects);
  - if BRIGHT objects are being missed, that is a real detection/cross-match
    bug worth fixing -- a far more valuable finding.

Loads ONLY the science exposure (fast). Honest either way.
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
    ap.add_argument("--match-arcsec", type=float, default=2.5)
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
    det_ra, det_dec, det_mag = [], [], []
    for ccd in ccds:
        try:
            srcs = detect_sources_in_image(
                np.asarray(ccd.science, float), ccd.wcs, mjd=mjd,
                image_id=f"p{ccd.ccdnum}", fwhm_px=3.5,
                threshold_sigma=args.sigma, zeropoint_mag=ccd.magzero)
        except Exception:
            continue
        for s in srcs:
            det_ra.append(s.ra); det_dec.append(s.dec); det_mag.append(s.mag)
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

    # bin every predicted-known-on-footprint by PREDICTED apparent mag
    bins = [(0, 18), (18, 19), (19, 20), (20, 21), (21, 21.5),
            (21.5, 22), (22, 22.5), (22.5, 23), (23, 99)]
    found = {b: 0 for b in bins}
    total = {b: 0 for b in bins}
    for i in range(len(recs)):
        if np.isnan(eph[i, 0]) or np.isnan(eph[i, 2]):
            continue
        cd = math.cos(math.radians(eph[i, 1]))
        sep = np.hypot((det_ra - eph[i, 0]) * cd, det_dec - eph[i, 1]) * 3600
        if sep.size == 0 or sep.min() > 120:    # not on a covered CCD
            continue
        m = float(eph[i, 2])
        for b in bins:
            if b[0] <= m < b[1]:
                total[b] += 1
                if sep.min() <= args.match_arcsec:
                    found[b] += 1
                break

    print(f"\n=== RECALL vs PREDICTED MAGNITUDE (real DECam, {len(ccds)} CCDs) ===")
    print(f"  {'mag bin':>12}   {'recovered/total':>16}   recall")
    cum_f = cum_t = 0
    bright_f = bright_t = 0
    for b in bins:
        f, t = found[b], total[b]
        cum_f += f; cum_t += t
        if b[1] <= 21.5:
            bright_f += f; bright_t += t
        if t == 0:
            continue
        bar = "#" * int(round(20 * f / max(t, 1)))
        print(f"  {b[0]:5.1f}-{b[1]:<5.1f}   {f:7d}/{t:<7d}      "
              f"{f/t*100:5.0f}%  {bar}")
    print(f"\n  recall for DETECTABLE objects (predicted r < 21.5): "
          f"{bright_f}/{bright_t} = {bright_f/max(bright_t,1)*100:.0f}%")
    print(f"  overall (all predicted, incl. fainter than limit): "
          f"{cum_f}/{cum_t} = {cum_f/max(cum_t,1)*100:.0f}%")
    print(f"  fraction of denominator fainter than r=21.5: "
          f"{(cum_t-bright_t)/max(cum_t,1)*100:.0f}% (invisible to any method)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
