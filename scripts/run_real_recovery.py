"""Real end-to-end recovery of known asteroids from recent DECam pixels.

THE proof we lacked: take a recent DECam exposure (small epoch delta so
catalog positions are arcsec-accurate), extract sources from the real
pixels, and confirm that real detections fall on the predicted positions
of KNOWN MPC asteroids -- i.e., recover real, named, catalogued objects
from raw pixels, confirmed against the MPC catalog.

Unlike the 9-year-old field (where 2-body/N-body propagation could not
reach arcsec accuracy), the 2024 field is ~1.2 yr from the MPCORB epoch,
so 2-body propagation is good to arcsec for numbered asteroids.

Usage:
  python scripts/run_real_recovery.py <exposure.fits.fz> \
      --db <path> --max-ccds 60 --match-arcsec 3.0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exposure", help="path to a DECam instcal FITS")
    ap.add_argument("--db", default=os.environ.get("ARIADNE_FULL_DB", "data/full_discovery.db"),
                      help="DB with the MPCORB known_objects catalog")
    ap.add_argument("--max-ccds", type=int, default=60)
    ap.add_argument("--detect-sigma", type=float, default=5.0)
    ap.add_argument("--match-arcsec", type=float, default=3.0)
    ap.add_argument("--max-mag", type=float, default=22.5)
    ap.add_argument("--gaia", action="store_true",
                      help="apply Gaia refinement (slow; instcal WCS usually enough)")
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field

    print(f"Loading {args.exposure} ...", flush=True)
    t0 = time.time()
    inst = load_decam_instcal(args.exposure, read_dqm=False)
    print(f"  {len(inst.ccds)} CCDs, mjd={inst.mjd:.4f}, band={inst.band}, "
          f"exptime={inst.exptime_s}s  ({time.time()-t0:.0f}s)", flush=True)

    # 1. Extract sources from every CCD via the instcal WCS
    all_dets = []   # (ra, dec, mag, ccd_name)
    t0 = time.time()
    for ccd in inst.ccds[:args.max_ccds]:
        if ccd.wcs is None:
            continue
        try:
            srcs = detect_sources_in_image(
                ccd.science, ccd.wcs, mjd=inst.mjd,
                image_id=f"ccd{ccd.ccdnum}",
                fwhm_px=3.5, threshold_sigma=args.detect_sigma,
                zeropoint_mag=(ccd.magzero if ccd.magzero > 0 else None))
        except Exception:
            continue
        for s in srcs:
            all_dets.append((s.ra, s.dec, s.mag, ccd.name))
    print(f"  extracted {len(all_dets)} real sources from "
          f"{min(len(inst.ccds), args.max_ccds)} CCDs  ({time.time()-t0:.0f}s)",
          flush=True)

    if not all_dets:
        print("  no sources extracted; aborting", flush=True)
        return 1

    # Footprint actually covered
    ras = [d[0] for d in all_dets]; decs = [d[1] for d in all_dets]
    ra_lo, ra_hi = min(ras), max(ras)
    dec_lo, dec_hi = min(decs), max(decs)
    print(f"  covered footprint: RA {ra_lo:.3f}..{ra_hi:.3f}, "
          f"Dec {dec_lo:.3f}..{dec_hi:.3f}", flush=True)

    # 2. Predict catalog asteroids in the covered footprint at this epoch
    db = open_db(args.db)
    n_known = db.conn.execute("SELECT COUNT(*) FROM known_objects").fetchone()[0]
    print(f"  catalog: {n_known:,} MPCORB orbits", flush=True)
    t0 = time.time()
    in_field = pick_orbits_in_field(
        db, inst.mjd, (ra_lo, ra_hi), (dec_lo, dec_hi),
        max_mag=args.max_mag, limit_candidates=1600000)
    print(f"  {len(in_field)} catalog asteroids predicted on the covered "
          f"pixels at mag<{args.max_mag}  ({time.time()-t0:.0f}s)", flush=True)

    # 3. Cross-match: for each predicted asteroid, find the nearest real
    #    detection. Recovery = a real detection within match_arcsec.
    import numpy as np
    det_ra = np.array([d[0] for d in all_dets])
    det_dec = np.array([d[1] for d in all_dets])
    # cache detections for fast re-analysis
    np.savez("data/real_decam_recovery/_dets_cache.npz",
              ra=det_ra, dec=det_dec,
              mag=np.array([d[2] for d in all_dets]), mjd=inst.mjd)
    recovered = []
    misses = []
    nearest_offsets = []
    for rec, p_ra, p_dec, p_mag, rho in in_field:
        cd = math.cos(math.radians(p_dec))
        sep = np.hypot((det_ra - p_ra) * cd, det_dec - p_dec) * 3600.0
        j = int(np.argmin(sep))
        nearest_offsets.append((rec.designation, p_mag, float(sep[j]),
                                  p_ra, p_dec))
        if sep[j] <= args.match_arcsec:
            recovered.append((rec.designation, p_mag, float(sep[j]),
                                all_dets[j][2]))
        else:
            misses.append((rec.designation, p_mag, float(sep[j])))

    # Diagnostic: distribution of nearest-detection offsets (reveals a
    # systematic prediction error vs random coverage gaps)
    offs = np.array([o[2] for o in nearest_offsets])
    print()
    print("  Nearest-detection offset distribution (all predicted):")
    print(f"    median={np.median(offs):.1f}\"  "
          f"p10={np.percentile(offs,10):.1f}\"  "
          f"p90={np.percentile(offs,90):.1f}\"  min={offs.min():.1f}\"")
    print("  Brightest 8 predicted asteroids -> nearest real detection:")
    for desig, mag, sep, pra, pdec in sorted(nearest_offsets,
                                                  key=lambda x: x[1])[:8]:
        print(f"    {desig:12s} mag={mag:5.1f} pred=({pra:.4f},{pdec:.4f}) "
              f"nearest={sep:.1f}\"")

    # 4. Report -- overall + by magnitude bin
    print()
    print(f"=== REAL END-TO-END RECOVERY (match radius {args.match_arcsec}\") ===")
    print(f"  predicted in footprint: {len(in_field)}")
    print(f"  recovered:              {len(recovered)} "
          f"({len(recovered)/max(len(in_field),1)*100:.0f}%)")
    print()
    bins = [(0, 18), (18, 19), (19, 20), (20, 21), (21, 22.5)]
    print(f"  {'mag bin':>10s}  {'predicted':>9s}  {'recovered':>9s}  {'recall':>7s}")
    for lo, hi in bins:
        pred_n = sum(1 for r in in_field if lo <= r[3] < hi)
        rec_n = sum(1 for r in recovered if lo <= r[1] < hi)
        if pred_n:
            print(f"  {lo:.0f}-{hi:<4.1f}    {pred_n:9d}  {rec_n:9d}  "
                  f"{rec_n/pred_n*100:6.0f}%")
    print()
    print("  Recovered known asteroids (designation, mag, offset, det_mag):")
    for desig, mag, sep, dmag in sorted(recovered, key=lambda x: x[1])[:25]:
        print(f"    {desig:12s} mag={mag:5.1f}  offset={sep:4.2f}\"  "
              f"det_mag={dmag:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
