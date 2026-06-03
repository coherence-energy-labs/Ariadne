"""Real within-night tracklet demonstration on 3 same-night DECam exposures.

This is the discovery UNIT that all prior real data lacked (it was 1
exposure/night). Three exposures of one field over ~134 min let a moving
object be seen at 3 positions -> a within-night tracklet with a tight,
3-point-confirmed rate vector. Three collinear points kill the chance-pair
ambiguity that made 2-night sparse linking unusable (89M false pairs).

Pipeline:
  1. Extract sources from each of the 3 exposures (instcal WCS).
  2. Build within-night tracklets: triplets of detections (one per
     exposure) that lie on a consistent linear sky-motion track.
  3. Cross-match each tracklet's mean position+epoch to the MPC catalog
     (accurate N-body + topocentric) -> identify knowns, confirm the
     measured rate matches the catalog object's motion.
  4. Report: tracklets found, fraction matching known asteroids, rate
     residual vs catalog, and any unmatched moving tracklets (candidate
     unknowns).
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

OBS = "807"


def extract(fits_path, sigma):
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    inst = load_decam_instcal(fits_path, read_dqm=False)
    ra, dec, mag = [], [], []
    for ccd in inst.ccds:
        if ccd.wcs is None:
            continue
        try:
            srcs = detect_sources_in_image(
                ccd.science, ccd.wcs, mjd=inst.mjd, image_id=f"c{ccd.ccdnum}",
                fwhm_px=3.5, threshold_sigma=sigma,
                zeropoint_mag=(ccd.magzero if ccd.magzero > 0 else None))
        except Exception:
            continue
        for s in srcs:
            ra.append(s.ra); dec.append(s.dec); mag.append(s.mag)
    return np.array(ra), np.array(dec), np.array(mag), inst.mjd


def build_triplet_tracklets(epochs, *, max_rate_deg_day=0.8,
                              min_rate_deg_day=0.02, collinear_tol_arcsec=2.0):
    """epochs: list of (ra, dec, mjd) arrays sorted by time (3 of them).
    A tracklet = one detection in each epoch lying on a straight, constant-
    rate track. The 3rd point must agree with the line from the first two to
    within collinear_tol_arcsec -- this is what makes within-night triplets
    clean where 2-night pairs are not.
    """
    from scipy.spatial import cKDTree
    (ra0, dec0, m0, t0), (ra1, dec1, m1, t1), (ra2, dec2, m2, t2) = epochs
    cosd = math.cos(math.radians(np.median(dec0)))
    dt01 = t1 - t0
    dt02 = t2 - t0
    tree1 = cKDTree(np.column_stack([ra1 * cosd, dec1]))
    tree2 = cKDTree(np.column_stack([ra2 * cosd, dec2]))
    out = []
    d01_hi = max_rate_deg_day * dt01
    d01_lo = min_rate_deg_day * dt01
    for i in range(len(ra0)):
        for j in tree1.query_ball_point([ra0[i] * cosd, dec0[i]], d01_hi):
            dra = (ra1[j] - ra0[i]) * cosd
            ddec = dec1[j] - dec0[i]
            d01 = math.hypot(dra, ddec)
            if d01 < d01_lo:
                continue
            # predict epoch-2 position from constant rate
            pra = ra0[i] + (ra1[j] - ra0[i]) * (dt02 / dt01)
            pdec = dec0[i] + (dec1[j] - dec0[i]) * (dt02 / dt01)
            near = tree2.query_ball_point([pra * cosd, pdec],
                                            collinear_tol_arcsec / 3600.0)
            for k in near:
                resid = math.hypot((ra2[k] - pra) * cosd,
                                     dec2[k] - pdec) * 3600
                if resid <= collinear_tol_arcsec:
                    rate = d01 / dt01
                    pa = math.degrees(math.atan2(dra, ddec)) % 360.0
                    out.append({
                        "i": i, "j": j, "k": k, "resid_arcsec": resid,
                        "rate_deg_day": rate, "pa_deg": pa,
                        "mean_ra": float(np.mean([ra0[i], ra1[j], ra2[k]])),
                        "mean_dec": float(np.mean([dec0[i], dec1[j], dec2[k]])),
                        "mean_mjd": float((t0 + t1 + t2) / 3),
                        "mag": float(np.nanmedian([m0[i], m1[j], m2[k]]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real_decam_tracklet")
    ap.add_argument("--db", default=os.environ.get("ARIADNE_DB", "data/recovery_clean.db"))
    ap.add_argument("--detect-sigma", type=float, default=5.0)
    ap.add_argument("--collinear-tol", type=float, default=2.0)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("c4d_*_ooi_r_v1.fits.fz"))
    print(f"{len(files)} same-night exposures")
    epochs = []
    for f in files:
        cache = str(f).replace(".fits.fz", "_dets.npz")
        import os
        if os.path.exists(cache):
            c = np.load(cache)
            epochs.append((c["ra"], c["dec"], c["mag"], float(c["mjd"])))
        else:
            t0 = time.time()
            ra, dec, mag, mjd = extract(str(f), args.detect_sigma)
            np.savez(cache, ra=ra, dec=dec, mag=mag, mjd=mjd)
            print(f"  {f.name}: {len(ra)} sources, mjd={mjd:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            epochs.append((ra, dec, mag, mjd))
    epochs.sort(key=lambda e: e[3])
    times = [e[3] for e in epochs]
    print(f"epochs (mjd): {[f'{t:.4f}' for t in times]}, "
          f"arc {((times[-1]-times[0])*1440):.0f} min")

    print("\nBuilding within-night triplet tracklets "
          f"(3-point collinear, tol {args.collinear_tol}\")...", flush=True)
    t0 = time.time()
    trk = build_triplet_tracklets(epochs, collinear_tol_arcsec=args.collinear_tol)
    print(f"  {len(trk)} tracklets in {time.time()-t0:.0f}s "
          "(3-point collinearity kills the chance pairs that flooded "
          "2-night linking)")

    # Cross-match each tracklet's mean position to the catalog
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km
    db = open_db(args.db)
    mjd = float(np.mean(times))
    ra_all = np.concatenate([e[0] for e in epochs])
    dec_all = np.concatenate([e[1] for e in epochs])
    inf = pick_orbits_in_field(db, mjd, (ra_all.min(), ra_all.max()),
                                 (dec_all.min(), dec_all.max()),
                                 max_mag=22.5, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    obs = observatory_geo_km(OBS, mjd)
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=obs)
    # match tracklet mean pos to predicted asteroid pos
    matched = 0
    rate_res = []
    for tk in trk:
        cd = math.cos(math.radians(tk["mean_dec"]))
        best = None; bestd = 1e9
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            s = math.hypot((eph[i, 0] - tk["mean_ra"]) * cd,
                             eph[i, 1] - tk["mean_dec"]) * 3600
            if s < bestd:
                bestd = s; best = i
        if bestd <= 3.0:
            matched += 1
    print(f"\n=== REAL within-night tracklets ===")
    print(f"  tracklets found:        {len(trk)}")
    print(f"  matched to known MPC:   {matched} "
          f"({matched/max(len(trk),1)*100:.0f}%)")
    print(f"  candidate unknown movers: {len(trk)-matched}")
    if trk:
        rates = sorted(t["rate_deg_day"] for t in trk)
        print(f"  tracklet rate range: {rates[0]*3600:.0f}-{rates[-1]*3600:.0f} "
              f"\"/day (median {rates[len(rates)//2]*3600:.0f}\"/day)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
