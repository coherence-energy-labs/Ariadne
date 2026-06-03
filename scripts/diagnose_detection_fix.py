"""Which detection change recovers the asteroids whose flux is in the pixels?

The reconciliation showed 81%% of detectable known asteroids have flux at the
predicted spot but only 31%% are extracted. Two suspected defects in
detect_sources_in_image:
  (1) a broken post-filter `sharpness * fwhm_px in [1,8]` (sharpness is NOT a
      FWHM -> silently drops real sources);
  (2) a fixed fwhm_px=3.5 detection kernel instead of the image's measured
      seeing (mismatched matched-filter loses faint sources).

This measures money-sample recall (corrected mag<21, on clean chip) under:
  V0  current     : fwhm=3.5, 5sigma, broken sharpness*fwhm filter
  V1  measured    : measured FWHM, 5sigma, DAO default cuts, NO proxy filter
  V2  trail-tol   : V1 + roundhi relaxed (allow mild trailing)
Whichever closes the gap to the ~81%% flux-present ceiling is the fix.
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
    ap.add_argument("--n-ccd", type=int, default=60)
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km
    from ariadne.discovery.imaging.trailed_rate import stellar_psf_fwhm
    from photutils.background import Background2D, MedianBackground
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    print("loading exposure ...", flush=True)
    t0 = time.time()
    inst = load_decam_instcal(str(SCI), read_dqm=False)
    mjd = inst.mjd
    ccds = [c for c in inst.ccds if c.wcs is not None and c.magzero > 0][:args.n_ccd]
    print(f"  {len(ccds)} CCDs ({time.time()-t0:.0f}s)", flush=True)

    # predicted knowns
    corners_ra, corners_dec = [], []
    for c in ccds:
        H, W = np.asarray(c.science).shape
        for (x, y) in [(0, 0), (W, 0), (0, H), (W, H)]:
            r, d = c.wcs.pixel_to_world_values(x, y)
            corners_ra.append(float(r)); corners_dec.append(float(d))
    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (min(corners_ra), max(corners_ra)),
                                 (min(corners_dec), max(corners_dec)),
                                 max_mag=24.0, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km("807", mjd))
    ra = eph[:, 0]; dec = eph[:, 1]; mag = eph[:, 2]

    # money sample: corrected mag<21, assign to CCD
    money = []   # (ccd_index, ra, dec)
    for ci, c in enumerate(ccds):
        H, W = np.asarray(c.science).shape
        good = np.isfinite(ra) & np.isfinite(mag) & (mag < 21.0)
        xs, ys = c.wcs.world_to_pixel_values(ra, dec)
        inb = good & (xs >= 20) & (xs < W - 20) & (ys >= 20) & (ys < H - 20)
        for i in np.where(inb)[0]:
            money.append((ci, ra[i], dec[i]))
    ccds_with = sorted({m[0] for m in money})
    print(f"  money sample: {len(money)} knowns (mag<21) on {len(ccds_with)} CCDs",
          flush=True)

    def recall(positions_by_ccd):
        rec = 0
        for ci, mra, mdec in money:
            P = positions_by_ccd.get(ci)
            if P is None or len(P) == 0:
                continue
            pr, pd = P[:, 0], P[:, 1]
            cd = math.cos(math.radians(mdec))
            sep = np.hypot((pr - mra) * cd, pd - mdec) * 3600
            if sep.size and sep.min() <= 2.5:
                rec += 1
        return rec

    pos = {"V0": {}, "V1": {}, "V2": {}}
    fwhms = []
    te = time.time()
    for ci in ccds_with:
        c = ccds[ci]
        data = np.asarray(c.science, float)
        try:
            bkg = Background2D(data, box_size=(64, 64), bkg_estimator=MedianBackground())
            sub = data - bkg.background
        except Exception:
            sub = data - np.nanmedian(data)
        _m, _md, std = sigma_clipped_stats(sub, sigma=3.0)
        if not np.isfinite(std) or std <= 0:
            continue
        meas_fwhm = stellar_psf_fwhm(data, fwhm_px=4.0)
        fwhms.append(meas_fwhm)

        def to_radec(tbl):
            if tbl is None or len(tbl) == 0:
                return np.empty((0, 2))
            xc = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
            yc = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
            rr, dd = c.wcs.pixel_to_world_values(np.array(tbl[xc]), np.array(tbl[yc]))
            return np.column_stack([np.atleast_1d(rr) % 360.0, np.atleast_1d(dd)])

        # V0: current behaviour (fwhm 3.5, 5 sigma, broken sharpness*fwhm filter)
        t0v = DAOStarFinder(fwhm=3.5, threshold=5.0 * std)(sub)
        if t0v is not None and len(t0v):
            keep = [(1.0 <= float(r["sharpness"]) * 3.5 <= 8.0) for r in t0v]
            t0v = t0v[np.array(keep, bool)] if len(keep) else t0v
        pos["V0"][ci] = to_radec(t0v)

        # V1: measured FWHM, 5 sigma, DAO default cuts, no proxy filter
        t1 = DAOStarFinder(fwhm=max(meas_fwhm, 2.0), threshold=5.0 * std)(sub)
        pos["V1"][ci] = to_radec(t1)

        # V2: V1 + relaxed roundness (allow mild trailing)
        t2 = DAOStarFinder(fwhm=max(meas_fwhm, 2.0), threshold=5.0 * std,
                            roundlo=-1.5, roundhi=1.5, sharplo=0.1)(sub)
        pos["V2"][ci] = to_radec(t2)

    print(f"  detection variants over {len(ccds_with)} CCDs ({time.time()-te:.0f}s)",
          flush=True)
    print(f"  measured stellar FWHM: median {np.median(fwhms):.1f}px "
          f"(detection assumed 3.5px)")
    n = len(money)
    for v in ["V0", "V1", "V2"]:
        r = recall(pos[v])
        print(f"  {v}: money-sample recall {r}/{n} = {r/max(n,1)*100:.0f}%")
    print(f"  (flux-present ceiling was ~81%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
