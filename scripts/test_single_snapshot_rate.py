"""MAKE-OR-BREAK: can we recover an asteroid's RATE from a SINGLE exposure?

A point source dragged at angular rate w over exposure time T_exp leaves a
trail of length L = w * T_exp. Convolved with the seeing PSF, the source's
second-moment tensor gains variance L^2/12 ALONG the motion while the
PERPENDICULAR axis stays at the PSF width. So:

    a^2 - b^2 = L^2 / 12         (a,b = major/minor RMS, pixels)
    L_px      = sqrt(12 (a^2 - b^2))
    rate      = L_px * pixscale / T_exp
    PA_motion = orientation of the major axis (mod 180 deg)

This is self-calibrating: the minor axis IS the local PSF, so no separate
baseline is needed (we still measure stars to confirm the instrumental PSF
is round). We test it on the KNOWN asteroids in a real 90 s DECam exposure,
comparing the elongation-implied rate/PA to the catalog (N-body) truth.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FITS = "data/real_decam_recovery/c4d_240824_013234_ooi_r_v1.fits.fz"
DB = "C:/Users/Josh/AppData/Local/Temp/recovery_clean.db"
OBS = "807"
PIXSCALE = 0.263
STAMP = 15  # half-size


def _measure(stamp):
    """Flux-weighted centroid + major/minor RMS (px) + PA (rad) of a stamp."""
    from ariadne.discovery.imaging.morphology import _second_moments, _theta_from_moments
    bg = float(np.median(stamp))
    sub = np.clip(stamp - bg, 0, None)
    if sub.sum() <= 0:
        return None
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    xc = float((sub * xx).sum() / sub.sum())
    yc = float((sub * yy).sum() / sub.sum())
    a, b = _second_moments(stamp, xc, yc, bg)
    th = _theta_from_moments(stamp, xc, yc, bg)
    snr = float(sub.max() / (np.std(stamp[stamp < np.percentile(stamp, 84)]) + 1e-9))
    return dict(a=a, b=b, theta=th, snr=snr, peak=float(sub.max()))


def main():
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    print(f"Loading {Path(FITS).name} ...", flush=True)
    inst = load_decam_instcal(FITS, read_dqm=False)
    mjd = inst.mjd
    texp = inst.exptime_s
    print(f"  mjd={mjd:.4f}, exptime={texp:.0f}s")

    # Known asteroids in field + catalog rate/PA via finite-difference ephemeris
    ccds = [c for c in inst.ccds if c.wcs is not None]
    # field bbox from CCD centers
    ras, decs = [], []
    for c in ccds:
        ny, nx = c.science.shape
        w = c.wcs.pixel_to_world_values(nx / 2, ny / 2)
        ras.append(float(w[0])); decs.append(float(w[1]))
    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (min(ras) - 0.2, max(ras) + 0.2),
                                 (min(decs) - 0.2, max(decs) + 0.2),
                                 max_mag=21.0, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    obs = observatory_geo_km(OBS, mjd)
    dt = 0.02
    e0 = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=obs)
    e1 = bulk_ephemeris_at_mjd_nbody(recs, mjd + dt, observer_geo_km=obs)

    # Instrumental PSF check: measure bright round sources (stars) on one CCD
    star_ab = []
    cc = ccds[len(ccds) // 2]
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    data = np.asarray(cc.science, float)
    _m, _md, std = sigma_clipped_stats(data, sigma=3.0)
    tbl = DAOStarFinder(fwhm=4.0, threshold=30 * std)(data - _md)
    if tbl is not None:
        xcol = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
        ycol = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
        for row in list(tbl)[:60]:
            x, y = int(row[xcol]), int(row[ycol])
            if x < STAMP or y < STAMP or x > data.shape[1] - STAMP or y > data.shape[0] - STAMP:
                continue
            st = data[y - STAMP:y + STAMP + 1, x - STAMP:x + STAMP + 1]
            m = _measure(st)
            if m and m["snr"] > 20:
                star_ab.append((m["a"], m["b"]))
    if star_ab:
        sa = np.array(star_ab)
        print(f"  stellar PSF: major={np.median(sa[:,0]):.2f}px minor={np.median(sa[:,1]):.2f}px "
              f"(instrumental elongation a-b={np.median(sa[:,0]-sa[:,1]):.2f}px; should be ~0)")
        psf_floor = np.median(sa[:, 0] ** 2 - sa[:, 1] ** 2)  # instrumental a^2-b^2
    else:
        psf_floor = 0.0

    # For each known asteroid: locate CCD+pixel, measure elongation, compare
    print(f"\n  {'desig':10s} {'V':>5s} {'cat_rate':>9s} {'meas_rate':>9s} "
          f"{'cat_PA':>7s} {'meas_PA':>7s} {'a/b':>5s} {'snr':>5s}")
    print("  " + "-" * 66)
    results = []
    for i, rec in enumerate(recs):
        if np.isnan(e0[i, 0]):
            continue
        ra, dec, vmag = e0[i, 0], e0[i, 1], e0[i, 2]
        if vmag > 20.0:
            continue
        cd = math.cos(math.radians(dec))
        # catalog on-sky rate (arcsec/hr) + PA
        dra = (e1[i, 0] - ra) * cd
        ddec = e1[i, 1] - dec
        cat_rate = math.hypot(dra, ddec) * 3600 / (dt * 24)  # "/hr
        cat_pa = math.degrees(math.atan2(dra, ddec)) % 180.0
        # find CCD containing it
        for c in ccds:
            ny, nx = c.science.shape
            try:
                px, py = c.wcs.world_to_pixel_values(ra, dec)
            except Exception:
                continue
            px = float(px); py = float(py)
            if STAMP <= px < nx - STAMP and STAMP <= py < ny - STAMP:
                st = np.asarray(c.science, float)[int(py) - STAMP:int(py) + STAMP + 1,
                                                    int(px) - STAMP:int(px) + STAMP + 1]
                m = _measure(st)
                if m is None or m["snr"] < 8:
                    break
                # trail length from EXCESS elongation over instrumental PSF
                excess = (m["a"] ** 2 - m["b"] ** 2) - psf_floor
                L_px = math.sqrt(12 * excess) if excess > 0 else 0.0
                meas_rate = L_px * PIXSCALE / texp * 3600  # "/hr
                meas_pa = (math.degrees(m["theta"])) % 180.0
                results.append((rec.designation, vmag, cat_rate, meas_rate,
                                  cat_pa, meas_pa, m["a"] / max(m["b"], 1e-6),
                                  m["snr"]))
                break
    results.sort(key=lambda r: -r[2])  # by catalog rate desc
    for d, v, cr, mr, cpa, mpa, ab, snr in results:
        dpa = abs((mpa - cpa + 90) % 180 - 90)
        print(f"  {d:10s} {v:5.1f} {cr:8.0f}\" {mr:8.0f}\" {cpa:6.0f} {mpa:6.0f} "
              f"{ab:5.2f} {snr:5.0f}  dPA={dpa:.0f}")

    # Correlation: does measured rate track catalog rate?
    if len(results) >= 5:
        cr = np.array([r[2] for r in results])
        mr = np.array([r[3] for r in results])
        from scipy.stats import spearmanr, pearsonr
        rho, p = spearmanr(cr, mr)
        print(f"\n  Spearman(cat_rate, meas_rate) = {rho:.2f} (p={p:.3f}), n={len(results)}")
        print(f"  => {'SIGNAL: elongation tracks rate' if rho > 0.4 and p < 0.05 else 'WEAK/NO signal at this depth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
