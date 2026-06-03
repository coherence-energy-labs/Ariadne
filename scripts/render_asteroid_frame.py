"""Render a real DECam frame to a viewable PNG, marking a recovered asteroid.

Loads the science FITS, finds a KNOWN asteroid that the pipeline recovers
(extracted source within match radius of the N-body-predicted position), then
renders (a) the full CCD it lands on and (b) a zoomed stamp, with the asteroid
circled. Saves PNGs and opens them in the OS image viewer.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sci", default="data/real_decam_recovery/c4d_240824_013234_ooi_r_v1.fits.fz")
    ap.add_argument("--n-ccd", type=int, default=24)
    ap.add_argument("--out", default="data/_viz")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt   # only plt.imsave (no figures/patches -> no py3.14 deepcopy bug)

    def _rgb(a):
        """Asinh-scaled grayscale -> uint8 RGB (shows faint + bright together)."""
        f = a[np.isfinite(a)]
        lo, hi = np.percentile(f, [25.0, 99.6])
        d = np.arcsinh(np.clip((a - lo) / (0.15 * max(hi - lo, 1e-6)), 0, None))
        g = np.clip(d / (d.max() + 1e-9), 0, 1)
        u = (g * 255).astype(np.uint8)
        return np.stack([u, u, u], axis=-1)

    def _ring(rgb, cx, cy, R, w=2, color=(255, 40, 40)):
        H, W, _ = rgb.shape
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(xx - cx, yy - cy)
        rgb[(r >= R - w) & (r <= R + w)] = color

    def _box(rgb, cx, cy, half, w=3, color=(255, 40, 40)):
        H, W, _ = rgb.shape
        x0, x1 = int(cx - half), int(cx + half); y0, y1 = int(cy - half), int(cy + half)
        for yy in (y0, y1):
            rgb[max(0, yy - w):min(H, yy + w), max(0, x0):min(W, x1)] = color
        for xx in (x0, x1):
            rgb[max(0, y0):min(H, y1), max(0, xx - w):min(W, xx + w)] = color

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"loading {Path(args.sci).name} ...", flush=True)
    inst = load_decam_instcal(str(args.sci), read_dqm=False)
    mjd = inst.mjd
    ccds = [c for c in inst.ccds if c.wcs is not None and c.magzero > 0][:args.n_ccd]

    # extract + collect detections per CCD
    det = []  # (ra, dec, x, y, flux, ccd_index)
    for ci, c in enumerate(ccds):
        try:
            srcs = detect_sources_in_image(np.asarray(c.science, float), c.wcs, mjd=mjd,
                                           image_id=f"c{c.ccdnum}", fwhm_px=4.0,
                                           threshold_sigma=5.0, zeropoint_mag=c.magzero)
        except Exception:
            continue
        for s in srcs:
            det.append((s.ra, s.dec, s.x, s.y, s.flux, ci))
    if not det:
        print("no detections"); return 1
    dra = np.array([d[0] for d in det]); ddec = np.array([d[1] for d in det])
    print(f"  {len(ccds)} CCDs, {len(det)} detections; cross-matching knowns ...", flush=True)

    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (dra.min(), dra.max()), (ddec.min(), ddec.max()),
                                 max_mag=22.0, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km("807", mjd))

    # find recovered knowns (extracted source within 2.5") -> pick the brightest
    best = None
    for i in range(len(recs)):
        if np.isnan(eph[i, 0]):
            continue
        cd = math.cos(math.radians(eph[i, 1]))
        sep = np.hypot((dra - eph[i, 0]) * cd, ddec - eph[i, 1]) * 3600
        j = int(np.argmin(sep))
        if sep[j] <= 2.5:
            d = det[j]
            cand = {"desig": str(getattr(recs[i], "designation", "?")),
                    "ra": eph[i, 0], "dec": eph[i, 1], "mag": float(eph[i, 2]),
                    "x": d[2], "y": d[3], "flux": d[4], "ccd": d[5], "sep": float(sep[j])}
            if best is None or cand["flux"] > best["flux"]:
                best = cand
    if best is None:
        print("no known asteroid recovered in the scanned CCDs"); return 1
    print(f"  RECOVERED asteroid {best['desig']}: predicted mag {best['mag']:.1f}, "
          f"detected {best['sep']:.2f}\" from JPL/MPC orbit, on CCD index {best['ccd']}", flush=True)

    ccd = ccds[best["ccd"]]
    img = np.asarray(ccd.science, float)
    img = np.where(np.isfinite(img), img, np.nanmedian(img[np.isfinite(img)]))
    H, W = img.shape
    xi, yi = int(round(best["x"])), int(round(best["y"]))

    # panel 1: full CCD, red box on the asteroid (origin lower -> flipud for display)
    rgb = _rgb(img)
    _box(rgb, xi, yi, 70, w=3)
    p1 = out / "asteroid_ccd.png"
    plt.imsave(str(p1), np.flipud(rgb))

    # panel 2: zoom stamp, red ring on the asteroid
    s = 45
    x0, x1 = max(0, xi - s), min(W, xi + s); y0, y1 = max(0, yi - s), min(H, yi + s)
    stamp = img[y0:y1, x0:x1]
    rgb2 = _rgb(stamp)
    _ring(rgb2, best["x"] - x0, best["y"] - y0, 11, w=2)
    p2 = out / "asteroid_zoom.png"
    plt.imsave(str(p2), np.flipud(rgb2))

    print(f"  wrote {p1}\n  wrote {p2}", flush=True)
    for p in (p2, p1):
        try:
            os.startfile(str(p))   # opens in the Windows default image viewer
        except Exception as e:
            print(f"  (could not auto-open {p.name}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
