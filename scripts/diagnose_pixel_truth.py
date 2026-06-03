"""Per-object PIXEL TRUTH: why is each predicted known asteroid missed?

For every predicted known on the exposure, map its (ra,dec) to the actual
CCD pixel and answer, from the pixels themselves:

  off-chip   : predicted position falls in no CCD (inter-CCD gap / off detector)
  masked     : lands on NaN / bad-pixel-masked pixels (no detection possible)
  flux       : a >5-sigma peak sits within ~4px of the predicted spot (DETECTED)
  no-flux    : on clean pixels but nothing there (genuinely not detected)

Cross-tabulated by phase-CORRECTED predicted magnitude and by predicted TRAIL
length (sky-plane rate * t_exp). The money number: of objects that SHOULD be
trivially detectable -- corrected mag < 21, trail < 5px, on clean chip -- what
fraction actually have flux? That isolates a real detection/ephemeris bug from
"fainter than I labeled" / "trailed" / "in a gap".
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
PIXSCALE = 0.263


def stamp_diag(sci, px, py, half=9, search=4):
    """Return (snr_at_predicted, masked_frac) or (None, 1.0) if off-array."""
    H, W = sci.shape
    xi, yi = int(round(px)), int(round(py))
    if xi < half or yi < half or xi >= W - half or yi >= H - half:
        return None, 1.0
    st = sci[yi - half:yi + half + 1, xi - half:xi + half + 1]
    finite = np.isfinite(st)
    masked_frac = 1.0 - finite.mean()
    if finite.sum() < 0.3 * st.size:
        return None, masked_frac
    bg = float(np.nanmedian(st))
    mad = 1.4826 * float(np.nanmedian(np.abs(st - bg))) + 1e-6
    c = st[half - search:half + search + 1, half - search:half + search + 1]
    peak = float(np.nanmax(c))
    return (peak - bg) / mad, masked_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ccd", type=int, default=60)
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image

    print("loading science exposure ...", flush=True)
    t0 = time.time()
    inst = load_decam_instcal(str(SCI), read_dqm=False)
    mjd = inst.mjd
    texp = float(getattr(inst, "exptime", 0.0) or 90.0)
    ccds = [c for c in inst.ccds if c.wcs is not None][:args.n_ccd]
    print(f"  {len(ccds)} CCDs, t_exp={texp:.0f}s ({time.time()-t0:.0f}s)", flush=True)

    # run the ACTUAL extraction pipeline per CCD -> global (ra,dec) detection list
    ext_ra, ext_dec = [], []
    te = time.time()
    for c in ccds:
        if not (c.magzero and c.magzero > 0):
            continue
        try:
            srcs = detect_sources_in_image(
                np.asarray(c.science, float), c.wcs, mjd=mjd,
                image_id=f"x{c.ccdnum}", fwhm_px=3.5, threshold_sigma=5.0,
                zeropoint_mag=c.magzero)
        except Exception:
            continue
        for s in srcs:
            ext_ra.append(s.ra); ext_dec.append(s.dec)
    ext_ra = np.array(ext_ra); ext_dec = np.array(ext_dec)
    print(f"  extraction: {len(ext_ra)} sources ({time.time()-te:.0f}s)", flush=True)

    # field bbox for catalog pick
    ra_lo = ra_hi = dec_lo = dec_hi = None
    corners_ra, corners_dec = [], []
    for c in ccds:
        H, W = np.asarray(c.science).shape
        for (x, y) in [(0, 0), (W, 0), (0, H), (W, H)]:
            r, d = c.wcs.pixel_to_world_values(x, y)
            corners_ra.append(float(r)); corners_dec.append(float(d))
    ra_lo, ra_hi = min(corners_ra), max(corners_ra)
    dec_lo, dec_hi = min(corners_dec), max(corners_dec)

    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (ra_lo, ra_hi), (dec_lo, dec_hi),
                                 max_mag=24.0, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    obs = observatory_geo_km("807", mjd)
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=obs)
    dt = 0.02
    eph2 = bulk_ephemeris_at_mjd_nbody(recs, mjd + dt,
                                        observer_geo_km=observatory_geo_km("807", mjd + dt))
    print(f"  {len(recs)} predicted knowns in field bbox", flush=True)

    # assign each predicted known to a CCD pixel (vectorised per CCD)
    ra = eph[:, 0]; dec = eph[:, 1]
    N = len(recs)
    on_ccd = np.full(N, -1, int); px = np.full(N, np.nan); py = np.full(N, np.nan)
    for ci, c in enumerate(ccds):
        H, W = np.asarray(c.science).shape
        good = np.isfinite(ra)
        xs, ys = c.wcs.world_to_pixel_values(ra, dec)
        inb = good & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        take = inb & (on_ccd < 0)
        on_ccd[take] = ci; px[take] = xs[take]; py[take] = ys[take]

    # trail (px) from sky-plane rate
    cd = np.cos(np.radians(dec))
    rate_as_hr = np.hypot((eph2[:, 0] - ra) * cd, eph2[:, 1] - dec) * 3600.0 / (dt * 24.0)
    trail_px = rate_as_hr * (texp / 3600.0) / PIXSCALE

    cats = {"off-chip": 0, "masked": 0, "flux": 0, "no-flux": 0}
    # money sample: corrected mag<21, trail<5px, on clean chip
    money_total = money_flux = money_ext = 0
    # mag-binned: clean-chip count, flux-present, AND pipeline-extracted-recovered
    magbins = [(0, 20), (20, 21), (21, 22), (22, 99)]
    perbin = {b: {"clean": 0, "flux": 0, "ext": 0} for b in magbins}
    sci_cache = {ci: np.asarray(ccds[ci].science, float) for ci in range(len(ccds))}
    for i in range(N):
        if np.isnan(ra[i]) or on_ccd[i] < 0:
            cats["off-chip"] += 1
            continue
        snr, mfrac = stamp_diag(sci_cache[on_ccd[i]], px[i], py[i])
        if snr is None:
            cats["masked"] += 1
            continue
        if mfrac > 0.5:
            cats["masked"] += 1
            continue
        detected = snr > 5.0
        cats["flux" if detected else "no-flux"] += 1
        # was this object recovered by the ACTUAL extraction pipeline (<2.5")?
        ext_ok = False
        if ext_ra.size:
            cd_i = math.cos(math.radians(dec[i]))
            esep = np.hypot((ext_ra - ra[i]) * cd_i, ext_dec - dec[i]) * 3600
            ext_ok = bool(esep.min() <= 2.5)
        m = float(eph[i, 2])
        if not np.isnan(m):
            for b in magbins:
                if b[0] <= m < b[1]:
                    perbin[b]["clean"] += 1
                    if detected:
                        perbin[b]["flux"] += 1
                    if ext_ok:
                        perbin[b]["ext"] += 1
                    break
        if (not np.isnan(m)) and m < 21.0 and trail_px[i] < 5.0:
            money_total += 1
            if detected:
                money_flux += 1
            if ext_ok:
                money_ext += 1

    print(f"\n=== PIXEL-TRUTH for {N} predicted knowns ({len(ccds)} CCDs) ===")
    for k, v in cats.items():
        print(f"  {k:<10}: {v:4d}  ({v/max(N,1)*100:4.0f}%)")
    print(f"\n  on clean chip, by phase-CORRECTED mag:  flux-present  vs  pipeline-extracted(<2.5\")")
    for b in magbins:
        cl = perbin[b]["clean"]; fl = perbin[b]["flux"]; ex = perbin[b]["ext"]
        if cl:
            print(f"    {b[0]:2d}-{b[1]:<2d}: flux {fl:3d}/{cl:<3d}={fl/cl*100:3.0f}%   "
                  f"extracted {ex:3d}/{cl:<3d}={ex/cl*100:3.0f}%")
    print(f"\n  MONEY (corrected-mag<21, trail<5px, clean-chip, n={money_total}): "
          f"flux-present {money_flux/max(money_total,1)*100:.0f}%  "
          f"pipeline-extracted {money_ext/max(money_total,1)*100:.0f}%")
    print(f"  trail stats (all): median {np.nanmedian(trail_px):.1f}px "
          f"90th pct {np.nanpercentile(trail_px,90):.1f}px "
          f"(>5px = trail-limited)")
    print(f"  corrected-mag stats: median {np.nanmedian(eph[:,2]):.1f} "
          f"min {np.nanmin(eph[:,2]):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
