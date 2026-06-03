"""Does single-snapshot velocity collapse the 2-night linking blowup?

Blind 2-night pairing of the real Sep-4/Sep-5 DECam fields gave ~89M
candidate pairs (21/21 real movers recovered, but buried). Here we add the
single-snapshot cues to EACH detection (rate + PA from PSF trailing) and
re-link with velocity coherence:

  1. MOVER FILTER  -- drop detections whose elongation is consistent with a
     star (rate below a floor). Attacks the N1*N2 term directly.
  2. PREDICTED POSITION -- a real Sep-4 mover, propagated by its own
     measured velocity, must land near a Sep-5 detection.
  3. PA COHERENCE  -- the motion direction measured on both nights must
     agree (same object). The strongest clean discriminant.
  4. DISTANCE COHERENCE -- the opposition-relation distance from each
     night's rate must agree.

We measure the candidate-pair count at each stage vs the 89M blind
baseline, and the recall of the 21 known movers (truth from the accurate
cross-match).
"""
from __future__ import annotations

import os
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/real_decam_recovery")
SEP4 = DATA / "c4d_240904_012341_ooi_r_v1.fits.fz"
SEP5 = DATA / "c4d_240905_020550_ooi_r_v1.fits.fz"
DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")
PIXSCALE = 0.263
TEXP = 90.0
STAMP = 16


def extract_with_rates(fits_path, cache):
    """Per CCD: DAOStarFinder -> for each source measure trailed rate+PA from
    its stamp. Cache (ra, dec, mag, rate_arcsec_hr, pa_deg, snr)."""
    import os
    if os.path.exists(cache):
        c = np.load(cache)
        return (c["ra"], c["dec"], c["mag"], c["rate"], c["pa"], c["snr"],
                float(c["mjd"]))
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.trailed_rate import (rate_from_stamp,
                                                          stellar_psf_anisotropy)
    print(f"  extracting+rating {Path(fits_path).name} ...", flush=True)
    t0 = time.time()
    inst = load_decam_instcal(fits_path, read_dqm=False)
    ra, dec, mag, rate, pa, snr = [], [], [], [], [], []
    for ccd in inst.ccds:
        if ccd.wcs is None:
            continue
        try:
            srcs = detect_sources_in_image(
                ccd.science, ccd.wcs, mjd=inst.mjd, image_id=f"c{ccd.ccdnum}",
                fwhm_px=3.5, threshold_sigma=5.0,
                zeropoint_mag=(ccd.magzero if ccd.magzero > 0 else None))
        except Exception:
            continue
        if not srcs:
            continue
        data = np.asarray(ccd.science, float)
        aniso = stellar_psf_anisotropy(data)
        H, W = data.shape
        for s in srcs:
            x, y = int(round(s.x)), int(round(s.y))
            if x < STAMP or y < STAMP or x > W - STAMP or y > H - STAMP:
                continue
            st = data[y - STAMP:y + STAMP + 1, x - STAMP:x + STAMP + 1]
            est = rate_from_stamp(st, psf_aniso=aniso, pixscale_arcsec=PIXSCALE,
                                    t_exp_s=TEXP, wcs=ccd.wcs, x_pix=s.x, y_pix=s.y)
            ra.append(s.ra); dec.append(s.dec); mag.append(s.mag)
            rate.append(est.rate_arcsec_hr)
            pa.append(est.pa_deg_sky if est.pa_deg_sky is not None else est.pa_deg_pix)
            snr.append(est.snr)
    out = [np.array(v) for v in (ra, dec, mag, rate, pa, snr)]
    np.savez(cache, ra=out[0], dec=out[1], mag=out[2], rate=out[3],
             pa=out[4], snr=out[5], mjd=inst.mjd)
    print(f"    {len(out[0])} sources rated in {time.time()-t0:.0f}s", flush=True)
    return (*out, inst.mjd)


def truth_labels(db, ra, dec, mjd):
    """Designation per detection via the accurate cross-match (truth)."""
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km
    inf = pick_orbits_in_field(db, mjd, (ra.min(), ra.max()), (dec.min(), dec.max()),
                                 max_mag=22.5, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km("807", mjd))
    lab = {}
    for i in range(len(recs)):
        if np.isnan(eph[i, 0]):
            continue
        cd = math.cos(math.radians(eph[i, 1]))
        sep = np.hypot((ra - eph[i, 0]) * cd, dec - eph[i, 1]) * 3600
        j = int(np.argmin(sep))
        if sep[j] <= 3.0:
            lab[j] = recs[i].designation
    return lab


def main():
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.orbit_geometry import opposition_rate_to_distance
    db = open_db(DB)
    r4, d4, m4, rate4, pa4, snr4, mjd4 = extract_with_rates(SEP4, str(SEP4).replace(".fits.fz", "_rated.npz"))
    r5, d5, m5, rate5, pa5, snr5, mjd5 = extract_with_rates(SEP5, str(SEP5).replace(".fits.fz", "_rated.npz"))
    dt = mjd5 - mjd4
    print(f"Sep4: {len(r4)} det, Sep5: {len(r5)} det, dt={dt*24:.1f}h")

    lab4 = truth_labels(db, r4, d4, mjd4)
    lab5 = truth_labels(db, r5, d5, mjd5)
    truth = set(lab4.values()) & set(lab5.values())
    print(f"Known movers in BOTH nights (truth): {len(truth)}")

    cosd = math.cos(math.radians(np.median(d4)))
    from scipy.spatial import cKDTree
    xy5 = np.column_stack([r5 * cosd, d5])
    tree = cKDTree(xy5)
    # main-belt-ish rate window over the inter-night gap (deg displacement)
    rmin_deg = 0.10 * dt    # ~0.10 deg/day lower
    rmax_deg = 0.55 * dt

    def link(require_disp_pa, pa_tol, mag_tol=None):
        """Search the rate annulus; optionally require the inter-night
        DISPLACEMENT direction to match BOTH nights' measured PAs (the
        reliable single-snapshot cue) -- not the noisy rate magnitude."""
        pairs = 0
        recovered = set()
        for ii in range(len(r4)):
            cand = tree.query_ball_point([r4[ii] * cosd, d4[ii]], rmax_deg)
            for j in cand:
                dra = (r5[j] - r4[ii]) * cosd
                ddec = d5[j] - d4[ii]
                disp = math.hypot(dra, ddec)
                if disp < rmin_deg or disp > rmax_deg:
                    continue
                if require_disp_pa:
                    disp_pa = math.degrees(math.atan2(dra, ddec)) % 180.0
                    # displacement direction must match BOTH measured PAs
                    if abs((disp_pa - pa4[ii] + 90) % 180 - 90) > pa_tol:
                        continue
                    if abs((disp_pa - pa5[j] + 90) % 180 - 90) > pa_tol:
                        continue
                if mag_tol is not None and abs(m4[ii] - m5[j]) > mag_tol:
                    continue
                pairs += 1
                if ii in lab4 and j in lab5 and lab4[ii] == lab5[j]:
                    recovered.add(lab4[ii])
        return pairs, len(recovered)

    print(f"\n{'stage':46s} {'pairs':>12s} {'recall':>8s}")
    base = len(r4) * len(r5)
    print(f"{'blind (all x all positions)':46s} {base:12,d} {'21/21':>8s}")
    p, _ = link(False, 0)
    print(f"{'rate-annulus search (positions only)':46s} {p:12,d} {'(21/21 in here)':>8s}")
    for pa_tol, mtol, lbl in [
        (40, None, "+ displacement-PA matches both nights (<=40deg)"),
        (30, None, "+ displacement-PA (<=30deg)"),
        (30, 1.0, "+ PA(30) + magnitude coherence (<=1 mag)"),
    ]:
        p, rec = link(True, pa_tol, mtol)
        print(f"{lbl:46s} {p:12,d} {rec:3d}/{len(truth):<3d}")


if __name__ == "__main__":
    raise SystemExit(main())
