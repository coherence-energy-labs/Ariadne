"""Does PSF-MATCHED (Alard-Lupton) differencing improve real-data recall?

The honest follow-up to test_difference_completeness.py, which showed CRUDE
scalar differencing did NOT help (plain 4/30 -> diff 3/27) because mismatched
seeing leaves dipole residuals at every star. This compares three extractions
on the SAME real DECam CCDs, cross-matched to known asteroids:

  plain        : extract on the science frame (no subtraction)
  crude-diff   : extract on science - scalar*median_reference
  psf-diff     : extract on science - (median_reference (*) matching_kernel)

A median of the 3 OTHER same-field epochs is the star-only reference (movers
sit at different positions each epoch, so the median rejects them). Recall is
counted against JPL/MPC-predicted known asteroids on the covered footprint.
This tells us, on real pixels, whether the proper kernel buys completeness.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/real_decam_recovery")
SCI = DATA / "c4d_240824_013234_ooi_r_v1.fits.fz"
REFS = [DATA / "c4d_240809_043854_ooi_r_v1.fits.fz",
        DATA / "c4d_240904_012341_ooi_r_v1.fits.fz",
        DATA / "c4d_240905_020550_ooi_r_v1.fits.fz"]
DB = "C:/Users/Josh/AppData/Local/Temp/recovery_clean.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ccd", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=5.0)
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.difference import (
        _estimate_shift_xc, _shift_image, subtract_reference,
        psf_matched_difference)
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    print("loading 4 epochs ...", flush=True)
    sci_inst = load_decam_instcal(str(SCI), read_dqm=False)
    ref_insts = [load_decam_instcal(str(r), read_dqm=False) for r in REFS]
    mjd = sci_inst.mjd
    ref_by_num = [{c.ccdnum: c for c in ri.ccds if c.wcs is not None}
                  for ri in ref_insts]

    plain_radec, crude_radec, psf_radec = [], [], []
    ccds = [c for c in sci_inst.ccds if c.wcs is not None and c.magzero > 0][:args.n_ccd]
    n_psf, n_scalar = 0, 0
    t0 = time.time()
    for ccd in ccds:
        sci = np.asarray(ccd.science, float)
        aligned = []
        for rb in ref_by_num:
            rc = rb.get(ccd.ccdnum)
            if rc is None:
                continue
            ref = np.asarray(rc.science, float)
            if ref.shape != sci.shape:
                continue
            dx, dy = _estimate_shift_xc(sci, ref, max_shift_px=80)
            aligned.append(_shift_image(ref, dx, dy))
        if len(aligned) < 2:
            continue
        reference = np.median(np.stack(aligned), axis=0)

        def extract(image, tag):
            try:
                return detect_sources_in_image(
                    image, ccd.wcs, mjd=mjd, image_id=f"{tag}{ccd.ccdnum}",
                    fwhm_px=3.5, threshold_sigma=args.sigma,
                    zeropoint_mag=ccd.magzero)
            except Exception:
                return []

        for s in extract(sci, "p"):
            plain_radec.append((s.ra, s.dec))
        # crude scalar difference
        cr = subtract_reference(sci, reference, max_shift_px=4)
        for s in extract(cr.residual, "c"):
            crude_radec.append((s.ra, s.dec))
        # PSF-matched difference
        pm = psf_matched_difference(sci, reference, max_shift_px=4)
        if pm.method == "psf-matched":
            n_psf += 1
        else:
            n_scalar += 1
        for s in extract(pm.residual, "m"):
            psf_radec.append((s.ra, s.dec))

    print(f"  {len(ccds)} CCDs, plain={len(plain_radec)} crude={len(crude_radec)} "
          f"psf={len(psf_radec)} sources ({time.time()-t0:.0f}s); "
          f"psf-matched on {n_psf} CCDs, scalar-fallback on {n_scalar}", flush=True)

    pr = np.array([p[0] for p in plain_radec]); pd = np.array([p[1] for p in plain_radec])
    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (pr.min(), pr.max()), (pd.min(), pd.max()),
                                 max_mag=22.5, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km("807", mjd))

    def recall(radec):
        dets = np.array(radec)
        dr = dets[:, 0]; dd = dets[:, 1]
        rec = 0; n = 0
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            cd = math.cos(math.radians(eph[i, 1]))
            sep = np.hypot((dr - eph[i, 0]) * cd, dd - eph[i, 1]) * 3600
            if sep.min() > 120:        # not on a covered CCD
                continue
            n += 1
            if sep.min() <= 2.5:
                rec += 1
        return rec, n

    rp, npc = recall(plain_radec)
    rc, ncc = recall(crude_radec)
    rm, nmc = recall(psf_radec)
    print(f"\n=== DIFFERENCE-IMAGING RECALL (real DECam, {len(ccds)} CCDs) ===")
    print(f"  plain extraction       : {rp}/{npc} = {rp/max(npc,1)*100:.0f}%"
          f"   ({len(plain_radec)} sources)")
    print(f"  crude scalar difference: {rc}/{ncc} = {rc/max(ncc,1)*100:.0f}%"
          f"   ({len(crude_radec)} sources)")
    print(f"  PSF-matched difference : {rm}/{nmc} = {rm/max(nmc,1)*100:.0f}%"
          f"   ({len(psf_radec)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
