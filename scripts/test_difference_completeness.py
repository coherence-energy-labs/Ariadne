"""Does difference imaging improve known-asteroid RECALL on real pixels?

Removing the static star field is the biggest sensitivity boost in
moving-object detection: asteroids sitting on/near stars become visible
and the faint limit improves. We have 4 same-field epochs of (330,-10);
the asteroids are at different positions each epoch, so a MEDIAN of the
other 3 epochs is a star-only reference. Differencing the science frame
against it leaves the science epoch's asteroids (stars removed).

We re-extract on the residual and compare known-asteroid recall to plain
extraction on the same CCDs -- an honest measurement of the completeness
gain.
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
SCI = DATA / "c4d_240824_013234_ooi_r_v1.fits.fz"
REFS = [DATA / "c4d_240809_043854_ooi_r_v1.fits.fz",
        DATA / "c4d_240904_012341_ooi_r_v1.fits.fz",
        DATA / "c4d_240905_020550_ooi_r_v1.fits.fz"]
DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")
N_CCD = 10


def main():
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.difference import _estimate_shift_xc, _shift_image
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    print("loading 4 epochs ...", flush=True)
    sci_inst = load_decam_instcal(str(SCI), read_dqm=False)
    ref_insts = [load_decam_instcal(str(r), read_dqm=False) for r in REFS]
    mjd = sci_inst.mjd
    # index reference CCDs by ccdnum
    ref_by_num = [{c.ccdnum: c for c in ri.ccds if c.wcs is not None}
                  for ri in ref_insts]

    plain_radec = []   # (ra,dec) from plain extraction
    diff_radec = []    # (ra,dec) from difference-image extraction
    ccds = [c for c in sci_inst.ccds if c.wcs is not None and c.magzero > 0][:N_CCD]
    t0 = time.time()
    for ccd in ccds:
        sci = np.asarray(ccd.science, float)
        # build median reference from the 3 other epochs, aligned to science
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
        # normalise + subtract
        scale = np.median(sci) / max(np.median(reference), 1e-6)
        residual = sci - reference * scale

        # plain extraction
        try:
            ps = detect_sources_in_image(sci, ccd.wcs, mjd=mjd,
                                          image_id=f"p{ccd.ccdnum}", fwhm_px=3.5,
                                          threshold_sigma=5.0, zeropoint_mag=ccd.magzero)
            for s in ps:
                plain_radec.append((s.ra, s.dec))
        except Exception:
            pass
        # difference extraction (positives on the residual)
        try:
            ds = detect_sources_in_image(residual, ccd.wcs, mjd=mjd,
                                          image_id=f"d{ccd.ccdnum}", fwhm_px=3.5,
                                          threshold_sigma=5.0, zeropoint_mag=ccd.magzero)
            for s in ds:
                diff_radec.append((s.ra, s.dec))
        except Exception:
            pass
    print(f"  {len(ccds)} CCDs, plain={len(plain_radec)} diff={len(diff_radec)} "
          f"sources ({time.time()-t0:.0f}s)", flush=True)

    # predicted known asteroids on the covered CCDs
    pr = np.array([p[0] for p in plain_radec]); pd = np.array([p[1] for p in plain_radec])
    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (pr.min(), pr.max()), (pd.min(), pd.max()),
                                 max_mag=22.5, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km("807", mjd))
    plain = np.array(plain_radec); diff = np.array(diff_radec)

    def recall(dets, only_covered=True):
        rec = 0; n = 0
        dr = dets[:, 0]; dd = dets[:, 1]
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            cd = math.cos(math.radians(eph[i, 1]))
            # restrict to the covered CCDs' footprint
            sep = np.hypot((dr - eph[i, 0]) * cd, dd - eph[i, 1]) * 3600
            if only_covered and sep.min() > 120:   # not on a covered CCD
                continue
            n += 1
            if sep.min() <= 2.5:
                rec += 1
        return rec, n

    rp, npc = recall(plain)
    rd, ndc = recall(diff)
    print(f"\n=== DIFFERENCE-IMAGING COMPLETENESS (real DECam, {len(ccds)} CCDs) ===")
    print(f"  known asteroids recovered (plain extraction):      {rp}/{npc}"
          f" = {rp/max(npc,1)*100:.0f}%")
    print(f"  known asteroids recovered (difference image):      {rd}/{ndc}"
          f" = {rd/max(ndc,1)*100:.0f}%")
    print(f"  net completeness change: {rp} -> {rd} recoveries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
