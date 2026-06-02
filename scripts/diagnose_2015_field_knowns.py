"""Are there matchable KNOWN asteroids in the 2015 discovery field?

The end-to-end benchmark found 0 recoverable knowns. Before blaming the linker,
check the TRUTH layer: at MJD ~57130 (2015) -- ~10 yr from the catalog's
osculating epoch -- does the cross-match still predict + match known asteroids?
If recall here is near 0 while it was ~91% on the 2024 field, the issue is
far-epoch orbit propagation / the coarse field-filter, NOT the discovery chain.
Reports predicted-known count, catalog epoch span, and match counts at several
radii (separating 'none predicted' from 'predicted but not matching').
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB = "C:/Users/Josh/AppData/Local/Temp/recovery_clean.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sci", default="data/decam_discovery_field/c4d_150418_081903_ooi_VR_v4.fits.fz")
    ap.add_argument("--n-ccd", type=int, default=20)
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    inst = load_decam_instcal(str(args.sci), read_dqm=False)
    mjd = inst.mjd
    ccds = [c for c in inst.ccds if c.wcs is not None and c.magzero > 0][:args.n_ccd]
    dr, dd = [], []
    for c in ccds:
        try:
            for s in detect_sources_in_image(np.asarray(c.science, float), c.wcs,
                                              mjd=mjd, image_id="x", fwhm_px=4.0,
                                              threshold_sigma=5.0, zeropoint_mag=c.magzero):
                dr.append(s.ra); dd.append(s.dec)
        except Exception:
            continue
    dr = np.array(dr); dd = np.array(dd)
    print(f"sci mjd={mjd:.3f} (year~{2000 + (mjd-51544.5)/365.25:.1f}); "
          f"{len(ccds)} CCDs, {len(dr)} detections")

    db = open_db(DB)
    inf = pick_orbits_in_field(db, mjd, (dr.min(), dr.max()), (dd.min(), dd.max()),
                                 max_mag=23.0, limit_candidates=1600000)
    recs = [x[0] for x in inf]
    epochs = np.array([getattr(r, "epoch_mjd", np.nan) for r in recs], float)
    print(f"pick_orbits returned {len(recs)} candidates; "
          f"catalog epoch_mjd median={np.nanmedian(epochs):.0f} "
          f"(delta to sci = {np.nanmedian(epochs)-mjd:.0f} d = {(np.nanmedian(epochs)-mjd)/365.25:.1f} yr)")
    if not recs:
        print("=> 0 candidates returned by the coarse field-filter at this epoch."); return 0
    eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km("807", mjd))

    rings = [2.5, 5, 10, 30, 120]
    onfoot = 0
    hits = {r: 0 for r in rings}
    for i in range(len(recs)):
        if np.isnan(eph[i, 0]):
            continue
        cd = math.cos(math.radians(eph[i, 1]))
        sep = np.hypot((dr - eph[i, 0]) * cd, dd - eph[i, 1]) * 3600
        if sep.size == 0 or sep.min() > 120:
            continue
        onfoot += 1
        for r in rings:
            if sep.min() <= r:
                hits[r] += 1
    print(f"predicted knowns on covered footprint: {onfoot}")
    for r in rings:
        print(f"  matched within {r:5.1f}\": {hits[r]}")
    print("  (2024 field gave ~73-94% within 2.5\"; if near-0 here -> far-epoch "
          "propagation broke the prediction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
