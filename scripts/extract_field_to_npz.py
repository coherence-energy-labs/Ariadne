"""Extract all FITS in a directory to a combined meas.npz for the discovery run.

Uses the fixed, self-calibrating detection (measure PSF FWHM once per exposure,
then detect all CCDs at that scale) and writes (ra, dec, mjd, mag, fwhm,
exposure) -- the same schema run_nsc_discovery.py consumes -- so the FITS path
and the NSC path share one discovery runner.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/decam_deep_field")
    ap.add_argument("--n-ccd", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--max-det-per-exposure", type=int, default=15000,
                    help="skip noise-inflated frames above this detection count")
    ap.add_argument("--out", default="data/decam_deep_field/meas.npz")
    args = ap.parse_args()

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.trailed_rate import measure_image_fwhm

    files = sorted(Path(args.data_dir).glob("c4d_*_ooi_*_v*.fits.fz"))
    cache_dir = Path(args.data_dir) / "_cache"; cache_dir.mkdir(exist_ok=True)
    RA, DEC, MJD, MAG, FW, EXP = [], [], [], [], [], []
    t0 = time.time()
    for f in files:
        cache = cache_dir / (f.stem + f".s{args.sigma:.0f}.npz")
        if cache.exists():
            c = np.load(cache)
            ra, dec, mag, mjd, fwhm = c["ra"], c["dec"], c["mag"], float(c["mjd"]), float(c["fwhm"])
        else:
            try:
                inst = load_decam_instcal(str(f), read_dqm=False)
            except Exception as e:
                print(f"  SKIP {f.name}: {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            mjd = inst.mjd
            ccds = [c for c in inst.ccds if c.wcs is not None and c.magzero > 0][:args.n_ccd]
            fwhms = [m for c in ccds[:4]
                     if (m := measure_image_fwhm(np.asarray(c.science, float), fwhm_guess=4.0))]
            fwhm = float(np.median(fwhms)) if fwhms else 4.0
            ra, dec, mag = [], [], []
            for c in ccds:
                try:
                    for s in detect_sources_in_image(np.asarray(c.science, float), c.wcs,
                                                      mjd=mjd, image_id=f.stem, fwhm_px=fwhm,
                                                      threshold_sigma=args.sigma,
                                                      zeropoint_mag=c.magzero, auto_fwhm=False):
                        ra.append(s.ra); dec.append(s.dec); mag.append(s.mag)
                except Exception:
                    continue
            ra = np.array(ra); dec = np.array(dec); mag = np.array(mag)
            np.savez(cache, ra=ra, dec=dec, mag=mag, mjd=mjd, fwhm=fwhm)
        if args.max_det_per_exposure and len(ra) > args.max_det_per_exposure:
            print(f"  SKIP {f.name}: anomalous {len(ra)} detections", flush=True)
            continue
        RA.append(ra); DEC.append(dec); MJD.append(np.full(len(ra), mjd))
        MAG.append(mag); FW.append(np.full(len(ra), fwhm))
        EXP.append(np.full(len(ra), f.stem, dtype=object))
        print(f"  {f.name}: {len(ra)} det, mjd {mjd:.3f}, fwhm {fwhm:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if not RA:
        print("no detections"); return 1
    out = Path(args.out)
    np.savez(out, ra=np.concatenate(RA), dec=np.concatenate(DEC),
             mjd=np.concatenate(MJD), mag=np.concatenate(MAG),
             fwhm=np.concatenate(FW), exposure=np.concatenate(EXP))
    print(f"\n  wrote {sum(len(r) for r in RA)} detections from {len(RA)} exposures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
