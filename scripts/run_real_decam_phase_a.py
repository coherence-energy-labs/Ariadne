"""Phase A: process a real DECam Community Pipeline single-epoch exposure
end-to-end. Measure astrometric precision vs Gaia DR3.

Success criterion: pull a real DECam instcal FITS from NOIRLab, run
source extraction + Gaia astrometric refinement on each CCD, output a
combined catalog with (ra, dec, mag, flux, fwhm) where the post-
refinement astrometric residual vs Gaia is <0.2 arcsec.

This is the first concrete milestone toward production NEO/TNO
discovery: prove the pipeline works on REAL survey data at production-
quality astrometric precision.

Run:
  python scripts/run_real_decam_phase_a.py \\
      --ra 180 --dec -10 --radius 0.5 \\
      --mjd-min 58200 --mjd-max 58400 --max-exposures 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ra", type=float, default=180.0,
                     help="Cone center RA, deg")
    p.add_argument("--dec", type=float, default=-10.0,
                     help="Cone center Dec, deg (default in DES area)")
    p.add_argument("--radius", type=float, default=0.5,
                     help="Cone radius, deg")
    p.add_argument("--mjd-min", type=float, default=None,
                     help="MJD lower bound")
    p.add_argument("--mjd-max", type=float, default=None,
                     help="MJD upper bound")
    p.add_argument("--band", default="r",
                     help="Filter band (g/r/i/z/Y), default r")
    p.add_argument("--max-exposures", type=int, default=1,
                     help="Number of exposures to process (1 is sufficient for Phase A)")
    p.add_argument("--out", default="data/real_decam_phase_a",
                     help="Output directory")
    p.add_argument("--max-ccds", type=int, default=4,
                     help="Process only the first N CCDs (default 4; set to 62 for the full mosaic)")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print("=" * 70, flush=True)
    print("PHASE A: real DECam Community Pipeline E2E", flush=True)
    print("=" * 70, flush=True)
    print(f"target: ra={args.ra} dec={args.dec} r={args.radius} band={args.band}",
          flush=True)
    print(f"output: {out_root}", flush=True)

    # ---------------------------------------------------------------
    # [1] Query NOIRLab for DECam exposures in the cone
    # ---------------------------------------------------------------
    print(f"\n[1] NOIRLab SIA2 query ...", flush=True)
    t0 = time.time()
    from ariadne.discovery.imaging.noirlab_sia2 import (
        query_decam_exposures, download_decam_exposure)
    records = query_decam_exposures(
        ra=args.ra, dec=args.dec, radius_deg=args.radius,
        mjd_min=args.mjd_min, mjd_max=args.mjd_max,
        band=args.band, proc_type="instcal",
        max_results=args.max_exposures * 3)   # over-fetch since some may fail
    print(f"    {len(records)} records returned in {time.time()-t0:.1f}s",
           flush=True)
    if not records:
        report = {"status": "no_records",
                    "query": {"ra": args.ra, "dec": args.dec,
                                "radius": args.radius, "band": args.band}}
        (out_root / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\n[*] no records; report -> {out_root}/report.json",
               flush=True)
        return 1
    for r in records[:5]:
        print(f"      {r.obs_id}  mjd={r.obs_mjd:.4f}  band={r.band}  "
              f"exp={r.exptime_s:.0f}s  proc={r.proc_type}", flush=True)

    # ---------------------------------------------------------------
    # [2] Download instcal FITS
    # ---------------------------------------------------------------
    print(f"\n[2] download instcal FITS ...", flush=True)
    workdir = out_root / "fits_cache"
    workdir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for rec in records:
        if len(downloaded) >= args.max_exposures:
            break
        local = download_decam_exposure(rec, workdir)
        if local is not None:
            size_mb = local.stat().st_size / 1e6
            print(f"      {local.name} ({size_mb:.1f} MB)", flush=True)
            downloaded.append((rec, local))
        else:
            print(f"      skip (download failed): {rec.archive_id}",
                  flush=True)
    if not downloaded:
        report = {"status": "no_downloads", "n_records": len(records)}
        (out_root / "report.json").write_text(json.dumps(report, indent=2))
        return 1

    # ---------------------------------------------------------------
    # [3] Multi-extension parse: per-CCD images + WCS
    # ---------------------------------------------------------------
    print(f"\n[3] parse multi-extension FITS ...", flush=True)
    from ariadne.discovery.imaging.decam_instcal import (
        load_decam_instcal, iterate_ccds, calibrate_magnitudes)
    per_exposure = []
    for rec, path in downloaded:
        try:
            inst = load_decam_instcal(path)
        except Exception as e:
            print(f"      failed to parse {path.name}: {str(e)[:100]}",
                   flush=True)
            continue
        print(f"      {path.name}: {inst.n_ccds} CCDs, band={inst.band}, "
              f"mjd={inst.mjd:.4f}, exp={inst.exptime_s:.0f}s",
              flush=True)
        per_exposure.append((rec, inst))

    # ---------------------------------------------------------------
    # [4] Per-CCD source extraction + Gaia refinement + photometric ZP
    # ---------------------------------------------------------------
    print(f"\n[4] per-CCD detect + Gaia refine + photometric calibration",
           flush=True)
    from ariadne.discovery.imaging.source_extraction import (
        detect_sources_in_image, Source)
    from ariadne.discovery.imaging.gaia_refine import refine_to_gaia
    import math as _math
    all_sources = []
    per_ccd_stats = []
    for rec, inst in per_exposure:
        for ccd in inst.ccds[:args.max_ccds]:
            if ccd.wcs is None:
                continue
            try:
                srcs = detect_sources_in_image(
                    ccd.science, ccd.wcs,
                    mjd=ccd.mjd,
                    image_id=f"{rec.obs_id}_ccd{ccd.ccdnum}",
                    fwhm_px=3.0, threshold_sigma=4.0)
            except Exception as e:
                print(f"        ccd{ccd.ccdnum}: detect failed {str(e)[:80]}",
                       flush=True)
                continue
            n_raw = len(srcs)
            # Gaia refinement: pulls (ra, dec) onto Gaia DR3 frame.
            # IMPORTANT: in a DECam mosaic the WCS.CRVAL is the boresight
            # (telescope pointing) -- the SAME for every CCD. To get the
            # CCD's actual sky footprint center, project the CCD's pixel
            # center through the WCS.
            ny, nx = ccd.science.shape
            try:
                ctr_world = ccd.wcs.pixel_to_world_values(nx / 2, ny / 2)
                ctr_ra = float(ctr_world[0]) % 360.0
                ctr_dec = float(ctr_world[1])
            except Exception:
                ctr_ra = float(ccd.wcs.wcs.crval[0])
                ctr_dec = float(ccd.wcs.wcs.crval[1])
            try:
                refined_srcs, refinement = refine_to_gaia(
                    srcs,
                    image_centre_ra_deg=ctr_ra,
                    image_centre_dec_deg=ctr_dec,
                    image_radius_deg=0.15,
                    match_tol_arcsec=2.0,
                    accept_rms_arcsec=0.5)
            except Exception:
                refined_srcs = srcs
                refinement = None
            # Calibrate magnitudes with this CCD's MAGZERO
            calibrated_srcs = []
            for s in refined_srcs:
                if s.flux > 0 and ccd.magzero > 0:
                    mag_ab = float(-2.5 * _math.log10(s.flux) + ccd.magzero)
                else:
                    mag_ab = -99.0
                calibrated_srcs.append(Source(
                    ra=s.ra, dec=s.dec, flux=s.flux, mag=mag_ab,
                    fwhm_px=s.fwhm_px, mjd=s.mjd, image_id=s.image_id,
                    x=s.x, y=s.y,
                ))
            all_sources.extend(calibrated_srcs)
            stats = {
                "obs_id": rec.obs_id, "ccdnum": ccd.ccdnum,
                "ccd_name": ccd.name, "n_raw": n_raw,
                "n_refined": len(calibrated_srcs),
                "magzero": float(ccd.magzero), "seeing_arcsec": float(ccd.seeing_arcsec),
            }
            if refinement is not None:
                stats["gaia_n_matches"] = int(refinement.n_matches)
                stats["gaia_rms_arcsec"] = float(refinement.rms_residual_arcsec)
                stats["gaia_mean_dra_arcsec"] = float(refinement.mean_dra_arcsec)
                stats["gaia_mean_ddec_arcsec"] = float(refinement.mean_ddec_arcsec)
            per_ccd_stats.append(stats)
            ccd_label = f"ccd{ccd.ccdnum:02d}({ccd.name})"
            gaia_str = (
                f"Gaia: {refinement.n_matches} matches, "
                f"residual {refinement.rms_residual_arcsec:.3f}\""
                if refinement else "Gaia: skipped")
            print(f"        {ccd_label}: {n_raw} sources, ZP={ccd.magzero:.2f}, "
                  f"{gaia_str}", flush=True)

    print(f"\n[5] total {len(all_sources)} calibrated sources across "
          f"{len(per_ccd_stats)} CCDs", flush=True)
    # ---------------------------------------------------------------
    # [6] Report
    # ---------------------------------------------------------------
    rms_residuals = [s["gaia_rms_arcsec"] for s in per_ccd_stats
                       if "gaia_rms_arcsec" in s
                       and s["gaia_rms_arcsec"] > 0]
    median_rms = float(sorted(rms_residuals)[len(rms_residuals) // 2]) \
                   if rms_residuals else 0.0
    report = {
        "status": "ok",
        "query": {"ra": args.ra, "dec": args.dec,
                    "radius": args.radius, "band": args.band,
                    "mjd_min": args.mjd_min, "mjd_max": args.mjd_max},
        "n_records": len(records),
        "n_downloaded": len(downloaded),
        "n_ccds_processed": len(per_ccd_stats),
        "n_calibrated_sources": len(all_sources),
        "median_gaia_rms_arcsec": median_rms,
        "phase_a_success_threshold_arcsec": 0.2,
        "phase_a_passed": (median_rms < 0.2 and median_rms > 0),
        "per_ccd_stats": per_ccd_stats[:20],
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[*] report -> {out_root}/report.json", flush=True)
    print(f"    median Gaia residual: {median_rms:.3f}\"  "
          f"(target < 0.2\"; "
          f"{'PASSED' if report['phase_a_passed'] else 'FAILED'})",
           flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
