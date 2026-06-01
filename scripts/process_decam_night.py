"""Per-night driver: ingest one night's worth of DECam exposures into the
persistent detection database.

This is the operational entry point. For each exposure given:
  1. parse the multi-extension instcal FITS,
  2. per-CCD: source extraction + Gaia refinement + photometric ZP,
  3. insert detections into the DB,
  4. build within-night tracklets from same-night source pairs,
  5. link tonight's tracklets against the DB's open chains
     (multi-night attribution),
  6. seed new chains for unmatched tonight tracklets that pair with
     past-night tracklets within the seed window,
  7. write a per-night summary report.

Run:
  python scripts/process_decam_night.py \\
      --db data/decam_db.sqlite \\
      --fits one.fits.fz two.fits.fz \\
      --max-ccds 4 \\
      --night-mjd 60450

Or to pull from NOIRLab:
  python scripts/process_decam_night.py \\
      --db data/decam_db.sqlite \\
      --query-ra 15 --query-dec -30 --query-radius 1 \\
      --query-mjd-min 56000 --query-mjd-max 58000 \\
      --max-exposures 3 --max-ccds 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path


def _build_within_night_tracklets(detections_with_ids, *,
                                       max_pair_separation_arcsec=200.0,
                                       min_pair_separation_arcsec=0.3):
    """Form within-night tracklets from same-image detection pairs.

    `detections_with_ids` is a list of (db_id, Source) tuples for one
    night's detections. Returns a list of TrackletRow records.
    """
    from ariadne.discovery.imaging.detection_db import TrackletRow
    # Group by image_id
    by_image: dict[str, list] = {}
    for db_id, src in detections_with_ids:
        by_image.setdefault(src.image_id, []).append((db_id, src))
    tracklets = []
    max_sep_deg = max_pair_separation_arcsec / 3600.0
    min_sep_deg = min_pair_separation_arcsec / 3600.0
    # Form pairs across exposures of the same night that have CLOSE positions
    image_ids = sorted(by_image.keys(), key=lambda iid: by_image[iid][0][1].mjd)
    for i, iid_a in enumerate(image_ids):
        for iid_b in image_ids[i + 1:]:
            srcs_a = by_image[iid_a]
            srcs_b = by_image[iid_b]
            if not srcs_a or not srcs_b:
                continue
            mjd_a = srcs_a[0][1].mjd
            mjd_b = srcs_b[0][1].mjd
            if mjd_b <= mjd_a:
                continue
            dt_hr = (mjd_b - mjd_a) * 24.0
            if dt_hr < 1e-3 or dt_hr > 24.0:
                continue   # not within same night
            for db_id_a, sa in srcs_a:
                cos_dec = math.cos(math.radians(sa.dec))
                for db_id_b, sb in srcs_b:
                    dra = (sb.ra - sa.ra) * cos_dec
                    ddec = sb.dec - sa.dec
                    d_deg = math.hypot(dra, ddec)
                    if d_deg < min_sep_deg or d_deg > max_sep_deg:
                        continue
                    rate_arcsec_hr = d_deg * 3600.0 / dt_hr
                    pa_deg = math.degrees(math.atan2(dra, ddec)) % 360.0
                    tracklets.append(TrackletRow(
                        detection_a_id=db_id_a,
                        detection_b_id=db_id_b,
                        mean_mjd=0.5 * (mjd_a + mjd_b),
                        mean_ra=0.5 * (sa.ra + sb.ra),
                        mean_dec=0.5 * (sa.dec + sb.dec),
                        rate_arcsec_hr=rate_arcsec_hr,
                        pa_deg=pa_deg,
                        rate_sigma=0.0,
                        mag=0.5 * (sa.mag + sb.mag) if sa.mag > -50 and sb.mag > -50 else -99.0,
                        night=int(mjd_a),
                    ))
    return tracklets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True,
                     help="SQLite DB path (created if missing)")
    p.add_argument("--fits", nargs="*", default=None,
                     help="Local FITS file(s) to process")
    p.add_argument("--query-ra", type=float, default=None,
                     help="Cone RA for NOIRLab query (if --fits not given)")
    p.add_argument("--query-dec", type=float, default=None,
                     help="Cone Dec for NOIRLab query")
    p.add_argument("--query-radius", type=float, default=0.5,
                     help="Cone radius (deg)")
    p.add_argument("--query-mjd-min", type=float, default=None)
    p.add_argument("--query-mjd-max", type=float, default=None)
    p.add_argument("--query-band", default="r")
    p.add_argument("--max-exposures", type=int, default=3)
    p.add_argument("--max-ccds", type=int, default=4)
    p.add_argument("--detect-sigma", type=float, default=4.0)
    p.add_argument("--gaia-radius-deg", type=float, default=0.15)
    p.add_argument("--gaia-min-match", type=int, default=8)
    p.add_argument("--gaia-accept-arcsec", type=float, default=0.5)
    p.add_argument("--link-window-days", type=float, default=30.0)
    p.add_argument("--seed-window-days", type=float, default=7.0)
    p.add_argument("--out", default="data/decam_night_report")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print("=" * 70, flush=True)
    print("PHASE B: per-night DECam ingestion + DB linking", flush=True)
    print("=" * 70, flush=True)

    # ---------------------------------------------------------------
    # Resolve input FITS list
    # ---------------------------------------------------------------
    fits_paths = []
    workdir = Path(tempfile.mkdtemp(prefix="phase_b_"))
    if args.fits:
        for f in args.fits:
            fits_paths.append(Path(f))
    elif args.query_ra is not None and args.query_dec is not None:
        from ariadne.discovery.imaging.noirlab_sia2 import (
            query_decam_exposures, download_decam_exposure)
        print(f"\n[1] NOIRLab query ra={args.query_ra} dec={args.query_dec} "
              f"r={args.query_radius} ...", flush=True)
        t0 = time.time()
        records = query_decam_exposures(
            ra=args.query_ra, dec=args.query_dec,
            radius_deg=args.query_radius,
            mjd_min=args.query_mjd_min, mjd_max=args.query_mjd_max,
            band=args.query_band, proc_type="instcal",
            max_results=args.max_exposures * 3)
        print(f"    {len(records)} records in {time.time()-t0:.1f}s",
               flush=True)
        for rec in records:
            if len(fits_paths) >= args.max_exposures:
                break
            local = download_decam_exposure(rec, workdir, timeout_s=600)
            if local is not None:
                fits_paths.append(local)
                size_mb = local.stat().st_size / 1e6
                print(f"      downloaded {local.name} ({size_mb:.1f} MB)",
                      flush=True)
    if not fits_paths:
        print("    no FITS to process; aborting", flush=True)
        return 1

    # ---------------------------------------------------------------
    # Open DB, ingest each exposure
    # ---------------------------------------------------------------
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow)
    db = open_db(args.db)
    print(f"\n[2] DB open: {args.db}", flush=True)
    pre_stats = db.stats()
    print(f"    pre-stats: dets={pre_stats['n_detections']} "
          f"trk={pre_stats['n_tracklets']} chains={pre_stats['n_chains']}",
          flush=True)

    from ariadne.discovery.imaging.decam_instcal import (
        load_decam_instcal, iterate_ccds)
    from ariadne.discovery.imaging.source_extraction import (
        detect_sources_in_image, Source)
    from ariadne.discovery.imaging.gaia_refine import refine_to_gaia
    import math as _m

    detections_by_night: dict[int, list] = {}
    for path in fits_paths:
        print(f"\n[3] parse {path.name} ...", flush=True)
        try:
            inst = load_decam_instcal(path, read_dqm=True)
        except Exception as e:
            print(f"    parse failed: {str(e)[:120]}", flush=True)
            continue
        print(f"    {inst.n_ccds} CCDs, mjd={inst.mjd:.4f} band={inst.band}",
              flush=True)
        for ccd in inst.ccds[:args.max_ccds]:
            if ccd.wcs is None:
                continue
            try:
                srcs = detect_sources_in_image(
                    ccd.science, ccd.wcs,
                    mjd=ccd.mjd, image_id=f"{path.stem}_ccd{ccd.ccdnum}",
                    fwhm_px=3.0, threshold_sigma=args.detect_sigma)
            except Exception:
                continue
            ny, nx = ccd.science.shape
            try:
                ctr = ccd.wcs.pixel_to_world_values(nx / 2, ny / 2)
                ctr_ra = float(ctr[0]) % 360.0
                ctr_dec = float(ctr[1])
            except Exception:
                continue
            try:
                refined, refinement = refine_to_gaia(
                    srcs, image_centre_ra_deg=ctr_ra,
                    image_centre_dec_deg=ctr_dec,
                    image_radius_deg=args.gaia_radius_deg,
                    match_tol_arcsec=2.0,
                    accept_rms_arcsec=args.gaia_accept_arcsec)
            except Exception:
                refined, refinement = srcs, None
            # Skip CCD if Gaia residual too high to trust
            if (refinement and refinement.n_matches >= args.gaia_min_match
                    and refinement.rms_residual_arcsec < args.gaia_accept_arcsec):
                astrom_sigma = refinement.rms_residual_arcsec
                refinement_str = (f"Gaia: {refinement.n_matches} matches, "
                                     f"resid {astrom_sigma:.3f}\"")
            else:
                astrom_sigma = 0.5    # conservative default when no Gaia
                refinement_str = "Gaia: skipped"
            rows = []
            for s in refined:
                mag_ab = (-2.5 * _m.log10(s.flux) + ccd.magzero
                            if (s.flux > 0 and ccd.magzero > 0) else -99.0)
                rows.append(DetectionRow(
                    image_id=s.image_id, ccd_id=ccd.name,
                    mjd=s.mjd, ra=s.ra, dec=s.dec,
                    mag=mag_ab, flux=s.flux, fwhm_px=s.fwhm_px,
                    x_pix=s.x, y_pix=s.y,
                    astrom_sigma_arcsec=astrom_sigma))
            ids = db.insert_detections(rows)
            print(f"      ccd{ccd.ccdnum:02d}({ccd.name}): {len(rows)} dets, "
                  f"{refinement_str}", flush=True)
            night = int(ccd.mjd)
            detections_by_night.setdefault(night, []).extend(
                [(did, _row_to_source(r, ccd.mjd))
                  for did, r in zip(ids, rows)])

    # ---------------------------------------------------------------
    # [4] Per-night within-night tracklet build + persist
    # ---------------------------------------------------------------
    print(f"\n[4] within-night tracklet build", flush=True)
    night_tracklets: list[dict] = []
    for night, dets in detections_by_night.items():
        rows = _build_within_night_tracklets(dets,
                                                max_pair_separation_arcsec=200.0,
                                                min_pair_separation_arcsec=0.3)
        print(f"    night {night}: {len(rows)} tracklets from {len(dets)} dets",
              flush=True)
        for r in rows:
            tid = db.insert_tracklet(r)
            night_tracklets.append({
                "tracklet_id": tid,
                "mean_mjd": r.mean_mjd,
                "mean_ra": r.mean_ra,
                "mean_dec": r.mean_dec,
                "rate_arcsec_hr": r.rate_arcsec_hr,
                "pa_deg": r.pa_deg,
            })

    # ---------------------------------------------------------------
    # [5] Multi-night linking
    # ---------------------------------------------------------------
    print(f"\n[5] multi-night linking", flush=True)
    from ariadne.discovery.imaging.multi_night_linker import link_tonight
    report = link_tonight(
        db, night_tracklets,
        link_window_days=args.link_window_days,
        seed_window_days=args.seed_window_days,
        position_tol_arcsec=120.0, rate_tol_pct=50.0)
    print(f"    extended {report.n_chains_extended} existing chains, "
          f"seeded {report.n_chains_seeded} new chains", flush=True)
    print(f"    {report.n_tracklets_unmatched} tracklets unmatched",
          flush=True)

    # ---------------------------------------------------------------
    # [6] Stats + report
    # ---------------------------------------------------------------
    final_stats = db.stats()
    print(f"\n[6] post-run DB stats:", flush=True)
    for k, v in final_stats.items():
        print(f"    {k} = {v}", flush=True)
    report_out = {
        "fits_processed": [str(p) for p in fits_paths],
        "pre_stats": pre_stats,
        "post_stats": final_stats,
        "tonight_tracklets": len(night_tracklets),
        "chains_extended": report.n_chains_extended,
        "chains_seeded": report.n_chains_seeded,
        "tracklets_unmatched": report.n_tracklets_unmatched,
        "linking_wall_s": report.wall_seconds,
    }
    (out_root / "report.json").write_text(json.dumps(report_out, indent=2))
    print(f"\n[*] report -> {out_root}/report.json", flush=True)
    return 0


def _row_to_source(row, mjd):
    """Adapter: re-wrap a DetectionRow as a Source for tracklet building."""
    from ariadne.discovery.imaging.source_extraction import Source
    return Source(ra=row.ra, dec=row.dec, flux=row.flux,
                    mag=row.mag, fwhm_px=row.fwhm_px,
                    mjd=mjd, image_id=row.image_id,
                    x=row.x_pix, y=row.y_pix)


if __name__ == "__main__":
    sys.exit(main())
