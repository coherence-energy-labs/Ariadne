"""Full operational discovery driver: A -> C -> B -> IOD -> D, one night.

The end-to-end production pipeline:

  1. (A) Query / download / parse DECam exposures
  2.     Detect sources + Gaia astrometric refinement + photometric ZP
  3.     Persist detections to the DB
  4. (B) Build within-night tracklets
  5. (C) Cross-match detections against MPC catalog; flag KNOWN
  6.     Multi-night linking against DB-resident OPEN chains
  7. (IOD) Run robust_iod on every newly-extended or newly-seeded chain
  8. (D) For each chain that passes the grade-A QC, write ADES + 80-col
         submission packet to disk
  9.     Print operational summary

Usage:
  # From real NOIRLab data:
  python scripts/run_full_discovery_night.py \\
      --db data/discovery_db.sqlite \\
      --query-ra 15 --query-dec -30 --query-radius 1.0 \\
      --query-mjd-min 57680 --query-mjd-max 57700 \\
      --max-exposures 4 --max-ccds 4

  # From local FITS:
  python scripts/run_full_discovery_night.py \\
      --db data/discovery_db.sqlite \\
      --fits one.fits.fz two.fits.fz

  # First time: ingest MPCORB once
  python scripts/run_full_discovery_night.py --db data/discovery_db.sqlite \\
      --ingest-mpcorb data/mpc_catalog/MPCORB.DAT.gz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


def _maybe_ingest_mpcorb(db, path: str | None, *, limit: int | None = None) -> int:
    if not path:
        return 0
    p = Path(path)
    if not p.exists():
        print(f"    MPCORB file not found: {p}", flush=True)
        return 0
    from ariadne.discovery.imaging.mpc_catalog import ingest_mpcorb_to_db
    n = ingest_mpcorb_to_db(db, p, limit=limit)
    print(f"    ingested {n} known orbital records into DB", flush=True)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True,
                     help="SQLite DB path (created if missing)")
    p.add_argument("--fits", nargs="*", default=None,
                     help="Local FITS file(s) to process")
    p.add_argument("--query-ra", type=float, default=None)
    p.add_argument("--query-dec", type=float, default=None)
    p.add_argument("--query-radius", type=float, default=0.5)
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
    p.add_argument("--mpc-match-arcsec", type=float, default=3.0)
    p.add_argument("--mpc-target-mjd", type=float, default=None,
                     help="Force ephemeris epoch for MPC cross-match; "
                            "default uses tonight's median detection MJD")
    p.add_argument("--mpc-limit-known", type=int, default=5000,
                     help="Cap on # known orbits to ephem (for speed)")
    p.add_argument("--ingest-mpcorb", default=None,
                     help="Path to MPCORB.DAT[.gz] to ingest before linking")
    p.add_argument("--mpcorb-ingest-limit", type=int, default=None,
                     help="Cap on # records to ingest from MPCORB")
    p.add_argument("--iod-rms-acceptance", type=float, default=10.0)
    p.add_argument("--iod-min-tracklets", type=int, default=2)
    p.add_argument("--iod-n-draws", type=int, default=3)
    p.add_argument("--observatory-code", default="W84")
    p.add_argument("--submission-out", default="data/discovery_submissions")
    p.add_argument("--out", default="data/discovery_night_report")
    args = p.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    sub_root = Path(args.submission_out); sub_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("FULL DISCOVERY NIGHT: A -> C -> B -> IOD -> D", flush=True)
    print("=" * 70, flush=True)

    # ============================================================
    # Setup
    # ============================================================
    from ariadne.discovery.imaging.detection_db import open_db, DetectionRow
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import (
        detect_sources_in_image, Source)
    from ariadne.discovery.imaging.gaia_refine import refine_to_gaia
    from ariadne.discovery.imaging.noirlab_sia2 import (
        query_decam_exposures, download_decam_exposure)
    from ariadne.discovery.imaging.mpc_catalog import flag_known_in_db
    from ariadne.discovery.imaging.multi_night_linker import link_tonight
    from ariadne.discovery.imaging.chain_iod import (
        run_iod_on_all_open_chains)
    from ariadne.discovery.imaging.mpc_submission import (
        evaluate_grade_a, write_submission_packet)

    db = open_db(args.db)
    pre_stats = db.stats()
    print(f"\nDB: {args.db}", flush=True)
    print(f"  pre: dets={pre_stats['n_detections']} "
            f"trk={pre_stats['n_tracklets']} "
            f"chains={pre_stats['n_chains']} "
            f"known={pre_stats['n_known_objects']}",
            flush=True)

    if args.ingest_mpcorb:
        print(f"\n[*] Ingesting MPCORB ...", flush=True)
        _maybe_ingest_mpcorb(db, args.ingest_mpcorb,
                                limit=args.mpcorb_ingest_limit)

    # ============================================================
    # [A] Resolve FITS list (local or NOIRLab)
    # ============================================================
    fits_paths: list[Path] = []
    workdir = Path(tempfile.mkdtemp(prefix="full_disc_"))
    if args.fits:
        fits_paths = [Path(f) for f in args.fits]
        print(f"\n[A] {len(fits_paths)} local FITS files", flush=True)
    elif args.query_ra is not None and args.query_dec is not None:
        print(f"\n[A] NOIRLab query ra={args.query_ra} dec={args.query_dec}",
                flush=True)
        records = query_decam_exposures(
            ra=args.query_ra, dec=args.query_dec,
            radius_deg=args.query_radius,
            mjd_min=args.query_mjd_min, mjd_max=args.query_mjd_max,
            band=args.query_band, proc_type="instcal",
            max_results=args.max_exposures * 2)
        print(f"    {len(records)} records returned", flush=True)
        for rec in records:
            if len(fits_paths) >= args.max_exposures:
                break
            local = download_decam_exposure(rec, workdir, timeout_s=600)
            if local is not None:
                fits_paths.append(local)
                print(f"    downloaded {local.name} "
                        f"({local.stat().st_size/1e6:.0f} MB)", flush=True)
    if not fits_paths:
        print("    no FITS to process; aborting", flush=True)
        return 1

    # ============================================================
    # [A+B] Per-exposure detect + ingest + per-night tracklets
    # ============================================================
    print(f"\n[A+B] Detect + ingest + within-night tracklets",
            flush=True)
    all_inserted_per_night: dict[int, list[tuple[int, Source]]] = {}
    for path in fits_paths:
        try:
            inst = load_decam_instcal(path, read_dqm=True)
        except Exception as exc:
            print(f"    parse failed for {path.name}: {exc}", flush=True)
            continue
        print(f"    {path.name}: mjd={inst.mjd:.4f} band={inst.band}",
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
            if (refinement and refinement.n_matches >= args.gaia_min_match
                    and refinement.rms_residual_arcsec
                    < args.gaia_accept_arcsec):
                astrom_sigma = refinement.rms_residual_arcsec
            else:
                astrom_sigma = 0.5
            rows = []
            for s in refined:
                mag_ab = (-2.5 * math.log10(s.flux) + ccd.magzero
                            if (s.flux > 0 and ccd.magzero > 0) else -99.0)
                rows.append(DetectionRow(
                    image_id=s.image_id, ccd_id=ccd.name,
                    mjd=s.mjd, ra=s.ra, dec=s.dec,
                    mag=mag_ab, flux=s.flux, fwhm_px=s.fwhm_px,
                    x_pix=s.x, y_pix=s.y,
                    astrom_sigma_arcsec=astrom_sigma))
            ids = db.insert_detections(rows)
            night = int(ccd.mjd)
            for det_id, det_row in zip(ids, rows):
                sm = Source(ra=det_row.ra, dec=det_row.dec,
                              flux=det_row.flux, mag=det_row.mag,
                              fwhm_px=det_row.fwhm_px, mjd=ccd.mjd,
                              image_id=det_row.image_id,
                              x=det_row.x_pix, y=det_row.y_pix)
                all_inserted_per_night.setdefault(night, []).append(
                    (det_id, sm))
            gaia_str = (f"Gaia: {refinement.n_matches} match, "
                          f"{refinement.rms_residual_arcsec:.3f}\""
                          if refinement else "Gaia: skip")
            print(f"      ccd{ccd.ccdnum:02d} {ccd.name}: "
                    f"{len(rows)} dets, {gaia_str}", flush=True)

    # Build within-night tracklets via the proper module
    from ariadne.discovery.imaging.within_night_tracklets import (
        build_within_night_tracklets)
    night_tracklets: list[dict] = []
    for night, det_rows in all_inserted_per_night.items():
        rows = build_within_night_tracklets(det_rows)
        print(f"    night {night}: {len(rows)} tracklets from "
                f"{len(det_rows)} dets", flush=True)
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

    # ============================================================
    # [C] MPC catalog cross-match
    # ============================================================
    if pre_stats["n_known_objects"] > 0 or args.ingest_mpcorb:
        print(f"\n[C] MPC catalog cross-match", flush=True)
        target_mjd = (args.mpc_target_mjd if args.mpc_target_mjd
                        else (sum(r["mean_mjd"] for r in night_tracklets)
                                / max(len(night_tracklets), 1)
                                if night_tracklets
                                else (db.stats().get("mjd_max")
                                        or 60450.0)))
        t0 = time.time()
        n_flagged = flag_known_in_db(
            db, target_mjd=target_mjd,
            mjd_box_days=0.5,
            match_radius_arcsec=args.mpc_match_arcsec,
            limit_known=args.mpc_limit_known)
        print(f"    flagged {n_flagged} detections as KNOWN "
                f"(target_mjd={target_mjd:.4f}, "
                f"limit_known={args.mpc_limit_known}, "
                f"wall {time.time()-t0:.1f}s)", flush=True)
    else:
        print(f"\n[C] MPC cross-match skipped (no known_objects in DB; "
                f"pass --ingest-mpcorb to populate)", flush=True)

    # ============================================================
    # [B-link] Multi-night linking
    # ============================================================
    print(f"\n[B-link] Multi-night chain extension", flush=True)
    if night_tracklets:
        report = link_tonight(
            db, night_tracklets,
            link_window_days=args.link_window_days,
            seed_window_days=args.seed_window_days,
            position_tol_arcsec=120.0, rate_tol_pct=50.0)
        print(f"    extended {report.n_chains_extended} chains, "
                f"seeded {report.n_chains_seeded} new chains, "
                f"{report.n_tracklets_unmatched} tracklets unmatched",
                flush=True)
    else:
        print(f"    no tonight tracklets to link", flush=True)

    # ============================================================
    # [IOD] Run robust_iod on all open chains
    # ============================================================
    print(f"\n[IOD] Robust IOD on open chains", flush=True)
    neural_weights = None
    np_path = Path("data/neural_orbit_prior_weights.json")
    if np_path.exists():
        try:
            from ariadne.discovery.imaging.neural_orbit_prior import load_weights
            neural_weights = load_weights(np_path)
        except Exception:
            pass
    iod_results = run_iod_on_all_open_chains(
        db,
        min_tracklets=args.iod_min_tracklets,
        rms_acceptance_arcsec=args.iod_rms_acceptance,
        n_draws=args.iod_n_draws,
        neural_weights=neural_weights)
    n_iod_attempt = len(iod_results)
    n_iod_success = sum(1 for _, r in iod_results if r.get("success"))
    n_iod_strategies = Counter(
        r["strategy"] for _, r in iod_results if r.get("success"))
    print(f"    {n_iod_success}/{n_iod_attempt} chains have a fit",
            flush=True)
    for strat, count in n_iod_strategies.most_common():
        print(f"      {strat}: {count}", flush=True)

    # ============================================================
    # [D] Grade-A QC + submission packets
    # ============================================================
    print(f"\n[D] Grade-A QC + submission packets", flush=True)
    n_grade_a = 0
    n_packets = 0
    for chain_id, res in iod_results:
        if not res.get("success"):
            continue
        chain = db.get_chain(chain_id)
        if chain is None:
            continue
        dets = db.get_chain_detections(chain_id)
        # Skip known objects -- don't re-submit existing catalog entries
        if any(d.get("known_designation") for d in dets):
            continue
        grade = evaluate_grade_a(chain, dets)
        if not grade.passed:
            continue
        n_grade_a += 1
        files = write_submission_packet(
            chain, dets, sub_root,
            observatory_code=args.observatory_code,
            designation_hint=f"ARI{chain_id:04d}")
        n_packets += 1
        print(f"    chain {chain_id}: grade-A, "
                f"n_obs={grade.n_observations}, "
                f"arc={grade.arc_days:.2f}d, "
                f"rms={grade.astrometric_rms_arcsec:.3f}\", "
                f"packet -> {files['ades'].name}",
                flush=True)
    print(f"    {n_grade_a} chains graded A, {n_packets} submission packets "
            f"in {sub_root}", flush=True)

    # ============================================================
    # Summary
    # ============================================================
    post_stats = db.stats()
    print(f"\n{'='*70}", flush=True)
    print("OPERATIONAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  fits processed     : {len(fits_paths)}", flush=True)
    print(f"  new detections     : "
            f"{post_stats['n_detections'] - pre_stats['n_detections']}",
            flush=True)
    print(f"  new tracklets      : "
            f"{post_stats['n_tracklets'] - pre_stats['n_tracklets']}",
            flush=True)
    print(f"  open chains        : {post_stats['n_chains_open']}", flush=True)
    print(f"  chains IOD'd       : {n_iod_success}", flush=True)
    print(f"  grade-A submissions: {n_grade_a}", flush=True)
    print(f"  DB: {args.db}", flush=True)
    report_out = {
        "fits_processed": [str(p) for p in fits_paths],
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "iod_attempted": n_iod_attempt,
        "iod_success": n_iod_success,
        "iod_strategy_counts": dict(n_iod_strategies),
        "grade_a_submissions": n_grade_a,
        "submission_dir": str(sub_root),
    }
    (out_root / "report.json").write_text(json.dumps(report_out, indent=2))
    print(f"\n[*] report -> {out_root}/report.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
