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
    p.add_argument("--no-streaks", action="store_true",
                     help="disable fast-mover streak detection")
    p.add_argument("--no-self-check", action="store_true",
                     help="disable the ephemeris-vs-Horizons self-check gate")
    p.add_argument("--no-discovery", action="store_true",
                     help="disable the within-night discovery loop (>=3 exp)")
    p.add_argument("--single-snapshot-rates", action="store_true",
                     help="measure per-source trail rate + flag incomers "
                          "(1-exposure screen; per-source stamp moments)")
    p.add_argument("--streak-min-length-px", type=float, default=20.0,
                     help="reject streaks shorter than this (px)")
    p.add_argument("--pixel-scale-arcsec", type=float, default=0.263,
                     help="detector pixel scale (DECam = 0.263)")
    p.add_argument("--gaia-radius-deg", type=float, default=0.15)
    p.add_argument("--gaia-min-match", type=int, default=8)
    p.add_argument("--gaia-accept-arcsec", type=float, default=0.5)
    p.add_argument("--link-window-days", type=float, default=30.0)
    p.add_argument("--seed-window-days", type=float, default=7.0)
    p.add_argument("--mpc-match-arcsec", type=float, default=2.5,
                     help="cross-match radius; calibrated on real DECam: "
                          "recall plateaus by ~1.5\", 2.5\" keeps the NEO "
                          "ephemeris tail at ~4x lower chance-FP than 3\"")
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
    # Single-snapshot + discovery engines (wired below)
    from ariadne.discovery.imaging.ephemeris_selfcheck import (
        validate_against_horizons)
    from ariadne.discovery.imaging.discovery_pipeline import (
        run_discovery, false_positive_floor)
    from ariadne.discovery.imaging.mpc_catalog import (
        observatory_geo_km, OrbitalElements)
    import json as _json

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
    night_streak_tracklets: list[dict] = []
    # det_id -> (rate_arcsec_hr, snr) from the single-snapshot trail measure;
    # used for incomer screening AFTER cross-match (unmatched sources only)
    snapshot_rate: dict[int, tuple] = {}
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
            # ----------------------------------------------------------
            # Single-snapshot velocity (trailed_rate): measure each source's
            # trail rate+PA from its PSF and flag INCOMER candidates (bright +
            # untrailed -> nominally distant -> implausibly large -> likely
            # nearby/radial). This is the 1-exposure-per-night screen that
            # tracklet linking cannot do. Opt-in (per-source stamp moments
            # are not free). Connects trailed_rate + orbit_geometry.
            if args.single_snapshot_rates and ccd.magzero > 0:
                try:
                    from ariadne.discovery.imaging.trailed_rate import (
                        rate_from_stamp, fit_trailed_psf,
                        stellar_psf_anisotropy, stellar_psf_fwhm)
                    import numpy as _np
                    sci = _np.asarray(ccd.science, float)
                    aniso = stellar_psf_anisotropy(sci)
                    psf_fwhm = stellar_psf_fwhm(sci)   # MEASURED seeing
                    H, W = sci.shape
                    for det_id, s in zip(ids, refined):
                        x, y = int(round(s.x)), int(round(s.y))
                        if x < 18 or y < 18 or x > W - 18 or y > H - 18:
                            continue
                        st = sci[y - 18:y + 18 + 1, x - 18:x + 18 + 1]
                        # Two-stage: cheap moments screen, then refine the
                        # ELONGATED candidates with the matched-filter
                        # trailed-PSF fit (unbiased, far less noisy at faint
                        # flux). Keeps the per-source cost bounded.
                        est = rate_from_stamp(
                            st, psf_aniso=aniso,
                            pixscale_arcsec=args.pixel_scale_arcsec,
                            t_exp_s=inst.exptime_s)
                        if est.rate_arcsec_hr > 50.0:
                            est = fit_trailed_psf(
                                st, psf_fwhm_px=psf_fwhm,
                                pixscale_arcsec=args.pixel_scale_arcsec,
                                t_exp_s=inst.exptime_s)
                        snapshot_rate[det_id] = (est.rate_arcsec_hr, est.snr)
                except Exception as exc:
                    print(f"        single-snapshot rates skipped: {exc}",
                            flush=True)

            gaia_str = (f"Gaia: {refinement.n_matches} match, "
                          f"{refinement.rms_residual_arcsec:.3f}\""
                          if refinement else "Gaia: skip")
            print(f"      ccd{ccd.ccdnum:02d} {ccd.name}: "
                    f"{len(rows)} dets, {gaia_str}", flush=True)

            # ----------------------------------------------------------
            # Streak detection for fast NEOs that trail across the
            # exposure (DAOStarFinder rejects elongated PSFs). Each
            # asteroid-candidate streak becomes two timed detections +
            # an instant within-night tracklet (rate vector).
            # ----------------------------------------------------------
            if not args.no_streaks and inst.exptime_s > 0:
                try:
                    from ariadne.discovery.imaging.streaks import (
                        detect_streaks)
                    from ariadne.discovery.imaging.streak_tracklets import (
                        ingest_streaks)
                    ny_s, nx_s = ccd.science.shape
                    frame_diag = math.hypot(nx_s, ny_s)
                    streaks = detect_streaks(
                        ccd.science, sigma_threshold=args.detect_sigma,
                        min_length_px=args.streak_min_length_px,
                        max_width_px=6.0)
                    s_det_ids, s_trk_ids = ingest_streaks(
                        db, streaks, ccd.wcs,
                        image_id=f"{path.stem}_ccd{ccd.ccdnum}",
                        exposure_start_mjd=ccd.mjd,
                        exposure_seconds=inst.exptime_s,
                        pixel_scale_arcsec=args.pixel_scale_arcsec,
                        psf_fwhm_px=3.0, frame_diagonal_px=frame_diag,
                        zeropoint_mag=(ccd.magzero if ccd.magzero > 0
                                         else None),
                        asteroid_only=True, ccd_id=ccd.name)
                    if s_trk_ids:
                        # Streak tracklets join the night's tracklet pool
                        for tid in s_trk_ids:
                            t = db.get_tracklet(tid)
                            night_streak_tracklets.append({
                                "tracklet_id": tid,
                                "mean_mjd": t["mean_mjd"],
                                "mean_ra": t["mean_ra"],
                                "mean_dec": t["mean_dec"],
                                "rate_arcsec_hr": t["rate_arcsec_hr"],
                                "pa_deg": t["pa_deg"]})
                        print(f"        + {len(s_trk_ids)} streak "
                                f"tracklet(s) (fast movers)", flush=True)
                except Exception as exc:
                    print(f"        streak detection skipped: {exc}",
                            flush=True)

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
    # Streak tracklets (fast movers) are already persisted; add them to the
    # linker pool so single-exposure NEO trails can seed/extend chains.
    if night_streak_tracklets:
        print(f"    + {len(night_streak_tracklets)} streak tracklet(s) "
                f"added to linker pool", flush=True)
        night_tracklets.extend(night_streak_tracklets)

    # ============================================================
    # [DISC] Within-night DISCOVERY loop (>=3 exposures per night)
    #   tracklets -> remove knowns -> vet -> unknown candidate list,
    #   with a scrambled-control false-positive floor and per-candidate
    #   single-snapshot distance/class/incomer annotation.
    # ============================================================
    if not args.no_discovery:
        from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
        for night, det_rows in all_inserted_per_night.items():
            by_img: dict[str, list] = {}
            for _did, s in det_rows:
                by_img.setdefault(s.image_id.split("#")[0], []).append(s)
            if len(by_img) < 3:
                continue   # discovery linking needs >=3 same-night exposures
            epochs = []
            for img, srcs in by_img.items():
                epochs.append({
                    "ra": __import__("numpy").array([s.ra for s in srcs]),
                    "dec": __import__("numpy").array([s.dec for s in srcs]),
                    "mag": __import__("numpy").array([s.mag for s in srcs]),
                    "mjd": srcs[0].mjd})
            print(f"\n[DISC] discovery loop, night {night} "
                    f"({len(by_img)} exposures)", flush=True)
            try:
                res = run_discovery(epochs, db,
                                      observatory_code=args.observatory_code)
                floor = false_positive_floor(epochs, n_trials=3)
                print(f"    {res.note}", flush=True)
                print(f"    chance-alignment floor ~{floor:.1f} candidates "
                        f"(candidates are significant only above this)",
                        flush=True)
                obs_geo = observatory_geo_km(args.observatory_code,
                                               epochs[0]["mjd"])
                import math as _m
                for c in res.candidates[:10]:
                    a = _m.radians(c.mean_ra); d = _m.radians(c.mean_dec)
                    los = [_m.cos(d) * _m.cos(a), _m.cos(d) * _m.sin(a), _m.sin(d)]
                    et = (c.mean_mjd - 51544.5) * 86400.0
                    from ariadne.data.ephemeris import body_state
                    import numpy as _np
                    Rh = _np.array(body_state("EARTH", et, "J2000", "SUN")[:3]) \
                        + (obs_geo if obs_geo is not None else 0)
                    post = snapshot_posterior(c.rate_arcsec_hr, c.mag, Rh,
                                                _np.array(los), n=4000)
                    tag = " INCOMER!" if post.incomer_flag else ""
                    print(f"      cand ({c.mean_ra:.4f},{c.mean_dec:.4f}) "
                            f"mag={c.mag:.1f} rate={c.rate_arcsec_hr:.0f}\"/hr "
                            f"dist={post.distance_med:.1f}AU "
                            f"[{post.distance_lo:.1f},{post.distance_hi:.1f}] "
                            f"{c.orbit_class}{tag}", flush=True)
            except Exception as exc:
                print(f"    discovery loop skipped: {exc}", flush=True)

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
            limit_known=None,
            observatory_code=args.observatory_code)
        print(f"    flagged {n_flagged} detections as KNOWN "
                f"(target_mjd={target_mjd:.4f}, obs={args.observatory_code}, "
                f"accurate N-body+topocentric ephemeris, "
                f"wall {time.time()-t0:.1f}s)", flush=True)

        # [C-check] Ephemeris self-validation gate (best-effort, network-
        # gated): confirm the predicted positions of a few bright numbered
        # in-field asteroids match JPL Horizons. This is the guard that
        # would have caught the ecliptic-vs-equatorial frame bug instead of
        # silently flagging 0. Skips quietly if offline.
        if not args.no_self_check:
            try:
                obs_geo = observatory_geo_km(args.observatory_code, target_mjd)
                d_ra = [d["ra"] for d in db.query_detections_by_cone(
                    mjd_range=(target_mjd - 0.5, target_mjd + 0.5),
                    ra_range=None, dec_range=None, limit=50000)]
                cur = db.conn.cursor()
                rows = cur.execute(
                    "SELECT designation, epoch_mjd, orbital_elements FROM "
                    "known_objects WHERE designation GLOB "
                    "'[0-9][0-9][0-9][0-9][0-9]' LIMIT 4000").fetchall()
                recs = []
                for r in rows[:4000]:
                    try:
                        el = _json.loads(r["orbital_elements"])
                        recs.append(OrbitalElements(
                            designation=r["designation"],
                            epoch_mjd=float(r["epoch_mjd"]), a_au=el["a_au"],
                            e=el["e"], i_deg=el["i_deg"], Omega_deg=el["Omega_deg"],
                            omega_deg=el["omega_deg"], M_deg=el["M_deg"],
                            H_mag=el.get("H", 0.0)))
                    except Exception:
                        continue
                # pick a few numbered objects actually in the field
                from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
                infield = pick_orbits_in_field(
                    db, target_mjd,
                    (min(d_ra), max(d_ra)) if d_ra else (0, 360),
                    (-90, 90), max_mag=20.0, limit_candidates=1600000)
                numbered = [x[0] for x in infield
                            if str(x[0].designation).isdigit()][:6]
                if numbered:
                    rep = validate_against_horizons(
                        numbered, target_mjd, observer_code=args.observatory_code,
                        observer_geo_km=obs_geo, tol_arcsec=5.0)
                    flag = "PASS" if rep.passed else "*** FAIL ***"
                    print(f"    [self-check] ephemeris vs Horizons: "
                            f"median {rep.median_offset_arcsec:.1f}\" over "
                            f"{rep.n_checked} numbered objects -> {flag}",
                            flush=True)
                    if not rep.passed:
                        print(f"      WARNING: cross-match may be miscalibrated "
                                f"-- {rep.diagnosis}", flush=True)
            except Exception as exc:
                print(f"    [self-check] skipped ({str(exc)[:60]})", flush=True)

        # [FAST] Post-cross-match VERY-FAST-MOVER screen. Calibrated on real
        # DECam: the per-object single-snapshot trail rate is noise-limited
        # below ~500"/hr (17% of stars scatter >100"/hr; 0% scatter >500"/hr),
        # so only rate > FAST_RATE flagged, on UNMATCHED + high-SNR sources.
        # These are genuine fast movers (NEOs/inner) that the catalog misses;
        # complements the streak detector. NOTE: the no-trail "incomer" case
        # is NOT screenable from one frame -- a stationary incomer is
        # indistinguishable from a star; that needs star-catalog removal +
        # a confirming second epoch.
        FAST_RATE = 500.0   # "/hr -- the clean per-object threshold (0% FP)
        if args.single_snapshot_rates and snapshot_rate:
            cur = db.conn.cursor()
            n_fast = 0; n_checked = 0
            for det_id, (rate, snr) in snapshot_rate.items():
                row = cur.execute(
                    "SELECT ra, dec, mag, known_designation FROM detections "
                    "WHERE id=?", (det_id,)).fetchone()
                if row is None or row["known_designation"] or snr < 20:
                    continue
                n_checked += 1
                if rate >= FAST_RATE:
                    n_fast += 1
                    print(f"    !! FAST-MOVER candidate ({row['ra']:.4f},"
                            f"{row['dec']:.4f}) mag={row['mag']:.1f} "
                            f"rate={rate:.0f}\"/hr snr={snr:.0f}", flush=True)
            print(f"    [fast-mover screen] {n_fast} flagged from {n_checked} "
                    f"unmatched high-SNR sources (rate>={FAST_RATE:.0f}\"/hr)",
                    flush=True)
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
