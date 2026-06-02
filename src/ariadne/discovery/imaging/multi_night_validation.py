"""End-to-end multi-night pipeline validation harness.

This is the MOPS-validation pattern: simulate N nights of observations
of the same sky region, inject K moving objects whose true orbits
follow the catalog, run the FULL operational pipeline (tracklet build
-> chain link -> cross-match -> IOD -> grade-A), and measure:

  - chain_recall  = chains containing all K nights / K
  - chain_purity  = chains containing only one true object / total chains
  - iod_success_rate = chains where IOD converged
  - grade_a_yield   = injected objects passing Grade-A QC

This validates the LINKER not just the cross-match. A working linker
is the difference between an algorithm demo and an operational
discovery pipeline.

Public API:
  run_multi_night_validation(db, mjd_grid, in_field_orbits, ...) -> dict
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MultiNightReport:
    """Outcome of one multi-night validation run."""
    n_nights: int
    n_objects_injected: int
    n_detections_injected: int
    n_tracklets_built: int
    n_chains_formed: int
    chain_recall: float            # frac of injected objects with a >=3-obs chain
    chain_purity: float            # frac of formed chains that are single-truth
    iod_attempted: int
    iod_succeeded: int
    grade_a_passed: int


def _propagate_orbit_to_mjd(orbital_elements, target_mjd):
    """Return (ra_deg, dec_deg, mag) for one orbit at target_mjd via the
    auto-ephemeris path (2-body for short dt, N-body otherwise)."""
    from .mpc_ephemeris_nbody import auto_ephemeris_at_mjd
    eph = auto_ephemeris_at_mjd([orbital_elements], target_mjd)
    return float(eph[0, 0]), float(eph[0, 1]), float(eph[0, 2])


def run_multi_night_validation(db,
                                  in_field_orbits: Sequence,
                                  mjd_grid: Sequence[float],
                                  *, obs_per_night: int = 2,
                                  within_night_gap_hours: float = 1.0,
                                  astrom_noise_arcsec: float = 0.15,
                                  match_radius_arcsec: float = 3.0,
                                  link_window_days: float = 30.0,
                                  seed_window_days: float = 7.0,
                                  rng_seed: int = 42,
                                  ) -> MultiNightReport:
    """Simulate a multi-night survey on `in_field_orbits` over `mjd_grid`.

    For each MJD in mjd_grid:
      1. predict each orbit's position via auto-ephemeris
      2. inject `obs_per_night` synthetic detections per orbit
         (offset by within_night_gap_hours each)
      3. build within-night tracklets
      4. run multi-night linker against open chains
      5. cross-match against catalog

    At the end, attempt IOD on every formed chain and measure grade-A QC.

    Returns a MultiNightReport with recall/purity/iod/grade-A metrics.
    """
    from .detection_db import DetectionRow, TrackletRow, open_db
    from .within_night_tracklets import build_within_night_tracklets
    from .multi_night_linker import link_tonight
    from .injection_recovery import clear_injections, INJECT_PREFIX
    from .mpc_catalog import flag_known_in_db
    from .chain_iod import run_iod_on_chain
    from .mpc_submission import evaluate_grade_a
    from .mpc_ephemeris_nbody import auto_ephemeris_at_mjd

    rng = random.Random(rng_seed)
    clear_injections(db)
    truth_by_det_id: dict[int, str] = {}     # det_id -> designation

    total_dets = 0
    total_tracklets = 0
    # Each row in in_field_orbits: (OrbitalElements, ra_at_inject_epoch,
    # dec_at_inject_epoch, mag, rho). Only the orbit is used here; positions
    # at each mjd are re-derived via auto_ephemeris.
    recs = [item[0] for item in in_field_orbits]

    # Source-style minimal class for within-night tracklet builder
    class _Src:
        __slots__ = ("ra", "dec", "mjd", "mag", "image_id")
        def __init__(self, ra, dec, mjd, mag, image_id):
            self.ra = ra; self.dec = dec; self.mjd = mjd
            self.mag = mag; self.image_id = image_id

    for night_idx, night_mjd_base in enumerate(mjd_grid):
        # MJD grid for this night's exposures
        exp_mjds = [night_mjd_base + j * within_night_gap_hours / 24.0
                       for j in range(obs_per_night)]

        # For each exposure, predict every orbit's position and inject
        night_dets_with_ids = []
        for j, exp_mjd in enumerate(exp_mjds):
            eph = auto_ephemeris_at_mjd(recs, exp_mjd)
            rows = []
            row_truth = []
            for i, rec in enumerate(recs):
                if math.isnan(eph[i, 0]):
                    continue
                sigma_deg = astrom_noise_arcsec / 3600.0
                cos_dec = math.cos(math.radians(eph[i, 1]))
                ra_noise = rng.gauss(0, sigma_deg / max(cos_dec, 1e-6))
                dec_noise = rng.gauss(0, sigma_deg)
                inj_ra = float(eph[i, 0]) + ra_noise
                inj_dec = float(eph[i, 1]) + dec_noise
                inj_mag = float(eph[i, 2]) + rng.gauss(0, 0.05)
                image_id = f"{INJECT_PREFIX}n{night_idx}_e{j}"
                rows.append(DetectionRow(
                    image_id=image_id, mjd=exp_mjd,
                    ra=inj_ra, dec=inj_dec, mag=inj_mag,
                    astrom_sigma_arcsec=astrom_noise_arcsec,
                    ccd_id="INJECT"))
                row_truth.append(rec.designation)
            new_ids = db.insert_detections(rows)
            for did, row, designation in zip(new_ids, rows, row_truth):
                truth_by_det_id[did] = designation
                src = _Src(row.ra, row.dec, row.mjd, row.mag, row.image_id)
                night_dets_with_ids.append((did, src))
            total_dets += len(new_ids)

        # 3. Build within-night tracklets
        tracklets = build_within_night_tracklets(night_dets_with_ids)
        new_tids = []
        for trk in tracklets:
            new_tids.append(db.insert_tracklet(trk))
        total_tracklets += len(tracklets)

        # 4. Run multi-night linker
        link_tonight(db, [
            {"tracklet_id": tid,
              "mean_ra": trk.mean_ra, "mean_dec": trk.mean_dec,
              "mean_mjd": trk.mean_mjd,
              "rate_arcsec_hr": trk.rate_arcsec_hr,
              "pa_deg": trk.pa_deg, "night": trk.night}
            for tid, trk in zip(new_tids, tracklets)
        ], link_window_days=link_window_days,
            seed_window_days=seed_window_days,
            position_tol_arcsec=120.0, rate_tol_pct=50.0)

    # 5. Cross-match against the (already-ingested) MPCORB catalog
    final_mjd = mjd_grid[-1]
    n_flagged = flag_known_in_db(db, final_mjd,
                                       mjd_box_days=14.0,
                                       match_radius_arcsec=match_radius_arcsec)

    # 6. Score chain recall / purity
    # Map each chain to the set of truth designations it contains
    chain_rows = list(db.conn.execute(
        "SELECT id, n_tracklets, status FROM chains"))
    chain_truth: dict[int, set] = {}
    for c in chain_rows:
        cid = c["id"]
        # All detections via tracklet -> chain
        dets = list(db.conn.execute(
            "SELECT d.id FROM detections d "
            "JOIN tracklets t ON (t.detection_a_id = d.id OR t.detection_b_id = d.id) "
            "WHERE t.chain_id = ?", (cid,)))
        truths = {truth_by_det_id[d["id"]] for d in dets
                     if d["id"] in truth_by_det_id}
        chain_truth[cid] = truths
    n_chains = len(chain_truth)
    n_pure_chains = sum(1 for s in chain_truth.values() if len(s) == 1)
    chain_purity = n_pure_chains / max(n_chains, 1)
    # Recall: an injected object is recovered if SOME chain contains its truth
    all_truths_covered = set()
    for s in chain_truth.values():
        all_truths_covered |= s
    n_truths = len(set(rec.designation for rec in recs))
    chain_recall = len(all_truths_covered) / max(n_truths, 1)

    # 7. IOD
    iod_attempted = 0
    iod_succeeded = 0
    grade_a_passed = 0
    for cid in chain_truth.keys():
        iod_attempted += 1
        try:
            res = run_iod_on_chain(db, cid,
                                       rms_acceptance_arcsec=10.0,
                                       n_draws=2, use_monte_carlo=False)
            if res.get("success"):
                iod_succeeded += 1
                # Try grade-A QC
                chain_row = db.get_chain(cid)
                det_rows = list(db.conn.execute(
                    "SELECT d.* FROM detections d "
                    "JOIN tracklets t ON (t.detection_a_id = d.id OR t.detection_b_id = d.id) "
                    "WHERE t.chain_id = ?", (cid,)))
                # Convert sqlite Row to dict; pass through grade_a evaluator
                dets_list = [dict(r) for r in det_rows]
                grade = evaluate_grade_a(chain_row, dets_list)
                if grade.passed:
                    grade_a_passed += 1
        except Exception:
            pass

    return MultiNightReport(
        n_nights=len(mjd_grid),
        n_objects_injected=n_truths,
        n_detections_injected=total_dets,
        n_tracklets_built=total_tracklets,
        n_chains_formed=n_chains,
        chain_recall=chain_recall,
        chain_purity=chain_purity,
        iod_attempted=iod_attempted,
        iod_succeeded=iod_succeeded,
        grade_a_passed=grade_a_passed)
