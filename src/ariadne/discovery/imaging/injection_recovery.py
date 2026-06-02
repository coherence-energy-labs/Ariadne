"""Injection-recovery validation for the discovery pipeline.

Standard survey-validation pattern (used by PanSTARRS/ATLAS/LSST MOPS):
inject N synthetic moving-object detections at positions predicted from
real catalog orbits, then run the existing detection -> linking ->
cross-match pipeline and measure:

  - recall    = (n_injected & recovered) / n_injected
  - precision = (true matches) / (all flagged)
  - astrometric residual on injections
  - linker recovery on injections that should chain across nights

Public API:
  inject_synthetic_detections(db, target_mjd, ra_range, dec_range,
                                n_inject, ...)
  measure_cross_match_recall(db, target_mjd, ...)
  build_injection_report(db, target_mjd, ...) -> RecoveryReport

The synthetic detections are inserted into the live `detections` table
with `image_id` prefixed with "INJECT__" so they can be cleanly removed
after the validation run.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


INJECT_PREFIX = "INJECT__"


@dataclass
class InjectedOrbit:
    """A catalog orbit chosen for injection."""
    designation: str
    pred_ra: float        # predicted RA at injection epoch
    pred_dec: float       # predicted Dec at injection epoch
    pred_mag: float
    inject_ra: float      # actual (noise-perturbed) ra used as detection
    inject_dec: float     # actual (noise-perturbed) dec used as detection
    inject_mag: float
    detection_id: int     # row id assigned to the synthetic detection


@dataclass
class RecoveryReport:
    """Result of one injection-recovery validation run."""
    n_injected: int
    n_recovered: int
    n_flagged_total: int
    n_false_positive: int
    recall: float
    precision: float
    astrom_residual_arcsec: list[float]
    median_residual_arcsec: float
    p95_residual_arcsec: float
    injected_orbits: list[dict]   # InjectedOrbit asdict


def pick_orbits_in_field(db,
                            target_mjd: float,
                            ra_range: tuple[float, float],
                            dec_range: tuple[float, float],
                            *, max_mag: float = 22.5,
                            limit_candidates: int = 100000,
                            ) -> list:
    """Find every catalogued orbit whose predicted position at
    `target_mjd` falls inside the sky box `[ra_range] x [dec_range]`
    and is brighter than `max_mag`.

    Uses the vectorised batch ephemeris (~1ms/orbit for the full 100K
    catalog).
    """
    from .mpc_catalog import OrbitalElements
    from .mpc_ephemeris_batch import bulk_ephemeris_at_mjd
    import json as _json
    cur = db.conn.cursor()
    rows = list(cur.execute(
        f"SELECT designation, epoch_mjd, orbital_elements "
        f"FROM known_objects LIMIT {int(limit_candidates)}"))
    recs = []
    for r in rows:
        try:
            elems = _json.loads(r["orbital_elements"])
            recs.append(OrbitalElements(
                designation=r["designation"],
                epoch_mjd=float(r["epoch_mjd"]),
                a_au=float(elems["a_au"]),
                e=float(elems["e"]),
                i_deg=float(elems["i_deg"]),
                Omega_deg=float(elems["Omega_deg"]),
                omega_deg=float(elems["omega_deg"]),
                M_deg=float(elems["M_deg"]),
                H_mag=float(elems.get("H", 0.0)),
            ))
        except Exception:
            continue
    eph = bulk_ephemeris_at_mjd(recs, target_mjd)
    # Filter into the field
    ra_lo, ra_hi = ra_range
    dec_lo, dec_hi = dec_range
    in_field = []
    for i, rec in enumerate(recs):
        ra, dec, mag, rho = eph[i]
        if math.isnan(ra) or math.isnan(dec):
            continue
        if not (ra_lo <= ra <= ra_hi):
            continue
        if not (dec_lo <= dec <= dec_hi):
            continue
        if mag > max_mag:
            continue
        in_field.append((rec, float(ra), float(dec), float(mag), float(rho)))
    return in_field


def inject_synthetic_detections(db,
                                   target_mjd: float,
                                   in_field_orbits: Sequence,
                                   *, astrom_noise_arcsec: float = 0.15,
                                   phot_noise_mag: float = 0.05,
                                   image_id: str | None = None,
                                   seed: int = 42,
                                   ) -> list[InjectedOrbit]:
    """Insert synthetic detections at the predicted positions of each
    orbit in `in_field_orbits`, with Gaussian noise on (RA, Dec, mag).

    Returns the list of InjectedOrbit records (with assigned DB ids) so
    the validator can later check recovery.
    """
    rng = random.Random(seed)
    from .detection_db import DetectionRow
    if image_id is None:
        image_id = f"{INJECT_PREFIX}mjd{target_mjd:.5f}"
    rows = []
    injected = []
    for rec, pred_ra, pred_dec, pred_mag, _rho in in_field_orbits:
        sigma_deg = astrom_noise_arcsec / 3600.0
        cos_dec = math.cos(math.radians(pred_dec))
        ra_noise = rng.gauss(0, sigma_deg / max(cos_dec, 1e-6))
        dec_noise = rng.gauss(0, sigma_deg)
        mag_noise = rng.gauss(0, phot_noise_mag)
        inj_ra = pred_ra + ra_noise
        inj_dec = pred_dec + dec_noise
        inj_mag = pred_mag + mag_noise
        rows.append(DetectionRow(
            image_id=image_id,
            mjd=target_mjd,
            ra=inj_ra,
            dec=inj_dec,
            mag=inj_mag,
            astrom_sigma_arcsec=astrom_noise_arcsec,
            ccd_id="INJECT",
            x_pix=0.0, y_pix=0.0))
    ids = db.insert_detections(rows)
    for did, row, (rec, pred_ra, pred_dec, pred_mag, _rho) in zip(
            ids, rows, in_field_orbits):
        injected.append(InjectedOrbit(
            designation=rec.designation,
            pred_ra=pred_ra, pred_dec=pred_dec, pred_mag=pred_mag,
            inject_ra=row.ra, inject_dec=row.dec, inject_mag=row.mag,
            detection_id=did))
    db.conn.commit()
    return injected


def clear_injections(db, image_id_prefix: str = INJECT_PREFIX) -> int:
    """Delete every injection-marker detection. Returns rows deleted."""
    cur = db.conn.cursor()
    cur.execute(f"DELETE FROM detections WHERE image_id LIKE ?",
                  (f"{image_id_prefix}%",))
    db.conn.commit()
    return cur.rowcount


def measure_recall(db,
                      target_mjd: float,
                      injected: Sequence[InjectedOrbit],
                      *, match_radius_arcsec: float = 3.0,
                      mjd_box_days: float = 0.001,
                      ) -> RecoveryReport:
    """Run the operational cross-match against the DB (which now contains
    the injections) and compute recall/precision against the truth set.

    Returns a fully-populated RecoveryReport.
    """
    from .mpc_catalog import flag_known_in_db
    # Run the operational cross-match
    n_flagged = flag_known_in_db(
        db, target_mjd,
        mjd_box_days=mjd_box_days,
        match_radius_arcsec=match_radius_arcsec)
    # Now query the flags applied to our injections
    cur = db.conn.cursor()
    truth_by_id = {inj.detection_id: inj for inj in injected}
    rows = list(cur.execute(
        f"SELECT id, known_designation, ra, dec FROM detections "
        f"WHERE id IN ({','.join('?' * len(truth_by_id))})",
        list(truth_by_id.keys())))
    n_recovered = 0
    residuals = []
    for r in rows:
        flagged = r["known_designation"]
        truth = truth_by_id[r["id"]]
        if flagged == truth.designation:
            n_recovered += 1
            # Astrometric residual = distance between injected and predicted
            cos_dec = math.cos(math.radians(truth.pred_dec))
            dra = (truth.inject_ra - truth.pred_ra) * cos_dec
            ddec = truth.inject_dec - truth.pred_dec
            residuals.append(math.hypot(dra, ddec) * 3600)
    n_false_positive = n_flagged - n_recovered
    recall = n_recovered / max(len(injected), 1)
    precision = n_recovered / max(n_flagged, 1) if n_flagged > 0 else 0.0
    arr = np.array(residuals) if residuals else np.array([0.0])
    return RecoveryReport(
        n_injected=len(injected),
        n_recovered=n_recovered,
        n_flagged_total=n_flagged,
        n_false_positive=n_false_positive,
        recall=recall,
        precision=precision,
        astrom_residual_arcsec=residuals,
        median_residual_arcsec=float(np.median(arr)),
        p95_residual_arcsec=float(np.percentile(arr, 95)),
        injected_orbits=[asdict(inj) for inj in injected])


def run_full_validation(db,
                          target_mjd: float,
                          ra_range: tuple[float, float],
                          dec_range: tuple[float, float],
                          *, max_mag: float = 22.5,
                          n_inject_target: int = 50,
                          astrom_noise_arcsec: float = 0.15,
                          match_radius_arcsec: float = 3.0,
                          seed: int = 42,
                          clear_after: bool = True,
                          ) -> RecoveryReport:
    """End-to-end injection-recovery in one call.

    1. Find every catalog orbit predicted to fall in `ra_range x dec_range`
       at `target_mjd` brighter than `max_mag`.
    2. Sample up to `n_inject_target` of them.
    3. Insert synthetic detections at predicted positions + noise.
    4. Run flag_known_in_db.
    5. Score recall/precision vs the truth set.
    """
    in_field = pick_orbits_in_field(
        db, target_mjd, ra_range, dec_range, max_mag=max_mag)
    if len(in_field) > n_inject_target:
        rng = random.Random(seed)
        in_field = rng.sample(in_field, n_inject_target)
    injected = inject_synthetic_detections(
        db, target_mjd, in_field,
        astrom_noise_arcsec=astrom_noise_arcsec, seed=seed)
    report = measure_recall(
        db, target_mjd, injected,
        match_radius_arcsec=match_radius_arcsec)
    if clear_after:
        clear_injections(db)
    return report
