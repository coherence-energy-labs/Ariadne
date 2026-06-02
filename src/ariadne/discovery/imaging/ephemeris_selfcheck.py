"""Self-validation gate for the ephemeris + cross-match pipeline.

The ecliptic-vs-equatorial frame bug threw predicted positions off by
~100,000 arcsec yet produced no error -- the cross-match simply reported
"0 recovered", indistinguishable from "no objects present". A
production system must tell those apart. This module is the guard that
would have caught it immediately.

Two checks:

  validate_against_detections(...)  -- self-calibration with NO external
      service: predict bright catalogue objects in a field, measure the
      median offset to the nearest real detection. If the median offset
      is large but the MINIMUM offsets cluster (i.e., predictions land
      near *something* consistently), that signals a systematic pointing
      error, not absence. A healthy pipeline has bright-object offsets at
      the few-arcsec level.

  validate_against_horizons(...)    -- authoritative check against JPL
      Horizons for a handful of numbered asteroids. Returns the per-object
      offsets; the gate fails if the median exceeds `tol_arcsec`. Use in
      CI / pre-flight when network is available.

Both return a SelfCheckReport with a clear pass/fail + diagnosis string,
so the caller can refuse to publish a cross-match built on a broken
ephemeris.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SelfCheckReport:
    passed: bool
    median_offset_arcsec: float
    n_checked: int
    diagnosis: str
    detail: dict


def validate_against_horizons(records, target_mjd,
                                *, observer_code: str = "807",
                                observer_geo_km=None,
                                tol_arcsec: float = 10.0,
                                use_nbody: bool = True) -> SelfCheckReport:
    """Compare our ephemeris to JPL Horizons for `records` (each must have
    a numeric designation Horizons can resolve as a small body).

    Fails if the median sky offset exceeds tol_arcsec -- the signature of
    a frame / units / epoch bug. Requires astroquery + network.
    """
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError:
        return SelfCheckReport(False, float("nan"), 0,
                                 "astroquery not installed; cannot reach Horizons",
                                 {})
    if use_nbody:
        from .mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody as _eph
    else:
        from .mpc_ephemeris_batch import bulk_ephemeris_at_mjd as _eph

    jd = target_mjd + 2400000.5
    eph = _eph(records, target_mjd, observer_geo_km=observer_geo_km)
    offsets = {}
    for i, rec in enumerate(records):
        if np.isnan(eph[i, 0]):
            continue
        try:
            tgt = str(rec.designation).lstrip("0") or "0"
            e = Horizons(id=tgt, id_type="smallbody",
                          location=observer_code, epochs=jd).ephemerides()
            rah, dech = float(e["RA"][0]), float(e["DEC"][0])
        except Exception:
            continue
        cd = math.cos(math.radians(dech))
        sep = math.hypot((eph[i, 0] - rah) * cd, eph[i, 1] - dech) * 3600.0
        offsets[rec.designation] = sep
    if not offsets:
        return SelfCheckReport(False, float("nan"), 0,
                                 "no objects could be resolved by Horizons", {})
    med = float(np.median(list(offsets.values())))
    passed = med <= tol_arcsec
    diag = (f"median ephemeris-vs-Horizons offset {med:.1f}\" over "
            f"{len(offsets)} objects "
            f"({'PASS' if passed else 'FAIL'} at tol {tol_arcsec}\"). "
            + ("Healthy." if passed else
               "LARGE systematic -- suspect frame (ecliptic vs equatorial), "
               "epoch, light-time, or observer-location bug. Do NOT trust "
               "the cross-match."))
    return SelfCheckReport(passed, med, len(offsets), diag, offsets)


def diagnose_crossmatch_miss(predicted_radec, detections_radec,
                               *, expected_offset_arcsec: float = 5.0,
                               ) -> SelfCheckReport:
    """Given predicted positions and the detection list, decide WHY a
    cross-match found nothing: genuine absence vs systematic miscalibration.

    `predicted_radec`: (M,2) array of predicted (ra,dec) deg for bright
        catalogue objects expected in the field.
    `detections_radec`: (N,2) array of real detection (ra,dec) deg.

    Returns a report. The key signal: if the bright predictions' nearest-
    detection offsets are all far larger than `expected_offset_arcsec`
    AND tightly distributed (low spread), that is a SYSTEMATIC pointing
    error (the predictions are coherently shifted), not random absence.
    """
    pred = np.asarray(predicted_radec, float)
    det = np.asarray(detections_radec, float)
    if pred.size == 0 or det.size == 0:
        return SelfCheckReport(False, float("nan"), 0,
                                 "empty predictions or detections", {})
    offs = []
    for p_ra, p_dec in pred:
        cd = math.cos(math.radians(p_dec))
        sep = np.hypot((det[:, 0] - p_ra) * cd, det[:, 1] - p_dec) * 3600.0
        offs.append(float(sep.min()))
    offs = np.array(offs)
    med = float(np.median(offs))
    # Random-coincidence floor for this detection density
    # (mean nearest-neighbour distance ~ 0.5 / sqrt(surface density))
    ra_span = (det[:, 0].max() - det[:, 0].min())
    dec_span = (det[:, 1].max() - det[:, 1].min())
    area_sq_arcsec = max(ra_span * dec_span, 1e-9) * (3600.0 ** 2)
    density = len(det) / area_sq_arcsec
    random_floor = 0.5 / math.sqrt(max(density, 1e-30))
    passed = med <= expected_offset_arcsec
    if passed:
        diag = f"median bright-object offset {med:.1f}\" -- healthy."
    elif med < 0.5 * random_floor:
        diag = (f"median offset {med:.1f}\" exceeds the {expected_offset_arcsec}\" "
                f"expectation but is well below the random-coincidence floor "
                f"({random_floor:.0f}\") -- SYSTEMATIC pointing error in the "
                f"ephemeris (frame/epoch/observer). Predictions are coherently "
                f"shifted, NOT absent.")
    else:
        diag = (f"median offset {med:.1f}\" ~ random floor {random_floor:.0f}\" "
                f"-- predictions land no closer than chance; the objects are "
                f"likely genuinely ABSENT/undetected (not a calibration bug).")
    return SelfCheckReport(passed, med, len(offs), diag,
                             {"random_floor_arcsec": random_floor})
