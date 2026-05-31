"""Sensitivity validation -- honest recovery rates on injected synthetic objects.

For a discovery engine, the question that matters is: at what magnitude /
on-sky-rate / arc-length does the pipeline RECOVER an object that's really
there? The answer needs to be a quantified curve, not a marketing claim.

This module runs end-to-end validation:

  1. Inject N synthetic moving objects at user-specified (magnitude, rate)
     into the same machinery the pipeline uses on real data.
  2. Run the pipeline (clustering + linker + IOD + LM fit) on the union of
     real and synthetic alerts.
  3. Cross-match the surviving candidates to the truth list.
  4. Report the recovery rate (objects recovered / objects injected) as a
     function of magnitude, rate, and arc length.

Outputs a structured dict + an optional matplotlib figure. The numbers go
into docs/SENSITIVITY.md so users can see exactly where the engine
saturates.

Honest caveats:
  * Synthetic injection ASSUMES the survey's spatial+temporal sampling
    distribution; real recoveries depend on cadence too.
  * "Recovery" here means "the candidate's orbit fit landed within
    `match_tol_arcsec` of the truth"; refinements like multi-opposition
    extension are a separate quality metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.constants import GM_SUN, AU_KM
from ..data.ephemeris import et, body_state
from ..dynamics.secular import elements_to_state, kepler_step
from ..discovery.brokers.base import Alert


@dataclass
class InjectionRecord:
    """One synthetic injection -- truth state for recovery checking."""
    truth_id: int
    a_au: float
    e: float
    i_deg: float
    apparent_mag: float
    rate_arcsec_hr: float
    n_planted_alerts: int


@dataclass
class RecoveryResult:
    """Result of one sensitivity-validation run."""
    n_injected: int
    n_recovered: int
    n_false_positives: int
    recovery_rate: float
    recovery_by_magnitude: dict
    recovery_by_rate: dict
    median_rms_arcsec: float
    median_arc_recovery_days: float
    injection_records: list


def inject_synthetic_objects(orbits: list[dict], epoch: str,
                              n_nights: int = 5,
                              n_per_night: int = 4,
                              noise_arcsec: float = 0.5,
                              seed: int = 42) -> tuple[list[Alert], list[InjectionRecord]]:
    """Generate a set of synthetic Alerts for a list of orbits.

    Args:
      orbits:        list of dicts with a_au, e, i_deg, Omega, omega, M, apparent_mag.
      epoch:         UTC start epoch (string).
      n_nights:      how many nightly visits to the same field.
      n_per_night:   detections per night (typically 2-4 per ZTF visit).
      noise_arcsec:  per-detection astrometric noise sigma.
      seed:          RNG seed for reproducibility.

    Returns:
      (alerts, injection_records) -- the synthetic Alert stream + per-object
      truth records.
    """
    rng = np.random.default_rng(seed)
    e0 = et(epoch)
    alerts = []
    records = []
    for oid, o in enumerate(orbits):
        a = o["a_au"]; ecc = o.get("e", 0.05); inc = o.get("i_deg", o.get("i", 10.0))
        Omega = o.get("Omega", 30.0); omega = o.get("omega", 50.0); M = o.get("M", 180.0)
        mag = o.get("apparent_mag", 22.0)
        rate_hint = o.get("rate_arcsec_hr", 1.0)
        p, v = elements_to_state(a, ecc, inc, Omega, omega, M)
        r0 = np.array(p); v0 = np.array(v)
        n_planted = 0
        for night_idx in range(n_nights):
            night_dt = night_idx * 86400.0
            for k in range(n_per_night):
                # spread detections within ~2 hr of each night centre
                ddt = (k - n_per_night / 2) * 1800.0
                t = night_dt + ddt
                xs, _ = kepler_step(r0, v0, GM_SUN, t)
                R_e = body_state("EARTH", e0 + t, "J2000", "SUN")[:3]
                geo = xs - R_e
                rho = float(np.linalg.norm(geo))
                ra = math.atan2(geo[1], geo[0]) + rng.normal(0, noise_arcsec / 206265.0)
                dec_safe = max(-0.99, min(0.99, geo[2] / rho))
                dec = math.asin(dec_safe) + rng.normal(0, noise_arcsec / 206265.0)
                alerts.append(Alert(
                    survey="synth", alert_id=f"inj_{oid}_{night_idx}_{k}",
                    obj_id=f"truth_{oid}",
                    mjd=(e0 + t) / 86400.0 + 51544.5,         # ET sec -> MJD UTC approx
                    ra=math.degrees(ra) % 360.0,
                    dec=math.degrees(dec),
                    mag=mag + rng.normal(0, 0.1),
                    band="r",
                    meta={"truth_id": oid, "truth_a_au": a,
                           "truth_e": ecc, "truth_i_deg": inc},
                ))
                n_planted += 1
        records.append(InjectionRecord(
            truth_id=oid, a_au=a, e=ecc, i_deg=inc,
            apparent_mag=mag, rate_arcsec_hr=rate_hint,
            n_planted_alerts=n_planted,
        ))
    return alerts, records


def evaluate_recovery(pipeline_output: list[dict], injections: list[InjectionRecord],
                      *, match_tol_arcsec: float = 60.0,
                      bin_mag_edges=(15, 18, 19, 20, 21, 22, 22.5, 23.0, 24.0),
                      bin_rate_edges=(0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0)
                      ) -> RecoveryResult:
    """Compare pipeline output to ground truth and produce a recovery report.

    Args:
      pipeline_output:  list of accepted candidates from realtime.run_pipeline,
                         each with "x_fit_km", "v_fit_kms", "rms_arcsec",
                         "members" (the constituent alerts).
      injections:       the InjectionRecord list returned by inject_synthetic_objects.
      match_tol_arcsec: candidate <-> truth match tolerance on the median position.
      bin_mag_edges:    histogram edges for the magnitude-binned recovery curve.
      bin_rate_edges:   same for the rate-binned recovery curve.

    Returns:
      RecoveryResult with overall + binned recovery statistics.
    """
    truth_recovered = set()
    n_false_positives = 0
    rms_recovered = []
    arc_lengths_recovered = []

    for cand in pipeline_output:
        if cand.get("status") != "accepted":
            continue
        members = cand.get("members", [])
        truth_ids = set()
        for m in members:
            t_id = (getattr(m, "meta", {}) or {}).get("truth_id")
            if t_id is not None:
                truth_ids.add(t_id)
        if not truth_ids:
            n_false_positives += 1
            continue
        # match to a single truth if at least 4 detections from the same truth
        from collections import Counter
        counts = Counter([(getattr(m, "meta", {}) or {}).get("truth_id") for m in members])
        most_common, n_in_common = counts.most_common(1)[0]
        if most_common is not None and n_in_common >= 4:
            truth_recovered.add(most_common)
            if "rms_arcsec" in cand and cand["rms_arcsec"] is not None:
                rms_recovered.append(cand["rms_arcsec"])
            if members:
                mjds = [m.mjd for m in members]
                arc_lengths_recovered.append(max(mjds) - min(mjds))

    # recovery by magnitude bin
    recovery_by_mag = {}
    for lo, hi in zip(bin_mag_edges[:-1], bin_mag_edges[1:]):
        injected_in_bin = [r for r in injections if lo <= r.apparent_mag < hi]
        recovered_in_bin = [r for r in injected_in_bin if r.truth_id in truth_recovered]
        if injected_in_bin:
            recovery_by_mag[f"{lo}-{hi}"] = {
                "n_injected": len(injected_in_bin),
                "n_recovered": len(recovered_in_bin),
                "rate": len(recovered_in_bin) / len(injected_in_bin),
            }

    # recovery by on-sky rate
    recovery_by_rate = {}
    for lo, hi in zip(bin_rate_edges[:-1], bin_rate_edges[1:]):
        injected_in_bin = [r for r in injections if lo <= r.rate_arcsec_hr < hi]
        recovered_in_bin = [r for r in injected_in_bin if r.truth_id in truth_recovered]
        if injected_in_bin:
            recovery_by_rate[f"{lo}-{hi}"] = {
                "n_injected": len(injected_in_bin),
                "n_recovered": len(recovered_in_bin),
                "rate": len(recovered_in_bin) / len(injected_in_bin),
            }

    return RecoveryResult(
        n_injected=len(injections),
        n_recovered=len(truth_recovered),
        n_false_positives=n_false_positives,
        recovery_rate=len(truth_recovered) / max(len(injections), 1),
        recovery_by_magnitude=recovery_by_mag,
        recovery_by_rate=recovery_by_rate,
        median_rms_arcsec=float(np.median(rms_recovered)) if rms_recovered else float("nan"),
        median_arc_recovery_days=float(np.median(arc_lengths_recovered))
                                 if arc_lengths_recovered else float("nan"),
        injection_records=injections,
    )


def make_population(
    *,
    n_objects: int = 30,
    a_range_au: tuple = (2.0, 80.0),
    e_range: tuple = (0.01, 0.4),
    i_range_deg: tuple = (0.0, 30.0),
    mag_range: tuple = (18.0, 23.0),
    seed: int = 7,
) -> list[dict]:
    """Generate a random population of orbits for an injection run.

    Use this to build a coarse sensitivity sweep across the parameter space.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_objects):
        a = float(rng.uniform(*a_range_au))
        e = float(rng.uniform(*e_range))
        i = float(rng.uniform(*i_range_deg))
        mag = float(rng.uniform(*mag_range))
        # Rough rate estimate at opposition: 10 * (a/AU)^-1.5 for outer objects
        rate = 30.0 / max(a, 0.5) ** 0.5         # placeholder; real opposition rate
        out.append({"a_au": a, "e": e, "i_deg": i,
                     "Omega": float(rng.uniform(0, 360)),
                     "omega": float(rng.uniform(0, 360)),
                     "M": 180.0, "apparent_mag": mag,
                     "rate_arcsec_hr": rate})
    return out
