"""Large synthetic labelled corpus for inference calibration.

Generates 200-500 realistic LabelledCase records spanning every orbital
class the inference engine knows about, with realistic per-class evidence
distributions (rate, magnitude, morphology, arc). The corpus is
deterministic (seeded RNG) so calibration runs are reproducible.

Per-class evidence distributions (drawn from the same rate/mag windows
inference.EXPECTED_RATES and inference._magnitude_log_likelihood use, so
the calibrator sees in-distribution data):

  MBA / IMB / OMB    rate 5-25"/hr, mag 17-22, point morphology
  HILDA / JTROJAN    rate 2-8"/hr, mag 16-22, point
  CENTAUR            rate 1-5"/hr, mag 18-23
  Classical / Hot KBO rate 0.3-3"/hr, mag 19-25
  RESONANT / SCATTERED rate 0.4-3.5, mag 19-25
  DETACHED / SEDNOID rate 0.05-1, mag 21-25
  APOLLO / ATEN      rate 15-600"/hr, mag 10-20
  AMOR / MARS_CROSSER rate 10-60"/hr, mag 15-22
  COMET_HYPERBOLIC   rate 1-200"/hr, mag 14-22

Adversarial / artefact cases (~20% of corpus):
  cosmic_ray         morphology COSMIC_RAY, n_detections=1
  satellite_trail    rate >1000"/hr, morphology STREAK
  stellar_variable   rate ~0, stationary
  subtraction_residual high RMS, n_detections=1
"""
from __future__ import annotations

import random

from .benchmarking import LabelledCase
from .inference import Evidence


_CLASS_SPECS = {
    "MBA":           {"rate": (5, 25), "mag": (17, 22), "morph": "POINT"},
    "IMB":           {"rate": (8, 30), "mag": (17, 22), "morph": "POINT"},
    "OMB":           {"rate": (3, 18), "mag": (17, 22), "morph": "POINT"},
    "HILDA":         {"rate": (2, 8),  "mag": (16, 22), "morph": "POINT"},
    "JTROJAN":       {"rate": (2, 6),  "mag": (16, 22), "morph": "POINT"},
    "CENTAUR":       {"rate": (1, 5),  "mag": (18, 23), "morph": "POINT"},
    "CLASSICAL_KBO": {"rate": (0.3, 3),"mag": (19, 25), "morph": "POINT"},
    "HOT_CLASSICAL": {"rate": (0.3, 3),"mag": (19, 25), "morph": "POINT"},
    "RESONANT_KBO":  {"rate": (0.4, 3.5),"mag": (19, 25), "morph": "POINT"},
    "SCATTERED_KBO": {"rate": (0.4, 3.5),"mag": (19, 25), "morph": "POINT"},
    "DETACHED":      {"rate": (0.2, 2),"mag": (21, 25), "morph": "POINT"},
    "SEDNOID":       {"rate": (0.05, 1),"mag": (21, 25), "morph": "POINT"},
    "APOLLO":        {"rate": (15, 600),"mag": (12, 20), "morph": "POINT"},
    "ATEN":          {"rate": (15, 600),"mag": (12, 20), "morph": "POINT"},
    "AMOR":          {"rate": (10, 120),"mag": (15, 22), "morph": "POINT"},
    "MARS_CROSSER":  {"rate": (8, 60), "mag": (16, 22), "morph": "POINT"},
}

_ARTEFACT_SPECS = {
    "cosmic_ray":           {"morph": "COSMIC_RAY", "rate": None,
                              "n_det": 1, "arc": 0},
    "satellite_trail":      {"morph": "STREAK", "rate": (3000, 8000),
                              "n_det": 1, "arc": 0},
    "stellar_variable":     {"morph": "POINT", "rate": (0.0, 0.05),
                              "n_det": 3, "arc": 5, "stationary": True},
    "subtraction_residual": {"morph": "POINT", "rate": (0.05, 2),
                              "n_det": 1, "arc": 0, "rms_high": True},
}


def make_large_corpus(n_per_class: int = 20, n_artefacts_per_kind: int = 8,
                      seed: int = 20260601) -> list[LabelledCase]:
    """Generate a deterministic large labelled corpus for calibration."""
    rng = random.Random(seed)
    cases = []
    case_id = 0

    # Moving objects
    for label, spec in _CLASS_SPECS.items():
        for _ in range(n_per_class):
            rate_lo, rate_hi = spec["rate"]
            mag_lo, mag_hi = spec["mag"]
            n_det = rng.randint(4, 12)
            arc_days = rng.uniform(3, 30)
            ev = Evidence(
                ra_deg=rng.uniform(0, 360),
                dec_deg=rng.uniform(-60, 60),
                rate_arcsec_hr=rng.uniform(rate_lo, rate_hi),
                apparent_mag=rng.uniform(mag_lo, mag_hi),
                band="r",
                morphology_label=spec["morph"],
                morphology_confidence=rng.uniform(0.75, 0.99),
                n_detections=n_det,
                arc_days=arc_days,
                rms_arcsec=rng.uniform(0.5, 3.0),
                skybot_match_names=[],
            )
            cases.append(LabelledCase(
                case_id=f"corp_{case_id:04d}",
                evidence=ev, truth_label=label,
                source="synthetic_class_spec",
                split="validation" if case_id % 5 != 0 else "holdout",
            ))
            case_id += 1

    # Artefacts
    for label, spec in _ARTEFACT_SPECS.items():
        for _ in range(n_artefacts_per_kind):
            rate = None
            if spec.get("rate"):
                rate = rng.uniform(*spec["rate"])
            n_det = spec.get("n_det", 1)
            arc = spec.get("arc", 0)
            rms = (rng.uniform(20, 60) if spec.get("rms_high")
                   else rng.uniform(0.5, 3.0))
            sky_context = {"stationary": True} if spec.get("stationary") else {}
            ev = Evidence(
                ra_deg=rng.uniform(0, 360),
                dec_deg=rng.uniform(-60, 60),
                rate_arcsec_hr=rate,
                apparent_mag=rng.uniform(16, 22),
                band="r",
                morphology_label=spec["morph"],
                morphology_confidence=rng.uniform(0.8, 0.95),
                n_detections=n_det,
                arc_days=arc,
                rms_arcsec=rms,
                sky_context=sky_context,
            )
            cases.append(LabelledCase(
                case_id=f"corp_{case_id:04d}",
                evidence=ev, truth_label=label,
                source="synthetic_artefact",
                adversarial=True,
                split="validation" if case_id % 5 != 0 else "holdout",
            ))
            case_id += 1

    return cases
