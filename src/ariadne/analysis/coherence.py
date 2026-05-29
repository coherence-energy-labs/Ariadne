"""Trajectory coherence / robustness metrics (MASTER_PLAN.md — coherence lens).

"Coherence" here is operationalized as ROBUSTNESS: how well a trajectory holds
together under a small injection error. A coherent (robust) path barely deviates;
an incoherent (chaotic, knife-edge) path amplifies a tiny error enormously -- the
"decoherence" of nearby trajectories.

Two universal, propagator-agnostic measures (work for CR3BP or full ephemeris):
  - endpoint_sensitivity: km of final-position deviation per 1 m/s injection error
    (the operationally meaningful robustness number a mission's nav budget cares about),
  - decoherence_rate: the finite-time Lyapunov exponent of a nondimensional state
    perturbation -- the exponential rate at which nearby trajectories diverge.

`prop` is any callable: state6 -> final state6.
"""
from __future__ import annotations

import math

import numpy as np


def endpoint_sensitivity(prop, s0, dv_kms: float = 1e-3) -> float:
    """Worst-case final-position deviation (km) per 1 m/s of initial velocity error.

    Higher = more fragile / less coherent. Probes +/- each velocity component.
    """
    s0 = np.asarray(s0, float)
    base = np.asarray(prop(s0), float)
    worst = 0.0
    for i in (3, 4, 5):
        for sgn in (+1.0, -1.0):
            sp = s0.copy()
            sp[i] += sgn * dv_kms
            f = np.asarray(prop(sp), float)
            worst = max(worst, float(np.linalg.norm(f[:3] - base[:3])))
    return worst / (dv_kms * 1000.0)        # km per (m/s)


def decoherence_rate(prop, s0, T_seconds: float,
                     L: float = 384400.0, V: float = 1.025,
                     eps: float = 1e-7) -> float:
    """Finite-time Lyapunov exponent (per day) of a nondimensional state perturbation.

    Position scaled by L (km), velocity by V (km/s) so the 6-vector is dimensionless.
    FTLE = ln(amplification) / T.

    ROUGH INDICATOR ONLY — a single-window finite-difference FTLE conflates a stable
    orbit's *polynomial* along-track drift with genuine *exponential* chaos, and it is
    window-length dependent (a 1-day LEO can score higher than a 49-day WSB arc). Use
    `endpoint_sensitivity` as the reliable coherence/robustness measure; this is kept only
    as a coarse secondary signal.
    """
    s0 = np.asarray(s0, float)
    scale = np.array([L, L, L, V, V, V])
    base = np.asarray(prop(s0), float)
    amp = 0.0
    for i in range(6):
        sp = s0.copy()
        sp[i] += eps * scale[i]
        f = np.asarray(prop(sp), float)
        d = np.linalg.norm((f - base) / scale) / eps
        amp = max(amp, d)
    days = T_seconds / 86400.0
    return math.log(max(amp, 1.0)) / days     # per day


def coherence_score(sensitivity_km_per_ms: float) -> float:
    """A 0..1 coherence score from endpoint sensitivity (1 = perfectly robust)."""
    return 1.0 / (1.0 + sensitivity_km_per_ms / 100.0)
