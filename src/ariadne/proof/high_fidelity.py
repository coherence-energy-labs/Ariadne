"""High-fidelity evidence generators for route promotion."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from ..data.constants import GM_SUN
from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..optimize.lambert import lambert
from .promotion import PromotionEvidence


DEFAULT_HELIOCENTRIC_PERTURBERS = (
    "MERCURY BARYCENTER",
    "VENUS",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)


DAY = 86400.0


def _route_dict(route: Any) -> dict:
    if hasattr(route, "__dataclass_fields__"):
        return asdict(route)
    if isinstance(route, dict):
        return route
    raise TypeError("route must be a route dictionary or MissionRoute dataclass")


def _event_coord(event: dict) -> np.ndarray:
    coords = event.get("coordinates_km")
    if not isinstance(coords, (list, tuple)) or len(coords) != 3:
        raise ValueError("route event is missing finite coordinates_km")
    out = np.asarray(coords, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("route event coordinates_km must be finite")
    return out


def _body_key(name: str) -> str:
    key = name.upper().replace("_", " ")
    aliases = {
        "MARS": "MARS BARYCENTER",
        "JUPITER": "JUPITER BARYCENTER",
        "SATURN": "SATURN BARYCENTER",
    }
    return aliases.get(key, key)


def _usable_perturbers(dep_body: str, arr_body: str, perturbers: tuple[str, ...]) -> tuple[str, ...]:
    blocked = {_body_key(dep_body), _body_key(arr_body), "SUN"}
    return tuple(p for p in perturbers if _body_key(p) not in blocked)


def _propagate(r0: np.ndarray, v0: np.ndarray, et0: float, tof_s: float,
               perturbers: tuple[str, ...]):
    return propagate_test_particle(
        r0,
        v0,
        et0,
        (0.0, tof_s),
        central="SUN",
        perturbers=perturbers,
        rtol=2e-10,
        atol=1e-5,
    ).y[:, -1]


def _propagate_independent(r0: np.ndarray, v0: np.ndarray, et0: float, tof_s: float,
                           perturbers: tuple[str, ...]):
    return propagate_test_particle(
        r0,
        v0,
        et0,
        (0.0, tof_s),
        central="SUN",
        perturbers=perturbers,
        method="Radau",
        rtol=5e-11,
        atol=1e-6,
    ).y[:, -1]


def _retarget_velocity(
    r0: np.ndarray,
    v0: np.ndarray,
    target_r: np.ndarray,
    et0: float,
    tof_s: float,
    perturbers: tuple[str, ...],
    *,
    tol_km: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(v0, dtype=float).copy()
    h = 1e-4
    state = _propagate(r0, v, et0, tof_s, perturbers)
    for _ in range(max_iter):
        miss = state[:3] - target_r
        miss_norm = float(np.linalg.norm(miss))
        if miss_norm <= tol_km:
            break
        jac = np.zeros((3, 3))
        for i in range(3):
            vp = v.copy()
            vp[i] += h
            sp = _propagate(r0, vp, et0, tof_s, perturbers)
            jac[:, i] = (sp[:3] - state[:3]) / h
        step, *_ = np.linalg.lstsq(jac, -miss, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        v = v + step
        state = _propagate(r0, v, et0, tof_s, perturbers)
    return v, state, float(np.linalg.norm(state[:3] - target_r))


def nbody_replay_evidence(
    route: Any,
    *,
    perturbers: tuple[str, ...] = DEFAULT_HELIOCENTRIC_PERTURBERS,
    position_tolerance_km: float = 10.0,
    max_retarget_correction_mps: float = 500.0,
    max_iter: int = 8,
) -> PromotionEvidence:
    """Generate n-body replay evidence for a direct heliocentric Lambert route.

    The route's endpoint coordinates define the two-body Lambert seed. The seed
    is propagated under Sun-centered n-body perturbations, then the departure
    velocity is corrected by finite-difference targeting so the same arrival
    position is hit in the higher-fidelity dynamics.
    """

    r = _route_dict(route)
    sequence = tuple(r.get("sequence") or ())
    events = list(r.get("events") or ())
    if len(sequence) < 2 or len(events) < 2:
        raise ValueError("n-body replay requires at least two sequence bodies and two route events")
    dep_event = events[0]
    arr_event = events[-1]
    r0 = _event_coord(dep_event)
    r_target = _event_coord(arr_event)
    raw = r.get("raw") or {}
    et0 = raw.get("et_dep")
    tof_days = r.get("tof_days")
    if not isinstance(et0, (int, float)) or not math.isfinite(float(et0)):
        raise ValueError("route raw evidence must include finite et_dep")
    if not isinstance(tof_days, (int, float)) or not math.isfinite(float(tof_days)):
        raise ValueError("route must include finite tof_days")
    tof_s = float(tof_days) * DAY
    dep_body = str(sequence[0])
    arr_body = str(sequence[-1])
    active_perturbers = _usable_perturbers(dep_body, arr_body, perturbers)
    v1, v2 = lambert(r0, r_target, tof_s, GM_SUN)
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        raise ValueError("Lambert seed produced non-finite velocity")
    uncorrected = _propagate(r0, v1, float(et0), tof_s, active_perturbers)
    uncorrected_miss = float(np.linalg.norm(uncorrected[:3] - r_target))
    corrected_v, corrected_state, corrected_miss = _retarget_velocity(
        r0,
        v1,
        r_target,
        float(et0),
        tof_s,
        active_perturbers,
        tol_km=position_tolerance_km,
        max_iter=max_iter,
    )
    correction_mps = float(np.linalg.norm(corrected_v - v1) * 1000.0)
    final_velocity_delta_mps = float(np.linalg.norm(corrected_state[3:] - v2) * 1000.0)
    status = corrected_miss <= position_tolerance_km and correction_mps <= max_retarget_correction_mps
    metrics = {
        "uncorrected_position_residual_km": uncorrected_miss,
        "max_position_residual_km": corrected_miss,
        "retarget_correction_dv_mps": correction_mps,
        "final_velocity_delta_mps": final_velocity_delta_mps,
        "tof_days": float(tof_days),
        "perturber_count": len(active_perturbers),
        "perturbers": active_perturbers,
        "position_tolerance_km": position_tolerance_km,
        "max_retarget_correction_mps": max_retarget_correction_mps,
    }
    return PromotionEvidence(
        rung="nbody_replay",
        status="pass" if status else "fail",
        source="ariadne.proof.high_fidelity.nbody_replay_evidence",
        metrics=metrics,
        notes=("Sun-centered n-body finite-difference retarget from Lambert seed",),
    )


def covariance_envelope_evidence(
    route: Any,
    *,
    perturbers: tuple[str, ...] = DEFAULT_HELIOCENTRIC_PERTURBERS,
    sigma_position_km: float = 1e-3,
    sigma_velocity_mps: float = 1e-3,
    max_position_3sigma_km: float = 1_000.0,
    max_dv_3sigma_mps: float = 10.0,
) -> PromotionEvidence:
    """Generate deterministic arrival-dispersion evidence for a route.

    This is a sigma-point covariance envelope, not a random Monte Carlo. It
    perturbs the Lambert departure state by +/- one sigma along each Cartesian
    position and velocity axis, propagates every perturbed state in the same
    Sun-centered n-body model, and reports the largest 3-sigma arrival
    displacement. The velocity sigma is also reported as a maneuver-execution
    envelope, so promotion can gate both navigation dispersion and execution
    tolerance.
    """

    r = _route_dict(route)
    sequence = tuple(r.get("sequence") or ())
    events = list(r.get("events") or ())
    if len(sequence) < 2 or len(events) < 2:
        raise ValueError("covariance envelope requires at least two sequence bodies and route events")
    dep_event = events[0]
    arr_event = events[-1]
    r0 = _event_coord(dep_event)
    r_target = _event_coord(arr_event)
    raw = r.get("raw") or {}
    et0 = raw.get("et_dep")
    tof_days = r.get("tof_days")
    if not isinstance(et0, (int, float)) or not math.isfinite(float(et0)):
        raise ValueError("route raw evidence must include finite et_dep")
    if not isinstance(tof_days, (int, float)) or not math.isfinite(float(tof_days)):
        raise ValueError("route must include finite tof_days")
    if sigma_position_km < 0.0 or sigma_velocity_mps < 0.0:
        raise ValueError("sigma values must be non-negative")
    tof_s = float(tof_days) * DAY
    dep_body = str(sequence[0])
    arr_body = str(sequence[-1])
    active_perturbers = _usable_perturbers(dep_body, arr_body, perturbers)
    v_seed, _ = lambert(r0, r_target, tof_s, GM_SUN)
    if not np.all(np.isfinite(v_seed)):
        raise ValueError("Lambert seed produced non-finite velocity")

    nominal = _propagate(r0, v_seed, float(et0), tof_s, active_perturbers)
    endpoints = []
    sigma_v_kms = sigma_velocity_mps / 1000.0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            rp = r0.copy()
            rp[axis] += sign * sigma_position_km
            endpoints.append(_propagate(rp, v_seed, float(et0), tof_s, active_perturbers))
            vp = v_seed.copy()
            vp[axis] += sign * sigma_v_kms
            endpoints.append(_propagate(r0, vp, float(et0), tof_s, active_perturbers))
    deltas = np.array([state[:3] - nominal[:3] for state in endpoints], dtype=float)
    norms = np.linalg.norm(deltas, axis=1)
    one_sigma_position_km = float(np.max(norms)) if len(norms) else 0.0
    position_3sigma_km = 3.0 * one_sigma_position_km
    dv_3sigma_mps = 3.0 * sigma_velocity_mps
    nominal_miss_km = float(np.linalg.norm(nominal[:3] - r_target))
    status = position_3sigma_km <= max_position_3sigma_km and dv_3sigma_mps <= max_dv_3sigma_mps
    metrics = {
        "position_1sigma_km": one_sigma_position_km,
        "position_3sigma_km": position_3sigma_km,
        "dv_1sigma_mps": sigma_velocity_mps,
        "dv_3sigma_mps": dv_3sigma_mps,
        "sigma_position_km": sigma_position_km,
        "sigma_velocity_mps": sigma_velocity_mps,
        "nominal_nbody_position_residual_km": nominal_miss_km,
        "tof_days": float(tof_days),
        "sigma_point_count": len(endpoints),
        "perturber_count": len(active_perturbers),
        "perturbers": active_perturbers,
        "max_position_3sigma_km": max_position_3sigma_km,
        "max_dv_3sigma_mps": max_dv_3sigma_mps,
    }
    return PromotionEvidence(
        rung="covariance_envelope",
        status="pass" if status else "fail",
        source="ariadne.proof.high_fidelity.covariance_envelope_evidence",
        metrics=metrics,
        notes=("Deterministic +/- Cartesian sigma-point n-body arrival dispersion",),
    )


def independent_crosscheck_evidence(
    route: Any,
    *,
    perturbers: tuple[str, ...] = DEFAULT_HELIOCENTRIC_PERTURBERS,
    max_position_delta_km: float = 10.0,
    max_velocity_delta_mps: float = 0.05,
    retarget_tolerance_km: float = 10.0,
    max_iter: int = 8,
) -> PromotionEvidence:
    """Cross-check a route with an independent implicit integrator.

    The n-body replay rung targets the arrival using Ariadne's default DOP853
    propagator. This rung recomputes the final state with Radau, an independent
    implicit integration scheme, using the same corrected departure velocity.
    The promoted claim is numerical method consistency, not external GMAT/Monte
    validation; GMAT can still be added as another evidence source when present.
    """

    r = _route_dict(route)
    sequence = tuple(r.get("sequence") or ())
    events = list(r.get("events") or ())
    if len(sequence) < 2 or len(events) < 2:
        raise ValueError("independent cross-check requires at least two sequence bodies and route events")
    r0 = _event_coord(events[0])
    r_target = _event_coord(events[-1])
    raw = r.get("raw") or {}
    et0 = raw.get("et_dep")
    tof_days = r.get("tof_days")
    if not isinstance(et0, (int, float)) or not math.isfinite(float(et0)):
        raise ValueError("route raw evidence must include finite et_dep")
    if not isinstance(tof_days, (int, float)) or not math.isfinite(float(tof_days)):
        raise ValueError("route must include finite tof_days")
    tof_s = float(tof_days) * DAY
    active_perturbers = _usable_perturbers(str(sequence[0]), str(sequence[-1]), perturbers)
    v_seed, _ = lambert(r0, r_target, tof_s, GM_SUN)
    corrected_v, default_state, default_miss = _retarget_velocity(
        r0,
        v_seed,
        r_target,
        float(et0),
        tof_s,
        active_perturbers,
        tol_km=retarget_tolerance_km,
        max_iter=max_iter,
    )
    independent_state = _propagate_independent(r0, corrected_v, float(et0), tof_s, active_perturbers)
    position_delta_km = float(np.linalg.norm(independent_state[:3] - default_state[:3]))
    velocity_delta_mps = float(np.linalg.norm(independent_state[3:] - default_state[3:]) * 1000.0)
    status = position_delta_km <= max_position_delta_km and velocity_delta_mps <= max_velocity_delta_mps
    metrics = {
        "position_delta_km": position_delta_km,
        "velocity_delta_mps": velocity_delta_mps,
        "default_position_residual_km": default_miss,
        "tof_days": float(tof_days),
        "perturber_count": len(active_perturbers),
        "perturbers": active_perturbers,
        "default_integrator": "DOP853",
        "crosscheck_integrator": "Radau",
        "max_position_delta_km": max_position_delta_km,
        "max_velocity_delta_mps": max_velocity_delta_mps,
    }
    return PromotionEvidence(
        rung="independent_crosscheck",
        status="pass" if status else "fail",
        source="ariadne.proof.high_fidelity.independent_crosscheck_evidence",
        metrics=metrics,
        notes=("DOP853 targeted n-body replay cross-checked with Radau integration",),
    )
