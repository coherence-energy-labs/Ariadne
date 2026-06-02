"""Gravity-assist multi-flyby trajectories, globally optimized (MASTER_PLAN.md - Stage 22).

A direct Earth->Jupiter transfer needs a launch C3 ~ 85 km^2/s^2 -- beyond most launchers. A
gravity-assist chain (e.g. Galileo's Venus-Earth-Earth-Jupiter, "VEEGA") borrows momentum from
the planets and slashes the launch energy. We model the chain with patched conics: a Lambert arc
on each leg, and at each intermediate planet a flyby that ROTATES v_inf but cannot change ``|v_inf|``
(any ``|v_inf|`` mismatch is a powered-flyby Delta-v; the required turn must be within the flyby's
turn authority). A global optimizer (differential evolution) searches the launch epoch and the
leg times of flight.

Standard gravity, patched-conic -- the standard tool for first-cut gravity-assist tour design.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import differential_evolution

from ..data.constants import (
    GM_SUN, GM_EARTH, GM_VENUS, GM_JUPITER, GM_SATURN,
    R_EARTH, R_VENUS, R_JUPITER, R_SATURN,
)
from ..data.ephemeris import body_state, utc
from ..dynamics.secular import kepler_step
from ..optimize.lambert import lambert

DAY = 86400.0

# flyby bodies -> (GM, body radius km)
_BODY = {
    "EARTH": (GM_EARTH, R_EARTH),
    "EARTH BARYCENTER": (GM_EARTH, R_EARTH),
    "VENUS": (GM_VENUS, R_VENUS),
    "VENUS BARYCENTER": (GM_VENUS, R_VENUS),
    "JUPITER": (GM_JUPITER, R_JUPITER),
    "JUPITER BARYCENTER": (GM_JUPITER, R_JUPITER),
    "SATURN": (GM_SATURN, R_SATURN),
    "SATURN BARYCENTER": (GM_SATURN, R_SATURN),
}


#: Reference Galileo-class VEEGA (Earth-Venus-Earth-Earth-Jupiter) from the Stage-22 global
#: optimizer (epoch base 2029-01-01). Reproduces launch C3 = 16.8 km^2/s^2 with all flybys
#: feasible -- a 5x launch-energy cut vs the direct ~85, in the Galileo trajectory class.
GALILEO_VEEGA = {
    "bodies": ["EARTH", "VENUS", "EARTH", "EARTH", "JUPITER BARYCENTER"],
    "epoch_base": "2029-01-01T00:00:00",
    "offset_days": 344.363835,
    "tofs_days": [164.319906, 343.300573, 730.500241, 1082.878784],
}


def _max_turn(vinf, gm, r_flyby):
    return 2.0 * math.asin(1.0 / (1.0 + r_flyby * vinf * vinf / gm))


def reference_veega():
    """Evaluate the stored Galileo-class VEEGA reference solution (fast, deterministic)."""
    from ..data.ephemeris import et
    v = GALILEO_VEEGA
    e0 = et(v["epoch_base"]) + v["offset_days"] * DAY
    return evaluate_chain(v["bodies"], e0, v["tofs_days"])


def _leg_with_optional_dsm(r1, r2, tof, mu, dsm_spec):
    """One Lambert leg, optionally with a mid-course Deep-Space Maneuver.

    `dsm_spec` accepts three shapes:
      None  -> no DSM, plain Lambert
      float -> 1-DOF DSM at frac=dsm_spec; impulse direction Lambert-fitted
      4-tuple (frac, dvx, dvy, dvz) -> 4-DOF DSM with explicit 3D kick

    1-DOF path (frac only): coast to r_DSM via baseline Lambert, then Lambert from r_DSM to r_arr.
    Impulse magnitude = ||v_post - v_pre||. Geometry-determined; one knob.

    4-DOF path: coast to r_DSM via baseline Lambert, apply explicit (dvx, dvy, dvz) kick (the
    optimizer-chosen impulse), then ballistically propagate. The arrival position is no longer
    exactly r_arr -- the caller's objective must penalise this mismatch (added as `pos_err_km` in
    info). Optimizer learns to trade kick magnitude vs position match; can recover the 1-DOF
    solution with zero kick or beat it with directional freedom.

    Returns (v_dep, v_arr_actual, dv_impulse, info).
    """
    # ---- no-DSM ----
    if dsm_spec is None:
        v1, v2 = lambert(r1, r2, tof, mu)
        return v1, v2, 0.0, None

    # ---- 1-DOF (legacy float input) ----
    if np.isscalar(dsm_spec):
        f = float(dsm_spec)
        if not (0.02 < f < 0.98):
            v1, v2 = lambert(r1, r2, tof, mu)
            return v1, v2, 0.0, None
        v_dep, _ = lambert(r1, r2, tof, mu)
        t_dsm = f * tof
        r_dsm, v_pre = kepler_step(np.asarray(r1), v_dep, mu, t_dsm)
        if not (np.all(np.isfinite(r_dsm)) and np.all(np.isfinite(v_pre))):
            raise ValueError("DSM leg: Kepler propagation diverged")
        v_post, v_arr = lambert(r_dsm, r2, tof - t_dsm, mu)
        dv = float(np.linalg.norm(v_post - v_pre))
        info = {"frac": f, "dv_kms": dv, "dof": 1, "pos_err_km": 0.0}
        return v_dep, v_arr, dv, info

    # ---- 4-DOF (frac + 3D kick) ----
    spec = np.asarray(dsm_spec, dtype=float)
    if spec.size != 4:
        raise ValueError(f"dsm_spec must be None, float, or 4-tuple; got shape {spec.shape}")
    f = float(spec[0])
    kick = spec[1:4]
    if not (0.02 < f < 0.98) and float(np.linalg.norm(kick)) < 1e-9:
        # effectively no-DSM
        v1, v2 = lambert(r1, r2, tof, mu)
        return v1, v2, 0.0, None
    v_dep, _ = lambert(r1, r2, tof, mu)
    t_dsm = max(1e-3 * tof, min((1 - 1e-3) * tof, f * tof))
    r_dsm, v_pre = kepler_step(np.asarray(r1), v_dep, mu, t_dsm)
    if not (np.all(np.isfinite(r_dsm)) and np.all(np.isfinite(v_pre))):
        raise ValueError("DSM 4DOF leg: Kepler propagation diverged")
    # apply explicit kick
    v_post = v_pre + kick
    # propagate ballistically for remaining time
    r_arr_actual, v_arr_actual = kepler_step(np.asarray(r_dsm), v_post, mu, tof - t_dsm)
    if not (np.all(np.isfinite(r_arr_actual)) and np.all(np.isfinite(v_arr_actual))):
        raise ValueError("DSM 4DOF leg: post-kick propagation diverged")
    pos_err_km = float(np.linalg.norm(r_arr_actual - np.asarray(r2)))
    dv = float(np.linalg.norm(kick))
    info = {"frac": f, "dv_kms": dv, "dof": 4, "kick_kms": kick.tolist(),
            "pos_err_km": pos_err_km}
    return v_dep, v_arr_actual, dv, info


def evaluate_chain(bodies, et0, tofs_days, flyby_alt_km=300.0,
                   leo_alt=200.0, mu=GM_SUN, dsm_fracs=None):
    """Evaluate a gravity-assist chain. bodies[0]=launch, bodies[-1]=arrival.

    `dsm_fracs` (optional): list of length len(bodies)-1, each entry is None (no DSM on that leg)
    or a float in (0.02, 0.98) giving the fraction of the leg TOF at which to place a DSM. Adding
    DSMs lets the optimizer trade ~50-200 m/s of impulsive Delta-v to slash multi-km/s of powered-
    flyby mismatch -- a real Galileo-class refinement.

    Returns a dict: launch C3, per-flyby v_inf mismatch + required/max turn, arrival v_inf,
    total deterministic Delta-v (LEO injection + flyby mismatches + DSM impulses), and feasibility.
    """
    epochs = [et0]
    for t in tofs_days:
        epochs.append(epochs[-1] + t * DAY)
    states = [body_state(b, e, "J2000", "SUN") for b, e in zip(bodies, epochs)]

    n_legs = len(bodies) - 1
    if dsm_fracs is None:
        dsm_fracs = [None] * n_legs
    elif len(dsm_fracs) != n_legs:
        raise ValueError(f"dsm_fracs length {len(dsm_fracs)} != n_legs {n_legs}")

    legs = []                                  # each: (v_dep, v_arr, dsm_dv_kms, dsm_info)
    for i in range(n_legs):
        tof = (epochs[i + 1] - epochs[i])
        try:
            v1, v2, dv_dsm, dsm_info = _leg_with_optional_dsm(
                states[i][:3], states[i + 1][:3], tof, mu, dsm_fracs[i])
        except Exception:
            return None
        if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
            return None
        legs.append((v1, v2, dv_dsm, dsm_info))

    # launch C3
    vinf_launch = legs[0][0] - states[0][3:]
    c3 = float(vinf_launch @ vinf_launch)
    r_leo = R_EARTH + leo_alt
    dv_launch = math.sqrt(c3 + 2.0 * GM_EARTH / r_leo) - math.sqrt(GM_EARTH / r_leo)

    # intermediate flybys
    flybys, mismatch_dv, infeasible = [], 0.0, 0.0
    for i in range(1, len(bodies) - 1):
        body = bodies[i]
        vb = states[i][3:]
        vin = legs[i - 1][1] - vb
        vout = legs[i][0] - vb
        m_in, m_out = np.linalg.norm(vin), np.linalg.norm(vout)
        mism = abs(m_in - m_out)
        mismatch_dv += mism
        turn_req = math.acos(max(-1.0, min(1.0, float(vin @ vout) / (m_in * m_out))))
        gm, rb = _BODY.get(body, (GM_EARTH, R_EARTH))
        turn_max = _max_turn(0.5 * (m_in + m_out), gm, rb + flyby_alt_km)
        if turn_req > turn_max:
            infeasible += (turn_req - turn_max)
        flybys.append({"body": body, "vinf_in_kms": m_in, "vinf_out_kms": m_out,
                       "mismatch_ms": mism * 1000.0, "turn_req_deg": math.degrees(turn_req),
                       "turn_max_deg": math.degrees(turn_max),
                       "feasible": turn_req <= turn_max})

    vinf_arr = float(np.linalg.norm(legs[-1][1] - states[-1][3:]))
    dsm_dv_total = sum(l[2] for l in legs)
    pos_err_total_km = sum(l[3]["pos_err_km"] if l[3] else 0.0 for l in legs)
    total = dv_launch + mismatch_dv + dsm_dv_total
    return {"bodies": bodies, "et0": et0, "epochs": epochs, "tofs_days": list(tofs_days),
            "c3": c3, "dep_vinf_kms": math.sqrt(c3), "dv_launch_ms": dv_launch * 1000.0,
            "flybys": flybys, "mismatch_dv_ms": mismatch_dv * 1000.0,
            "dsm_dv_ms": dsm_dv_total * 1000.0,
            "dsm_per_leg_ms": [l[2] * 1000.0 for l in legs],
            "dsm_info": [l[3] for l in legs],
            "dsm_fracs": [None if l[3] is None else l[3]["frac"] for l in legs],
            "pos_err_total_km": pos_err_total_km,
            "arr_vinf_kms": vinf_arr, "total_dv_ms": total * 1000.0,
            "infeasible": infeasible, "tof_total_days": sum(tofs_days),
            "r1": states[0][:3], "v1": legs[0][0]}


def optimize_chain(bodies, et_start, dep_window_days, tof_bounds, flyby_alt_km=300.0,
                   maxiter=120, seed=0, turn_penalty=5.0, dsm_legs=None,
                   dsm_frac_bounds=(0.1, 0.9), dsm_dof=1,
                   dsm_kick_bound_kms=2.0, pos_err_penalty_per_km=0.5,
                   warm_start_1dof=None, popsize=15):
    """Global differential-evolution search over (launch epoch, leg TOFs).

    Minimizes total Delta-v = launch injection + flyby v_inf mismatches + DSM impulses, with an
    infeasible-turn penalty and (for 4-DOF DSMs) a position-error penalty.

    `dsm_legs` = list of leg indices to place a DSM on.
    `dsm_dof` = 1 (frac only -- Lambert-determined impulse) or 4 (frac + 3D Delta-v kick).
    `dsm_kick_bound_kms` = bound on each kick component for 4-DOF.
    `pos_err_penalty_per_km` = m/s per km of arrival-position miss (4-DOF only).
    """
    bounds = [(0.0, dep_window_days)] + list(tof_bounds)
    n_basic = len(bounds)
    if dsm_legs:
        if dsm_dof == 1:
            bounds = bounds + [tuple(dsm_frac_bounds)] * len(dsm_legs)
        elif dsm_dof == 4:
            for _ in dsm_legs:
                bounds = bounds + [tuple(dsm_frac_bounds),
                                   (-dsm_kick_bound_kms, dsm_kick_bound_kms),
                                   (-dsm_kick_bound_kms, dsm_kick_bound_kms),
                                   (-dsm_kick_bound_kms, dsm_kick_bound_kms)]
        else:
            raise ValueError(f"dsm_dof must be 1 or 4, got {dsm_dof}")

    def _unpack_dsm(x):
        if not dsm_legs:
            return None
        specs = [None] * (len(bodies) - 1)
        for k, idx in enumerate(dsm_legs):
            if dsm_dof == 1:
                specs[idx] = float(x[n_basic + k])
            else:
                base = n_basic + 4 * k
                specs[idx] = tuple(float(x[base + j]) for j in range(4))
        return specs

    def obj(x):
        r = evaluate_chain(bodies, et_start + x[0] * DAY, x[1:n_basic], flyby_alt_km,
                           dsm_fracs=_unpack_dsm(x))
        if r is None:
            return 1e12
        pos_err_cost = pos_err_penalty_per_km * r.get("pos_err_total_km", 0.0)
        return (r["dv_launch_ms"] + r["mismatch_dv_ms"] + r.get("dsm_dv_ms", 0.0)
                + turn_penalty * 1000.0 * r["infeasible"] + pos_err_cost)

    init_pop = None
    if warm_start_1dof is not None and dsm_dof == 4 and dsm_legs:
        # warm start: lift the 1-DOF solution into the 4-DOF search by setting kicks=0
        # and rebuild a population around it
        x_1dof = np.asarray(warm_start_1dof)
        assert x_1dof.size == n_basic + len(dsm_legs), \
            f"warm_start_1dof must have {n_basic + len(dsm_legs)} entries, got {x_1dof.size}"
        x_seed = np.zeros(len(bounds))
        x_seed[:n_basic] = x_1dof[:n_basic]
        for k in range(len(dsm_legs)):
            x_seed[n_basic + 4 * k] = x_1dof[n_basic + k]    # frac
            x_seed[n_basic + 4 * k + 1: n_basic + 4 * k + 4] = 0.0    # kicks=0 by default
        # population: x_seed + perturbations within bounds
        rng = np.random.default_rng(seed)
        pop_n = popsize * len(bounds)
        init_pop = np.tile(x_seed, (pop_n, 1))
        for j, (lo, hi) in enumerate(bounds):
            scale = 0.05 * (hi - lo)
            init_pop[1:, j] = np.clip(x_seed[j] + rng.normal(0.0, scale, size=pop_n - 1),
                                       lo, hi)

    kwargs = dict(seed=seed, maxiter=maxiter, tol=1e-7, mutation=(0.5, 1.5),
                  recombination=0.7, polish=True, popsize=popsize)
    if init_pop is not None:
        kwargs["init"] = init_pop
    res = differential_evolution(obj, bounds, **kwargs)
    best = evaluate_chain(bodies, et_start + res.x[0] * DAY, res.x[1:n_basic], flyby_alt_km,
                          dsm_fracs=_unpack_dsm(res.x))
    if best:
        best["utc_launch"] = utc(best["et0"])
        best["utc_arrival"] = utc(best["epochs"][-1])
        best["dsm_legs"] = list(dsm_legs) if dsm_legs else []
        best["dsm_dof"] = int(dsm_dof)
    return best
