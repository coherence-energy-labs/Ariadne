"""Gravity-assist multi-flyby trajectories, globally optimized (MASTER_PLAN.md - Stage 22).

A direct Earth->Jupiter transfer needs a launch C3 ~ 85 km^2/s^2 -- beyond most launchers. A
gravity-assist chain (e.g. Galileo's Venus-Earth-Earth-Jupiter, "VEEGA") borrows momentum from
the planets and slashes the launch energy. We model the chain with patched conics: a Lambert arc
on each leg, and at each intermediate planet a flyby that ROTATES v_inf but cannot change |v_inf|
(any |v_inf| mismatch is a powered-flyby Delta-v; the required turn must be within the flyby's
turn authority). A global optimizer (differential evolution) searches the launch epoch and the
leg times of flight.

Standard gravity, patched-conic -- the standard tool for first-cut gravity-assist tour design.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import differential_evolution

from ..data.constants import (GM_SUN, GM_EARTH, GM_VENUS, R_EARTH, R_VENUS)
from ..data.ephemeris import body_state, utc
from ..dynamics.secular import kepler_step
from ..optimize.lambert import lambert

DAY = 86400.0

# flyby bodies -> (GM, body radius km)
_BODY = {
    "EARTH": (GM_EARTH, R_EARTH),
    "VENUS": (GM_VENUS, R_VENUS),
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


def _leg_with_optional_dsm(r1, r2, tof, mu, dsm_frac):
    """One Lambert leg, optionally with a mid-course Deep-Space Maneuver.

    No-DSM path (dsm_frac is None or outside (0.02, 0.98)) returns Lambert(r1, r2, tof) directly.

    DSM path: first solve the baseline Lambert(r1, r2, tof) to get the cruise velocity at r1;
    propagate ballistically by `dsm_frac * tof` to (r_DSM, v_pre); solve a second
    Lambert(r_DSM, r2, (1-dsm_frac)*tof) for (v_post, v_arr_new); the DSM impulse magnitude is
    ||v_post - v_pre||. The leg's terminal velocities are (v_dep, v_arr_new); the impulse adds
    to powered Delta-v but lets v_arr_new differ from the original v_arr -- the key freedom that
    lets the next flyby's |v_inf_in| match the prior flyby's |v_inf_out| (Galileo-class DSM).
    """
    if dsm_frac is None or not (0.02 < float(dsm_frac) < 0.98):
        v1, v2 = lambert(r1, r2, tof, mu)
        return v1, v2, 0.0, None
    f = float(dsm_frac)
    v_dep, _v_arr_no_dsm = lambert(r1, r2, tof, mu)
    t_dsm = f * tof
    r_dsm, v_pre = kepler_step(np.asarray(r1), v_dep, mu, t_dsm)
    if not (np.all(np.isfinite(r_dsm)) and np.all(np.isfinite(v_pre))):
        raise ValueError("DSM leg: Kepler propagation diverged")
    v_post, v_arr = lambert(r_dsm, r2, tof - t_dsm, mu)
    dv = float(np.linalg.norm(v_post - v_pre))
    info = {"r_km": np.asarray(r_dsm).tolist(), "v_pre_kms": v_pre.tolist(),
            "v_post_kms": v_post.tolist(), "frac": f, "dv_kms": dv}
    return v_dep, v_arr, dv, info


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
    total = dv_launch + mismatch_dv + dsm_dv_total
    return {"bodies": bodies, "et0": et0, "epochs": epochs, "tofs_days": list(tofs_days),
            "c3": c3, "dep_vinf_kms": math.sqrt(c3), "dv_launch_ms": dv_launch * 1000.0,
            "flybys": flybys, "mismatch_dv_ms": mismatch_dv * 1000.0,
            "dsm_dv_ms": dsm_dv_total * 1000.0,
            "dsm_per_leg_ms": [l[2] * 1000.0 for l in legs],
            "dsm_info": [l[3] for l in legs],
            "dsm_fracs": [None if l[3] is None else l[3]["frac"] for l in legs],
            "arr_vinf_kms": vinf_arr, "total_dv_ms": total * 1000.0,
            "infeasible": infeasible, "tof_total_days": sum(tofs_days),
            "r1": states[0][:3], "v1": legs[0][0]}


def optimize_chain(bodies, et_start, dep_window_days, tof_bounds, flyby_alt_km=300.0,
                   maxiter=120, seed=0, turn_penalty=5.0, dsm_legs=None,
                   dsm_frac_bounds=(0.1, 0.9)):
    """Global differential-evolution search over (launch epoch, leg TOFs).

    Minimizes total Delta-v = launch injection + flyby v_inf mismatches + DSM impulses, with an
    infeasible-turn penalty. Optional `dsm_legs` = list of leg indices to place a DSM on; each adds
    one decision variable (frac in `dsm_frac_bounds`). Returns the best chain."""
    bounds = [(0.0, dep_window_days)] + list(tof_bounds)
    n_basic = len(bounds)
    if dsm_legs:
        bounds = bounds + [tuple(dsm_frac_bounds)] * len(dsm_legs)

    def _unpack_dsm(x):
        if not dsm_legs:
            return None
        fracs = [None] * (len(bodies) - 1)
        for k, idx in enumerate(dsm_legs):
            fracs[idx] = float(x[n_basic + k])
        return fracs

    def obj(x):
        r = evaluate_chain(bodies, et_start + x[0] * DAY, x[1:n_basic], flyby_alt_km,
                           dsm_fracs=_unpack_dsm(x))
        if r is None:
            return 1e12
        return (r["dv_launch_ms"] + r["mismatch_dv_ms"] + r.get("dsm_dv_ms", 0.0)
                + turn_penalty * 1000.0 * r["infeasible"])

    res = differential_evolution(obj, bounds, seed=seed, maxiter=maxiter, tol=1e-7,
                                 mutation=(0.5, 1.5), recombination=0.7, polish=True)
    best = evaluate_chain(bodies, et_start + res.x[0] * DAY, res.x[1:n_basic], flyby_alt_km,
                          dsm_fracs=_unpack_dsm(res.x))
    if best:
        best["utc_launch"] = utc(best["et0"])
        best["utc_arrival"] = utc(best["epochs"][-1])
        best["dsm_legs"] = list(dsm_legs) if dsm_legs else []
    return best
