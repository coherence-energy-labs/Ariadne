"""Interplanetary transfers, epoch-swept and globally optimized (MASTER_PLAN.md - Stage 21).

The variable we had been ignoring -- the LAUNCH EPOCH (time of year / planet geometry) -- is
here a free optimization variable. For a departure body and an arrival body we solve the
heliocentric Lambert problem on the real DE440 ephemeris over a grid of (departure date, time
of flight): the classic porkchop. From it we read the launch WINDOWS, the global minimum, and
the time-vs-energy Pareto front (faster transfers cost more C3 -- no free lunch, but the
optimizer finds the best trade). A global optimizer (differential evolution) pins the true
optimum without grid quantization.

Departure cost is the LEO injection for the required C3; arrival cost is capture into a circular
orbit at the target (or just the arrival v_inf for a flyby / aerocapture). Standard gravity,
patched-conic -- the industry-standard first-cut tool, on real ephemeris.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import differential_evolution

from ..data.constants import (
    GM_SUN, GM_EARTH, GM_MARS, GM_VENUS, GM_JUPITER, GM_SATURN,
    R_EARTH, R_MARS, R_VENUS, R_JUPITER, R_SATURN,
)
from ..data.ephemeris import body_state, utc
from ..optimize.lambert import lambert

DAY = 86400.0

# arrival capture targets: body -> (GM, capture orbit radius km)
_CAPTURE = {
    "MARS BARYCENTER": (GM_MARS, R_MARS + 400.0),
    "MARS": (GM_MARS, R_MARS + 400.0),
    "VENUS": (GM_VENUS, R_VENUS + 400.0),
}

_PARKING = {
    "EARTH": (GM_EARTH, R_EARTH + 200.0),
    "VENUS": (GM_VENUS, R_VENUS + 400.0),
    "MARS": (GM_MARS, R_MARS + 400.0),
    "MARS BARYCENTER": (GM_MARS, R_MARS + 400.0),
    "JUPITER BARYCENTER": (GM_JUPITER, R_JUPITER + 1000.0),
    "SATURN BARYCENTER": (GM_SATURN, R_SATURN + 1000.0),
}


def lambert_transfer(dep_body, arr_body, et_dep, tof_days, leo_alt=200.0,
                     capture=True, prograde=True, mu=GM_SUN):
    """One heliocentric Lambert transfer. Returns a cost dict, or None if Lambert fails."""
    tof = tof_days * DAY
    sd = body_state(dep_body, et_dep, "J2000", "SUN")
    sa = body_state(arr_body, et_dep + tof, "J2000", "SUN")
    return _lambert_transfer_from_states(
        dep_body, arr_body, float(et_dep), float(tof_days), sd, sa,
        leo_alt=leo_alt, capture=capture, prograde=prograde, mu=mu)


def _lambert_transfer_from_states(dep_body, arr_body, et_dep, tof_days, sd, sa,
                                  leo_alt=200.0, capture=True, prograde=True,
                                  mu=GM_SUN):
    """Lambert transfer with caller-supplied ephemeris states.

    Porkchop grids evaluate many TOFs for the same departure epochs. Supplying
    states avoids repeated SPICE calls for the departure body while keeping the
    public `lambert_transfer` semantics identical.
    """
    tof = tof_days * DAY
    try:
        v1, v2 = lambert(sd[:3], sa[:3], tof, mu, prograde=prograde)
    except Exception:
        return None
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return None
    c3 = float(np.dot(v1 - sd[3:], v1 - sd[3:]))
    vinf_arr = float(np.linalg.norm(v2 - sa[3:]))
    if dep_body in _PARKING:
        gm_dep, r_park = _PARKING[dep_body]
        if dep_body == "EARTH":
            r_park = R_EARTH + leo_alt
        dv_dep = math.sqrt(c3 + 2.0 * gm_dep / r_park) - math.sqrt(gm_dep / r_park)
    else:
        dv_dep = math.sqrt(c3)
    dv_arr = 0.0
    if capture and arr_body in _CAPTURE:
        gm, rc = _CAPTURE[arr_body]
        dv_arr = math.sqrt(vinf_arr ** 2 + 2.0 * gm / rc) - math.sqrt(gm / rc)
    return {"c3": c3, "dep_vinf_kms": math.sqrt(c3), "arr_vinf_kms": vinf_arr,
            "dv_dep_ms": dv_dep * 1000.0, "dv_arr_ms": dv_arr * 1000.0,
            "total_ms": (dv_dep + dv_arr) * 1000.0, "tof_days": float(tof_days),
            "et_dep": float(et_dep), "r1": sd[:3], "v1": v1,
            "et_arr": float(et_dep + tof), "r2": sa[:3], "v2": v2}


def porkchop(dep_body, arr_body, et_start, dep_days, tof_range,
             n_dep=60, n_tof=60, metric="total_ms", **kw):
    """Compute a porkchop grid. Returns dep/tof grids + C3 and total-Delta-v arrays + the grid min."""
    dep_grid = et_start + np.linspace(0.0, dep_days, n_dep) * DAY
    tof_grid = np.linspace(tof_range[0], tof_range[1], n_tof)
    C3 = np.full((n_tof, n_dep), np.nan)
    TOT = np.full((n_tof, n_dep), np.nan)
    best = None
    dep_states = [body_state(dep_body, ed, "J2000", "SUN") for ed in dep_grid]
    for i, tof in enumerate(tof_grid):
        for j, (ed, sd) in enumerate(zip(dep_grid, dep_states)):
            sa = body_state(arr_body, ed + float(tof) * DAY, "J2000", "SUN")
            r = _lambert_transfer_from_states(
                dep_body, arr_body, float(ed), float(tof), sd, sa, **kw)
            if r is None:
                continue
            C3[i, j] = r["c3"]
            TOT[i, j] = r["total_ms"]
            if best is None or r[metric] < best[metric]:
                best = r
    return {"dep_grid": dep_grid, "tof_grid": tof_grid, "C3": C3, "total_ms": TOT,
            "grid_best": best, "dep_body": dep_body, "arr_body": arr_body}


def optimize_window(dep_body, arr_body, et_start, dep_days, tof_range,
                    metric="total_ms", maxiter=60, seed=0, **kw):
    """Global (differential-evolution) search for the optimal (epoch, TOF). Returns the best transfer."""
    def obj(x):
        r = lambert_transfer(dep_body, arr_body, et_start + x[0] * DAY, float(x[1]), **kw)
        return r[metric] if r else 1e12

    res = differential_evolution(obj, [(0.0, dep_days), (tof_range[0], tof_range[1])],
                                 seed=seed, maxiter=maxiter, tol=1e-7, polish=True)
    best = lambert_transfer(dep_body, arr_body, et_start + res.x[0] * DAY, float(res.x[1]), **kw)
    if best:
        best["utc_dep"] = utc(best["et_dep"])
        best["utc_arr"] = utc(best["et_arr"])
    return best


def launch_windows(dep_body, arr_body, et_start, years=6.0, tof_range=(120.0, 400.0),
                   n_dep=220, n_tof=40, **kw):
    """For each departure date, the best-over-TOF total Delta-v -> the launch-window cadence."""
    dep_grid = et_start + np.linspace(0.0, years * 365.25, n_dep) * DAY
    tofs = np.linspace(tof_range[0], tof_range[1], n_tof)
    best_dv = np.full(n_dep, np.nan)
    best_tof = np.full(n_dep, np.nan)
    dep_states = [body_state(dep_body, ed, "J2000", "SUN") for ed in dep_grid]
    for j, (ed, sd) in enumerate(zip(dep_grid, dep_states)):
        vals = []
        for t in tofs:
            sa = body_state(arr_body, ed + float(t) * DAY, "J2000", "SUN")
            rec = _lambert_transfer_from_states(
                dep_body, arr_body, float(ed), float(t), sd, sa, **kw)
            vals.append((rec or {"total_ms": np.inf})["total_ms"])
        k = int(np.nanargmin(vals))
        best_dv[j] = vals[k]
        best_tof[j] = tofs[k]
    return {"dep_grid": dep_grid, "best_dv_ms": best_dv, "best_tof_days": best_tof}


def time_energy_pareto(pork):
    """Non-dominated (total Delta-v, TOF) set from a porkchop grid -- the time/energy trade."""
    pts = []
    for i, tof in enumerate(pork["tof_grid"]):
        col = pork["total_ms"][i]
        if np.all(np.isnan(col)):
            continue
        j = int(np.nanargmin(col))
        pts.append({"tof_days": float(tof), "total_ms": float(col[j]),
                    "dep_et": float(pork["dep_grid"][j])})
    front = []
    for p in pts:
        if not any(q is not p and q["total_ms"] <= p["total_ms"] and q["tof_days"] <= p["tof_days"]
                   and (q["total_ms"] < p["total_ms"] or q["tof_days"] < p["tof_days"]) for q in pts):
            front.append(p)
    return sorted(front, key=lambda p: p["tof_days"])


def coherent_knee(front):
    """The time/energy balance point: nearest the utopia corner after [0,1] normalization.

    This is the coherence principle applied to route choice -- the most BALANCED route, not the
    cheapest (slowest) nor the fastest (priciest)."""
    if len(front) < 3:
        return front[0] if front else None
    t = np.array([p["tof_days"] for p in front], float)
    d = np.array([p["total_ms"] for p in front], float)
    tn = (t - t.min()) / (np.ptp(t) + 1e-30)
    dn = (d - d.min()) / (np.ptp(d) + 1e-30)
    return front[int(np.argmin(np.hypot(tn, dn)))]
