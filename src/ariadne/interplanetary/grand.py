"""The unified multi-objective grand optimizer (MASTER_PLAN.md - Stage 23).

The synthesis: not "minimum Delta-v" or "minimum time" alone, but a route chosen by BALANCING
three objectives -- energy (Delta-v), time (TOF), and ROBUSTNESS (insensitivity to launch-date
slip) -- with the coherence principle as the balancer. This is the operational meaning of "the
most coherent route": the point of the trade space that is best across all three at once, not an
extreme on any single axis.

  - window_sensitivity : m/s of extra Delta-v per day of launch slip -- the robustness axis
    (a sharp porkchop tip is fragile; a broad basin is robust/coherent).
  - coherence_score     : 1 minus the weighted utopia-distance over the normalized objectives.
    Weights let a mission dial time vs energy vs robustness; sweeping them traces the choice.

Also contrasts propulsion regimes: a high-thrust (impulsive Lambert) transfer vs a low-thrust
heliocentric estimate (Edelbaum). HONEST: the Edelbaum figure is the heliocentric orbit-change
Delta-v only (it omits the Earth-escape and target-capture spirals and inclination); low-thrust's
real edge is high specific impulse -> low propellant mass and launch flexibility, NOT lower Delta-v.
"""
from __future__ import annotations

import math

import numpy as np

from ..data.constants import GM_SUN
from ..data.ephemeris import body_state, utc
from .porkchop import lambert_transfer, porkchop, time_energy_pareto

DAY = 86400.0


def window_sensitivity(dep_body, arr_body, et_dep, tof_days, dslip_days=5.0, **kw):
    """Extra total Delta-v per day of launch-date slip (m/s/day). Lower = more robust."""
    base = lambert_transfer(dep_body, arr_body, et_dep, tof_days, **kw)
    if base is None:
        return math.inf
    slips = []
    for s in (-dslip_days, dslip_days):
        r = lambert_transfer(dep_body, arr_body, et_dep + s * DAY, tof_days, **kw)
        if r is not None:
            slips.append(abs(r["total_ms"] - base["total_ms"]) / abs(s))
    return float(np.mean(slips)) if slips else math.inf


def build_tradeoff(dep_body, arr_body, et_start, dep_days=540, tof_range=(120, 400),
                   n_dep=50, n_tof=40, **kw):
    """The (energy, time, robustness) trade space: the time/energy Pareto + a robustness axis."""
    pk = porkchop(dep_body, arr_body, et_start, dep_days, tof_range, n_dep, n_tof, **kw)
    front = time_energy_pareto(pk)
    for p in front:
        p["sensitivity_ms_per_day"] = window_sensitivity(
            dep_body, arr_body, p["dep_et"], p["tof_days"], **kw)
        p["utc_dep"] = utc(p["dep_et"])
    return front


def _normalize(vals):
    v = np.asarray(vals, float)
    lo, span = v.min(), (np.ptp(v) + 1e-30)
    return (v - lo) / span


def rank_by_coherence(tradeoff, weights=(1.0, 1.0, 1.0)):
    """Rank routes by the coherence score (balance across energy, time, robustness).

    weights = (w_energy, w_time, w_robust). Returns the tradeoff list with a 'coherence' field,
    sorted best-first. The most coherent route is the one nearest the (normalized) utopia corner.
    """
    if not tradeoff:
        return []
    en = _normalize([p["total_ms"] for p in tradeoff])
    tn = _normalize([p["tof_days"] for p in tradeoff])
    rn = _normalize([p["sensitivity_ms_per_day"] for p in tradeoff])
    we, wt, wr = weights
    wsum = we + wt + wr
    for k, p in enumerate(tradeoff):
        dist = math.sqrt((we * en[k] ** 2 + wt * tn[k] ** 2 + wr * rn[k] ** 2) / wsum)
        p["coherence"] = 1.0 - dist
    return sorted(tradeoff, key=lambda p: -p["coherence"])


def most_coherent_route(tradeoff, weights=(1.0, 1.0, 1.0)):
    ranked = rank_by_coherence(tradeoff, weights)
    return ranked[0] if ranked else None


def low_thrust_heliocentric(dep_body, arr_body, et_dep, accel_mm_s2=0.2):
    """Edelbaum heliocentric orbit-change estimate for a low-thrust transfer.

    Delta-v = |v_dep_helio - v_arr_helio| (coplanar circular); time = Delta-v / a_T. HONEST: this
    is the heliocentric leg only -- it omits Earth-escape and target-capture spirals and inclination.
    """
    rd = float(np.linalg.norm(body_state(dep_body, et_dep, "J2000", "SUN")[:3]))
    ra = float(np.linalg.norm(body_state(arr_body, et_dep, "J2000", "SUN")[:3]))
    vd = math.sqrt(GM_SUN / rd)
    va = math.sqrt(GM_SUN / ra)
    dv = abs(vd - va)                                  # km/s, Edelbaum coplanar
    accel = accel_mm_s2 * 1e-6                          # mm/s^2 -> km/s^2
    time_days = (dv / accel) / DAY
    return {"dv_ms": dv * 1000.0, "tof_days": time_days, "accel_mm_s2": accel_mm_s2,
            "note": "heliocentric orbit-change only; excludes escape/capture spirals + inclination"}
