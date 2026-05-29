"""Sun-assisted low-energy / weak-stability-boundary (WSB) lunar transfer
(MASTER_PLAN.md Stage 10).

A direct Earth->Moon transfer arrives hyperbolically (high v_infinity, expensive
lunar-orbit insertion). A low-energy transfer instead lets solar perturbation lower the
lunar-arrival energy over a longer, multi-revolution flight, enabling a cheaper capture.

We construct it the way Belbruno found ballistic captures: BACKWARD from a near-ballistic
lunar capture. A capture state at low lunar periapsis is propagated backward in the full
DE440 ephemeris (Earth + Sun + Moon point masses); its velocity vector, periapsis phase,
and capture energy are optimized so the backward arc returns to a perigee at LEO altitude
while MINIMIZING the total LEO->LLO Delta-v. The forward transfer is that arc reversed.

Result (epoch 2025-11-12): a LEO-departing transfer at 3,907 m/s (TOF 48.8 d) -- below the
direct transfer (3,953) and below the published Coimbra 3,925 m/s, at the cost of a longer
flight time. NOTE: the converged solution is a MULTI-REVOLUTION, Sun-perturbed low-energy
transfer with apogee near lunar distance (~2-3 revs), NOT a deep Sun-Earth-L1/L2 exterior
(1.5e6 km) WSB -- both are low-energy, this is the basin the optimizer found. CAVEATS: a
two-impulse patched model; the WSB region is chaotic so the solution is stored as a fixed,
deterministically-reproducible state (SOLUTION_PARAMS); a flight-grade design would add
finite burns + navigation. Found from the dynamics, not fitted.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from ..data.constants import EARTH_MOON, GM_EARTH, GM_MOON, R_EARTH, R_MOON
from ..data.ephemeris import body_state, et
from ..dynamics.ephemeris_nbody import propagate_test_particle

COIMBRA_MS = 3925.0


def _rodrigues(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(angle) + np.cross(axis, v) * math.sin(angle)
            + axis * np.dot(axis, v) * (1.0 - math.cos(angle)))


def _frame(epoch_et):
    st = body_state("MOON", epoch_et, "J2000", "EARTH")
    mp, mv = st[:3], st[3:]
    n = np.cross(mp, mv); n /= np.linalg.norm(n)
    e1 = mp / np.linalg.norm(mp)
    e2 = np.cross(n, e1)
    return mp, mv, n, e1, e2


def _capture_state(epoch_et, frac, alpha, beta, phi_deg, r_peri, mp, mv, n, e1, e2):
    v_esc = math.sqrt(2.0 * GM_MOON / r_peri)
    phi = math.radians(phi_deg)
    u = math.cos(phi) * e1 + math.sin(phi) * e2
    w = np.cross(n, u); w /= np.linalg.norm(w)
    w = _rodrigues(w, n, alpha)
    w = _rodrigues(w, u, beta)
    pos = mp + r_peri * u
    vel = mv + frac * v_esc * w
    return pos, vel, v_esc


def _evaluate(params, epoch_et, frame, r_peri, r_leo, v_circ_llo, t_back):
    mp, mv, n, e1, e2 = frame
    frac, alpha, beta, phi = params
    pos, vel, v_esc = _capture_state(epoch_et, frac, alpha, beta, phi, r_peri,
                                     mp, mv, n, e1, e2)
    sol = propagate_test_particle(pos, vel, epoch_et, (0.0, -t_back * 86400.0),
                                  perturbers=("SUN", "MOON"),
                                  t_eval=np.linspace(0.0, -t_back * 86400.0, 700))
    d = np.sqrt(sol.y[0] ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2)
    i = int(np.argmin(d))
    r_pe, v_pe = float(d[i]), float(np.linalg.norm(sol.y[3:, i]))
    dv_tli = v_pe - math.sqrt(GM_EARTH / max(r_pe, R_EARTH))
    dv_loi = max(frac * v_esc - v_circ_llo, 0.0)
    v_inf = math.sqrt(max(frac ** 2 - 1.0, 0.0)) * v_esc
    return {"r_pe": r_pe, "tli": dv_tli, "loi": dv_loi,
            "total_ms": (dv_tli + dv_loi) * 1000.0, "v_inf": v_inf,
            "tof_days": abs(sol.t[i]) / 86400.0, "sol": sol, "peri_index": i}


#: the converged Stage-10 WSB solution (epoch 2025-11-12) -> 3,907 m/s, deterministically.
#: FULL PRECISION is required: the WSB trajectory is in a chaotic region, so 4-decimal
#: rounding shifts the Earth perigee by thousands of km. evaluate_transfer(SOLUTION_PARAMS)
#: reproduces 3,907 m/s exactly; this fixed state is the canonical, reproducible artifact.
SOLUTION_PARAMS = [1.0497739028321098, 0.2949913490297107,
                   5.373891049016198e-05, 246.77973994414444]


def evaluate_transfer(params=None, epoch: str = "2025-11-12T00:00:00",
                      leo_alt: float = 200.0, llo_alt: float = 100.0,
                      t_back: float = 65.0) -> dict:
    """Evaluate one WSB transfer for given capture params (fast, no optimization)."""
    if params is None:
        params = SOLUTION_PARAMS
    epoch_et = et(epoch)
    frame = _frame(epoch_et)
    r_peri = R_MOON + llo_alt
    r_leo = R_EARTH + leo_alt
    v_circ_llo = math.sqrt(GM_MOON / r_peri)
    r = _evaluate(params, epoch_et, frame, r_peri, r_leo, v_circ_llo, t_back)
    return {"perigee_alt_km": r["r_pe"] - R_EARTH, "v_inf": r["v_inf"],
            "tli_ms": r["tli"] * 1000.0, "loi_ms": r["loi"] * 1000.0,
            "total_ms": r["total_ms"], "tof_days": r["tof_days"],
            "params": list(params), "epoch": epoch, "coimbra_ms": COIMBRA_MS}


def wsb_transfer(epoch: str = "2025-11-12T00:00:00", leo_alt: float = 200.0,
                 llo_alt: float = 100.0, starts=None, maxiter: int = 260,
                 t_back: float = 65.0):
    """Optimize a LEO-departing WSB lunar transfer. Returns the best transfer dict."""
    epoch_et = et(epoch)
    frame = _frame(epoch_et)
    r_peri = R_MOON + llo_alt
    r_leo = R_EARTH + leo_alt
    v_circ_llo = math.sqrt(GM_MOON / r_peri)

    if starts is None:
        # lead with the known converged solution (fast, reliable), then fallbacks
        starts = [[1.0498, 0.295, 0.0001, 246.78], [1.05, 0.0, 0.0, 247.0],
                  [1.0, 0.3, 0.0, 247.0], [1.02, -0.3, 0.2, 250.0]]

    def obj(p):
        r = _evaluate(p, epoch_et, frame, r_peri, r_leo, v_circ_llo, t_back)
        return r["total_ms"] + 2.0 * max(0.0, abs(r["r_pe"] - r_leo) - 50.0)

    best, best_obj = None, np.inf
    for s in starts:
        res = minimize(obj, s, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 0.5})
        r = _evaluate(res.x, epoch_et, frame, r_peri, r_leo, v_circ_llo, t_back)
        if abs(r["r_pe"] - r_leo) < 2000.0 and r["total_ms"] < best_obj:
            best, best_obj = {**r, "params": res.x.tolist(), "epoch": epoch}, r["total_ms"]

    if best is None:
        return None
    best["perigee_alt_km"] = best["r_pe"] - R_EARTH
    best["tli_ms"] = best["tli"] * 1000.0
    best["loi_ms"] = best["loi"] * 1000.0
    best["coimbra_ms"] = COIMBRA_MS
    return best


def transfer_trajectory(best, epoch: str = "2025-11-12T00:00:00",
                        llo_alt: float = 100.0, t_back: float = 65.0):
    """Re-propagate the best WSB transfer; return (t, Y) of the forward trajectory
    (LEO -> ... -> lunar capture), in Earth-centered J2000 (km)."""
    epoch_et = et(epoch)
    frame = _frame(epoch_et)
    r_peri = R_MOON + llo_alt
    v_circ_llo = math.sqrt(GM_MOON / r_peri)
    r = _evaluate(best["params"], epoch_et, frame, r_peri, r_peri, v_circ_llo, t_back)
    sol = r["sol"]
    # backward arc -> reverse for forward (LEO first)
    return sol.t[::-1], sol.y[:, ::-1]
