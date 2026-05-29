"""Ballistic lunar capture via invariant manifolds (MASTER_PLAN.md §3.12, Stage 6).

A libration-orbit unstable manifold delivers a spacecraft to a low periapsis about
the Moon with NO insertion burn beyond circularization. This module finds the
manifold trajectory whose lunar periapsis sits at the target low-lunar-orbit (LLO)
altitude and computes the real Moon-relative speed there (rotating-frame state ->
Moon-centered inertial), hence the true ballistic-capture insertion Delta-v.

This is the low-energy lunar-orbit insertion computed from REAL CR3BP dynamics,
not a vis-viva estimate.
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON, GM_MOON, R_MOON
from ..manifolds.manifold import manifold_seeds, manifold_trajectory


def moon_relative_speed(state, mu: float, v_star: float) -> float:
    """Moon-centered inertial speed (km/s) from an Earth-Moon rotating-frame state."""
    x, y, z, vx, vy, vz = state
    rho = np.array([x - (1.0 - mu), y, z])
    v_rot = np.array([vx, vy, vz])
    omega = np.array([0.0, 0.0, 1.0])
    v_inertial = v_rot + np.cross(omega, rho)      # add frame rotation
    return float(np.linalg.norm(v_inertial) * v_star)


def _lunar_periapsis(mu, Y):
    """Index, distance(nondim) of closest approach to the Moon along a trajectory."""
    d = np.sqrt((Y[0] - (1.0 - mu)) ** 2 + Y[1] ** 2 + Y[2] ** 2)
    i = int(np.argmin(d))
    return i, float(d[i])


def ballistic_capture(orbit, llo_alt: float = 100.0, system=EARTH_MOON,
                      n_seeds: int = 200, displacement: float = 1e-4,
                      t_max: float = 6.0):
    """Find the unstable-manifold trajectory that ballistically reaches LLO altitude.

    Returns the best transfer (dict) with the real Moon-relative periapsis speed,
    the circularization Delta-v, and the coast time of flight, or None.
    """
    mu, L, V, T = system.mu, system.L_star, system.V_star, system.T_star
    r_llo = R_MOON + llo_alt
    r_llo_nd = r_llo / L
    v_circ_llo = np.sqrt(GM_MOON / r_llo)

    best = None
    for branch in (+1, -1):
        seeds, lam = manifold_seeds(mu, orbit, n_seeds=n_seeds,
                                    displacement=displacement, stable=False,
                                    branch=branch)
        for s in seeds:
            t, Y = manifold_trajectory(mu, s, stable=False, t_max=t_max, n=1200)
            i, d_nd = _lunar_periapsis(mu, Y)
            # interested in trajectories whose periapsis is at/below LLO (capturable)
            if d_nd * L > 0.6 * r_llo and d_nd * L < 4.0 * r_llo:
                # closeness of this periapsis to the LLO target
                err = abs(d_nd - r_llo_nd)
                state = Y[:, i]
                v_peri = moon_relative_speed(state, mu, V)
                dv = v_peri - v_circ_llo
                cand = {"branch": branch, "periapsis_km": d_nd * L,
                        "periapsis_alt_km": d_nd * L - R_MOON,
                        "v_peri_kms": v_peri, "v_circ_llo_kms": v_circ_llo,
                        "dv_capture_kms": dv, "tof_days": abs(t[i]) * T / 86400.0,
                        "match_err_km": err * L, "state": state,
                        "seed": s, "peri_index": i}
                if best is None or cand["match_err_km"] < best["match_err_km"]:
                    best = cand
    return best
