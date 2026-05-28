"""High-fidelity ephemeris dynamics (MASTER_PLAN.md §3.11 rung 5).

Two propagators, both using real JPL DE440 positions from SPICE:

  - propagate_test_particle: a massless spacecraft in a body-centered inertial
    (J2000) frame, under the central body's point mass plus third-body
    perturbations (with the indirect term for the accelerating frame). This is
    the production propagator for trajectory work.

  - propagate_nbody: a self-consistent mutual gravitation integration of several
    massive bodies (e.g. Sun, Earth, Moon) in the SSB frame, plus optional
    external point-mass perturbers whose positions come from SPICE. Used to
    validate the dynamics against the ephemeris itself.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..data.ephemeris import body_gm, body_pos, body_state

_INT = dict(method="DOP853", rtol=1e-11, atol=1e-7)


# --------------------------------------------------------------------------- #
# Test-particle propagator (body-centered inertial)
# --------------------------------------------------------------------------- #
def test_particle_accel(et_t, r, gm_central, perturbers, gm_pert, center):
    a = -gm_central * r / np.linalg.norm(r) ** 3
    for body, gmb in zip(perturbers, gm_pert):
        s = body_pos(body, et_t, "J2000", center)      # body rel. to center
        d = s - r
        a += gmb * (d / np.linalg.norm(d) ** 3 - s / np.linalg.norm(s) ** 3)
    return a


def propagate_test_particle(r0, v0, et0, t_span, central="EARTH",
                            perturbers=("SUN", "MOON"), t_eval=None, **kwargs):
    """Propagate a spacecraft (km, km/s) in the `central`-centered J2000 frame.

    t_span is in seconds relative to et0. Perturbers' positions are queried from
    SPICE at each evaluation.
    """
    gm_central = body_gm(central)
    gm_pert = [body_gm(b) for b in perturbers]

    def rhs(t, y):
        a = test_particle_accel(et0 + t, y[:3], gm_central,
                                perturbers, gm_pert, central)
        return np.concatenate([y[3:], a])

    opts = {**_INT, **kwargs}
    y0 = np.concatenate([np.asarray(r0, float), np.asarray(v0, float)])
    return solve_ivp(rhs, t_span, y0, t_eval=t_eval, **opts)


# --------------------------------------------------------------------------- #
# Mutual n-body propagator (SSB inertial) -- for validation against SPICE
# --------------------------------------------------------------------------- #
def propagate_nbody(bodies, et0, t_span, external=(), t_eval=None, **kwargs):
    """Integrate the mutual gravitation of `bodies` (massive) from SPICE ICs.

    bodies   : list of SPICE names integrated self-consistently (e.g.
               ["SUN","EARTH","MOON"]).
    external : additional perturbers (positions from SPICE, treated as fixed-mass
               sources at their ephemeris positions), e.g. planet barycenters.
    Returns (sol, gms) where sol.y is [r1..rK, v1..vK] in SSB J2000 (km, km/s).
    """
    gms = np.array([body_gm(b) for b in bodies])
    gms_ext = [body_gm(b) for b in external]
    k = len(bodies)

    y0 = np.concatenate(
        [body_state(b, et0, "J2000", "SSB")[:3] for b in bodies]
        + [body_state(b, et0, "J2000", "SSB")[3:] for b in bodies])

    def rhs(t, y):
        R = y[:3 * k].reshape(k, 3)
        acc = np.zeros((k, 3))
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                d = R[j] - R[i]
                acc[i] += gms[j] * d / np.linalg.norm(d) ** 3
            for b, gmb in zip(external, gms_ext):
                sb = body_pos(b, et0 + t, "J2000", "SSB")
                d = sb - R[i]
                acc[i] += gmb * d / np.linalg.norm(d) ** 3
        return np.concatenate([y[3 * k:], acc.ravel()])

    opts = {**_INT, **kwargs}
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, **opts)
    return sol, gms


def nbody_position(sol, k, index, t_index):
    """Extract body `index`'s position (km) from a propagate_nbody solution."""
    return sol.y[3 * index:3 * index + 3, t_index]
