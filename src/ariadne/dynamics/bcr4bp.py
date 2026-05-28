"""Bicircular Restricted Four-Body Problem (BCR4BP). See MASTER_PLAN.md §3.11.

Earth-Moon rotating (synodic) frame, with the Sun added on a circular orbit about
the Earth-Moon barycenter. This is the lowest-fidelity model that captures the
solar perturbation responsible for low-energy lunar capture.

Solar acceleration in the synodic frame (origin at the EM barycenter, which itself
accelerates toward the Sun):

    a_sun = -m_S (R - R_S)/|R - R_S|^3   (direct attraction)
            - m_S R_S / a_S^3            (indirect / frame-acceleration term)

so a_sun -> 0 at the barycenter and reduces to a tidal field across the EM system.
With m_S = 0 the EOM are identical to the CR3BP.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from ..data.constants import AU_KM, GM_SUN, System
from .cr3bp import eom

_INT = dict(method="DOP853", rtol=1e-12, atol=1e-12)


def sun_params(system: System) -> dict:
    """BCR4BP solar parameters for a given Earth-Moon-like system (nondim).

    m_S  : solar mass in system-mass units,
    a_S  : Sun distance in characteristic lengths,
    omega_S : Sun angular rate in the synodic frame (rad / nondim time);
              negative (the Sun appears to move retrograde), period ~ 29.5 d.
    """
    m_S = GM_SUN / system.gm_total
    a_S = AU_KM / system.L_star
    # inertial solar mean motion (nondim) = T* * sqrt(GM_sun / AU^3)
    n_S = system.T_star * math.sqrt(GM_SUN / AU_KM ** 3)
    omega_S = n_S - 1.0
    return {"m_S": m_S, "a_S": a_S, "omega_S": omega_S}


def sun_position(t: float, a_S: float, omega_S: float, theta0: float = 0.0) -> np.ndarray:
    ang = theta0 + omega_S * t
    return np.array([a_S * math.cos(ang), a_S * math.sin(ang), 0.0])


def solar_acceleration(R, t, m_S, a_S, omega_S, theta0=0.0) -> np.ndarray:
    """Solar perturbing acceleration on the spacecraft (3-vector)."""
    R_S = sun_position(t, a_S, omega_S, theta0)
    d = np.asarray(R[:3], float) - R_S
    dist3 = (d @ d) ** 1.5
    return -m_S * d / dist3 - m_S * R_S / a_S ** 3


def eom_bcr4bp(t, s, mu, m_S, a_S, omega_S, theta0=0.0) -> np.ndarray:
    """BCR4BP equations of motion (CR3BP base + solar perturbation)."""
    ds = eom(t, s, mu)
    ds[3:] += solar_acceleration(s[:3], t, m_S, a_S, omega_S, theta0)
    return ds


def propagate_bcr4bp(s0, t_span, mu, m_S, a_S, omega_S, theta0=0.0,
                     t_eval=None, **kwargs):
    """Integrate the BCR4BP state."""
    opts = {**_INT, **kwargs}
    return solve_ivp(eom_bcr4bp, t_span, np.asarray(s0, float),
                     args=(mu, m_S, a_S, omega_S, theta0), t_eval=t_eval, **opts)
