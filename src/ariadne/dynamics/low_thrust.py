"""Low-thrust (continuous-acceleration) CR3BP dynamics (MASTER_PLAN.md — Stage 13).

A different propulsion regime: instead of impulsive burns, a small continuous thrust
acceleration steers the spacecraft over a long arc. This changes the optimization
landscape (and is where "ride the field gradients" has the most teeth -- you can
continuously align thrust with the dynamical flow). EOM = CR3BP + a_T * u_hat.

Energy bookkeeping: the Jacobi constant is no longer conserved. For tangential thrust
(u = v/|v|), dC/dt = -2 * a_T * |v|, so C decreases monotonically (energy rises) -- a
spiral. The effective Delta-v is the time integral of the thrust acceleration.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .cr3bp import eom, jacobi_constant

_INT = dict(method="DOP853", rtol=1e-11, atol=1e-12)


def thrust_direction(s, mode="tangential"):
    v = s[3:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros(3)
    if mode == "tangential":
        return v / nv
    if mode == "antitangential":
        return -v / nv
    raise ValueError(f"unknown thrust mode {mode}")


def eom_low_thrust(t, s, mu, accel, mode="tangential"):
    ds = eom(t, s, mu)
    ds[3:] += accel * thrust_direction(s, mode)
    return ds


def propagate_low_thrust(s0, t_span, mu, accel, mode="tangential", t_eval=None, **kw):
    opts = {**_INT, **kw}
    return solve_ivp(eom_low_thrust, t_span, np.asarray(s0, float),
                     args=(mu, accel, mode), t_eval=t_eval, **opts)


def delta_v(accel, t_span_nondim, v_star):
    """Effective Delta-v (km/s) = thrust acceleration * burn time."""
    return accel * (t_span_nondim[1] - t_span_nondim[0]) * v_star


def energy_gain_rate_check(s, mu, accel):
    """Predicted dC/dt for tangential thrust (= -2 a_T |v|) -- used to validate the dynamics."""
    return -2.0 * accel * np.linalg.norm(s[3:])
