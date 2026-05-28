"""Linear analysis at the collinear libration points (MASTER_PLAN.md §3.5).

Provides the in-plane oscillation frequency omega (the Lyapunov-orbit frequency),
the real unstable rate lambda, the amplitude ratio kappa, and the vertical
frequency omega_v — and a linear first guess for the Lyapunov differential
corrector.

In-plane linearized characteristic equation (collinear, U_xy = 0):
    Lambda^2 + (4 - Uxx - Uyy) Lambda + Uxx*Uyy = 0,   Lambda = lambda^2
one root > 0 (real saddle, rate lambda), one root < 0 (center, omega = sqrt(-Lambda)).
"""
from __future__ import annotations

import numpy as np

from ..dynamics.cr3bp import _omega_hessian
from .lagrange import lagrange_points


def collinear_linear_modes(mu: float, point: str = "L1") -> dict:
    """Linear modal data at a collinear point ('L1','L2','L3')."""
    s = lagrange_points(mu)[point]
    H = _omega_hessian(s, mu)
    Uxx, Uyy, Uzz = H[0, 0], H[1, 1], H[2, 2]
    b = 4.0 - Uxx - Uyy
    disc = b * b - 4.0 * Uxx * Uyy
    sq = np.sqrt(disc)
    lam_center = (-b - sq) / 2.0   # negative root -> center
    lam_saddle = (-b + sq) / 2.0   # positive root -> saddle
    omega = np.sqrt(-lam_center)
    lam = np.sqrt(lam_saddle)
    kappa = (omega ** 2 + Uxx) / (2.0 * omega)
    omega_v = np.sqrt(-Uzz)        # vertical (out-of-plane) frequency, Uzz < 0
    return {
        "point": point, "xL": float(s[0]),
        "Uxx": float(Uxx), "Uyy": float(Uyy), "Uzz": float(Uzz),
        "omega": float(omega), "lambda": float(lam),
        "kappa": float(kappa), "omega_v": float(omega_v),
        "linear_period": float(2.0 * np.pi / omega),
    }


def linear_lyapunov_guess(mu: float, point: str, amplitude: float):
    """Linear initial guess for a planar Lyapunov orbit of x-amplitude `amplitude`.

    Returns (state6, linear_period). The state is on the x-axis with a
    perpendicular crossing (vx = 0), suitable for the symmetric corrector.
    """
    m = collinear_linear_modes(mu, point)
    x0 = m["xL"] - amplitude
    vy0 = m["kappa"] * amplitude * m["omega"]
    state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
    return state, m["linear_period"]
