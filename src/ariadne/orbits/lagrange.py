"""Lagrange (libration) point solver for the CR3BP. See MASTER_PLAN.md §3.4.

Collinear points L1, L2, L3 are found by root-finding Omega_x = 0 on the x-axis;
triangular points L4, L5 are exact.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..dynamics.cr3bp import omega_gradient, jacobi_constant


def _omega_x_axis(x: float, mu: float) -> float:
    """Omega_x evaluated on the x-axis (y = z = 0)."""
    a = x + mu
    b = x - 1.0 + mu
    return x - (1.0 - mu) * a / abs(a) ** 3 - mu * b / abs(b) ** 3


def lagrange_points(mu: float, tol: float = 1e-14) -> dict[str, np.ndarray]:
    """Return the five Lagrange points as full states (position, zero velocity).

    Keys: 'L1','L2','L3','L4','L5'. Each value is a length-6 array.
    """
    d = 1e-4  # offset to avoid the singularities at the primaries
    # L1: between the primaries  (-mu, 1-mu)
    x_l1 = brentq(_omega_x_axis, -mu + d, 1.0 - mu - d, args=(mu,), xtol=tol)
    # L2: beyond the smaller primary  (x > 1-mu)
    x_l2 = brentq(_omega_x_axis, 1.0 - mu + d, 2.0, args=(mu,), xtol=tol)
    # L3: beyond the larger primary  (x < -mu)
    x_l3 = brentq(_omega_x_axis, -2.0, -mu - d, args=(mu,), xtol=tol)

    def state(x, y):
        return np.array([x, y, 0.0, 0.0, 0.0, 0.0])

    sqrt3_2 = np.sqrt(3.0) / 2.0
    return {
        "L1": state(x_l1, 0.0),
        "L2": state(x_l2, 0.0),
        "L3": state(x_l3, 0.0),
        "L4": state(0.5 - mu, sqrt3_2),
        "L5": state(0.5 - mu, -sqrt3_2),
    }


def lagrange_jacobi(mu: float) -> dict[str, float]:
    """Jacobi constant at each Lagrange point (zero velocity, so C = 2*Omega)."""
    return {k: jacobi_constant(s, mu) for k, s in lagrange_points(mu).items()}


def gradient_residual(mu: float) -> dict[str, float]:
    """|grad Omega| at each Lagrange point. Should be ~0 (internal correctness)."""
    return {k: float(np.linalg.norm(omega_gradient(s, mu)))
            for k, s in lagrange_points(mu).items()}
