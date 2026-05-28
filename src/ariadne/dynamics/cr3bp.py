"""Circular Restricted Three-Body Problem (CR3BP) dynamics.

Rotating (synodic), nondimensional frame. Larger primary (mass 1-mu) at
(-mu, 0, 0); smaller primary (mass mu) at (1-mu, 0, 0). See MASTER_PLAN.md §3.1-3.6.

State vector s = [x, y, z, vx, vy, vz].

Provides:
  - pseudo_potential(s, mu)          Omega
  - jacobi_constant(s, mu)           C = 2*Omega - v^2
  - eom(t, s, mu)                    equations of motion (6-vector)
  - jacobian(s, mu)                  6x6 variational matrix A = d(sdot)/ds
  - eom_stm(t, y, mu)               augmented EOM for state(6)+STM(36) = 42
  - propagate(...)                   integrate the state
  - propagate_stm(...)               integrate state + STM (monodromy etc.)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

_DEFAULT = dict(method="DOP853", rtol=1e-12, atol=1e-12)


def _radii(x, y, z, mu):
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    return r1, r2


def pseudo_potential(s, mu: float) -> float:
    """Effective potential Omega in the rotating frame."""
    x, y, z = s[0], s[1], s[2]
    r1, r2 = _radii(x, y, z, mu)
    return (0.5 * (x * x + y * y)
            + (1.0 - mu) / r1 + mu / r2
            + 0.5 * mu * (1.0 - mu))


def omega_gradient(s, mu: float) -> np.ndarray:
    """Gradient (Omega_x, Omega_y, Omega_z)."""
    x, y, z = s[0], s[1], s[2]
    r1, r2 = _radii(x, y, z, mu)
    r1_3, r2_3 = r1 ** 3, r2 ** 3
    om1 = (1.0 - mu)
    ox = x - om1 * (x + mu) / r1_3 - mu * (x - 1.0 + mu) / r2_3
    oy = y - om1 * y / r1_3 - mu * y / r2_3
    oz = -om1 * z / r1_3 - mu * z / r2_3
    return np.array([ox, oy, oz])


def jacobi_constant(s, mu: float) -> float:
    """Jacobi integral C = 2*Omega - v^2 (conserved in the CR3BP)."""
    v2 = s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    return 2.0 * pseudo_potential(s, mu) - v2


def eom(t, s, mu: float) -> np.ndarray:
    """CR3BP equations of motion. sdot = f(s)."""
    ox, oy, oz = omega_gradient(s, mu)
    vx, vy, vz = s[3], s[4], s[5]
    return np.array([vx, vy, vz,
                     2.0 * vy + ox,
                     -2.0 * vx + oy,
                     oz])


def _omega_hessian(s, mu: float) -> np.ndarray:
    """Second partials of Omega (the U block of the variational matrix)."""
    x, y, z = s[0], s[1], s[2]
    r1, r2 = _radii(x, y, z, mu)
    om1 = 1.0 - mu
    r1_3, r2_3 = r1 ** 3, r2 ** 3
    r1_5, r2_5 = r1 ** 5, r2 ** 5
    a = x + mu
    b = x - 1.0 + mu

    Uxx = 1.0 - om1 / r1_3 - mu / r2_3 + 3 * om1 * a * a / r1_5 + 3 * mu * b * b / r2_5
    Uyy = 1.0 - om1 / r1_3 - mu / r2_3 + 3 * om1 * y * y / r1_5 + 3 * mu * y * y / r2_5
    Uzz = -om1 / r1_3 - mu / r2_3 + 3 * om1 * z * z / r1_5 + 3 * mu * z * z / r2_5
    Uxy = 3 * om1 * a * y / r1_5 + 3 * mu * b * y / r2_5
    Uxz = 3 * om1 * a * z / r1_5 + 3 * mu * b * z / r2_5
    Uyz = 3 * om1 * y * z / r1_5 + 3 * mu * y * z / r2_5

    return np.array([[Uxx, Uxy, Uxz],
                     [Uxy, Uyy, Uyz],
                     [Uxz, Uyz, Uzz]])


def jacobian(s, mu: float) -> np.ndarray:
    """6x6 variational matrix A = d(sdot)/ds along state s."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = _omega_hessian(s, mu)
    # Coriolis block
    A[3, 4] = 2.0
    A[4, 3] = -2.0
    return A


def eom_stm(t, y, mu: float) -> np.ndarray:
    """Augmented EOM: state (6) + flattened STM (36). Phidot = A @ Phi."""
    s = y[:6]
    Phi = y[6:].reshape(6, 6)
    sdot = eom(t, s, mu)
    Phidot = jacobian(s, mu) @ Phi
    return np.concatenate([sdot, Phidot.ravel()])


def propagate(s0, t_span, mu: float, t_eval=None, **kwargs):
    """Integrate the CR3BP state. Returns a scipy solve_ivp result.

    s0 : initial state (6,). t_span : (t0, tf) nondimensional.
    """
    opts = {**_DEFAULT, **kwargs}
    return solve_ivp(eom, t_span, np.asarray(s0, dtype=float),
                     args=(mu,), t_eval=t_eval, **opts)


def propagate_stm(s0, t_span, mu: float, t_eval=None, **kwargs):
    """Integrate state + STM. Returns (sol, stm_final) where stm_final is 6x6.

    The monodromy matrix is propagate_stm over exactly one orbit period.
    """
    opts = {**_DEFAULT, **kwargs}
    y0 = np.concatenate([np.asarray(s0, dtype=float), np.eye(6).ravel()])
    sol = solve_ivp(eom_stm, t_span, y0, args=(mu,), t_eval=t_eval, **opts)
    stm_final = sol.y[6:, -1].reshape(6, 6)
    return sol, stm_final
