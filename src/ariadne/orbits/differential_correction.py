"""Differential correction for periodic orbits (MASTER_PLAN.md §3.7).

Planar Lyapunov orbits via the symmetric single-shooting method: start on the
x-axis with a perpendicular crossing (y=0, vx=0), integrate to the next x-axis
crossing, and Newton-correct (using the STM) so that the crossing is also
perpendicular (vx=0). The full period is twice the crossing time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.cr3bp import eom, eom_stm, jacobi_constant, propagate_stm

_INT = dict(method="DOP853", rtol=1e-12, atol=1e-12)


@dataclass
class PeriodicOrbit:
    s0: np.ndarray              # initial state (perpendicular x-axis crossing)
    period: float
    jacobi: float
    point: str = ""
    family: str = "lyapunov"
    half_period_residual: float = 0.0   # |vx| at the half-period crossing
    meta: dict = field(default_factory=dict)


def _y_crossing_event():
    def ev(t, y, mu):
        return y[1]
    ev.terminal = False
    ev.direction = -1.0   # y returning through zero (started with vy > 0)
    return ev


def correct_lyapunov(mu: float, guess, period_guess: float,
                     tol: float = 1e-11, max_iter: int = 50) -> PeriodicOrbit:
    """Correct a planar Lyapunov guess to a periodic orbit.

    Holds x0 fixed (sets amplitude); varies vy0 to null vx at the half-period
    crossing. Returns a PeriodicOrbit. Raises RuntimeError on non-convergence.
    """
    s = np.asarray(guess, dtype=float).copy()
    ev = _y_crossing_event()
    max_t = 1.4 * period_guess
    t_min = 0.25 * period_guess   # reject the spurious t~0 crossing

    for _ in range(max_iter):
        y0 = np.concatenate([s, np.eye(6).ravel()])
        sol = solve_ivp(eom_stm, (0.0, max_t), y0, args=(mu,),
                        events=ev, **_INT)
        tev, yev = sol.t_events[0], sol.y_events[0]
        mask = tev > t_min
        if not mask.any():
            max_t *= 1.6
            continue
        idx = int(np.argmax(mask))   # first real crossing past t_min
        te = tev[idx]
        ye = yev[idx]
        s_cross = ye[:6]
        Phi = ye[6:].reshape(6, 6)

        vx_f = s_cross[3]
        if abs(vx_f) < tol:
            period = 2.0 * te
            return PeriodicOrbit(
                s0=s.copy(), period=float(period),
                jacobi=float(jacobi_constant(s, mu)),
                half_period_residual=float(abs(vx_f)),
            )

        sdot = eom(te, s_cross, mu)
        xddot = sdot[3]
        ydot = s_cross[4]
        # dvx_f = (Phi[3,4] - xddot/ydot * Phi[1,4]) dvy0,  set dvx_f = -vx_f
        denom = Phi[3, 4] - (xddot / ydot) * Phi[1, 4]
        s[4] += -vx_f / denom

    raise RuntimeError("Lyapunov corrector failed to converge")


def monodromy(mu: float, orbit: PeriodicOrbit) -> np.ndarray:
    """Monodromy matrix M = Phi(T) for a periodic orbit."""
    _, M = propagate_stm(orbit.s0, (0.0, orbit.period), mu)
    return M


def stability_indices(M: np.ndarray) -> dict:
    """Stability indices from the monodromy matrix of a planar Lyapunov orbit.

    Because the planar orbit keeps z = vz = 0, the monodromy decouples into an
    in-plane (x,y,vx,vy) block and an out-of-plane (z,vz) block. The vertical
    index nu_v = 0.5*trace(out-of-plane block); the halo orbit bifurcates from
    the Lyapunov family where nu_v passes through +1.
    """
    iz = [2, 5]
    Bv = M[np.ix_(iz, iz)]
    nu_v = 0.5 * float(np.trace(Bv))

    ip = [0, 1, 3, 4]
    Bp = M[np.ix_(ip, ip)]
    eig = np.linalg.eigvals(Bp)
    # drop the trivial pair nearest 1, take the dominant remaining multiplier
    order = np.argsort(np.abs(eig - 1.0))
    rest = eig[order[2:]]
    lam = rest[np.argmax(np.abs(rest))]
    nu_p = 0.5 * float(np.real(lam + 1.0 / lam))
    return {"nu_vertical": nu_v, "nu_planar": nu_p,
            "max_multiplier": float(np.max(np.abs(eig)))}
