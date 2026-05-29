"""The coherence field over CR3BP phase space (MASTER_PLAN.md - Stage 25, a falsifiable test).

Takes the project's "coherence" idea seriously and makes it a concrete, computable scalar field,
then tests -- falsifiably -- whether it has predictive value for the natural transport structure.

We operationalize coherence as REGULARITY: a coherent state stays orderly under perturbation; an
incoherent one is near a separatrix and diverges fast. The standard, rigorous local measure of
this is the Fast Lyapunov Indicator (FLI) -- the supremum over a finite time of the log-growth of
a tangent (deviation) vector, read off the state-transition matrix. High FLI = low coherence
(chaotic / near a manifold tube); low FLI = high coherence (regular / KAM-like).

  coherence(state) = -FLI(state)        (higher = more coherent)

The falsifiable claim (H1, tested in validate/stage25.py): the LOW-coherence ridges of this field
coincide with the invariant-manifold tubes computed independently -- i.e. a purely LOCAL measure
locates the transport highways that normally need the global STM/eigenvector machinery.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.cr3bp import eom_stm, pseudo_potential

_INT = dict(method="DOP853", rtol=1e-9, atol=1e-9)


def fli(state, mu, t_max=4.0, n_t=40):
    """Fast Lyapunov Indicator: sup over [0,t_max] of log10 of the max STM stretching."""
    y0 = np.concatenate([np.asarray(state, float), np.eye(6).ravel()])
    sol = solve_ivp(eom_stm, (0.0, t_max), y0, args=(mu,),
                    t_eval=np.linspace(0.0, t_max, n_t), **_INT)
    best = 0.0
    for k in range(sol.y.shape[1]):
        Phi = sol.y[6:, k].reshape(6, 6)
        s = float(np.linalg.svd(Phi, compute_uv=False)[0])   # largest singular value
        if s > 0:
            best = max(best, np.log10(s))
    return best


def _prograde_velocity(x, y, speed):
    """Prograde tangential velocity (about the barycenter) of the given speed."""
    r = np.hypot(x, y)
    if r < 1e-9:
        return 0.0, speed
    return -y / r * speed, x / r * speed


def accessible_speed(x, y, mu, jacobi):
    """Speed from the Jacobi constant at (x,y): v^2 = 2*Omega - C, or None if forbidden."""
    v2 = 2.0 * pseudo_potential([x, y, 0.0, 0.0, 0.0, 0.0], mu) - jacobi
    return float(np.sqrt(v2)) if v2 > 0.0 else None


def coherence_map(mu, jacobi, xlim, ylim, n=40, t_max=4.0):
    """FLI/coherence field over an (x,y) grid at fixed Jacobi energy (prograde velocity).

    Returns (xs, ys, FLI[n,n]); NaN in the energetically forbidden region. coherence = -FLI.
    """
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    F = np.full((n, n), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            sp = accessible_speed(x, y, mu, jacobi)
            if sp is None:
                continue
            vx, vy = _prograde_velocity(x, y, sp)
            F[i, j] = fli([x, y, 0.0, vx, vy, 0.0], mu, t_max=t_max)
    return xs, ys, F


def fli_at_points(points_xy, mu, jacobi, t_max=4.0):
    """FLI at specific (x,y) points (prograde velocity at the energy). Skips forbidden points."""
    out = []
    for x, y in points_xy:
        sp = accessible_speed(x, y, mu, jacobi)
        if sp is None:
            continue
        vx, vy = _prograde_velocity(x, y, sp)
        out.append(fli([x, y, 0.0, vx, vy, 0.0], mu, t_max=t_max))
    return np.array(out)
