"""Near-Rectilinear Halo Orbits via pseudo-arclength continuation (MASTER_PLAN.md - Stage 19).

NRHOs are the highly-elongated, near-stable end of the L1/L2 halo family -- the orbit class
NASA's Gateway flies (the 9:2 synodic-resonant L2 southern NRHO: period ~6.5 d, low perilune
over the lunar pole, apolune ~70,000 km). Naive z-amplitude continuation cannot reach them: the
family turns, and a fixed-z corrector falls off the branch. We use PSEUDO-ARCLENGTH continuation
on the x-z-symmetric single-shooting problem, which rounds the turning point.

Free variables X = [x0, z0, vy0] (state [x0,0,z0,0,vy0,0], a perpendicular x-z-plane crossing).
Constraints F(X) = [vx, vz] = 0 at the next y=0 crossing (perpendicularity -> periodic). That is
2 equations in 3 unknowns -> a 1-parameter family. The family tangent is the null vector of the
2x3 STM Jacobian (= cross product of its rows). Each step predicts along the tangent and Newton-
corrects orthogonally to it. We march until the period (or perilune) reaches the NRHO target.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.cr3bp import eom, eom_stm, jacobi_constant, propagate
from .halo import HaloOrbit, halo_family

_INT = dict(method="DOP853", rtol=1e-12, atol=1e-12)


def _half_period(mu, X, period_guess, t_min_frac=0.25):
    """Integrate [x0,0,z0,0,vy0,0] to the first y=0 crossing past t_min. Returns te, sc, Phi."""
    s = np.array([X[0], 0.0, X[1], 0.0, X[2], 0.0])
    y0 = np.concatenate([s, np.eye(6).ravel()])

    def ev(t, y, mu):
        return y[1]
    ev.direction = -1.0
    max_t = 1.4 * period_guess
    t_min = t_min_frac * period_guess
    for _ in range(6):
        sol = solve_ivp(eom_stm, (0.0, max_t), y0, args=(mu,), events=ev, **_INT)
        tev, yev = sol.t_events[0], sol.y_events[0]
        mask = tev > t_min
        if mask.any():
            k = int(np.argmax(mask))
            return float(tev[k]), yev[k][:6], yev[k][6:].reshape(6, 6)
        max_t *= 1.6
    raise RuntimeError("no y=0 crossing found")


def _F_and_jac(mu, X, period_guess):
    """Perpendicularity residual F=[vx,vz] and its 2x3 Jacobian dF/dX at X=[x0,z0,vy0]."""
    te, sc, Phi = _half_period(mu, X, period_guess)
    sd = eom(te, sc, mu)
    ydot = sc[4]
    cols = (0, 2, 4)                       # x0, z0, vy0
    M = np.empty((2, 3))
    for ri, r in enumerate((3, 5)):        # vx, vz
        for ci, c in enumerate(cols):
            M[ri, ci] = Phi[r, c] - (sd[r] / ydot) * Phi[1, c]
    return np.array([sc[3], sc[5]]), M, te, sc


def _tangent(M, prev=None):
    t = np.cross(M[0], M[1])
    n = np.linalg.norm(t)
    if n < 1e-14:
        raise RuntimeError("degenerate family tangent")
    t = t / n
    if prev is not None and np.dot(t, prev) < 0:
        t = -t
    return t


def _correct(mu, X_pred, tangent, period_guess, tol=1e-11, max_iter=25):
    """Pseudo-arclength Newton: solve F=0 with dX orthogonal to the family tangent."""
    X = np.asarray(X_pred, float).copy()
    for _ in range(max_iter):
        F, M, te, sc = _F_and_jac(mu, X, period_guess)
        if np.linalg.norm(F) < tol:
            return X, te, sc, M
        A = np.vstack([M, tangent])
        b = np.array([-F[0], -F[1], 0.0])
        X = X + np.linalg.solve(A, b)
        period_guess = 2.0 * te
    raise RuntimeError("NRHO corrector did not converge")


def _perilune_km(mu, s0, period, l_star, n=600):
    sol = propagate(s0, (0.0, period), mu, t_eval=np.linspace(0.0, period, n))
    d = np.sqrt((sol.y[0] - (1 - mu)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2)
    return float(d.min()) * l_star


def nrho_family(mu, point="L2", t_star_days=None, l_star=None,
                target_period_d=6.56, ds=5e-3, max_steps=400, southern=True):
    """Continue the halo family (pseudo-arclength) to the NRHO at ~target_period_d.

    Returns (nrho HaloOrbit, list of family HaloOrbits traversed). Period decreases along
    the branch toward the near-rectilinear regime; we stop at/below the target period.
    """
    # seed with a moderate halo a few steps from the bifurcation
    seed = halo_family(mu, point, n=4, dz=3e-3)[-1]
    sign = -1.0 if southern else 1.0
    X = np.array([seed.s0[0], sign * abs(seed.s0[2]), seed.s0[4]])
    period = seed.period

    F, M, te, sc = _F_and_jac(mu, X, period)
    # orient the tangent toward INCREASING |z| (into the family, away from planar)
    tan = _tangent(M)
    if np.sign(tan[1]) != np.sign(X[1]):
        tan = -tan

    fam, nrho = [], None
    for _ in range(max_steps):
        X_pred = X + ds * tan
        try:
            Xc, te, sc, M = _correct(mu, X_pred, tan, 2.0 * te)
        except RuntimeError:
            ds *= 0.5
            if ds < 1e-5:
                break
            continue
        period = 2.0 * te
        s0 = np.array([Xc[0], 0.0, Xc[1], 0.0, Xc[2], 0.0])
        orb = HaloOrbit(s0=s0, period=period, jacobi=float(jacobi_constant(s0, mu)),
                        z_amplitude=float(abs(Xc[1])), point=point,
                        residual=float(np.linalg.norm([sc[3], sc[5]])))
        fam.append(orb)
        new_tan = _tangent(M, prev=tan)
        tan, X = new_tan, Xc
        if t_star_days is not None and period * t_star_days <= target_period_d:
            nrho = orb
            break
    return nrho, fam
