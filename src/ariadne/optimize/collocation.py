"""Direct-transcription trajectory optimization (MASTER_PLAN.md §3.13).

Hermite-Simpson collocation: discretize the trajectory at N+1 nodes, enforce the
dynamics as Simpson "defect" constraints between nodes, and minimize a
Simpson-integrated running cost with an NLP (SLSQP). This is the robust,
guess-tolerant optimizer used to refine transfers. Validated against the
closed-form minimum-energy double-integrator solution (see tests).

solve_hermite_simpson(f, L, x0, xf, T, N, nu, ...) -> dict(X, U, t, J, success).
  f(x, u) -> dx/dt   (numpy arrays, shapes (nx,), (nu,))
  L(x, u) -> running cost integrand (scalar)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def solve_hermite_simpson(f, L, x0, xf, T: float, N: int, nu: int,
                          u_bounds=None, max_iter: int = 400, ftol: float = 1e-9):
    x0 = np.asarray(x0, float)
    xf = np.asarray(xf, float)
    nx = x0.size
    h = T / N
    n_state = (N + 1) * nx

    def unpack(z):
        return z[:n_state].reshape(N + 1, nx), z[n_state:].reshape(N + 1, nu)

    def _segments(X, U):
        for k in range(N):
            xk, xk1 = X[k], X[k + 1]
            uk, uk1 = U[k], U[k + 1]
            fk, fk1 = f(xk, uk), f(xk1, uk1)
            xm = 0.5 * (xk + xk1) + (h / 8.0) * (fk - fk1)
            um = 0.5 * (uk + uk1)
            yield k, xk, xk1, uk, uk1, fk, fk1, xm, um

    def objective(z):
        X, U = unpack(z)
        J = 0.0
        for _, xk, xk1, uk, uk1, _, _, xm, um in _segments(X, U):
            J += (h / 6.0) * (L(xk, uk) + 4.0 * L(xm, um) + L(xk1, uk1))
        return J

    def defects(z):
        X, U = unpack(z)
        d = []
        for _, xk, xk1, uk, uk1, fk, fk1, xm, um in _segments(X, U):
            fm = f(xm, um)
            d.extend(xk1 - xk - (h / 6.0) * (fk + 4.0 * fm + fk1))
        return np.array(d)

    def boundary(z):
        X, _ = unpack(z)
        return np.concatenate([X[0] - x0, X[-1] - xf])

    # initial guess: linear state interpolation, zero control
    X_guess = np.linspace(x0, xf, N + 1)
    U_guess = np.zeros((N + 1, nu))
    z0 = np.concatenate([X_guess.ravel(), U_guess.ravel()])

    bounds = None
    if u_bounds is not None:
        bounds = [(None, None)] * n_state + [u_bounds] * ((N + 1) * nu)

    cons = [{"type": "eq", "fun": defects}, {"type": "eq", "fun": boundary}]
    res = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": max_iter, "ftol": ftol})

    X, U = unpack(res.x)
    return {"X": X, "U": U, "t": np.linspace(0.0, T, N + 1),
            "J": float(objective(res.x)), "success": bool(res.success),
            "max_defect": float(np.max(np.abs(defects(res.x)))), "res": res}
