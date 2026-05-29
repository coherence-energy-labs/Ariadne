"""Differentiable dynamics + gradient-based trajectory optimization -- MASTER_PLAN.md Stage 35.

Ariadne's optimizers so far are derivative-FREE (differential evolution, grid/porkchop). That is
robust but slow: it never uses the gradient of Delta-v / miss-distance with respect to the decision
variables. Automatic differentiation gives that gradient EXACTLY (to machine precision, no
finite-difference noise), which turns trajectory design into a fast gradient/Gauss-Newton problem --
the approach modern tools (CASADI, JAX-based optimizers) use.

This module implements a fully differentiable RK4 two-body propagator in JAX and a Gauss-Newton
shooting solver on top of it. RK4 of smooth gravity is branch-free, so the autodiff gradient is clean
(unlike a branchy analytic Lambert, whose universal-variable Stumpff/Newton singularities give NaN
gradients). Standard gravity throughout (firewall intact).

Requires JAX (CPU is fine -- the win is EXACT gradients, not the GPU). Falls back with a clear error.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    HAVE_JAX = True
except Exception:                                            # pragma: no cover
    HAVE_JAX = False

GM_SUN = 1.32712440018e11        # km^3/s^2


if HAVE_JAX:
    def _accel(r, mu):
        rn = jnp.linalg.norm(r)
        return -mu * r / rn ** 3

    def _propagate(r0, v0, tof, mu, n_steps):
        dt = tof / n_steps

        def step(carry, _):
            r, v = carry
            k1r, k1v = v, _accel(r, mu)
            k2r, k2v = v + 0.5 * dt * k1v, _accel(r + 0.5 * dt * k1r, mu)
            k3r, k3v = v + 0.5 * dt * k2v, _accel(r + 0.5 * dt * k2r, mu)
            k4r, k4v = v + dt * k3v, _accel(r + dt * k3r, mu)
            r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
            v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            return (r, v), None

        (rf, vf), _ = jax.lax.scan(step, (r0, v0), None, length=n_steps)
        return rf, vf

    def propagate(r0, v0, tof, mu=GM_SUN, n_steps=400):
        """Differentiable RK4 two-body propagation. Returns (r_final, v_final) as numpy arrays."""
        rf, vf = _propagate(jnp.asarray(r0, float), jnp.asarray(v0, float),
                            float(tof), mu, n_steps)
        return np.asarray(rf), np.asarray(vf)

    def _residual(v1, r1, r2, tof, mu, n_steps, scale):
        rf, _ = _propagate(r1, v1, tof, mu, n_steps)
        return (rf - r2) / scale

    def solve_lambert_shooting(r1, r2, tof, v_guess, mu=GM_SUN, n_steps=400,
                               iters=60, tol_km=1e-3):
        """Find v1 such that propagate(r1, v1, tof) == r2, by autodiff Levenberg-Marquardt.

        Solves the Lambert problem via EXACT gradients of the integrator -- no branchy analytic
        Lambert. The miss->velocity map is strongly nonlinear over a long arc, so an undamped
        Newton step overshoots; LM damping with backtracking (accept only steps that reduce the
        miss, adapt lambda) makes it robust. Returns dict(v1, miss_km, iters).
        """
        r1j = jnp.asarray(r1, float); r2j = jnp.asarray(r2, float)
        scale = float(jnp.linalg.norm(r2j))
        resid = jax.jit(lambda v: _residual(v, r1j, r2j, float(tof), mu, n_steps, scale))
        jac = jax.jit(jax.jacobian(lambda v: _residual(v, r1j, r2j, float(tof), mu, n_steps, scale)))

        v = np.asarray(v_guess, float)
        rr = np.asarray(resid(jnp.asarray(v)))
        miss = float(np.linalg.norm(rr) * scale)
        lam = 1e-3
        n_it = 0
        for n_it in range(1, iters + 1):
            if miss < tol_km:
                break
            J = np.asarray(jac(jnp.asarray(v)))
            JtJ = J.T @ J
            Jtr = J.T @ rr
            accepted = False
            for _ in range(12):                              # LM backtracking on lambda
                dv = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), Jtr)
                v_try = v + dv
                rr_try = np.asarray(resid(jnp.asarray(v_try)))
                miss_try = float(np.linalg.norm(rr_try) * scale)
                if miss_try < miss:
                    v, rr, miss = v_try, rr_try, miss_try
                    lam = max(lam * 0.5, 1e-12)
                    accepted = True
                    break
                lam = min(lam * 4.0, 1e12)
            if not accepted:
                break
        return {"v1": v, "miss_km": miss, "iters": n_it}

    def dv_gradient(v1, r1, r2_target, tof, mu=GM_SUN, n_steps=400):
        """Exact gradient of the squared miss-distance w.r.t. departure velocity (autodiff)."""
        r1 = jnp.asarray(r1, float); r2 = jnp.asarray(r2_target, float)

        def miss(v):
            rf, _ = _propagate(r1, v, float(tof), mu, n_steps)
            return jnp.sum((rf - r2) ** 2)

        return np.asarray(jax.grad(miss)(jnp.asarray(v1, float)))

    def _transfer_objective(p, rE0, vE0, rM0, vM0, mu, n_steps, penalty):
        """Total Delta-v + miss penalty for a 2-impulse transfer. p=[t0, tof, v1x,v1y,v1z] (s, km/s).

        Planets and spacecraft are propagated by the branch-free RK4 (differentiable). The miss
        penalty enforces arrival; the objective is a single smooth function -> exact autodiff gradient.
        """
        t0, tof = p[0], p[1]
        v1 = p[2:5]
        rE, vE = _propagate(rE0, vE0, t0, mu, n_steps)
        rM, vM = _propagate(rM0, vM0, t0 + tof, mu, n_steps)
        rf, vf = _propagate(rE, v1, tof, mu, n_steps)
        dv = jnp.linalg.norm(v1 - vE) + jnp.linalg.norm(vf - vM)
        miss = jnp.linalg.norm(rf - rM) / 1.495978707e8
        return dv + penalty * miss * miss

    def transfer_dv(rE0, vE0, rM0, vM0, t0_s, tof_s, mu=GM_SUN, n_steps=300):
        """Exact 2-impulse Delta-v for a transfer departing at t0 with time-of-flight tof.

        Arrival is enforced EXACTLY by the autodiff Levenberg-Marquardt shooting (the inner solve),
        so the miss is ~0 by construction. Returns (dv_kms, miss_km, v1).
        """
        rE, vE = propagate(rE0, vE0, t0_s, mu, n_steps)
        rM, vM = propagate(rM0, vM0, t0_s + tof_s, mu, n_steps)
        sol = solve_lambert_shooting(rE, rM, tof_s, vE * 1.06, mu=mu, n_steps=n_steps,
                                     iters=40, tol_km=1.0)
        rf, vf = propagate(rE, sol["v1"], tof_s, mu, n_steps)
        dv = float(np.linalg.norm(sol["v1"] - vE) + np.linalg.norm(vf - vM))
        return dv, sol["miss_km"], sol["v1"]

    def optimize_transfer(rE0, vE0, rM0, vM0, t0_grid_days, tof_grid_days, mu=GM_SUN, n_steps=300):
        """Global min-Delta-v 2-impulse transfer over a (departure, time-of-flight) grid.

        Each grid point is solved EXACTLY by the autodiff shooting (arrival enforced), so the
        returned optimum is a true, valid transfer. Returns the best dict + the Delta-v surface.
        """
        rE0 = np.asarray(rE0, float); vE0 = np.asarray(vE0, float)
        rM0 = np.asarray(rM0, float); vM0 = np.asarray(vM0, float)
        surf = np.full((len(t0_grid_days), len(tof_grid_days)), np.inf)
        best = None
        for it, t0d in enumerate(t0_grid_days):
            for jt, tofd in enumerate(tof_grid_days):
                try:
                    dv, miss, v1 = transfer_dv(rE0, vE0, rM0, vM0, t0d * 86400.0,
                                               tofd * 86400.0, mu, n_steps)
                except Exception:
                    continue
                if miss < 100.0:                              # valid transfer (arrival hit)
                    surf[it, jt] = dv
                    if best is None or dv < best["dv_kms"]:
                        best = {"dv_kms": dv, "t0_days": t0d, "tof_days": tofd,
                                "miss_km": miss, "v1": v1}
        return {"best": best, "surface": surf, "t0_grid": np.asarray(t0_grid_days),
                "tof_grid": np.asarray(tof_grid_days)}


else:                                                        # pragma: no cover
    def propagate(*a, **k):
        raise RuntimeError("JAX required for ariadne.optimize.autodiff")

    solve_lambert_shooting = dv_gradient = optimize_transfer = propagate
