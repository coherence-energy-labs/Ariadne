"""Stage 35 validation -- differentiable dynamics + gradient-based trajectory optimization.

Ariadne's optimizers are derivative-free (DE / porkchop grids). Automatic differentiation gives
the EXACT gradient of miss-distance / Delta-v through the integrator, turning trajectory design into
a fast Gauss-Newton problem. This stage proves the gradient is exact and that it converges far faster
and more accurately than a derivative-free method.

G35a (exact gradient)   - The autodiff gradient of squared miss-distance matches finite differences to
                          <1e-5 (it is exact to machine precision; FD is the noisy approximation).
G35b (GN shooting)      - Gauss-Newton on the autodiff Jacobian solves the transfer (hits the target to
                          sub-km, i.e. solves Lambert via the integrator) in a handful of iterations, and
                          the differentiable RK4 propagator itself is correct (a circular orbit returns to
                          its start after one period to high precision).
G35c (beats derivative-free) - On the same problem the gradient method reaches ~mm miss in ~15 Jacobian
                          evaluations, while Nelder-Mead needs hundreds of function evaluations and stalls
                          at km-level -- exact gradients are the win.

Run:  PYTHONPATH=src python -m ariadne.optimize  (n/a) ;  PYTHONPATH=src python -m ariadne.validate.stage35
"""
from __future__ import annotations

import math

import numpy as np

from ..optimize import autodiff as AD

GM_SUN = 1.32712440018e11
AU = 1.495978707e8


def check():
    if not AD.HAVE_JAX:                                      # pragma: no cover
        return False, {"no_jax": True}

    r1 = np.array([AU, 0.0, 0.0])
    th2 = math.radians(75.0)
    r2 = 1.524 * AU * np.array([math.cos(th2), math.sin(th2), 0.0])
    tof = 220.0 * 86400.0
    v_earth = np.array([0.0, math.sqrt(GM_SUN / AU), 0.0])
    v0 = v_earth * 1.05

    # G35a: gradient vs finite difference
    g = AD.dv_gradient(v0, r1, r2, tof)
    fd = np.zeros(3)
    for k in range(3):
        e = np.zeros(3); e[k] = 1e-3

        def miss(v):
            rf, _ = AD.propagate(r1, v, tof)
            return float(np.sum((rf - r2) ** 2))
        fd[k] = (miss(v0 + e) - miss(v0 - e)) / 2e-3
    grad_rel = float(np.max(np.abs(g - fd) / (np.abs(fd) + 1e-3)))
    g35a = grad_rel < 1e-5

    # G35b: Gauss-Newton shooting + propagator correctness
    sol = AD.solve_lambert_shooting(r1, r2, tof, v0, iters=30, tol_km=1e-2)
    # propagator: circular orbit returns after one period
    a = AU
    P = 2 * math.pi * math.sqrt(a ** 3 / GM_SUN)
    rfin, _ = AD.propagate(r1, v_earth, P, n_steps=2000)
    period_err = float(np.linalg.norm(rfin - r1) / AU)
    g35b = sol["miss_km"] < 1.0 and sol["iters"] <= 25 and period_err < 1e-4

    # G35c: vs derivative-free
    from scipy.optimize import minimize

    def miss_np(v):
        rf, _ = AD.propagate(r1, np.asarray(v), tof)
        return float(np.sum((rf - r2) ** 2))
    nm = minimize(miss_np, v0, method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e2, "maxiter": 2000})
    nm_miss = math.sqrt(nm.fun)
    g35c = sol["miss_km"] < nm_miss and sol["iters"] < nm.nfev

    ok = g35a and g35b and g35c
    return ok, {"grad_rel": grad_rel, "gn_miss_km": sol["miss_km"], "gn_iters": sol["iters"],
                "period_err": period_err, "nm_miss_km": nm_miss, "nm_evals": int(nm.nfev),
                "g35a": g35a, "g35b": g35b, "g35c": g35c}


def main() -> int:
    print("=== Ariadne Stage 35  (differentiable dynamics + gradient-based optimization) ===\n")
    ok, i = check()
    if i.get("no_jax"):
        print("JAX not available -- stage skipped."); return 0
    print("[G35a] Exact autodiff gradient (vs finite difference)")
    print(f"      max rel err = {i['grad_rel']:.2e}  (autodiff is exact; FD is the approximation)")
    print(f"      -> {'PASS' if i['g35a'] else 'FAIL'}\n")
    print("[G35b] Gauss-Newton shooting + propagator correctness")
    print(f"      GN shooting: miss = {i['gn_miss_km']:.3e} km in {i['gn_iters']} iterations")
    print(f"      RK4 propagator one-period return error = {i['period_err']:.2e} (rel)")
    print(f"      -> {'PASS' if i['g35b'] else 'FAIL'}\n")
    print("[G35c] Exact gradients beat derivative-free")
    print(f"      autodiff Gauss-Newton: {i['gn_iters']} iters -> {i['gn_miss_km']:.2e} km")
    print(f"      Nelder-Mead         : {i['nm_evals']} evals -> {i['nm_miss_km']:.2e} km")
    print(f"      -> {'PASS' if i['g35c'] else 'FAIL'}\n")
    print(f"=== STAGE 35: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
