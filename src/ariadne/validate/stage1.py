"""Stage 1 validation gates (MASTER_PLAN.md §9, §10).

G1 - Jacobi constant conserved to < 1e-10 over a long propagation, AND the STM
     (variational equations) matches a finite-difference of the flow.
G2 - Lagrange points: grad(Omega) ~ 0 to < 1e-10 at all five, and the collinear
     positions match published Earth-Moon values.

Run:  PYTHONPATH=src python -m ariadne.validate.stage1
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON
from ..dynamics.cr3bp import jacobi_constant, propagate, propagate_stm
from ..orbits.lagrange import lagrange_points, gradient_residual

# Published Earth-Moon collinear Lagrange x-positions (nondimensional).
PUBLISHED_EM = {"L1": 0.8369151324, "L2": 1.1556821603, "L3": -1.0050626453}


def check_jacobi(mu: float) -> tuple[bool, float]:
    """Propagate a generic bounded state and check |dC| over many time units."""
    # An L4-region state with a small velocity perturbation: stays bounded.
    s0 = np.array([0.5 - mu, np.sqrt(3) / 2, 0.0, 0.02, -0.01, 0.005])
    c0 = jacobi_constant(s0, mu)
    sol = propagate(s0, (0.0, 50.0), mu, t_eval=np.linspace(0, 50, 2001))
    c = np.array([jacobi_constant(sol.y[:, i], mu) for i in range(sol.y.shape[1])])
    max_dev = float(np.max(np.abs(c - c0)))
    return max_dev < 1e-10, max_dev


def check_stm(mu: float) -> tuple[bool, float]:
    """Compare variational STM against central finite-differences of the flow."""
    s0 = np.array([0.85, 0.02, 0.01, 0.05, 0.12, -0.03])
    T = 1.0
    _, stm = propagate_stm(s0, (0.0, T), mu)

    h = 1e-7
    fd = np.zeros((6, 6))
    for i in range(6):
        sp = s0.copy(); sp[i] += h
        sm = s0.copy(); sm[i] -= h
        yp = propagate(sp, (0.0, T), mu).y[:, -1]
        ym = propagate(sm, (0.0, T), mu).y[:, -1]
        fd[:, i] = (yp - ym) / (2 * h)

    err = float(np.max(np.abs(stm - fd)))
    return err < 1e-5, err


def check_lagrange(mu: float) -> tuple[bool, dict]:
    pts = lagrange_points(mu)
    resid = gradient_residual(mu)
    max_resid = max(resid.values())
    pos_err = {k: abs(pts[k][0] - PUBLISHED_EM[k]) for k in PUBLISHED_EM}
    max_pos_err = max(pos_err.values())
    ok = (max_resid < 1e-10) and (max_pos_err < 1e-3)
    info = {"x": {k: float(pts[k][0]) for k in pts},
            "grad_residual_max": max_resid,
            "position_error_max": max_pos_err}
    return ok, info


def main() -> int:
    mu = EARTH_MOON.mu
    print(f"=== Ariadne Stage 1 validation  (Earth-Moon, mu={mu:.12f}) ===\n")
    print(f"Characteristic scales: L*={EARTH_MOON.L_star:.1f} km  "
          f"T*={EARTH_MOON.T_star:.1f} s ({EARTH_MOON.T_star/86400:.4f} d)  "
          f"V*={EARTH_MOON.V_star:.5f} km/s\n")

    ok_j, dev = check_jacobi(mu)
    print(f"[G1a] Jacobi conservation : max|dC| = {dev:.3e}   "
          f"{'PASS' if ok_j else 'FAIL'}  (need < 1e-10)")

    ok_s, err = check_stm(mu)
    print(f"[G1b] STM vs finite-diff  : max err = {err:.3e}   "
          f"{'PASS' if ok_s else 'FAIL'}  (need < 1e-5)")

    ok_l, info = check_lagrange(mu)
    print(f"[G2 ] Lagrange points     : grad|.| = {info['grad_residual_max']:.3e}, "
          f"pos err = {info['position_error_max']:.3e}   "
          f"{'PASS' if ok_l else 'FAIL'}")
    for k in ("L1", "L2", "L3", "L4", "L5"):
        print(f"        {k}: x = {info['x'][k]:+.10f}")

    all_ok = ok_j and ok_s and ok_l
    print(f"\n=== STAGE 1: {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
