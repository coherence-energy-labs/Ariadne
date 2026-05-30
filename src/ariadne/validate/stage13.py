"""Stage 13 validation gates (MASTER_PLAN.md — engine generalization + new regime).

G_jov - The Ariadne CR3BP/manifold engine generalizes to the Jupiter (Galilean) moon systems
        with only constant changes: sensible Lagrange-point distances and periodic Lyapunov
        orbits for all four moons, and a sensible moon-to-moon tour Delta-v baseline.
G_lt  - Low-thrust dynamics are correct: with zero thrust the Jacobi constant is conserved
        (= CR3BP), and with tangential thrust the energy changes at the predicted rate
        dC/dt = -2 a_T |v| (the continuous-thrust power theorem).

Run:  PYTHONPATH=src python -m ariadne.validate.stage13
"""
from __future__ import annotations

import numpy as np

from ..data.constants import GALILEAN, EARTH_MOON
from ..dynamics.cr3bp import jacobi_constant
from ..dynamics.low_thrust import (
    propagate_low_thrust, energy_gain_rate_check, delta_v,
)
from ..transfers.jovian import moon_libration, moon_tour_deltav


def check_jovian() -> tuple[bool, dict]:
    rows = [moon_libration(S) for S in GALILEAN]
    tour = moon_tour_deltav()
    ok = (all(1e3 < r["L1_km"] < 1e5 for r in rows)
          and all(0.1 < r["lyap_period_d"] < 30 for r in rows)
          and all(r["orbit"].half_period_residual < 1e-9 for r in rows)
          and all(1000 < leg["dv_ms"] < 6000 for leg in tour))
    return ok, {"rows": rows, "tour": tour,
                "tour_total": sum(leg["dv_ms"] for leg in tour)}


def check_low_thrust(mu) -> tuple[bool, dict]:
    s0 = np.array([0.5 - mu, np.sqrt(3) / 2, 0.0, 0.4, -0.2, 0.05])
    # zero thrust -> CR3BP (Jacobi conserved)
    sol0 = propagate_low_thrust(s0, (0.0, 5.0), mu, 0.0,
                                t_eval=np.linspace(0, 5, 400))
    c = np.array([jacobi_constant(sol0.y[:, i], mu) for i in range(sol0.y.shape[1])])
    drift0 = float(np.max(np.abs(c - c[0])))
    # tangential thrust -> dC/dt matches -2 a|v|
    a = 1e-3
    dt = 0.02
    sol = propagate_low_thrust(s0, (0.0, dt), mu, a, mode="tangential")
    dC = jacobi_constant(sol.y[:, -1], mu) - jacobi_constant(s0, mu)
    rate_meas = dC / dt
    rate_pred = energy_gain_rate_check(s0, mu, a)
    rel = abs(rate_meas - rate_pred) / abs(rate_pred)
    dv = delta_v(a, (0.0, 1.0), EARTH_MOON.V_star)        # over 1 nondim time
    ok = (drift0 < 1e-9) and (rel < 1e-2)
    return ok, {"zero_thrust_drift": drift0, "rate_meas": rate_meas,
                "rate_pred": rate_pred, "rate_rel_err": rel, "dv_per_nondim_kms": dv}


def main() -> int:
    print("=== Ariadne Stage 13 validation  (Jovian generalization + low-thrust regime) ===\n")
    okj, ij = check_jovian()
    print("[G_jov] Engine generalizes to the Galilean moon systems")
    for r in ij["rows"]:
        print(f"      {r['system']:<18s} L1={r['L1_km']:7.0f} km  "
              f"Lyapunov period={r['lyap_period_d']:.3f} d  "
              f"(periodic to {r['orbit'].half_period_residual:.1e})")
    tour_legs = [f"{l['from'][:3]}->{l['to'][:3]} {l['dv_ms']:.0f}" for l in ij["tour"]]
    print(f"      moon tour (Hohmann baseline): {' + '.join(tour_legs)}"
          f"  = {ij['tour_total']:.0f} m/s")
    print(f"      -> {'PASS' if okj else 'FAIL'}\n")

    okl, il = check_low_thrust(EARTH_MOON.mu)
    print("[G_lt] Low-thrust (continuous-acceleration) dynamics")
    print(f"      zero-thrust Jacobi drift = {il['zero_thrust_drift']:.2e} (= CR3BP)")
    print(f"      tangential thrust dC/dt: measured {il['rate_meas']:.4f}, "
          f"predicted -2a|v| {il['rate_pred']:.4f}  (rel err {il['rate_rel_err']:.2e})")
    print(f"      effective Delta-v at a=1e-3 over 1 nondim time = "
          f"{il['dv_per_nondim_kms']*1000:.1f} m/s")
    print(f"      -> {'PASS' if okl else 'FAIL'}\n")

    ok = okj and okl
    print(f"=== STAGE 13: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
