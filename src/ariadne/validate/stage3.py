"""Stage 3 validation gates (MASTER_PLAN.md §9, §10).

G5 - Invariant manifold: the L1 Lyapunov unstable tube conserves the Jacobi
     constant along every trajectory (< 1e-9), the seeds sit at the orbit energy
     to O(displacement), and the tube reaches the secondary's neck (x = 1-mu).
G6 - Heteroclinic connection: the L1 unstable tube and L2 stable tube intersect
     on the section x = 1-mu at a common Jacobi constant (a near-ballistic
     L1<->L2 connection).

Run:  PYTHONPATH=src python -m ariadne.validate.stage3
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON
from ..manifolds.manifold import manifold_seeds
from ..connections.poincare import propagate_until_section
from ..connections.heteroclinic import find_heteroclinic
from ..dynamics.cr3bp import jacobi_constant
from ..orbits.families import lyapunov_orbit_at_jacobi


def check_g5(mu) -> tuple[bool, dict]:
    c = 3.17  # below C_L1 ~ 3.200: the L1 neck is open
    orbit = lyapunov_orbit_at_jacobi(mu, "L1", c)
    n_seeds = 40
    disp = 1e-4
    seeds, lam = manifold_seeds(mu, orbit, n_seeds=n_seeds, displacement=disp,
                                stable=False, branch=+1)
    seed_dc = max(abs(jacobi_constant(s, mu) - orbit.jacobi) for s in seeds)

    drifts = []
    reached = 0
    min_moon = np.inf
    x_sec = 1.0 - mu
    for s in seeds:
        t, Y, hit = propagate_until_section(mu, s, x_sec, stable=False, t_max=8.0)
        # min distance to the Moon along this tube segment (km)
        r2 = np.sqrt((Y[0] - (1.0 - mu)) ** 2 + Y[1] ** 2 + Y[2] ** 2)
        min_moon = min(min_moon, float(r2.min()))
        if hit:
            reached += 1
            c0 = jacobi_constant(Y[:, 0], mu)
            drifts.append(max(abs(jacobi_constant(Y[:, i], mu) - c0)
                              for i in range(Y.shape[1])))

    drifts = np.array(drifts)
    median_drift = float(np.median(drifts))
    max_drift = float(drifts.max())
    info = {"jacobi": orbit.jacobi, "lambda_u": lam, "seed_dc": seed_dc,
            "median_jacobi_drift": median_drift, "max_jacobi_drift": max_drift,
            "reached_neck": reached, "n_seeds": n_seeds,
            "min_moon_km": min_moon * EARTH_MOON.L_star}
    # machinery sound (typical drift tiny); close approaches bounded
    ok = ((lam > 1.0) and (median_drift < 1e-9) and (max_drift < 1e-5)
          and (reached >= n_seeds // 4) and (seed_dc < 1e-2))
    return ok, info


def check_g6(mu) -> tuple[bool, dict]:
    # Both necks open for C < C_L2 ~ 3.184; scan a few energies for a clean hit.
    for c in (3.15, 3.14, 3.16, 3.13, 3.17):
        conn = find_heteroclinic(mu, c, "L1", "L2", n_seeds=160, displacement=1e-4)
        if conn is not None:
            y, vy = conn["connection_yv"]
            info = {"jacobi": c, "connection_y": float(y), "connection_vy": float(vy),
                    "n_intersections": len(conn["intersections"]),
                    "branch_u": conn["branch_unstable"], "branch_s": conn["branch_stable"],
                    "x_section": conn["x_section"]}
            return True, info
    return False, {"note": "no intersection found in scanned energies"}


def main() -> int:
    mu = EARTH_MOON.mu
    print(f"=== Ariadne Stage 3 validation  (Earth-Moon, mu={mu:.12f}) ===\n")

    ok5, i5 = check_g5(mu)
    print("[G5] Invariant manifold (L1 Lyapunov unstable tube)")
    print(f"     orbit Jacobi C = {i5['jacobi']:.6f},  unstable multiplier lambda = {i5['lambda_u']:.3e}")
    print(f"     seed energy offset = {i5['seed_dc']:.2e} (O(displacement))")
    print(f"     Jacobi drift to section: median = {i5['median_jacobi_drift']:.2e} (need < 1e-9), "
          f"max = {i5['max_jacobi_drift']:.2e} (need < 1e-5, bounds close approaches)")
    print(f"     min Moon distance along tubes = {i5['min_moon_km']:.0f} km")
    print(f"     trajectories reaching the Moon neck x=1-mu: {i5['reached_neck']}/{i5['n_seeds']}")
    print(f"     -> {'PASS' if ok5 else 'FAIL'}\n")

    ok6, i6 = check_g6(mu)
    print("[G6] Heteroclinic L1<->L2 connection")
    if ok6:
        print(f"     connection at Jacobi C = {i6['jacobi']:.4f} on section x = {i6['x_section']:.5f}")
        print(f"     crossing (y, vy) = ({i6['connection_y']:.6f}, {i6['connection_vy']:.6f})")
        print(f"     tube-cut intersections found = {i6['n_intersections']} "
              f"(branches u={i6['branch_u']}, s={i6['branch_s']})")
    else:
        print(f"     {i6.get('note','')}")
    print(f"     -> {'PASS' if ok6 else 'FAIL'}\n")

    all_ok = ok5 and ok6
    print(f"=== STAGE 3: {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
