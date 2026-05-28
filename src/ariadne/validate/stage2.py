"""Stage 2 validation gates (MASTER_PLAN.md §9, §10).

G3 - Reproduce a periodic libration orbit: the corrected Lyapunov orbit is truly
     periodic (state returns after one period to < 1e-9), and in the small-amplitude
     limit its period matches the linear theory 2*pi/omega.
G4 - Family continuation: generate the L1 Lyapunov family; the Jacobi/period curves
     are monotonic and well-formed, and the halo bifurcation (vertical stability
     index through +1) is located.

Run:  PYTHONPATH=src python -m ariadne.validate.stage2
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON
from ..dynamics.cr3bp import propagate
from ..orbits.differential_correction import correct_lyapunov
from ..orbits.families import lyapunov_family, find_halo_bifurcation
from ..orbits.linear import collinear_linear_modes, linear_lyapunov_guess


def periodicity_residual(mu, orbit) -> float:
    sol = propagate(orbit.s0, (0.0, orbit.period), mu)
    return float(np.max(np.abs(sol.y[:, -1] - orbit.s0)))


def check_g3(mu) -> tuple[bool, dict]:
    info = {}
    # small-amplitude period vs linear theory
    m = collinear_linear_modes(mu, "L1")
    s0, Tg = linear_lyapunov_guess(mu, "L1", 1e-4)
    tiny = correct_lyapunov(mu, s0, Tg)
    period_rel_err = abs(tiny.period - m["linear_period"]) / m["linear_period"]
    info["linear_period"] = m["linear_period"]
    info["tiny_period"] = tiny.period
    info["period_rel_err"] = period_rel_err

    # a representative finite-amplitude orbit, check true periodicity
    s0b, Tgb = linear_lyapunov_guess(mu, "L1", 0.02)
    orb = correct_lyapunov(mu, s0b, Tgb)
    res = periodicity_residual(mu, orb)
    info["rep_amplitude"] = 0.02
    info["rep_period"] = orb.period
    info["rep_jacobi"] = orb.jacobi
    info["periodicity_residual"] = res

    ok = (period_rel_err < 1e-3) and (res < 1e-9) and (tiny.half_period_residual < 1e-11)
    return ok, info


def check_g4(mu) -> tuple[bool, dict]:
    fam = lyapunov_family(mu, "L1", amplitude0=1e-3, dx=2e-3, n=40)
    amps = np.array([m.amplitude for m in fam])
    jac = np.array([m.orbit.jacobi for m in fam])
    per = np.array([m.orbit.period for m in fam])
    nuv = np.array([m.nu_vertical for m in fam])

    # all members truly periodic
    max_res = max(periodicity_residual(mu, m.orbit) for m in fam[::5])
    # Jacobi decreases monotonically as amplitude (energy) grows
    monotonic = bool(np.all(np.diff(jac) < 1e-9))
    bif = find_halo_bifurcation(fam)

    info = {"n": len(fam), "amp_range": (float(amps.min()), float(amps.max())),
            "jacobi_range": (float(jac.max()), float(jac.min())),
            "period_range": (float(per.min()), float(per.max())),
            "nu_v_range": (float(nuv.min()), float(nuv.max())),
            "max_periodicity_residual": float(max_res),
            "monotonic_jacobi": monotonic, "halo_bifurcation": bif}
    ok = (len(fam) >= 20) and monotonic and (max_res < 1e-8) and (bif is not None)
    return ok, info


def main() -> int:
    mu = EARTH_MOON.mu
    print(f"=== Ariadne Stage 2 validation  (Earth-Moon, mu={mu:.12f}) ===\n")

    ok3, i3 = check_g3(mu)
    print("[G3] Periodic Lyapunov orbit")
    print(f"     linear period 2pi/omega = {i3['linear_period']:.8f}")
    print(f"     tiny-amp period         = {i3['tiny_period']:.8f}  "
          f"(rel err {i3['period_rel_err']:.2e}, need < 1e-3)")
    print(f"     rep orbit (Ax=0.02): period={i3['rep_period']:.6f}, "
          f"C={i3['rep_jacobi']:.6f}, periodicity={i3['periodicity_residual']:.2e}")
    print(f"     -> {'PASS' if ok3 else 'FAIL'}\n")

    ok4, i4 = check_g4(mu)
    print("[G4] Lyapunov family continuation")
    print(f"     members: {i4['n']}, amplitude {i4['amp_range'][0]:.4f}..{i4['amp_range'][1]:.4f}")
    print(f"     Jacobi {i4['jacobi_range'][0]:.5f} -> {i4['jacobi_range'][1]:.5f} "
          f"(monotonic decreasing: {i4['monotonic_jacobi']})")
    print(f"     period {i4['period_range'][0]:.4f}..{i4['period_range'][1]:.4f}")
    print(f"     vertical index nu_v in [{i4['nu_v_range'][0]:.4f}, {i4['nu_v_range'][1]:.4f}]")
    print(f"     max periodicity residual across family = {i4['max_periodicity_residual']:.2e}")
    if i4["halo_bifurcation"]:
        b = i4["halo_bifurcation"]
        print(f"     HALO BIFURCATION at amplitude={b['amplitude']:.5f}, C={b['jacobi']:.6f}")
    else:
        print("     halo bifurcation: NOT found in scanned range")
    print(f"     -> {'PASS' if ok4 else 'FAIL'}\n")

    all_ok = ok3 and ok4
    print(f"=== STAGE 2: {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
