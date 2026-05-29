"""Stage 7 validation gates (MASTER_PLAN.md §9, §10).

G_halo - 3D halo orbits: a halo family branches from the Lyapunov vertical bifurcation
         (Earth-Moon L1, C ~ 3.186, consistent with Stage 2), with periodic 3D orbits.
G9     - Genesis mechanism: a real Sun-Earth L1 halo (period ~178 d, matching SOHO/Genesis)
         whose invariant manifold carries a spacecraft from the halo (~1.49e6 km out) down
         to Earth's vicinity -- the interplanetary superhighway Genesis flew.
G10*   - Independent cross-validation (the GMAT cross-check is exported but GMAT-the-app is
         not run here; instead we cross-check with INDEPENDENT tools): two ephemeris
         libraries (spiceypy vs jplephem) reading DE440 agree to < 1 m, and two independent
         integrators (DOP853 vs Radau) agree on an ephemeris-perturbed orbit to < 100 m.

Run:  PYTHONPATH=src python -m ariadne.validate.stage7
"""
from __future__ import annotations

import os

import numpy as np

from ..data.constants import EARTH_MOON, SUN_EARTH
from ..data.ephemeris import body_gm, body_state, et
from ..data.kernels import KERNEL_DIR, ensure_kernels
from ..dynamics.cr3bp import propagate
from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..orbits.halo import halo_family
from ..transfers.genesis import genesis_halo, earth_approach


def check_halo() -> tuple[bool, dict]:
    mu = EARTH_MOON.mu
    halos = halo_family(mu, "L1", n=15, dz=2e-3)
    res = max(np.max(np.abs(propagate(h.s0, (0.0, h.period), mu).y[:, -1] - h.s0))
              for h in halos[::3])
    z = [h.z_amplitude for h in halos]
    info = {"n": len(halos), "bifurcation_C": halos[0].jacobi,
            "max_periodicity": res, "z_max_km": z[-1] * EARTH_MOON.L_star,
            "z_increasing": all(b > a for a, b in zip(z, z[1:]))}
    ok = (len(halos) >= 10 and res < 1e-9
          and 3.17 < halos[0].jacobi < 3.19 and info["z_increasing"])
    return ok, info


def check_genesis() -> tuple[bool, dict]:
    h, _ = genesis_halo()
    period_d = h.period * SUN_EARTH.T_star / 86400.0
    r = earth_approach(h, t_max=8.0, n_seeds=120)
    info = {"period_days": period_d, "z_amp_km": h.z_amplitude * SUN_EARTH.L_star,
            "earth_km": r["earth_km"], "l1_km": r["l1_earth_km"],
            "fraction_of_l1": r["fraction_of_l1"],
            "via": ("stable" if r["stable"] else "unstable")}
    ok = (170.0 < period_d < 185.0) and (r["fraction_of_l1"] < 0.1)
    return ok, info


def check_crossval() -> tuple[bool, dict]:
    ensure_kernels()
    from jplephem.spk import SPK
    e = et("2025-06-01T00:00:00")
    jd = 2451545.0 + e / 86400.0
    k = SPK.open(os.path.join(KERNEL_DIR, "de440s.bsp"))
    moon_jpl = np.array(k[3, 301].compute(jd)) - np.array(k[3, 399].compute(jd))
    k.close()
    moon_spice = body_state("MOON", e, "J2000", "EARTH")[:3]
    eph_diff_km = float(np.linalg.norm(moon_spice - moon_jpl))

    # DOP853 vs Radau on the same ephemeris-perturbed orbit (1 day)
    r0 = np.array([7000.0, 1000.0, 500.0])
    v0 = np.array([1.0, 7.5, 0.3])
    span = (0.0, 86400.0)
    a = propagate_test_particle(r0, v0, e, span, perturbers=("SUN", "MOON"))
    b = propagate_test_particle(r0, v0, e, span, perturbers=("SUN", "MOON"),
                                method="Radau", rtol=1e-11, atol=1e-9)
    integ_diff_km = float(np.linalg.norm(a.y[:3, -1] - b.y[:3, -1]))

    info = {"ephem_lib_diff_m": eph_diff_km * 1000.0,
            "integrator_diff_m": integ_diff_km * 1000.0}
    ok = (eph_diff_km < 1e-3) and (integ_diff_km < 0.1)
    return ok, info


def main() -> int:
    print("=== Ariadne Stage 7 validation  (halos, Genesis, independent cross-checks) ===\n")

    okh, ih = check_halo()
    print("[G_halo] 3D halo orbits (Earth-Moon L1)")
    print(f"      {ih['n']} halos; family branches at C = {ih['bifurcation_C']:.5f} "
          f"(Stage-2 bifurcation ~3.1864)")
    print(f"      z-amplitude up to {ih['z_max_km']:.0f} km (increasing: {ih['z_increasing']}); "
          f"max periodicity {ih['max_periodicity']:.2e}")
    print(f"      -> {'PASS' if okh else 'FAIL'}\n")

    okg, ig = check_genesis()
    print("[G9] Genesis mechanism (Sun-Earth L1 halo + manifold to Earth)")
    print(f"      halo period = {ig['period_days']:.1f} d (real SOHO/Genesis ~178 d), "
          f"z-amplitude {ig['z_amp_km']:.0f} km")
    print(f"      manifold closest Earth approach = {ig['earth_km']:.0f} km "
          f"({ig['fraction_of_l1']*100:.1f}% of the {ig['l1_km']:.0f} km L1 distance) "
          f"via {ig['via']} manifold")
    print(f"      -> {'PASS' if okg else 'FAIL'}\n")

    okc, ic = check_crossval()
    print("[G10*] Independent cross-validation")
    print(f"      ephemeris libraries (spiceypy vs jplephem on DE440): "
          f"{ic['ephem_lib_diff_m']:.2e} m")
    print(f"      integrators (DOP853 vs Radau, ephemeris-perturbed, 1 day): "
          f"{ic['integrator_diff_m']:.3e} m")
    print(f"      -> {'PASS' if okc else 'FAIL'}\n")

    all_ok = okh and okg and okc
    print(f"=== STAGE 7: {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    print("NOTE: GMAT script is exported (docs/examples/) but GMAT-the-app is not installed\n"
          "here; the independent library + integrator cross-checks serve the G10 purpose.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
