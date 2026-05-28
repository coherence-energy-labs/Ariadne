"""Stage 5 validation gates (MASTER_PLAN.md §9, §10).

G7a - Ephemeris pipeline: SPICE/DE440 furnishes; UTC<->ET round-trips; the Earth-Moon
      distance and Earth-Sun distance from DE440 are in their known ranges.
G7b - Test-particle propagator: a two-body orbit (Earth point mass) closes on itself
      after one period to < 1 m and conserves energy.
G7c - Real n-body: a self-consistent Sun-Earth-Moon (+planet perturber) integration from
      DE440 initial conditions tracks the DE440 ephemeris over days to within a few km.

Run:  PYTHONPATH=src python -m ariadne.validate.stage5
"""
from __future__ import annotations

import math

import numpy as np

from ..data.ephemeris import body_gm, body_state, et, utc
from ..dynamics.ephemeris_nbody import propagate_test_particle, propagate_nbody

EPOCH = "2025-06-01T00:00:00"


def check_g7a() -> tuple[bool, dict]:
    e = et(EPOCH)
    back = utc(e, "ISOC", 0)
    moon = body_state("MOON", e, "J2000", "EARTH")
    sun = body_state("SUN", e, "J2000", "EARTH")
    d_moon = float(np.linalg.norm(moon[:3]))
    v_moon = float(np.linalg.norm(moon[3:]))
    d_sun = float(np.linalg.norm(sun[:3]))
    info = {"epoch": EPOCH, "utc_roundtrip": back, "earth_moon_km": d_moon,
            "moon_speed_kms": v_moon, "earth_sun_km": d_sun}
    ok = (back.startswith("2025-06-01")
          and 356000 < d_moon < 407000
          and 0.9 < v_moon < 1.1
          and 1.45e8 < d_sun < 1.53e8)
    return ok, info


def check_g7b() -> tuple[bool, dict]:
    mu = body_gm("EARTH")
    r = 8000.0
    r0 = np.array([r, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(mu / r), 0.0])
    T = 2 * math.pi * math.sqrt(r ** 3 / mu)
    e0 = et(EPOCH)
    sol = propagate_test_particle(r0, v0, e0, (0.0, T), perturbers=())
    rf, vf = sol.y[:3, -1], sol.y[3:, -1]

    def energy(rr, vv):
        return 0.5 * vv @ vv - mu / np.linalg.norm(rr)

    pos_err = float(np.linalg.norm(rf - r0))     # km
    e_drift = abs(energy(rf, vf) - energy(r0, v0))
    info = {"period_s": T, "closure_m": pos_err * 1000.0, "energy_drift": e_drift}
    ok = (pos_err < 1e-3) and (e_drift < 1e-6)
    return ok, info


def check_g7c() -> tuple[bool, dict]:
    bodies = ["SUN", "EARTH", "MOON"]
    external = ["JUPITER BARYCENTER", "VENUS BARYCENTER",
                "MARS BARYCENTER", "SATURN BARYCENTER"]
    e0 = et(EPOCH)
    days = 2.0
    sol, _ = propagate_nbody(bodies, e0, (0.0, days * 86400.0), external=external)

    err = {}
    for i, b in enumerate(bodies):
        integ = sol.y[3 * i:3 * i + 3, -1]
        truth = body_state(b, e0 + days * 86400.0, "J2000", "SSB")[:3]
        err[b] = float(np.linalg.norm(integ - truth))
    max_err = max(err["EARTH"], err["MOON"])
    info = {"days": days, "err_km": err, "max_earth_moon_km": max_err}
    # residual = missing minor perturbers / relativity / oblateness; ~0.02 km at 2 d
    ok = max_err < 1.0
    return ok, info


def main() -> int:
    print("=== Ariadne Stage 5 validation  (real JPL DE440 ephemeris) ===\n")

    ok_a, ia = check_g7a()
    print("[G7a] Ephemeris pipeline (SPICE / DE440)")
    print(f"      UTC round-trip: {ia['utc_roundtrip']}")
    print(f"      Earth-Moon distance = {ia['earth_moon_km']:,.0f} km "
          f"(perigee~363k, apogee~406k)")
    print(f"      Moon speed = {ia['moon_speed_kms']:.4f} km/s, "
          f"Earth-Sun = {ia['earth_sun_km']:,.0f} km")
    print(f"      -> {'PASS' if ok_a else 'FAIL'}\n")

    ok_b, ib = check_g7b()
    print("[G7b] Test-particle propagator (two-body closure)")
    print(f"      orbit period = {ib['period_s']:.2f} s; one-period closure = "
          f"{ib['closure_m']:.3e} m; energy drift = {ib['energy_drift']:.2e}")
    print(f"      -> {'PASS' if ok_b else 'FAIL'}\n")

    ok_c, ic = check_g7c()
    print("[G7c] Real n-body vs DE440 (Sun-Earth-Moon + 4 planet perturbers)")
    print(f"      after {ic['days']:.0f} days: "
          f"Earth err = {ic['err_km']['EARTH']:.4f} km, "
          f"Moon err = {ic['err_km']['MOON']:.4f} km, "
          f"Sun err = {ic['err_km']['SUN']:.4f} km")
    print(f"      -> {'PASS' if ok_c else 'FAIL'}\n")

    all_ok = ok_a and ok_b and ok_c
    print(f"=== STAGE 5 (G7): {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
