"""Stage 4 validation gates (MASTER_PLAN.md §9, §10).

G8a - BCR4BP: reduces to the CR3BP when the solar mass is removed (EOM match to
      1e-12); the solar acceleration vanishes at the barycenter and is a sane
      tidal magnitude across the Earth-Moon system.
G8b - Delta-v budget reproduces the Apollo-class direct Earth->Moon transfer
      (LEO->LLO ~ 3.9-4.0 km/s) and textbook LEO/LLO circular speeds.
G8c - The ballistic-capture (low-energy) lunar insertion is cheaper than direct
      insertion -- the mechanism behind the Coimbra low-energy result.

Run:  PYTHONPATH=src python -m ariadne.validate.stage4
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON
from ..dynamics.cr3bp import eom
from ..dynamics.bcr4bp import eom_bcr4bp, solar_acceleration, sun_params
from ..optimize.budget import earth_moon_budget


def check_g8a(mu) -> tuple[bool, dict]:
    sp = sun_params(EARTH_MOON)
    rng = np.random.default_rng(0)
    # no-Sun limit: BCR4BP(m_S=0) == CR3BP
    max_diff = 0.0
    for _ in range(200):
        s = rng.uniform(-1.5, 1.5, size=6)
        d = eom_bcr4bp(0.3, s, mu, 0.0, sp["a_S"], sp["omega_S"]) - eom(0.3, s, mu)
        max_diff = max(max_diff, float(np.max(np.abs(d))))
    # solar accel vanishes at the barycenter
    a_origin = np.linalg.norm(
        solar_acceleration(np.zeros(3), 0.0, sp["m_S"], sp["a_S"], sp["omega_S"]))
    # tidal magnitude at the Moon
    a_moon = np.linalg.norm(
        solar_acceleration(np.array([1 - mu, 0, 0]), 0.0,
                           sp["m_S"], sp["a_S"], sp["omega_S"]))
    info = {**sp, "noSun_eom_diff": max_diff,
            "a_sun_origin": a_origin, "a_sun_moon": a_moon}
    # tidal accel at the Moon ~ 1.1e-2 nondim (= 3.05e-8 km/s^2, matches the SI
    # solar tidal field 2 G M_sun d / r_sun^3 across the Earth-Moon system).
    ok = (max_diff < 1e-12) and (a_origin < 1e-10) and (1e-3 < a_moon < 5e-2)
    return ok, info


def check_g8b(b) -> tuple[bool, dict]:
    ok = (3.85 < b["total_direct"] < 4.05
          and 7.6 < b["v_circ_leo"] < 7.9
          and 1.5 < b["v_circ_llo"] < 1.75
          and 2.9 < b["dv_tli"] < 3.3)
    return ok, b


def check_g8c(b) -> tuple[bool, dict]:
    ok = (b["capture_saving"] > 0.1) and (b["total_low_energy"] < b["total_direct"])
    return ok, b


def main() -> int:
    mu = EARTH_MOON.mu
    print(f"=== Ariadne Stage 4 validation  (Earth-Moon, mu={mu:.12f}) ===\n")

    ok_a, ia = check_g8a(mu)
    print("[G8a] BCR4BP (bicircular Sun-perturbed model)")
    print(f"      sun params: m_S={ia['m_S']:.1f}, a_S={ia['a_S']:.3f}, "
          f"omega_S={ia['omega_S']:+.5f} (period {2*np.pi/abs(ia['omega_S']):.3f} ~ 29.5 d)")
    print(f"      no-Sun limit EOM vs CR3BP = {ia['noSun_eom_diff']:.2e} (need < 1e-12)")
    print(f"      |a_sun| at barycenter = {ia['a_sun_origin']:.2e} (need < 1e-10)")
    print(f"      |a_sun| at Moon (tidal) = {ia['a_sun_moon']:.2e}")
    print(f"      -> {'PASS' if ok_a else 'FAIL'}\n")

    b = earth_moon_budget()
    ok_b, _ = check_g8b(b)
    print("[G8b] Delta-v budget vs Apollo-class direct transfer (LEO->LLO)")
    print(f"      v_circ LEO = {b['v_circ_leo']:.4f} km/s, v_circ LLO = {b['v_circ_llo']:.4f} km/s")
    print(f"      TLI = {b['dv_tli']*1000:.0f} m/s,  direct LOI = {b['dv_loi_direct']*1000:.0f} m/s "
          f"(v_inf={b['v_inf_direct']:.3f} km/s)")
    print(f"      TOTAL direct = {b['total_direct']*1000:.0f} m/s "
          f"(known Apollo-class ~3900-4000 m/s; Coimbra 'previous best' ~3992)")
    print(f"      -> {'PASS' if ok_b else 'FAIL'}\n")

    ok_c, _ = check_g8c(b)
    print("[G8c] Low-energy (ballistic-capture) mechanism")
    print(f"      ballistic LOI = {b['dv_loi_ballistic']*1000:.0f} m/s "
          f"(near-parabolic arrival)")
    print(f"      capture saving vs direct = {b['capture_saving']*1000:.0f} m/s")
    print(f"      TOTAL low-energy class = {b['total_low_energy']*1000:.0f} m/s "
          f"(Coimbra optimized result: 3925 m/s)")
    print(f"      -> {'PASS' if ok_c else 'FAIL'}\n")

    all_ok = ok_a and ok_b and ok_c
    print(f"=== STAGE 4: {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    print("NOTE: this reproduces the low-energy MECHANISM and Delta-v CLASS of the\n"
          "Coimbra result. Matching the exact 3925 m/s end-to-end requires their\n"
          "boundary conditions + full-ephemeris collocation (Stage 5).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
