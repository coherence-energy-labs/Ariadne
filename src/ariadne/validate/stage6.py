"""Stage 6 validation gates (MASTER_PLAN.md §9, §10).

G8r - Ballistic lunar capture from REAL CR3BP manifold dynamics: a libration-orbit
      unstable manifold delivers the spacecraft to LLO altitude with a near-parabolic
      arrival, so the lunar-orbit-insertion burn is markedly cheaper than a direct
      hyperbolic insertion.
G8t - End-to-end low-energy LEO->LLO transfer (TLI + ballistic capture) lands in the
      low-energy class and brackets the published Coimbra 3,925 m/s result.

HONEST SCOPE: the TLI is the standard Hohmann value (held common with the direct case);
the manifold's rigorous contribution is the cheaper capture. Matching 3,925 m/s exactly
needs the paper's boundary conditions + a Sun-assisted (BCR4BP) departure optimization
(Stage 7). We report what the CR3BP construction actually yields, and do not fake the number.

Run:  PYTHONPATH=src python -m ariadne.validate.stage6
"""
from __future__ import annotations

from ..data.constants import EARTH_MOON
from ..orbits.families import lyapunov_orbit_at_jacobi
from ..transfers.lunar_capture import ballistic_capture
from ..transfers.low_energy_lunar import low_energy_lunar_transfer
from ..optimize.budget import earth_moon_budget


def check_capture(mu) -> tuple[bool, dict]:
    orbit = lyapunov_orbit_at_jacobi(mu, "L1", 3.15)
    cap = ballistic_capture(orbit, llo_alt=100.0)
    direct_loi = earth_moon_budget()["dv_loi_direct"]
    info = {"periapsis_alt_km": cap["periapsis_alt_km"],
            "v_peri_kms": cap["v_peri_kms"],
            "loi_ballistic_ms": cap["dv_capture_kms"] * 1000.0,
            "loi_direct_ms": direct_loi * 1000.0,
            "saving_ms": (direct_loi - cap["dv_capture_kms"]) * 1000.0,
            "tof_days": cap["tof_days"]}
    ok = (abs(cap["periapsis_alt_km"] - 100.0) < 60.0
          and 0.5 < cap["dv_capture_kms"] < 0.8
          and cap["dv_capture_kms"] < direct_loi)
    return ok, info


def check_transfer(mu) -> tuple[bool, dict]:
    best, recs, base = low_energy_lunar_transfer()
    info = {"best": best, "baseline": base, "n": len(recs)}
    ok = (best is not None
          and 3600.0 < best["total_ms"] < 3960.0
          and (base["direct_total_ms"] - best["total_ms"]) > 100.0)
    return ok, info


def main() -> int:
    mu = EARTH_MOON.mu
    print("=== Ariadne Stage 6 validation  (low-energy lunar transfer, real CR3BP) ===\n")

    ok1, i1 = check_capture(mu)
    print("[G8r] Ballistic lunar capture (real manifold dynamics)")
    print(f"      periapsis altitude = {i1['periapsis_alt_km']:.1f} km (target 100)")
    print(f"      Moon-relative periapsis speed = {i1['v_peri_kms']:.4f} km/s "
          f"(parabolic at 100 km = 2.310)")
    print(f"      ballistic LOI = {i1['loi_ballistic_ms']:.0f} m/s  vs  direct LOI "
          f"{i1['loi_direct_ms']:.0f} m/s  ->  saving {i1['saving_ms']:.0f} m/s")
    print(f"      manifold coast TOF = {i1['tof_days']:.1f} days")
    print(f"      -> {'PASS' if ok1 else 'FAIL'}\n")

    ok2, i2 = check_transfer(mu)
    b, base = i2["best"], i2["baseline"]
    print("[G8t] End-to-end low-energy LEO->LLO transfer (TLI + ballistic capture)")
    print(f"      best: {b['point']} C={b['jacobi']:.2f}  "
          f"TLI {b['tli_ms']:.0f} + LOI {b['loi_ms']:.0f} = {b['total_ms']:.0f} m/s  "
          f"(TOF {b['tof_days']:.1f} d)")
    print(f"      direct (hyperbolic) = {base['direct_total_ms']:.0f} m/s; "
          f"Coimbra optimized = {base['coimbra_ms']:.0f} m/s")
    print(f"      saving vs direct = {base['direct_total_ms'] - b['total_ms']:.0f} m/s")
    print(f"      -> {'PASS' if ok2 else 'FAIL'}\n")

    all_ok = ok1 and ok2
    print(f"=== STAGE 6: {'ALL GATES PASS' if all_ok else 'FAILURE'} ===")
    print("NOTE: rigorous result = the ballistic-capture LOI from real manifold dynamics.\n"
          "Total brackets Coimbra 3,925 m/s; exact match needs their BCs + BCR4BP\n"
          "departure optimization (Stage 7). Numbers are reported, not fitted.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
