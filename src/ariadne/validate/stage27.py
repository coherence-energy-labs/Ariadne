"""Stage 27 validation -- principled coherence field + trajectory-residual hidden-mass detector.

Uses REAL data end to end: JPL DE440 planet positions (SPICE), DE440-consistent GM constants,
the published osculating elements of the actual clustered eTNOs (the Batygin & Brown 2016
evidence set), and the published Planet 9 hypothesis parameters.

G27a (Newton recovery) - The principled coherence field tau_c = 1 + Phi/c^2 (from the total
                         potential of all known masses) gives g_coh = -c^2 grad ln(tau_c) that
                         reduces to Newtonian gravity to exactly |Phi|/c^2 (~1e-8 at Earth) -- the
                         framework's own "Newton recovery", reproduced inside Ariadne. The
                         -c^2 grad ln(tau_c) form is verified to machine precision in strong field.
G27b (residual detector) - The residual acceleration from an unmodeled mass equals (model-with) minus
                         (model-without) and matches the analytic GM/d^2 -- a self-consistent
                         trajectory-residual detector (the Neptune/Cassini method).
G27c (real-data Planet 9) - Evaluated at the REAL clustered eTNOs, the residual a hypothesized
                         Planet 9 imparts rises ABOVE the unmodeled-Kuiper-belt noise floor (so it is
                         distinguishable from small-body noise), while being only ~1e-5..1e-4 of the
                         solar pull -- which is why a real detection needs secular (Myr) accumulation
                         and many objects, NOT an instantaneous snapshot (beyond this short-arc engine).

Run:  PYTHONPATH=src python -m ariadne.validate.stage27
"""
from __future__ import annotations

import numpy as np

from ..data.constants import (GM_SUN, GM_MERCURY, GM_VENUS, GM_EARTH, GM_MARS,
                              GM_JUPITER, GM_SATURN, GM_URANUS, GM_NEPTUNE)
from ..data.ephemeris import et, body_state
from ..fields.tau_c import (newtonian_accel, coherence_accel, coherence_accel_fd,
                            tau_c, potential, C2)
from ..fields.hidden_mass import (CLUSTERED_ETNOS, PLANET9, elements_to_position,
                                  residual_accel, kuiper_noise_floor, GM_EARTH as GME)

EPOCH = "2026-01-01T00:00:00"
_PLANETS = [("MERCURY BARYCENTER", GM_MERCURY), ("VENUS BARYCENTER", GM_VENUS),
            ("EARTH", GM_EARTH), ("MARS BARYCENTER", GM_MARS),
            ("JUPITER BARYCENTER", GM_JUPITER), ("SATURN BARYCENTER", GM_SATURN),
            ("URANUS BARYCENTER", GM_URANUS), ("NEPTUNE BARYCENTER", GM_NEPTUNE)]


def _known_masses(e0):
    masses = [(GM_SUN, np.zeros(3))]
    for name, gm in _PLANETS:
        masses.append((gm, body_state(name, e0, "J2000", "SUN")[:3]))
    return masses


def check() -> tuple[bool, dict]:
    e0 = et(EPOCH)
    masses = _known_masses(e0)

    # G27a: Newton recovery at Earth + strong-field FD implementation check
    xE = body_state("EARTH", e0, "J2000", "SUN")[:3]
    gN, gC = newtonian_accel(xE, masses), coherence_accel(xE, masses)
    rel = float(np.linalg.norm(gC - gN) / np.linalg.norm(gN))
    phi_over_c2 = abs(potential(xE, masses)) / C2
    newton_ok = abs(rel - phi_over_c2) / phi_over_c2 < 1e-3 and rel < 1e-6
    # strong-field form check (no float64 cancellation)
    sf = [(8.99e15, np.zeros(3))]
    xsf = np.array([1.0e6, 0.0, 0.0])
    fd_rel = float(np.linalg.norm(coherence_accel_fd(xsf, sf, 1.0) - coherence_accel(xsf, sf))
                   / np.linalg.norm(coherence_accel(xsf, sf)))
    g27a = newton_ok and fd_rel < 1e-6

    # G27b: residual detector self-consistency
    gm_x = PLANET9["m_earth"] * GME
    pos_x = elements_to_position(PLANET9["a_au"], PLANET9["e"], PLANET9["i"],
                                 PLANET9["Omega"], PLANET9["omega"], 180.0)
    test = elements_to_position(263.1, 0.70, 24.0, 90.8, 293.8, 180.0)
    a_without = newtonian_accel(test, masses)
    a_with = newtonian_accel(test, masses + [(gm_x, pos_x)])
    res = residual_accel(test, gm_x, pos_x)
    g27b = float(np.linalg.norm((a_with - a_without) - res)) < 1e-18

    # G27c: real clustered-eTNO detectability vs the unmodeled-Kuiper floor
    rows = []
    for o in CLUSTERED_ETNOS:
        x = elements_to_position(o["a_au"], o["e"], o["i"], o["Omega"], o["omega"], 180.0)
        sig = float(np.linalg.norm(residual_accel(x, gm_x, pos_x))) * 1000.0
        solar = float(np.linalg.norm(newtonian_accel(x, [(GM_SUN, np.zeros(3))]))) * 1000.0
        floor = kuiper_noise_floor(x) * 1000.0
        rows.append({"name": o["name"], "r_au": np.linalg.norm(x) / 1.495979e8,
                     "signal_ms2": sig, "floor_ms2": floor, "solar_ms2": solar,
                     "sig_over_floor": sig / floor, "sig_frac_solar": sig / solar})
    n_above = sum(1 for r in rows if r["sig_over_floor"] > 1.0)
    g27c = n_above >= len(rows) - 1          # P9 rises above the small-body floor for the cluster

    ok = g27a and g27b and g27c
    return ok, {"rel": rel, "phi_over_c2": phi_over_c2, "fd_rel": fd_rel,
                "rows": rows, "n_above": n_above, "g27a": g27a, "g27b": g27b, "g27c": g27c}


def main() -> int:
    print("=== Ariadne Stage 27  (principled coherence field + hidden-mass detector) ===\n")
    ok, i = check()

    print("[G27a] Newton recovery: g_coh = -c^2 grad ln(tau_c) reduces to Newtonian gravity")
    print(f"      |g_coh - g_N|/|g_N| at Earth = {i['rel']:.3e}   |Phi|/c^2 = {i['phi_over_c2']:.3e}  (match)")
    print(f"      -c^2 grad ln(tau_c) form check (strong field) rel err = {i['fd_rel']:.2e}")
    print(f"      -> {'PASS' if i['g27a'] else 'FAIL'}\n")

    print("[G27b] Trajectory-residual detector self-consistency (residual = model_with - model_without)")
    print(f"      -> {'PASS' if i['g27b'] else 'FAIL'}\n")

    print("[G27c] Real clustered eTNOs vs a hypothesized Planet 9 "
          f"({PLANET9['m_earth']} M_earth @ {PLANET9['a_au']} AU)")
    print(f"      {'object':<12s} {'r(AU)':>7s} {'P9 signal':>11s} {'Kuiper floor':>13s} "
          f"{'sig/floor':>10s} {'sig/solar':>10s}")
    for r in i["rows"]:
        print(f"      {r['name']:<12s} {r['r_au']:7.0f} {r['signal_ms2']:11.2e} "
              f"{r['floor_ms2']:13.2e} {r['sig_over_floor']:10.1f} {r['sig_frac_solar']:10.2e}")
    print(f"      Planet 9 rises above the unmodeled-small-body floor for {i['n_above']}/{len(i['rows'])} "
          f"real eTNOs  -> {'PASS' if i['g27c'] else 'FAIL'}\n")

    print("  HONEST: the residual is real and distinguishable from small-body noise, but it is only")
    print("  ~1e-5..1e-4 of the solar pull -- so an actual detection needs SECULAR (Myr) accumulation")
    print("  over many objects (the eTNO clustering), not this instantaneous snapshot. Real precedent:")
    print("  Cassini range tracking already CONSTRAINS Planet 9 by this exact residual method.")
    print("  Long-term secular integration is the noted next tool; the engine here is short-arc.\n")
    print(f"=== STAGE 27: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
