"""Stage 20 validation gates (MASTER_PLAN.md - multi-moon tour mining via Tisserand graph).

G20a (physical)  - The Tisserand/v_inf structure of the Galilean tour is physical: each leg's
                   v_inf is a sensible fraction of the moon's orbital speed, and every flyby has
                   ample turn authority (max turn angle well above the small redirection needed).
G20b (assist)    - A gravity-assist tour's DETERMINISTIC Delta-v (the v_inf mismatches a flyby
                   cannot fix) is more than 10x below the propulsive Hohmann baseline -- the
                   gravity-assist saving that makes the Petit Grand Tour possible.

Honest: this is the energy/Tisserand structure (phasing, resonance timing, and plane changes not
modeled). Separately, the planar L1<->L2 transport graph (Stages 14-15) DOES build for a Jovian
system, but is sparse in the narrow small-mu libration band -- the Tisserand graph is the right
tool for INTER-moon tours, the transport graph for INTRA-system libration routing.

Run:  PYTHONPATH=src python -m ariadne.validate.stage20
"""
from __future__ import annotations

import math

from ..data.constants import GM_JUPITER
from ..transfers.tisserand import moon_tour, GALILEAN_TOUR


def check() -> tuple[bool, dict]:
    t = moon_tour(flyby_alt_km=200.0)

    # G20a: physical v_inf + turn authority
    vmax = max(math.sqrt(GM_JUPITER / m[1]) for m in GALILEAN_TOUR)
    physical = all(0.0 < leg["vinf_inner_kms"] < vmax and 0.0 < leg["vinf_outer_kms"] < vmax
                   for leg in t["legs"])
    turn_ok = all(leg["turn_inner_deg"] > 20.0 and leg["turn_outer_deg"] > 20.0
                  for leg in t["legs"])
    g20a = physical and turn_ok

    # G20b: gravity-assist saving
    g20b = t["ga_deterministic_dv_ms"] < 0.1 * t["hohmann_dv_ms"]

    ok = g20a and g20b
    return ok, {**t, "g20a": g20a, "g20b": g20b}


def main() -> int:
    print("=== Ariadne Stage 20 validation  (multi-moon tour mining via Tisserand graph) ===\n")
    ok, t = check()

    print("[G20a] Galilean gravity-assist tour (Io -> Callisto): v_inf + flyby turn authority")
    for leg in t["legs"]:
        print(f"      {leg['from']:<9s}-> {leg['to']:<9s}  v_inf {leg['vinf_inner_kms']:.3f} / "
              f"{leg['vinf_outer_kms']:.3f} km/s   max turn {leg['turn_inner_deg']:.0f} / "
              f"{leg['turn_outer_deg']:.0f} deg")
    print(f"      -> {'PASS' if t['g20a'] else 'FAIL'}\n")

    print("[G20b] Gravity-assist saving vs propulsive Hohmann")
    for j in t["junctions"]:
        print(f"      junction at {j['moon']:<9s}: v_inf {j['vinf_in']:.3f} -> {j['vinf_out']:.3f} km/s"
              f"  => {j['dv_ms']:.0f} m/s")
    print(f"      gravity-assist deterministic Delta-v = {t['ga_deterministic_dv_ms']:.0f} m/s")
    print(f"      Hohmann baseline                     = {t['hohmann_dv_ms']:.0f} m/s")
    print(f"      saving = {t['saving_ms']:.0f} m/s "
          f"({t['hohmann_dv_ms'] / max(t['ga_deterministic_dv_ms'], 1e-9):.1f}x cheaper)")
    print(f"      -> {'PASS' if t['g20b'] else 'FAIL'}\n")

    print(f"=== STAGE 20: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
