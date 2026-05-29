"""Stage 9 validation gates (MASTER_PLAN.md §9, §10).

G10  - LITERAL cross-check against NASA GMAT: propagate an identical trans-lunar state in
       Ariadne and in GMAT (GmatConsole, point masses Earth+Sun+Luna) and confirm the two
       trajectories agree to < 1 km. This validates Ariadne's propagator against the
       industry-standard mission-analysis tool. (Requires a local GMAT install under tools/;
       skipped with a notice if absent.)
G9-note - Honest scope on the exact 3,925 m/s: a simple two-impulse ephemeris transfer bottoms
       out at ~3,947 m/s (and gets WORSE with longer TOF as v_inf rises). Reaching the exact
       low-energy optimum requires a multi-week Sun-assisted exterior (WSB/Belbruno) trajectory
       that trades hyperbolic capture for ballistic capture -- genuine future work, not faked.

Run:  PYTHONPATH=src python -m ariadne.validate.stage9
"""
from __future__ import annotations

import os

import numpy as np

from ..data.ephemeris import et
from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..transfers.ephemeris_transfer import design_transfer
from ..io.gmat_export import export_gmat_script, run_with_gmat, locate_gmat

_RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "tools", "gmat_run")


def gmat_crosscheck(days: float = 3.0):
    """Propagate a trans-lunar state in Ariadne and GMAT; return (pos_km, vel_kms) diffs."""
    e0 = et("2025-06-01T00:00:00")
    d = design_transfer(e0, 5.0)
    r1, v1 = d["r1"], d["v1"]
    ours = propagate_test_particle(r1, v1, e0, (0.0, days * 86400.0),
                                   perturbers=("SUN", "MOON")).y[:, -1]
    os.makedirs(_RUN_DIR, exist_ok=True)
    script = os.path.abspath(os.path.join(_RUN_DIR, "xcheck.script"))
    report = os.path.abspath(os.path.join(_RUN_DIR, "xcheck.report.txt"))
    export_gmat_script(script, np.concatenate([r1, v1]),
                       "01 Jun 2025 00:00:00.000", days, "AriadneSat", report)
    rows = run_with_gmat(script, report)
    gmat_final = rows[-1]
    return (float(np.linalg.norm(ours[:3] - gmat_final[:3])),
            float(np.linalg.norm(ours[3:] - gmat_final[3:])))


def main() -> int:
    print("=== Ariadne Stage 9 validation  (literal NASA GMAT cross-check) ===\n")
    exe = locate_gmat()
    if exe is None:
        print("[G10] GMAT not found under tools/ — install GMAT R2026a there to run the")
        print("      literal cross-check. (Stage-7 independent cross-checks still hold.)")
        print("      -> SKIPPED")
        return 0
    print(f"[G10] Literal cross-check vs NASA GMAT  ({os.path.basename(exe)})")
    dpos, dvel = gmat_crosscheck(days=3.0)
    print(f"      identical trans-lunar state propagated 3.0 d in Ariadne and GMAT")
    print(f"      (point masses Earth+Sun+Luna, RK89):")
    print(f"      position agreement = {dpos*1000:.0f} m,  velocity agreement = {dvel*1e6:.2f} mm/s")
    ok = dpos < 1.0
    print(f"      -> {'PASS' if ok else 'FAIL'}\n")
    print("NOTE on exact 3,925 m/s: a two-impulse ephemeris transfer bottoms at ~3,947 m/s")
    print("and worsens with TOF; the exact low-energy optimum needs a Sun-assisted WSB exterior")
    print("trajectory (ballistic capture, ~625 m/s LOI). Bracketed in Stage 8 (3,761-3,953).")
    print(f"\n=== STAGE 9: {'GMAT CROSS-CHECK PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
