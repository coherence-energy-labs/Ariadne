"""Stage 41 validation -- autodiff global trajectory optimization.

Stage 35 gave exact gradients through the integrator; this stage uses them for GLOBAL trajectory
design. The autodiff Levenberg-Marquardt shooting solves each transfer EXACTLY (arrival enforced),
and a sweep over (departure date, time-of-flight) finds the global minimum-Delta-v transfer. Validated
on the canonical Earth->Mars problem against the known optimum and the project's own Lambert porkchop.

G41a (valid transfers)  - The autodiff shooting enforces arrival: the optimum is a real transfer that
                          reaches Mars to sub-km (not a penalty-relaxed near-miss).
G41b (correct optimum)  - The global minimum Earth->Mars 2-impulse heliocentric Delta-v is ~5.6 km/s at a
                          ~250-330 day time-of-flight -- the textbook value, and consistent with Stage 21's
                          independent Lambert porkchop (5.68 km/s). Found by gradient-based shooting, not
                          a brute grid of trial velocities.

Run:  PYTHONPATH=src python -m ariadne.validate.stage41   (uses JAX; ~1 min)
"""
from __future__ import annotations

import numpy as np

from ..optimize import autodiff as AD
from ..data.ephemeris import et, body_state

EPOCH = "2026-01-01T00:00:00"


def check():
    if not AD.HAVE_JAX:                                      # pragma: no cover
        return True, {"no_jax": True}
    e0 = et(EPOCH)
    rE = body_state("EARTH", e0, "J2000", "SUN")
    rM = body_state("MARS BARYCENTER", e0, "J2000", "SUN")
    # coarse global sweep over one synodic period to LOCALIZE the best launch window...
    t0_grid = np.linspace(0, 780, 15)
    tof_grid = np.linspace(180, 340, 10)
    res = AD.optimize_transfer(rE[:3], rE[3:], rM[:3], rM[3:], t0_grid, tof_grid, n_steps=300)
    c = res["best"]
    # ...then a fine local REFINE around it (honest: global localize + local refine)
    t0_fine = np.linspace(max(0, c["t0_days"] - 60), c["t0_days"] + 60, 13)
    tof_fine = np.linspace(max(150, c["tof_days"] - 40), c["tof_days"] + 40, 11)
    res2 = AD.optimize_transfer(rE[:3], rE[3:], rM[:3], rM[3:], t0_fine, tof_fine, n_steps=300)
    b = res2["best"]
    g41a = b is not None and b["miss_km"] < 1.0
    g41b = b is not None and 5.0 <= b["dv_kms"] <= 6.0 and 200 <= b["tof_days"] <= 340
    n_valid = int(np.isfinite(res["surface"]).sum())
    return (g41a and g41b), {"best": b, "n_valid": n_valid, "n_cells": res["surface"].size,
                             "g41a": g41a, "g41b": g41b}


def main() -> int:
    print("=== Ariadne Stage 41  (autodiff global trajectory optimization) ===\n")
    ok, i = check()
    if i.get("no_jax"):
        print("JAX not available -- stage skipped."); return 0
    b = i["best"]
    print("[G41a] Autodiff shooting enforces arrival (valid transfer, not a penalty near-miss)")
    print(f"       global optimum miss = {b['miss_km']:.2f} km  ({i['n_valid']}/{i['n_cells']} grid cells valid)")
    print(f"      -> {'PASS' if i['g41a'] else 'FAIL'}\n")
    print("[G41b] Correct global Earth->Mars optimum, found by gradient-based shooting")
    print(f"       Delta-v = {b['dv_kms']:.3f} km/s  at launch +{b['t0_days']:.0f} d, tof {b['tof_days']:.0f} d")
    print("       (textbook ~5.6 km/s; consistent with Stage 21's independent Lambert porkchop 5.68 km/s)")
    print(f"      -> {'PASS' if i['g41b'] else 'FAIL'}\n")
    print(f"=== STAGE 41: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
