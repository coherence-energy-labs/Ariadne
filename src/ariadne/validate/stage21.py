"""Stage 21 validation gates (MASTER_PLAN.md - interplanetary porkchop + global Lambert).

G21a (windows)  - Sweeping the LAUNCH EPOCH over years recovers the real Earth->Mars launch-window
                  cadence (~26 months, the Mars synodic period) at realistic cost (C3 a few to ~20
                  km^2/s^2; total Delta-v with a propulsive Mars capture ~5-7 km/s).
G21b (global)   - A global optimizer (differential evolution) over (epoch, TOF) finds an optimum at
                  least as good as the porkchop grid minimum, at a sensible date/geometry.
G21c (pareto)   - The time-vs-energy Pareto front is monotonic (faster transfers cost more) and has
                  a balance knee -- the coherence principle applied to route choice.

This frees the variable we had been ignoring (time of year / planet geometry) and answers
"the best/fastest way to Mars across all epochs" on the real DE440 ephemeris.

Run:  PYTHONPATH=src python -m ariadne.validate.stage21
"""
from __future__ import annotations

import numpy as np

from ..data.ephemeris import et, utc
from ..interplanetary.porkchop import (
    porkchop, optimize_window, launch_windows, time_energy_pareto, coherent_knee,
)

START = "2026-01-01T00:00:00"
DEP, ARR = "EARTH", "MARS BARYCENTER"


def _windows(lw, min_sep_days=300.0):
    """Local minima of best-over-TOF Delta-v, deduped so each launch window appears once.

    Raw local minima can include a window's secondary lobe; we cluster minima within
    min_sep_days and keep the deepest per cluster (the actual window)."""
    dv, dep = lw["best_dv_ms"], lw["dep_grid"]
    med = np.nanmedian(dv)
    raw = [{"utc": utc(dep[k])[:7], "et": float(dep[k]), "total_ms": float(dv[k]),
            "tof_days": float(lw["best_tof_days"][k])}
           for k in range(2, len(dv) - 2)
           if dv[k] < dv[k - 1] and dv[k] < dv[k + 1] and dv[k] < med]
    out = []
    for w in sorted(raw, key=lambda r: r["et"]):
        if out and (w["et"] - out[-1]["et"]) / 86400.0 < min_sep_days:
            if w["total_ms"] < out[-1]["total_ms"]:
                out[-1] = w                       # keep the deeper of the cluster
        else:
            out.append(w)
    return out


def check() -> tuple[bool, dict]:
    e0 = et(START)
    lw = launch_windows(DEP, ARR, e0, years=6.0, tof_range=(120, 400), n_dep=150, n_tof=28)
    wins = _windows(lw)
    # cadence: gaps between consecutive windows ~ 26 months (780 d)
    gaps_d = [(_b["et"] - _a["et"]) / 86400.0 for _a, _b in zip(wins[:-1], wins[1:])]
    cadence_ok = len(wins) >= 2 and all(600 < g < 950 for g in gaps_d)
    cost_ok = all(4000 < w["total_ms"] < 8000 for w in wins)
    g21a = cadence_ok and cost_ok

    pk = porkchop(DEP, ARR, e0, dep_days=540, tof_range=(120, 400), n_dep=50, n_tof=40)
    gb = pk["grid_best"]
    opt = optimize_window(DEP, ARR, e0, dep_days=540, tof_range=(120, 400), maxiter=50)
    g21b = (opt is not None and opt["total_ms"] <= gb["total_ms"] + 1.0
            and 5.0 <= opt["c3"] <= 25.0 and 120 <= opt["tof_days"] <= 400)

    front = time_energy_pareto(pk)
    monotonic = all(a["total_ms"] >= b["total_ms"] - 1.0
                    for a, b in zip(front[:-1], front[1:]))   # longer TOF -> cheaper
    knee = coherent_knee(front)
    g21c = len(front) >= 3 and monotonic and knee is not None

    ok = g21a and g21b and g21c
    return ok, {"windows": wins, "gaps_months": [g / 30.44 for g in gaps_d],
                "grid_best": gb, "opt": opt, "front": front, "knee": knee,
                "g21a": g21a, "g21b": g21b, "g21c": g21c}


def main() -> int:
    print("=== Ariadne Stage 21 validation  (interplanetary porkchop + global Lambert) ===\n")
    ok, i = check()

    print("[G21a] Earth->Mars launch windows (epoch swept 6 yr) -- the ~26-month Mars cadence")
    for w in i["windows"]:
        print(f"      {w['utc']}  total {w['total_ms']:.0f} m/s  TOF {w['tof_days']:.0f} d")
    print(f"      window gaps (months): {[round(g, 1) for g in i['gaps_months']]}")
    print(f"      -> {'PASS' if i['g21a'] else 'FAIL'}\n")

    o, gb = i["opt"], i["grid_best"]
    print("[G21b] Global optimum (differential evolution over epoch + TOF)")
    print(f"      depart {o['utc_dep'][:10]}  arrive {o['utc_arr'][:10]}  TOF {o['tof_days']:.0f} d")
    print(f"      C3 {o['c3']:.2f} km^2/s^2  dep v_inf {o['dep_vinf_kms']:.3f}  arr v_inf {o['arr_vinf_kms']:.3f} km/s")
    print(f"      total Delta-v {o['total_ms']:.0f} m/s (grid best {gb['total_ms']:.0f})")
    print(f"      -> {'PASS' if i['g21b'] else 'FAIL'}\n")

    print("[G21c] Time-vs-energy Pareto + the coherence-balanced knee")
    for p in i["front"][::max(1, len(i["front"]) // 6)]:
        print(f"      TOF {p['tof_days']:.0f} d -> {p['total_ms']:.0f} m/s")
    if i["knee"]:
        print(f"      balanced knee: TOF {i['knee']['tof_days']:.0f} d, {i['knee']['total_ms']:.0f} m/s")
    print(f"      -> {'PASS' if i['g21c'] else 'FAIL'}\n")

    print(f"=== STAGE 21: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
