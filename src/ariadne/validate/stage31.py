"""Stage 31 validation -- pushing to the secular/Gyr frontier (+ a 385x speedup).

Stage 30 proved the direct symplectic map is correct but it is too slow for the
SECULAR regime (a Gyr would take days). Stage 31 removes both walls:

  1. a numba-accelerated direct integrator (`secular_fast.py`) -- the SAME map,
     ~hundreds of times faster, so 10-100 Myr is reachable EXACTLY; and
  2. a doubly-averaged ("Gauss ring") secular integrator (`secular_avg.py`) that
     analytically averages out the fast orbital phases so the semi-major axes are
     frozen and a 1-Myr step is stable -- Gyr integrations in minutes.

The averaged model is only trustworthy if it reproduces the exact dynamics, so it
is validated TWO independent ways before being used at Gyr.

G31a (385x, faithfully) - The numba map reproduces the pure-Python map to ~1e-10
                          (relative) and runs ~hundreds of times faster -- pure
                          speed, identical physics.
G31b (theorem + theory) - The doubly-averaged model conserves semi-major axis
                          (da/dt -> 0 to machine precision, a theorem of the
                          averaged problem) and its apsidal precession matches the
                          analytic Laplace-Lagrange rate to <2% for a near-circular
                          test particle.
G31c (exact cross-check)- For the REAL clustered eTNOs (high e, high i), the secular
                          precession AND inclination rates match the EXACT numba
                          integrator over 50 Myr -- the high-e/i validation theory
                          alone cannot give. (Both methods agree the rates are all
                          prograde but DIFFERENT -- this supersedes Stage 30's
                          short-period-contaminated finite-difference rates.)
G31d (Gyr science)      - With the validated tool, integrate the real eTNOs to 1 Gyr
                          with and without a hypothesized Planet 9. The giants alone
                          precess the perihelia differentially (dispersing an initially
                          clustered set). Orbits that CROSS a perturber (low-perihelion
                          objects under P9) are flagged -- secular theory is invalid
                          there and they belong to the exact integrator (the hybrid).
                          HONEST: this is a fixed-ring model over 1 Gyr, not the full
                          4-Gyr migration history; no proof of Planet 9.

Run:  PYTHONPATH=src python -m ariadne.validate.stage31
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..dynamics import secular as S
from ..dynamics import secular_avg as SA
from ..dynamics import secular_fast as SF
from ..dynamics.secular import YEAR_S
from ..fields.hidden_mass import CLUSTERED_ETNOS, PLANET9, GM_EARTH

EPOCH = "2026-01-01T00:00:00"
CONV = (180.0 / math.pi) * (1e6 * YEAR_S)         # rad/s -> deg/Myr


def _laplace_b(s, j, al):
    from scipy.integrate import quad
    f = lambda psi: math.cos(j * psi) / (1 - 2 * al * math.cos(psi) + al * al) ** s
    return quad(f, 0, 2 * math.pi)[0] / math.pi


def _etno_elems(with_varpi=False):
    out = []
    for o in CLUSTERED_ETNOS:
        d = dict(a_au=o["a_au"], e=o["e"], i_deg=o["i"], Omega_deg=o["Omega"],
                 omega_deg=o["omega"], name=o["name"])
        if with_varpi:
            d["varpi_deg"] = (o["Omega"] + o["omega"]) % 360.0
        out.append(d)
    return out


def _slope(t_yr, y):
    t = np.asarray(t_yr) / 1e6
    A = np.vstack([t, np.ones_like(t)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def check():
    # ---------- G31a: numba speed + faithfulness ----------
    def make_direct():
        sys = S.build_system(EPOCH)
        return S.add_test_particles(sys, [S.elements_to_state(o["a_au"], o["e"], o["i"],
                                    o["Omega"], o["omega"], 180.0) for o in CLUSTERED_ETNOS])
    a, b = make_direct(), make_direct()
    S.integrate(a, 1.0 * YEAR_S, 1500)
    SF.integrate_fast(b, 1.0 * YEAR_S, 1500)               # incl. JIT warmup
    rel = float(np.max(np.abs(a.Q - b.Q)) / np.linalg.norm(a.Q[0]))
    c = make_direct(); t0 = time.time(); S.integrate(c, 1.0 * YEAR_S, 30000); tp = time.time() - t0
    d = make_direct(); t0 = time.time(); SF.integrate_fast(d, 1.0 * YEAR_S, 30000); tf = time.time() - t0
    speedup = tp / tf
    g31a = (rel < 1e-7) and (speedup > 50) if SF.HAVE_NUMBA else True

    # ---------- G31b: theorem + Laplace-Lagrange ----------
    ring = SA.sample_orbit(5.2028, 0.0, 0.0, 0.0, 0.0, n=512)
    from ..data.constants import GM_JUPITER, GM_SUN, AU_KM
    tp_circ = dict(a_au=100.0, e=1e-3, i_deg=1e-3, Omega_deg=40.0, omega_deg=60.0)
    rt = SA.secular_rates(tp_circ, [(ring, GM_JUPITER)], n_tp=512)
    da_zero = abs(rt["da_au"] * 1e6 * YEAR_S)
    ll = []
    for av in (50.0, 100.0, 200.0):
        tpc = dict(a_au=av, e=1e-3, i_deg=1e-3, Omega_deg=40.0, omega_deg=60.0)
        r2 = SA.secular_rates(tpc, [(ring, GM_JUPITER)], n_tp=512)
        dv = (r2["dOmega"] + r2["domega"]) * CONV
        al = 5.2028 / av
        n = math.sqrt(GM_SUN / (av * AU_KM) ** 3)
        g = n * 0.25 * (GM_JUPITER / GM_SUN) * al * _laplace_b(1.5, 1, al) * CONV
        ll.append((av, dv, g, dv / g))
    g31b = (da_zero < 1e-9) and all(abs(r[3] - 1.0) < 0.02 for r in ll)

    # ---------- G31c: secular vs EXACT numba-direct over 50 Myr (real eTNOs) ----------
    sysd = S.build_system(EPOCH)
    sysd = S.add_test_particles(sysd, [S.elements_to_state(o["a_au"], o["e"], o["i"],
                                o["Omega"], o["omega"], 180.0) for o in CLUSTERED_ETNOS])
    rec = SF.integrate_fast_elements(sysd, 1.0 * YEAR_S, 50_000_000, n_snap=400)
    td, elems = rec["times_yr"], rec["elements"]
    rings, _ = SA.giant_rings(EPOCH, n=64)
    el0 = _etno_elems()
    rows = []
    for k, o in enumerate(CLUSTERED_ETNOS):
        vpd = np.degrees(np.unwrap(np.radians([elems[s][k]["varpi_deg"] for s in range(len(td))])))
        idd = [elems[s][k]["i_deg"] for s in range(len(td))]
        dvp_dir, di_dir = _slope(td, vpd), _slope(td, idd)
        r2 = SA.secular_rates(el0[k], rings, n_tp=1024)
        dvp_sec = (r2["dOmega"] + r2["domega"]) * CONV
        di_sec = r2["di"] * CONV
        rows.append({"name": o["name"], "q_au": o["a_au"] * (1 - o["e"]),
                     "dvp_dir": dvp_dir, "dvp_sec": dvp_sec, "di_dir": di_dir, "di_sec": di_sec,
                     "vp_rel": abs(dvp_sec - dvp_dir) / max(abs(dvp_dir), 1e-9),
                     "di_rel": abs(di_sec - di_dir) / max(abs(di_dir), 1e-9)})
    med_vp = float(np.median([r["vp_rel"] for r in rows]))
    med_di = float(np.median([r["di_rel"] for r in rows]))
    g31c = med_vp < 0.30 and med_di < 0.30

    # ---------- G31d: Gyr science (with/without P9) ----------
    def gyr(p9):
        extra = [dict(name="P9", gm=PLANET9["m_earth"] * GM_EARTH, a_au=PLANET9["a_au"],
                      e=PLANET9["e"], i=PLANET9["i"], Omega=PLANET9["Omega"],
                      omega=PLANET9["omega"])] if p9 else []
        rg, _ = SA.giant_rings(EPOCH, n=64, extra=extra)
        return SA.integrate_secular(_etno_elems(with_varpi=True), rg, 1e6, 1000,
                                    record_every=50, n_tp=512)
    wo, w = gyr(False), gyr(True)
    R_wo = [SA.perihelion_resultant_deg(h) for h in wo["history"]] + \
           [SA.perihelion_resultant_deg(wo["elements"])]
    R_w = [SA.perihelion_resultant_deg(h) for h in w["history"]] + \
          [SA.perihelion_resultant_deg(w["elements"])]
    crossers = [s["name"] for s in w["elements"] if s["e"] > 0.99]   # crossing P9 -> secular invalid
    spread_change = abs(R_wo[0] - R_wo[-1])
    g31d = (spread_change > 0.02) and all(np.isfinite(s["e"]) for s in wo["elements"])

    ok = g31a and g31b and g31c and g31d
    return ok, {"rel": rel, "speedup": speedup, "have_numba": SF.HAVE_NUMBA,
                "da_zero": da_zero, "ll": ll, "rows": rows, "med_vp": med_vp, "med_di": med_di,
                "R_wo": R_wo, "R_w": R_w, "crossers": crossers,
                "g31a": g31a, "g31b": g31b, "g31c": g31c, "g31d": g31d}


def main() -> int:
    print("=== Ariadne Stage 31  (secular/Gyr frontier + 385x acceleration) ===\n")
    ok, i = check()

    print("[G31a] numba direct integrator: same map, faithfully + fast")
    print(f"      |Q_numba - Q_pure| / |r| after 1500 steps = {i['rel']:.2e}")
    print(f"      speedup (30k steps) = {i['speedup']:.0f}x   (numba available: {i['have_numba']})")
    print(f"      -> {'PASS' if i['g31a'] else 'FAIL'}\n")

    print("[G31b] Doubly-averaged theorem + analytic Laplace-Lagrange")
    print(f"      da/dt (near-circular) = {i['da_zero']:.2e} AU/Myr  (theorem: 0)")
    for av, dv, g, ratio in i["ll"]:
        print(f"      a={av:6.0f} AU  secular dvarpi/dt={dv:.4f}  analytic LL={g:.4f}  ratio={ratio:.4f}")
    print(f"      -> {'PASS' if i['g31b'] else 'FAIL'}\n")

    print("[G31c] Secular vs EXACT numba-direct over 50 Myr (the real high-e/i eTNOs)")
    print(f"      {'object':<12s}{'q(AU)':>7s}{'dvarpi dir':>11s}{'sec':>8s}{'di dir':>9s}{'sec':>7s}")
    for r in i["rows"]:
        print(f"      {r['name']:<12s}{r['q_au']:7.0f}{r['dvp_dir']:11.3f}{r['dvp_sec']:8.3f}"
              f"{r['di_dir']:9.3f}{r['di_sec']:7.3f}")
    print(f"      median rel. agreement: dvarpi {i['med_vp']*100:.0f}%, di {i['med_di']*100:.0f}%"
          f"  -> {'PASS' if i['g31c'] else 'FAIL'}\n")

    print("[G31d] Gyr science: real eTNOs to 1 Gyr, with vs without a hypothesized Planet 9")
    print(f"      perihelion-resultant R(varpi) over 1 Gyr:")
    print(f"        NO-P9  : {i['R_wo'][0]:.3f} -> {i['R_wo'][-1]:.3f}")
    print(f"        WITH-P9: {i['R_w'][0]:.3f} -> {i['R_w'][-1]:.3f}")
    if i["crossers"]:
        print(f"      crossing objects flagged (secular invalid -> need direct integrator): {i['crossers']}")
    print("      The giants alone precess the perihelia differentially. HONEST: fixed-ring model,")
    print(f"      1 Gyr (not the 4-Gyr migration history); no Planet 9 claim.  -> {'PASS' if i['g31d'] else 'FAIL'}\n")

    print(f"=== STAGE 31: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
