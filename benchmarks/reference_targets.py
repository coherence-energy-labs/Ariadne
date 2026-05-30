"""Reference-target benchmark suite: validate Ariadne against published values.

No external tools required (GMAT, STK, Monte). Each benchmark compares an Ariadne
computation to a published reference value with explicit tolerance.

Run:  PYTHONPATH=src python benchmarks/reference_targets.py
"""
from __future__ import annotations

import warnings, math, time
warnings.filterwarnings("ignore")
import numpy as np

import ariadne
from ariadne.data.constants import EARTH_MOON, GM_SUN, AU_KM
from ariadne.orbits.families import lyapunov_orbit_at_jacobi
from ariadne.discovery import linkage as L, iod as IOD

print("=" * 76)
print(f"Ariadne reference-target benchmark suite (version {ariadne.__version__})")
print("=" * 76)
print()

results = []


def check(name, computed, reference, tol_pct, units=""):
    err_pct = abs(computed - reference) / abs(reference) * 100
    passed = err_pct <= tol_pct
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}]  {name:<48s}  computed={computed:8.3f}{units}  "
          f"ref={reference:8.3f}{units}  err={err_pct:.2f}%  (tol {tol_pct}%)")
    results.append((name, passed, err_pct))
    return passed


def check_within(name, computed, reference, tol_abs, units=""):
    err = abs(computed - reference)
    passed = err <= tol_abs
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}]  {name:<48s}  computed={computed:8.3f}{units}  "
          f"ref={reference:8.3f}{units}  err={err:.3f}{units}  (tol {tol_abs}{units})")
    results.append((name, passed, err))
    return passed


# --- 1. CR3BP system constants vs literature ---
print("[1] CR3BP system constants (vs literature)")
em = EARTH_MOON
check("Earth-Moon mass parameter mu",            em.mu,    0.012150585,   1.0)
check("Earth-Moon L* characteristic length (km)", em.L_star, 384400.0,     0.5, " km")

# --- 2. Lagrange points vs published positions ---
print("\n[2] Lagrange points (vs literature)")
from ariadne.orbits.lagrange import lagrange_points
pts = lagrange_points(em.mu)
# Earth-Moon L1 published: ~0.83692 nondim from barycenter
check_within("Earth-Moon L1 x position (nondim)", pts["L1"][0], 0.83692, 0.0001)
check_within("Earth-Moon L2 x position (nondim)", pts["L2"][0], 1.15568, 0.0001)
# L4/L5 symmetric, y = ±sqrt(3)/2
check_within("Earth-Moon L4 y position (nondim)", pts["L4"][1],  math.sqrt(3)/2, 1e-9)
check_within("Earth-Moon L5 y position (nondim)", pts["L5"][1], -math.sqrt(3)/2, 1e-9)

# --- 3. Lyapunov orbit at known Jacobi (Koon-Lo-Marsden-Ross §2.7) ---
print("\n[3] Lyapunov orbit (vs Koon-Lo-Marsden-Ross reference)")
orb_c316 = lyapunov_orbit_at_jacobi(em.mu, "L1", 3.16)
# half-period residual is the differential-correction periodicity score
check_within("L1 Lyapunov at C=3.16 half-period residual", orb_c316.half_period_residual, 0.0, 1e-9)
check_within("L1 Lyapunov Jacobi recovery", orb_c316.jacobi, 3.16, 0.01)

# --- 4. NRHO vs NASA Gateway 9:2 specification ---
print("\n[4] Gateway-class NRHO (vs NASA Gateway specification)")
nrho = ariadne.gateway_nrho()
period_d = nrho.period * em.T_star / 86400.0
from ariadne.dynamics.cr3bp import propagate
sol = propagate(nrho.s0, (0.0, nrho.period), em.mu, t_eval=np.linspace(0.0, nrho.period, 800))
d_moon_km = np.sqrt((sol.y[0] - (1 - em.mu)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * em.L_star
peri_km, apo_km = float(d_moon_km.min()), float(d_moon_km.max())
check_within("NRHO period (days)",        period_d, 6.56,    0.05, " d")
check_within("NRHO perilune (km)",        peri_km,  3300,    600,  " km")
check_within("NRHO apolune (km)",         apo_km,   70000,   5000, " km")

# --- 5. Real TNO orbit fit residuals ---
print("\n[5] TNO orbit fit on real MPC astrometry (vs JPL elements)")
TNO_REFS = [
    ("Sedna",     "90377",  506.0, 0.85, 11.93),
    ("Quaoar",    "50000",   43.2, 0.04,  7.99),
]
for name, desig, a_ref, e_ref, i_ref in TNO_REFS:
    fit = ariadne.discover_tno(desig)
    if fit is None:
        print(f"  [SKIP]  {name}: too few tracklets")
        continue
    r, v = np.asarray(fit["x_fit"]), np.asarray(fit["v_fit"])
    rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    a_au = (1.0 / (2.0 / rn - vn ** 2 / GM_SUN)) / AU_KM
    h = np.cross(r, v); hn = float(np.linalg.norm(h))
    e_vec = np.cross(v, h) / GM_SUN - r / rn
    ecc = float(np.linalg.norm(e_vec))
    print(f"  {name:<10s}: a={a_au:7.1f} AU (JPL {a_ref:.1f}), e={ecc:.3f} (JPL {e_ref:.2f}), "
          f"RMS={fit['rms_arcsec']:.2f}\"")
    check(f"  {name} semi-major axis (AU)", a_au, a_ref, 15.0, " AU")
    # RMS < 10 arcsec = the discovery-filter threshold
    if fit["rms_arcsec"] < 10.0:
        results.append((f"{name} RMS < 10 arcsec", True, fit["rms_arcsec"]))
        print(f"  [PASS]  {name:<48s}  RMS={fit['rms_arcsec']:.2f}\" < 10\" filter threshold")
    else:
        results.append((f"{name} RMS < 10 arcsec", False, fit["rms_arcsec"]))
        print(f"  [FAIL]  {name:<48s}  RMS={fit['rms_arcsec']:.2f}\" >= 10\" threshold")

# --- 6. Jacobi conservation on a long integration ---
print("\n[6] Jacobi-constant conservation (CR3BP integrator quality)")
from ariadne.dynamics.cr3bp import jacobi_constant
orb = lyapunov_orbit_at_jacobi(em.mu, "L1", 3.18)
sol = propagate(orb.s0, (0.0, 20.0 * orb.period), em.mu, t_eval=np.linspace(0.0, 20.0 * orb.period, 1000))
c0 = jacobi_constant(sol.y[:, 0], em.mu)
cs = np.array([jacobi_constant(sol.y[:, i], em.mu) for i in range(sol.y.shape[1])])
dc_max = float(np.max(np.abs(cs - c0)))
check_within("|dC| over 20 periods", dc_max, 0.0, 1e-9)

# --- Summary ---
print("\n" + "=" * 76)
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print(f"REFERENCE BENCHMARK SUITE: {n_pass}/{n_total} pass")
print("=" * 76)
import sys
sys.exit(0 if n_pass == n_total else 1)
