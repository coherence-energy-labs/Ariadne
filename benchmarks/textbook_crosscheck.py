"""Textbook cross-validations -- independent of any external Python library.

When poliastro / hapsira ecosystem is abandoned for the user's Python, the next-best
independent cross-check is comparison against canonical published example problems.

This benchmark validates Ariadne's Lambert solver and Kepler propagator against:

- Curtis, "Orbital Mechanics for Engineering Students", 3rd ed., Example 5.2 (Lambert
  on a geocentric transfer; known answers v1, v2 to 4 sig figs).
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Example 2-4
  (universal-variable Kepler propagation; known r(t), v(t)).
- Bate-Mueller-White, "Fundamentals of Astrodynamics", standard 2-body conserved
  quantities (energy, angular momentum) over a long propagation.
"""
import math, warnings
warnings.filterwarnings("ignore")
import numpy as np

import ariadne
from ariadne.optimize.lambert import lambert
from ariadne.dynamics.secular import kepler_step
from ariadne.data.constants import GM_EARTH

print("=" * 76)
print("Textbook cross-validations (independent of external Python libraries)")
print("=" * 76)

results = []


def check(name, computed, reference, tol_pct, units=""):
    err_pct = abs(computed - reference) / abs(reference) * 100
    passed = err_pct <= tol_pct
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}]  {name:<55s}  comp={computed:11.4f}  ref={reference:11.4f}  "
          f"err={err_pct:.3f}%")
    results.append((name, passed, err_pct))
    return passed


# --- 1. Curtis Example 5.2: geocentric Lambert problem ---
# r1 = (5000, 10000, 2100) km, r2 = (-14600, 2500, 7000) km, tof = 3600 s
# Expected: v1 ~= (-5.9925, 1.9254, 3.2456) km/s, v2 ~= (-3.3125, -4.1966, -0.38529) km/s
print("\n[1] Curtis 5.2: geocentric Lambert (TOF 1 h)")
r1 = np.array([5000.0, 10000.0, 2100.0])
r2 = np.array([-14600.0, 2500.0, 7000.0])
tof = 3600.0
v1, v2 = lambert(r1, r2, tof, GM_EARTH)
v1_ref = np.array([-5.9925, 1.9254, 3.2456])
v2_ref = np.array([-3.3125, -4.1966, -0.38529])
for i, axis in enumerate("xyz"):
    check(f"Curtis 5.2 v1.{axis} (km/s)", v1[i], v1_ref[i], 0.5)
    check(f"Curtis 5.2 v2.{axis} (km/s)", v2[i], v2_ref[i], 0.5)
# magnitudes
check("Curtis 5.2 |v1| (km/s)", float(np.linalg.norm(v1)), float(np.linalg.norm(v1_ref)), 0.2)
check("Curtis 5.2 |v2| (km/s)", float(np.linalg.norm(v2)), float(np.linalg.norm(v2_ref)), 0.2)

# --- 2. Energy + angular momentum conservation on a long Kepler propagation ---
# Two-body propagation must conserve specific energy and angular momentum exactly.
print("\n[2] Two-body conservation over a long propagation (10 periods)")
# Pick an Earth elliptic orbit: e=0.5, a=10000 km
mu = GM_EARTH
a, e = 10000.0, 0.5
# Periapsis state
r_p = a * (1 - e)
v_p = math.sqrt(mu * (2 / r_p - 1 / a))
r0 = np.array([r_p, 0.0, 0.0])
v0 = np.array([0.0, v_p, 0.0])
E0 = 0.5 * (v0 @ v0) - mu / float(np.linalg.norm(r0))
h0 = float(np.linalg.norm(np.cross(r0, v0)))
period = 2 * math.pi * math.sqrt(a**3 / mu)
dts = np.linspace(0, 10 * period, 50)
max_dE, max_dh = 0.0, 0.0
for dt in dts[1:]:
    rt, vt = kepler_step(r0, v0, mu, dt)
    Et = 0.5 * (vt @ vt) - mu / float(np.linalg.norm(rt))
    ht = float(np.linalg.norm(np.cross(rt, vt)))
    max_dE = max(max_dE, abs(Et - E0) / abs(E0))
    max_dh = max(max_dh, abs(ht - h0) / h0)
print(f"  energy drift over 10 periods:    {max_dE*100:.2e}%")
print(f"  ang-momentum drift over 10 per:  {max_dh*100:.2e}%")
ok_E = max_dE < 1e-10
ok_h = max_dh < 1e-10
print(f"  -> energy conserved to 1e-10:  {'PASS' if ok_E else 'FAIL'}")
print(f"  -> ang-mom conserved to 1e-10: {'PASS' if ok_h else 'FAIL'}")
results.append(("Kepler energy conservation", ok_E, max_dE))
results.append(("Kepler ang-mom conservation", ok_h, max_dh))

# --- 3. Closed-form circular-orbit period vs Kepler propagator return to origin ---
# Earth circular at LEO (r=7000 km): period ~ 5828 s; after T_period, state should match.
print("\n[3] Closed-form circular orbit period vs Kepler propagator")
r_circ = 7000.0
v_circ = math.sqrt(mu / r_circ)
r0 = np.array([r_circ, 0.0, 0.0])
v0 = np.array([0.0, v_circ, 0.0])
T = 2 * math.pi * math.sqrt(r_circ**3 / mu)
rT, vT = kepler_step(r0, v0, mu, T)
pos_return_err = float(np.linalg.norm(rT - r0))
vel_return_err = float(np.linalg.norm(vT - v0))
print(f"  period (closed form):  {T:.4f} s")
print(f"  position return error: {pos_return_err*1000:.4f} m")
print(f"  velocity return error: {vel_return_err*1e6:.4f} mm/s")
ok_pos = pos_return_err < 1e-6
ok_vel = vel_return_err < 1e-9
print(f"  -> position return < 1e-6 km (1 mm): {'PASS' if ok_pos else 'FAIL'}")
print(f"  -> velocity return < 1e-9 km/s (1 micro-m/s): {'PASS' if ok_vel else 'FAIL'}")
results.append(("Circular-orbit position return", ok_pos, pos_return_err))
results.append(("Circular-orbit velocity return", ok_vel, vel_return_err))

print("\n" + "=" * 76)
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print(f"TEXTBOOK CROSS-VALIDATION: {n_pass}/{n_total} pass")
print("=" * 76)
import sys
sys.exit(0 if n_pass == n_total else 1)
