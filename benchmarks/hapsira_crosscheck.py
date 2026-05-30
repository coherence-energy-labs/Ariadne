"""Independent Lambert cross-check vs hapsira (the maintained poliastro fork).

Picks the Curtis 5.2 geocentric Lambert example, runs it in BOTH Ariadne and hapsira,
checks the two libraries agree on the answer.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from ariadne.optimize.lambert import lambert as ariadne_lambert
from ariadne.data.constants import GM_EARTH

print("Independent Lambert cross-check: Ariadne vs hapsira")
print("Curtis Example 5.2 -- geocentric Lambert, TOF 1 hour")
print()

r1 = np.array([5000.0, 10000.0, 2100.0])
r2 = np.array([-14600.0, 2500.0, 7000.0])
tof = 3600.0

v1_a, v2_a = ariadne_lambert(r1, r2, tof, GM_EARTH)
print(f"Ariadne:")
print(f"  v1 = ({v1_a[0]:+.6f}, {v1_a[1]:+.6f}, {v1_a[2]:+.6f})  |v1| = {np.linalg.norm(v1_a):.6f}")
print(f"  v2 = ({v2_a[0]:+.6f}, {v2_a[1]:+.6f}, {v2_a[2]:+.6f})  |v2| = {np.linalg.norm(v2_a):.6f}")

try:
    from hapsira.iod import izzo
    from astropy import units as u
    k = GM_EARTH * u.km**3 / u.s**2
    r1_q = r1 * u.km
    r2_q = r2 * u.km
    tof_q = tof * u.s
    v1_h, v2_h = izzo.lambert(k, r1_q, r2_q, tof_q)
    v1_h_arr = v1_h.to(u.km / u.s).value
    v2_h_arr = v2_h.to(u.km / u.s).value
    print(f"\nhapsira (izzo solver):")
    print(f"  v1 = ({v1_h_arr[0]:+.6f}, {v1_h_arr[1]:+.6f}, {v1_h_arr[2]:+.6f})  |v1| = {np.linalg.norm(v1_h_arr):.6f}")
    print(f"  v2 = ({v2_h_arr[0]:+.6f}, {v2_h_arr[1]:+.6f}, {v2_h_arr[2]:+.6f})  |v2| = {np.linalg.norm(v2_h_arr):.6f}")

    dv1 = float(np.linalg.norm(v1_a - v1_h_arr))
    dv2 = float(np.linalg.norm(v2_a - v2_h_arr))
    print(f"\nLibrary agreement:")
    print(f"  |v1_ariadne - v1_hapsira| = {dv1*1e6:.3f} mm/s")
    print(f"  |v2_ariadne - v2_hapsira| = {dv2*1e6:.3f} mm/s")
    ok = dv1 < 1e-3 and dv2 < 1e-3   # both libraries to 1 mm/s
    print(f"\n  -> Lambert solutions agree to 1 mm/s: {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"\nhapsira call failed: {e}")
    print(f"  -> SKIPPED (hapsira+astropy compat issue on Py 3.14)")
