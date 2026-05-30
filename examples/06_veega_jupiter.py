"""Galileo-class Venus-Earth-Earth Gravity Assist (VEEGA) to Jupiter.

A direct Earth->Jupiter transfer needs a launch C3 ~ 85 km^2/s^2 -- beyond most launchers.
A gravity-assist chain (Galileo's VEEGA) borrows momentum from Venus + Earth + Earth and
slashes the launch energy. Ariadne reproduces this in patched conics with optional per-leg
Deep-Space Maneuvers.

Run:  PYTHONPATH=src python examples/06_veega_jupiter.py
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

from ariadne.interplanetary.flyby import reference_veega
from ariadne.data.ephemeris import body_state

# Evaluate the stored Galileo-class VEEGA reference (5-leg patched-conic)
veega = reference_veega()
print("Galileo-class VEEGA (Earth-Venus-Earth-Earth-Jupiter)")
print(f"  launch C3 = {veega['c3']:.2f} km^2/s^2  (v_inf = {veega['dep_vinf_kms']:.2f} km/s)")
print(f"  TOF total = {veega['tof_total_days']/365.25:.1f} years")
print(f"  launch dv = {veega['dv_launch_ms']:.0f} m/s")
print(f"  flyby mismatch (powered) = {veega['mismatch_dv_ms']:.0f} m/s")
print(f"  TOTAL    = {veega['total_dv_ms']:.0f} m/s")
print(f"\nFlybys:")
for f in veega["flybys"]:
    print(f"  {f['body']:<6s}  v_inf {f['vinf_in_kms']:.2f}/{f['vinf_out_kms']:.2f} km/s   "
          f"turn {f['turn_req_deg']:.0f}/{f['turn_max_deg']:.0f} deg   "
          f"feasible = {f['feasible']}")
print(f"  arrival v_inf at Jupiter = {veega['arr_vinf_kms']:.2f} km/s")

# Plot heliocentric trajectory: planet positions at each leg's epoch
fig, ax = plt.subplots(figsize=(9, 9))
planet_colors = {"EARTH": "royalblue", "VENUS": "darkorange", "JUPITER BARYCENTER": "tan"}
AU_KM = 149597870.7

# planet orbits as reference circles
for body, R_au, color in [("VENUS", 0.72, "darkorange"),
                          ("EARTH", 1.0, "royalblue"),
                          ("JUPITER BARYCENTER", 5.20, "tan")]:
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_au * np.cos(th), R_au * np.sin(th), color=color, ls='--', lw=0.5, alpha=0.6)

# spacecraft trajectory: planet position at each epoch (connected with arcs)
xs, ys = [], []
for body, ep in zip(veega['bodies'], veega['epochs']):
    state = body_state(body, ep, "J2000", "SUN")
    xs.append(state[0] / AU_KM); ys.append(state[1] / AU_KM)

# Lambert arcs between consecutive bodies (simple connecting line is approximate, but reads)
for i in range(len(xs) - 1):
    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color='crimson', lw=1.5, alpha=0.8)
for body, x, y in zip(veega['bodies'], xs, ys):
    ax.scatter([x], [y], s=160, c=planet_colors.get(body, 'gray'),
               edgecolors='black', linewidth=1.2, zorder=5)
    ax.annotate(body.replace(" BARYCENTER", ""), (x, y), xytext=(8, 8),
                textcoords='offset points', fontsize=8)
ax.scatter([0], [0], s=300, c='gold', edgecolors='black', label='Sun', zorder=4)
ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
ax.set_title(f"Galileo-class VEEGA to Jupiter: C3 = {veega['c3']:.1f} km²/s² "
             f"({veega['tof_total_days']/365.25:.1f} y)\n"
             f"vs direct Earth->Jupiter min C3 ~ 85 km²/s² (4.6x energy reduction)")
ax.set_aspect('equal'); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("examples_out/06_veega_jupiter.png", dpi=120)
print("\nWrote examples_out/06_veega_jupiter.png")
