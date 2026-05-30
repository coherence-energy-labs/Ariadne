"""Earth-Mars Lambert porkchop -- launch energy as a function of (epoch, time-of-flight).

A "porkchop plot" shows the launch C3 (or Δv) over a 2D grid of departure-epoch and
time-of-flight pairs. The contours identify the cheapest launch windows. Ariadne computes
this from the Lambert universal-variable solver on the real DE440 ephemeris.

Run:  PYTHONPATH=src python examples/07_lambert_porkchop.py
"""
import warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

from ariadne.data.constants import GM_SUN
from ariadne.data.ephemeris import et, body_state, utc
from ariadne.optimize.lambert import lambert

DAY = 86400.0
LAUNCH_START = "2026-01-01T00:00:00"
DEP_DAYS = np.linspace(0, 540, 28)     # ~1.5 yr launch window
TOF_DAYS = np.linspace(120, 360, 26)   # 120 to 360 day transfers

e0 = et(LAUNCH_START)
print(f"Earth -> Mars Lambert porkchop")
print(f"  launch window:  {utc(e0)[:10]}  -> {utc(e0 + DEP_DAYS[-1]*DAY)[:10]}")
print(f"  TOF range:      {TOF_DAYS[0]:.0f} -> {TOF_DAYS[-1]:.0f} days")
print(f"  grid:           {len(DEP_DAYS)} x {len(TOF_DAYS)} = {len(DEP_DAYS)*len(TOF_DAYS)} Lambert solves")

C3_grid = np.full((len(TOF_DAYS), len(DEP_DAYS)), np.nan)
for j, dep in enumerate(DEP_DAYS):
    s_e = body_state("EARTH", e0 + dep * DAY, "J2000", "SUN")
    for i, tof in enumerate(TOF_DAYS):
        s_m = body_state("MARS BARYCENTER", e0 + (dep + tof) * DAY, "J2000", "SUN")
        try:
            v1, _ = lambert(s_e[:3], s_m[:3], tof * DAY, GM_SUN)
            vinf = v1 - s_e[3:]
            C3_grid[i, j] = float(vinf @ vinf)
        except Exception:
            pass

# Min finder
i_min, j_min = np.unravel_index(np.nanargmin(C3_grid), C3_grid.shape)
c3_min = C3_grid[i_min, j_min]
dep_best = DEP_DAYS[j_min]; tof_best = TOF_DAYS[i_min]
print(f"\nMinimum C3 in this grid:")
print(f"  C3 = {c3_min:.2f} km^2/s^2   (v_inf = {math.sqrt(c3_min):.2f} km/s)")
print(f"  depart {utc(e0 + dep_best*DAY)[:10]}   TOF = {tof_best:.0f} days")
print(f"  arrive {utc(e0 + (dep_best + tof_best)*DAY)[:10]}")
print(f"  Reference: Earth->Mars Hohmann C3 ~ 8.9; real Mars launch windows hit ~10-25")

# Plot porkchop
fig, ax = plt.subplots(figsize=(11, 6.5))
levels = [10, 12, 15, 20, 25, 30, 40, 50, 75]
cf = ax.contourf(DEP_DAYS, TOF_DAYS, np.clip(C3_grid, 0, 75), levels=20, cmap='viridis')
cs = ax.contour(DEP_DAYS, TOF_DAYS, np.clip(C3_grid, 0, 75), levels=levels,
                colors='white', linewidths=0.6)
ax.clabel(cs, fmt='%d', fontsize=8)
ax.scatter([dep_best], [tof_best], s=200, marker='*',
           edgecolor='red', facecolor='yellow', linewidth=2, zorder=5,
           label=f"min C3 = {c3_min:.1f}")
ax.set_xlabel(f"days after {LAUNCH_START[:10]}"); ax.set_ylabel("time of flight (days)")
ax.set_title(f"Earth -> Mars Lambert porkchop (DE440 ephemeris)\n"
             f"depart {utc(e0 + dep_best*DAY)[:10]} + {tof_best:.0f} d TOF -> "
             f"C3 = {c3_min:.1f} km²/s² (v_inf {math.sqrt(c3_min):.2f} km/s)")
plt.colorbar(cf, ax=ax, label='launch C3 (km²/s²)')
ax.legend(loc='upper right')
plt.tight_layout(); plt.savefig("examples_out/07_lambert_porkchop.png", dpi=120)
print("\nWrote examples_out/07_lambert_porkchop.png")
