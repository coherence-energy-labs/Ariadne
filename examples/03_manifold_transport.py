"""Cislunar transport graph: the natural Earth-Moon highways.

Each L1/L2 libration orbit has stable+unstable manifold "tubes" -- 4D surfaces in phase space
that carry trajectories ballistically toward (stable) or away from (unstable) the orbit.
Cutting two tubes by a common Poincare section (x = 1-mu) yields curves whose crossings are
exact heteroclinic-class connections: a spacecraft transitions from orbit A's unstable tube to
orbit B's stable tube with a Delta-v equal to ||vx_A - vx_B|| (energy-determined, machine-precise).

Ariadne builds this transport graph for the Earth-Moon L1+L2 family + the Gateway NRHO.

Run:  PYTHONPATH=src python examples/03_manifold_transport.py
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

from ariadne.data.constants import EARTH_MOON
from ariadne.orbits.halo import halo_family
from ariadne.orbits.nrho import nrho_family
from ariadne.connections.poincare_3d import tube_section_cut_3d, closest_approach_4d
from ariadne.dynamics.cr3bp import pseudo_potential

MU = EARTH_MOON.mu
L_STAR = EARTH_MOON.L_star
V_STAR = EARTH_MOON.V_star
T_DAYS = EARTH_MOON.T_star / 86400.0

# Build orbits
print("Building L1 halo, L2 halo, and Gateway NRHO...")
l1 = halo_family(MU, point="L1", n=10, dz=4e-3, fam_n=40, lyap_amp0=2e-3, lyap_dx=4e-3)[5]
l2 = halo_family(MU, point="L2", n=10, dz=4e-3, fam_n=40, lyap_amp0=2e-3, lyap_dx=4e-3)[5]
nrho, _ = nrho_family(MU, "L2", t_star_days=T_DAYS, l_star=L_STAR,
                      target_period_d=6.56, ds=4e-3)
print(f"  L1 halo C={l1.jacobi:.4f}, L2 halo C={l2.jacobi:.4f}, NRHO C={nrho.jacobi:.4f}")

# ----- L1<->L2 halo via x=1-mu Poincare section -----
print("\nL1 halo <-> L2 halo heteroclinic on x = 1-mu (axis=0):")
t0 = time.time()
l1_u = tube_section_cut_3d(MU, l1, x_sec=1 - MU, stable=False, branch=+1,
                            n_seeds=200, t_max=12.0, axis=0)
l2_s = tube_section_cut_3d(MU, l2, x_sec=1 - MU, stable=True, branch=+1,
                            n_seeds=200, t_max=12.0, axis=0)
r = closest_approach_4d(l1_u["yzvyvz"], l2_s["yzvyvz"])
pa, pb = r["point_a"], r["point_b"]
y_int, z_int = 0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])
vy_a, vz_a = float(pa[2]), float(pa[3])
vy_b, vz_b = float(pb[2]), float(pb[3])
om = pseudo_potential([1 - MU, y_int, z_int, 0.0, 0.0, 0.0], MU)
import math
vx_a = math.sqrt(2 * om - l1.jacobi - vy_a ** 2 - vz_a ** 2)
vx_b = math.sqrt(2 * om - l2.jacobi - vy_b ** 2 - vz_b ** 2)
dv12 = math.sqrt((vx_a - vx_b) ** 2 + (vy_a - vy_b) ** 2 + (vz_a - vz_b) ** 2) * V_STAR * 1000
print(f"  patch Delta-v = {dv12:.1f} m/s   ({time.time()-t0:.0f}s)")

# ----- NRHO <-> L2 halo via y=0 Poincare section -----
print("\nNRHO <-> L2 halo heteroclinic on y = 0 (axis=1):")
t0 = time.time()
nu = tube_section_cut_3d(MU, nrho, x_sec=0.0, stable=False, branch=-1,
                         n_seeds=160, t_max=12.0, axis=1)
ls = tube_section_cut_3d(MU, l2, x_sec=0.0, stable=True, branch=+1,
                         n_seeds=160, t_max=12.0, axis=1)
r = closest_approach_4d(nu["yzvyvz"], ls["yzvyvz"])
pa, pb = r["point_a"], r["point_b"]
x_int, z_int = 0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])
vx_a, vz_a = float(pa[2]), float(pa[3])
vx_b, vz_b = float(pb[2]), float(pb[3])
om = pseudo_potential([x_int, 0.0, z_int, 0.0, 0.0, 0.0], MU)
vy_a = math.sqrt(2 * om - nrho.jacobi - vx_a ** 2 - vz_a ** 2)
vy_b = math.sqrt(2 * om - l2.jacobi - vx_b ** 2 - vz_b ** 2)
dv_nrho = math.sqrt((vx_a - vx_b) ** 2 + (vy_a - vy_b) ** 2 + (vz_a - vz_b) ** 2) * V_STAR * 1000
print(f"  patch Delta-v = {dv_nrho:.1f} m/s   ({time.time()-t0:.0f}s)")
print(f"  crossing at (x={x_int:.4f}, y=0, z={z_int:.4f}, ~lunar vicinity)")

print(f"\nCislunar transport graph summary:")
print(f"  L1 halo (C={l1.jacobi:.4f}) --[{dv12:5.1f} m/s, x=1-mu section]--> L2 halo (C={l2.jacobi:.4f})")
print(f"  NRHO    (C={nrho.jacobi:.4f}) --[{dv_nrho:5.1f} m/s, y=0    section]--> L2 halo (C={l2.jacobi:.4f})")

# Visualise the manifold tubes
from ariadne.connections.poincare import propagate_until_section
from ariadne.manifolds.manifold import manifold_seeds
fig, ax = plt.subplots(figsize=(9, 7))
for name, orbit, color in [("L1 halo", l1, 'forestgreen'),
                           ("L2 halo", l2, 'crimson'),
                           ("NRHO",    nrho, 'royalblue')]:
    from ariadne.dynamics.cr3bp import propagate
    period = orbit.period
    sol = propagate(orbit.s0, (0.0, period), MU, t_eval=np.linspace(0.0, period, 200))
    ax.plot(sol.y[0] * L_STAR, sol.y[1] * L_STAR, color=color, lw=2.0, label=name)
ax.scatter([(1 - MU) * L_STAR], [0], s=120, c='gray', label='Moon')
ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
ax.set_title(f"Cislunar libration orbits: L1/L2 halos + Gateway NRHO\n"
             f"L1<->L2 patch {dv12:.0f} m/s,  NRHO<->L2 patch {dv_nrho:.0f} m/s")
ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("examples_out/03_manifold_transport.png", dpi=120)
print("\nWrote examples_out/03_manifold_transport.png")
