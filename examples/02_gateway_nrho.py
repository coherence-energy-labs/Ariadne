"""Construct NASA's Gateway-class Near-Rectilinear Halo Orbit + verify its near-stability.

The Gateway lunar space station orbits the Moon in a 9:2 resonant Near-Rectilinear Halo Orbit
(NRHO) at L2 -- period ~6.56 days, perilune ~3,200 km (low pass over the south pole), apolune
~70,000 km. The NRHO is dynamically chosen because its Floquet multiplier is tiny (~2) compared
to a deep libration orbit's (>>1000), meaning station-keeping is cheap.

Ariadne builds the orbit from scratch by pseudo-arclength continuation along the L2 halo family,
then validates the geometry + stability against the Gateway specification.

Run:  PYTHONPATH=src python examples/02_gateway_nrho.py
"""
import numpy as np
import matplotlib.pyplot as plt

from ariadne.data.constants import EARTH_MOON, R_MOON
from ariadne.orbits.nrho import nrho_family
from ariadne.orbits.families import lyapunov_orbit_at_jacobi
from ariadne.orbits.differential_correction import monodromy
from ariadne.dynamics.cr3bp import propagate

MU = EARTH_MOON.mu
L_STAR = EARTH_MOON.L_star
T_DAYS = EARTH_MOON.T_star / 86400.0

# Pseudo-arclength continuation: walk along the L2 halo family to Gateway's ~6.56 d period
print("Building Gateway-class NRHO via pseudo-arclength continuation of L2 halo family...")
nrho, family = nrho_family(MU, "L2", t_star_days=T_DAYS, l_star=L_STAR,
                           target_period_d=6.56, ds=4e-3)
print(f"  family: {len(family)} members, period {family[0].period*T_DAYS:.2f}d -> "
      f"{family[-1].period*T_DAYS:.2f}d")

# Sample the NRHO + its perilune/apolune
sol = propagate(nrho.s0, (0.0, nrho.period), MU, t_eval=np.linspace(0.0, nrho.period, 800))
d_moon_km = np.sqrt((sol.y[0] - (1 - MU)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * L_STAR
peri_km, apo_km = float(d_moon_km.min()), float(d_moon_km.max())

# Near-stability: NRHO Floquet vs a deep L1 Lyapunov for scale
floq_nrho = float(np.max(np.abs(np.linalg.eigvals(monodromy(MU, nrho)))))
floq_lyap = float(np.max(np.abs(np.linalg.eigvals(
    monodromy(MU, lyapunov_orbit_at_jacobi(MU, "L1", 3.16))))))

print(f"\nNRHO geometry:")
print(f"  period   = {nrho.period * T_DAYS:.3f} d   (Gateway spec: ~6.56)")
print(f"  perilune = {peri_km:.0f} km   (alt {peri_km - R_MOON:.0f} km over the pole)")
print(f"  apolune  = {apo_km:.0f} km   (Gateway spec: ~70,000)")
print(f"  periodic to residual {nrho.residual:.1e}")
print(f"\nNear-stability:")
print(f"  NRHO Floquet multiplier         = {floq_nrho:.2f}")
print(f"  L1 Lyapunov (C=3.16) for scale  = {floq_lyap:.0f}")
print(f"  NRHO is {floq_lyap / floq_nrho:.0f}x more stable (why Gateway flies an NRHO)")

fig = plt.figure(figsize=(11, 4.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol.y[0] * L_STAR, sol.y[1] * L_STAR, sol.y[2] * L_STAR, lw=1.0, color='crimson')
ax1.scatter([(1 - MU) * L_STAR], [0], [0], s=100, c='gray', label='Moon')
ax1.set_xlabel("x [km]"); ax1.set_ylabel("y [km]"); ax1.set_zlabel("z [km]")
ax1.set_title("Gateway NRHO (3D, synodic frame)")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol.t * T_DAYS, d_moon_km, color='crimson', lw=1.2)
ax2.axhline(R_MOON, color='gray', ls='--', label=f"Moon surface ({R_MOON} km)")
ax2.set_xlabel("time [d]"); ax2.set_ylabel("distance to Moon [km]")
ax2.set_title(f"Moon distance: peri {peri_km:.0f} km / apo {apo_km:.0f} km")
ax2.grid(alpha=0.3); ax2.legend()
plt.tight_layout()
plt.savefig("examples_out/02_gateway_nrho.png", dpi=120)
print("\nWrote examples_out/02_gateway_nrho.png")
