"""Build the Earth-Moon L1 Lyapunov orbit family in 30 lines.

Periodic Lyapunov orbits are the simplest periodic orbits around the L1/L2 libration points
(the saddle points of the CR3BP). Their family is parameterised by amplitude (or equivalently
Jacobi constant). The unstable/stable manifolds of these orbits are the natural transport
"highways" between Earth and Moon -- the foundation of low-energy mission design.

Run:  PYTHONPATH=src python examples/01_lyapunov_family.py
"""
import numpy as np
import matplotlib.pyplot as plt

from ariadne.data.constants import EARTH_MOON
from ariadne.orbits.families import lyapunov_family
from ariadne.dynamics.cr3bp import propagate

MU = EARTH_MOON.mu
L_STAR = EARTH_MOON.L_star

# Generate 30 Lyapunov orbits at L1 by amplitude continuation
family = lyapunov_family(MU, point="L1", amplitude0=1e-3, dx=2e-3, n=30)
print(f"L1 Lyapunov family: {len(family)} orbits")
print(f"Jacobi range: {family[0].orbit.jacobi:.3f} -> {family[-1].orbit.jacobi:.3f}")

# Plot every other orbit, colored by Jacobi constant
fig, ax = plt.subplots(figsize=(9, 7))
cmap = plt.cm.viridis
norm = plt.Normalize(family[-1].orbit.jacobi, family[0].orbit.jacobi)
for m in family[::2]:
    sol = propagate(m.orbit.s0, (0.0, m.orbit.period), MU,
                    t_eval=np.linspace(0.0, m.orbit.period, 200))
    ax.plot(sol.y[0] * L_STAR, sol.y[1] * L_STAR, color=cmap(norm(m.orbit.jacobi)), lw=0.7)

ax.scatter([(1 - MU) * L_STAR], [0], s=80, c='gray', label='Moon')
ax.scatter([-MU * L_STAR], [0], s=180, c='royalblue', label='Earth')
ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
ax.set_title(f"Earth-Moon L1 Lyapunov family ({len(family)} orbits, amplitude-continued)")
ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.3)
plt.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, label="Jacobi constant")
plt.tight_layout()
plt.savefig("examples_out/01_lyapunov_family.png", dpi=120)
print("\nWrote examples_out/01_lyapunov_family.png")
