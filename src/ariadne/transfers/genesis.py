"""Genesis-type Sun-Earth halo transport (MASTER_PLAN.md Stage 7, G9).

The Genesis mission flew a Sun-Earth L1 halo orbit and used its invariant manifolds
as a free "interplanetary superhighway" between the halo and Earth's vicinity. This
module builds a real Sun-Earth L1 halo (period ~178 days, matching SOHO/Genesis) and
measures how close its stable/unstable manifolds carry a spacecraft to Earth -- i.e.
it demonstrates the manifold connection that Genesis exploited.
"""
from __future__ import annotations

import numpy as np

from ..data.constants import SUN_EARTH
from ..orbits.halo import halo_family
from ..orbits.lagrange import lagrange_points
from ..manifolds.manifold import manifold_seeds, manifold_trajectory


def genesis_halo(system=SUN_EARTH, index: int = 6):
    """A representative Sun-Earth L1 halo (period ~178 d)."""
    halos = halo_family(system.mu, "L1", n=10, dz=1.5e-4, fam_n=120,
                        lyap_amp0=1e-5, lyap_dx=4e-5)
    return halos[min(index, len(halos) - 1)], halos


def earth_approach(halo, system=SUN_EARTH, n_seeds: int = 120,
                   t_max: float = 6.0, displacement: float = 1e-6):
    """Closest approach to Earth over the halo's stable and unstable manifolds.

    Returns a dict with the minimum Earth distance (km), the L1 distance (km), and
    which manifold/branch achieved it.
    """
    mu, L = system.mu, system.L_star
    xL = lagrange_points(mu)["L1"][0]
    l1_earth_km = (1.0 - mu - xL) * L

    best = {"earth_km": np.inf, "stable": None, "branch": None}
    for stable in (True, False):
        for br in (+1, -1):
            seeds, _ = manifold_seeds(mu, halo, n_seeds=n_seeds,
                                      displacement=displacement, stable=stable,
                                      branch=br)
            for s in seeds:
                _, Y = manifold_trajectory(mu, s, stable=stable, t_max=t_max, n=1500)
                d = np.sqrt((Y[0] - (1.0 - mu)) ** 2 + Y[1] ** 2 + Y[2] ** 2).min() * L
                if d < best["earth_km"]:
                    best = {"earth_km": float(d), "stable": stable, "branch": br}
    best["l1_earth_km"] = float(l1_earth_km)
    best["fraction_of_l1"] = best["earth_km"] / l1_earth_km
    best["period_days"] = halo.period * system.T_star / 86400.0
    best["z_amplitude_km"] = halo.z_amplitude * L
    return best
