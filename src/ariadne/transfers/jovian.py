"""Galilean-moon transport (MASTER_PLAN.md — Stage 13, discovery-engine generalization).

Points the Ariadne CR3BP/manifold engine at the Jupiter moon systems (Io, Europa,
Ganymede, Callisto) -- the home of multi-moon "Petit Grand Tour" low-energy routes
(Koon-Lo-Marsden-Ross). Demonstrates the engine GENERALIZES to a new system with no
code changes (only mass/length constants), validates each moon's libration structure,
and computes a moon-to-moon transfer Delta-v table (the baseline a manifold/gravity-assist
tour reduces).

Honest scope: the Galilean tour concept is known (KLMR); this shows our engine reproduces
the structure in a new system and sets up route discovery -- not a new route in itself.
"""
from __future__ import annotations

import math

import numpy as np

from ..data.constants import GALILEAN, GM_JUPITER
from ..orbits.lagrange import lagrange_points
from ..orbits.linear import collinear_linear_modes, linear_lyapunov_guess
from ..orbits.differential_correction import correct_lyapunov
from ..manifolds.manifold import manifold_seeds, manifold_trajectory


def moon_libration(system) -> dict:
    """Lagrange points + a small L1 Lyapunov orbit for a Jupiter-moon system."""
    mu = system.mu
    L = lagrange_points(mu)
    modes = collinear_linear_modes(mu, "L1")
    # small amplitude appropriate to the tight (small-mu) system
    amp = 5e-5
    s0, Tg = linear_lyapunov_guess(mu, "L1", amp)
    orb = correct_lyapunov(mu, s0, Tg)
    orb.point = "L1"
    return {"system": system.name, "mu": mu,
            "L1_km": (1 - mu - L["L1"][0]) * system.L_star,
            "L2_km": (L["L2"][0] - (1 - mu)) * system.L_star,
            "lyap_period_d": orb.period * system.T_star / 86400.0,
            "omega": modes["omega"], "orbit": orb, "L_star": system.L_star}


def manifold_reach_km(system, orbit, t_max=8.0, n_seeds=60, displacement=1e-5):
    """Closest approach to Jupiter reached by the L1 stable/unstable manifold (km)."""
    mu, Lkm = system.mu, system.L_star
    best = math.inf
    for stable in (True, False):
        for br in (+1, -1):
            seeds, _ = manifold_seeds(mu, orbit, n_seeds=n_seeds,
                                      displacement=displacement, stable=stable, branch=br)
            for s in seeds:
                _, Y = manifold_trajectory(mu, s, stable=stable, t_max=t_max, n=600)
                # distance to Jupiter (at -mu)
                d = np.sqrt((Y[0] + mu) ** 2 + Y[1] ** 2 + Y[2] ** 2).min() * Lkm
                best = min(best, d)
    return best


def moon_tour_deltav():
    """Hohmann Delta-v (Jupiter-centric two-body) to step between adjacent moon orbits."""
    legs = []
    for a, b in zip(GALILEAN[:-1], GALILEAN[1:]):
        r1, r2 = a.L_star, b.L_star
        at = 0.5 * (r1 + r2)
        v1 = math.sqrt(GM_JUPITER / r1)
        vp = math.sqrt(GM_JUPITER * (2 / r1 - 1 / at))
        va = math.sqrt(GM_JUPITER * (2 / r2 - 1 / at))
        v2 = math.sqrt(GM_JUPITER / r2)
        dv = (vp - v1) + (v2 - va)
        legs.append({"from": a.secondary, "to": b.secondary, "dv_ms": dv * 1000.0})
    return legs
