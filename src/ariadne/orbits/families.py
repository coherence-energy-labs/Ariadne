"""Orbit-family continuation (MASTER_PLAN.md §3.8).

Natural-parameter continuation of the planar Lyapunov family: step the x-amplitude
(initial x0) away from the libration point and re-correct vy0 at each step. Tracks
the Jacobi constant, period, and stability indices, and locates the halo
bifurcation (where the vertical stability index passes through +1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .differential_correction import (
    PeriodicOrbit, correct_lyapunov, monodromy, stability_indices,
)
from .lagrange import lagrange_points
from .linear import linear_lyapunov_guess


@dataclass
class FamilyMember:
    amplitude: float        # x-distance of x0 from the libration point
    orbit: PeriodicOrbit
    nu_vertical: float
    nu_planar: float


def lyapunov_family(mu: float, point: str = "L1",
                    amplitude0: float = 1e-3, dx: float = 2e-3,
                    n: int = 40) -> list[FamilyMember]:
    """Generate a planar Lyapunov family by natural-parameter continuation in x0."""
    xL = lagrange_points(mu)[point][0]
    members: list[FamilyMember] = []

    s0, Tg = linear_lyapunov_guess(mu, point, amplitude0)
    orb = correct_lyapunov(mu, s0, Tg)
    orb.point = point

    for _ in range(n):
        M = monodromy(mu, orb)
        si = stability_indices(M)
        members.append(FamilyMember(
            amplitude=abs(orb.s0[0] - xL), orbit=orb,
            nu_vertical=si["nu_vertical"], nu_planar=si["nu_planar"],
        ))
        # step amplitude outward (x0 moves further from xL) and re-correct
        guess = orb.s0.copy()
        guess[0] = orb.s0[0] - dx
        try:
            nxt = correct_lyapunov(mu, guess, orb.period)
        except RuntimeError:
            break
        nxt.point = point
        orb = nxt

    return members


def find_halo_bifurcation(members: list[FamilyMember]) -> dict | None:
    """Locate where the vertical stability index crosses +1 (halo bifurcation).

    Returns interpolated amplitude and Jacobi constant, or None if not bracketed.
    """
    for a, b in zip(members[:-1], members[1:]):
        if (a.nu_vertical - 1.0) * (b.nu_vertical - 1.0) <= 0.0 and a.nu_vertical != b.nu_vertical:
            t = (1.0 - a.nu_vertical) / (b.nu_vertical - a.nu_vertical)
            amp = a.amplitude + t * (b.amplitude - a.amplitude)
            jac = a.orbit.jacobi + t * (b.orbit.jacobi - a.orbit.jacobi)
            return {"amplitude": float(amp), "jacobi": float(jac)}
    return None
