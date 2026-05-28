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
    """Generate a planar Lyapunov family by natural-parameter continuation in x0.

    Uses a tangent predictor for vy0 (initialized from linear theory, then refined
    by finite differences) so each corrector call starts inside its convergence
    basin. This is what makes the larger-period L2 family robust.
    """
    from .linear import collinear_linear_modes

    xL = lagrange_points(mu)[point][0]
    modes = collinear_linear_modes(mu, point)
    members: list[FamilyMember] = []

    def record(orb):
        si = stability_indices(monodromy(mu, orb))
        members.append(FamilyMember(amplitude=abs(orb.s0[0] - xL), orbit=orb,
                                    nu_vertical=si["nu_vertical"],
                                    nu_planar=si["nu_planar"]))

    s0, Tg = linear_lyapunov_guess(mu, point, amplitude0)
    orb = correct_lyapunov(mu, s0, Tg)
    orb.point = point
    record(orb)

    prev_x0, prev_vy0 = float(orb.s0[0]), float(orb.s0[4])
    slope = -modes["kappa"] * modes["omega"]   # d(vy0)/d(x0), linear estimate

    for _ in range(n - 1):
        x0_new = prev_x0 - dx
        vy0_pred = prev_vy0 + slope * (x0_new - prev_x0)
        guess = np.array([x0_new, 0.0, 0.0, 0.0, vy0_pred, 0.0])
        try:
            nxt = correct_lyapunov(mu, guess, orb.period)
        except RuntimeError:
            break
        nxt.point = point
        if nxt.s0[0] != prev_x0:
            slope = (nxt.s0[4] - prev_vy0) / (nxt.s0[0] - prev_x0)
        prev_x0, prev_vy0 = float(nxt.s0[0]), float(nxt.s0[4])
        orb = nxt
        record(orb)

    return members


def lyapunov_orbit_at_jacobi(mu: float, point: str, c_target: float,
                             n: int = 70, dx: float = 2e-3,
                             tol: float = 1e-9) -> PeriodicOrbit:
    """Find the planar Lyapunov orbit at a target Jacobi constant.

    Generates the family to bracket c_target, then secant-refines on x0 (warm
    started) until the orbit's Jacobi matches. Raises ValueError if c_target is
    outside the family's energy range.
    """
    fam = lyapunov_family(mu, point, amplitude0=1e-3, dx=dx, n=n)
    cs = [m.orbit.jacobi for m in fam]
    bracket = None
    for i in range(len(fam) - 1):
        if (cs[i] - c_target) * (cs[i + 1] - c_target) <= 0.0:
            bracket = (fam[i].orbit, fam[i + 1].orbit)
            break
    if bracket is None:
        raise ValueError(f"C={c_target} outside family range "
                         f"[{cs[-1]:.5f}, {cs[0]:.5f}] for {point}")

    o0, o1 = bracket
    xa, ca, vya = float(o0.s0[0]), o0.jacobi, float(o0.s0[4])
    xb, cb, vyb = float(o1.s0[0]), o1.jacobi, float(o1.s0[4])
    t_guess, orb = o0.period, o0
    for _ in range(40):
        if abs(cb - ca) < 1e-15:
            break
        xn = xb - (cb - c_target) * (xb - xa) / (cb - ca)
        # predict vy0 along the family line through the two bracketing points
        vy_pred = vya + (vyb - vya) * (xn - xa) / (xb - xa)
        guess = np.array([xn, 0.0, 0.0, 0.0, vy_pred, 0.0])
        orb = correct_lyapunov(mu, guess, t_guess)
        orb.point = point
        if abs(orb.jacobi - c_target) < tol:
            return orb
        xa, ca, vya = xb, cb, vyb
        xb, cb, vyb = xn, orb.jacobi, float(orb.s0[4])
        t_guess = orb.period
    return orb


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
