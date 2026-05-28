"""Heteroclinic connections between libration-point orbits (MASTER_PLAN.md §3.10).

At a fixed Jacobi constant, the unstable tube of one orbit and the stable tube of
another are cut by a common surface of section. Where the two cuts intersect in
the (y, vy) plane there is a near-ballistic heteroclinic connection: a trajectory
that departs the first orbit and asymptotically arrives at the second. This is the
mathematical form of "leave orbit A on tube X, coast, arrive at orbit B."
"""
from __future__ import annotations

import numpy as np

from ..orbits.families import lyapunov_orbit_at_jacobi
from .poincare import tube_section_cut


def _segment_intersection(p1, p2, p3, p4):
    """Intersection point of segments p1p2 and p3p4, or None."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-15:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def loop_intersections(curve_a: np.ndarray, curve_b: np.ndarray) -> list[tuple]:
    """All intersection points between two polylines of (y, vy) points."""
    pts = []
    for i in range(len(curve_a) - 1):
        for j in range(len(curve_b) - 1):
            p = _segment_intersection(curve_a[i], curve_a[i + 1],
                                      curve_b[j], curve_b[j + 1])
            if p is not None:
                pts.append(p)
    return pts


def find_heteroclinic(mu: float, c_target: float,
                      source: str = "L1", target: str = "L2",
                      x_sec: float | None = None, n_seeds: int = 160,
                      displacement: float = 1e-4) -> dict | None:
    """Find an L1<->L2 (or general) Lyapunov heteroclinic connection at Jacobi c_target.

    Cuts the source orbit's unstable tube and the target orbit's stable tube on the
    plane x = x_sec (default: through the secondary, x = 1-mu), tries both branches,
    and returns the first intersection found.

    Returns a dict with the connection coordinates and the two orbits, or None.
    """
    if x_sec is None:
        x_sec = 1.0 - mu

    o_src = lyapunov_orbit_at_jacobi(mu, source, c_target)
    o_tgt = lyapunov_orbit_at_jacobi(mu, target, c_target)

    best = None
    for b_u in (+1, -1):
        cut_u = tube_section_cut(mu, o_src, x_sec, stable=False, branch=b_u,
                                 n_seeds=n_seeds, displacement=displacement)
        if len(cut_u["yv"]) < 3:
            continue
        for b_s in (+1, -1):
            cut_s = tube_section_cut(mu, o_tgt, x_sec, stable=True, branch=b_s,
                                     n_seeds=n_seeds, displacement=displacement)
            if len(cut_s["yv"]) < 3:
                continue
            pts = loop_intersections(cut_u["yv"], cut_s["yv"])
            if pts:
                best = {
                    "jacobi": c_target, "source": source, "target": target,
                    "x_section": x_sec, "branch_unstable": b_u, "branch_stable": b_s,
                    "intersections": pts, "connection_yv": pts[0],
                    "unstable_cut": cut_u["yv"], "stable_cut": cut_s["yv"],
                    "orbit_source": o_src, "orbit_target": o_tgt,
                }
                return best
    return best
