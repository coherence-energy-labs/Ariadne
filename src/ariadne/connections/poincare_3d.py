"""3D-orbit Poincare cuts -- the extension of poincare.tube_section_cut to z/vz-active orbits.

Planar Lyapunov tubes cut by x = x_sec produce a closed curve in (y, vy). 3D halos / NRHOs
(z, vz active) cut by the same plane produce a closed curve in the 4-space (y, z, vy, vz):
the tube still has only one orbit-phase parameter, but each crossing carries two more coords.
Two such 1-curves in 4-space generically do not intersect exactly; their closest 4D approach
is the natural patch point for a heteroclinic-class connection, with the closest-approach
position gap exposed as a discretisation-quality metric on the edge.

References:
- Goemoery & Howell 2010, "Halo orbit transfer via the invariant manifolds" (3D section method).
- Anderson & Lo 2015, "Spatial halo transfer via Poincare sections."
"""
from __future__ import annotations

import numpy as np

from ..manifolds.manifold import manifold_seeds
from .poincare import first_section_crossing


def tube_section_cut_3d(mu: float, orbit, x_sec: float, stable: bool = False,
                        branch: int = +1, n_seeds: int = 200,
                        displacement: float = 1e-4, t_max: float = 12.0,
                        axis: int = 0):
    """Cut a 3D-orbit manifold tube with the plane axis_value = x_sec.

    Default axis=0 (x = x_sec) reproduces the existing x-section behaviour. For NRHO
    transport, axis=1 (y = x_sec, typically x_sec=0) is the right choice -- NRHO crosses
    y=0 every period while its tube barely diverges from x=x_NRHO. Each crossing keeps the
    4-tuple of (state components NOT on the section's axis OR matched to it). For axis=0:
    (y, z, vy, vz). For axis=1: (x, z, vx, vz). For axis=2: (x, y, vx, vy).
    """
    seeds, lam = manifold_seeds(mu, orbit, n_seeds=n_seeds,
                                displacement=displacement, stable=stable, branch=branch)
    # which state components survive the section (the two positions + their conjugate vels)
    pos_idxs = [j for j in (0, 1, 2) if j != axis]
    vel_idxs = [j + 3 for j in pos_idxs]
    # which axis is the "energy-determined" velocity? matches the section axis.
    vel_axis = axis + 3
    crossings, states, idx = [], [], []
    for i, seed in enumerate(seeds):
        st = first_section_crossing(mu, seed, x_sec, stable, t_max=t_max,
                                    require_vx_positive=(axis == 0), axis=axis)
        if st is None and axis != 0:
            # for non-x sections, try the other sign convention too (vx negative branch)
            st = first_section_crossing(mu, seed, x_sec, stable, t_max=t_max,
                                        require_vx_positive=False, axis=axis)
        if st is not None:
            crossings.append([st[pos_idxs[0]], st[pos_idxs[1]],
                              st[vel_idxs[0]], st[vel_idxs[1]]])
            states.append(st)
            idx.append(i)
    return {"yzvyvz": np.array(crossings), "states": np.array(states),
            "seed_idx": np.array(idx), "lambda_u": float(lam), "axis": int(axis),
            "vel_axis": int(vel_axis)}


def _seg_seg_distance(p1, p2, q1, q2):
    """Closest-approach between two N-dim line segments. Returns (dist, s, t, point_p, point_q).

    `s, t` are fractional parameters on [0, 1] along the first/second segment. Standard
    Eberly / Sunday formulation, dimension-agnostic.
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = float(u @ u); b = float(u @ v); c = float(v @ v)
    d = float(u @ w); e = float(v @ w)
    denom = a * c - b * b
    if abs(denom) < 1e-15:
        s, t = 0.0, 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    pp = p1 + s * u
    qq = q1 + t * v
    return float(np.linalg.norm(pp - qq)), float(s), float(t), pp, qq


def closest_approach_4d(curveA: np.ndarray, curveB: np.ndarray):
    """Brute-force minimum 4D-distance segment-pair between two polylines.

    `curveA`, `curveB` shape (n, 4). Returns dict with gap_4d, idx_a, idx_b, s_a, t_b,
    point_a, point_b -- the 4D crossing-coordinate states at the closest-approach pair --
    or None on empty inputs. O(n_a * n_b) but n is small (~hundreds) and only invoked once
    per (src, dst) edge candidate.
    """
    if len(curveA) < 2 or len(curveB) < 2:
        return None
    best_d = np.inf
    best = None
    for i in range(len(curveA) - 1):
        a0, a1 = curveA[i], curveA[i + 1]
        for j in range(len(curveB) - 1):
            b0, b1 = curveB[j], curveB[j + 1]
            d, s, t, pa, pb = _seg_seg_distance(a0, a1, b0, b1)
            if d < best_d:
                best_d = d
                best = (i, j, s, t, pa, pb)
    if best is None:
        return None
    i, j, s, t, pa, pb = best
    return {"gap_4d": best_d, "idx_a": i, "idx_b": j, "s_a": s, "t_b": t,
            "point_a": pa, "point_b": pb}
