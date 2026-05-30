"""Poincare surfaces of section and tube cuts (MASTER_PLAN.md §3.10).

A manifold tube, cut by a surface of section x = const, traces a closed curve in
the (y, vy) plane. Heteroclinic connections appear where two such cuts (at the
same Jacobi constant) intersect.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.cr3bp import eom
from ..manifolds.manifold import manifold_seeds

_INT = dict(method="DOP853", rtol=1e-12, atol=1e-12)


def _section_event(x_sec: float, axis: int = 0):
    """Event function for the plane axis_value = x_sec. Default axis=0 ⇒ x = x_sec.

    axis indexes the position component: 0 → x, 1 → y, 2 → z. Used as the section
    normal direction for generalised Poincaré cuts (NRHO needs y = 0, halos use
    x = 1 − μ, etc.).
    """
    def ev(t, s, mu):
        return s[axis] - x_sec
    ev.terminal = False
    ev.direction = 0.0
    return ev


def first_section_crossing(mu: float, seed, x_sec: float, stable: bool,
                           t_max: float = 10.0, require_vx_positive: bool = True,
                           axis: int = 0):
    """First crossing of the plane axis_value = x_sec by a manifold trajectory.

    Forward integration for unstable seeds, backward for stable seeds. Returns the
    crossing state (6,) or None. By default keeps only crossings with vx > 0 (the
    branch consistent with an L1->secondary->L2 connection on the x-section); set
    require_vx_positive=False for sections (like y=0) where another sign convention
    selects the branch.
    """
    tf = -t_max if stable else t_max
    ev = _section_event(x_sec, axis=axis)
    sol = solve_ivp(eom, (0.0, tf), np.asarray(seed, float), args=(mu,),
                    events=ev, **_INT)
    tev, yev = sol.t_events[0], sol.y_events[0]
    for k in range(len(tev)):
        st = yev[k]
        if (not require_vx_positive) or st[3] > 0.0:
            return st
    return None


def propagate_until_section(mu: float, seed, x_sec: float, stable: bool = False,
                            t_max: float = 10.0):
    """Propagate a manifold seed until the FIRST crossing of x = x_sec (terminal).

    Returns (t, Y) covering only the tube segment from the orbit to the section,
    i.e. before any deep close approach beyond it. (t_events empty if not reached.)
    """
    tf = -t_max if stable else t_max

    def ev(t, s, mu):
        return s[0] - x_sec
    ev.terminal = True
    ev.direction = 0.0

    sol = solve_ivp(eom, (0.0, tf), np.asarray(seed, float), args=(mu,),
                    events=ev, **_INT)
    reached = sol.t_events[0].size > 0
    return sol.t, sol.y, reached


def tube_section_cut(mu: float, orbit, x_sec: float, stable: bool = False,
                     branch: int = +1, n_seeds: int = 120,
                     displacement: float = 1e-4, t_max: float = 10.0):
    """Cut a manifold tube with the plane x = x_sec.

    Returns dict with:
      yv      : (k,2) array of (y, vy) crossing coordinates (the tube cut),
      states  : (k,6) full crossing states,
      seed_idx: indices of the seeds that crossed,
      lambda_u: the unstable multiplier.
    """
    seeds, lam = manifold_seeds(mu, orbit, n_seeds=n_seeds,
                                displacement=displacement, stable=stable,
                                branch=branch)
    yv, states, idx = [], [], []
    for i, seed in enumerate(seeds):
        st = first_section_crossing(mu, seed, x_sec, stable, t_max=t_max)
        if st is not None:
            yv.append([st[1], st[4]])
            states.append(st)
            idx.append(i)
    return {"yv": np.array(yv), "states": np.array(states),
            "seed_idx": np.array(idx), "lambda_u": lam}
