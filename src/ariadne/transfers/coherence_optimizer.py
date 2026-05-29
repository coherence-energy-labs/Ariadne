"""Coherence-weighted trajectory optimizer (MASTER_PLAN.md — Stage 12).

Implements the coherence-guided objective: instead of optimizing Delta-v alone, trade
Delta-v against ROBUSTNESS (the Stage-11 coherence metric = endpoint sensitivity to
injection error). This finds smoother, lower-correction, lower-risk routes -- the
"coherence-guided Earth->Moon optimizer."

    J(transfer) = z(Delta-v) + w_robust * z(endpoint_sensitivity)

(the |grad(1/tau_c)| term is dropped: at Earth-Moon scale tau_c collapses to the
Newtonian potential, so it adds no information beyond the tidal field already present;
the robustness term IS the coherence contribution.)

Outputs the Delta-v-vs-coherence Pareto front and its knee (the best compromise:
maximum robustness gain per unit Delta-v). Standard gravity throughout (eta=1 firewall).
"""
from __future__ import annotations

import numpy as np

from ..data.constants import R_MOON
from ..data.ephemeris import et
from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..transfers.ephemeris_transfer import design_transfer
from ..transfers.wsb import _capture_state, _frame, SOLUTION_PARAMS, evaluate_transfer
from ..analysis.coherence import endpoint_sensitivity


def coherence_frontier(epoch="2025-06-01T00:00:00", tof_grid=(3.0, 4.0, 5.0, 6.0),
                       leo_alt=200.0, llo_alt=100.0, include_wsb=True):
    """Compute (Delta-v, endpoint-sensitivity) for the transfer family. Returns list of dicts."""
    e0 = et(epoch)
    pts = []
    for tof in tof_grid:
        d = design_transfer(e0, float(tof), leo_alt, llo_alt)
        if d is None:
            continue
        s0 = np.concatenate([d["r1"], d["v1"]]); T = tof * 86400.0
        prop = lambda s, T=T: propagate_test_particle(
            s[:3], s[3:], e0, (0, T), perturbers=("SUN", "MOON")).y[:, -1]
        pts.append({"label": f"direct {tof:.0f}d", "dv_ms": d["total_ms"],
                    "sensitivity": endpoint_sensitivity(prop, s0), "tof_days": tof})
    if include_wsb:
        ec = et("2025-11-12T00:00:00"); fr = _frame(ec)
        fa, al, be, ph = SOLUTION_PARAMS
        pos, vel, _ = _capture_state(ec, fa, al, be, ph, R_MOON + llo_alt, *fr)
        s0w = np.concatenate([pos, vel]); Tw = 48.8 * 86400.0
        propw = lambda s: propagate_test_particle(
            s[:3], s[3:], ec, (0, -Tw), perturbers=("SUN", "MOON")).y[:, -1]
        bw = evaluate_transfer(SOLUTION_PARAMS)
        pts.append({"label": "WSB 49d", "dv_ms": bw["total_ms"],
                    "sensitivity": endpoint_sensitivity(propw, s0w), "tof_days": 48.8})
    return pts


def pareto_front(points):
    """Non-dominated set minimizing BOTH Delta-v and sensitivity. Sorted by Delta-v."""
    front = []
    for p in points:
        dominated = any(
            q is not p and q["dv_ms"] <= p["dv_ms"] and q["sensitivity"] <= p["sensitivity"]
            and (q["dv_ms"] < p["dv_ms"] or q["sensitivity"] < p["sensitivity"])
            for q in points)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["dv_ms"])


def knee(front):
    """The Pareto knee: point nearest the utopia corner after [0,1] normalization
    (the best Delta-v-vs-robustness compromise)."""
    if len(front) < 3:
        return front[0] if front else None
    dv = np.array([p["dv_ms"] for p in front], float)
    s = np.array([p["sensitivity"] for p in front], float)
    dvn = (dv - dv.min()) / (np.ptp(dv) + 1e-30)
    sn = (s - s.min()) / (np.ptp(s) + 1e-30)
    return front[int(np.argmin(np.hypot(dvn, sn)))]


def weighted_optimum(points, w_robust):
    """Pick the transfer minimizing z(Delta-v) + w_robust * z(sensitivity)."""
    dv = np.array([p["dv_ms"] for p in points], float)
    s = np.array([p["sensitivity"] for p in points], float)
    dvn = (dv - dv.min()) / (np.ptp(dv) + 1e-30)
    sn = (s - s.min()) / (np.ptp(s) + 1e-30)
    return points[int(np.argmin(dvn + w_robust * sn))]
