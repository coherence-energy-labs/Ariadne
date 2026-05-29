"""Route verification -- the burden of proof (MASTER_PLAN.md - Stage 15 / §7.4).

A mined route is just a sequence of graph edges until it is shown to be a real
trajectory. Two rungs of the fidelity ladder, applied honestly:

  1. CR3BP verification (`verify_route`): each patch must be a genuine section crossing.
     We check (a) CONTINUITY -- the pre-burn and post-burn states share the same position
     (you burn where you are); (b) ENERGY bookkeeping -- the pre-state's Jacobi equals the
     source orbit's and the post-state's equals the destination's, i.e. the burn supplies
     exactly the energy change; (c) the burn magnitude equals the edge's Delta-v.

  2. Perturbation survivability (`ephemeris_survivability`): propagate the connecting state
     in CR3BP and in the Sun-perturbed BCR4BP over a fixed horizon (same synodic frame, no
     coordinate conversion) and measure the divergence in km. A route survives into the
     real regime if that divergence is bounded/correctable rather than a chaotic escape.

HONEST scope: rung 1 is a rigorous CR3BP proof; rung 2 quantifies how much higher-fidelity
re-targeting a route would need. Driving that residual to zero in the full DE440 ephemeris
plus a GMAT cross-check is exactly what Stages 8-10 already did for the Earth->Moon TRANSFER
leg (to ~50 m / 149 m); a dedicated libration-to-libration ephemeris re-targeter is the
remaining tool and is noted as such (not claimed here).
"""
from __future__ import annotations

import numpy as np

from ..dynamics.cr3bp import jacobi_constant, propagate
from ..dynamics.bcr4bp import propagate_bcr4bp, sun_params


def _edge(graph, u, v):
    return next((e for e in graph.edges.get(u, []) if e.dst == v), None)


def verify_route(graph, path, tol_pos=1e-9, tol_jac=1e-6) -> dict:
    """CR3BP verification of every patch in a route. Returns per-leg residuals + verdict."""
    mu = graph.mu
    legs = []
    ok = True
    for u, v in zip(path[:-1], path[1:]):
        e = _edge(graph, u, v)
        if e is None or "pre" not in e.meta:
            ok = False
            legs.append({"leg": (u, v), "error": "no edge / no patch state"})
            continue
        pre = np.array(e.meta["pre"], float)
        post = np.array(e.meta["post"], float)
        pos_gap = float(np.linalg.norm(pre[:3] - post[:3]))
        c_pre = jacobi_constant(pre, mu)
        c_post = jacobi_constant(post, mu)
        dv_check = abs(abs(pre[3] - post[3]) - e.dv)
        e_src = abs(c_pre - graph.nodes[u].jacobi)
        e_dst = abs(c_post - graph.nodes[v].jacobi)
        leg_ok = (pos_gap < tol_pos and dv_check < 1e-9
                  and e_src < tol_jac and e_dst < tol_jac)
        ok = ok and leg_ok
        legs.append({"leg": (u, v), "pos_gap": pos_gap, "dv_ms": graph.dv_ms(e.dv),
                     "jacobi_src_resid": e_src, "jacobi_dst_resid": e_dst,
                     "dv_resid": dv_check, "ok": leg_ok})
    return {"path": path, "legs": legs, "ok": ok,
            "total_dv_ms": sum(l.get("dv_ms", 0.0) for l in legs)}


def ephemeris_survivability(graph, path, system, t_horizon=4.0, n=400) -> dict:
    """Max CR3BP-vs-BCR4BP divergence (km) of the route's patch states over t_horizon.

    Small/bounded divergence => the route is a correctable arc that survives the solar
    perturbation; a blow-up => fragile. Quantifies the higher-fidelity re-targeting budget.
    """
    mu = graph.mu
    sp = sun_params(system)
    ts = np.linspace(0.0, t_horizon, n)
    worst = 0.0
    per_leg = []
    for u, v in zip(path[:-1], path[1:]):
        e = _edge(graph, u, v)
        if e is None or "post" not in e.meta:
            continue
        s0 = np.array(e.meta["post"], float)
        cr = propagate(s0, (0.0, t_horizon), mu, t_eval=ts)
        bc = propagate_bcr4bp(s0, (0.0, t_horizon), mu, sp["m_S"], sp["a_S"],
                              sp["omega_S"], t_eval=ts)
        kmax = min(cr.y.shape[1], bc.y.shape[1])
        div = np.linalg.norm(cr.y[:3, :kmax] - bc.y[:3, :kmax], axis=0) * system.L_star
        d = float(div.max())
        per_leg.append({"leg": (u, v), "divergence_km": d})
        worst = max(worst, d)
    return {"worst_divergence_km": worst, "t_horizon_days": t_horizon * system.T_star / 86400.0,
            "per_leg": per_leg}
