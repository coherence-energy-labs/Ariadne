"""Efficiency benchmark: graph search vs brute force (MASTER_PLAN.md - Stage 14, G11).

Runs Dijkstra (SSSP), A* (admissible energy heuristic), and exhaustive brute force on
the same transport graph and reports, honestly:
  - do they agree on the optimal cost? (they must -- all three are exact)
  - how much less work does the graph search do? (node/edge expansions, wall clock)
  - is the A* heuristic provably admissible? (so its optimality is guaranteed, not luck)

Also reports how the optimum route changes when the edge weight includes the coherence
(fragility) penalty -- the bridge to the Stage 11/12 robustness lens.
"""
from __future__ import annotations

import math

from .search import (
    dijkstra, astar, brute_force, reconstruct_path,
    calibrate_energy_heuristic, energy_heuristic, is_admissible, dist_to_target, timed,
)


def benchmark_route(graph, source, target, w_robust=0.0):
    """Compare the three routers on source->target. Returns a result dict."""
    dj, t_dj = timed(dijkstra, graph, source, "dv", w_robust)
    dj_path = reconstruct_path(dj["prev"], source, target)
    dj_cost = dj["dist"].get(target, math.inf)

    k = calibrate_energy_heuristic(graph, target, "dv", w_robust)
    h = energy_heuristic(graph, target, k)
    admissible = is_admissible(graph, target, h, "dv", w_robust)
    asr, t_as = timed(astar, graph, source, target, h, "dv", w_robust)

    bf, t_bf = timed(brute_force, graph, source, target, "dv", w_robust)

    tol = 1e-9
    optimal_match = (math.isfinite(dj_cost) and math.isfinite(bf["cost"])
                     and abs(dj_cost - bf["cost"]) < tol
                     and abs(asr["cost"] - bf["cost"]) < tol)

    return {
        "source": source, "target": target, "w_robust": w_robust,
        "heuristic_slope": k, "admissible": admissible,
        "optimal_match": optimal_match,
        "dijkstra": {"cost": dj_cost, "path": dj_path,
                     "expansions": dj["expansions"], "time": t_dj},
        "astar": {"cost": asr["cost"], "path": asr["path"],
                  "expansions": asr["expansions"], "time": t_as},
        "brute": {"cost": bf["cost"], "path": bf["path"],
                  "expansions": bf["expansions"], "time": t_bf},
        "speedup_astar_vs_brute": (bf["expansions"] / asr["expansions"]
                                   if asr["expansions"] else math.inf),
        "speedup_dijkstra_vs_brute": (bf["expansions"] / dj["expansions"]
                                      if dj["expansions"] else math.inf),
    }


def ballistic_edges(graph, dv_ms_threshold=50.0):
    """Edges whose physical patch Delta-v is below threshold -- the near-ballistic
    (heteroclinic) connections. Used as a physical sanity check on the graph."""
    out = []
    for elist in graph.edges.values():
        for e in elist:
            dv_ms = graph.dv_ms(e.dv)
            if dv_ms < dv_ms_threshold:
                out.append((e.src, e.dst, dv_ms))
    return sorted(out, key=lambda r: r[2])


def coherence_shift(graph, source, target, weights=(0.0, 0.5, 2.0)):
    """How the optimal route changes as the coherence (robustness) weight grows."""
    rows = []
    for w in weights:
        dj = dijkstra(graph, source, "dv", w)
        path = reconstruct_path(dj["prev"], source, target)
        rows.append({"w_robust": w, "cost": dj["dist"].get(target, math.inf),
                     "path": path})
    return rows
