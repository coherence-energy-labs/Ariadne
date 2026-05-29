"""Shortest-path search over the transport graph (MASTER_PLAN.md - Stage 14, G11).

Three routers, deliberately, so the efficiency claim can be measured honestly:

  - dijkstra        : single-source shortest path (SSSP). Optimal. Expands each node
                      at most once. This is the classical result the project is built on.
  - astar           : A* with an admissible energy heuristic. Optimal too (because the
                      heuristic never overestimates), but expands fewer nodes by aiming
                      at the target instead of flooding outward.
  - brute_force     : exhaustive enumeration of every simple path source->target. Also
                      optimal, but its work explodes combinatorially -- the baseline the
                      graph search is supposed to beat.

The point of G11 is NOT that Dijkstra/A* find a better answer (they cannot beat the
optimum; neither can brute force). It is that they find the SAME optimum for a tiny
fraction of the work. We count node expansions and wall-clock and verify the optima agree.
"""
from __future__ import annotations

import heapq
import math
import time

INF = math.inf


def _w(edge, weight, w_robust):
    if weight == "dv":
        return edge.dv + w_robust * edge.fragility
    if weight == "fragility":
        return edge.fragility
    raise ValueError(weight)


def dijkstra(graph, source, weight="dv", w_robust=0.0):
    """Single-source shortest path. Returns dist, prev, and #expansions."""
    dist = {source: 0.0}
    prev: dict[str, str] = {}
    pq = [(0.0, source)]
    visited: set[str] = set()
    expansions = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        expansions += 1
        for e in graph.edges.get(u, []):
            nd = d + _w(e, weight, w_robust)
            if nd < dist.get(e.dst, INF):
                dist[e.dst] = nd
                prev[e.dst] = u
                heapq.heappush(pq, (nd, e.dst))
    return {"dist": dist, "prev": prev, "expansions": expansions}


def dist_to_target(graph, target, weight="dv", w_robust=0.0):
    """Exact shortest distance from every node TO target (Dijkstra on the reversed graph).

    This is h*(n) -- the perfect heuristic -- used to calibrate and verify admissibility.
    """
    return dijkstra(graph.reversed(), target, weight, w_robust)["dist"]


def reconstruct_path(prev, source, target):
    if target not in prev and target != source:
        return None
    path = [target]
    while path[-1] != source:
        p = prev.get(path[-1])
        if p is None:
            return None
        path.append(p)
    return list(reversed(path))


def calibrate_energy_heuristic(graph, target, weight="dv", w_robust=0.0, safety=0.9):
    """Largest admissible slope k for h(n) = k * |C_n - C_target|.

    Admissible means h(n) <= h*(n) for all n. We take k = safety * min over reachable,
    different-energy nodes of h*(n)/|C_n - C_target|, guaranteeing admissibility by
    construction (safety < 1 keeps a margin against floating point).
    """
    hstar = dist_to_target(graph, target, weight, w_robust)
    Ct = graph.nodes[target].jacobi
    ratios = []
    for key, h in hstar.items():
        if key == target or not math.isfinite(h):
            continue
        dC = abs(graph.nodes[key].jacobi - Ct)
        if dC > 1e-9:
            ratios.append(h / dC)
    k = safety * min(ratios) if ratios else 0.0
    return k


def energy_heuristic(graph, target, k):
    """Admissible A* heuristic: a lower bound on remaining cost from the energy gap."""
    Ct = graph.nodes[target].jacobi

    def h(node_key):
        return k * abs(graph.nodes[node_key].jacobi - Ct)
    return h


def is_admissible(graph, target, h, weight="dv", w_robust=0.0, tol=1e-9):
    """Verify h(n) <= true distance-to-target for every node (so A* stays optimal)."""
    hstar = dist_to_target(graph, target, weight, w_robust)
    for key in graph.nodes:
        if key == target:
            continue
        true_d = hstar.get(key, INF)
        if not math.isfinite(true_d):
            continue                       # unreachable: any finite h is fine
        if h(key) > true_d + tol:
            return False
    return True


def astar(graph, source, target, h, weight="dv", w_robust=0.0):
    """A* shortest path source->target. Returns cost, path, and #expansions."""
    g_cost = {source: 0.0}
    prev: dict[str, str] = {}
    pq = [(h(source), 0.0, source)]
    closed: set[str] = set()
    expansions = 0
    while pq:
        _, gu, u = heapq.heappop(pq)
        if u in closed:
            continue
        if u == target:
            return {"cost": gu, "path": reconstruct_path(prev, source, target),
                    "expansions": expansions}
        closed.add(u)
        expansions += 1
        for e in graph.edges.get(u, []):
            ng = gu + _w(e, weight, w_robust)
            if ng < g_cost.get(e.dst, INF):
                g_cost[e.dst] = ng
                prev[e.dst] = u
                heapq.heappush(pq, (ng + h(e.dst), ng, e.dst))
    return {"cost": INF, "path": None, "expansions": expansions}


def brute_force(graph, source, target, weight="dv", w_robust=0.0, max_depth=None):
    """Exhaustive min-cost simple path by DFS. Returns cost, path, and #expansions.

    `expansions` counts edge relaxations (partial-path extensions) -- the work measure
    comparable to Dijkstra/A* node expansions. This blows up combinatorially; that is
    the whole point of the baseline.
    """
    if max_depth is None:
        max_depth = len(graph.nodes)
    best = {"cost": INF, "path": None}
    counter = {"n": 0}

    def dfs(u, cost, path, on_path):
        if u == target:
            if cost < best["cost"]:
                best["cost"] = cost
                best["path"] = list(path)
            return
        if len(path) >= max_depth:
            return
        for e in graph.edges.get(u, []):
            if e.dst in on_path:
                continue
            counter["n"] += 1
            nc = cost + _w(e, weight, w_robust)
            if nc >= best["cost"]:          # bound: cannot improve
                continue
            on_path.add(e.dst)
            path.append(e.dst)
            dfs(e.dst, nc, path, on_path)
            path.pop()
            on_path.discard(e.dst)

    dfs(source, 0.0, [source], {source})
    return {"cost": best["cost"], "path": best["path"], "expansions": counter["n"]}


def timed(fn, *a, **k):
    """Run fn, return (result, elapsed_seconds). (Wall clock for the benchmark.)"""
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
