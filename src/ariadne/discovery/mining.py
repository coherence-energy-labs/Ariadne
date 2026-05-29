"""Route mining over the transport graph (MASTER_PLAN.md - Stage 15, G12).

The discovery engine. Stage 14 found the single optimal route; here we enumerate the
k-shortest LOOPLESS routes (Yen's algorithm) to build a ranked route catalog -- the
non-obvious multi-hop alternatives the optimum hides. Each route is scored by total
Delta-v, hop count, and accumulated fragility (the coherence/robustness cost), and we
extract the Delta-v-vs-robustness Pareto set so a mission can pick its trade.

Yen's algorithm reuses a constrained Dijkstra (edges/nodes can be masked) as its
shortest-path subroutine, so the catalog inherits Stage 14's exactness.
"""
from __future__ import annotations

import heapq
import math

from ..transport_graph.search import _w

INF = math.inf


def _constrained_dijkstra(graph, source, target, removed_edges, removed_nodes,
                          weight="dv", w_robust=0.0):
    """Shortest path source->target avoiding removed edges/nodes. (cost, path) or (inf, None)."""
    if source in removed_nodes or target in removed_nodes:
        return INF, None
    dist = {source: 0.0}
    prev: dict[str, str] = {}
    pq = [(0.0, source)]
    visited: set[str] = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for e in graph.edges.get(u, []):
            if e.dst in removed_nodes or (e.src, e.dst) in removed_edges:
                continue
            nd = d + _w(e, weight, w_robust)
            if nd < dist.get(e.dst, INF):
                dist[e.dst] = nd
                prev[e.dst] = u
                heapq.heappush(pq, (nd, e.dst))
    if target not in dist:
        return INF, None
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    return dist[target], list(reversed(path))


def _path_cost(graph, path, weight="dv", w_robust=0.0):
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = next((e for e in graph.edges.get(u, []) if e.dst == v), None)
        if e is None:
            return INF
        total += _w(e, weight, w_robust)
    return total


def yen_k_shortest(graph, source, target, K=5, weight="dv", w_robust=0.0):
    """K shortest LOOPLESS paths source->target (Yen 1971). Returns [(cost, path), ...]."""
    c0, p0 = _constrained_dijkstra(graph, source, target, set(), set(), weight, w_robust)
    if p0 is None:
        return []
    A = [(c0, p0)]
    B: list[tuple[float, list]] = []
    accepted = {tuple(p0)}
    while len(A) < K:
        _, prev_path = A[-1]
        for i in range(len(prev_path) - 1):
            spur = prev_path[i]
            root = prev_path[:i + 1]
            removed_edges = set()
            for _, p in A:
                if len(p) > i and p[:i + 1] == root:
                    removed_edges.add((p[i], p[i + 1]))
            removed_nodes = set(root[:-1])
            sc, sp = _constrained_dijkstra(graph, spur, target, removed_edges,
                                           removed_nodes, weight, w_robust)
            if sp is None:
                continue
            total = _path_cost(graph, root, weight, w_robust) + sc
            cand_path = root[:-1] + sp
            key = tuple(cand_path)
            if key in accepted or any(tuple(b[1]) == key for b in B):
                continue
            B.append((total, cand_path))
        if not B:
            break
        B.sort(key=lambda x: x[0])
        cost, path = B.pop(0)
        A.append((cost, path))
        accepted.add(tuple(path))
    return A


def describe_route(graph, path, weight="dv"):
    """Annotate a route with physical Delta-v (m/s), hop count, and total fragility."""
    dv_nondim = _path_cost(graph, path, "dv", 0.0)
    frag = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = next((e for e in graph.edges.get(u, []) if e.dst == v), None)
        if e:
            frag += e.fragility
    return {"path": path, "hops": len(path) - 1,
            "dv_ms": graph.dv_ms(dv_nondim), "fragility": frag}


def route_catalog(graph, source, target, K=8):
    """Ranked catalog of the K shortest routes, each described physically."""
    routes = yen_k_shortest(graph, source, target, K=K)
    return [describe_route(graph, p) for _, p in routes]


def pareto_routes(catalog):
    """Non-dominated routes minimizing BOTH dv_ms and fragility."""
    front = []
    for r in catalog:
        dominated = any(
            q is not r and q["dv_ms"] <= r["dv_ms"] and q["fragility"] <= r["fragility"]
            and (q["dv_ms"] < r["dv_ms"] or q["fragility"] < r["fragility"])
            for q in catalog)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["dv_ms"])
