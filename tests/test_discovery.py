"""Stage 15 tests: route mining (Yen) + verification pipeline (G12)."""
import math

import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.transport_graph.graph import TransportGraph, build_transport_graph
from ariadne.discovery.mining import (
    yen_k_shortest, route_catalog, pareto_routes, _constrained_dijkstra,
)
from ariadne.discovery.verify import verify_route, ephemeris_survivability


def _synthetic():
    g = TransportGraph(mu=EARTH_MOON.mu, v_star=EARTH_MOON.V_star)
    for k, c in [("A", 3.10), ("B", 3.13), ("C", 3.13), ("D", 3.16), ("E", 3.17)]:
        g.add_manual_node(k, jacobi=c)
    for s, d, w in [("A", "B", 1.0), ("A", "C", 2.0), ("B", "C", 0.1),
                    ("B", "D", 2.0), ("C", "D", 1.0), ("D", "E", 1.0),
                    ("A", "D", 5.0), ("C", "E", 3.0)]:
        g.add_manual_edge(s, d, w)
    return g


def test_yen_returns_distinct_cost_ordered_paths():
    g = _synthetic()
    routes = yen_k_shortest(g, "A", "E", K=5)
    costs = [c for c, _ in routes]
    paths = [tuple(p) for _, p in routes]
    assert routes[0][1] == ["A", "B", "C", "D", "E"]      # the known optimum
    assert abs(costs[0] - 3.1) < 1e-12
    assert costs == sorted(costs)                          # non-decreasing
    assert len(set(paths)) == len(paths)                   # all distinct


def test_constrained_dijkstra_respects_masks():
    g = _synthetic()
    c0, p0 = _constrained_dijkstra(g, "A", "E", set(), set())
    c1, p1 = _constrained_dijkstra(g, "A", "E", {("B", "C")}, set())
    assert p0 == ["A", "B", "C", "D", "E"]
    assert p1 != p0 and c1 >= c0                           # masking an edge cannot help


def test_pareto_routes_are_nondominated():
    g = _synthetic()
    cat = route_catalog(g, "A", "E", K=6)
    # give routes differing fragility so the Pareto set is non-trivial
    for r in cat:
        r["fragility"] = float(r["hops"])                  # fewer hops = more robust
    pf = pareto_routes(cat)
    for r in pf:
        assert not any(q is not r and q["dv_ms"] <= r["dv_ms"]
                       and q["fragility"] <= r["fragility"]
                       and (q["dv_ms"] < r["dv_ms"] or q["fragility"] < r["fragility"])
                       for q in cat)


@pytest.mark.slow
def test_verify_route_on_real_graph_is_exact():
    """A real route's patches are true crossings: continuity and energy exact."""
    g = build_transport_graph(EARTH_MOON, [3.14, 3.16], points=("L1", "L2"), n_seeds=60)
    cat = route_catalog(g, "L1@3.140", "L2@3.160", K=4)
    assert cat, "no route mined"
    v = verify_route(g, cat[0]["path"])
    assert v["ok"]
    for leg in v["legs"]:
        assert leg["pos_gap"] < 1e-9                        # burn where you are
        assert leg["jacobi_src_resid"] < 1e-9              # energy-exact (no interpolation)
        assert leg["jacobi_dst_resid"] < 1e-9


@pytest.mark.slow
def test_survivability_is_finite_and_bounded():
    g = build_transport_graph(EARTH_MOON, [3.14, 3.16], points=("L1", "L2"), n_seeds=60)
    cat = route_catalog(g, "L1@3.140", "L2@3.160", K=2)
    s = ephemeris_survivability(g, cat[0]["path"], EARTH_MOON, t_horizon=2.0)
    assert math.isfinite(s["worst_divergence_km"])
    assert s["worst_divergence_km"] < EARTH_MOON.L_star    # stays in the EM neighborhood
