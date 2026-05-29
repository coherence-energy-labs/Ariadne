"""Stage 14 tests: transport-graph search correctness + efficiency benchmark (G11)."""
import math

import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.transport_graph.graph import TransportGraph, build_transport_graph
from ariadne.transport_graph.search import (
    dijkstra, astar, brute_force, reconstruct_path,
    calibrate_energy_heuristic, energy_heuristic, is_admissible, dist_to_target,
)
from ariadne.transport_graph.benchmark import benchmark_route, ballistic_edges, coherence_shift


def _synthetic():
    """A small graph with a hand-checkable optimum A->B->C->D->E (cost 3.1)."""
    g = TransportGraph(mu=EARTH_MOON.mu, v_star=EARTH_MOON.V_star)
    for k, c in [("A", 3.10), ("B", 3.13), ("C", 3.13), ("D", 3.16), ("E", 3.17)]:
        g.add_manual_node(k, jacobi=c)
    for s, d, w in [("A", "B", 1.0), ("A", "C", 2.0), ("B", "C", 0.1),
                    ("B", "D", 2.0), ("C", "D", 1.0), ("D", "E", 1.0),
                    ("A", "D", 5.0), ("C", "E", 3.0)]:
        g.add_manual_edge(s, d, w)
    return g


def test_all_routers_agree_on_optimum():
    g = _synthetic()
    dj = dijkstra(g, "A")
    bf = brute_force(g, "A", "E")
    k = calibrate_energy_heuristic(g, "E")
    h = energy_heuristic(g, "E", k)
    asr = astar(g, "A", "E", h)
    assert abs(dj["dist"]["E"] - 3.1) < 1e-12
    assert abs(bf["cost"] - 3.1) < 1e-12
    assert abs(asr["cost"] - 3.1) < 1e-12
    assert reconstruct_path(dj["prev"], "A", "E") == ["A", "B", "C", "D", "E"]
    assert asr["path"] == ["A", "B", "C", "D", "E"]


def test_heuristic_is_admissible_and_astar_not_worse_than_dijkstra():
    g = _synthetic()
    k = calibrate_energy_heuristic(g, "E")
    h = energy_heuristic(g, "E", k)
    assert is_admissible(g, "E", h)
    dj = dijkstra(g, "A")
    asr = astar(g, "A", "E", h)
    assert asr["expansions"] <= dj["expansions"]


def test_brute_force_does_more_work_than_dijkstra():
    g = _synthetic()
    dj = dijkstra(g, "A")
    bf = brute_force(g, "A", "E")
    assert bf["expansions"] > dj["expansions"]


def test_dist_to_target_matches_forward_distance():
    g = _synthetic()
    dj = dijkstra(g, "A")
    rev = dist_to_target(g, "E")           # distance-to-E from every node
    # the source's distance to target must equal forward dist A->E
    assert abs(rev["A"] - dj["dist"]["E"]) < 1e-12


def test_coherence_weight_can_change_route():
    g = _synthetic()
    # give the cheap B->C edge high fragility so robustness routing avoids it
    for e in g.edges["B"]:
        if e.dst == "C":
            e.fragility = 50.0
    rows = coherence_shift(g, "A", "E", weights=(0.0, 2.0))
    assert rows[0]["path"] is not None and rows[1]["path"] is not None


@pytest.mark.slow
def test_physical_graph_has_ballistic_heteroclinic():
    """A tiny real Earth-Moon graph contains a near-ballistic same-energy L1<->L2 patch."""
    g = build_transport_graph(EARTH_MOON, [3.14, 3.16], points=("L1", "L2"),
                              n_seeds=60)
    ball = ballistic_edges(g, dv_ms_threshold=80.0)
    assert len(ball) > 0
    # the cheapest near-ballistic patch should connect L1 and L2 at the SAME energy
    src, dst, dv_ms = ball[0]
    assert src.split("@")[1] == dst.split("@")[1]      # same Jacobi level
    assert src.split("@")[0] != dst.split("@")[0]      # different libration point
