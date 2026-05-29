"""Stage 14 validation gates (MASTER_PLAN.md - transport-graph search + efficiency benchmark).

G11a (optimality) - Dijkstra (SSSP) and A* return the SAME minimum-Delta-v route as an
                    exhaustive brute-force search. Graph search does not sacrifice the optimum.
G11b (efficiency) - A* (and Dijkstra) reach that optimum while expanding dramatically fewer
                    nodes/edges than brute force (the benchmark the project promised).
G11c (admissible) - The A* energy heuristic provably never overestimates the true remaining
                    cost, so A*'s optimality is guaranteed by construction, not luck.
G11d (physical)   - The graph encodes real IPTN structure: it contains a near-ballistic
                    (< 50 m/s) same-energy L1<->L2 patch -- the known heteroclinic connection.

Run:  PYTHONPATH=src python -m ariadne.validate.stage14
"""
from __future__ import annotations

from ..data.constants import EARTH_MOON
from ..transport_graph.graph import build_transport_graph
from ..transport_graph.benchmark import benchmark_route, ballistic_edges, coherence_shift


ENERGIES = [3.120, 3.140, 3.160, 3.172]    # Jacobi grid inside both L1 and L2 families


def build():
    # n_seeds=120: converged resolution. With the energy-exact edge model (v_x from energy
    # conservation at each crossing) the optimal ROUTE TOPOLOGY is stable across resolution
    # and the cost converges to ~17 m/s (90->12.7, 120->16.1, 150->16.9; same 3-hop route) --
    # a genuine multi-hop optimum, not an artifact. See MASTER_PLAN Stage 14 convergence note.
    return build_transport_graph(EARTH_MOON, ENERGIES, points=("L1", "L2"),
                                 n_seeds=120)


def check(graph) -> tuple[bool, dict]:
    keys = list(graph.nodes)
    source = f"L1@{ENERGIES[0]:.3f}"
    target = f"L2@{ENERGIES[-1]:.3f}"
    bench = benchmark_route(graph, source, target)
    ball = ballistic_edges(graph, dv_ms_threshold=50.0)
    shift = coherence_shift(graph, source, target, weights=(0.0, 1.0, 5.0, 20.0))

    g11a = bench["optimal_match"]
    g11b = (bench["speedup_astar_vs_brute"] >= 5.0
            and bench["astar"]["expansions"] <= bench["dijkstra"]["expansions"])
    g11c = bool(bench["admissible"])
    g11d = len(ball) > 0
    ok = g11a and g11b and g11c and g11d
    return ok, {"bench": bench, "ballistic": ball, "shift": shift,
                "g11a": g11a, "g11b": g11b, "g11c": g11c, "g11d": g11d,
                "source": source, "target": target, "n_nodes": len(keys)}


def main() -> int:
    print("=== Ariadne Stage 14 validation  (transport-graph search + efficiency benchmark) ===\n")
    g = build()
    n_edges = sum(len(v) for v in g.edges.values())
    print(f"Transport graph: {len(g.nodes)} nodes (L1/L2 Lyapunov at {len(ENERGIES)} energies), "
          f"{n_edges} patch edges.\n")

    ok, info = check(g)
    b = info["bench"]

    def fmt_path(p):
        return " -> ".join(p) if p else "(none)"

    print(f"[G11a] Optimality  (route {info['source']}  =>  {info['target']})")
    print(f"      brute force : {g.dv_ms(b['brute']['cost']):8.1f} m/s   {fmt_path(b['brute']['path'])}")
    print(f"      Dijkstra    : {g.dv_ms(b['dijkstra']['cost']):8.1f} m/s   {fmt_path(b['dijkstra']['path'])}")
    print(f"      A*          : {g.dv_ms(b['astar']['cost']):8.1f} m/s   {fmt_path(b['astar']['path'])}")
    print(f"      -> all agree on the optimum: {'PASS' if info['g11a'] else 'FAIL'}\n")

    print("[G11b] Efficiency  (work to find that optimum)")
    print(f"      brute force : {b['brute']['expansions']:6d} edge-expansions   {b['brute']['time']*1e3:7.1f} ms")
    print(f"      Dijkstra    : {b['dijkstra']['expansions']:6d} node-expansions   {b['dijkstra']['time']*1e3:7.1f} ms")
    print(f"      A*          : {b['astar']['expansions']:6d} node-expansions   {b['astar']['time']*1e3:7.1f} ms")
    print(f"      A* vs brute speedup: {b['speedup_astar_vs_brute']:.1f}x fewer expansions"
          f"  -> {'PASS' if info['g11b'] else 'FAIL'}\n")

    print(f"[G11c] Heuristic admissibility (slope k = {b['heuristic_slope']:.3f})")
    print(f"      h(n) <= true remaining cost for all n: {'PASS' if info['g11c'] else 'FAIL'}\n")

    print("[G11d] Physical sanity: near-ballistic heteroclinic patches in the graph")
    for src, dst, dv_ms in info["ballistic"][:4]:
        print(f"      {src} -> {dst}: {dv_ms:.1f} m/s")
    print(f"      found {len(info['ballistic'])} edge(s) < 50 m/s  -> {'PASS' if info['g11d'] else 'FAIL'}\n")

    print("[info] Coherence-weighted route choice (robustness penalty w):")
    for r in info["shift"]:
        print(f"      w={r['w_robust']:>4}: {fmt_path(r['path'])}")
    paths = {tuple(r["path"]) for r in info["shift"] if r["path"]}
    if len(paths) == 1:
        print("      (the min-Delta-v patch is also the least-fragile here, so the robustness")
        print("       weight does not change it -- the two objectives agree on this graph.)")
    else:
        print("      (raising the robustness weight selects a different, more coherent route.)")
    print()

    print(f"=== STAGE 14: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
