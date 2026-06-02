"""Build a proof artifact for transport-graph admissible search."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ariadne.data.constants import EARTH_MOON
from ariadne.proof import stable_hash
from ariadne.transport_graph.benchmark import benchmark_route
from ariadne.transport_graph.graph import TransportGraph


def _synthetic_graph() -> TransportGraph:
    g = TransportGraph(mu=EARTH_MOON.mu, v_star=EARTH_MOON.V_star)
    for key, c in [("A", 3.10), ("B", 3.13), ("C", 3.13), ("D", 3.16), ("E", 3.17)]:
        g.add_manual_node(key, jacobi=c)
    for src, dst, dv in [
        ("A", "B", 1.0), ("A", "C", 2.0), ("B", "C", 0.1),
        ("B", "D", 2.0), ("C", "D", 1.0), ("D", "E", 1.0),
        ("A", "D", 5.0), ("C", "E", 3.0),
    ]:
        g.add_manual_edge(src, dst, dv)
    return g


def build_artifact() -> dict:
    graph = _synthetic_graph()
    benchmark = benchmark_route(graph, "A", "E")
    passed = (
        bool(benchmark["admissible"])
        and bool(benchmark["optimal_match"])
        and benchmark["astar"]["expansions"] <= benchmark["dijkstra"]["expansions"]
        and benchmark["speedup_astar_vs_brute"] >= 1.0
    )
    payload = {
        "schema": "ariadne.transport_admissibility_benchmark.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "passed": passed,
        "admissible": bool(benchmark["admissible"]),
        "optimal_match": bool(benchmark["optimal_match"]),
        "astar_expansions": benchmark["astar"]["expansions"],
        "dijkstra_expansions": benchmark["dijkstra"]["expansions"],
        "brute_expansions": benchmark["brute"]["expansions"],
        "speedup_astar_vs_brute": benchmark["speedup_astar_vs_brute"],
        "benchmark": benchmark,
    }
    payload["certificate_hash"] = stable_hash({k: v for k, v in payload.items() if k != "created_utc"})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/benchmarks/transport_admissibility")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    path = out / "metrics.json"
    path.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"passed={artifact['passed']}")
    print(f"admissible={artifact['admissible']}")
    print(f"optimal_match={artifact['optimal_match']}")
    print(f"certificate_hash={artifact['certificate_hash']}")
    print(f"metrics={path}")
    return 0 if artifact["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
