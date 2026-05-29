"""Stage 15 validation gates (MASTER_PLAN.md - discovery engine + route verification, G12).

G12a (discovery)     - The engine mines a ranked catalog of >= 3 DISTINCT loopless routes
                       (Yen k-shortest), automatically -- not a single hand-coded transfer.
G12b (verification)  - Every mined route passes CR3BP verification: each patch is a true
                       section crossing (position continuity exact; each side's energy equals
                       its orbit's Jacobi to machine precision; the burn equals the edge Delta-v).
G12c (survivability) - The optimal route's connecting state stays a bounded, correctable arc
                       under the solar perturbation (CR3BP-vs-BCR4BP divergence is finite and
                       within the Earth-Moon neighborhood, not a chaotic escape).

HONEST verdict: G12a-c establish automated discovery + a rigorous CR3BP proof + a quantified
re-targeting budget. Driving that residual to zero in the full DE440 ephemeris AND a GMAT
cross-check is what Stages 8-10 already did for the Earth->Moon TRANSFER leg (~50 m / 149 m);
a dedicated libration-to-libration ephemeris re-targeter is the remaining tool, noted not claimed.
"Novel" here means automatically discovered + verified IPTN structure -- NOT unknown to science.

Run:  PYTHONPATH=src python -m ariadne.validate.stage15
"""
from __future__ import annotations

import math

from ..data.constants import EARTH_MOON
from ..transport_graph.graph import build_transport_graph
from ..discovery.mining import route_catalog, pareto_routes
from ..discovery.verify import verify_route, ephemeris_survivability

ENERGIES = [3.120, 3.140, 3.160, 3.172]
SOURCE = "L1@3.120"
TARGET = "L2@3.172"


def build():
    return build_transport_graph(EARTH_MOON, ENERGIES, points=("L1", "L2"), n_seeds=120)


def check(graph) -> tuple[bool, dict]:
    catalog = route_catalog(graph, SOURCE, TARGET, K=8)
    pf = pareto_routes(catalog)
    verifications = [verify_route(graph, r["path"]) for r in catalog]
    surv = ephemeris_survivability(graph, catalog[0]["path"], EARTH_MOON, t_horizon=2.0)

    distinct = len({tuple(r["path"]) for r in catalog}) == len(catalog)
    g12a = len(catalog) >= 3 and distinct
    g12b = all(v["ok"] for v in verifications)
    g12c = (math.isfinite(surv["worst_divergence_km"])
            and surv["worst_divergence_km"] < 0.5 * EARTH_MOON.L_star)
    ok = g12a and g12b and g12c
    return ok, {"catalog": catalog, "pareto": pf, "verifications": verifications,
                "surv": surv, "g12a": g12a, "g12b": g12b, "g12c": g12c}


def main() -> int:
    print("=== Ariadne Stage 15 validation  (discovery engine + route verification) ===\n")
    g = build()
    n_edges = sum(len(v) for v in g.edges.values())
    print(f"Transport graph: {len(g.nodes)} nodes, {n_edges} edges.  "
          f"Mining {SOURCE} -> {TARGET}\n")

    ok, info = check(g)

    print("[G12a] Discovery: ranked route catalog (Yen k-shortest)")
    for i, r in enumerate(info["catalog"]):
        tag = "  <- optimal" if i == 0 else ("  (Pareto)" if r in info["pareto"] else "")
        print(f"      {i+1}. {r['dv_ms']:7.1f} m/s  {r['hops']} hop(s)  "
              f"{' -> '.join(r['path'])}{tag}")
    print(f"      {len(info['catalog'])} distinct routes  -> {'PASS' if info['g12a'] else 'FAIL'}\n")

    print("[G12b] Verification: every route's patches are true section crossings")
    worst_pos = max((l["pos_gap"] for v in info["verifications"] for l in v["legs"]
                     if "pos_gap" in l), default=0.0)
    worst_jac = max((max(l["jacobi_src_resid"], l["jacobi_dst_resid"])
                     for v in info["verifications"] for l in v["legs"]
                     if "jacobi_src_resid" in l), default=0.0)
    print(f"      all {len(info['verifications'])} routes verified: "
          f"{all(v['ok'] for v in info['verifications'])}")
    print(f"      worst position-continuity gap : {worst_pos:.2e} (nondim)")
    print(f"      worst Jacobi-vs-orbit residual : {worst_jac:.2e}")
    print(f"      -> {'PASS' if info['g12b'] else 'FAIL'}\n")

    s = info["surv"]
    print("[G12c] Survivability: solar perturbation on the optimal route")
    print(f"      CR3BP-vs-BCR4BP divergence over {s['t_horizon_days']:.1f} d: "
          f"{s['worst_divergence_km']:.0f} km  ({s['worst_divergence_km']/EARTH_MOON.L_star:.3f} L*)")
    print(f"      -> bounded/correctable: {'PASS' if info['g12c'] else 'FAIL'}\n")

    print("[note] 'Novel' = automatically discovered + verified IPTN structure, not unknown to")
    print("       science. Full DE440 re-convergence + GMAT was closed for the Earth->Moon")
    print("       transfer leg in Stages 8-10 (~50 m / 149 m); a libration ephemeris re-targeter")
    print("       is the remaining fidelity tool (noted, not claimed).\n")

    print(f"=== STAGE 15: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
