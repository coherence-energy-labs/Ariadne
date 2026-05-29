"""Build the Ariadne atlas (MASTER_PLAN.md §12, Stage 16).

Assembles the atlas dict from the engine, then `store.write_atlas` persists it:
  - libration structure for every system in ATLAS_SYSTEMS (the generalization spread,
    mu ~ 1.6e-8 .. 7e-3: Mars-Phobos, Saturn moons, Sun-Mars, the DART binary asteroid),
  - the full Earth-Moon transport graph + its ranked route catalog (Stage 14/15 artifacts).

Route mining for the other systems is system-agnostic (same code path) and is left as a
straightforward extension; here Earth-Moon carries the graph/routes and the rest carry
their libration summaries, which is enough to demonstrate a multi-system persistent atlas.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..data.constants import EARTH_MOON, ATLAS_SYSTEMS
from ..transfers.jovian import moon_libration
from ..transport_graph.graph import build_transport_graph
from ..discovery.mining import route_catalog

EM_ENERGIES = [3.120, 3.140, 3.160, 3.172]


def _params(S):
    return {"mu": S.mu, "L_star": S.L_star, "T_star": S.T_star, "V_star": S.V_star,
            "primary": S.primary, "secondary": S.secondary}


def _lib_summary(m):
    return {"L1_km": float(m["L1_km"]), "L2_km": float(m["L2_km"]),
            "lyap_period_d": float(m["lyap_period_d"]),
            "half_period_residual": float(m["orbit"].half_period_residual)}


def build_atlas(em_energies=None, n_seeds=120, route_K=8, version="0.16") -> dict:
    """Build the full atlas dict (Earth-Moon graph+routes + multi-system libration)."""
    em_energies = list(em_energies or EM_ENERGIES)
    systems = {}

    # Earth-Moon: full transport graph + ranked routes
    g = build_transport_graph(EARTH_MOON, em_energies, points=("L1", "L2"), n_seeds=n_seeds)
    nodes = [{"key": k, "point": n.point, "jacobi": n.jacobi} for k, n in g.nodes.items()]
    edges = [{"src": e.src, "dst": e.dst, "dv_ms": g.dv_ms(e.dv), "fragility": e.fragility}
             for elist in g.edges.values() for e in elist]
    src, tgt = f"L1@{em_energies[0]:.3f}", f"L2@{em_energies[-1]:.3f}"
    routes = [{"path": r["path"], "dv_ms": r["dv_ms"], "hops": r["hops"]}
              for r in route_catalog(g, src, tgt, K=route_K)]
    systems["Earth-Moon"] = {"params": _params(EARTH_MOON),
                             "libration": _lib_summary(moon_libration(EARTH_MOON)),
                             "graph": {"nodes": nodes, "edges": edges},
                             "routes": routes}

    # Generalization spread: libration structure for each system (engine ports by constants only)
    for S in ATLAS_SYSTEMS:
        systems[S.name] = {"params": _params(S), "libration": _lib_summary(moon_libration(S))}

    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version,
        "note": "Ariadne atlas: multi-system libration + Earth-Moon transport graph + ranked routes",
        "config": {"em_energies": em_energies, "n_seeds": n_seeds, "route_K": route_K},
    }
    return {"provenance": prov, "systems": systems}
