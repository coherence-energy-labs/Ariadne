"""Stage 19 validation gates (MASTER_PLAN.md - 3D halos + the Gateway-class NRHO).

G19a (NRHO geometry)  - Pseudo-arclength continuation of the L2 halo family reaches a
                        Near-Rectilinear Halo Orbit matching NASA's Gateway 9:2 NRHO:
                        period ~6.5-6.6 d, perilune ~2,800-3,800 km (low pass over the pole),
                        apolune ~60,000-78,000 km, periodic to < 1e-9.
G19b (near-stability) - The NRHO is FAR more stable than a deep libration orbit: its maximum
                        Floquet multiplier is small (the reason Gateway flies an NRHO -- cheap
                        stationkeeping), vs the L1 Lyapunov orbit's multiplier in the thousands.
G19c (3D graph build) - The 3D-transport-graph builder (connections.poincare_3d +
                        transport_graph.graph.build_transport_graph_3d) constructs a graph of
                        L1/L2 halos by intersecting their 3D-manifold tubes as 1-curves in the
                        (y, z, vy, vz) 4-space cut by x = 1-mu, finding the closest 4D approach
                        of each pair as the patch point.
G19d (3D heteroclinic) - At least one 3D-manifold tube pair produces a section intersection,
                         yielding a finite-Delta-v edge -- the genuine 3D extension of the
                         planar (y, vy) heteroclinic-connection method to spatially-extended
                         orbits (halos, NRHOs). Cislunar/Gateway-class transfer design.

Honest scope: NRHO's small Floquet multiplier means its tube barely reaches x = 1-mu within a
reasonable propagation time (~12 nondim ~ 50 days); the Gateway NRHO node has zero crossings at
that section. For NRHO transport graph edges, a section closer to the NRHO itself (e.g., y = 0,
which the NRHO crosses every period) is the correct choice -- a future-extension item.

Run:  PYTHONPATH=src python -m ariadne.validate.stage19
"""
from __future__ import annotations

import warnings

import numpy as np

from ..data.constants import EARTH_MOON, R_MOON
from ..dynamics.cr3bp import propagate
from ..orbits.differential_correction import monodromy
from ..orbits.families import lyapunov_orbit_at_jacobi
from ..orbits.halo import halo_family
from ..orbits.nrho import nrho_family
from ..transport_graph.graph import build_transport_graph_3d

T_DAYS = EARTH_MOON.T_star / 86400.0
LSTAR = EARTH_MOON.L_star
MU = EARTH_MOON.mu


def _peri_apo_km(s0, period):
    sol = propagate(s0, (0.0, period), MU, t_eval=np.linspace(0.0, period, 800))
    d = np.sqrt((sol.y[0] - (1 - MU)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * LSTAR
    return float(d.min()), float(d.max())


def _max_floquet(orbit):
    return float(np.max(np.abs(np.linalg.eigvals(monodromy(MU, orbit)))))


def _check_3d_graph():
    """G19c + G19d: build the 3D transport graph for an L1+L2 halo pair, verify >=1 edge."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l1 = halo_family(MU, point="L1", n=10, dz=4e-3, fam_n=40,
                         lyap_amp0=2e-3, lyap_dx=4e-3)
        l2 = halo_family(MU, point="L2", n=10, dz=4e-3, fam_n=40,
                         lyap_amp0=2e-3, lyap_dx=4e-3)
        l1_pick = l1[len(l1) // 2]
        l2_pick = min(l2, key=lambda h: abs(h.jacobi - l1_pick.jacobi))
        specs = [{"key": "L1_halo", "point": "L1", "orbit": l1_pick},
                 {"key": "L2_halo", "point": "L2", "orbit": l2_pick}]
        g = build_transport_graph_3d(EARTH_MOON, specs, n_seeds=120,
                                     displacement=1e-4, t_max=12.0, max_pos_gap=8e-3)
    g19c = (len(g.nodes) == 2)
    n_edges = sum(len(e) for e in g.edges.values())
    g19d = (n_edges >= 1)
    best_edge = None
    for elist in g.edges.values():
        for e in elist:
            if best_edge is None or e.dv < best_edge.dv:
                best_edge = e
    return g19c, g19d, {"n_nodes": len(g.nodes), "n_edges": n_edges,
                        "l1_jacobi": float(l1_pick.jacobi),
                        "l2_jacobi": float(l2_pick.jacobi),
                        "best_dv_ms": g.dv_ms(best_edge.dv) if best_edge else None,
                        "best_meta": best_edge.meta if best_edge else None}


def check() -> tuple[bool, dict]:
    nrho, fam = nrho_family(MU, "L2", t_star_days=T_DAYS, l_star=LSTAR,
                            target_period_d=6.56, ds=4e-3)
    info = {"n_family": len(fam),
            "period_range_d": (fam[0].period * T_DAYS, fam[-1].period * T_DAYS) if fam else None}
    if nrho is None:
        return False, {**info, "nrho": None, "g19a": False, "g19b": False,
                       "g19c": False, "g19d": False}

    peri, apo = _peri_apo_km(nrho.s0, nrho.period)
    period_d = nrho.period * T_DAYS
    floq_nrho = _max_floquet(nrho)
    floq_lyap = _max_floquet(lyapunov_orbit_at_jacobi(MU, "L1", 3.16))

    g19a = (6.3 <= period_d <= 6.8 and 2800 <= peri <= 3800
            and 60000 <= apo <= 78000 and nrho.residual < 1e-9)
    g19b = floq_nrho < 100.0 and floq_nrho < 0.1 * floq_lyap

    g19c, g19d, graph_info = _check_3d_graph()
    ok = g19a and g19b and g19c and g19d
    info.update({"nrho": nrho, "period_d": period_d, "peri_km": peri, "apo_km": apo,
                 "peri_alt_km": peri - R_MOON, "floq_nrho": floq_nrho,
                 "floq_lyap": floq_lyap, "g19a": g19a, "g19b": g19b,
                 "g19c": g19c, "g19d": g19d, "graph": graph_info})
    return ok, info


def main() -> int:
    print("=== Ariadne Stage 19 validation  (3D halos + Gateway NRHO + 3D transport graph) ===\n")
    ok, i = check()
    if i["period_range_d"]:
        print(f"L2 halo->NRHO continuation: {i['n_family']} members, "
              f"period {i['period_range_d'][0]:.2f} d -> {i['period_range_d'][1]:.2f} d\n")

    print("[G19a] Near-Rectilinear Halo Orbit geometry (vs NASA Gateway 9:2 NRHO)")
    if i.get("nrho") is not None:
        print(f"      period   = {i['period_d']:.3f} d   (Gateway ~6.56 d)")
        print(f"      perilune = {i['peri_km']:.0f} km  (alt {i['peri_alt_km']:.0f} km over the pole)")
        print(f"      apolune  = {i['apo_km']:.0f} km   (Gateway ~70,000 km)")
        print(f"      periodic to {i['nrho'].residual:.1e}")
    print(f"      -> {'PASS' if i['g19a'] else 'FAIL'}\n")

    print("[G19b] Near-stability (why Gateway flies an NRHO)")
    if "floq_nrho" in i:
        print(f"      NRHO max Floquet multiplier   = {i['floq_nrho']:.2f}")
        print(f"      L1 Lyapunov (C=3.16) for scale = {i['floq_lyap']:.0f}")
    print(f"      -> {'PASS' if i['g19b'] else 'FAIL'}\n")

    g = i["graph"]
    print("[G19c] 3D transport graph builds (L1 + L2 halo manifolds, x=1-mu section)")
    print(f"      {g['n_nodes']} nodes (L1@C={g['l1_jacobi']:.4f}, L2@C={g['l2_jacobi']:.4f})  "
          f"{g['n_edges']} edges")
    print(f"      -> {'PASS' if i['g19c'] else 'FAIL'}\n")

    print("[G19d] 3D-tube heteroclinic edge (4D closest-approach in (y,z,vy,vz))")
    if g["best_dv_ms"] is not None:
        m = g["best_meta"]
        print(f"      best patch dv = {g['best_dv_ms']:.1f} m/s   "
              f"position gap = {m['pos_gap'] * LSTAR:.0f} km   "
              f"at (y={m['y']:+.3f}, z={m['z']:+.3f})")
    print(f"      -> {'PASS' if i['g19d'] else 'FAIL'}\n")

    print(f"=== STAGE 19: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
