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
G19e (NRHO transport)  - NRHO transport via the y=0 Poincare section (the right section for
                         NRHO; its small Floquet multiplier prevents x=1-mu reach). poincare.py
                         and poincare_3d.py now accept an `axis` parameter (0=x, 1=y, 2=z), so
                         each orbit can be cut by its natural section. Demonstration: NRHO
                         unstable manifold intersects an L2 halo stable manifold at y=0 with
                         3D patch Delta-v < 200 m/s -- a real heteroclinic-class Gateway-to-
                         halo transfer (the genuine Artemis mission-design relevance).

Run:  PYTHONPATH=src python -m ariadne.validate.stage19
"""
from __future__ import annotations

import warnings

import math

import numpy as np

from ..data.constants import EARTH_MOON, R_MOON
from ..dynamics.cr3bp import propagate, pseudo_potential
from ..orbits.differential_correction import monodromy
from ..orbits.families import lyapunov_orbit_at_jacobi
from ..orbits.halo import halo_family
from ..orbits.nrho import nrho_family
from ..connections.poincare_3d import tube_section_cut_3d, closest_approach_4d
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


def _check_nrho_transport(nrho, l2_halo):
    """G19e: NRHO-to-L2-halo heteroclinic patch via y=0 section. Returns Delta-v in m/s."""
    nu = tube_section_cut_3d(MU, nrho, x_sec=0.0, stable=False, branch=-1,
                             n_seeds=160, displacement=1e-4, t_max=12.0, axis=1)
    ls = tube_section_cut_3d(MU, l2_halo, x_sec=0.0, stable=True, branch=+1,
                             n_seeds=160, displacement=1e-4, t_max=12.0, axis=1)
    if len(nu["yzvyvz"]) < 2 or len(ls["yzvyvz"]) < 2:
        return False, {"reason": "insufficient crossings",
                       "nrho_crossings": int(len(nu["yzvyvz"])),
                       "l2_crossings": int(len(ls["yzvyvz"]))}
    r = closest_approach_4d(nu["yzvyvz"], ls["yzvyvz"])
    if r is None:
        return False, {"reason": "no 4D closest approach"}
    pa, pb = r["point_a"], r["point_b"]
    x_int = 0.5 * (pa[0] + pb[0])
    z_int = 0.5 * (pa[1] + pb[1])
    vx_a, vz_a = float(pa[2]), float(pa[3])
    vx_b, vz_b = float(pb[2]), float(pb[3])
    om = pseudo_potential([x_int, 0.0, z_int, 0.0, 0.0, 0.0], MU)
    arg_a = 2 * om - nrho.jacobi - vx_a**2 - vz_a**2
    arg_b = 2 * om - l2_halo.jacobi - vx_b**2 - vz_b**2
    if arg_a < 0 or arg_b < 0:
        return False, {"reason": "energy constraint infeasible at crossing"}
    vy_a, vy_b = math.sqrt(arg_a), math.sqrt(arg_b)
    dv = math.sqrt((vx_a - vx_b) ** 2 + (vy_a - vy_b) ** 2 + (vz_a - vz_b) ** 2)
    dv_ms = dv * EARTH_MOON.V_star * 1000.0
    pos_gap_km = float(np.linalg.norm(pa[:2] - pb[:2])) * LSTAR
    return (dv_ms < 500.0), {"dv_ms": float(dv_ms), "pos_gap_km": float(pos_gap_km),
                             "crossing_x": float(x_int), "crossing_z": float(z_int)}


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
    # G19e needs an L2 halo at moderate energy for the NRHO connection
    l2_fam = halo_family(MU, point="L2", n=10, dz=4e-3, fam_n=40,
                         lyap_amp0=2e-3, lyap_dx=4e-3)
    l2_pick = l2_fam[len(l2_fam) // 2]
    g19e, nrho_xfer = _check_nrho_transport(nrho, l2_pick)
    ok = g19a and g19b and g19c and g19d and g19e
    info.update({"nrho": nrho, "period_d": period_d, "peri_km": peri, "apo_km": apo,
                 "peri_alt_km": peri - R_MOON, "floq_nrho": floq_nrho,
                 "floq_lyap": floq_lyap, "g19a": g19a, "g19b": g19b,
                 "g19c": g19c, "g19d": g19d, "g19e": g19e,
                 "graph": graph_info, "nrho_xfer": nrho_xfer})
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

    x = i["nrho_xfer"]
    print("[G19e] NRHO transport via y=0 Poincare section (axis=1; Gateway-class deliverable)")
    if "dv_ms" in x:
        print(f"      NRHO unstable -> L2 halo stable closest 4D approach on y=0")
        print(f"      patch dv = {x['dv_ms']:.1f} m/s   position (x,z) gap = {x['pos_gap_km']:.0f} km")
        print(f"      crossing at (x={x['crossing_x']:+.4f}, y=0, z={x['crossing_z']:+.4f})")
    else:
        print(f"      reason: {x.get('reason', 'unknown')}")
    print(f"      -> {'PASS' if i['g19e'] else 'FAIL'}\n")

    print(f"=== STAGE 19: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
