"""Stage 46 validation -- Certified Route Promotion.

This is the proof-carrying trajectory layer. A candidate route must now carry a
machine-checkable certificate: CR3BP patch proof, BCR4BP survivability,
deterministic robustness envelope, DE440 multiple-shooting promotion, and explicit
GMAT replay status. The certificate is canonical JSON with a SHA-256 payload hash.

G46a (certificate integrity)  - canonical replay hash validates and tampering is detected.
G46b (fail closed)            - a route without patch-state evidence is rejected.
G46c (physical promotion)     - an Earth-Moon L1->L2 heteroclinic is promoted through
                                CR3BP, BCR4BP, robustness, and DE440 retargeting.

Run:  PYTHONPATH=src python -m ariadne.validate.stage46
"""
from __future__ import annotations

import copy

import numpy as np

from ..certification import (CertificateThresholds, certify_heteroclinic_route,
                             certify_transport_route, graph_from_heteroclinic,
                             validate_certificate)
from ..connections.heteroclinic import find_heteroclinic
from ..data.constants import EARTH_MOON
from ..dynamics.cr3bp import pseudo_potential
from ..transport_graph.graph import Edge, Node, TransportGraph


MU = EARTH_MOON.mu


def _synthetic_good_graph():
    g = TransportGraph(MU, EARTH_MOON.V_star)
    c = 3.12
    x, y, vy = 1.0 - MU, 0.08, 0.04
    om = pseudo_potential([x, y, 0.0, 0.0, 0.0, 0.0], MU)
    vx = float(np.sqrt(2.0 * om - c - vy * vy))
    s = [x, y, 0.0, vx, vy, 0.0]
    g.add_node(Node("A", "L1", c))
    g.add_node(Node("B", "L2", c))
    g.add_edge(Edge("A", "B", 0.0, meta={"pre": s, "post": s}))
    return g, ["A", "B"]


def _synthetic_bad_graph():
    g = TransportGraph(MU, EARTH_MOON.V_star)
    g.add_manual_node("A", 3.12)
    g.add_manual_node("B", 3.12)
    g.add_manual_edge("A", "B", 0.0)
    return g, ["A", "B"]


def check() -> tuple[bool, dict]:
    thresholds = CertificateThresholds(
        bcr4bp_divergence_max_km=300_000.0,
        monte_carlo_sensitivity_max_km_per_ms=300_000.0,
        de440_total_dv_max_ms=10_000.0,
    )

    good_g, good_path = _synthetic_good_graph()
    good = certify_transport_route(good_g, good_path, EARTH_MOON, thresholds=thresholds,
                                   monte_carlo_samples=8)
    vgood = validate_certificate(good)
    tampered = copy.deepcopy(good)
    tampered["path"] = ["A", "C"]
    vtamp = validate_certificate(tampered)
    g46a = vgood["hash_ok"] and not vtamp["hash_ok"]

    bad_g, bad_path = _synthetic_bad_graph()
    bad = certify_transport_route(bad_g, bad_path, EARTH_MOON, thresholds=thresholds,
                                  monte_carlo_samples=4)
    vbad = validate_certificate(bad)
    g46b = bad["status"] == "rejected" and not vbad["ok"]

    conn = find_heteroclinic(MU, 3.15, "L1", "L2", n_seeds=120)
    if conn is None:
        return False, {"g46a": g46a, "g46b": g46b, "g46c": False,
                       "reason": "heteroclinic not found"}
    phys = certify_heteroclinic_route(conn, EARTH_MOON, thresholds=thresholds,
                                      require_de440=True, n_patch=10, t_leg=2.6,
                                      monte_carlo_samples=8)
    vphys = validate_certificate(phys)
    r = phys["rungs"]
    g46c = (phys["status"] == "certified" and vphys["ok"]
            and r["cr3bp_patch_proof"]["status"] == "pass"
            and r["bcr4bp_survivability"]["status"] == "pass"
            and r["robustness_envelope"]["status"] == "pass"
            and r["de440_retarget"]["status"] == "pass")

    g_h, path_h = graph_from_heteroclinic(conn, EARTH_MOON)
    return g46a and g46b and g46c, {
        "g46a": g46a, "g46b": g46b, "g46c": g46c,
        "good_status": good["status"], "bad_status": bad["status"],
        "route_id": phys["route_id"],
        "path": path_h,
        "cr3bp_dv_ms": r["cr3bp_patch_proof"]["total_dv_ms"],
        "bcr4bp_worst_km": r["bcr4bp_survivability"]["worst_divergence_km"],
        "robust_km_per_ms": r["robustness_envelope"]["worst_sensitivity_km_per_ms"],
        "de440_dv_ms": r["de440_retarget"]["total_dv_ms"],
        "de440_resid_km": r["de440_retarget"]["max_resid_km"],
        "gmat_status": r.get("gmat_replay", {}).get("status"),
        "hash_ok": vphys["hash_ok"],
        "cert_status": phys["status"],
        "n_edges": sum(len(v) for v in g_h.edges.values()),
    }


def main() -> int:
    print("=== Ariadne Stage 46 validation  (Certified Route Promotion) ===\n")
    ok, i = check()

    print("[G46a] Certificate integrity")
    print(f"      canonical SHA-256 validates; tamper detection active -> {'PASS' if i['g46a'] else 'FAIL'}\n")

    print("[G46b] Fail-closed behavior")
    print(f"      valid synthetic status={i.get('good_status')} ; missing-evidence status={i.get('bad_status')}")
    print(f"      -> {'PASS' if i['g46b'] else 'FAIL'}\n")

    print("[G46c] Physical route promotion: Earth-Moon L1->L2 heteroclinic")
    if i.get("g46c"):
        print(f"      path: {' -> '.join(i['path'])}   route_id={i['route_id'][:16]}...")
        print(f"      CR3BP patch Delta-v = {i['cr3bp_dv_ms']:.6f} m/s")
        print(f"      BCR4BP worst divergence = {i['bcr4bp_worst_km']:.0f} km")
        print(f"      robustness envelope = {i['robust_km_per_ms']:.1f} km per m/s")
        print(f"      DE440 retarget: Delta-v = {i['de440_dv_ms']:.1f} m/s, "
              f"max residual = {i['de440_resid_km']:.3e} km")
        print(f"      GMAT replay status = {i['gmat_status']} (explicitly recorded, not implied)")
        print(f"      certificate status={i['cert_status']} hash_ok={i['hash_ok']}")
    else:
        print(f"      reason: {i.get('reason', 'promotion gate failed')}")
    print(f"      -> {'PASS' if i.get('g46c') else 'FAIL'}\n")

    print(f"=== STAGE 46: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
