"""Stage 46 tests: proof-carrying route certificates."""
import copy

import numpy as np

from ariadne.certification import (CertificateThresholds, certify_transport_route,
                                   validate_certificate)
from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import pseudo_potential
from ariadne.transport_graph.graph import Edge, Node, TransportGraph


def _certifiable_graph():
    g = TransportGraph(EARTH_MOON.mu, EARTH_MOON.V_star)
    c = 3.12
    x, y, vy = 1.0 - EARTH_MOON.mu, 0.08, 0.04
    om = pseudo_potential([x, y, 0.0, 0.0, 0.0, 0.0], EARTH_MOON.mu)
    vx = float(np.sqrt(2.0 * om - c - vy * vy))
    s = [x, y, 0.0, vx, vy, 0.0]
    g.add_node(Node("A", "L1", c))
    g.add_node(Node("B", "L2", c))
    g.add_edge(Edge("A", "B", 0.0, meta={"pre": s, "post": s}))
    return g, ["A", "B"]


def test_certificate_hash_validates_and_detects_tamper():
    g, path = _certifiable_graph()
    cert = certify_transport_route(
        g, path, EARTH_MOON,
        thresholds=CertificateThresholds(bcr4bp_divergence_max_km=200_000.0),
        monte_carlo_samples=4,
    )
    assert validate_certificate(cert)["hash_ok"]
    tampered = copy.deepcopy(cert)
    tampered["path"] = ["A", "tampered"]
    assert not validate_certificate(tampered)["hash_ok"]


def test_certificate_fails_closed_without_patch_evidence():
    g = TransportGraph(EARTH_MOON.mu, EARTH_MOON.V_star)
    g.add_manual_node("A", 3.12)
    g.add_manual_node("B", 3.12)
    g.add_manual_edge("A", "B", 0.0)
    cert = certify_transport_route(g, ["A", "B"], EARTH_MOON, monte_carlo_samples=2)
    assert cert["status"] == "rejected"
    assert cert["rungs"]["cr3bp_patch_proof"]["status"] == "fail"
    assert "cr3bp_patch_proof" in validate_certificate(cert)["missing_or_failed_required"]


def test_de440_required_rejects_when_not_supplied():
    g, path = _certifiable_graph()
    cert = certify_transport_route(g, path, EARTH_MOON, require_de440=True,
                                   monte_carlo_samples=2)
    assert cert["status"] == "rejected"
    assert cert["rungs"]["de440_retarget"]["status"] == "not_run"
