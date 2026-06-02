import copy
import json
from dataclasses import asdict

from ariadne.interplanetary.navigator import MissionRoute, RouteEvent
from ariadne.proof.promotion import (
    PromotionEvidence,
    PromotionThresholds,
    load_routes_from_navigator_report,
    promote_route,
    promote_routes,
    write_promotion_report,
)


def _route(**overrides):
    base = dict(
        route_id="route_test",
        name="test Earth Mars",
        engine="direct_lambert",
        sequence=("EARTH", "MARS BARYCENTER"),
        target="MARS",
        fidelity="patched_conic_ephemeris",
        total_dv_ms=3600.0,
        tof_days=250.0,
        c3_km2_s2=9.0,
        arrival_vinf_kms=2.5,
        risk=0.2,
        feasible=True,
        assumptions=("heliocentric Lambert on real ephemerides",),
        validations=("finite launch C3", "finite arrival v-infinity"),
        events=(
            RouteEvent(
                body="EARTH",
                epoch_utc="2028-01-01T00:00:00",
                role="departure",
                coordinates_km=(1.0, 2.0, 3.0),
            ),
            RouteEvent(
                body="MARS BARYCENTER",
                epoch_utc="2028-09-07T00:00:00",
                role="arrival",
                coordinates_km=(4.0, 5.0, 6.0),
            ),
        ),
        certificate_hash="abc",
    )
    base.update(overrides)
    return MissionRoute(**base)


def _high_fidelity_evidence():
    return [
        PromotionEvidence(
            "nbody_replay",
            "pass",
            "unit-test",
            {"max_position_residual_km": 1.0, "max_velocity_residual_mps": 0.01},
            "nbody",
        ),
        PromotionEvidence(
            "covariance_envelope",
            "pass",
            "unit-test",
            {"position_3sigma_km": 100.0, "dv_3sigma_mps": 1.0},
            "cov",
        ),
        PromotionEvidence(
            "independent_crosscheck",
            "pass",
            "unit-test",
            {"position_delta_km": 1.0, "velocity_delta_mps": 0.01},
            "cross",
        ),
    ]


def test_promote_route_basic_rungs_pass_and_hash_validates():
    cert = promote_route(_route())

    assert cert.status == "promoted"
    assert cert.validate_hash()
    assert {r.rung: r.status for r in cert.rungs}["patched_conic_sanity"] == "pass"


def test_promote_route_rejects_impossible_costs():
    cert = promote_route(_route(total_dv_ms=1e9))

    assert cert.status == "rejected"
    sanity = {r.rung: r for r in cert.rungs}["patched_conic_sanity"]
    assert sanity.required
    assert sanity.status == "fail"
    assert "total_dv_ms" in sanity.message


def test_high_fidelity_requirement_fails_closed_without_evidence():
    cert = promote_route(_route(), require_high_fidelity=True)

    assert cert.status == "rejected"
    failed = [r.rung for r in cert.rungs if r.required and r.status != "pass"]
    assert failed == ["nbody_replay", "covariance_envelope", "independent_crosscheck"]


def test_high_fidelity_requirement_passes_with_bounded_evidence():
    cert = promote_route(_route(), require_high_fidelity=True,
                         external_evidence=_high_fidelity_evidence())

    assert cert.status == "promoted"
    assert cert.validate_hash()


def test_high_fidelity_evidence_rejects_bad_residual():
    evidence = _high_fidelity_evidence()
    evidence[0] = PromotionEvidence(
        "nbody_replay",
        "pass",
        "unit-test",
        {"max_position_residual_km": 1000.0, "max_velocity_residual_mps": 0.01},
        "nbody",
    )
    cert = promote_route(
        _route(),
        require_high_fidelity=True,
        external_evidence=evidence,
        thresholds=PromotionThresholds(max_nbody_position_residual_km=10.0),
    )

    assert cert.status == "rejected"
    rung = {r.rung: r for r in cert.rungs}["nbody_replay"]
    assert rung.status == "fail"
    assert "position residual" in rung.message


def test_promotion_report_writes_and_loads_routes(tmp_path):
    route = _route()
    navigator_report = tmp_path / "navigator_report.json"
    navigator_report.write_text(json.dumps({"routes": [asdict(route)]}), encoding="utf-8")

    routes = load_routes_from_navigator_report(navigator_report)
    report = promote_routes(routes)
    outputs = write_promotion_report(report, tmp_path / "promotion")

    assert report.status == "pass"
    assert report.validate_hash()
    assert "promotion_report.json" in outputs["json"]


def test_certificate_hash_detects_tamper():
    cert = promote_route(_route())
    tampered = copy.deepcopy(cert)
    object.__setattr__(tampered, "status", "promoted-even-more")

    assert cert.validate_hash()
    assert not tampered.validate_hash()
