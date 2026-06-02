from dataclasses import asdict

import numpy as np

from ariadne.interplanetary.navigator import MissionRoute, RouteEvent
from ariadne.proof.high_fidelity import (
    covariance_envelope_evidence,
    independent_crosscheck_evidence,
    nbody_replay_evidence,
)
from ariadne.proof.promotion import promote_route


def _route():
    return MissionRoute(
        route_id="route_linear",
        name="linear test route",
        engine="direct_lambert",
        sequence=("EARTH", "MARS BARYCENTER"),
        target="MARS",
        fidelity="patched_conic_ephemeris",
        total_dv_ms=1000.0,
        tof_days=1.0,
        c3_km2_s2=1.0,
        arrival_vinf_kms=1.0,
        risk=0.1,
        feasible=True,
        assumptions=("test",),
        validations=("test",),
        events=(
            RouteEvent("EARTH", "2028-01-01T00:00:00", "departure", (1.0e8, 0.0, 0.0)),
            RouteEvent("MARS BARYCENTER", "2028-01-02T00:00:00", "arrival", (1.0001e8, 1000.0, 0.0)),
        ),
        raw={"et_dep": 0.0},
        certificate_hash="abc",
    )


def test_nbody_replay_evidence_can_feed_required_promotion(monkeypatch):
    import ariadne.proof.high_fidelity as hf

    def fake_lambert(r0, r1, tof_s, mu):
        v = (np.asarray(r1) - np.asarray(r0)) / tof_s
        return v, v

    def fake_propagate(r0, v0, et0, tof_s, perturbers):
        return np.concatenate([np.asarray(r0) + np.asarray(v0) * tof_s, np.asarray(v0)])

    monkeypatch.setattr(hf, "lambert", fake_lambert)
    monkeypatch.setattr(hf, "_propagate", fake_propagate)

    evidence = nbody_replay_evidence(asdict(_route()))
    cert = promote_route(
        _route(),
        external_evidence=(evidence,),
        required_external_rungs=("nbody_replay",),
    )

    assert evidence.status == "pass"
    assert evidence.metrics["max_position_residual_km"] == 0.0
    assert cert.status == "promoted"
    assert cert.validate_hash()


def test_required_nbody_replay_rejects_bad_evidence(monkeypatch):
    import ariadne.proof.high_fidelity as hf

    def fake_lambert(r0, r1, tof_s, mu):
        v = (np.asarray(r1) - np.asarray(r0)) / tof_s
        return v, v

    def fake_propagate(r0, v0, et0, tof_s, perturbers):
        drift = np.array([1000.0, 0.0, 0.0])
        return np.concatenate([np.asarray(r0) + np.asarray(v0) * tof_s + drift, np.asarray(v0)])

    monkeypatch.setattr(hf, "lambert", fake_lambert)
    monkeypatch.setattr(hf, "_propagate", fake_propagate)

    evidence = nbody_replay_evidence(asdict(_route()), max_iter=0)
    cert = promote_route(
        _route(),
        external_evidence=(evidence,),
        required_external_rungs=("nbody_replay",),
    )

    assert evidence.status == "fail"
    assert cert.status == "rejected"


def test_covariance_envelope_evidence_can_feed_required_promotion(monkeypatch):
    import ariadne.proof.high_fidelity as hf

    def fake_lambert(r0, r1, tof_s, mu):
        v = (np.asarray(r1) - np.asarray(r0)) / tof_s
        return v, v

    def fake_propagate(r0, v0, et0, tof_s, perturbers):
        return np.concatenate([np.asarray(r0) + np.asarray(v0) * tof_s, np.asarray(v0)])

    monkeypatch.setattr(hf, "lambert", fake_lambert)
    monkeypatch.setattr(hf, "_propagate", fake_propagate)

    evidence = covariance_envelope_evidence(
        asdict(_route()),
        sigma_position_km=0.001,
        sigma_velocity_mps=0.001,
        max_position_3sigma_km=1.0,
    )
    cert = promote_route(
        _route(),
        external_evidence=(evidence,),
        required_external_rungs=("covariance_envelope",),
    )

    assert evidence.status == "pass"
    assert evidence.metrics["sigma_point_count"] == 12
    assert cert.status == "promoted"


def test_covariance_envelope_rejects_large_dispersion(monkeypatch):
    import ariadne.proof.high_fidelity as hf

    def fake_lambert(r0, r1, tof_s, mu):
        v = (np.asarray(r1) - np.asarray(r0)) / tof_s
        return v, v

    def fake_propagate(r0, v0, et0, tof_s, perturbers):
        return np.concatenate([np.asarray(r0) + np.asarray(v0) * tof_s, np.asarray(v0)])

    monkeypatch.setattr(hf, "lambert", fake_lambert)
    monkeypatch.setattr(hf, "_propagate", fake_propagate)

    evidence = covariance_envelope_evidence(
        asdict(_route()),
        sigma_position_km=1.0,
        sigma_velocity_mps=10.0,
        max_position_3sigma_km=0.1,
        max_dv_3sigma_mps=1.0,
    )
    cert = promote_route(
        _route(),
        external_evidence=(evidence,),
        required_external_rungs=("covariance_envelope",),
    )

    assert evidence.status == "fail"
    assert cert.status == "rejected"


def test_independent_crosscheck_evidence_can_feed_required_promotion(monkeypatch):
    import ariadne.proof.high_fidelity as hf

    def fake_lambert(r0, r1, tof_s, mu):
        v = (np.asarray(r1) - np.asarray(r0)) / tof_s
        return v, v

    def fake_propagate(r0, v0, et0, tof_s, perturbers):
        return np.concatenate([np.asarray(r0) + np.asarray(v0) * tof_s, np.asarray(v0)])

    monkeypatch.setattr(hf, "lambert", fake_lambert)
    monkeypatch.setattr(hf, "_propagate", fake_propagate)
    monkeypatch.setattr(hf, "_propagate_independent", fake_propagate)

    evidence = independent_crosscheck_evidence(asdict(_route()))
    cert = promote_route(
        _route(),
        external_evidence=(evidence,),
        required_external_rungs=("independent_crosscheck",),
    )

    assert evidence.status == "pass"
    assert evidence.metrics["position_delta_km"] == 0.0
    assert cert.status == "promoted"


def test_independent_crosscheck_rejects_integrator_disagreement(monkeypatch):
    import ariadne.proof.high_fidelity as hf

    def fake_lambert(r0, r1, tof_s, mu):
        v = (np.asarray(r1) - np.asarray(r0)) / tof_s
        return v, v

    def fake_propagate(r0, v0, et0, tof_s, perturbers):
        return np.concatenate([np.asarray(r0) + np.asarray(v0) * tof_s, np.asarray(v0)])

    def fake_independent(r0, v0, et0, tof_s, perturbers):
        return np.concatenate([
            np.asarray(r0) + np.asarray(v0) * tof_s + np.array([100.0, 0.0, 0.0]),
            np.asarray(v0),
        ])

    monkeypatch.setattr(hf, "lambert", fake_lambert)
    monkeypatch.setattr(hf, "_propagate", fake_propagate)
    monkeypatch.setattr(hf, "_propagate_independent", fake_independent)

    evidence = independent_crosscheck_evidence(asdict(_route()), max_position_delta_km=1.0)
    cert = promote_route(
        _route(),
        external_evidence=(evidence,),
        required_external_rungs=("independent_crosscheck",),
    )

    assert evidence.status == "fail"
    assert cert.status == "rejected"
