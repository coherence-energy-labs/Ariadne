import copy
import json

import pytest

from ariadne.proof.closure import (
    ArtifactEvidence,
    ClosureLedger,
    ResidualSignal,
    SubsystemContract,
    build_closure_report,
    load_json_artifact,
    write_closure_report,
)


def test_closure_report_passes_and_hash_validates(tmp_path):
    metrics = {
        "passed": True,
        "certificate_hash": "abc123",
        "accuracy": 0.986,
        "ece": 0.0065,
    }
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text(json.dumps(metrics), encoding="utf-8")

    evidence = ArtifactEvidence.from_json_artifact(
        "discovery_mpc_500",
        "discovery_inference",
        "labelled_benchmark",
        artifact_path,
        required=True,
        metric_fields=("accuracy", "ece"),
    )
    report = build_closure_report(
        contracts=[
            SubsystemContract(
                "discovery_inference",
                required_kinds=("labelled_benchmark",),
                minimum_metrics={"accuracy": 0.9},
                maximum_metrics={"ece": 0.02},
            )
        ],
        evidence=[evidence],
    )

    assert report.status == "complete"
    assert report.readiness_score == 1.0
    assert report.validate_hash()
    outputs = write_closure_report(report, tmp_path / "closure")
    assert load_json_artifact(outputs["json"])["certificate_hash"] == report.certificate_hash
    assert "Ariadne Closure Report" in (tmp_path / "closure" / "closure_report.md").read_text()


def test_required_evidence_fails_closed_and_records_residual():
    evidence = ArtifactEvidence(
        artifact_id="navigator_missing",
        subsystem="solar_navigator",
        kind="navigator_benchmark",
        status="failed",
        required=True,
    )
    report = build_closure_report(
        contracts=[SubsystemContract("solar_navigator", required_kinds=("navigator_benchmark",))],
        evidence=[evidence],
    )

    assert report.status == "partial"
    assert report.critical_failures == 1
    assert report.readiness_score == 0.0
    assert report.residuals[0].residual_id == "navigator_missing:required_not_passing"


def test_metric_thresholds_fail_on_missing_or_low_values():
    report = build_closure_report(
        contracts=[
            SubsystemContract(
                "visuals",
                required_kinds=("figure_audit",),
                minimum_metrics={"png_valid_fraction": 1.0},
            )
        ],
        evidence=[
            ArtifactEvidence(
                artifact_id="figure_audit",
                subsystem="visuals",
                kind="figure_audit",
                status="pass",
                metrics={"png_valid_fraction": 0.75},
            )
        ],
    )

    failures = [g for g in report.gates if not g.passed]
    assert len(failures) == 1
    assert "does not satisfy" in failures[0].message


def test_residual_priority_order_is_severity_times_confidence():
    high_conf = ResidualSignal(
        "high_conf",
        "navigator",
        "model",
        severity=0.7,
        confidence=1.0,
        description="known broken promotion rung",
    )
    scarier_but_uncertain = ResidualSignal(
        "uncertain",
        "navigator",
        "model",
        severity=1.0,
        confidence=0.1,
        description="possible edge case without evidence",
    )
    report = build_closure_report(
        contracts=[],
        evidence=[],
        residuals=[scarier_but_uncertain, high_conf],
    )

    assert [r.residual_id for r in report.residuals] == ["high_conf", "uncertain"]


def test_duplicate_ids_and_invalid_residuals_are_rejected():
    ledger = ClosureLedger()
    ledger.add_contract(SubsystemContract("a"))
    with pytest.raises(ValueError):
        ledger.add_contract(SubsystemContract("a"))
    ledger.add_evidence(ArtifactEvidence("e", "a", "k"))
    with pytest.raises(ValueError):
        ledger.add_evidence(ArtifactEvidence("e", "a", "k"))
    with pytest.raises(ValueError):
        ledger.add_residual(ResidualSignal("bad", "a", "x", 1.5, 1.0, "bad"))


def test_report_hash_detects_tamper():
    report = build_closure_report(
        contracts=[SubsystemContract("a", required_kinds=("k",))],
        evidence=[ArtifactEvidence("e", "a", "k", status="pass")],
    )
    tampered = copy.deepcopy(report)
    object.__setattr__(tampered, "readiness_score", 0.0)
    assert report.validate_hash()
    assert not tampered.validate_hash()
