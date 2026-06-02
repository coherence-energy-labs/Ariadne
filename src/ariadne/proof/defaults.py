"""Default Ariadne closure contracts and artifact collection."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from .closure import (
    ArtifactEvidence,
    ResidualSignal,
    SubsystemContract,
    build_closure_report,
)
from .visuals import navigator_visual_contract_evidence


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as fh:
        header = fh.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def audit_png_directory(figures_dir: str | Path, *, artifact_id: str = "docs_figures_png_audit") -> ArtifactEvidence:
    """Validate PNG signatures and dimensions for a figure directory."""

    root = Path(figures_dir)
    pngs = sorted(root.glob("*.png"))
    valid = []
    invalid = []
    min_width = None
    min_height = None
    for path in pngs:
        dims = _png_dimensions(path)
        if dims is None:
            invalid.append(path.name)
            continue
        valid.append(path.name)
        min_width = dims[0] if min_width is None else min(min_width, dims[0])
        min_height = dims[1] if min_height is None else min(min_height, dims[1])
    total = len(pngs)
    fraction = 1.0 if total == 0 else len(valid) / total
    metrics = {
        "n_png": total,
        "n_valid_png": len(valid),
        "png_valid_fraction": fraction,
        "min_width_px": min_width or 0,
        "min_height_px": min_height or 0,
        "invalid_png": invalid,
    }
    status = total > 0 and fraction == 1.0
    payload = {"status": status, "metrics": metrics}
    return ArtifactEvidence(
        artifact_id=artifact_id,
        subsystem="visuals",
        kind="figure_audit",
        path=str(root),
        status="pass" if status else "fail",
        content_hash=None,
        metrics=metrics,
        required=True,
        notes=("PNG signature and IHDR dimension audit",),
        certificate_hash=None,
    )


def default_contracts() -> tuple[SubsystemContract, ...]:
    """Closure gates for the current Ariadne system surface."""

    return (
        SubsystemContract(
            "dream_calibration",
            required_kinds=("dream_run",),
            minimum_metrics={"experiment_count": 1.0},
        ),
        SubsystemContract(
            "discovery_inference",
            required_kinds=("labelled_benchmark",),
            minimum_metrics={"accuracy": 0.9, "safe_accuracy": 0.95, "macro_f1": 0.9},
            maximum_metrics={"ece": 0.05},
        ),
        SubsystemContract(
            "discovery_robustness",
            required_kinds=("adversarial_benchmark",),
            minimum_metrics={"safe_accuracy": 0.94},
        ),
        SubsystemContract(
            "solar_navigator",
            required_kinds=("navigator_benchmark",),
        ),
        SubsystemContract(
            "visuals",
            required_kinds=("figure_audit", "reviewer_visual_contract"),
            minimum_metrics={"png_valid_fraction": 1.0, "n_png": 10.0, "semantic_route_fraction": 1.0},
        ),
        SubsystemContract(
            "trajectory_certification",
            soft_required_kinds=("route_certificate",),
        ),
        SubsystemContract(
            "transport_search",
            required_kinds=("admissibility_benchmark",),
            minimum_metrics={"speedup_astar_vs_brute": 1.0},
        ),
        SubsystemContract(
            "artifact_integrity",
            required_kinds=("artifact_manifest",),
            minimum_metrics={"file_count": 10.0},
        ),
    )


def known_residuals() -> tuple[ResidualSignal, ...]:
    """Repo-backed residuals that should remain visible until closed by evidence."""

    return (
        ResidualSignal(
            residual_id="navigator_non_earth_origin_extension",
            subsystem="solar_navigator",
            category="model_scope",
            severity=0.45,
            confidence=0.9,
            description="Direct non-Earth origins are supported; non-Earth gravity-assist templates still need expansion.",
            recommended_action=(
                "Add non-Earth gravity-assist templates and benchmark coverage for non-Earth departure bodies."
            ),
        ),
        ResidualSignal(
            residual_id="route_multifidelity_promotion_ladder",
            subsystem="trajectory_certification",
            category="flight_grade_proof",
            severity=0.48,
            confidence=0.95,
            description=(
                "Direct Earth-Mars routes now promote through n-body, covariance, and independent integrator cross-checks; remaining ladder work is moon-system and low-thrust expansion."
            ),
            recommended_action=(
                "Extend strict promotion to moon-system ephemerides, low-thrust/DSM arcs, and installed GMAT/Monte external replays."
            ),
        ),
        ResidualSignal(
            residual_id="discovery_long_arc_nbody_default",
            subsystem="discovery_inference",
            category="fidelity_default",
            severity=0.35,
            confidence=0.9,
            description="Long-arc IOD now auto-promotes to N-body LM; remaining work is broader real-corpus burn-down.",
            recommended_action=(
                "Expand real-corpus regression coverage for long-arc N-body fits and monitor fallback rates."
            ),
        ),
        ResidualSignal(
            residual_id="deep_field_rate_constrained_cross_night_linking",
            subsystem="discovery_pipeline",
            category="operational_scaling",
            severity=0.38,
            confidence=0.9,
            description="Rate-constrained cross-night linking exists; remaining work is deep near-ecliptic real-field scale validation.",
            recommended_action=(
                "Benchmark the rate-constrained linker on near-ecliptic deep cadence data and track runtime/false-link curves."
            ),
        ),
        ResidualSignal(
            residual_id="reviewer_grade_visual_contract",
            subsystem="visuals",
            category="explainability",
            severity=0.30,
            confidence=0.95,
            description="Reviewer visual contract now gates navigator PNGs and route-card semantics; remaining work is richer pixel-level/metadata scoring.",
            recommended_action=(
                "Add optional OCR/metadata sidecar scoring for units, labels, uncertainty overlays, and scale bars."
            ),
        ),
    )


def collect_default_evidence(project_root: str | Path) -> tuple[ArtifactEvidence, ...]:
    """Collect existing Ariadne evidence artifacts without mutating the project."""

    root = Path(project_root)
    evidence: list[ArtifactEvidence] = []

    dream = root / "data" / "benchmarks" / "dream_lab" / "dream_run.json"
    if dream.exists():
        evidence.append(ArtifactEvidence.from_json_artifact(
            "dream_lab_run",
            "dream_calibration",
            "dream_run",
            dream,
            required=True,
            status_field="status",
            metric_fields=("experiment_count",),
        ))

    navigator_summary = root / "data" / "benchmarks" / "solar_navigator_benchmark" / "benchmark_summary.json"
    if navigator_summary.exists():
        evidence.append(ArtifactEvidence.from_json_artifact(
            "solar_navigator_benchmark",
            "solar_navigator",
            "navigator_benchmark",
            navigator_summary,
            required=True,
            status_field="passed",
            metric_fields=("elapsed_s",),
        ))

    discovery_metrics = root / "data" / "benchmarks" / "real_corpus_mpc_500" / "benchmark" / "metrics.json"
    if discovery_metrics.exists():
        payload = json.loads(discovery_metrics.read_text(encoding="utf-8"))
        reliability = payload.get("reliability")
        if isinstance(reliability, dict) and "ece" in reliability and "ece" not in payload:
            payload = {**payload, "ece": reliability["ece"]}
        evidence.append(ArtifactEvidence.from_payload(
            "discovery_real_corpus_mpc_500",
            "discovery_inference",
            "labelled_benchmark",
            payload,
            required=True,
            status=payload.get("accuracy", 0.0) >= 0.9,
            metric_fields=("accuracy", "safe_accuracy", "macro_f1", "ece", "n"),
        ))

    adversarial_metrics = root / "data" / "benchmarks" / "inference_adversarial_builtin" / "metrics.json"
    if adversarial_metrics.exists():
        payload = json.loads(adversarial_metrics.read_text(encoding="utf-8"))
        evidence.append(ArtifactEvidence.from_payload(
            "discovery_adversarial_builtin",
            "discovery_robustness",
            "adversarial_benchmark",
            payload,
            required=True,
            status=payload.get("safe_accuracy", 0.0) >= 0.94,
            metric_fields=("accuracy", "safe_accuracy", "macro_f1", "n"),
        ))

    figures = root / "docs" / "figures"
    if figures.exists():
        evidence.append(audit_png_directory(figures))
    navigator_benchmark = root / "data" / "benchmarks" / "solar_navigator_benchmark"
    if navigator_benchmark.exists():
        evidence.append(navigator_visual_contract_evidence(navigator_benchmark))

    promotion = root / "data" / "benchmarks" / "route_promotion_full_strict" / "promotion_report.json"
    if not promotion.exists():
        promotion = root / "data" / "benchmarks" / "route_promotion_nbody_covariance" / "promotion_report.json"
    if not promotion.exists():
        promotion = root / "data" / "benchmarks" / "route_promotion_nbody" / "promotion_report.json"
    if not promotion.exists():
        promotion = root / "data" / "benchmarks" / "route_promotion" / "promotion_report.json"
    if promotion.exists():
        evidence.append(ArtifactEvidence.from_json_artifact(
            "route_promotion_report",
            "trajectory_certification",
            "route_certificate",
            promotion,
            required=False,
            status_field="status",
            metric_fields=("route_count", "promoted_count", "rejected_count"),
        ))

    transport = root / "data" / "benchmarks" / "transport_admissibility" / "metrics.json"
    if transport.exists():
        payload = json.loads(transport.read_text(encoding="utf-8"))
        evidence.append(ArtifactEvidence.from_payload(
            "transport_admissibility",
            "transport_search",
            "admissibility_benchmark",
            payload,
            required=True,
            status=payload.get("passed", False),
            metric_fields=("speedup_astar_vs_brute", "astar_expansions", "brute_expansions"),
        ))

    manifest = root / "data" / "benchmarks" / "artifact_integrity" / "artifact_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        evidence.append(ArtifactEvidence.from_payload(
            "artifact_integrity_manifest",
            "artifact_integrity",
            "artifact_manifest",
            payload,
            required=True,
            status=payload.get("file_count", 0) >= 10,
            metric_fields=("file_count", "total_size_bytes"),
        ))

    return tuple(evidence)


def build_default_ariadne_closure(project_root: str | Path):
    """Build the default closure report for the current Ariadne checkout."""

    return build_closure_report(
        contracts=default_contracts(),
        evidence=collect_default_evidence(project_root),
        residuals=known_residuals(),
    )
