"""Residual DreamLab for Ariadne.

DreamLab turns closure residuals into a deterministic experiment queue. It is a
nightly calibration brain: mine proof reports, rank residuals, propose concrete
evidence-producing experiments, and hash the queue so follow-up runs can prove
what changed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .closure import stable_hash


@dataclass(frozen=True)
class DreamExperiment:
    experiment_id: str
    subsystem: str
    category: str
    priority: float
    command: str
    success_artifact: str
    expected_gate: str
    rationale: str


@dataclass(frozen=True)
class DreamRun:
    schema: str
    created_utc: str
    source_closure_hash: str
    experiment_count: int
    experiments: tuple[DreamExperiment, ...]
    status: str
    certificate_hash: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def validate_hash(self) -> bool:
        body = self.to_dict()
        expected = body.pop("certificate_hash")
        body.pop("created_utc", None)
        return expected == stable_hash(body)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _priority(residual: dict) -> float:
    sev = max(0.0, min(1.0, float(residual.get("severity", 0.0))))
    conf = max(0.0, min(1.0, float(residual.get("confidence", 0.0))))
    return round(sev * (0.25 + 0.75 * conf), 6)


def _experiment_for(residual: dict) -> DreamExperiment:
    rid = str(residual.get("residual_id", "unknown"))
    subsystem = str(residual.get("subsystem", "unknown"))
    category = str(residual.get("category", "unknown"))
    priority = _priority(residual)
    if rid == "route_multifidelity_promotion_ladder":
        return DreamExperiment(
            experiment_id="dream_route_full_strict_promotion",
            subsystem=subsystem,
            category=category,
            priority=priority,
            command=(
                "PYTHONPATH=src python scripts/promote_navigator_routes.py "
                "data/benchmarks/solar_navigator_benchmark/earth_mars_small/navigator_report.json "
                "--out-dir data/benchmarks/route_promotion_full_strict "
                "--generate-nbody-evidence --require-nbody-replay "
                "--generate-covariance-evidence --require-covariance-envelope "
                "--generate-crosscheck-evidence --require-independent-crosscheck"
            ),
            success_artifact="data/benchmarks/route_promotion_full_strict/promotion_report.json",
            expected_gate="trajectory_certification:route_certificate:soft_required",
            rationale="Promote direct routes through n-body, covariance, and independent integrator cross-check rungs.",
        )
    if rid == "reviewer_grade_visual_contract":
        return DreamExperiment(
            experiment_id="dream_visual_contract_audit",
            subsystem=subsystem,
            category=category,
            priority=priority,
            command="PYTHONPATH=src python scripts/build_closure_report.py --project-root . --out-dir data/benchmarks/closure",
            success_artifact="data/benchmarks/closure/closure_report.json",
            expected_gate="visuals:reviewer_visual_contract:required",
            rationale="Ensure figure and route-card artifacts carry machine-checkable visual semantics.",
        )
    if rid == "discovery_long_arc_nbody_default":
        command = "PYTHONPATH=src python -m pytest tests/test_mpc_ephemeris_nbody.py tests/test_iod_advanced.py -q"
        artifact = "pytest:nbody_long_arc"
    elif rid == "deep_field_rate_constrained_cross_night_linking":
        command = "PYTHONPATH=src python -m pytest tests/test_multi_night_linker.py tests/test_triplet_linker.py -q"
        artifact = "pytest:deep_field_linking"
    elif rid == "navigator_non_earth_origin_extension":
        command = "PYTHONPATH=src python -m pytest tests/test_solar_navigator.py -q"
        artifact = "pytest:non_earth_origin_regression"
    else:
        command = "PYTHONPATH=src python scripts/build_closure_report.py --project-root . --out-dir data/benchmarks/closure"
        artifact = "data/benchmarks/closure/closure_report.json"
    return DreamExperiment(
        experiment_id=f"dream_{rid}",
        subsystem=subsystem,
        category=category,
        priority=priority,
        command=command,
        success_artifact=artifact,
        expected_gate=f"{subsystem}:{category}",
        rationale=str(residual.get("recommended_action") or residual.get("description") or "Close residual with evidence."),
    )


def build_dream_run(closure_report: str | Path | dict) -> DreamRun:
    if isinstance(closure_report, dict):
        payload = closure_report
    else:
        payload = json.loads(Path(closure_report).read_text(encoding="utf-8"))
    residuals = payload.get("residuals") or []
    experiments = tuple(
        sorted((_experiment_for(r) for r in residuals), key=lambda e: (-e.priority, e.experiment_id))
    )
    body = {
        "schema": "ariadne.dream_run.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_closure_hash": payload.get("certificate_hash", ""),
        "experiment_count": len(experiments),
        "experiments": experiments,
        "status": "pass" if experiments else "idle",
    }
    cert_hash = stable_hash({k: v for k, v in body.items() if k != "created_utc"})
    return DreamRun(certificate_hash=cert_hash, **body)


def write_dream_run(run: DreamRun, outdir: str | Path) -> dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "dream_run.json"
    md_path = out / "dream_run.md"
    json_path.write_text(json.dumps(run.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Ariadne DreamLab Run",
        "",
        f"- status: {run.status}",
        f"- experiments: {run.experiment_count}",
        f"- source_closure_hash: `{run.source_closure_hash}`",
        f"- certificate_hash: `{run.certificate_hash}`",
        "",
        "| experiment | subsystem | priority | artifact |",
        "|---|---|---:|---|",
    ]
    for exp in run.experiments:
        lines.append(
            f"| `{exp.experiment_id}` | {exp.subsystem} | {exp.priority:.3f} | `{exp.success_artifact}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
