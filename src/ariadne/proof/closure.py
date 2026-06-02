"""Ariadne closure ledger and residual proof engine.

The ledger is the system-level audit surface for Ariadne. It does three jobs:

* records evidence artifacts from route certificates, navigator benchmarks,
  discovery benchmarks, validation runs, and visual products;
* evaluates subsystem contracts with fail-closed gate semantics;
* ranks residuals so the next engineering pass attacks the largest verified
  weakness rather than the loudest anecdote.

All public outputs are canonicalized and hash-stamped. A report can therefore
be compared across runs, stored in release artifacts, or used as an input to a
nightly calibration loop.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PASS_STATUSES = {"pass", "passed", "ok", "certified", "complete"}
SOFT_STATUSES = {"warning", "not_run", "not_available", "degraded", "unknown"}
FAIL_STATUSES = {"fail", "failed", "rejected", "error", "missing", "blocked"}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json(payload: Any) -> str:
    """Return stable compact JSON for certificates and replay hashes."""

    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(payload: Any) -> str:
    """SHA-256 over canonical JSON."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_json_artifact(path: str | Path) -> dict:
    """Load a JSON artifact as a dictionary and reject malformed evidence."""

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"artifact {p} must contain a JSON object")
    return data


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _norm_status(status: str | bool | None) -> str:
    if isinstance(status, bool):
        return "pass" if status else "fail"
    if status is None:
        return "unknown"
    text = str(status).strip().lower()
    if text in PASS_STATUSES:
        return "pass"
    if text in FAIL_STATUSES:
        return "fail"
    if text in SOFT_STATUSES:
        return text
    return text or "unknown"


@dataclass(frozen=True)
class ArtifactEvidence:
    """One external or generated proof artifact."""

    artifact_id: str
    subsystem: str
    kind: str
    path: str | None = None
    status: str = "unknown"
    certificate_hash: str | None = None
    content_hash: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    required: bool = False
    notes: tuple[str, ...] = ()

    @classmethod
    def from_json_artifact(
        cls,
        artifact_id: str,
        subsystem: str,
        kind: str,
        path: str | Path,
        *,
        required: bool = False,
        status_field: str = "passed",
        certificate_field: str = "certificate_hash",
        metric_fields: Iterable[str] = (),
        notes: Iterable[str] = (),
    ) -> "ArtifactEvidence":
        """Create evidence from a JSON object and hash the file bytes."""

        p = Path(path)
        payload = load_json_artifact(p)
        metrics = {name: payload.get(name) for name in metric_fields if name in payload}
        return cls(
            artifact_id=artifact_id,
            subsystem=subsystem,
            kind=kind,
            path=str(p),
            status=_norm_status(payload.get(status_field)),
            certificate_hash=payload.get(certificate_field),
            content_hash=_file_hash(p),
            metrics=metrics,
            required=required,
            notes=tuple(str(n) for n in notes),
        )

    @classmethod
    def from_payload(
        cls,
        artifact_id: str,
        subsystem: str,
        kind: str,
        payload: dict,
        *,
        required: bool = False,
        status: str | bool | None = None,
        metric_fields: Iterable[str] = (),
        notes: Iterable[str] = (),
    ) -> "ArtifactEvidence":
        """Create evidence from an in-memory JSON-like payload."""

        metrics = {name: payload.get(name) for name in metric_fields if name in payload}
        return cls(
            artifact_id=artifact_id,
            subsystem=subsystem,
            kind=kind,
            status=_norm_status(payload.get("status", status)),
            certificate_hash=payload.get("certificate_hash"),
            content_hash=stable_hash(payload),
            metrics=metrics,
            required=required,
            notes=tuple(str(n) for n in notes),
        )


@dataclass(frozen=True)
class GateResult:
    """One contract gate verdict."""

    gate_id: str
    subsystem: str
    status: str
    severity: str
    evidence_ids: tuple[str, ...] = ()
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return _norm_status(self.status) == "pass"


@dataclass(frozen=True)
class ResidualSignal:
    """A ranked model, data, performance, or UX residual."""

    residual_id: str
    subsystem: str
    category: str
    severity: float
    confidence: float
    description: str
    evidence_ids: tuple[str, ...] = ()
    recommended_action: str = ""

    def priority(self) -> float:
        sev = min(max(float(self.severity), 0.0), 1.0)
        conf = min(max(float(self.confidence), 0.0), 1.0)
        return sev * (0.25 + 0.75 * conf)


@dataclass(frozen=True)
class SubsystemContract:
    """Required evidence and thresholds for one subsystem."""

    subsystem: str
    required_kinds: tuple[str, ...] = ()
    minimum_metrics: dict[str, float] = field(default_factory=dict)
    maximum_metrics: dict[str, float] = field(default_factory=dict)
    soft_required_kinds: tuple[str, ...] = ()

    def evaluate(self, evidence: Iterable[ArtifactEvidence]) -> tuple[GateResult, ...]:
        rows = [e for e in evidence if e.subsystem == self.subsystem]
        by_kind: dict[str, list[ArtifactEvidence]] = {}
        for row in rows:
            by_kind.setdefault(row.kind, []).append(row)

        gates: list[GateResult] = []
        for kind in self.required_kinds:
            candidates = by_kind.get(kind, [])
            passing = [e for e in candidates if _norm_status(e.status) == "pass"]
            gates.append(GateResult(
                gate_id=f"{self.subsystem}:{kind}:required",
                subsystem=self.subsystem,
                status="pass" if passing else "fail",
                severity="critical",
                evidence_ids=tuple(e.artifact_id for e in candidates),
                message="" if passing else f"required evidence kind {kind!r} has no passing artifact",
            ))

        for kind in self.soft_required_kinds:
            candidates = by_kind.get(kind, [])
            passing = [e for e in candidates if _norm_status(e.status) == "pass"]
            gates.append(GateResult(
                gate_id=f"{self.subsystem}:{kind}:soft_required",
                subsystem=self.subsystem,
                status="pass" if passing else "warning",
                severity="major",
                evidence_ids=tuple(e.artifact_id for e in candidates),
                message="" if passing else f"soft-required evidence kind {kind!r} is not passing",
            ))

        gates.extend(self._metric_gates(rows, minimum=True))
        gates.extend(self._metric_gates(rows, minimum=False))
        return tuple(gates)

    def _metric_gates(self, rows: list[ArtifactEvidence], *, minimum: bool) -> list[GateResult]:
        thresholds = self.minimum_metrics if minimum else self.maximum_metrics
        op = ">=" if minimum else "<="
        out: list[GateResult] = []
        for metric, threshold in thresholds.items():
            values: list[tuple[str, float]] = []
            for row in rows:
                raw = row.metrics.get(metric)
                if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                    values.append((row.artifact_id, float(raw)))
            if not values:
                out.append(GateResult(
                    gate_id=f"{self.subsystem}:{metric}:{op}",
                    subsystem=self.subsystem,
                    status="fail",
                    severity="critical",
                    message=f"metric {metric!r} has no finite evidence",
                    metrics={"threshold": threshold},
                ))
                continue
            best = max(values, key=lambda item: item[1]) if minimum else min(values, key=lambda item: item[1])
            ok = best[1] >= threshold if minimum else best[1] <= threshold
            out.append(GateResult(
                gate_id=f"{self.subsystem}:{metric}:{op}",
                subsystem=self.subsystem,
                status="pass" if ok else "fail",
                severity="critical",
                evidence_ids=(best[0],),
                message="" if ok else f"metric {metric}={best[1]:.6g} does not satisfy {op} {threshold:.6g}",
                metrics={"observed": best[1], "threshold": threshold},
            ))
        return out


@dataclass
class ClosureLedger:
    """Mutable builder for Ariadne system closure reports."""

    contracts: list[SubsystemContract] = field(default_factory=list)
    evidence: list[ArtifactEvidence] = field(default_factory=list)
    residuals: list[ResidualSignal] = field(default_factory=list)

    def add_contract(self, contract: SubsystemContract) -> None:
        if any(c.subsystem == contract.subsystem for c in self.contracts):
            raise ValueError(f"duplicate subsystem contract {contract.subsystem!r}")
        self.contracts.append(contract)

    def add_evidence(self, evidence: ArtifactEvidence) -> None:
        if any(e.artifact_id == evidence.artifact_id for e in self.evidence):
            raise ValueError(f"duplicate evidence artifact_id {evidence.artifact_id!r}")
        if evidence.required and _norm_status(evidence.status) != "pass":
            self.residuals.append(ResidualSignal(
                residual_id=f"{evidence.artifact_id}:required_not_passing",
                subsystem=evidence.subsystem,
                category="evidence",
                severity=1.0,
                confidence=1.0,
                description=f"required {evidence.kind} evidence is {evidence.status}",
                evidence_ids=(evidence.artifact_id,),
                recommended_action="Regenerate or repair the required evidence artifact before claiming closure.",
            ))
        self.evidence.append(evidence)

    def add_residual(self, residual: ResidualSignal) -> None:
        if any(r.residual_id == residual.residual_id for r in self.residuals):
            raise ValueError(f"duplicate residual_id {residual.residual_id!r}")
        if not 0.0 <= residual.severity <= 1.0:
            raise ValueError("residual severity must be in [0, 1]")
        if not 0.0 <= residual.confidence <= 1.0:
            raise ValueError("residual confidence must be in [0, 1]")
        self.residuals.append(residual)

    def evaluate(self) -> "ClosureReport":
        gates: list[GateResult] = []
        for contract in self.contracts:
            gates.extend(contract.evaluate(self.evidence))

        critical_failures = [g for g in gates if g.severity == "critical" and not g.passed]
        warnings = [g for g in gates if g.severity != "critical" and not g.passed]
        ranked = tuple(sorted(self.residuals, key=lambda r: (-r.priority(), r.residual_id)))
        readiness = self._readiness_score(gates)
        blocking_residuals = [r for r in ranked if r.priority() >= 0.5]
        status = "complete" if readiness == 1.0 and not critical_failures and not blocking_residuals else "partial"
        payload = {
            "schema": "ariadne.closure_report.v1",
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": status,
            "readiness_score": readiness,
            "critical_failures": len(critical_failures),
            "warnings": len(warnings),
            "blocking_residuals": len(blocking_residuals),
            "contracts": self.contracts,
            "evidence": self.evidence,
            "gates": gates,
            "residuals": ranked,
        }
        report_hash = stable_hash({k: v for k, v in payload.items() if k != "created_utc"})
        return ClosureReport(certificate_hash=report_hash, **payload)

    @staticmethod
    def _readiness_score(gates: list[GateResult]) -> float:
        if not gates:
            return 0.0
        weights = {"critical": 1.0, "major": 0.35, "minor": 0.15}
        total = 0.0
        earned = 0.0
        for gate in gates:
            w = weights.get(gate.severity, 0.25)
            total += w
            if gate.passed:
                earned += w
        return round(earned / total, 6) if total else 0.0


@dataclass(frozen=True)
class ClosureReport:
    """Immutable closure report emitted by `ClosureLedger.evaluate`."""

    schema: str
    created_utc: str
    status: str
    readiness_score: float
    critical_failures: int
    warnings: int
    blocking_residuals: int
    contracts: list[SubsystemContract]
    evidence: list[ArtifactEvidence]
    gates: list[GateResult]
    residuals: tuple[ResidualSignal, ...]
    certificate_hash: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def validate_hash(self) -> bool:
        body = self.to_dict()
        expected = body.pop("certificate_hash")
        body.pop("created_utc", None)
        return expected == stable_hash(body)


def build_closure_report(
    contracts: Iterable[SubsystemContract],
    evidence: Iterable[ArtifactEvidence],
    residuals: Iterable[ResidualSignal] = (),
) -> ClosureReport:
    """Build a closure report from declarative contracts and evidence."""

    ledger = ClosureLedger()
    for contract in contracts:
        ledger.add_contract(contract)
    for row in evidence:
        ledger.add_evidence(row)
    for residual in residuals:
        ledger.add_residual(residual)
    return ledger.evaluate()


def _markdown(report: ClosureReport) -> str:
    rows = report.to_dict()
    lines = [
        "# Ariadne Closure Report",
        "",
        f"- status: {report.status}",
        f"- readiness_score: {report.readiness_score:.6f}",
        f"- critical_failures: {report.critical_failures}",
        f"- warnings: {report.warnings}",
        f"- blocking_residuals: {report.blocking_residuals}",
        f"- certificate_hash: `{report.certificate_hash}`",
        "",
        "## Gate Results",
        "",
        "| gate | subsystem | status | severity | message |",
        "|---|---|---:|---:|---|",
    ]
    for gate in rows["gates"]:
        message = str(gate.get("message", "")).replace("\n", " ")
        lines.append(
            f"| `{gate['gate_id']}` | {gate['subsystem']} | {gate['status']} | "
            f"{gate['severity']} | {message} |"
        )
    lines.extend(["", "## Highest-Priority Residuals", ""])
    if not rows["residuals"]:
        lines.append("No residuals recorded.")
    else:
        lines.extend(["| residual | subsystem | category | priority | action |", "|---|---|---|---:|---|"])
        for residual in rows["residuals"][:12]:
            priority = ResidualSignal(
                residual_id=residual["residual_id"],
                subsystem=residual["subsystem"],
                category=residual["category"],
                severity=float(residual["severity"]),
                confidence=float(residual["confidence"]),
                description=residual["description"],
                evidence_ids=tuple(residual.get("evidence_ids", ())),
                recommended_action=residual.get("recommended_action", ""),
            ).priority()
            action = str(residual.get("recommended_action", "")).replace("\n", " ")
            lines.append(
                f"| `{residual['residual_id']}` | {residual['subsystem']} | "
                f"{residual['category']} | {priority:.3f} | {action} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_closure_report(report: ClosureReport, outdir: str | Path) -> dict[str, str]:
    """Write canonical JSON plus reviewer-readable Markdown."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "closure_report.json"
    md_path = out / "closure_report.md"
    json_path.write_text(json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
