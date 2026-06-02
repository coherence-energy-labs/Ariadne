"""Multi-fidelity promotion certificates for mission routes.

This module sits above the navigator and below final mission design. A route can
be promising, mathematically sane, and still not flight-grade. Promotion makes
that boundary explicit by checking each rung, recording pass/fail evidence, and
hashing the result.

The first rungs validate route-card integrity, patched-conic physical sanity,
event chronology, and ephemeris coordinate evidence. Higher rungs accept real
evidence produced by dedicated solvers: n-body replay, covariance dispersion,
independent tool cross-checks, and visual specification audits. If a rung is
required and the evidence is missing, promotion fails closed.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .closure import stable_hash


@dataclass(frozen=True)
class PromotionThresholds:
    """Numeric limits used by the route promotion gates."""

    max_total_dv_ms: float = 80_000.0
    max_launch_c3_km2_s2: float = 250.0
    max_arrival_vinf_kms: float = 35.0
    max_risk: float = 0.95
    min_tof_days: float = 1.0
    max_tof_days: float = 30_000.0
    max_nbody_position_residual_km: float = 10.0
    max_nbody_velocity_residual_mps: float = 0.05
    max_nbody_retarget_correction_mps: float = 500.0
    max_covariance_position_3sigma_km: float = 1_000.0
    max_covariance_dv_3sigma_mps: float = 10.0
    max_crosscheck_position_delta_km: float = 10.0
    max_crosscheck_velocity_delta_mps: float = 0.05


@dataclass(frozen=True)
class PromotionEvidence:
    """External evidence for a high-fidelity promotion rung."""

    rung: str
    status: str
    source: str
    metrics: dict[str, Any] = field(default_factory=dict)
    certificate_hash: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionRung:
    """One route promotion rung verdict."""

    rung: str
    status: str
    required: bool
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence_hash: str | None = None


@dataclass(frozen=True)
class RoutePromotionCertificate:
    """Full promotion certificate for one route."""

    schema: str
    created_utc: str
    route_id: str
    route_name: str
    target: str
    sequence: tuple[str, ...]
    status: str
    required_rungs: tuple[str, ...]
    rungs: tuple[PromotionRung, ...]
    thresholds: PromotionThresholds
    route_fingerprint: str
    certificate_hash: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def validate_hash(self) -> bool:
        body = self.to_dict()
        expected = body.pop("certificate_hash")
        body.pop("created_utc", None)
        return expected == stable_hash(body)


@dataclass(frozen=True)
class PromotionReport:
    """Promotion bundle for multiple navigator routes."""

    schema: str
    created_utc: str
    route_count: int
    promoted_count: int
    rejected_count: int
    certificates: tuple[RoutePromotionCertificate, ...]
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _route_dict(route: Any) -> dict[str, Any]:
    if hasattr(route, "__dataclass_fields__"):
        return _jsonable(asdict(route))
    if isinstance(route, dict):
        return copy.deepcopy(route)
    raise TypeError("route must be a MissionRoute dataclass or a route dictionary")


def _events(route: dict) -> list[dict]:
    return [e for e in route.get("events", ()) if isinstance(e, dict)]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _status_ok(status: str) -> bool:
    return str(status).lower() in {"pass", "passed", "ok", "certified", "complete"}


def _rung(name: str, status: str, required: bool, message: str = "",
          metrics: dict[str, Any] | None = None, evidence_hash: str | None = None) -> PromotionRung:
    return PromotionRung(
        rung=name,
        status=status,
        required=required,
        message=message,
        metrics=metrics or {},
        evidence_hash=evidence_hash,
    )


def _route_card_integrity(route: dict, required: bool) -> PromotionRung:
    failures = []
    sequence = route.get("sequence") or ()
    if len(sequence) < 2:
        failures.append("sequence must contain at least origin and target")
    for key in ("route_id", "name", "engine", "target", "fidelity", "certificate_hash"):
        if not route.get(key):
            failures.append(f"missing {key}")
    if not route.get("assumptions"):
        failures.append("route card has no assumptions")
    if not route.get("validations"):
        failures.append("route card has no validations")
    return _rung(
        "route_card_integrity",
        "fail" if failures else "pass",
        required,
        "; ".join(failures),
        {"sequence_length": len(sequence), "assumptions": len(route.get("assumptions") or ()),
         "validations": len(route.get("validations") or ())},
    )


def _patched_conic_sanity(route: dict, thresholds: PromotionThresholds, required: bool) -> PromotionRung:
    failures = []
    total_dv = route.get("total_dv_ms")
    tof = route.get("tof_days")
    c3 = route.get("c3_km2_s2")
    vinf = route.get("arrival_vinf_kms")
    risk = route.get("risk")
    if not _finite(total_dv) or float(total_dv) < 0.0 or float(total_dv) > thresholds.max_total_dv_ms:
        failures.append("total_dv_ms outside threshold")
    if not _finite(tof) or not thresholds.min_tof_days <= float(tof) <= thresholds.max_tof_days:
        failures.append("tof_days outside threshold")
    if c3 is not None and (not _finite(c3) or float(c3) < 0.0 or float(c3) > thresholds.max_launch_c3_km2_s2):
        failures.append("launch C3 outside threshold")
    if vinf is not None and (not _finite(vinf) or float(vinf) < 0.0 or float(vinf) > thresholds.max_arrival_vinf_kms):
        failures.append("arrival v-infinity outside threshold")
    if risk is not None and (not _finite(risk) or not 0.0 <= float(risk) <= thresholds.max_risk):
        failures.append("risk outside threshold")
    if route.get("feasible") is False:
        failures.append("route is marked infeasible")
    return _rung(
        "patched_conic_sanity",
        "fail" if failures else "pass",
        required,
        "; ".join(failures),
        {"total_dv_ms": total_dv, "tof_days": tof, "c3_km2_s2": c3,
         "arrival_vinf_kms": vinf, "risk": risk},
    )


def _event_chronology(route: dict, required: bool) -> PromotionRung:
    events = _events(route)
    if not events:
        return _rung("event_chronology", "warning", required, "no route events supplied")
    epochs = [e.get("epoch_utc") for e in events]
    missing = [i for i, t in enumerate(epochs) if not t]
    ordered = epochs == sorted(epochs)
    failures = []
    if missing:
        failures.append(f"missing epochs at indices {missing}")
    if not ordered:
        failures.append("events are not chronological")
    return _rung(
        "event_chronology",
        "fail" if failures else "pass",
        required,
        "; ".join(failures),
        {"event_count": len(events)},
    )


def _ephemeris_coordinate_evidence(route: dict, required: bool) -> PromotionRung:
    events = _events(route)
    if not events:
        return _rung("ephemeris_coordinate_evidence", "fail" if required else "warning", required,
                     "no route events supplied")
    missing = []
    for i, event in enumerate(events):
        coords = event.get("coordinates_km")
        if not isinstance(coords, (list, tuple)) or len(coords) != 3 or not all(_finite(x) for x in coords):
            missing.append(i)
    return _rung(
        "ephemeris_coordinate_evidence",
        "fail" if missing else "pass",
        required,
        "" if not missing else f"events missing finite 3D coordinates: {missing}",
        {"event_count": len(events), "coordinate_count": len(events) - len(missing)},
    )


def _external_rung(
    rung: str,
    evidence: dict[str, PromotionEvidence],
    required: bool,
    thresholds: PromotionThresholds,
) -> PromotionRung:
    row = evidence.get(rung)
    if row is None:
        return _rung(rung, "not_run" if required else "not_available", required, "no evidence supplied")
    metrics = dict(row.metrics)
    failures = []
    if not _status_ok(row.status):
        failures.append(f"evidence status is {row.status}")
    if rung == "nbody_replay":
        pos = metrics.get("max_position_residual_km")
        vel = metrics.get("max_velocity_residual_mps")
        correction = metrics.get("retarget_correction_dv_mps")
        if not _finite(pos) or float(pos) > thresholds.max_nbody_position_residual_km:
            failures.append("n-body position residual outside threshold")
        if vel is not None and (not _finite(vel) or float(vel) > thresholds.max_nbody_velocity_residual_mps):
            failures.append("n-body velocity residual outside threshold")
        if correction is not None and (
            not _finite(correction)
            or float(correction) > thresholds.max_nbody_retarget_correction_mps
        ):
            failures.append("n-body retarget correction outside threshold")
        if vel is None and correction is None:
            failures.append("n-body replay supplied no velocity residual or correction metric")
    elif rung == "covariance_envelope":
        pos = metrics.get("position_3sigma_km")
        dv = metrics.get("dv_3sigma_mps")
        if not _finite(pos) or float(pos) > thresholds.max_covariance_position_3sigma_km:
            failures.append("position covariance outside threshold")
        if not _finite(dv) or float(dv) > thresholds.max_covariance_dv_3sigma_mps:
            failures.append("delta-v covariance outside threshold")
    elif rung == "independent_crosscheck":
        pos = metrics.get("position_delta_km")
        vel = metrics.get("velocity_delta_mps")
        if not _finite(pos) or float(pos) > thresholds.max_crosscheck_position_delta_km:
            failures.append("cross-check position delta outside threshold")
        if not _finite(vel) or float(vel) > thresholds.max_crosscheck_velocity_delta_mps:
            failures.append("cross-check velocity delta outside threshold")
    status = "fail" if failures else "pass"
    return _rung(rung, status, required, "; ".join(failures), metrics, stable_hash(row))


def promote_route(
    route: Any,
    *,
    thresholds: PromotionThresholds | None = None,
    external_evidence: Iterable[PromotionEvidence] = (),
    require_high_fidelity: bool = False,
    required_external_rungs: Iterable[str] = (),
    require_ephemeris_coordinates: bool = True,
) -> RoutePromotionCertificate:
    """Promote one route through the configured fidelity ladder."""

    thresholds = thresholds or PromotionThresholds()
    r = _route_dict(route)
    evidence = {row.rung: row for row in external_evidence}
    required = ["route_card_integrity", "patched_conic_sanity", "event_chronology"]
    if require_ephemeris_coordinates:
        required.append("ephemeris_coordinate_evidence")
    if require_high_fidelity:
        required.extend(["nbody_replay", "covariance_envelope", "independent_crosscheck"])
    for rung in required_external_rungs:
        if rung not in required:
            required.append(str(rung))

    rungs = [
        _route_card_integrity(r, "route_card_integrity" in required),
        _patched_conic_sanity(r, thresholds, "patched_conic_sanity" in required),
        _event_chronology(r, "event_chronology" in required),
        _ephemeris_coordinate_evidence(r, "ephemeris_coordinate_evidence" in required),
        _external_rung("nbody_replay", evidence, "nbody_replay" in required, thresholds),
        _external_rung("covariance_envelope", evidence, "covariance_envelope" in required, thresholds),
        _external_rung("independent_crosscheck", evidence, "independent_crosscheck" in required, thresholds),
    ]
    failed_required = [row.rung for row in rungs if row.required and row.status != "pass"]
    status = "promoted" if not failed_required else "rejected"
    payload = {
        "schema": "ariadne.route_promotion_certificate.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "route_id": r.get("route_id", ""),
        "route_name": r.get("name", ""),
        "target": r.get("target", ""),
        "sequence": tuple(r.get("sequence") or ()),
        "status": status,
        "required_rungs": tuple(required),
        "rungs": tuple(rungs),
        "thresholds": thresholds,
        "route_fingerprint": stable_hash(r),
    }
    cert_hash = stable_hash({k: v for k, v in payload.items() if k != "created_utc"})
    return RoutePromotionCertificate(certificate_hash=cert_hash, **payload)


def promote_routes(
    routes: Iterable[Any],
    *,
    thresholds: PromotionThresholds | None = None,
    external_evidence_by_route: dict[str, Iterable[PromotionEvidence]] | None = None,
    require_high_fidelity: bool = False,
    required_external_rungs: Iterable[str] = (),
    require_ephemeris_coordinates: bool = True,
) -> PromotionReport:
    """Promote a set of routes and emit a bundle certificate."""

    evidence_by_route = external_evidence_by_route or {}
    certs = []
    for route in routes:
        r = _route_dict(route)
        rid = str(r.get("route_id", ""))
        certs.append(promote_route(
            r,
            thresholds=thresholds,
            external_evidence=evidence_by_route.get(rid, ()),
            require_high_fidelity=require_high_fidelity,
            required_external_rungs=required_external_rungs,
            require_ephemeris_coordinates=require_ephemeris_coordinates,
        ))
    promoted = [c for c in certs if c.status == "promoted"]
    payload = {
        "schema": "ariadne.route_promotion_report.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "route_count": len(certs),
        "promoted_count": len(promoted),
        "rejected_count": len(certs) - len(promoted),
        "certificates": tuple(certs),
        "status": "pass" if certs and len(promoted) == len(certs) else "fail",
    }
    cert_hash = stable_hash({k: v for k, v in payload.items() if k != "created_utc"})
    return PromotionReport(certificate_hash=cert_hash, **payload)


def write_promotion_report(report: PromotionReport, outdir: str | Path) -> dict[str, str]:
    """Write route promotion JSON and concise Markdown summary."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "promotion_report.json"
    md_path = out / "promotion_report.md"
    json_path.write_text(json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Ariadne Route Promotion Report",
        "",
        f"- status: {report.status}",
        f"- routes: {report.route_count}",
        f"- promoted: {report.promoted_count}",
        f"- rejected: {report.rejected_count}",
        f"- certificate_hash: `{report.certificate_hash}`",
        "",
        "| route | target | status | failed required rungs |",
        "|---|---|---:|---|",
    ]
    for cert in report.certificates:
        failed = [r.rung for r in cert.rungs if r.required and r.status != "pass"]
        lines.append(
            f"| `{cert.route_id}` | {cert.target} | {cert.status} | "
            f"{', '.join(failed) if failed else ''} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def load_routes_from_navigator_report(path: str | Path) -> tuple[dict, ...]:
    """Read route dictionaries from a navigator report JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    routes = payload.get("routes")
    if not isinstance(routes, list):
        raise ValueError("navigator report JSON must contain a routes list")
    return tuple(route for route in routes if isinstance(route, dict))
