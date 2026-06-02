"""Benchmark and validation harness for the solar-system navigator."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .navigator import (
    NavigatorConstraints,
    NavigatorReport,
    navigate_solar_system,
    stable_hash,
    write_navigator_report,
)


@dataclass(frozen=True)
class NavigatorValidation:
    """Validation outcome for one navigator report."""
    target: str
    passed: bool
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    route_count: int = 0
    pareto_count: int = 0
    fastest_id: str | None = None
    cheapest_id: str | None = None
    balanced_id: str | None = None


@dataclass(frozen=True)
class NavigatorBenchmarkCase:
    """One real-ephemeris benchmark case."""
    name: str
    constraints: NavigatorConstraints
    expected_target: str
    min_routes: int = 1
    max_best_c3: float | None = None
    max_best_arrival_vinf: float | None = None
    max_best_total_dv_ms: float | None = None
    require_moon_tour: bool = False


@dataclass(frozen=True)
class NavigatorBenchmarkResult:
    """Full benchmark result bundle."""
    schema: str
    created_utc: str
    cases: tuple[dict, ...]
    validations: tuple[NavigatorValidation, ...]
    elapsed_s: float
    passed: bool
    certificate_hash: str
    artifacts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NavigatorBenchmarkComparison:
    """Diff two navigator benchmark summaries."""
    baseline_hash: str
    candidate_hash: str
    same_certificate: bool
    changed_cases: tuple[str, ...]
    route_count_delta: dict
    balanced_route_changed: dict
    cost_delta_ms: dict
    tof_delta_days: dict


def _png_valid(path: str | Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_report(report: NavigatorReport,
                    *,
                    min_routes: int = 1,
                    max_best_c3: float | None = None,
                    max_best_arrival_vinf: float | None = None,
                    max_best_total_dv_ms: float | None = None,
                    require_moon_tour: bool = False) -> NavigatorValidation:
    """Check route-card invariants and coarse real-data sanity bounds."""
    failures = []
    warnings = []
    routes = list(report.routes)
    if len(routes) < min_routes:
        failures.append(f"route_count {len(routes)} < {min_routes}")
    ids = [r.route_id for r in routes]
    if len(ids) != len(set(ids)):
        failures.append("route IDs are not unique")
    if report.balanced_id and report.balanced_id not in ids:
        failures.append("balanced_id missing from routes")
    if report.fastest_id and report.fastest_id not in ids:
        failures.append("fastest_id missing from routes")
    if report.cheapest_id and report.cheapest_id not in ids:
        failures.append("cheapest_id missing from routes")
    for route in routes:
        if route.total_dv_ms < 0.0 or not math.isfinite(route.total_dv_ms):
            failures.append(f"{route.route_id} invalid total_dv_ms")
        if route.tof_days <= 0.0 or not math.isfinite(route.tof_days):
            failures.append(f"{route.route_id} invalid tof_days")
        if route.c3_km2_s2 is not None and route.c3_km2_s2 < 0.0:
            failures.append(f"{route.route_id} negative C3")
        if not route.certificate_hash:
            failures.append(f"{route.route_id} missing certificate")
        if not route.assumptions or not route.validations:
            warnings.append(f"{route.route_id} has sparse route-card evidence")
    best = report.balanced
    if best is not None:
        if max_best_c3 is not None and (best.c3_km2_s2 or math.inf) > max_best_c3:
            failures.append(f"balanced C3 {(best.c3_km2_s2 or math.inf):.3g} > {max_best_c3}")
        if max_best_arrival_vinf is not None and (best.arrival_vinf_kms or math.inf) > max_best_arrival_vinf:
            failures.append(
                f"balanced arrival_vinf {(best.arrival_vinf_kms or math.inf):.3g} > {max_best_arrival_vinf}")
        if max_best_total_dv_ms is not None and best.total_dv_ms > max_best_total_dv_ms:
            failures.append(f"balanced total_dv_ms {best.total_dv_ms:.3g} > {max_best_total_dv_ms}")
    if require_moon_tour and not any("tisserand_moon_tour" in r.engine for r in routes):
        failures.append("required moon-tour route missing")
    return NavigatorValidation(
        target=report.target.name,
        passed=not failures,
        failures=tuple(failures),
        warnings=tuple(warnings),
        route_count=len(routes),
        pareto_count=len(report.pareto_front),
        fastest_id=report.fastest_id,
        cheapest_id=report.cheapest_id,
        balanced_id=report.balanced_id,
    )


def default_benchmark_cases() -> list[NavigatorBenchmarkCase]:
    """Small real-ephemeris cases that run quickly but touch real SPICE data."""
    return [
        NavigatorBenchmarkCase(
            name="earth_mars_small",
            expected_target="MARS",
            constraints=NavigatorConstraints(
                target="Mars",
                epoch_start="2026-01-01T00:00:00",
                departure_window_days=540.0,
                tof_range_days=(120.0, 420.0),
                n_dep=18,
                n_tof=14,
                include_gravity_assist=False,
            ),
            max_best_c3=35.0,
            max_best_arrival_vinf=6.0,
            max_best_total_dv_ms=9000.0,
        ),
        NavigatorBenchmarkCase(
            name="earth_enceladus_small",
            expected_target="ENCELADUS",
            constraints=NavigatorConstraints(
                target="Enceladus",
                epoch_start="2028-01-01T00:00:00",
                departure_window_days=900.0,
                tof_range_days=(700.0, 2400.0),
                n_dep=14,
                n_tof=12,
                include_gravity_assist=False,
            ),
            max_best_c3=140.0,
            max_best_arrival_vinf=9.0,
            max_best_total_dv_ms=10000.0,
            require_moon_tour=True,
        ),
    ]


def run_navigator_benchmark(cases: list[NavigatorBenchmarkCase] | None = None,
                            *,
                            outdir: str | Path | None = None,
                            make_plots: bool = True) -> NavigatorBenchmarkResult:
    """Run real-ephemeris navigator benchmark cases and optional artifact checks."""
    cases = cases or default_benchmark_cases()
    out = Path(outdir) if outdir is not None else None
    case_rows = []
    validations = []
    artifacts = {}
    t0 = time.perf_counter()
    for case in cases:
        report = navigate_solar_system(case.constraints)
        validation = validate_report(
            report,
            min_routes=case.min_routes,
            max_best_c3=case.max_best_c3,
            max_best_arrival_vinf=case.max_best_arrival_vinf,
            max_best_total_dv_ms=case.max_best_total_dv_ms,
            require_moon_tour=case.require_moon_tour,
        )
        validations.append(validation)
        case_artifacts = {}
        if out is not None:
            case_dir = out / case.name
            case_artifacts = write_navigator_report(
                report, case_dir, make_plots=make_plots)
            for key, path in list(case_artifacts.items()):
                if key.endswith("png") or key in {"porkchop_heatmap", "route_trade_space"}:
                    if not _png_valid(path):
                        validations[-1] = NavigatorValidation(
                            target=validation.target,
                            passed=False,
                            failures=(*validation.failures, f"invalid PNG artifact {key}"),
                            warnings=validation.warnings,
                            route_count=validation.route_count,
                            pareto_count=validation.pareto_count,
                            fastest_id=validation.fastest_id,
                            cheapest_id=validation.cheapest_id,
                            balanced_id=validation.balanced_id,
                        )
            artifacts[case.name] = case_artifacts
        best = report.balanced
        case_rows.append({
            "name": case.name,
            "target": report.target.name,
            "route_count": len(report.routes),
            "certificate_hash": report.certificate_hash,
            "balanced": None if best is None else {
                "route_id": best.route_id,
                "name": best.name,
                "sequence": best.sequence,
                "total_dv_ms": best.total_dv_ms,
                "tof_days": best.tof_days,
                "c3_km2_s2": best.c3_km2_s2,
                "arrival_vinf_kms": best.arrival_vinf_kms,
            },
        })
    elapsed = time.perf_counter() - t0
    payload = {
        "schema": "ariadne.solar_system_navigator_benchmark.v1",
        "cases": case_rows,
        "validations": validations,
        "elapsed_s": elapsed,
        "passed": all(v.passed for v in validations),
    }
    result = NavigatorBenchmarkResult(
        schema=payload["schema"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        cases=tuple(case_rows),
        validations=tuple(validations),
        elapsed_s=elapsed,
        passed=payload["passed"],
        certificate_hash=stable_hash(payload),
        artifacts=artifacts,
    )
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_summary.json").write_text(
            json.dumps(_jsonable(result), sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
    return result


def compare_benchmark_summaries(baseline: str | Path | dict,
                                candidate: str | Path | dict) -> NavigatorBenchmarkComparison:
    """Compare two `benchmark_summary.json` files."""
    def load(x):
        if isinstance(x, dict):
            return x
        return json.loads(Path(x).read_text(encoding="utf-8"))

    a = load(baseline)
    b = load(candidate)
    acases = {c["name"]: c for c in a.get("cases", [])}
    bcases = {c["name"]: c for c in b.get("cases", [])}
    names = sorted(set(acases) | set(bcases))
    changed = []
    route_delta = {}
    balanced_changed = {}
    cost_delta = {}
    tof_delta = {}
    for name in names:
        ca = acases.get(name, {})
        cb = bcases.get(name, {})
        route_delta[name] = int(cb.get("route_count", 0)) - int(ca.get("route_count", 0))
        ba = ca.get("balanced") or {}
        bb = cb.get("balanced") or {}
        balanced_changed[name] = ba.get("route_id") != bb.get("route_id")
        cost_delta[name] = float(bb.get("total_dv_ms", 0.0)) - float(ba.get("total_dv_ms", 0.0))
        tof_delta[name] = float(bb.get("tof_days", 0.0)) - float(ba.get("tof_days", 0.0))
        if route_delta[name] or balanced_changed[name] or abs(cost_delta[name]) > 1e-9 or abs(tof_delta[name]) > 1e-9:
            changed.append(name)
    return NavigatorBenchmarkComparison(
        baseline_hash=a.get("certificate_hash", ""),
        candidate_hash=b.get("certificate_hash", ""),
        same_certificate=a.get("certificate_hash") == b.get("certificate_hash"),
        changed_cases=tuple(changed),
        route_count_delta=route_delta,
        balanced_route_changed=balanced_changed,
        cost_delta_ms=cost_delta,
        tof_delta_days=tof_delta,
    )


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
