"""Reviewer-grade visual artifact audits."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from .closure import ArtifactEvidence, stable_hash


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as fh:
        header = fh.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def navigator_visual_contract_evidence(
    benchmark_dir: str | Path,
    *,
    min_width_px: int = 900,
    min_height_px: int = 600,
) -> ArtifactEvidence:
    """Audit navigator visual/report artifacts for reviewer-grade context.

    This is intentionally more than "is the PNG readable". It checks that each
    benchmark case has the three expected visual products, route cards, route
    events with coordinates, route assumptions/validations, and report-level
    certificates. Pixel-level OCR is not reliable enough for a truth gate, so
    the machine-readable route report and sidecar markdown carry the semantic
    contract while PNG signatures/dimensions carry the visual-render contract.
    """

    root = Path(benchmark_dir)
    reports = sorted(root.glob("*/navigator_report.json"))
    cases = []
    failures = []
    png_total = 0
    png_valid = 0
    route_total = 0
    routes_with_full_events = 0
    routes_with_assumptions = 0
    routes_with_validations = 0
    routes_with_certificates = 0
    required_pngs = ("mission_plate.png", "porkchop_heatmap.png", "route_trade_space.png")
    for report_path in reports:
        case_dir = report_path.parent
        case_name = case_dir.name
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not payload.get("certificate_hash"):
            failures.append(f"{case_name}: navigator report missing certificate_hash")
        route_cards = case_dir / "route_cards.md"
        if not route_cards.exists() or route_cards.stat().st_size == 0:
            failures.append(f"{case_name}: route_cards.md missing or empty")
        manifest_path = case_dir / "figure_manifest.json"
        manifest = None
        if not manifest_path.exists():
            failures.append(f"{case_name}: figure_manifest.json missing")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"{case_name}: figure_manifest.json unreadable: {exc}")
        if manifest is not None:
            if manifest.get("schema") != "ariadne.navigator_figure_manifest.v1":
                failures.append(f"{case_name}: figure manifest schema mismatch")
            if manifest.get("report_certificate_hash") != payload.get("certificate_hash"):
                failures.append(f"{case_name}: figure manifest certificate mismatch")
            units = manifest.get("units") or {}
            for key in ("delta_v", "time_of_flight", "c3", "arrival_vinf", "coordinates"):
                if key not in units:
                    failures.append(f"{case_name}: figure manifest missing unit {key}")
            roles = manifest.get("route_roles") or {}
            for key in ("fastest", "cheapest", "balanced"):
                if key not in roles:
                    failures.append(f"{case_name}: figure manifest missing route role {key}")
            required_figures = manifest.get("required_figures") or {}
            for fig_name in ("mission_plate", "porkchop_heatmap", "route_trade_space"):
                fig = required_figures.get(fig_name) or {}
                if not fig.get("path") or not fig.get("must_show"):
                    failures.append(f"{case_name}: figure manifest incomplete for {fig_name}")
        for name in required_pngs:
            png_total += 1
            png = case_dir / name
            dims = _png_dimensions(png) if png.exists() else None
            if dims is None:
                failures.append(f"{case_name}: {name} missing or invalid")
                continue
            if dims[0] < min_width_px or dims[1] < min_height_px:
                failures.append(f"{case_name}: {name} below minimum size {dims[0]}x{dims[1]}")
                continue
            png_valid += 1
        routes = payload.get("routes") or []
        route_total += len(routes)
        for route in routes:
            if route.get("certificate_hash"):
                routes_with_certificates += 1
            if route.get("assumptions"):
                routes_with_assumptions += 1
            if route.get("validations"):
                routes_with_validations += 1
            events = route.get("events") or []
            full = bool(events)
            for event in events:
                if event.get("role") == "moon-tour-node":
                    notes = str(event.get("notes", "")).lower()
                    if "tisserand" in notes and "phased moon ephemeris not attached" in notes:
                        continue
                coords = event.get("coordinates_km")
                if not isinstance(coords, list) or len(coords) != 3:
                    full = False
                    break
                if any(not isinstance(x, (int, float)) for x in coords):
                    full = False
                    break
            if full:
                routes_with_full_events += 1
        cases.append(case_name)

    png_fraction = 1.0 if png_total == 0 else png_valid / png_total
    semantic_fraction = 1.0 if route_total == 0 else min(
        routes_with_full_events,
        routes_with_assumptions,
        routes_with_validations,
        routes_with_certificates,
    ) / route_total
    status = bool(reports) and not failures and png_fraction == 1.0 and semantic_fraction == 1.0
    metrics = {
        "case_count": len(reports),
        "png_total": png_total,
        "png_valid": png_valid,
        "png_valid_fraction": png_fraction,
        "route_total": route_total,
        "routes_with_full_events": routes_with_full_events,
        "routes_with_assumptions": routes_with_assumptions,
        "routes_with_validations": routes_with_validations,
        "routes_with_certificates": routes_with_certificates,
        "semantic_route_fraction": semantic_fraction,
        "failure_count": len(failures),
        "manifest_count": sum(1 for p in reports if (p.parent / "figure_manifest.json").exists()),
        "cases": cases,
        "failures": failures,
    }
    return ArtifactEvidence(
        artifact_id="navigator_visual_contract",
        subsystem="visuals",
        kind="reviewer_visual_contract",
        path=str(root),
        status="pass" if status else "fail",
        certificate_hash=stable_hash(metrics),
        content_hash=stable_hash(metrics),
        metrics=metrics,
        required=True,
        notes=("Navigator PNG + route-card semantic visual contract",),
    )
