import base64
import json

from ariadne.proof.defaults import (
    audit_png_directory,
    build_default_ariadne_closure,
    collect_default_evidence,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _png(width: int, height: int) -> bytes:
    data = bytearray(PNG_1X1)
    data[16:20] = int(width).to_bytes(4, "big")
    data[20:24] = int(height).to_bytes(4, "big")
    return bytes(data)


def test_audit_png_directory_checks_signature_and_dimensions(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "good.png").write_bytes(PNG_1X1)
    (figures / "bad.png").write_bytes(b"not a png")

    evidence = audit_png_directory(figures)

    assert evidence.status == "fail"
    assert evidence.metrics["n_png"] == 2
    assert evidence.metrics["n_valid_png"] == 1
    assert evidence.metrics["png_valid_fraction"] == 0.5
    assert evidence.metrics["invalid_png"] == ["bad.png"]


def test_default_closure_collects_existing_artifact_shapes(tmp_path):
    dream = tmp_path / "data" / "benchmarks" / "dream_lab"
    dream.mkdir(parents=True)
    (dream / "dream_run.json").write_text(
        json.dumps({"status": "pass", "experiment_count": 2, "certificate_hash": "dreamhash"}),
        encoding="utf-8",
    )
    nav = tmp_path / "data" / "benchmarks" / "solar_navigator_benchmark"
    nav.mkdir(parents=True)
    (nav / "benchmark_summary.json").write_text(
        json.dumps({"passed": True, "certificate_hash": "navhash", "elapsed_s": 0.5}),
        encoding="utf-8",
    )
    disc = tmp_path / "data" / "benchmarks" / "real_corpus_mpc_500" / "benchmark"
    disc.mkdir(parents=True)
    (disc / "metrics.json").write_text(
        json.dumps({
            "accuracy": 0.98,
            "safe_accuracy": 1.0,
            "macro_f1": 0.97,
            "reliability": {"ece": 0.01},
            "n": 500,
            "certificate_hash": "dischash",
        }),
        encoding="utf-8",
    )
    adversarial = tmp_path / "data" / "benchmarks" / "inference_adversarial_builtin"
    adversarial.mkdir(parents=True)
    (adversarial / "metrics.json").write_text(
        json.dumps({
            "accuracy": 0.64,
            "safe_accuracy": 0.95,
            "macro_f1": 0.55,
            "n": 75,
            "certificate_hash": "advhash",
        }),
        encoding="utf-8",
    )
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    for i in range(10):
        (figures / f"fig_{i}.png").write_bytes(PNG_1X1)
    nav_case = tmp_path / "data" / "benchmarks" / "solar_navigator_benchmark" / "earth_mars_small"
    nav_case.mkdir(parents=True)
    for name in ("mission_plate.png", "porkchop_heatmap.png", "route_trade_space.png"):
        (nav_case / name).write_bytes(_png(1200, 800))
    (nav_case / "route_cards.md").write_text("route cards", encoding="utf-8")
    (nav_case / "navigator_report.json").write_text(
        json.dumps({
            "certificate_hash": "nav_case",
            "routes": [{
                "route_id": "r",
                "certificate_hash": "c",
                "assumptions": ["a"],
                "validations": ["v"],
                "events": [{"coordinates_km": [1, 2, 3]}, {"coordinates_km": [4, 5, 6]}],
            }],
        }),
        encoding="utf-8",
    )
    (nav_case / "figure_manifest.json").write_text(
        json.dumps({
            "schema": "ariadne.navigator_figure_manifest.v1",
            "report_certificate_hash": "nav_case",
            "units": {
                "delta_v": "m/s",
                "time_of_flight": "days",
                "c3": "km^2/s^2",
                "arrival_vinf": "km/s",
                "coordinates": "km",
            },
            "route_roles": {"fastest": "r", "cheapest": "r", "balanced": "r"},
            "required_figures": {
                "mission_plate": {"path": str(nav_case / "mission_plate.png"), "must_show": ["origin"]},
                "porkchop_heatmap": {"path": str(nav_case / "porkchop_heatmap.png"), "must_show": ["dv"]},
                "route_trade_space": {"path": str(nav_case / "route_trade_space.png"), "must_show": ["pareto"]},
            },
        }),
        encoding="utf-8",
    )
    promo = tmp_path / "data" / "benchmarks" / "route_promotion"
    promo.mkdir(parents=True)
    (promo / "promotion_report.json").write_text(
        json.dumps({
            "status": "pass",
            "route_count": 2,
            "promoted_count": 2,
            "rejected_count": 0,
            "certificate_hash": "promohash",
        }),
        encoding="utf-8",
    )
    transport = tmp_path / "data" / "benchmarks" / "transport_admissibility"
    transport.mkdir(parents=True)
    (transport / "metrics.json").write_text(
        json.dumps({
            "passed": True,
            "speedup_astar_vs_brute": 2.0,
            "astar_expansions": 4,
            "brute_expansions": 8,
            "certificate_hash": "transporthash",
        }),
        encoding="utf-8",
    )
    integrity = tmp_path / "data" / "benchmarks" / "artifact_integrity"
    integrity.mkdir(parents=True)
    (integrity / "artifact_manifest.json").write_text(
        json.dumps({
            "schema": "ariadne.artifact_integrity_manifest.v1",
            "file_count": 12,
            "total_size_bytes": 1234,
            "certificate_hash": "artifacthash",
        }),
        encoding="utf-8",
    )

    evidence = collect_default_evidence(tmp_path)
    report = build_default_ariadne_closure(tmp_path)

    assert {e.subsystem for e in evidence} == {
        "dream_calibration", "solar_navigator", "discovery_inference", "visuals",
        "trajectory_certification", "discovery_robustness", "transport_search",
        "artifact_integrity",
    }
    assert report.critical_failures == 0
    assert report.warnings == 0
    assert report.status == "complete"
    assert report.blocking_residuals == 0
    assert report.validate_hash()


def test_default_closure_fails_without_required_artifacts(tmp_path):
    report = build_default_ariadne_closure(tmp_path)

    assert report.status == "partial"
    assert report.critical_failures >= 3
    assert report.readiness_score < 1.0
