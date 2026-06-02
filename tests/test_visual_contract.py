import base64
import json

from ariadne.proof.visuals import navigator_visual_contract_evidence


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _png(width: int, height: int) -> bytes:
    data = bytearray(PNG_1X1)
    data[16:20] = int(width).to_bytes(4, "big")
    data[20:24] = int(height).to_bytes(4, "big")
    return bytes(data)


def _case(root, *, complete=True):
    case = root / "earth_mars_small"
    case.mkdir(parents=True)
    for name in ("mission_plate.png", "porkchop_heatmap.png", "route_trade_space.png"):
        (case / name).write_bytes(_png(1200, 800))
    (case / "route_cards.md").write_text("route cards", encoding="utf-8")
    route = {
        "route_id": "r",
        "certificate_hash": "c",
        "assumptions": ["a"],
        "validations": ["v"],
        "events": [
            {"coordinates_km": [1.0, 2.0, 3.0]},
            {"coordinates_km": [4.0, 5.0, 6.0]},
            {"role": "moon-tour-node",
             "notes": "abstract Tisserand screening node; phased moon ephemeris not attached"},
        ],
    }
    if not complete:
        route["events"][1].pop("coordinates_km")
    (case / "navigator_report.json").write_text(
        json.dumps({"certificate_hash": "report", "routes": [route]}),
        encoding="utf-8",
    )
    (case / "figure_manifest.json").write_text(
        json.dumps({
            "schema": "ariadne.navigator_figure_manifest.v1",
            "report_certificate_hash": "report",
            "units": {
                "delta_v": "m/s",
                "time_of_flight": "days",
                "c3": "km^2/s^2",
                "arrival_vinf": "km/s",
                "coordinates": "km",
            },
            "route_roles": {"fastest": "r", "cheapest": "r", "balanced": "r"},
            "required_figures": {
                "mission_plate": {"path": str(case / "mission_plate.png"), "must_show": ["origin"]},
                "porkchop_heatmap": {"path": str(case / "porkchop_heatmap.png"), "must_show": ["dv"]},
                "route_trade_space": {"path": str(case / "route_trade_space.png"), "must_show": ["pareto"]},
            },
        }),
        encoding="utf-8",
    )


def test_navigator_visual_contract_passes_complete_case(tmp_path):
    _case(tmp_path)

    evidence = navigator_visual_contract_evidence(tmp_path, min_width_px=900, min_height_px=600)

    assert evidence.status == "pass"
    assert evidence.metrics["semantic_route_fraction"] == 1.0
    assert evidence.metrics["png_valid_fraction"] == 1.0


def test_navigator_visual_contract_fails_missing_route_coordinates(tmp_path):
    _case(tmp_path, complete=False)

    evidence = navigator_visual_contract_evidence(tmp_path, min_width_px=900, min_height_px=600)

    assert evidence.status == "fail"
    assert evidence.metrics["semantic_route_fraction"] == 0.0
