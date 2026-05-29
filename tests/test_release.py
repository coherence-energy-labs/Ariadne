"""Stage 17 tests: open release bundle + reference routes."""
import os

from ariadne.atlas.release import export_release, reference_routes
from ariadne.atlas.store import read_atlas


def _tiny_atlas():
    return {
        "provenance": {"created_utc": "2026-05-29T00:00:00+00:00", "version": "test",
                       "note": "test", "config": {"n_seeds": 40}},
        "systems": {
            "Earth-Moon": {
                "params": {"mu": 0.0121, "L_star": 384400.0, "T_star": 1.0, "V_star": 1.0,
                           "primary": "Earth", "secondary": "Moon"},
                "libration": {"L1_km": 58000.0, "L2_km": 64500.0, "lyap_period_d": 11.9,
                              "half_period_residual": 1e-12},
                "routes": [{"path": ["L1@3.12", "L2@3.12"], "dv_ms": 0.4, "hops": 1}],
            },
            "Didymos-Dimorphos": {
                "params": {"mu": 6.9e-3, "L_star": 1.19, "T_star": 1.0, "V_star": 1.0,
                           "primary": "Didymos", "secondary": "Dimorphos"},
                "libration": {"L1_km": 0.15, "L2_km": 0.16, "lyap_period_d": 0.22,
                              "half_period_residual": 1e-13},
            },
        },
    }


def test_export_release_writes_all_artifacts(tmp_path):
    info = export_release(str(tmp_path), atlas=_tiny_atlas())
    for key in ("atlas_h5", "index_md", "reference_csv"):
        assert os.path.exists(info[key])
    # atlas round-trips
    back = read_atlas(info["atlas_h5"])
    assert set(back["systems"]) == {"Earth-Moon", "Didymos-Dimorphos"}
    # index mentions a system and the reference-routes heading
    index = open(info["index_md"], encoding="utf-8").read()
    assert "Didymos-Dimorphos" in index
    assert "Reference routes" in index


def test_reference_routes_are_honest():
    refs = reference_routes()
    # at least one genuinely GMAT-validated route, and only honest tags
    gmat = [r for r in refs if "GMAT-validated" in r["validation"]]
    assert len(gmat) >= 1
    # Earth->Moon transfers and libration routes are kept in separate classes
    classes = {r["route_class"] for r in refs}
    assert "Earth->Moon transfer" in classes
    assert "libration-network route" in classes
    # the cheap (~17 m/s) route must be a libration-network route, NOT an Earth->Moon transfer
    cheap = min(refs, key=lambda r: r["dv_ms"])
    assert cheap["route_class"] == "libration-network route"
