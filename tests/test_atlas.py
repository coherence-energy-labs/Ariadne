"""Stage 16 tests: HDF5 atlas round-trip + multi-system generalization."""
import pytest

from ariadne.data.constants import ATLAS_SYSTEMS
from ariadne.atlas.store import write_atlas, read_atlas
from ariadne.transfers.jovian import moon_libration


def _synthetic_atlas():
    return {
        "provenance": {"created_utc": "2026-05-29T00:00:00+00:00", "version": "test",
                       "note": "unit test", "config": {"n_seeds": 40, "route_K": 3}},
        "systems": {
            "Earth-Moon": {
                "params": {"mu": 0.0121, "L_star": 384400.0, "T_star": 375200.0,
                           "V_star": 1.024, "primary": "Earth", "secondary": "Moon"},
                "libration": {"L1_km": 58000.0, "L2_km": 64500.0,
                              "lyap_period_d": 11.9, "half_period_residual": 1e-12},
                "graph": {
                    "nodes": [{"key": "L1@3.12", "point": "L1", "jacobi": 3.12},
                              {"key": "L2@3.12", "point": "L2", "jacobi": 3.12}],
                    "edges": [{"src": "L1@3.12", "dst": "L2@3.12",
                               "dv_ms": 0.4, "fragility": 7.6}],
                },
                "routes": [{"path": ["L1@3.12", "L2@3.12"], "dv_ms": 0.4, "hops": 1}],
            },
            "Saturn-Titan": {
                "params": {"mu": 2.366e-4, "L_star": 1221870.0, "T_star": 1.0,
                           "V_star": 5.5, "primary": "Saturn", "secondary": "Titan"},
                "libration": {"L1_km": 51645.0, "L2_km": 54000.0,
                              "lyap_period_d": 7.46, "half_period_residual": 1e-12},
            },
        },
    }


def test_atlas_roundtrip_exact(tmp_path):
    a = _synthetic_atlas()
    path = tmp_path / "atlas.h5"
    write_atlas(str(path), a)
    b = read_atlas(str(path))

    assert set(a["systems"]) == set(b["systems"])
    assert b["provenance"]["version"] == "test"
    assert b["provenance"]["config"]["n_seeds"] == 40
    # params + libration preserved
    for name in a["systems"]:
        assert abs(a["systems"][name]["params"]["mu"]
                   - b["systems"][name]["params"]["mu"]) < 1e-15
        assert abs(a["systems"][name]["libration"]["L1_km"]
                   - b["systems"][name]["libration"]["L1_km"]) < 1e-9


def test_atlas_roundtrip_preserves_graph_and_routes(tmp_path):
    a = _synthetic_atlas()
    path = tmp_path / "atlas.h5"
    write_atlas(str(path), a)
    b = read_atlas(str(path))
    em = b["systems"]["Earth-Moon"]
    assert em["graph"]["nodes"][0]["key"] == "L1@3.12"
    assert em["graph"]["edges"][0]["dst"] == "L2@3.12"
    assert abs(em["graph"]["edges"][0]["dv_ms"] - 0.4) < 1e-9
    assert em["routes"][0]["path"] == ["L1@3.12", "L2@3.12"]
    assert em["routes"][0]["hops"] == 1
    # a system without a graph round-trips with just params + libration
    assert "graph" not in b["systems"]["Saturn-Titan"]


@pytest.mark.slow
def test_engine_generalizes_across_mass_ratio_spectrum():
    """Libration structure is sensible for every atlas system (mu ~ 1.6e-8 .. 7e-3)."""
    for S in ATLAS_SYSTEMS:
        m = moon_libration(S)
        assert m["L1_km"] > 0.0
        assert m["lyap_period_d"] > 0.0
        assert m["orbit"].half_period_residual < 1e-9
