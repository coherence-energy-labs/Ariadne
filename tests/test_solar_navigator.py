import json

import numpy as np


def _fake_porkchop(dep_body, arr_body, et_start, dep_days, tof_range,
                   n_dep=6, n_tof=5, **kwargs):
    dep_grid = et_start + np.linspace(0.0, dep_days, n_dep) * 86400.0
    tof_grid = np.linspace(tof_range[0], tof_range[1], n_tof)
    total = np.add.outer(np.linspace(9000.0, 7000.0, n_tof),
                         np.linspace(1200.0, 0.0, n_dep))
    c3 = total / 1000.0
    best = {
        "c3": 42.0,
        "dep_vinf_kms": 6.48,
        "arr_vinf_kms": 5.1,
        "dv_dep_ms": 5200.0,
        "dv_arr_ms": 0.0,
        "total_ms": 7200.0,
        "tof_days": float(tof_grid[-1]),
        "et_dep": float(dep_grid[-1]),
        "et_arr": float(dep_grid[-1] + tof_grid[-1] * 86400.0),
        "r1": np.array([1.0, 2.0, 3.0]),
        "r2": np.array([7.0, 8.0, 9.0]),
        "v1": np.array([0.0, 1.0, 0.0]),
    }
    return {
        "dep_grid": dep_grid,
        "tof_grid": tof_grid,
        "C3": c3,
        "total_ms": total,
        "grid_best": best,
        "dep_body": dep_body,
        "arr_body": arr_body,
    }


def _fake_direct_optimizer(dep_body, arr_body, et_start, dep_days, tof_range,
                           **kwargs):
    return {
        "c3": 38.0,
        "dep_vinf_kms": 6.16,
        "arr_vinf_kms": 4.7,
        "dv_dep_ms": 5000.0,
        "dv_arr_ms": 0.0,
        "total_ms": 6900.0,
        "tof_days": 1200.0,
        "et_dep": float(et_start + 10 * 86400.0),
        "et_arr": float(et_start + 1210 * 86400.0),
        "r1": np.array([4.0, 5.0, 6.0]),
        "r2": np.array([10.0, 11.0, 12.0]),
        "v1": np.array([0.0, 1.0, 0.0]),
    }


def _fake_flyby_optimizer(bodies, et_start, dep_window_days, tof_bounds, **kwargs):
    epochs = [float(et_start)]
    for lo, hi in tof_bounds:
        epochs.append(epochs[-1] + 0.5 * (lo + hi) * 86400.0)
    return {
        "bodies": bodies,
        "et0": float(et_start),
        "epochs": epochs,
        "tofs_days": [0.5 * (lo + hi) for lo, hi in tof_bounds],
        "c3": 18.0,
        "dep_vinf_kms": 4.24,
        "dv_launch_ms": 4300.0,
        "flybys": [{"body": b, "feasible": True, "mismatch_ms": 50.0}
                   for b in bodies[1:-1]],
        "mismatch_dv_ms": 150.0,
        "dsm_dv_ms": 0.0,
        "arr_vinf_kms": 6.2,
        "total_dv_ms": 4450.0,
        "infeasible": 0.0,
        "tof_total_days": sum(0.5 * (lo + hi) for lo, hi in tof_bounds),
        "r1": np.array([0.0, 0.0, 0.0]),
        "v1": np.array([0.0, 0.0, 0.0]),
    }


def test_navigator_builds_saturn_moon_route_with_tour():
    from ariadne.interplanetary.navigator import (
        NavigatorConstraints,
        navigate_solar_system,
    )

    report = navigate_solar_system(
        NavigatorConstraints(
            target="Enceladus",
            epoch_start="2028-01-01T00:00:00",
            n_dep=6,
            n_tof=5,
            optimize_flybys=True,
        ),
        porkchop_solver=_fake_porkchop,
        direct_optimizer=_fake_direct_optimizer,
        flyby_optimizer=_fake_flyby_optimizer,
    )
    assert report.schema == "ariadne.solar_system_navigator.v1"
    assert report.target.name == "ENCELADUS"
    assert report.routes
    assert report.balanced is not None
    assert any("ENCELADUS" in r.sequence for r in report.routes)
    assert any(r.engine.endswith("tisserand_moon_tour") for r in report.routes)
    assert report.fastest is not None
    assert report.cheapest is not None
    assert report.certificate_hash
    assert any(r.name.startswith("direct_pareto") for r in report.routes)


def test_navigator_constraints_can_filter_everything():
    from ariadne.interplanetary.navigator import (
        NavigatorConstraints,
        navigate_solar_system,
    )

    report = navigate_solar_system(
        NavigatorConstraints(target="Saturn", max_total_dv_ms=1.0),
        porkchop_solver=_fake_porkchop,
        direct_optimizer=_fake_direct_optimizer,
        flyby_optimizer=_fake_flyby_optimizer,
    )
    assert report.routes == ()
    assert report.balanced is None


def test_navigator_writes_json_and_pngs(tmp_path):
    from ariadne.interplanetary.navigator import (
        NavigatorConstraints,
        navigate_solar_system,
        write_navigator_report,
    )

    report = navigate_solar_system(
        NavigatorConstraints(target="Mars", n_dep=6, n_tof=5),
        porkchop_solver=_fake_porkchop,
        direct_optimizer=_fake_direct_optimizer,
        flyby_optimizer=_fake_flyby_optimizer,
    )
    artifacts = write_navigator_report(report, tmp_path, make_plots=False)
    loaded = json.loads((tmp_path / "navigator_report.json").read_text(encoding="utf-8"))
    assert loaded["certificate_hash"] == report.certificate_hash
    assert artifacts["report"].endswith("navigator_report.json")
    assert artifacts["route_cards"].endswith("route_cards.md")
    assert (tmp_path / "route_cards.md").exists()
    assert loaded["routes"][0]["certificate_hash"]


def test_public_api_exports_navigator():
    import ariadne
    from ariadne.interplanetary.navigator import NavigatorConstraints

    report = ariadne.navigate_solar_system(
        NavigatorConstraints(target="Mars", n_dep=6, n_tof=5),
        porkchop_solver=_fake_porkchop,
        direct_optimizer=_fake_direct_optimizer,
        flyby_optimizer=_fake_flyby_optimizer,
    )
    assert report.balanced is not None


def test_navigator_supports_non_earth_direct_origin():
    from ariadne.interplanetary.navigator import NavigatorConstraints, navigate_solar_system

    report = navigate_solar_system(
        NavigatorConstraints(origin="Mars", target="Venus", n_dep=6, n_tof=5),
        porkchop_solver=_fake_porkchop,
        direct_optimizer=_fake_direct_optimizer,
        flyby_optimizer=_fake_flyby_optimizer,
    )

    assert report.origin.name == "MARS"
    assert report.balanced is not None
    assert report.balanced.sequence[0] == "MARS BARYCENTER"


def test_navigator_report_validation_catches_bad_bounds():
    from ariadne.interplanetary.navigator import NavigatorConstraints, navigate_solar_system
    from ariadne.interplanetary.navigator_benchmark import validate_report

    report = navigate_solar_system(
        NavigatorConstraints(target="Mars", n_dep=6, n_tof=5),
        porkchop_solver=_fake_porkchop,
        direct_optimizer=_fake_direct_optimizer,
        flyby_optimizer=_fake_flyby_optimizer,
    )
    validation = validate_report(report, max_best_c3=1.0)
    assert not validation.passed
    assert any("C3" in failure for failure in validation.failures)


def test_benchmark_summary_comparison_detects_drift(tmp_path):
    from ariadne.interplanetary.navigator_benchmark import compare_benchmark_summaries

    a = {
        "certificate_hash": "a",
        "cases": [{
            "name": "case",
            "route_count": 2,
            "balanced": {"route_id": "r1", "total_dv_ms": 10.0, "tof_days": 5.0},
        }],
    }
    b = {
        "certificate_hash": "b",
        "cases": [{
            "name": "case",
            "route_count": 3,
            "balanced": {"route_id": "r2", "total_dv_ms": 12.5, "tof_days": 4.0},
        }],
    }
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")
    diff = compare_benchmark_summaries(pa, pb)
    assert not diff.same_certificate
    assert diff.changed_cases == ("case",)
    assert diff.route_count_delta["case"] == 1
    assert diff.balanced_route_changed["case"]
    assert diff.cost_delta_ms["case"] == 2.5
