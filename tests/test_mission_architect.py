import json


def _fake_direct_optimizer(et0, tof_grid, leo_alt=200.0, llo_alt=100.0, lead_deg=110.0):
    records = [
        {
            "tof_days": float(tof_grid[0]),
            "miss_km": 0.2,
            "v_inf_kms": 0.85,
            "dv_tli_ms": 3150.0,
            "dv_loi_ms": 860.0,
            "total_ms": 4010.0,
        },
        {
            "tof_days": float(tof_grid[-1]),
            "miss_km": 2.0,
            "v_inf_kms": 0.75,
            "dv_tli_ms": 3120.0,
            "dv_loi_ms": 820.0,
            "total_ms": 3940.0,
        },
    ]
    return min(records, key=lambda r: r["total_ms"]), records


def _fake_low_energy_solver(jacobi_values=(3.05,), leo_alt=200.0, llo_alt=100.0, **kwargs):
    records = [
        {
            "point": "L1",
            "jacobi": 3.10,
            "tli_ms": 3150.0,
            "loi_ms": 610.0,
            "total_ms": 3760.0,
            "tof_days": 8.5,
            "periapsis_alt_km": 105.0,
        }
    ]
    baseline = {"direct_total_ms": 4010.0, "coimbra_ms": 3925.0}
    return records[0], records, baseline


def _fake_frontier_solver(epoch, tof_grid, leo_alt=200.0, llo_alt=100.0, include_wsb=True):
    return [
        {"label": "direct 4d", "dv_ms": 4010.0, "sensitivity": 12.0, "tof_days": 4.0},
        {"label": "direct 6d", "dv_ms": 3940.0, "sensitivity": 20.0, "tof_days": 6.0},
        {"label": "WSB 49d", "dv_ms": 3600.0, "sensitivity": 35.0, "tof_days": 48.8},
    ]


def test_cislunar_architect_builds_ranked_certified_report():
    from ariadne.transfers.mission_architect import (
        MissionConstraints,
        architect_cislunar_round_trip,
    )

    report = architect_cislunar_round_trip(
        MissionConstraints(
            epoch="2025-06-01T00:00:00",
            outbound_tof_days=(4.0, 6.0),
            return_tof_days=(3.0,),
            lunar_stay_days=2.0,
        ),
        direct_optimizer=_fake_direct_optimizer,
        low_energy_solver=_fake_low_energy_solver,
        frontier_solver=_fake_frontier_solver,
    )
    assert report.schema == "ariadne.cislunar_mission_architect.v1"
    assert report.certificate_hash
    assert report.recommended is not None
    assert report.recommended.total_dv_ms > 0.0
    assert report.recommended.outbound.direction == "earth_to_moon"
    assert report.recommended.return_leg.direction == "moon_to_earth"
    assert any(c.free_return_capable for c in report.candidates)
    assert set(report.pareto_front)


def test_cislunar_architect_constraints_filter_candidates():
    from ariadne.transfers.mission_architect import (
        MissionConstraints,
        architect_cislunar_round_trip,
    )

    report = architect_cislunar_round_trip(
        MissionConstraints(
            outbound_tof_days=(4.0, 6.0),
            return_tof_days=(3.0,),
            include_low_energy=False,
            include_coherence=False,
            max_total_dv_ms=1000.0,
        ),
        direct_optimizer=_fake_direct_optimizer,
        low_energy_solver=_fake_low_energy_solver,
        frontier_solver=_fake_frontier_solver,
    )
    assert report.candidates == ()
    assert report.recommended is None


def test_write_architecture_report_roundtrip(tmp_path):
    from ariadne.transfers.mission_architect import (
        MissionConstraints,
        architect_cislunar_round_trip,
        write_architecture_report,
    )

    report = architect_cislunar_round_trip(
        MissionConstraints(outbound_tof_days=(4.0,), return_tof_days=(3.0,)),
        direct_optimizer=_fake_direct_optimizer,
        low_energy_solver=_fake_low_energy_solver,
        frontier_solver=_fake_frontier_solver,
    )
    path = tmp_path / "architect.json"
    payload = write_architecture_report(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["certificate_hash"] == report.certificate_hash
    assert payload["recommended_id"] == report.recommended_id
    assert loaded["candidates"][0]["certificate_hash"]


def test_public_api_exports_cislunar_architect():
    import ariadne

    report = ariadne.architect_cislunar_round_trip(
        direct_optimizer=_fake_direct_optimizer,
        low_energy_solver=_fake_low_energy_solver,
        frontier_solver=_fake_frontier_solver,
    )
    assert report.recommended is not None
