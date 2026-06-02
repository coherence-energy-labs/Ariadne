"""Stage 21 tests: interplanetary porkchop + global Lambert + GMAT export."""
import os

import numpy as np
import pytest

from ariadne.data.ephemeris import et
from ariadne.interplanetary.porkchop import (
    lambert_transfer, optimize_window, launch_windows, porkchop,
    time_energy_pareto, coherent_knee, _lambert_transfer_from_states,
)
from ariadne.interplanetary.gmat_helio import export_transfer_gmat, _gmat_gregorian

START = "2026-01-01T00:00:00"


def test_gmat_gregorian_format():
    assert _gmat_gregorian("2026-11-01T00:00:00.000") == "01 Nov 2026 00:00:00.000"
    assert _gmat_gregorian("2028-12-15T06:30:00") == "15 Dec 2028 06:30:00.000"


def test_state_cached_lambert_matches_public_transfer():
    from ariadne.data.ephemeris import body_state

    e0 = et("2026-11-01T00:00:00")
    tof = 300.0
    sd = body_state("EARTH", e0, "J2000", "SUN")
    sa = body_state("MARS BARYCENTER", e0 + tof * 86400.0, "J2000", "SUN")
    a = lambert_transfer("EARTH", "MARS BARYCENTER", e0, tof)
    b = _lambert_transfer_from_states("EARTH", "MARS BARYCENTER", e0, tof, sd, sa)
    assert a is not None and b is not None
    for key in ("c3", "arr_vinf_kms", "dv_dep_ms", "dv_arr_ms", "total_ms"):
        assert abs(a[key] - b[key]) < 1e-9


@pytest.mark.slow
def test_earth_mars_transfer_is_realistic():
    t = lambert_transfer("EARTH", "MARS BARYCENTER", et("2026-11-01T00:00:00"), 300.0)
    assert t is not None
    assert 5.0 < t["c3"] < 30.0                       # realistic Mars departure C3
    assert 2.0 < t["arr_vinf_kms"] < 5.0


@pytest.mark.slow
def test_global_optimizer_beats_or_matches_grid():
    e0 = et(START)
    pk = porkchop("EARTH", "MARS BARYCENTER", e0, dep_days=540, tof_range=(120, 400),
                  n_dep=40, n_tof=30)
    opt = optimize_window("EARTH", "MARS BARYCENTER", e0, dep_days=540,
                          tof_range=(120, 400), maxiter=40)
    assert opt["total_ms"] <= pk["grid_best"]["total_ms"] + 1.0
    assert 5.0 <= opt["c3"] <= 25.0


@pytest.mark.slow
def test_launch_windows_follow_mars_synodic_cadence():
    e0 = et(START)
    lw = launch_windows("EARTH", "MARS BARYCENTER", e0, years=6.0,
                        tof_range=(120, 400), n_dep=120, n_tof=24)
    dv, dep = lw["best_dv_ms"], lw["dep_grid"]
    med = np.nanmedian(dv)
    raw = [(float(dep[k]), float(dv[k])) for k in range(2, len(dv) - 2)
           if dv[k] < dv[k - 1] and dv[k] < dv[k + 1] and dv[k] < med]
    wins = []                                          # dedupe clusters within 300 d
    for e_, d_ in sorted(raw):
        if wins and (e_ - wins[-1][0]) / 86400.0 < 300.0:
            if d_ < wins[-1][1]:
                wins[-1] = (e_, d_)
        else:
            wins.append((e_, d_))
    assert len(wins) >= 2
    gaps_months = [(b[0] - a[0]) / 86400.0 / 30.44 for a, b in zip(wins[:-1], wins[1:])]
    assert all(20 < g < 32 for g in gaps_months)       # ~26-month Mars cadence


@pytest.mark.slow
def test_pareto_is_monotonic_with_a_knee():
    e0 = et(START)
    pk = porkchop("EARTH", "MARS BARYCENTER", e0, dep_days=540, tof_range=(120, 400),
                  n_dep=40, n_tof=35)
    front = time_energy_pareto(pk)
    assert len(front) >= 3
    assert all(a["total_ms"] >= b["total_ms"] - 1.0 for a, b in zip(front[:-1], front[1:]))
    assert coherent_knee(front) is not None


@pytest.mark.slow
def test_gmat_export_writes_sun_centered_script(tmp_path):
    t = lambert_transfer("EARTH", "MARS BARYCENTER", et("2026-11-01T00:00:00"), 300.0)
    path = export_transfer_gmat(str(tmp_path / "mars.script"), t)
    assert os.path.exists(path)
    text = open(path).read()
    assert "Origin = Sun" in text
    assert "BeginMissionSequence" in text
    assert "Probe.X = " in text
