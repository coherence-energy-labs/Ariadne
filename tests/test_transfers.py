"""Low-energy transfer tests (Gate G8r / G8t, Stage 6)."""
import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.orbits.families import lyapunov_orbit_at_jacobi
from ariadne.optimize.budget import earth_moon_budget
from ariadne.transfers.lunar_capture import ballistic_capture
from ariadne.transfers.low_energy_lunar import low_energy_lunar_transfer

MU = EARTH_MOON.mu


@pytest.mark.slow
def test_ballistic_capture_beats_direct():
    orb = lyapunov_orbit_at_jacobi(MU, "L1", 3.15)
    cap = ballistic_capture(orb, llo_alt=100.0)
    assert cap is not None
    # reaches LLO altitude, near-parabolic arrival, cheaper than direct hyperbolic LOI
    assert abs(cap["periapsis_alt_km"] - 100.0) < 60.0
    assert 0.5 < cap["dv_capture_kms"] < 0.8
    assert cap["v_peri_kms"] < 2.40                     # at/below parabolic (2.310)
    direct_loi = earth_moon_budget()["dv_loi_direct"]
    assert cap["dv_capture_kms"] < direct_loi


@pytest.mark.slow
def test_low_energy_total_brackets_coimbra():
    best, recs, base = low_energy_lunar_transfer()
    assert best is not None and len(recs) >= 4
    assert 3600.0 < best["total_ms"] < 3960.0
    # genuine saving vs the direct hyperbolic-capture transfer
    assert base["direct_total_ms"] - best["total_ms"] > 100.0
    # all candidate captures arrive near LLO altitude
    for r in recs:
        assert 0 < r["periapsis_alt_km"] < 250.0
