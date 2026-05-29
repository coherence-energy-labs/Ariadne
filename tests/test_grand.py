"""Stage 23 tests: unified multi-objective grand optimizer."""
import pytest

from ariadne.data.ephemeris import et
from ariadne.interplanetary.grand import (
    build_tradeoff, rank_by_coherence, most_coherent_route, low_thrust_heliocentric,
)

START = "2026-01-01T00:00:00"


@pytest.mark.slow
def test_tradeoff_has_no_free_lunch_structure():
    tr = build_tradeoff("EARTH", "MARS BARYCENTER", et(START), dep_days=540,
                        tof_range=(120, 400), n_dep=40, n_tof=30)
    cheapest = min(tr, key=lambda p: p["total_ms"])
    fastest = min(tr, key=lambda p: p["tof_days"])
    assert fastest["total_ms"] > cheapest["total_ms"]          # faster costs more energy
    assert fastest["sensitivity_ms_per_day"] >= cheapest["sensitivity_ms_per_day"]  # and is less robust


@pytest.mark.slow
def test_coherence_pick_is_a_middle_ground_and_weights_steer():
    tr = build_tradeoff("EARTH", "MARS BARYCENTER", et(START), dep_days=540,
                        tof_range=(120, 400), n_dep=40, n_tof=30)
    cheapest = min(tr, key=lambda p: p["total_ms"])
    fastest = min(tr, key=lambda p: p["tof_days"])
    balanced = most_coherent_route(tr, (1.0, 1.0, 1.0))
    assert fastest["tof_days"] < balanced["tof_days"] < cheapest["tof_days"]
    assert balanced["total_ms"] > cheapest["total_ms"]
    # weights steer the choice
    energy_first = most_coherent_route(tr, (3.0, 1.0, 0.5))
    time_first = most_coherent_route(tr, (0.5, 3.0, 0.5))
    assert time_first["tof_days"] <= energy_first["tof_days"]


@pytest.mark.slow
def test_low_thrust_estimate_is_sensible():
    lt = low_thrust_heliocentric("EARTH", "MARS BARYCENTER", et(START), accel_mm_s2=0.2)
    assert 2000.0 < lt["dv_ms"] < 8000.0
    assert lt["tof_days"] > 100.0
    assert "excludes" in lt["note"]                            # honest scope caveat present
