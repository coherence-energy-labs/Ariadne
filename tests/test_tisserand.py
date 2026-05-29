"""Stage 20 tests: Tisserand-graph multi-moon tour."""
import math

from ariadne.data.constants import GM_JUPITER
from ariadne.transfers.tisserand import (
    tisserand_parameter, vinf_at_moon, connecting_transfer, moon_tour, GALILEAN_TOUR,
)


def test_vinf_zero_for_cotangent_circular_orbit():
    """A Jupiter-centric circular orbit AT a moon's radius has v_inf = 0 (T = 3)."""
    a_m = 671100.0
    assert abs(tisserand_parameter(a_m, 0.0, a_m) - 3.0) < 1e-12
    assert vinf_at_moon(a_m, 0.0, a_m) < 1e-6


def test_connecting_transfer_touches_both_radii():
    a, e = connecting_transfer(421800.0, 1882700.0)
    rp, ra = a * (1 - e), a * (1 + e)
    assert abs(rp - 421800.0) < 1e-6
    assert abs(ra - 1882700.0) < 1e-6


def test_vinf_in_galilean_regime():
    """Connecting-transfer v_inf at the Galilean moons is ~1-2 km/s (the real tour regime)."""
    t = moon_tour()
    for leg in t["legs"]:
        assert 0.5 < leg["vinf_inner_kms"] < 3.0
        assert 0.5 < leg["vinf_outer_kms"] < 3.0


def test_gravity_assist_beats_hohmann_by_an_order_of_magnitude():
    t = moon_tour()
    assert t["ga_deterministic_dv_ms"] < 0.1 * t["hohmann_dv_ms"]
    assert t["hohmann_dv_ms"] > 5000.0                      # the propulsive baseline is large
    # every flyby has ample turn authority
    for leg in t["legs"]:
        assert leg["turn_inner_deg"] > 20.0 and leg["turn_outer_deg"] > 20.0
