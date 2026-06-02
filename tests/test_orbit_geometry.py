"""Tests for single-snapshot orbit geometry (the VALIDATED estimator).

The opposition rate<->distance relation and the single-snapshot distance/
class/incomer estimate were validated on real DECam asteroids
(85% CI-calibrated, ~0.22 AU median error, 79% orbit-class correct).
These tests pin the relation's correctness, monotonicity, round-tripping,
classification, and the incomer-flag logic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_opposition_rate_matches_main_belt():
    """A 2.5 AU main-belt object at opposition should move ~870 "/day
    (~36 "/hr), the value we measured for real main-belt asteroids."""
    from ariadne.discovery.imaging.orbit_geometry import opposition_rate
    r = opposition_rate(2.5)
    assert 800 < r < 950, f"main-belt opposition rate {r:.0f} '/day off"


def test_opposition_rate_monotonic_decreasing():
    from ariadne.discovery.imaging.orbit_geometry import opposition_rate
    rates = [opposition_rate(r) for r in [1.5, 2.0, 2.5, 3.0, 5.0, 40.0]]
    assert all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))


@pytest.mark.parametrize("r_true", [1.5, 2.0, 2.5, 3.2, 5.2, 30.0])
def test_rate_distance_round_trip(r_true):
    from ariadne.discovery.imaging.orbit_geometry import (
        opposition_rate, opposition_rate_to_distance)
    rate = opposition_rate(r_true)
    r_back = opposition_rate_to_distance(rate)
    assert abs(r_back - r_true) < 0.02, f"{r_true} -> {rate:.0f} -> {r_back}"


def test_distance_classification():
    from ariadne.discovery.imaging.orbit_geometry import classify_by_distance
    assert classify_by_distance(1.0) == "NEO/inner"
    assert classify_by_distance(2.6) == "main-belt"
    assert classify_by_distance(5.2) == "outer-belt/Hilda/Trojan"
    assert classify_by_distance(40.0) == "TNO/distant"


def test_solar_elongation_opposition():
    from ariadne.discovery.imaging.orbit_geometry import solar_elongation_deg
    # observer 1 AU from Sun at +x; an object on the anti-Sun line of sight
    R = np.array([1.496e8, 0.0, 0.0])      # Earth heliocentric (km)
    los = np.array([1.0, 0.0, 0.0])        # looking away from Sun
    assert abs(solar_elongation_deg(R, los) - 180.0) < 1.0
    los2 = np.array([-1.0, 0.0, 0.0])      # looking toward Sun
    assert abs(solar_elongation_deg(R, los2) - 0.0) < 1.0


def test_single_snapshot_main_belt():
    """A 36 "/hr object near opposition -> main-belt-ish distance + class."""
    from ariadne.discovery.imaging.orbit_geometry import single_snapshot_estimate
    R = np.array([1.496e8, 0.0, 0.0])
    los = np.array([1.0, 0.0, 0.0])        # opposition
    est = single_snapshot_estimate(36.0, 18.0, R, los)
    assert est.near_opposition
    assert 1.5 < est.helio_r_au < 3.5
    assert est.orbit_class in ("main-belt", "Mars-crosser/inner-belt")
    assert not est.incomer_flag


def test_incomer_flag_for_bright_slow_object():
    """A bright, slow (untrailed) object inverts to a large distance where
    the implied H is implausibly bright -> flagged as a possible nearby/
    incoming body."""
    from ariadne.discovery.imaging.orbit_geometry import single_snapshot_estimate
    R = np.array([1.496e8, 0.0, 0.0])
    los = np.array([1.0, 0.0, 0.0])
    est = single_snapshot_estimate(5.0, 18.0, R, los)   # very slow + bright
    assert est.incomer_flag, f"should flag incomer (H={est.implied_H:.1f})"
    assert "INCOMER" in est.note


def test_implied_H_at_opposition():
    from ariadne.discovery.imaging.orbit_geometry import implied_absolute_magnitude
    # main-belt: V=18 at r=3, Delta=2 -> H = 18 - 5 log10(6) = 18 - 3.89
    H = implied_absolute_magnitude(18.0, 3.0, 2.0, phase_deg=0.0)
    assert abs(H - (18.0 - 5 * math.log10(6.0))) < 1e-6
