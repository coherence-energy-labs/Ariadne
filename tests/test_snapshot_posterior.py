"""Tests for the calibrated reparametrized single-snapshot posterior."""
from __future__ import annotations

import math

import numpy as np
import pytest

R_EARTH = np.array([1.496e8, 0.0, 0.0])
LOS_OPP = np.array([1.0, 0.0, 0.0])     # anti-Sun (opposition)


def test_main_belt_posterior_brackets_truth():
    from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
    # 35 "/hr near opposition -> main-belt; CI should bracket ~1.5-2 AU
    post = snapshot_posterior(35.0, 18.0, R_EARTH, LOS_OPP, n=8000)
    assert post.near_opposition
    assert post.distance_lo < post.distance_med < post.distance_hi
    assert 1.0 < post.helio_r_med < 3.5
    assert list(post.class_probs)[0] in ("main-belt", "Mars-crosser/inner-belt")
    assert not post.incomer_flag


def test_faster_rate_gives_closer_distance():
    from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
    slow = snapshot_posterior(20.0, 18.0, R_EARTH, LOS_OPP, n=8000)
    fast = snapshot_posterior(80.0, 18.0, R_EARTH, LOS_OPP, n=8000)
    assert fast.distance_med < slow.distance_med, "faster -> nearer"


def test_incomer_flag_bright_slow():
    from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
    post = snapshot_posterior(5.0, 18.0, R_EARTH, LOS_OPP, n=8000)
    assert post.incomer_flag
    assert "INCOMER" in post.note


def test_posterior_is_deterministic():
    from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
    a = snapshot_posterior(40.0, 19.0, R_EARTH, LOS_OPP, n=5000)
    b = snapshot_posterior(40.0, 19.0, R_EARTH, LOS_OPP, n=5000)
    assert a.distance_med == b.distance_med


def test_off_opposition_flagged():
    from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
    los_quad = np.array([0.0, 1.0, 0.0])   # 90 deg from anti-Sun
    post = snapshot_posterior(35.0, 18.0, R_EARTH, los_quad, n=4000)
    assert not post.near_opposition
    assert "opposition" in post.note.lower()
