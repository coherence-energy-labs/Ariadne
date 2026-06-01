"""Tests for the photometric-identifier module."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def _make_chain(mags, t_starts=None):
    if t_starts is None:
        t_starts = [_et(60450 + k * 3.0) for k in range(len(mags))]
    return [{"t": t_starts[k], "ra": math.radians(180.0),
              "dec": math.radians(20.0),
              "rate_arcsec_hr": 2.0, "mag": mags[k]}
             for k in range(len(mags))]


def test_lightcurve_features_constant_brightness():
    from ariadne.discovery.imaging.photometric_identifier import lightcurve_features
    ch = _make_chain([21.5, 21.5, 21.5, 21.5])
    feat = lightcurve_features(ch)
    assert feat.n_observations == 4
    assert feat.mag_median == 21.5
    assert feat.mag_std == 0.0
    assert feat.n_outliers == 0


def test_lightcurve_features_variable_brightness():
    from ariadne.discovery.imaging.photometric_identifier import lightcurve_features
    ch = _make_chain([21.0, 22.0, 23.0, 22.5])
    feat = lightcurve_features(ch)
    assert feat.n_observations == 4
    assert feat.mag_std > 0.5


def test_lightcurve_features_handles_empty():
    from ariadne.discovery.imaging.photometric_identifier import lightcurve_features
    feat = lightcurve_features([])
    assert feat.n_observations == 0


def test_photometric_chain_score_high_for_constant_brightness():
    from ariadne.discovery.imaging.photometric_identifier import photometric_chain_score
    ch = _make_chain([21.5] * 5)
    score = photometric_chain_score(ch)
    assert score > 0.8


def test_photometric_chain_score_low_for_variable_brightness():
    from ariadne.discovery.imaging.photometric_identifier import photometric_chain_score
    ch = _make_chain([20.0, 22.0, 24.0, 22.0, 20.0])
    score = photometric_chain_score(ch)
    assert score < 0.5


def test_correlate_lightcurves_high_for_similar_chains():
    from ariadne.discovery.imaging.photometric_identifier import correlate_lightcurves
    ch1 = _make_chain([21.5, 21.5, 21.5])
    ch2 = _make_chain([21.5, 21.5, 21.5])
    sim = correlate_lightcurves(ch1, ch2)
    assert sim >= 0.85    # identical chains; periodic+smoothness components soften it from 1.0


def test_correlate_lightcurves_low_for_different_brightness():
    from ariadne.discovery.imaging.photometric_identifier import correlate_lightcurves
    ch_bright = _make_chain([18.0, 18.0, 18.0])
    ch_faint = _make_chain([24.0, 24.0, 24.0])
    sim = correlate_lightcurves(ch_bright, ch_faint)
    # Bright vs faint chains: magnitude similarity is ~0, period+smoothness
    # are both neutral/identical at ~0.5+1.0, so the score floors at ~0.3.
    assert sim <= 0.35


def test_match_chains_groups_similar():
    from ariadne.discovery.imaging.photometric_identifier import (
        match_chains_photometrically)
    # 4 chains: 2 bright, 2 faint
    chains = [_make_chain([18.0, 18.0, 18.0]),
                _make_chain([18.05, 18.05, 18.05]),
                _make_chain([24.0, 24.0, 24.0]),
                _make_chain([24.05, 24.05, 24.05])]
    groups = match_chains_photometrically(chains, similarity_threshold=0.7)
    # Should produce 2 groups, each of size 2
    sizes = sorted([len(g) for g in groups.values()])
    assert sizes == [2, 2]


def test_detect_period_runs_without_error():
    """Mostly just make sure the periodogram doesn't crash on small inputs."""
    from ariadne.discovery.imaging.photometric_identifier import _detect_period
    # 5 points, no real period
    times = [0, 6, 12, 18, 24]
    mags = [21.5, 21.5, 21.5, 21.5, 21.5]
    P, sig = _detect_period(times, mags)
    assert math.isfinite(P)
    assert math.isfinite(sig)
