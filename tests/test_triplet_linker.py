"""Tests for the within-night triplet (3-point collinear) tracklet linker.

Pins the property that makes within-night linking clean: a real moving
object (3 collinear, constant-rate points) is recovered, while static
stars and chance non-collinear triples are rejected.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _moving_object(ra0, dec0, rate_deg_day, pa_deg, times):
    """Positions of one constant-rate mover at each time."""
    cosd = math.cos(math.radians(dec0))
    out = []
    for t in times:
        dt = t - times[0]
        d = rate_deg_day * dt
        dra = d * math.sin(math.radians(pa_deg)) / cosd
        ddec = d * math.cos(math.radians(pa_deg))
        out.append((ra0 + dra, dec0 + ddec))
    return out


def test_recovers_a_real_mover_among_static_stars():
    from ariadne.discovery.imaging.triplet_linker import link_collinear_tracklets
    rng = np.random.default_rng(0)
    times = [60428.2512, 60428.2917, 60428.3440]
    # 200 static stars (same position every epoch) + 1 mover
    star_ra = 210.0 + rng.uniform(-0.1, 0.1, 200)
    star_dec = -12.0 + rng.uniform(-0.1, 0.1, 200)
    mover = _moving_object(210.05, -12.02, 0.3, 60.0, times)
    epochs = []
    for ei, t in enumerate(times):
        ra = np.append(star_ra, mover[ei][0])
        dec = np.append(star_dec, mover[ei][1])
        epochs.append((ra, dec, t))
    trk = link_collinear_tracklets(epochs, collinear_tol_arcsec=1.0)
    # The mover (last index in each epoch) must be recovered as a tracklet
    assert len(trk) >= 1
    movers = [t for t in trk if t.idx == (200, 200, 200)]
    assert len(movers) == 1, f"mover not cleanly recovered ({len(trk)} tracklets)"
    assert abs(movers[0].rate_deg_day - 0.3) < 0.02
    assert abs((movers[0].pa_deg - 60.0 + 180) % 360 - 180) < 5


def test_same_star_zero_motion_makes_no_tracklet():
    """The SAME star at the SAME position across epochs has zero motion and
    must fall below the min-rate cut (it is not a mover)."""
    from ariadne.discovery.imaging.triplet_linker import link_collinear_tracklets
    times = [60428.25, 60428.29, 60428.34]
    # A handful of well-separated stars, identical positions every epoch.
    ra = np.array([210.00, 210.05, 209.95, 210.02])
    dec = np.array([-12.00, -12.03, -11.97, -12.06])
    epochs = [(ra.copy(), dec.copy(), t) for t in times]
    trk = link_collinear_tracklets(epochs, min_rate_deg_day=0.02,
                                      collinear_tol_arcsec=1.0)
    # No same-star match is a mover; only cross-star chance triples could
    # appear, and these well-separated stars admit none.
    assert len(trk) == 0


def test_chance_alignment_contamination_is_controllable():
    """Chance triples of DIFFERENT stars CAN align into a constant-rate
    track (this is the real single-night contamination). It must shrink as
    the collinearity tolerance tightens -- the lever a real pipeline uses
    (with Gaia-refined astrometry) plus a confirming second night."""
    from ariadne.discovery.imaging.triplet_linker import link_collinear_tracklets
    rng = np.random.default_rng(1)
    times = [60428.25, 60428.29, 60428.34]
    ra = 210.0 + rng.uniform(-0.1, 0.1, 300)
    dec = -12.0 + rng.uniform(-0.1, 0.1, 300)
    epochs = [(ra.copy(), dec.copy(), t) for t in times]
    n_loose = len(link_collinear_tracklets(epochs, collinear_tol_arcsec=2.0))
    n_tight = len(link_collinear_tracklets(epochs, collinear_tol_arcsec=0.3))
    assert n_tight < n_loose, (
        f"tightening tolerance should cut chance triples "
        f"(loose={n_loose}, tight={n_tight})")
    assert n_tight <= n_loose / 2, "tightening should cut chance triples sharply"


def test_non_collinear_triple_rejected():
    from ariadne.discovery.imaging.triplet_linker import link_collinear_tracklets
    times = [60428.25, 60428.29, 60428.34]
    # 3 points that move but NOT on a straight constant-rate line:
    # epoch2 is offset perpendicular by 10" from the predicted line.
    e0 = (np.array([210.0]), np.array([-12.0]), times[0])
    e1 = (np.array([210.0 + 0.01]), np.array([-12.0 + 0.01]), times[1])
    # predicted epoch2 (constant rate) would be at ~+0.0129 each; bend it
    e2 = (np.array([210.0 + 0.0129]), np.array([-12.0 + 0.0129 - 10/3600.0]),
          times[2])
    trk = link_collinear_tracklets([e0, e1, e2], collinear_tol_arcsec=1.0)
    assert len(trk) == 0, "non-collinear triple should be rejected"


def test_rate_window_excludes_too_fast():
    from ariadne.discovery.imaging.triplet_linker import link_collinear_tracklets
    times = [60428.25, 60428.29, 60428.34]
    # A satellite-fast track at 5 deg/day, above max_rate_deg_day=0.8
    mover = _moving_object(210.0, -12.0, 5.0, 30.0, times)
    epochs = [(np.array([mover[ei][0]]), np.array([mover[ei][1]]), t)
              for ei, t in enumerate(times)]
    trk = link_collinear_tracklets(epochs, max_rate_deg_day=0.8,
                                      collinear_tol_arcsec=1.0)
    assert len(trk) == 0, "object above the rate window should be excluded"
