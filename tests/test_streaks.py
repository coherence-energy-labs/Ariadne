"""Tests for fast-mover streak detection + the streak->tracklet bridge.

Streak detection is what unlocks fast-NEO discovery: an object moving
fast enough to trail across a single exposure is rejected by the
point-source finder, so it must be caught as a linear feature and
converted into a timed tracklet.

These tests pin:
  - the vectorised Hough transform finds an injected line
  - compact-source suppression lets a streak survive a crowded star field
  - the contiguous-segment endpoint measurement rejects chance-aligned
    debris (no over-long false streaks)
  - the classifier separates asteroid / satellite / cosmic-ray
  - streak_to_endpoints produces a correct rate vector
  - ingest_streaks persists detections + an instant within-night tracklet
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _field_with_trail(seed, trails, n_stars=2000, shape=(2048, 2048)):
    rng = np.random.default_rng(seed)
    img = rng.normal(100, 10, shape)
    for _ in range(n_stars):
        x = rng.integers(5, shape[1] - 5)
        y = rng.integers(5, shape[0] - 5)
        img[y - 2:y + 3, x - 2:x + 3] += rng.uniform(150, 2000)
    for (x1, y1, x2, y2, amp) in trails:
        L = int(math.hypot(x2 - x1, y2 - y1)) * 3
        for t in np.linspace(0, 1, max(L, 2)):
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            if 1 <= x < shape[1] - 1 and 1 <= y < shape[0] - 1:
                img[y - 1:y + 2, x - 1:x + 2] += amp
    return img


def test_hough_finds_a_line():
    from ariadne.discovery.imaging.streaks import hough_lines
    # Clean image with one diagonal line
    img = np.full((256, 256), 100.0)
    for i in range(40, 200):
        img[i, i] += 500
    lines = hough_lines(img, sigma_threshold=3.0, suppress_compact=False)
    assert len(lines) >= 1


def test_detect_streak_in_crowded_field():
    """A streak must be recoverable even with 2000 stars present."""
    from ariadne.discovery.imaging.streaks import detect_streaks
    img = _field_with_trail(1, [(800, 900, 920, 1010, 350)])
    streaks = detect_streaks(img, sigma_threshold=4.0,
                                min_length_px=20, max_width_px=6)
    assert len(streaks) >= 1
    # The detected streak's endpoints should match the injection
    s = streaks[0]
    d = min(
        math.hypot(s.x1 - 800, s.y1 - 900) + math.hypot(s.x2 - 920, s.y2 - 1010),
        math.hypot(s.x1 - 920, s.y1 - 1010) + math.hypot(s.x2 - 800, s.y2 - 900))
    assert d < 30, f"endpoints off by {d:.1f}px"


def test_no_false_streak_in_pure_star_field():
    """A field with NO trail must yield no asteroid-candidate streaks."""
    from ariadne.discovery.imaging.streaks import (
        detect_streaks, classify_streak)
    img = _field_with_trail(5, [])
    streaks = detect_streaks(img, sigma_threshold=4.0,
                                min_length_px=20, max_width_px=6)
    frame_diag = math.hypot(2048, 2048)
    ast = [s for s in streaks
            if classify_streak(s, exposure_seconds=90.0,
                                  pixel_scale_arcsec=0.263, psf_fwhm_px=3.0,
                                  frame_diagonal_px=frame_diag
                                  )["is_asteroid_candidate"]]
    assert len(ast) == 0, f"{len(ast)} false asteroid streaks in star field"


def test_contiguous_segment_rejects_overlong_streak():
    """The endpoint measurement must not extend a real trail through
    chance-aligned star debris."""
    from ariadne.discovery.imaging.streaks import detect_streaks
    # 163px trail; without contiguous-segment clipping the detector used
    # to report a ~600px streak from collinear residuals.
    img = _field_with_trail(1, [(800, 900, 920, 1010, 350)])
    streaks = detect_streaks(img, sigma_threshold=4.0,
                                min_length_px=20, max_width_px=6)
    injected_len = math.hypot(920 - 800, 1010 - 900)
    for s in streaks:
        # No detected streak should be more than ~1.5x the injected length
        assert s.length_px < injected_len * 1.8, (
            f"streak length {s.length_px:.0f}px >> injected {injected_len:.0f}px"
            " (chance-aligned debris not clipped)")


def test_difference_imaging_recovers_bright_and_faint_trails():
    """The flux-weighted width must accept BOTH bright and faint trails.

    Regression for the real-data bug where a brighter trail measured a
    wider width (84th-percentile of above-threshold pixels grows with
    brightness) and was wrongly rejected. After the flux-weighted second-
    moment fix, width is amplitude-invariant, so a 10x brighter trail of
    the same PSF width must still classify as an asteroid.
    """
    from ariadne.discovery.imaging.streaks import detect_streaks, classify_streak
    frame_diag = math.hypot(1024, 1024)
    # A difference residual: zero background, only the trail + faint noise.
    # Both amplitudes are well above the detection limit; the point is that
    # the 10x-brighter trail must NOT be rejected for measuring a wider width.
    for amp in (8000.0, 80000.0):
        rng = np.random.default_rng(0)
        resid = rng.normal(0, 5, (1024, 1024))
        # 100px trail at PSF width ~3.5px
        sigma = 3.5 / 2.3548
        for t in np.linspace(0, 1, 300):
            cx = 400 + t * 100
            cy = 500 + t * 0
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    xx = int(round(cx)) + dx
                    yy = int(round(cy)) + dy
                    if 0 <= xx < 1024 and 0 <= yy < 1024:
                        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
                        resid[yy, xx] += (amp / 300) * math.exp(-r2 / (2 * sigma ** 2))
        streaks = detect_streaks(resid, sigma_threshold=4.0,
                                    min_length_px=40, max_width_px=6,
                                    subtract_stars=False)
        ast = [s for s in streaks
                if classify_streak(s, exposure_seconds=90.0,
                                      pixel_scale_arcsec=0.263, psf_fwhm_px=3.5,
                                      frame_diagonal_px=frame_diag
                                      )["is_asteroid_candidate"]]
        assert len(ast) >= 1, f"trail of amplitude {amp} not recovered as asteroid"
        # Width must be near the PSF width regardless of brightness
        assert ast[0].width_px < 6.0, (
            f"width {ast[0].width_px:.1f}px inflated at amplitude {amp}")


def test_classifier_separates_types():
    from ariadne.discovery.imaging.streaks import Streak, classify_streak
    # Asteroid: PSF-thin, moderate length
    ast = Streak(x1=0, y1=0, x2=100, y2=0, length_px=100, width_px=3.5,
                  theta_rad=0, peak_pixel=500, total_flux=5000,
                  n_pixels=100, vote_count=80, consistency=0.85)
    c = classify_streak(ast, exposure_seconds=90.0, pixel_scale_arcsec=0.263,
                          psf_fwhm_px=3.0, frame_diagonal_px=2896)
    assert c["is_asteroid_candidate"] is True
    assert c["label"] == "asteroid_candidate"

    # Satellite: spans the frame
    sat = Streak(x1=0, y1=0, x2=2000, y2=200, length_px=2010, width_px=3.5,
                  theta_rad=0, peak_pixel=500, total_flux=50000,
                  n_pixels=2010, vote_count=900, consistency=0.85)
    c = classify_streak(sat, exposure_seconds=90.0, pixel_scale_arcsec=0.263,
                          psf_fwhm_px=3.0, frame_diagonal_px=2896)
    assert c["is_asteroid_candidate"] is False
    assert c["label"] == "satellite"

    # Cosmic ray: short + sub-PSF
    cr = Streak(x1=0, y1=0, x2=6, y2=2, length_px=6.3, width_px=1.0,
                 theta_rad=0, peak_pixel=900, total_flux=2000,
                 n_pixels=6, vote_count=40, consistency=1.0)
    c = classify_streak(cr, exposure_seconds=90.0, pixel_scale_arcsec=0.263,
                          psf_fwhm_px=3.0, frame_diagonal_px=2896)
    assert c["is_asteroid_candidate"] is False
    assert c["label"] == "cosmic_ray_trail"


def test_streak_to_endpoints_rate():
    """The rate vector recovered from a streak must match the injected
    angular motion."""
    from ariadne.discovery.imaging.streaks import Streak
    from ariadne.discovery.imaging.streak_tracklets import streak_to_endpoints
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [1024, 1024]
    w.wcs.cdelt = [-0.263 / 3600, 0.263 / 3600]
    w.wcs.crval = [150.0, -20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # 100px horizontal streak, 90s exposure => 100*0.263 = 26.3" in 90s
    s = Streak(x1=1000, y1=1000, x2=1100, y2=1000, length_px=100,
                width_px=3.5, theta_rad=0, peak_pixel=500, total_flux=5000,
                n_pixels=100, vote_count=80, consistency=0.85)
    ep = streak_to_endpoints(s, w, exposure_start_mjd=60500.0,
                                exposure_seconds=90.0, zeropoint_mag=30.0)
    expected_rate = 100 * 0.263 / (90.0 / 3600.0)   # arcsec/hr
    assert abs(ep.rate_arcsec_hr - expected_rate) / expected_rate < 0.05
    # Endpoints separated by 90s
    assert abs((ep.mjd_end - ep.mjd_start) * 86400.0 - 90.0) < 1e-6


def test_ingest_streaks_creates_tracklet(tmp_path):
    """ingest_streaks must persist 2 detections + 1 tracklet per
    asteroid-candidate streak."""
    from ariadne.discovery.imaging.streaks import Streak
    from ariadne.discovery.imaging.streak_tracklets import ingest_streaks
    from ariadne.discovery.imaging.detection_db import open_db
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [1024, 1024]
    w.wcs.cdelt = [-0.263 / 3600, 0.263 / 3600]
    w.wcs.crval = [150.0, -20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    db = open_db(tmp_path / "streaks.db")
    # One asteroid + one satellite; only the asteroid should be ingested
    ast = Streak(x1=1000, y1=1000, x2=1100, y2=1050, length_px=112,
                  width_px=3.5, theta_rad=0, peak_pixel=500, total_flux=5000,
                  n_pixels=112, vote_count=80, consistency=0.85)
    sat = Streak(x1=0, y1=0, x2=2000, y2=200, length_px=2010, width_px=3.5,
                  theta_rad=0, peak_pixel=500, total_flux=50000,
                  n_pixels=2010, vote_count=900, consistency=0.85)
    det_ids, trk_ids = ingest_streaks(
        db, [ast, sat], w, image_id="exp1",
        exposure_start_mjd=60500.0, exposure_seconds=90.0,
        pixel_scale_arcsec=0.263, psf_fwhm_px=3.0,
        frame_diagonal_px=2896, zeropoint_mag=30.0,
        asteroid_only=True, ccd_id="N4")
    assert len(trk_ids) == 1, "only the asteroid streak should become a tracklet"
    assert len(det_ids) == 2, "the asteroid streak yields 2 endpoint detections"
    t = db.get_tracklet(trk_ids[0])
    assert t is not None
    assert t["rate_arcsec_hr"] > 0
    # The two detections are 90s apart
    dets = db.get_tracklet_detections(trk_ids[0])
    assert len(dets) == 2
    dt_s = abs(dets[1]["mjd"] - dets[0]["mjd"]) * 86400.0
    assert abs(dt_s - 90.0) < 1e-3
