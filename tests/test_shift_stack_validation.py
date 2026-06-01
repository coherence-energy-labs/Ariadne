"""Tests for the shift-and-stack IOD validator."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _fake_wcs(ra=180.0, dec=20.0, npix=512, pixscale_arcsec=1.0):
    """Build a minimal TAN WCS for testing."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.crval = [ra, dec]
    pixscale_deg = pixscale_arcsec / 3600.0
    w.wcs.cd = [[-pixscale_deg, 0.0], [0.0, pixscale_deg]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _planted_image(npix=64, signal_xy=None, signal_amp=2000, bg_mean=100,
                     bg_sigma=10, seed=0):
    """Make an image with a known PSF at signal_xy=(x,y)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(bg_mean, bg_sigma, size=(npix, npix)).astype(float)
    if signal_xy is not None:
        x, y = signal_xy
        ix, iy = int(round(x)), int(round(y))
        half = 6
        x0, x1 = max(0, ix - half), min(npix, ix + half + 1)
        y0, y1 = max(0, iy - half), min(npix, iy + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sigma = 1.5
        g = signal_amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += g
    return img


def test_crop_returns_correct_shape():
    from ariadne.discovery.imaging.shift_stack_validation import _crop
    img = np.arange(20 * 20).reshape(20, 20).astype(float)
    patch = _crop(img, x_c=10, y_c=10, half_size=4)
    assert patch.shape == (9, 9)


def test_crop_handles_out_of_bounds():
    from ariadne.discovery.imaging.shift_stack_validation import _crop
    img = np.ones((10, 10))
    # Centre is outside; expect NaNs at out-of-bounds pixels
    patch = _crop(img, x_c=20, y_c=20, half_size=3)
    assert patch.shape == (7, 7)
    assert np.isnan(patch).all()


def test_aperture_snr_detects_planted_signal():
    from ariadne.discovery.imaging.shift_stack_validation import measure_aperture_snr
    img = _planted_image(npix=32, signal_xy=(16, 16), signal_amp=2000, seed=42)
    _, snr = measure_aperture_snr(img, aperture_radius=3)
    assert snr > 5.0   # 5-sigma minimum for a 2000-count source


def test_aperture_snr_zero_on_pure_noise():
    from ariadne.discovery.imaging.shift_stack_validation import measure_aperture_snr
    img = _planted_image(npix=32, signal_xy=None, seed=42)
    _, snr = measure_aperture_snr(img, aperture_radius=3)
    # No real signal -> SNR should be low (under 3)
    assert snr < 3.0


def test_shift_stack_boosts_snr_when_aligned():
    """N aligned images should give ~sqrt(N) SNR boost over a single one."""
    from ariadne.discovery.imaging.shift_stack_validation import (
        shift_stack, measure_aperture_snr)
    # Plant the same signal in 4 images at different positions; predicted
    # positions match exactly so the stack is perfectly aligned.
    images = []
    positions = []
    for k in range(4):
        x, y = 30 + k * 2, 40 + k * 3   # different positions per image
        images.append(_planted_image(npix=80, signal_xy=(x, y),
                                       signal_amp=2000, seed=k))
        positions.append((x, y, True))
    _, snr_single = measure_aperture_snr(
        _crop_helper(images[0], positions[0][0], positions[0][1]),
        aperture_radius=3)
    stacked = shift_stack(images, positions, half_size=12)
    _, snr_stacked = measure_aperture_snr(stacked, aperture_radius=3)
    # Stack of 4 aligned -> boost should be at least 1.3x (theory sqrt(4)=2)
    assert snr_stacked >= 1.3 * snr_single


def _crop_helper(img, x, y, half_size=12):
    from ariadne.discovery.imaging.shift_stack_validation import _crop
    return _crop(img, x, y, half_size)


def test_shift_stack_no_boost_when_misaligned():
    """If predicted positions all point at pure noise, the stacked SNR
    should stay near zero -- no spurious signal builds up from noise alone."""
    from ariadne.discovery.imaging.shift_stack_validation import (
        shift_stack, measure_aperture_snr)
    # Plant signal at (40, 50) in 4 images but PREDICT it far away in
    # each image (so every prediction points at pure noise -- no shared
    # source at the predicted positions).
    images = []
    positions = []
    far_away_positions = [(10, 10), (10, 70), (70, 10), (70, 70)]
    for k in range(4):
        images.append(_planted_image(npix=80, signal_xy=(40, 50),
                                       signal_amp=2000, seed=k))
        positions.append((*far_away_positions[k], True))
    stacked = shift_stack(images, positions, half_size=12)
    _, snr_stacked = measure_aperture_snr(stacked, aperture_radius=3)
    # Mostly noise stack -> SNR should stay near zero
    assert snr_stacked < 5.0
