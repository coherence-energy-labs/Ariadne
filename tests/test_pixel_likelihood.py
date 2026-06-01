"""Tests for the pixel-likelihood orbit refiner."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def _planted_patch(npix=17, x0=8.0, y0=8.0, amp=2000, sigma=1.5,
                     bg=100, noise=10, seed=0):
    """Synthesise a Gaussian PSF patch + Gaussian noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((npix, npix))
    img = (bg + amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
            + rng.normal(0, noise, size=(npix, npix)))
    return img


def test_patch_log_likelihood_higher_at_true_position():
    from ariadne.discovery.imaging.pixel_likelihood import patch_log_likelihood
    patch = _planted_patch(x0=8.0, y0=8.0, seed=0)
    # Likelihood at the true centre vs offset
    ll_true = patch_log_likelihood(patch, x0=8.0, y0=8.0, sigma_psf=1.5)
    ll_off = patch_log_likelihood(patch, x0=3.0, y0=3.0, sigma_psf=1.5)
    assert ll_true > ll_off


def test_patch_log_likelihood_finite_on_pure_noise():
    from ariadne.discovery.imaging.pixel_likelihood import patch_log_likelihood
    # Pure noise (no signal)
    rng = np.random.default_rng(0)
    patch = 100 + rng.normal(0, 10, size=(17, 17))
    ll = patch_log_likelihood(patch, x0=8.0, y0=8.0, sigma_psf=1.5)
    assert math.isfinite(ll)


def test_patch_log_likelihood_handles_empty():
    from ariadne.discovery.imaging.pixel_likelihood import patch_log_likelihood
    ll = patch_log_likelihood(np.array([]).reshape(0, 0), 0, 0)
    assert ll == 0.0


def test_gaussian_psf_peak_at_centre():
    from ariadne.discovery.imaging.pixel_likelihood import _gaussian_psf
    yy, xx = np.indices((11, 11))
    g = _gaussian_psf(xx, yy, x0=5, y0=5, amp=1000, sigma=1.5)
    assert g[5, 5] == pytest.approx(1000.0)
    assert g[0, 0] < 10  # tail


def test_crop_patch_returns_translated_centre():
    from ariadne.discovery.imaging.pixel_likelihood import _crop_patch
    img = np.arange(50 * 50).reshape(50, 50).astype(float)
    patch, xc_p, yc_p = _crop_patch(img, x_c=25.3, y_c=30.7, half_size=4)
    assert patch.shape == (9, 9)
    # The (xc_p, yc_p) should be near the patch centre (4, 4) plus subpixel offset
    assert 3.5 <= xc_p <= 4.5
    assert 3.5 <= yc_p <= 4.5


def test_refine_orbit_returns_valid_result_on_empty_images():
    """No images -> should not raise, should return the initial state."""
    from ariadne.discovery.imaging.pixel_likelihood import (
        refine_orbit_against_pixels)
    x_init = np.array([1.5e8, 0.0, 0.0])
    v_init = np.array([0.0, 29.78, 0.0])
    result = refine_orbit_against_pixels(
        x_init, v_init, _et(60450.0),
        images=[], wcs_list=[], image_ets=[],
        max_iter=10)
    # Should not raise; result type intact
    assert hasattr(result, "x_refined")
    assert hasattr(result, "v_refined")
