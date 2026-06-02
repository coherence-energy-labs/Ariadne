"""Tests for difference imaging, esp. the Alard-Lupton PSF-matched path.

The crude scalar subtraction leaves a bright/dark DIPOLE at every static star
when the two epochs have different seeing -- those dipoles are false positives
and raise the local noise, which is why crude differencing did not improve
real-data recall. The PSF-matched kernel must (a) cancel the static field to
the noise (no dipoles) and (b) leave an injected moving source standing as a
clear positive detection. We verify both, and for both PSF orderings
(reference sharper than science, and science sharper than reference).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("photutils")


def _gauss(img, x, y, flux, sigma):
    H, W = img.shape
    half = int(math.ceil(5 * sigma))
    x0 = max(0, int(x) - half); x1 = min(W, int(x) + half + 1)
    y0 = max(0, int(y) - half); y1 = min(H, int(y) + half + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += (flux / (2 * math.pi * sigma ** 2)) * np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def _star_grid(exclude_center=True):
    """Deterministic 6x6 grid of static-star positions, optional empty center."""
    coords = [40, 85, 130, 175, 220, 265]
    stars = [(x, y) for x in coords for y in coords]
    if exclude_center:
        # drop the star nearest the field center so a mover can sit there alone
        stars = [s for s in stars if not (s[0] == 130 and s[1] == 130)]
    return stars


def _field(shape, stars, fluxes, sigma, bg=100.0, rn=8.0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, rn, shape)
    for (x, y), f in zip(stars, fluxes):
        _gauss(img, x, y, f, sigma)
    return img


def _star_core_rms(result, stars):
    """RMS of residual/noise sampled at the static-star cores. Near 1 means the
    field cancelled to the noise; >> 1 means dipole residuals survive."""
    vals = []
    for (x, y) in stars:
        xi, yi = int(round(x)), int(round(y))
        vals.append(result.residual[yi, xi] / max(result.noise[yi, xi], 1e-6))
    return float(np.sqrt(np.mean(np.square(vals))))


def _mover_snr(result, mx, my):
    return result.residual[int(my), int(mx)] / max(result.noise[int(my), int(mx)], 1e-6)


def test_psf_matched_cancels_dipoles_reference_sharper():
    from ariadne.discovery.imaging.difference import (
        subtract_reference, psf_matched_difference)
    shape = (300, 300)
    stars = _star_grid()
    rng = np.random.default_rng(7)
    fluxes = rng.uniform(4e4, 2e5, len(stars))
    # reference SHARPER (1.5) + 15% brighter than science (2.4 seeing)
    sci = _field(shape, stars, fluxes, sigma=2.4, seed=1)
    ref = _field(shape, stars, fluxes * 1.15, sigma=1.5, seed=2)
    mx, my = 130.0, 130.0           # the empty grid center
    _gauss(sci, mx, my, 6e4, 2.4)   # mover, science only

    crude = subtract_reference(sci, ref)
    psf = psf_matched_difference(sci, ref)

    assert psf.method == "psf-matched"
    assert psf.n_stars >= 6
    # the kernel must cancel the static field far better than crude scalar
    assert _star_core_rms(psf, stars) < 0.5 * _star_core_rms(crude, stars)
    # and the mover must still stand out as a clear positive detection
    assert _mover_snr(psf, mx, my) > 5.0


def test_psf_matched_cancels_dipoles_science_sharper():
    from ariadne.discovery.imaging.difference import (
        subtract_reference, psf_matched_difference)
    shape = (300, 300)
    stars = _star_grid()
    rng = np.random.default_rng(11)
    fluxes = rng.uniform(4e4, 2e5, len(stars))
    # science SHARPER (1.5) than reference (2.6) -> exercises the Case-B branch
    sci = _field(shape, stars, fluxes, sigma=1.5, seed=3)
    ref = _field(shape, stars, fluxes * 0.9, sigma=2.6, seed=4)
    mx, my = 130.0, 130.0
    _gauss(sci, mx, my, 6e4, 1.5)   # mover, science only

    crude = subtract_reference(sci, ref)
    psf = psf_matched_difference(sci, ref)

    assert psf.method == "psf-matched"
    assert _star_core_rms(psf, stars) < 0.6 * _star_core_rms(crude, stars)
    assert _mover_snr(psf, mx, my) > 5.0


def test_psf_matched_falls_back_when_no_stars():
    """A pure-noise pair has no stars to fit -> graceful crude fallback."""
    from ariadne.discovery.imaging.difference import psf_matched_difference
    rng = np.random.default_rng(0)
    sci = rng.normal(100.0, 8.0, (120, 120))
    ref = rng.normal(100.0, 8.0, (120, 120))
    res = psf_matched_difference(sci, ref, min_stars=6)
    assert res.method == "scalar"   # fell back, did not crash


def test_kernel_flux_scale_tracks_brightness_ratio():
    """The fitted kernel's integral should approximate the science/reference
    stellar flux ratio (it absorbs the photometric scale)."""
    from ariadne.discovery.imaging.difference import psf_matched_difference
    shape = (300, 300)
    stars = _star_grid(exclude_center=False)
    rng = np.random.default_rng(5)
    fluxes = rng.uniform(5e4, 2e5, len(stars))
    ratio = 1.25
    sci = _field(shape, stars, fluxes * ratio, sigma=2.2, seed=1)
    ref = _field(shape, stars, fluxes, sigma=1.6, seed=2)
    psf = psf_matched_difference(sci, ref)
    assert psf.method == "psf-matched"
    # kernel integral ~ flux ratio (loose: photon noise + basis truncation)
    assert 0.9 * ratio < psf.flux_scale < 1.15 * ratio
