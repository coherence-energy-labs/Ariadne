"""Tests for shift-and-stack synthetic tracking."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _gaussian_image(npix, x0, y0, amp, sigma=1.5, bg=100.0, bg_sigma=10.0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, bg_sigma, size=(npix, npix))
    half = 6
    ix, iy = int(round(x0)), int(round(y0))
    yi0, yi1 = max(0, iy - half), min(npix, iy + half + 1)
    xi0, xi1 = max(0, ix - half), min(npix, ix + half + 1)
    yy, xx = np.mgrid[yi0:yi1, xi0:xi1]
    g = amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    img[yi0:yi1, xi0:xi1] += g
    return img


def test_shift_image_zero_shift_is_identity():
    from ariadne.discovery.imaging.synthetic_tracking import shift_image_bilinear
    img = _gaussian_image(50, 25, 25, 2000, seed=1)
    out = shift_image_bilinear(img, 0.0, 0.0)
    finite = np.isfinite(out)
    np.testing.assert_allclose(out[finite], img[finite], atol=1e-6)


def test_shift_image_moves_signal():
    """Shifting by (5, 3) should put a centered Gaussian at the new offset."""
    from ariadne.discovery.imaging.synthetic_tracking import shift_image_bilinear
    img = _gaussian_image(50, 25, 25, 2000, seed=2)
    # Shifting by +5 in x should make pixel (20, 25) hold the value
    # that was originally at (25, 25) (since out[i,j] takes from in[i, j+5]).
    out = shift_image_bilinear(img, 5.0, 0.0)
    # The signal originally at (25, 25) should now be at (25, 20)
    assert out[25, 20] > out[25, 30]
    assert out[25, 20] > 1000   # signal preserved


def test_shift_and_stack_aligned_signal_boosts_snr():
    """4 images, each with a Gaussian at different positions, stacked with
    the correct alignment shifts -> peak at reference position."""
    from ariadne.discovery.imaging.synthetic_tracking import (
        shift_and_stack, _aperture_snr)
    images = []
    positions = []
    # Signal moves diagonally at rate 5 pix per frame
    for k in range(4):
        x, y = 25 + 5 * k, 25 + 3 * k
        images.append(_gaussian_image(60, x, y, 1500, seed=10 + k))
        positions.append((x, y))
    # Stack with proper shifts: each image needs to be shifted so signal
    # lines up at the reference (25, 25).
    shifts = [(p[0] - 25, p[1] - 25) for p in positions]
    coadd = shift_and_stack(images, shifts)
    # SNR at reference pixel should be very high (4 aligned signals)
    patch = coadd[15:36, 15:36]   # 21x21 around reference
    snr = _aperture_snr(patch, aperture_radius=3)
    assert snr > 8.0


def test_shift_and_stack_misaligned_no_boost():
    """Wrong shifts -> signals don't pile up at the reference pixel."""
    from ariadne.discovery.imaging.synthetic_tracking import (
        shift_and_stack, _aperture_snr)
    images = []
    # Signal moves diagonally but stays AWAY from (50, 50) -- so zero-shift
    # coadd has no signal at the reference centre.
    for k in range(4):
        x, y = 5 + 4 * k, 5 + 4 * k   # spans (5,5) to (17,17)
        images.append(_gaussian_image(60, x, y, 1500, seed=20 + k))
    # WRONG shifts (zero) -> signals don't align at (50, 50)
    coadd = shift_and_stack(images, [(0, 0)] * 4)
    # Probe a patch around (50, 50) where there is no planted signal
    patch = coadd[40:61, 40:61]
    snr = _aperture_snr(patch, aperture_radius=3)
    # Empty-region stack -> SNR near 0 (noise only)
    assert snr < 5.0


def test_find_peaks_in_coadd_finds_planted_signal():
    """Plant a strong PSF + Gaussian noise. find_peaks_in_coadd should find it."""
    from ariadne.discovery.imaging.synthetic_tracking import find_peaks_in_coadd
    coadd = _gaussian_image(80, 40, 50, 5000, sigma=1.5, bg=100, bg_sigma=8, seed=42)
    peaks = find_peaks_in_coadd(coadd, snr_threshold=5.0,
                                  aperture_radius=3, min_separation_pix=8,
                                  n_top=5)
    assert len(peaks) >= 1
    # Closest peak should be near (40, 50)
    d = [math.hypot(p["x"] - 40, p["y"] - 50) for p in peaks]
    assert min(d) < 3   # within 3 pix


def test_find_peaks_no_signal_returns_few():
    """Pure noise -> few or zero peaks above 5-sigma."""
    from ariadne.discovery.imaging.synthetic_tracking import find_peaks_in_coadd
    rng = np.random.default_rng(0)
    coadd = rng.normal(100, 10, size=(80, 80))
    peaks = find_peaks_in_coadd(coadd, snr_threshold=5.0)
    # Pure noise rarely has 5-sigma peaks; allow a tiny background rate
    assert len(peaks) <= 3


def test_predicted_shift_zero_dt_is_zero():
    from ariadne.discovery.imaging.synthetic_tracking import predicted_shift
    dx, dy = predicted_shift(60450.0, 60450.0, 5.0, 90.0, 1.0)
    assert abs(dx) < 1e-9
    assert abs(dy) < 1e-9


def test_predicted_shift_east_motion_negative_x():
    """East-moving object in CD11<0 WCS has DECREASING pixel-x. To shift
    the image so it lands back at the reference, dx must be NEGATIVE
    (read from input[i, j + dx] means dx<0 reads from a smaller column,
    which is where the east-moved object sits)."""
    from ariadne.discovery.imaging.synthetic_tracking import predicted_shift
    dx, dy = predicted_shift(60450.0 + 1/24.0, 60450.0, 3600.0, 90.0, 1.0)
    assert dx < 0
    assert abs(dy) < 1e-3


def test_predicted_shift_north_motion_positive_y():
    """North-moving object in CD22>0 WCS has INCREASING pixel-y. Shift
    to bring back: dy positive (read from input[i + dy, j] where dy>0
    reads from a higher row, which is where the north-moved object is)."""
    from ariadne.discovery.imaging.synthetic_tracking import predicted_shift
    dx, dy = predicted_shift(60450.0 + 1/24.0, 60450.0, 3600.0, 0.0, 1.0)
    assert dy > 0
    assert abs(dx) < 1e-3


def test_fast_synthetic_tracking_recovers_moving_object():
    """Plant a Gaussian moving at rate=1"/hr pa=45 deg across 6 images
    spanning 6 days. fast_synthetic_tracking should find it."""
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.synthetic_tracking import (
        fast_synthetic_tracking)
    npix = 200
    pixscale = 1.0   # arcsec / pix
    image_mjds = [60450.0, 60450.083, 60453.0, 60453.083, 60456.0, 60456.083]
    t_ref = image_mjds[len(image_mjds)//2]
    rate_arcsec_hr = 1.0   # slow enough to stay in the 200-pix field over 6 days
    pa_deg = 45.0

    def make_wcs():
        w = WCS(naxis=2)
        w.wcs.crpix = [npix/2, npix/2]
        w.wcs.crval = [180.0, 20.0]
        w.wcs.cd = [[-pixscale/3600, 0], [0, pixscale/3600]]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w
    images = []; wcs_list = []
    for mjd in image_mjds:
        dt_hr = (mjd - t_ref) * 24.0
        motion_pix = rate_arcsec_hr * dt_hr / pixscale
        # In standard WCS east = -x, north = +y (our convention)
        dra_pix = motion_pix * math.sin(math.radians(pa_deg))
        ddec_pix = motion_pix * math.cos(math.radians(pa_deg))
        x_at_t = npix/2 - dra_pix
        y_at_t = npix/2 + ddec_pix
        # Bound-safe: rate=1"/hr * 72hr = 72 pix; npix=200 has 100-pix margin
        assert 10 < x_at_t < npix - 10
        assert 10 < y_at_t < npix - 10
        images.append(_gaussian_image(npix, x_at_t, y_at_t, 1500,
                                        sigma=1.5, bg=100, bg_sigma=8,
                                        seed=int(mjd * 1000)))
        wcs_list.append(make_wcs())

    candidates = fast_synthetic_tracking(
        images, wcs_list, image_mjds, t_ref_mjd=t_ref,
        rate_min_arcsec_hr=0.5, rate_max_arcsec_hr=3.0, n_rates=8,
        n_pa=8, snr_threshold=5.0, pixscale_arcsec=pixscale,
        n_top_per_hypothesis=3)
    assert len(candidates) >= 1
    # Best candidate has rate near 1 and PA near 45 (with grid tolerance)
    best = candidates[0]
    assert 0.7 <= best.rate_arcsec_hr <= 1.5
    pa_diff = abs(best.pa_deg - pa_deg)
    pa_diff = min(pa_diff, 360 - pa_diff)
    assert pa_diff < 50
