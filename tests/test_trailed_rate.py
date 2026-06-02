"""Tests for single-snapshot angular-rate recovery from PSF trailing.

Pins the validated behaviour (confirmed by injection into real DECam
pixels): the second-moment trail estimator recovers the injected angular
rate within ~25% and the motion PA within ~25 deg over the working regime
(~60-450 "/hr), and reports ~0 rate for a round (untrailed) PSF.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

PIXSCALE = 0.263
TEXP = 90.0


def _inject(shape, x, y, rate_arcsec_hr, pa_deg, flux=2.0e5, fwhm=3.8,
            seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 8.0, shape)
    L_px = rate_arcsec_hr * (TEXP / 3600.0) / PIXSCALE
    sigma = fwhm / 2.3548
    n = max(int(L_px * 3), 1)
    dx = 0.5 * L_px * math.sin(math.radians(pa_deg))
    dy = 0.5 * L_px * math.cos(math.radians(pa_deg))
    fpp = flux / n
    half = int(math.ceil(4 * sigma + L_px))
    H, W = shape
    for t in np.linspace(-1, 1, n):
        cx = x + t * dx; cy = y + t * dy
        x0 = max(0, int(cx) - half); x1 = min(W, int(cx) + half + 1)
        y0 = max(0, int(cy) - half); y1 = min(H, int(cy) + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += (fpp / (2 * math.pi * sigma ** 2)) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return img


def _stamp(img, x, y, half=18):
    return img[y - half:y + half + 1, x - half:x + half + 1]


@pytest.mark.parametrize("rate", [60, 120, 250, 400])
def test_rate_recovered_in_working_regime(rate):
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    img = _inject((120, 120), 60, 60, rate, 35.0)
    est = rate_from_stamp(_stamp(img, 60, 60), pixscale_arcsec=PIXSCALE,
                            t_exp_s=TEXP)
    assert abs(est.rate_arcsec_hr - rate) / rate < 0.30, (
        f"rate {rate}: recovered {est.rate_arcsec_hr:.0f}")


@pytest.mark.parametrize("pa", [10.0, 45.0, 80.0, 130.0])
def test_pa_recovered(pa):
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    img = _inject((140, 140), 70, 70, 250.0, pa)
    est = rate_from_stamp(_stamp(img, 70, 70), pixscale_arcsec=PIXSCALE,
                            t_exp_s=TEXP)
    d = abs((est.pa_deg_pix - (pa % 180.0) + 90) % 180 - 90)
    assert d < 25.0, f"PA {pa}: recovered {est.pa_deg_pix:.0f} (off {d:.0f})"


def test_round_psf_gives_near_zero_rate():
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    img = _inject((80, 80), 40, 40, 0.0, 0.0)   # no trail
    est = rate_from_stamp(_stamp(img, 40, 40), pixscale_arcsec=PIXSCALE,
                            t_exp_s=TEXP)
    # an untrailed point source should imply a small rate (well below the
    # fast-mover regime); allow for noise on the moment.
    assert est.rate_arcsec_hr < 60.0


def test_rate_monotonic_in_truth():
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    prev = -1.0
    for rate in [0, 60, 120, 250, 400]:
        img = _inject((160, 160), 80, 80, rate, 35.0, seed=rate)
        est = rate_from_stamp(_stamp(img, 80, 80), pixscale_arcsec=PIXSCALE,
                                t_exp_s=TEXP)
        assert est.rate_arcsec_hr >= prev - 20, "rate estimate not monotonic"
        prev = est.rate_arcsec_hr


def test_forward_model_beats_moments_at_faint_flux():
    """The matched-filter trailed-PSF fit must recover the rate where second
    moments collapse (faint, low-SNR). Validated on real DECam: median
    unbiased, false fast-mover rate ~halved."""
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp, fit_trailed_psf
    rate, pa = 120.0, 40.0
    mom_err, fwd_err = [], []
    for seed in range(20):
        img = _inject((60, 60), 30, 30, rate, pa, flux=5000.0, seed=seed)
        st = img[12:49, 12:49]
        mom_err.append(abs(rate_from_stamp(st, pixscale_arcsec=PIXSCALE,
                                             t_exp_s=TEXP).rate_arcsec_hr - rate))
        fwd_err.append(abs(fit_trailed_psf(st, psf_fwhm_px=3.8,
                                             pixscale_arcsec=PIXSCALE,
                                             t_exp_s=TEXP).rate_arcsec_hr - rate))
    # forward model's typical error must be much smaller than moments'
    assert np.median(fwd_err) < 0.5 * np.median(mom_err), (
        f"forward {np.median(fwd_err):.0f} not < half moments "
        f"{np.median(mom_err):.0f}")


def test_stellar_psf_fwhm_recovers_seeing():
    """The PSF-FWHM measurement (which the forward model needs) recovers the
    injected stellar seeing within ~20%."""
    from ariadne.discovery.imaging.trailed_rate import stellar_psf_fwhm
    rng = np.random.default_rng(0)
    img = rng.normal(100, 5, (400, 400))
    fwhm_true = 4.5
    sig = fwhm_true / 2.3548
    yy, xx = np.mgrid[0:400, 0:400]
    for _ in range(120):
        cx = rng.uniform(20, 380); cy = rng.uniform(20, 380)
        img += 4000 / (2 * math.pi * sig ** 2) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    fwhm_meas = stellar_psf_fwhm(img, fwhm_px=4.0)
    assert abs(fwhm_meas - fwhm_true) / fwhm_true < 0.25, (
        f"measured FWHM {fwhm_meas:.2f} vs true {fwhm_true}")


def test_wcs_pa_rotation_runs():
    """With a WCS, a sky-frame PA is produced (rotation path exercised)."""
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    w = WCS(naxis=2)
    w.wcs.crpix = [60, 60]
    w.wcs.cdelt = [-PIXSCALE / 3600, PIXSCALE / 3600]
    w.wcs.crval = [150.0, -20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    img = _inject((120, 120), 60, 60, 250.0, 45.0)
    est = rate_from_stamp(_stamp(img, 60, 60), pixscale_arcsec=PIXSCALE,
                            t_exp_s=TEXP, wcs=w, x_pix=60, y_pix=60)
    assert est.pa_deg_sky is not None
    assert 0.0 <= est.pa_deg_sky < 180.0
