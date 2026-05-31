"""PSF-fit centroiding -- sub-pixel position from a Gaussian model fit.

DAOStarFinder returns moment-based centroids (essentially weighted x_centroid /
y_centroid over the detection footprint). That centroid is good to ~0.2 pixel for
isolated bright sources but degrades to ~0.5 pixel for faint or blended sources.

A 2D-Gaussian PSF fit reaches the Cramer-Rao bound for centroid precision:

    sigma_x ~ FWHM / (S/N * sqrt(2 * pi))

For a 3-pixel FWHM source at S/N=10 that's ~0.06 pixel ~= 60 mas at DECam scale.
That precision is the difference between "tracklet links cleanly across nights"
and "tracklet falls apart in the linker."

This module:
  * fits a 2D circular Gaussian + flat background to a postage stamp around each
    seed source from `detect_sources_in_image` (or any (x,y) seed list),
  * returns a new list of Sources with REFINED (x,y) -> WCS-evaluated (RA,Dec)
    and a per-source 1-sigma position error estimate (in arcsec) for the linker
    to use as a weighting hint.

Sub-pixel precision uses scipy.optimize.least_squares, NOT photutils' built-in
fitter (which depends on a less-stable internal solver). Falls back gracefully
to the seed centroid when the fit doesn't converge.

Reference: King 1971 PSF formalism; Howell 2006 photometry text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .source_extraction import Source


@dataclass(frozen=True)
class PSFFit:
    """Result of one PSF fit on a postage stamp.

    Fields:
      x_sub, y_sub      : refined pixel position (sub-pixel)
      sigma_x, sigma_y  : 1-sigma position uncertainty in pixels
      sigma_arcsec      : 1-sigma position uncertainty in arcsec (via WCS scale)
      flux              : best-fit Gaussian flux (2*pi*amplitude*sigma^2)
      fwhm_px           : best-fit FWHM in pixels (= 2.355*sigma)
      success           : did the fit converge?
      chi2_red          : reduced chi-square (smaller is better; >5 is suspicious)
    """
    x_sub: float
    y_sub: float
    sigma_x: float
    sigma_y: float
    sigma_arcsec: float
    flux: float
    fwhm_px: float
    success: bool
    chi2_red: float


def _gaussian2d(params, x, y):
    """2D circular Gaussian + flat background. params = (A, x0, y0, sigma, b)."""
    A, x0, y0, sigma, b = params
    if sigma <= 0:
        return np.full_like(x, 1e30)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return A * np.exp(-0.5 * r2 / sigma ** 2) + b


def _wcs_pixel_scale_arcsec(wcs, x: float, y: float) -> float:
    """Approximate plate scale in arcsec/pixel at (x, y).

    Uses the local CD matrix via WCS finite-difference at 1-pixel offset.
    Falls back to 0.25"/px (DECam-like) if WCS query fails.
    """
    try:
        ra0, dec0 = wcs.pixel_to_world_values(x, y)
        ra1, dec1 = wcs.pixel_to_world_values(x + 1.0, y)
        dra = (ra1 - ra0) * math.cos(math.radians(dec0))
        ddec = dec1 - dec0
        return math.hypot(dra, ddec) * 3600.0
    except Exception:
        return 0.25


def fit_psf_postage_stamp(image_data: np.ndarray, x_seed: float, y_seed: float,
                          *, half_size: int = 5,
                          fwhm_guess_px: float = 3.0,
                          background_sigma: float | None = None,
                          wcs=None) -> PSFFit:
    """Fit a 2D Gaussian + flat background to a postage stamp around (x_seed, y_seed).

    Args:
      image_data:        full 2D image array (counts).
      x_seed, y_seed:    seed centroid (e.g. from DAOStarFinder).
      half_size:         stamp half-width in pixels; stamp = (2*half_size+1)^2.
                         Use 5-7 for typical seeing; 10 for very wide PSF.
      fwhm_guess_px:     initial guess for the Gaussian FWHM in pixels.
      background_sigma:  per-pixel noise sigma; if None, estimated from stamp std.
      wcs:               optional astropy WCS for computing sigma_arcsec.

    Returns:
      PSFFit with sub-pixel position + uncertainty + flux + chi2.
      .success=False if the fit diverged; caller should fall back to seed.
    """
    H, W = image_data.shape
    ix = int(round(x_seed))
    iy = int(round(y_seed))
    x_lo = max(0, ix - half_size); x_hi = min(W, ix + half_size + 1)
    y_lo = max(0, iy - half_size); y_hi = min(H, iy + half_size + 1)
    if x_hi - x_lo < 3 or y_hi - y_lo < 3:
        return PSFFit(x_seed, y_seed, 1.0, 1.0, _wcs_pixel_scale_arcsec(wcs, x_seed, y_seed),
                      0.0, fwhm_guess_px, False, 1e30)

    stamp = image_data[y_lo:y_hi, x_lo:x_hi].astype(float)
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    x_flat = xx.ravel().astype(float)
    y_flat = yy.ravel().astype(float)
    z_flat = stamp.ravel()

    # initial parameter guess
    b0 = float(np.percentile(stamp, 25))                 # lower-quartile = background
    A0 = float(stamp.max() - b0)
    sigma0 = fwhm_guess_px / 2.355
    p0 = np.array([A0, float(x_seed), float(y_seed), sigma0, b0])

    if background_sigma is None:
        background_sigma = float(np.std(stamp[stamp < np.percentile(stamp, 50)]))
        if background_sigma <= 0:
            background_sigma = max(1.0, abs(stamp.std()))

    def resid(p):
        return (_gaussian2d(p, x_flat, y_flat) - z_flat) / max(background_sigma, 1e-3)

    try:
        r = least_squares(resid, p0, method="lm", xtol=1e-10, ftol=1e-10, max_nfev=200)
        A, x0, y0, sigma, b = r.x
        if not r.success or sigma <= 0 or not (x_lo <= x0 <= x_hi) or not (y_lo <= y0 <= y_hi):
            return PSFFit(x_seed, y_seed, 1.0, 1.0,
                          _wcs_pixel_scale_arcsec(wcs, x_seed, y_seed),
                          0.0, fwhm_guess_px, False, float(np.sum(r.fun ** 2)))

        # Cramer-Rao centroid uncertainty:  sigma_pos = FWHM / (SNR * sqrt(2*pi))
        flux = 2.0 * math.pi * A * sigma ** 2
        snr = flux / (max(background_sigma, 1e-3) * sigma * math.sqrt(2 * math.pi))
        sigma_pos_px = (sigma * 2.355) / max(snr, 1.0) / math.sqrt(2 * math.pi)
        sigma_pos_px = max(sigma_pos_px, 0.005)         # floor: cosmic-ray bright srcs

        chi2 = float(np.sum(r.fun ** 2))
        dof = max(1, len(z_flat) - len(r.x))
        chi2_red = chi2 / dof

        sigma_arcsec = sigma_pos_px * _wcs_pixel_scale_arcsec(wcs, x0, y0)
        return PSFFit(x_sub=float(x0), y_sub=float(y0),
                      sigma_x=sigma_pos_px, sigma_y=sigma_pos_px,
                      sigma_arcsec=sigma_arcsec,
                      flux=flux, fwhm_px=float(sigma * 2.355),
                      success=True, chi2_red=chi2_red)
    except Exception:
        return PSFFit(x_seed, y_seed, 1.0, 1.0,
                      _wcs_pixel_scale_arcsec(wcs, x_seed, y_seed),
                      0.0, fwhm_guess_px, False, 1e30)


def refine_sources_psf(image_data: np.ndarray, sources: list[Source],
                       wcs, *,
                       half_size: int = 5,
                       fwhm_guess_px: float = 3.0,
                       discard_failed: bool = False) -> list[Source]:
    """Replace each Source's pixel position with a PSF-fit sub-pixel centroid.

    Recomputes (RA, Dec) from the refined (x, y) via the same WCS. The original
    Source.mag/flux is kept (PSF-fit flux is an additional sanity check, not a
    replacement for the input photometry calibration).

    Args:
      image_data:    full 2D image array.
      sources:       seed sources (e.g. from detect_sources_in_image).
      wcs:           astropy WCS for the image.
      half_size:     postage-stamp half-size.
      fwhm_guess_px: initial FWHM guess for the Gaussian.
      discard_failed: if True, drop sources whose PSF fit failed; if False,
                      keep them with their seed position.

    Returns:
      List of Source with refined (x, y, ra, dec). The list may be SHORTER than
      input if discard_failed=True.
    """
    out = []
    for s in sources:
        fit = fit_psf_postage_stamp(image_data, s.x, s.y,
                                     half_size=half_size,
                                     fwhm_guess_px=fwhm_guess_px,
                                     wcs=wcs)
        if not fit.success and discard_failed:
            continue
        if fit.success:
            ra, dec = wcs.pixel_to_world_values(fit.x_sub, fit.y_sub)
            ra = float(ra) % 360.0
            dec = float(dec)
        else:
            ra, dec = s.ra, s.dec
        out.append(Source(
            ra=ra, dec=dec, flux=s.flux, mag=s.mag,
            fwhm_px=fit.fwhm_px if fit.success else s.fwhm_px,
            mjd=s.mjd, image_id=s.image_id,
            x=fit.x_sub if fit.success else s.x,
            y=fit.y_sub if fit.success else s.y,
        ))
    return out
