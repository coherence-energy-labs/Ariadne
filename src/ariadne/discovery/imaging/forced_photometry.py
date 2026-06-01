"""Forced photometry at predicted positions.

For each within-night tracklet, predict positions on OTHER nights via
the tracklet's rate vector, then look for sub-threshold PSF signal at
each predicted pixel. This recovers detections that photutils' 5-sigma
threshold missed, enabling chains for objects detected on only one
night by the standard pipeline.

Public API:
  forced_photometry_at_position
      Fit a Gaussian PSF at a known (ra, dec) in an image. Return the
      best-fit amplitude + SNR + a Source object if SNR > threshold.

  enrich_sources_via_tracklets
      For each tracklet, extrapolate to all OTHER images via its rate
      vector, run forced_photometry at each predicted position, return
      a NEW list of sources that includes the original ones PLUS
      forced detections.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .source_extraction import Source


def _local_background_and_noise(image: np.ndarray, x: int, y: int,
                                  *, inner: int = 5, outer: int = 12
                                  ) -> tuple[float, float]:
    """Robust background median + MAD-sigma noise from an annulus at (x, y)."""
    ny, nx = image.shape
    if not (outer <= x < nx - outer and outer <= y < ny - outer):
        # Use whatever's in-image
        x0, x1 = max(0, x - outer), min(nx, x + outer + 1)
        y0, y1 = max(0, y - outer), min(ny, y + outer + 1)
    else:
        x0, x1 = x - outer, x + outer + 1
        y0, y1 = y - outer, y + outer + 1
    patch = image[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    cx, cy = patch.shape[1] // 2, patch.shape[0] // 2
    r = np.hypot(xx - cx, yy - cy)
    annulus = (r >= inner) & (r <= outer)
    bg_pix = patch[annulus]
    bg_pix = bg_pix[np.isfinite(bg_pix)]
    if bg_pix.size < 8:
        return float(np.nanmedian(patch)), 1.0
    bg_med = float(np.median(bg_pix))
    bg_mad = float(np.median(np.abs(bg_pix - bg_med)))
    bg_std = max(bg_mad * 1.4826, 1e-6)
    return bg_med, bg_std


def forced_photometry_at_position(image: np.ndarray, wcs,
                                     ra_deg: float, dec_deg: float,
                                     *, mjd: float, image_id: str,
                                     psf_sigma_pix: float = 1.5,
                                     snr_threshold: float = 2.0,
                                     fit_box_pix: int = 7,
                                     ) -> Source | None:
    """Fit a Gaussian PSF at a known (ra, dec) and return a Source if
    the SNR is above threshold.

    The PSF fit:
      1. Convert (ra, dec) to pixel (x, y) via WCS.
      2. Crop a (2*fit_box_pix + 1) box around (x, y).
      3. Fit amplitude analytically against a unit-PSF profile.
      4. SNR = signal / (per-pixel bg sigma * sqrt(aperture pixels))
      5. If SNR > threshold, return a Source with low-confidence flag.
    """
    try:
        x_pix, y_pix = wcs.world_to_pixel_values(ra_deg, dec_deg)
        x_pix = float(x_pix); y_pix = float(y_pix)
    except Exception:
        return None
    ny, nx = image.shape
    if not (fit_box_pix <= x_pix < nx - fit_box_pix
            and fit_box_pix <= y_pix < ny - fit_box_pix):
        return None

    bg_med, bg_std = _local_background_and_noise(image,
                                                   int(round(x_pix)),
                                                   int(round(y_pix)),
                                                   inner=fit_box_pix + 1,
                                                   outer=fit_box_pix + 6)

    # Fit Gaussian amplitude analytically
    half = fit_box_pix
    ix, iy = int(round(x_pix)), int(round(y_pix))
    patch = image[iy - half: iy + half + 1, ix - half: ix + half + 1].astype(float)
    yy, xx = np.indices(patch.shape)
    cx, cy = patch.shape[1] // 2, patch.shape[0] // 2
    # Sub-pixel offset
    dx_sub = x_pix - ix
    dy_sub = y_pix - iy
    profile = np.exp(-((xx - cx - dx_sub) ** 2
                         + (yy - cy - dy_sub) ** 2) / (2 * psf_sigma_pix ** 2))
    residual = patch - bg_med
    num = float(np.sum(residual * profile))
    den = float(np.sum(profile * profile))
    if den <= 0:
        return None
    A = num / den
    if A <= 0:
        return None

    # Aperture SNR
    aperture = profile > 0.1
    n_ap = int(aperture.sum())
    if n_ap == 0:
        return None
    signal = float(np.sum((patch - bg_med) * aperture))
    noise = bg_std * math.sqrt(n_ap)
    snr = signal / noise if noise > 0 else 0.0
    if snr < snr_threshold:
        return None

    # Construct a Source. Use a low-confidence FWHM = psf_sigma.
    fwhm = psf_sigma_pix * 2.355
    mag = -2.5 * math.log10(max(A, 1.0)) + 25.0 if A > 0 else -99.0
    return Source(
        ra=ra_deg % 360.0, dec=dec_deg,
        flux=float(A * n_ap), mag=float(mag),
        fwhm_px=float(fwhm), mjd=mjd, image_id=image_id,
        x=x_pix, y=y_pix,
    )


def enrich_sources_via_tracklets(images: Sequence[np.ndarray],
                                    wcs_list: Sequence,
                                    image_mjds: Sequence[float],
                                    image_ids: Sequence[str],
                                    tracklets: Sequence[dict],
                                    point_sources: Sequence[Source],
                                    *,
                                    psf_sigma_pix: float = 1.5,
                                    snr_threshold: float = 2.0,
                                    max_extrapolation_days: float = 14.0,
                                    ) -> list[Source]:
    """For each tracklet, predict positions on OTHER images via its rate
    vector + position-angle, run forced photometry, and merge any new
    sources into the source list.

    Returns a new list = original `point_sources` + non-duplicate forced
    detections.
    """
    if not tracklets or not images:
        return list(point_sources)

    # Existing source positions per image for dedup
    by_image_existing = {iid: [(s.x, s.y) for s in point_sources
                                 if s.image_id == iid]
                           for iid in image_ids}

    forced = []
    for tr in tracklets:
        rate = float(tr.get("rate_arcsec_hr", 0.0))
        if rate <= 0:
            continue
        # PA from dra/ddec
        dra = float(tr.get("dra", 0.0))
        ddec = float(tr.get("ddec", 0.0))
        if abs(dra) < 1e-12 and abs(ddec) < 1e-12:
            continue
        pa = math.atan2(dra, ddec)
        t_ref_et = float(tr["t"])
        t_ref_mjd = t_ref_et / 86400.0 + 51544.5
        ra_ref = math.degrees(tr["ra"]) % 360.0
        dec_ref = math.degrees(tr["dec"])
        cos_dec_ref = math.cos(math.radians(dec_ref))

        for img, wcs, mjd, iid in zip(images, wcs_list, image_mjds,
                                          image_ids):
            dt_days = mjd - t_ref_mjd
            if abs(dt_days) > max_extrapolation_days:
                continue
            if abs(dt_days) < 1e-6:
                continue
            dt_hr = dt_days * 24.0
            motion_arcsec = rate * dt_hr
            dra_arcsec = motion_arcsec * math.sin(pa) / max(cos_dec_ref, 1e-6)
            ddec_arcsec = motion_arcsec * math.cos(pa)
            ra_pred = ra_ref + dra_arcsec / 3600.0
            dec_pred = dec_ref + ddec_arcsec / 3600.0
            # Skip if a regular source already exists near this position
            try:
                x_pix, y_pix = wcs.world_to_pixel_values(ra_pred, dec_pred)
                x_pix = float(x_pix); y_pix = float(y_pix)
            except Exception:
                continue
            close_existing = False
            for ex, ey in by_image_existing.get(iid, ()):
                if abs(ex - x_pix) < 3 and abs(ey - y_pix) < 3:
                    close_existing = True
                    break
            if close_existing:
                continue
            src = forced_photometry_at_position(
                img, wcs, ra_pred, dec_pred,
                mjd=mjd, image_id=iid,
                psf_sigma_pix=psf_sigma_pix,
                snr_threshold=snr_threshold)
            if src is not None:
                forced.append(src)
                by_image_existing.setdefault(iid, []).append((src.x, src.y))

    return list(point_sources) + forced
