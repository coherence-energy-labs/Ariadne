"""Deblender: split a single blended detection into its constituent point sources.

The DAOStarFinder seed catalogue treats touching/overlapping pairs of point
sources as one detection. This module operates DOWNSTREAM of the morphology
classifier: when a Source comes back labelled BLEND, run the deblender to
extract individual point sources.

Two-pass algorithm:

  1. Peak-finding pass: locate local maxima above (bg + 5*noise) in the postage
     stamp, with a minimum separation of `min_sep_px` pixels.
  2. Joint PSF-fit pass: fit a sum of N circular Gaussians (one per peak) + a
     flat background. Initialised from the peak positions; LM optimises all
     positions + amplitudes simultaneously.

Output is a list of Source records replacing the original blended detection.
Each new Source carries the joint-fit centroid + flux + a marker in `image_id`
that records the blend split (e.g. "img42#blend2of3").

Use case: distinguishing two trojan binary components, or a near-Earth asteroid
that crossed a slow background star. Standard photutils deblend_sources is
threshold-based and tuned for galaxies; this one is tuned for point sources.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .source_extraction import Source


def _find_peaks(stamp: np.ndarray, bg: float, noise: float,
                min_sep_px: float = 2.5,
                sigma_above_bg: float = 7.0,
                max_peaks: int = 5,
                secondary_peak_min_frac: float = 0.25) -> list[tuple[int, int]]:
    """Locate up to `max_peaks` physically distinct local maxima.

    A candidate is a peak only when above (bg + sigma_above_bg * noise) AND
    the brightest pixel in its 3x3 neighborhood. Among candidates we keep:
      * the brightest one unconditionally,
      * each subsequent peak only if separated by >= min_sep_px from every
        previously-kept peak AND its amplitude (peak - bg) is at least
        secondary_peak_min_frac of the brightest peak's amplitude.
    The amplitude rule prevents the wings of a single bright star -- where
    Poisson fluctuations occasionally produce a local max -- from being
    counted as a second source.
    """
    thresh = bg + sigma_above_bg * max(noise, 1e-3)
    H, W = stamp.shape
    cands = []
    for j in range(1, H - 1):
        for i in range(1, W - 1):
            v = stamp[j, i]
            if v < thresh:
                continue
            if (v > stamp[j - 1, i] and v > stamp[j + 1, i] and
                    v > stamp[j, i - 1] and v > stamp[j, i + 1] and
                    v >= stamp[j - 1, i - 1] and v >= stamp[j + 1, i + 1]):
                cands.append((i, j, v))
    cands.sort(key=lambda t: -t[2])           # brightest first
    if not cands:
        return []
    brightest_amp = cands[0][2] - bg
    peaks = []
    for i, j, v in cands:
        amp = v - bg
        if peaks and amp < secondary_peak_min_frac * brightest_amp:
            continue
        if any(math.hypot(i - pi, j - pj) < min_sep_px for pi, pj in peaks):
            continue
        peaks.append((i, j))
        if len(peaks) >= max_peaks:
            break
    return peaks


def _multi_gaussian(params, x, y, n: int):
    """Sum of n circular Gaussians + flat background.

    params layout: [A0, x0, y0, A1, x1, y1, ..., sigma, bg]  (3n+2 parameters).
    All Gaussians share one sigma (point sources in the same image have same PSF).
    """
    sigma = params[-2]
    bg = params[-1]
    if sigma <= 0:
        return np.full_like(x, 1e30)
    out = np.full_like(x, bg, dtype=float)
    inv_2s2 = 1.0 / (2.0 * sigma ** 2)
    for k in range(n):
        A = params[3 * k]
        x0 = params[3 * k + 1]
        y0 = params[3 * k + 2]
        out += A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) * inv_2s2)
    return out


def deblend_source(image_data: np.ndarray, source: Source,
                   *, half_size: int = 7,
                   fwhm_guess_px: float = 3.0,
                   min_sep_px: int = 2,
                   sigma_above_bg: float = 5.0,
                   max_components: int = 4,
                   wcs=None) -> list[Source]:
    """Try to split a single Source into multiple overlapping point sources.

    Returns:
      Single-element list `[source]` if the stamp only has one detectable peak
      (nothing to deblend). Otherwise a list of N Sources, one per fitted
      component, with .image_id = "<orig>#blendKofN".
    """
    H, W = image_data.shape
    ix = int(round(source.x)); iy = int(round(source.y))
    x_lo = max(0, ix - half_size); x_hi = min(W, ix + half_size + 1)
    y_lo = max(0, iy - half_size); y_hi = min(H, iy + half_size + 1)
    stamp = image_data[y_lo:y_hi, x_lo:x_hi].astype(float)
    if stamp.size < 9:
        return [source]

    bg = float(np.percentile(stamp, 25))
    noise = float(np.std(stamp[stamp < np.percentile(stamp, 50)]))
    if noise <= 0:
        noise = max(1.0, abs(stamp.std()))

    peaks = _find_peaks(stamp, bg, noise,
                        min_sep_px=min_sep_px,
                        sigma_above_bg=sigma_above_bg,
                        max_peaks=max_components)
    if len(peaks) <= 1:
        return [source]

    # joint multi-Gaussian fit
    n = len(peaks)
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    x_flat = xx.ravel().astype(float)
    y_flat = yy.ravel().astype(float)
    z_flat = stamp.ravel()

    p0 = []
    for (px, py) in peaks:
        A0 = max(stamp[py, px] - bg, 1.0)
        p0.extend([A0, float(px), float(py)])
    p0.append(fwhm_guess_px / 2.355)
    p0.append(bg)
    p0 = np.array(p0)

    def resid(p):
        return (_multi_gaussian(p, x_flat, y_flat, n) - z_flat) / max(noise, 1e-3)

    try:
        r = least_squares(resid, p0, method="lm", xtol=1e-10, ftol=1e-10, max_nfev=400)
        if not r.success:
            return [source]
        sigma = r.x[-2]
        if sigma <= 0:
            return [source]
        out = []
        for k in range(n):
            A = r.x[3 * k]
            x_stamp = r.x[3 * k + 1]
            y_stamp = r.x[3 * k + 2]
            x_full = x_lo + x_stamp
            y_full = y_lo + y_stamp
            flux = 2.0 * math.pi * A * sigma ** 2
            if flux <= 0:
                continue
            if wcs is not None:
                try:
                    ra, dec = wcs.pixel_to_world_values(x_full, y_full)
                    ra = float(ra) % 360.0
                    dec = float(dec)
                except Exception:
                    ra, dec = source.ra, source.dec
            else:
                # No WCS: leave coords offset relative to seed (caller may re-WCS)
                ra = source.ra + (x_full - source.x) * 0.25 / 3600.0
                dec = source.dec + (y_full - source.y) * 0.25 / 3600.0
            mag = (source.mag if source.mag <= -50 else
                   source.mag - 2.5 * math.log10(max(flux / source.flux, 1e-3)))
            out.append(Source(
                ra=ra, dec=dec, flux=flux, mag=mag,
                fwhm_px=float(sigma * 2.355),
                mjd=source.mjd,
                image_id=f"{source.image_id}#blend{k+1}of{n}",
                x=x_full, y=y_full,
            ))
        if not out:
            return [source]
        return out
    except Exception:
        return [source]


def deblend_sources(image_data: np.ndarray, sources: list[Source],
                    *, only_blended_labels: list[str] | None = None,
                    morphology_verdicts: list | None = None,
                    **kwargs) -> list[Source]:
    """Apply the deblender to every source (or only those tagged BLEND).

    Args:
      sources:                input list.
      only_blended_labels:    list of MorphologyClass labels to deblend (default:
                              ["BLEND"]).
      morphology_verdicts:    parallel list of MorphologyVerdict; pass to skip
                              re-classification. If None, every source is deblended.
      kwargs:                 forwarded to deblend_source.

    Returns:
      Flat list of Sources with blended detections replaced by their components.
    """
    if only_blended_labels is None:
        only_blended_labels = ["BLEND"]
    out = []
    for i, s in enumerate(sources):
        if morphology_verdicts is not None and i < len(morphology_verdicts):
            if morphology_verdicts[i].label not in only_blended_labels:
                out.append(s)
                continue
        out.extend(deblend_source(image_data, s, **kwargs))
    return out
