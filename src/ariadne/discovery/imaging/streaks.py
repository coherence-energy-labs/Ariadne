"""Streak detector via Hough transform -- catch fast-mover trails in a single exposure.

A fast NEO or a low-Earth-orbit satellite crossing the field during a long
exposure produces a STREAK -- a linear feature several pixels wide and tens
to hundreds of pixels long. DAOStarFinder finds the BRIGHT centre of the
streak as a single source, missing the geometry entirely.

The Hough transform is the canonical way to detect lines in images:

  1. Threshold the image to a binary mask of "bright" pixels.
  2. Each bright pixel "votes" for every line that passes through it.
  3. Lines are parameterised as (rho, theta) where:
        rho = x*cos(theta) + y*sin(theta)
     -- the perpendicular distance from origin + angle of the normal.
  4. The Hough accumulator A[rho, theta] sums votes from every bright pixel.
  5. Local maxima in A correspond to genuine straight-line features.

After detecting candidate streaks, we cluster pixel groups along each line
to measure length, width, and brightness. The output Streak record carries
enough information to:

  * distinguish a fast NEO (~few arcsec/sec for a 30-second exposure) from
    a satellite (~ degrees/second; saturated, full-width trail), or from
    a cosmic-ray-induced trail (very thin, single-pixel-wide, no PSF wings).
  * compute the streak endpoint (RA, Dec) and time-bounds, giving a real
    sky-velocity estimate from a single exposure.
  * feed into the inference engine as MorphologyClass.STREAK evidence with
    a real angular position, not just a flag.

Reference: Hough 1962 (line detection patent); Duda & Hart 1972 (rho-theta
parameterisation); Sara et al. 2017 (asteroid streak detection in ZTF).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Streak:
    """One detected linear streak in an image.

    Fields:
      x1, y1, x2, y2:  pixel endpoints (sub-pixel).
      length_px:       sqrt((x2-x1)^2 + (y2-y1)^2).
      width_px:        FWHM perpendicular to the streak direction (PSF width).
      theta_rad:       angle of the streak axis (0 = horizontal, pi/2 = vertical).
      peak_pixel:      max pixel value along the streak (post-background).
      total_flux:      sum of pixel values minus background along the streak.
      n_pixels:        how many bright pixels participate.
      vote_count:      Hough-transform vote count (proxy for line strength).
      consistency:     0..1 -- how PSF-thin the streak is (1.0 = thin asteroid
                        trail, < 0.5 = wide satellite or diffuse extended).
    """
    x1: float; y1: float; x2: float; y2: float
    length_px: float
    width_px: float
    theta_rad: float
    peak_pixel: float
    total_flux: float
    n_pixels: int
    vote_count: int
    consistency: float


def _binarise(image: np.ndarray, sigma_threshold: float = 4.0) -> np.ndarray:
    """Threshold to a bool mask above (median + sigma_threshold * MAD)."""
    med = np.median(image)
    mad = np.median(np.abs(image - med))
    sigma = 1.4826 * mad
    return image > med + sigma_threshold * max(sigma, 1.0)


def hough_lines(image: np.ndarray,
                *,
                sigma_threshold: float = 4.0,
                n_theta: int = 180,
                rho_resolution: float = 1.0,
                min_votes: int = 30,
                top_n: int = 20) -> list[tuple[float, float, int]]:
    """Standard Hough line transform; return top-N (rho, theta, vote_count).

    Args:
      image:           2D image array.
      sigma_threshold: pixel must exceed (median + sigma*MAD) to vote.
      n_theta:         number of angle bins (default 180, 1-deg resolution).
      rho_resolution:  rho bin size in pixels.
      min_votes:       lines with fewer votes are discarded.
      top_n:           cap on number of returned lines.

    Returns:
      List of (rho_pixels, theta_radians, vote_count), sorted by vote_count
      descending.
    """
    H, W = image.shape
    mask = _binarise(image, sigma_threshold=sigma_threshold)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    thetas = np.linspace(-math.pi / 2, math.pi / 2, n_theta, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    rho_max = int(math.ceil(math.hypot(H, W) / rho_resolution))
    n_rho = 2 * rho_max + 1
    accumulator = np.zeros((n_rho, n_theta), dtype=np.int32)

    for x, y in zip(xs, ys):
        rhos = x * cos_t + y * sin_t
        rho_bins = np.round(rhos / rho_resolution).astype(int) + rho_max
        valid = (rho_bins >= 0) & (rho_bins < n_rho)
        for ti, rb in enumerate(rho_bins):
            if 0 <= rb < n_rho:
                accumulator[rb, ti] += 1

    # find local maxima in accumulator (suppress neighbours)
    lines = []
    acc_max = accumulator.max() if accumulator.size > 0 else 0
    threshold = max(min_votes, int(0.3 * acc_max))
    while len(lines) < top_n:
        idx = np.argmax(accumulator)
        votes = int(accumulator.flat[idx])
        if votes < threshold:
            break
        rho_bin, theta_bin = np.unravel_index(idx, accumulator.shape)
        rho = (rho_bin - rho_max) * rho_resolution
        theta = thetas[theta_bin]
        lines.append((float(rho), float(theta), votes))
        # suppress neighbourhood so we don't double-count the same line
        sy = max(0, rho_bin - 3); ey = min(n_rho, rho_bin + 4)
        sx = max(0, theta_bin - 3); ex = min(n_theta, theta_bin + 4)
        accumulator[sy:ey, sx:ex] = 0
    return lines


def _measure_streak(image: np.ndarray, mask: np.ndarray,
                    rho: float, theta: float,
                    *, width_px: float = 4.0,
                    bg_median: float | None = None) -> Streak | None:
    """Given a (rho, theta) line, measure endpoints + width + flux from the image."""
    H, W = image.shape
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    # signed perpendicular distance of each bright pixel from the line
    proj_perp = xs * cos_t + ys * sin_t - rho
    on_line = np.abs(proj_perp) <= width_px
    line_pix = np.where(on_line)[0]
    if len(line_pix) < 3:
        return None
    xs_line = xs[line_pix]
    ys_line = ys[line_pix]

    # parametric coordinate along the line direction:
    # tangent = (-sin_t, cos_t); so t = x*(-sin_t) + y*cos_t
    tang = -xs_line * sin_t + ys_line * cos_t
    t_min = tang.min(); t_max = tang.max()
    # endpoints (in image coords): solve for the foot on the line at t_min/t_max
    # foot_x = rho*cos_t - t*sin_t,  foot_y = rho*sin_t + t*cos_t
    x1 = rho * cos_t - t_min * sin_t
    y1 = rho * sin_t + t_min * cos_t
    x2 = rho * cos_t - t_max * sin_t
    y2 = rho * sin_t + t_max * cos_t
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 4.0:
        return None

    if bg_median is None:
        bg_median = float(np.median(image))

    pixel_vals = image[ys_line, xs_line] - bg_median
    peak = float(pixel_vals.max())
    total = float(pixel_vals.sum())
    n_pix = int(line_pix.size)

    # Width estimate: spread of perpendicular distances of bright pixels
    spread = float(np.percentile(np.abs(proj_perp[on_line]), 84))
    fwhm_perp = 2.355 * spread
    # Consistency: 1.0 = streak is exactly PSF-thin; <0.5 = wider than 3*PSF
    expected_psf_px = 3.0
    consistency = max(0.0, min(1.0, expected_psf_px / max(fwhm_perp, 1e-3)))

    return Streak(
        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
        length_px=length, width_px=fwhm_perp,
        theta_rad=theta, peak_pixel=peak, total_flux=total,
        n_pixels=n_pix, vote_count=0, consistency=consistency,
    )


def detect_streaks(image: np.ndarray,
                   *,
                   sigma_threshold: float = 4.0,
                   min_length_px: float = 8.0,
                   max_width_px: float = 6.0,
                   min_consistency: float = 0.3,
                   max_streaks: int = 10) -> list[Streak]:
    """End-to-end streak detection: Hough transform + measurement + filter.

    Args:
      image:               2D image array (background not pre-subtracted is fine).
      sigma_threshold:     binarisation cut (above median + sigma*MAD).
      min_length_px:       reject streaks shorter than this.
      max_width_px:        reject streaks wider than this (extended sources).
      min_consistency:     reject streaks too wide for the PSF (extended fuzz).
      max_streaks:         cap on returned count.

    Returns:
      List of Streak records, sorted by total_flux descending.
    """
    mask = _binarise(image, sigma_threshold=sigma_threshold)
    bg_med = float(np.median(image))
    lines = hough_lines(image, sigma_threshold=sigma_threshold,
                        top_n=max_streaks * 3)
    streaks = []
    for rho, theta, votes in lines:
        s = _measure_streak(image, mask, rho, theta,
                            width_px=max_width_px, bg_median=bg_med)
        if s is None:
            continue
        if s.length_px < min_length_px:
            continue
        if s.width_px > max_width_px:
            continue
        if s.consistency < min_consistency:
            continue
        # propagate vote count
        s = Streak(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
                   length_px=s.length_px, width_px=s.width_px,
                   theta_rad=s.theta_rad, peak_pixel=s.peak_pixel,
                   total_flux=s.total_flux, n_pixels=s.n_pixels,
                   vote_count=votes, consistency=s.consistency)
        streaks.append(s)
    streaks.sort(key=lambda x: x.total_flux, reverse=True)
    return streaks[:max_streaks]


def classify_streak(streak: Streak, exposure_seconds: float = 30.0,
                    pixel_scale_arcsec: float = 0.25) -> dict:
    """Distinguish asteroid trail / satellite trail / cosmic-ray trail.

    Heuristics (empirical for ground-based CCD imaging):

      * Asteroid (NEO):   length 5-200 px, width 1-2 PSFs, consistent intensity,
                          rate 0.5-30 arcsec/sec.
      * Satellite (LEO):  length > 200 px (often spans entire frame), width
                          consistent with PSF, rate >> 100 arcsec/sec.
      * Geosync sat:      length 5-30 px, width ~ PSF, rate 5-15 arcsec/sec.
      * Cosmic-ray trail: length < 10 px, width sub-PSF, often diagonal.

    Returns a dict {label, confidence, rate_arcsec_sec, rate_arcsec_hr}.
    """
    rate_px_per_s = streak.length_px / max(exposure_seconds, 1e-3)
    rate_arcsec_per_s = rate_px_per_s * pixel_scale_arcsec
    rate_arcsec_per_hr = rate_arcsec_per_s * 3600.0

    # Width-based label cuts
    if streak.consistency < 0.5:
        label = "extended_or_satellite_wide"
        conf = 0.5
    elif streak.length_px > 200:
        label = "satellite_LEO"
        conf = 0.85
    elif rate_arcsec_per_s > 5.0:
        label = "satellite_or_fast_NEO"
        conf = 0.6
    elif streak.length_px < 8 and streak.width_px < 2.0:
        label = "cosmic_ray_trail"
        conf = 0.7
    elif 0.5 < rate_arcsec_per_s < 5.0:
        label = "NEO_or_inner_main_belt"
        conf = 0.75
    elif rate_arcsec_per_s < 0.5:
        label = "slow_mover_or_artefact"
        conf = 0.5
    else:
        label = "unclassified_streak"
        conf = 0.4

    return {
        "label": label, "confidence": conf,
        "rate_arcsec_sec": rate_arcsec_per_s,
        "rate_arcsec_hr": rate_arcsec_per_hr,
        "length_px": streak.length_px, "width_px": streak.width_px,
        "consistency": streak.consistency,
    }
