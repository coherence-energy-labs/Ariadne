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


def mask_compact_sources(image: np.ndarray, mask: np.ndarray,
                            *, max_component_extent_px: float = 6.0,
                            min_axis_ratio: float = 2.0) -> np.ndarray:
    """Remove compact (star-like) connected components from a binary mask,
    keeping only elongated (streak-like) pixel groups.

    A real CCD has thousands of stars whose bright pixels swamp the Hough
    accumulator and produce spurious line votes from random alignments.
    Production streak finders (ZTF, PanSTARRS) suppress point sources
    first. We label connected components and drop any whose bounding-box
    is compact (both axes < max_component_extent_px) or whose elongation
    (major/minor axis ratio) is below min_axis_ratio.

    Returns a new boolean mask with only elongated components retained.
    """
    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError("scipy required for streak source masking") from e
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    out = np.zeros_like(mask)
    # Bounding-box slices per component
    slices = ndimage.find_objects(labels)
    for comp_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        major = max(h, w)
        minor = max(min(h, w), 1)
        axis_ratio = major / minor
        # Keep only elongated OR long components -- reject compact blobs
        if (major <= max_component_extent_px
                and axis_ratio < min_axis_ratio):
            continue
        out[labels == comp_id] = True
    return out


def subtract_and_mask_stars(image: np.ndarray,
                              *, fwhm_px: float = 3.5,
                              detect_sigma: float = 4.0,
                              mask_radius_factor: float = 3.0,
                              ) -> np.ndarray:
    """Background-subtract and remove point sources, returning a residual
    image in which stars are zeroed out so that only trails (and noise)
    survive.

    Real CCD stars have PSF wings, saturation spikes and bleed trails
    that connect into large elongated components -- those defeat a naive
    "keep elongated components" streak filter. The robust approach used
    by production fast-mover pipelines is to detect the stars explicitly
    and mask a disc around each before searching for linear residuals.

    Returns a background-subtracted residual with circular regions around
    every detected source set to zero.
    """
    try:
        import numpy as _np
        from photutils.background import Background2D, MedianBackground
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
    except ImportError as e:
        raise ImportError("photutils + astropy required for star-subtracted "
                            "streak detection") from e
    data = np.asarray(image, dtype=float)
    bkg = Background2D(data, box_size=(64, 64),
                         bkg_estimator=MedianBackground())
    resid = data - bkg.background
    _mean, _med, std = sigma_clipped_stats(resid, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm_px, threshold=detect_sigma * std)
    tbl = finder(resid)
    if tbl is not None and len(tbl) > 0:
        H, W = resid.shape
        yy, xx = np.ogrid[0:H, 0:W]
        r_mask = mask_radius_factor * fwhm_px
        r2 = r_mask * r_mask
        x_col = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
        y_col = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
        # Mask each detected source. Vectorised disc stamp per source.
        half = int(math.ceil(r_mask))
        for row in tbl:
            sx = float(row[x_col]); sy = float(row[y_col])
            x0 = max(0, int(sx) - half); x1 = min(W, int(sx) + half + 1)
            y0 = max(0, int(sy) - half); y1 = min(H, int(sy) + half + 1)
            sub_y = yy[y0:y1, :]
            sub_x = xx[:, x0:x1]
            dist2 = (sub_x - sx) ** 2 + (sub_y - sy) ** 2
            resid[y0:y1, x0:x1][dist2 <= r2] = 0.0
    return resid


def hough_lines(image: np.ndarray,
                *,
                sigma_threshold: float = 4.0,
                n_theta: int = 180,
                rho_resolution: float = 1.0,
                min_votes: int = 30,
                top_n: int = 20,
                premask: np.ndarray | None = None,
                suppress_compact: bool = True) -> list[tuple[float, float, int]]:
    """Vectorised Hough line transform; return top-N (rho, theta, vote_count).

    Args:
      image:           2D image array.
      sigma_threshold: pixel must exceed (median + sigma*MAD) to vote.
      n_theta:         number of angle bins (default 180, 1-deg resolution).
      rho_resolution:  rho bin size in pixels.
      min_votes:       lines with fewer votes are discarded.
      top_n:           cap on number of returned lines.
      premask:         optional precomputed boolean mask of voting pixels.
                       If None, derived from `image` via sigma threshold.
      suppress_compact: if True (and no premask given), remove star-like
                       compact connected components before voting. This
                       both speeds the transform up by orders of magnitude
                       and removes spurious votes from crowded fields.

    Returns:
      List of (rho_pixels, theta_radians, vote_count), sorted by vote_count
      descending.

    The accumulator is built with a single vectorised np.add.at over the
    outer product of bright-pixel coordinates and angle bins -- O(N*T) in
    numpy rather than a Python double loop.
    """
    H, W = image.shape
    if premask is not None:
        mask = premask
    else:
        mask = _binarise(image, sigma_threshold=sigma_threshold)
        if suppress_compact:
            mask = mask_compact_sources(image, mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    thetas = np.linspace(-math.pi / 2, math.pi / 2, n_theta, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    rho_max = int(math.ceil(math.hypot(H, W) / rho_resolution))
    n_rho = 2 * rho_max + 1
    accumulator = np.zeros((n_rho, n_theta), dtype=np.int32)

    # Vectorised vote: rho_matrix[i, t] = x_i*cos_t + y_i*sin_t  (N x T)
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    rho_matrix = np.outer(xs_f, cos_t) + np.outer(ys_f, sin_t)
    rho_bins = np.round(rho_matrix / rho_resolution).astype(np.int64) + rho_max
    # Theta index per column, broadcast to the full N x T grid
    theta_idx = np.broadcast_to(np.arange(n_theta), rho_bins.shape)
    valid = (rho_bins >= 0) & (rho_bins < n_rho)
    flat_rho = rho_bins[valid]
    flat_theta = theta_idx[valid]
    np.add.at(accumulator, (flat_rho, flat_theta), 1)

    # find local maxima in accumulator (suppress neighbours).
    # The relative floor (fraction of the strongest line) must be low
    # enough that a faint short streak co-occurring with a bright long
    # one is not suppressed -- min_votes is the real false-positive guard.
    lines = []
    acc_max = accumulator.max() if accumulator.size > 0 else 0
    threshold = max(min_votes, int(0.15 * acc_max))
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
    # A genuine trail has CONTIGUOUS pixel coverage along its length.
    # Unrelated collinear star residuals sit far away with large gaps.
    # Restrict the endpoints to the longest gap-limited contiguous run so
    # we don't extend the streak through chance-aligned debris.
    order = np.argsort(tang)
    tang_sorted = tang[order]
    max_gap = max(3.0 * width_px, 6.0)
    gaps = np.diff(tang_sorted)
    # Segment boundaries where the gap exceeds max_gap
    breaks = np.where(gaps > max_gap)[0]
    seg_starts = np.concatenate(([0], breaks + 1))
    seg_ends = np.concatenate((breaks, [len(tang_sorted) - 1]))
    # Pick the segment with the greatest extent in t
    best_i = 0
    best_extent = -1.0
    for si, ei in zip(seg_starts, seg_ends):
        extent = tang_sorted[ei] - tang_sorted[si]
        if extent > best_extent:
            best_extent = extent
            best_i = (si, ei)
    si, ei = best_i
    t_min = float(tang_sorted[si]); t_max = float(tang_sorted[ei])
    # Restrict the on-line pixel set to that contiguous segment for the
    # flux / width measurements too.
    seg_mask = (tang >= t_min) & (tang <= t_max)
    xs_line = xs_line[seg_mask]
    ys_line = ys_line[seg_mask]
    seg_line_pix = line_pix[seg_mask]
    if len(seg_line_pix) < 3:
        return None
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
    n_pix = int(seg_line_pix.size)

    # Width estimate: FLUX-WEIGHTED perpendicular second moment, which
    # measures the PSF width independent of source brightness. A simple
    # percentile of above-threshold pixel offsets grows with brightness
    # (a brighter trail pushes more wing pixels over the threshold), which
    # wrongly inflated the width of bright trails and rejected them. The
    # flux-weighted sigma of the perpendicular profile is amplitude-
    # invariant: a Gaussian profile has the same sigma at any peak height.
    seg_perp = proj_perp[seg_line_pix]
    w = np.maximum(pixel_vals, 0.0)
    w_sum = float(w.sum())
    if w_sum > 0:
        mean_perp = float(np.sum(w * seg_perp) / w_sum)
        var_perp = float(np.sum(w * (seg_perp - mean_perp) ** 2) / w_sum)
        sigma_perp = math.sqrt(max(var_perp, 0.0))
        fwhm_perp = 2.3548 * sigma_perp
    else:
        fwhm_perp = 2.3548 * float(np.percentile(np.abs(seg_perp), 84))
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
                   max_streaks: int = 10,
                   subtract_stars: bool = False,
                   star_fwhm_px: float = 3.5) -> list[Streak]:
    """End-to-end streak detection: Hough transform + measurement + filter.

    Args:
      image:               2D image array (background not pre-subtracted is fine).
      sigma_threshold:     binarisation cut (above median + sigma*MAD).
      min_length_px:       reject streaks shorter than this.
      max_width_px:        reject streaks wider than this (extended sources).
      min_consistency:     reject streaks too wide for the PSF (extended fuzz).
      max_streaks:         cap on returned count.
      subtract_stars:      if True, background-subtract and mask detected
                           point sources before searching for trails. This
                           is REQUIRED on real CCD images -- real stars have
                           wings/spikes/bleed trails that connect into
                           elongated components and defeat the compact-source
                           filter. Synthetic clean fields do not need it.
      star_fwhm_px:        PSF FWHM used for the star detection/masking step.

    Returns:
      List of Streak records, sorted by total_flux descending.
    """
    if subtract_stars:
        # Work on a star-subtracted residual: stars zeroed, only trails +
        # noise remain. Threshold the residual relative to its own noise.
        resid = subtract_and_mask_stars(
            image, fwhm_px=star_fwhm_px, detect_sigma=sigma_threshold)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        noise = 1.4826 * max(mad, 1e-6)
        streak_mask = resid > med + sigma_threshold * noise
        meas_image = resid
        bg_med = med
    else:
        raw_mask = _binarise(image, sigma_threshold=sigma_threshold)
        # Suppress compact (star-like) sources so only elongated trails vote
        # + are measured (adequate only for clean/synthetic fields).
        streak_mask = mask_compact_sources(image, raw_mask)
        meas_image = image
        bg_med = float(np.median(image))
    lines = hough_lines(image, sigma_threshold=sigma_threshold,
                        top_n=max_streaks * 3, premask=streak_mask)
    streaks = []
    for rho, theta, votes in lines:
        s = _measure_streak(meas_image, streak_mask, rho, theta,
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
    streaks = _dedupe_streaks(streaks, tol_px=max(max_width_px, 4.0))
    streaks.sort(key=lambda x: x.total_flux, reverse=True)
    return streaks[:max_streaks]


def _streak_endpoints_match(a: Streak, b: Streak, tol_px: float) -> bool:
    """True if two streaks describe the same physical trail (endpoints
    coincide, allowing for endpoint-order swap)."""
    def close(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= tol_px
    a1, a2 = (a.x1, a.y1), (a.x2, a.y2)
    b1, b2 = (b.x1, b.y1), (b.x2, b.y2)
    return (close(a1, b1) and close(a2, b2)) or \
           (close(a1, b2) and close(a2, b1))


def _dedupe_streaks(streaks: list[Streak], *, tol_px: float = 4.0
                      ) -> list[Streak]:
    """Merge near-duplicate streaks (parallel Hough lines a few pixels
    apart in rho describe one trail). Keep the highest-flux representative
    of each group."""
    kept: list[Streak] = []
    for s in sorted(streaks, key=lambda x: x.total_flux, reverse=True):
        if any(_streak_endpoints_match(s, k, tol_px) for k in kept):
            continue
        kept.append(s)
    return kept


def classify_streak(streak: Streak, exposure_seconds: float = 30.0,
                    pixel_scale_arcsec: float = 0.25,
                    *, psf_fwhm_px: float = 3.0,
                    frame_diagonal_px: float | None = None) -> dict:
    """Distinguish asteroid trail / satellite trail / cosmic-ray trail.

    Physical reasoning (a detected streak is, by definition, an object
    that moved more than ~1 PSF during the exposure -- so it is never a
    "slow mover"; the question is only WHICH kind of fast mover):

      * Asteroid (NEO):   width consistent with the PSF (it is a point
                          source dragged along the trail). Length implies
                          a plausible angular rate (0.05 - ~50 arcsec/sec).
                          Does NOT span the whole frame.
      * Satellite (LEO):  spans a large fraction of the frame; angular
                          rate >> 50 arcsec/sec (typically hundreds).
      * Cosmic-ray trail: short, sub-PSF width (sharp, no PSF wings).
      * Extended/wide:    width >> PSF (galaxy edge, diffraction, blend).

    Returns a dict with `label`, `confidence`, `is_asteroid_candidate`,
    and the angular-rate estimate in both arcsec/sec and arcsec/hr.
    """
    rate_px_per_s = streak.length_px / max(exposure_seconds, 1e-3)
    rate_arcsec_per_s = rate_px_per_s * pixel_scale_arcsec
    rate_arcsec_per_hr = rate_arcsec_per_s * 3600.0
    width_in_psf = streak.width_px / max(psf_fwhm_px, 1e-3)

    # A streak spanning a large fraction of the frame diagonal is a
    # satellite regardless of width.
    spans_frame = (frame_diagonal_px is not None
                     and streak.length_px > 0.5 * frame_diagonal_px)

    is_asteroid = False
    if streak.width_px < 0.5 * psf_fwhm_px and streak.length_px < 10:
        # sharp, short, sub-PSF -> cosmic ray
        label = "cosmic_ray_trail"; conf = 0.75
    elif width_in_psf > 2.5:
        # much wider than the PSF -> extended source or saturated blend
        label = "extended_or_blended"; conf = 0.6
    elif spans_frame or rate_arcsec_per_s > 50.0:
        # crosses the field / hypersonic angular rate -> satellite
        label = "satellite"; conf = 0.8
    elif 0.5 <= width_in_psf <= 2.5 and rate_arcsec_per_s <= 50.0:
        # PSF-thin trail at an asteroid-plausible rate
        label = "asteroid_candidate"; conf = 0.8
        is_asteroid = True
    else:
        label = "unclassified_streak"; conf = 0.4

    return {
        "label": label, "confidence": conf,
        "is_asteroid_candidate": is_asteroid,
        "rate_arcsec_sec": rate_arcsec_per_s,
        "rate_arcsec_hr": rate_arcsec_per_hr,
        "length_px": streak.length_px, "width_px": streak.width_px,
        "width_in_psf": width_in_psf,
        "consistency": streak.consistency,
    }
