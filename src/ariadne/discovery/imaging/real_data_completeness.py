"""Real-data completeness + recall characterisation via source injection.

This is the survey-grade calibration that synthetic-field tests cannot
provide: inject artificial sources of KNOWN magnitude directly into the
REAL DECam CCD pixels (with their real noise, real background, real
crowding, real artefacts), then run the ACTUAL extraction pipeline and
measure what fraction is recovered as a function of magnitude.

This is exactly how PanSTARRS / DES / Rubin measure their detection
completeness ("fake source injection" / "synthetic source recovery").

Public API:
  inject_psf_source(image, x, y, flux, fwhm_px)        -- add one fake star
  inject_trail(image, x1, y1, x2, y2, total_flux, fwhm_px) -- add one streak
  measure_point_source_completeness(ccd, mag_grid, ...)   -- recall vs mag
  measure_streak_recall(ccd, rate_grid, ...)              -- recall vs rate
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def inject_psf_source(image: np.ndarray, x: float, y: float,
                        flux: float, fwhm_px: float = 3.5) -> None:
    """Add a single Gaussian PSF source of total `flux` at (x, y) in place."""
    sigma = fwhm_px / 2.3548
    half = int(math.ceil(4 * sigma))
    H, W = image.shape
    x0 = int(round(x)); y0 = int(round(y))
    norm = flux / (2 * math.pi * sigma * sigma)
    for dy in range(-half, half + 1):
        yy = y0 + dy
        if yy < 0 or yy >= H:
            continue
        for dx in range(-half, half + 1):
            xx = x0 + dx
            if xx < 0 or xx >= W:
                continue
            r2 = (xx - x) ** 2 + (yy - y) ** 2
            image[yy, xx] += norm * math.exp(-r2 / (2 * sigma * sigma))


def inject_trail(image: np.ndarray, x1: float, y1: float,
                   x2: float, y2: float, total_flux: float,
                   fwhm_px: float = 3.5) -> None:
    """Add a uniform-surface-brightness trail (a PSF dragged from (x1,y1)
    to (x2,y2)) carrying `total_flux` spread along its length."""
    length = math.hypot(x2 - x1, y2 - y1)
    n_steps = max(int(length * 3), 2)
    flux_per_step = total_flux / n_steps
    for t in np.linspace(0, 1, n_steps):
        xx = x1 + t * (x2 - x1)
        yy = y1 + t * (y2 - y1)
        inject_psf_source(image, xx, yy, flux_per_step, fwhm_px)


@dataclass
class CompletenessPoint:
    mag: float
    n_injected: int
    n_recovered: int
    recall: float


@dataclass
class CompletenessReport:
    points: list[CompletenessPoint] = field(default_factory=list)
    limiting_mag_50: float = float("nan")   # mag at 50% recall
    limiting_mag_90: float = float("nan")   # mag at 90% recall
    zeropoint: float = float("nan")
    sky_noise: float = float("nan")


def _flux_for_mag(mag: float, zeropoint: float) -> float:
    """Invert mag = ZP - 2.5 log10(flux)  ->  flux."""
    return 10 ** ((zeropoint - mag) / 2.5)


def measure_point_source_completeness(
        science: np.ndarray, wcs, magzero: float, mjd: float,
        *, mag_grid: list[float] | None = None,
        n_per_mag: int = 200,
        fwhm_px: float = 3.5,
        detect_sigma: float = 5.0,
        match_tol_px: float = 2.5,
        edge_margin_px: int = 20,
        rng_seed: int = 0,
        ) -> CompletenessReport:
    """Inject `n_per_mag` fake PSF sources at each magnitude in `mag_grid`
    into the real CCD pixels, run the real DAOStarFinder extraction, and
    measure recall = recovered / injected per magnitude.

    Returns a CompletenessReport including the 50% and 90% limiting
    magnitudes (the standard survey depth metrics).
    """
    from .source_extraction import detect_sources_in_image

    if mag_grid is None:
        mag_grid = [19.0, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
    H, W = science.shape
    rng = np.random.default_rng(rng_seed)
    # sky noise estimate (robust)
    med = float(np.median(science))
    mad = float(np.median(np.abs(science - med)))
    sky_noise = 1.4826 * mad

    report = CompletenessReport(zeropoint=magzero, sky_noise=sky_noise)
    for mag in mag_grid:
        flux = _flux_for_mag(mag, magzero)
        work = science.copy()
        injected_xy = []
        for _ in range(n_per_mag):
            x = float(rng.integers(edge_margin_px, W - edge_margin_px))
            y = float(rng.integers(edge_margin_px, H - edge_margin_px))
            inject_psf_source(work, x, y, flux, fwhm_px)
            injected_xy.append((x, y))
        # Run the REAL extraction on the injected frame
        try:
            srcs = detect_sources_in_image(
                work, wcs, mjd=mjd, image_id="inj",
                fwhm_px=fwhm_px, threshold_sigma=detect_sigma,
                zeropoint_mag=magzero)
        except Exception:
            srcs = []
        # Match recovered sources to injections by pixel proximity
        rec_xy = np.array([[s.x, s.y] for s in srcs]) if srcs else np.empty((0, 2))
        n_rec = 0
        for (ix, iy) in injected_xy:
            if rec_xy.shape[0] == 0:
                break
            d = np.hypot(rec_xy[:, 0] - ix, rec_xy[:, 1] - iy)
            if d.min() <= match_tol_px:
                n_rec += 1
        recall = n_rec / n_per_mag
        report.points.append(CompletenessPoint(
            mag=mag, n_injected=n_per_mag, n_recovered=n_rec, recall=recall))

    # Interpolate the 50% / 90% limiting magnitudes (recall decreasing
    # with fainter mag)
    pts = sorted(report.points, key=lambda p: p.mag)
    report.limiting_mag_50 = _interp_limit(pts, 0.5)
    report.limiting_mag_90 = _interp_limit(pts, 0.9)
    return report


def _interp_limit(points, level: float) -> float:
    """Find the magnitude where recall crosses `level` (linear interp)."""
    for i in range(len(points) - 1):
        r1 = points[i].recall
        r2 = points[i + 1].recall
        if r1 >= level >= r2:
            m1, m2 = points[i].mag, points[i + 1].mag
            if r1 == r2:
                return m1
            frac = (r1 - level) / (r1 - r2)
            return m1 + frac * (m2 - m1)
    return float("nan")


@dataclass
class StreakRecallPoint:
    rate_arcsec_hr: float
    length_px: float
    mag: float
    n_injected: int
    n_recovered: int
    recall: float


def measure_streak_recall(
        science: np.ndarray, magzero: float,
        *, rate_grid_arcsec_hr: list[float] | None = None,
        mag: float = 21.0,
        exposure_seconds: float = 90.0,
        pixel_scale_arcsec: float = 0.263,
        n_per_rate: int = 30,
        fwhm_px: float = 3.5,
        detect_sigma: float = 4.0,
        match_tol_px: float = 15.0,
        edge_margin_px: int = 80,
        rng_seed: int = 0,
        ) -> list[StreakRecallPoint]:
    """Inject fake trails of a fixed magnitude at a grid of angular rates
    into the real CCD pixels, run the real streak detector, and measure
    recall as a function of rate.

    A higher rate spreads the same total flux over a longer trail (lower
    surface brightness), so recall falls off at both the slow end (too
    short to look like a trail) and the fast end (too faint per pixel).
    """
    from .streaks import detect_streaks, classify_streak

    if rate_grid_arcsec_hr is None:
        rate_grid_arcsec_hr = [200, 500, 1000, 2000, 4000, 8000, 16000]
    H, W = science.shape
    rng = np.random.default_rng(rng_seed)
    flux = _flux_for_mag(mag, magzero)
    frame_diag = math.hypot(W, H)
    out = []
    for rate in rate_grid_arcsec_hr:
        # length (px) = rate * exposure / pixel_scale
        length_px = rate * (exposure_seconds / 3600.0) / pixel_scale_arcsec
        n_rec = 0
        for _ in range(n_per_rate):
            work = science.copy()
            ang = rng.uniform(0, math.pi)
            cx = rng.uniform(edge_margin_px, W - edge_margin_px)
            cy = rng.uniform(edge_margin_px, H - edge_margin_px)
            dx = 0.5 * length_px * math.cos(ang)
            dy = 0.5 * length_px * math.sin(ang)
            x1, y1 = cx - dx, cy - dy
            x2, y2 = cx + dx, cy + dy
            inject_trail(work, x1, y1, x2, y2, flux, fwhm_px)
            streaks = detect_streaks(
                work, sigma_threshold=detect_sigma,
                min_length_px=min(20.0, max(length_px * 0.5, 6.0)),
                max_width_px=6.0)
            # Did we recover a streak near the injected midpoint?
            for s in streaks:
                sm_x = 0.5 * (s.x1 + s.x2)
                sm_y = 0.5 * (s.y1 + s.y2)
                if math.hypot(sm_x - cx, sm_y - cy) <= match_tol_px:
                    cls = classify_streak(
                        s, exposure_seconds=exposure_seconds,
                        pixel_scale_arcsec=pixel_scale_arcsec,
                        psf_fwhm_px=fwhm_px, frame_diagonal_px=frame_diag)
                    if cls["is_asteroid_candidate"]:
                        n_rec += 1
                        break
        out.append(StreakRecallPoint(
            rate_arcsec_hr=rate, length_px=length_px, mag=mag,
            n_injected=n_per_rate, n_recovered=n_rec,
            recall=n_rec / n_per_rate))
    return out
