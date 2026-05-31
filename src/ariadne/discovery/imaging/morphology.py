"""Per-detection morphology classifier: what KIND of object is this pixel cluster?

For every source the detector flags, we want to answer:

  POINT            single unresolved source (asteroid, star, distant galaxy)
  EXTENDED         clearly resolved (nearby galaxy, comet coma, asterism)
  STREAK           elongated trail (fast NEO, satellite, meteor)
  BLEND            two or more overlapping point sources mistaken as one
  COSMIC_RAY       sharper than the PSF (single-pixel or knife-edge spike)
  EDGE_ARTEFACT    sits on a detector edge / row defect
  UNKNOWN          fit failed / data missing

The discriminator runs on the same postage stamp the PSF fitter uses. Decisions
come from a small set of physically-motivated measurements:

  * PSF chi-square:        large -> NOT a PSF, candidate EXTENDED / BLEND / streak
  * sharpness:             > 1.5 -> COSMIC_RAY (sub-PSF spike)
  * ellipticity / theta:   > 0.6 -> STREAK (with axis ratio); rule is direction-
                            agnostic so applies to NEO trails AND satellite trails
  * second-moment second peak: two local maxima above noise = BLEND
  * aperture-vs-PSF flux ratio: > 1.4 -> EXTENDED (galaxy: aperture flux >> PSF)
  * distance-to-edge:      < margin -> EDGE_ARTEFACT

Rule thresholds are documented + tunable. Each decision returns a confidence in
[0, 1] so downstream code can filter on a soft score, not a binary verdict.

Reference: Bertin & Arnouts 1996 (SExtractor classification); Howell 2006.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .psf_centroid import fit_psf_postage_stamp, PSFFit
from .source_extraction import Source


class MorphologyClass:
    """Enum-like class labels (use strings, not int enums, for JSON friendliness)."""
    POINT = "POINT"
    EXTENDED = "EXTENDED"
    STREAK = "STREAK"
    BLEND = "BLEND"
    COSMIC_RAY = "COSMIC_RAY"
    EDGE_ARTEFACT = "EDGE_ARTEFACT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MorphologyVerdict:
    """Per-source morphology metrics + classification.

    Fields:
      label:           one of MorphologyClass.* constants.
      confidence:      0..1 -- how confident this label is.
      n_peaks:         number of detected local maxima in the stamp (1 -> POINT/EXTENDED,
                       >=2 -> BLEND).
      chi2_psf:        reduced chi-square of the PSF fit (1-3 is good, >5 = NOT PSF).
      ellipticity:     1 - b/a from second moments. 0 = circle, >0.6 = streak-like.
      theta_rad:       streak major-axis position angle (radians from +x).
      sharpness:       central-pixel / peak ratio approximation (>1.5 -> cosmic ray).
      aper_over_psf:   aperture flux / PSF-model flux (>1.4 -> EXTENDED galaxy-like).
      psf_fit:         the underlying PSFFit (for inspection).
    """
    label: str
    confidence: float
    n_peaks: int
    chi2_psf: float
    ellipticity: float
    theta_rad: float
    sharpness: float
    aper_over_psf: float
    psf_fit: PSFFit


def _second_moments(stamp: np.ndarray, x_c: float, y_c: float,
                    bg: float) -> tuple[float, float]:
    """Major/minor axis lengths in pixels (eigenvalues of the second-moment tensor)."""
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    sub = np.clip(stamp - bg, 0.0, None)
    total = sub.sum()
    if total <= 0:
        return 1.0, 1.0
    dx = xx - x_c
    dy = yy - y_c
    Mxx = (sub * dx ** 2).sum() / total
    Myy = (sub * dy ** 2).sum() / total
    Mxy = (sub * dx * dy).sum() / total
    tr = Mxx + Myy
    diff = math.sqrt(max(0.0, (Mxx - Myy) ** 2 + 4 * Mxy ** 2))
    a = math.sqrt(max(0.0, 0.5 * (tr + diff)))
    b = math.sqrt(max(0.0, 0.5 * (tr - diff)))
    return a, b


def _theta_from_moments(stamp: np.ndarray, x_c: float, y_c: float,
                        bg: float) -> float:
    """Streak position angle from second moments (radians)."""
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    sub = np.clip(stamp - bg, 0.0, None)
    total = sub.sum()
    if total <= 0:
        return 0.0
    dx = xx - x_c
    dy = yy - y_c
    Mxx = (sub * dx ** 2).sum() / total
    Myy = (sub * dy ** 2).sum() / total
    Mxy = (sub * dx * dy).sum() / total
    return 0.5 * math.atan2(2.0 * Mxy, Mxx - Myy)


def _count_peaks(stamp: np.ndarray, bg: float, noise: float,
                 min_sep_px: float = 3.0, sigma_above_bg: float = 7.0,
                 secondary_peak_min_frac: float = 0.30) -> int:
    """Count physically-distinct local maxima.

    A peak counts only when:
      * it sits above (bg + sigma_above_bg * noise),
      * it is the brightest pixel in its 3x3 neighborhood,
      * it is separated from every previously-found peak by >= min_sep_px,
      * AND (if there's already a peak) it has at least secondary_peak_min_frac
        of the brightest peak's amplitude. This last rule prevents noise
        wiggles in the wings of a single bright star from being counted as
        additional sources -- a real blend has two peaks of similar brightness.
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
                cands.append((v, i, j))
    cands.sort(reverse=True)            # brightest first
    peaks = []
    brightest_amp = (cands[0][0] - bg) if cands else 0.0
    for v, i, j in cands:
        amp = v - bg
        if peaks and amp < secondary_peak_min_frac * brightest_amp:
            continue
        if any(math.hypot(i - pi, j - pj) < min_sep_px for pi, pj in peaks):
            continue
        peaks.append((i, j))
    return len(peaks)


def classify_source(image_data: np.ndarray, source: Source,
                    *, half_size: int = 7,
                    aperture_radius_px: float = 5.0,
                    edge_margin_px: int = 4,
                    fwhm_guess_px: float = 3.0,
                    ellip_streak_threshold: float = 0.55,
                    chi2_extended_threshold: float = 5.0,
                    aper_extended_threshold: float = 1.4,
                    sharpness_cr_threshold: float = 1.7) -> MorphologyVerdict:
    """Classify one detected source into POINT / EXTENDED / STREAK / BLEND / COSMIC_RAY / EDGE.

    Args:
      image_data:        full 2-D image array (background-subtracted preferred).
      source:            the Source from detect_sources_in_image.
      half_size:         stamp half-width; the full stamp is (2*hs+1)^2.
      aperture_radius_px:circular aperture for aperture-vs-PSF flux comparison.
      edge_margin_px:    sources within this distance of the edge -> EDGE_ARTEFACT.
      fwhm_guess_px:     initial PSF guess for the underlying fit.
      *_threshold:       decision thresholds (see module docstring).

    Returns:
      MorphologyVerdict (label, confidence, raw metrics, underlying PSFFit).
    """
    H, W = image_data.shape
    ix = int(round(source.x)); iy = int(round(source.y))
    if (ix < edge_margin_px or iy < edge_margin_px or
            ix >= W - edge_margin_px or iy >= H - edge_margin_px):
        return MorphologyVerdict(
            label=MorphologyClass.EDGE_ARTEFACT, confidence=0.95,
            n_peaks=0, chi2_psf=float("nan"), ellipticity=0.0,
            theta_rad=0.0, sharpness=0.0, aper_over_psf=0.0,
            psf_fit=PSFFit(source.x, source.y, 1.0, 1.0, 0.5,
                           source.flux, source.fwhm_px, False, 1e30),
        )

    # 1. PSF fit (chi^2)
    psf = fit_psf_postage_stamp(image_data, source.x, source.y,
                                  half_size=half_size,
                                  fwhm_guess_px=fwhm_guess_px)
    chi2 = psf.chi2_red

    # 2. Stamp metrics
    x_lo = max(0, ix - half_size); x_hi = min(W, ix + half_size + 1)
    y_lo = max(0, iy - half_size); y_hi = min(H, iy + half_size + 1)
    stamp = image_data[y_lo:y_hi, x_lo:x_hi].astype(float)
    bg = float(np.percentile(stamp, 25))
    noise = float(np.std(stamp[stamp < np.percentile(stamp, 50)]))
    if noise <= 0:
        noise = max(1.0, float(abs(stamp.std())))
    peak = float(stamp.max())
    sharpness = (peak - bg) / max(noise * fwhm_guess_px ** 2, 1.0)
    # stamp-local centroid for moment math
    sub_for_moments = np.clip(stamp - bg, 0.0, None)
    if sub_for_moments.sum() > 0:
        yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
        x_c = (sub_for_moments * xx).sum() / sub_for_moments.sum()
        y_c = (sub_for_moments * yy).sum() / sub_for_moments.sum()
    else:
        x_c = stamp.shape[1] / 2; y_c = stamp.shape[0] / 2

    # second-moment major/minor (pixel units in stamp frame)
    a_px, b_px = _second_moments(stamp, x_c, y_c, bg)
    if a_px > 0:
        ellip = 1.0 - b_px / a_px
    else:
        ellip = 0.0
    theta = _theta_from_moments(stamp, x_c, y_c, bg)

    # peak count
    n_peaks = _count_peaks(stamp, bg, noise)

    # aperture flux (circular aperture, radius=aperture_radius_px)
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    rr = np.hypot(xx - x_c, yy - y_c)
    mask = rr <= aperture_radius_px
    aper_flux = float((stamp[mask] - bg).sum())
    psf_flux = max(psf.flux, 1e-6)
    aper_over_psf = aper_flux / psf_flux

    # 3. Decision tree -- order matters: edge > cosmic > blend > streak > extended > point
    # EDGE handled above.
    # COSMIC RAY: sharpness > threshold (sub-PSF spike). Confirmed by very small FWHM.
    if sharpness > sharpness_cr_threshold and psf.fwhm_px < 0.7 * fwhm_guess_px:
        return MorphologyVerdict(
            label=MorphologyClass.COSMIC_RAY,
            confidence=min(0.99, 0.5 + 0.5 * min(1.0, sharpness / sharpness_cr_threshold - 1)),
            n_peaks=n_peaks, chi2_psf=chi2, ellipticity=ellip,
            theta_rad=theta, sharpness=sharpness, aper_over_psf=aper_over_psf,
            psf_fit=psf,
        )

    # BLEND: two+ peaks separated by > min_sep_px above 5 sigma
    if n_peaks >= 2:
        return MorphologyVerdict(
            label=MorphologyClass.BLEND,
            confidence=min(0.99, 0.5 + 0.15 * (n_peaks - 1)),
            n_peaks=n_peaks, chi2_psf=chi2, ellipticity=ellip,
            theta_rad=theta, sharpness=sharpness, aper_over_psf=aper_over_psf,
            psf_fit=psf,
        )

    # STREAK: elongated (1 peak, high ellipticity)
    if ellip > ellip_streak_threshold:
        return MorphologyVerdict(
            label=MorphologyClass.STREAK,
            confidence=min(0.99, (ellip - ellip_streak_threshold) /
                            max(1e-6, (1.0 - ellip_streak_threshold))),
            n_peaks=n_peaks, chi2_psf=chi2, ellipticity=ellip,
            theta_rad=theta, sharpness=sharpness, aper_over_psf=aper_over_psf,
            psf_fit=psf,
        )

    # EXTENDED: aperture flux >> PSF flux, or PSF chi^2 large (poor PSF fit)
    if aper_over_psf > aper_extended_threshold or chi2 > chi2_extended_threshold:
        score = 0.5 * (max(0, aper_over_psf - aper_extended_threshold) /
                       max(1e-6, 2.0 - aper_extended_threshold))
        score += 0.5 * (min(1.0, max(0, chi2 - chi2_extended_threshold) /
                            max(1e-6, 20.0 - chi2_extended_threshold)))
        return MorphologyVerdict(
            label=MorphologyClass.EXTENDED,
            confidence=min(0.99, 0.4 + score),
            n_peaks=n_peaks, chi2_psf=chi2, ellipticity=ellip,
            theta_rad=theta, sharpness=sharpness, aper_over_psf=aper_over_psf,
            psf_fit=psf,
        )

    # Otherwise: POINT source (clean PSF, single peak, low ellipticity, sane chi^2)
    point_conf = 1.0
    point_conf *= max(0.2, 1.0 - chi2 / max(1e-6, chi2_extended_threshold))
    point_conf *= max(0.2, 1.0 - ellip / max(1e-6, ellip_streak_threshold))
    point_conf *= max(0.2, min(1.0, 1.5 / max(1e-6, aper_over_psf)))
    return MorphologyVerdict(
        label=MorphologyClass.POINT,
        confidence=min(0.99, max(0.25, point_conf)),
        n_peaks=n_peaks, chi2_psf=chi2, ellipticity=ellip,
        theta_rad=theta, sharpness=sharpness, aper_over_psf=aper_over_psf,
        psf_fit=psf,
    )


def classify_sources(image_data: np.ndarray, sources: list[Source],
                     **kwargs) -> list[tuple[Source, MorphologyVerdict]]:
    """Classify every source. Returns paired (source, verdict) list."""
    return [(s, classify_source(image_data, s, **kwargs)) for s in sources]


def filter_pointlike(image_data: np.ndarray, sources: list[Source],
                     *, min_confidence: float = 0.4, **kwargs) -> list[Source]:
    """Keep only point-source-like detections (asteroid candidates).

    Filters out: BLEND, EXTENDED, STREAK, COSMIC_RAY, EDGE_ARTEFACT.
    Use upstream of tracklet building when you want a clean asteroid-only catalogue.
    """
    out = []
    for s, v in classify_sources(image_data, sources, **kwargs):
        if v.label == MorphologyClass.POINT and v.confidence >= min_confidence:
            out.append(s)
    return out
