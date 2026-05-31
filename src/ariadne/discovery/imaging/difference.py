"""Difference imaging: subtract a reference frame to surface MOVING sources.

The single largest sensitivity boost in moving-object discovery: instead of
extracting sources from each image, subtract a static reference (or the
median of a stack) and run extraction on the residual. Static field stars and
galaxies cancel; only moving sources (asteroids, NEOs, TNOs) and transients
(SNe, novae, variable stars) remain. This routinely gives 2-3 magnitudes of
extra depth for moving-object detection on the same exposure.

Pipeline:

  1. ALIGN: register the science image to the reference via cross-correlation
     + sub-pixel interpolation (no need for full astrometric solution if the
     two frames are close on-sky).
  2. NORMALIZE: scale the reference to match the science image's flux level
     (PSF area-integrated counts).
  3. SUBTRACT: compute residual = science - aligned_reference.
  4. NOISE: estimate per-pixel sigma in the residual (combined variance).

This is a LIGHTWEIGHT implementation of the Alard-Lupton / ZOGY family. For
production work with PSF-matched kernels (the right way for changing seeing),
use hotpants or PyZOGY. The implementation here is good for the common case
where the science + reference were taken on the same telescope on similar nights.

Output: residual image array + per-pixel noise map. Feed downstream into
detect_sources_in_image to find the moving sources.

Reference: Alard & Lupton 1998 (image-subtraction formalism); Zackay-Ofek-Gal-Yam
2016 (ZOGY optimal subtraction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifferenceResult:
    """Result of one science-reference subtraction.

    Fields:
      residual:    science - aligned_reference, same shape as science.
      noise:       per-pixel sigma estimate (sqrt of combined variance).
      shift_px:    (dx, dy) the reference was shifted by to align.
      flux_scale:  the multiplicative scaling applied to the reference.
      n_sigma_max: peak |residual| in units of noise (sanity check).
    """
    residual: np.ndarray
    noise: np.ndarray
    shift_px: tuple
    flux_scale: float
    n_sigma_max: float


def _estimate_shift_xc(science: np.ndarray, reference: np.ndarray,
                       max_shift_px: int = 32) -> tuple[float, float]:
    """Cross-correlate (in Fourier space) to find the integer + sub-pixel shift.

    Returns (dx, dy) in pixels that reference should be shifted by to match
    science.
    """
    H, W = science.shape
    # FFT-based correlation; pad to power of 2 for speed
    F1 = np.fft.fft2(science - np.median(science))
    F2 = np.fft.fft2(reference - np.median(reference))
    xc = np.fft.ifft2(F1 * np.conj(F2)).real
    xc = np.fft.fftshift(xc)
    cy, cx = H // 2, W // 2

    # restrict to ±max_shift_px box
    lo_y = max(0, cy - max_shift_px); hi_y = min(H, cy + max_shift_px + 1)
    lo_x = max(0, cx - max_shift_px); hi_x = min(W, cx + max_shift_px + 1)
    box = xc[lo_y:hi_y, lo_x:hi_x]
    py, px = np.unravel_index(np.argmax(box), box.shape)
    dy = (py + lo_y) - cy
    dx = (px + lo_x) - cx

    # sub-pixel refinement via 1D parabolic interp on 3-point peak
    if 1 <= py < box.shape[0] - 1:
        f_m, f_0, f_p = box[py - 1, px], box[py, px], box[py + 1, px]
        denom = f_m - 2 * f_0 + f_p
        if abs(denom) > 1e-9:
            dy += 0.5 * (f_m - f_p) / denom
    if 1 <= px < box.shape[1] - 1:
        f_m, f_0, f_p = box[py, px - 1], box[py, px], box[py, px + 1]
        denom = f_m - 2 * f_0 + f_p
        if abs(denom) > 1e-9:
            dx += 0.5 * (f_m - f_p) / denom

    return float(dx), float(dy)


def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel-shift an image by (dx, dy) via Fourier phase shift.

    Out-of-bounds pixels wrap around (acceptable for fields where the moving
    object is far from any edge -- which we ensure by stamp-around-candidate).
    """
    H, W = image.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (kx * dx + ky * dy))
    return np.fft.ifft2(np.fft.fft2(image) * phase).real


def subtract_reference(science: np.ndarray, reference: np.ndarray,
                       *, max_shift_px: int = 32,
                       normalise: bool = True,
                       gain_e_per_adu: float = 1.0,
                       read_noise_e: float = 5.0) -> DifferenceResult:
    """Align reference to science, subtract, return residual + noise map.

    Args:
      science, reference:  2-D image arrays, same shape, same band.
      max_shift_px:        maximum expected dx/dy between science and reference.
      normalise:           if True, scale reference to match science flux median.
      gain_e_per_adu:      detector gain (used in noise calculation).
      read_noise_e:        per-pixel read noise in electrons.

    Returns:
      DifferenceResult with residual array + per-pixel noise sigma.
    """
    if science.shape != reference.shape:
        raise ValueError(f"science {science.shape} != reference {reference.shape}")
    if science.ndim != 2:
        raise ValueError("expected 2-D images")

    # 1. ALIGN
    dx, dy = _estimate_shift_xc(science, reference, max_shift_px=max_shift_px)
    aligned_ref = _shift_image(reference, dx, dy)

    # 2. NORMALIZE (flux scale via median)
    scale = 1.0
    if normalise:
        sci_med = float(np.median(science))
        ref_med = float(np.median(aligned_ref))
        if ref_med > 0:
            scale = sci_med / ref_med
            aligned_ref = aligned_ref * scale

    # 3. SUBTRACT
    residual = science - aligned_ref

    # 4. NOISE
    # Variance: photon noise from BOTH images (added in quadrature) + read noise.
    sci_var = np.maximum(science, 0.0) / gain_e_per_adu + read_noise_e ** 2
    ref_var = (np.maximum(aligned_ref, 0.0) / gain_e_per_adu + read_noise_e ** 2) * scale ** 2
    noise = np.sqrt(sci_var + ref_var)

    peak_sigma = float(np.max(np.abs(residual) / np.maximum(noise, 1e-6)))
    return DifferenceResult(residual=residual, noise=noise,
                            shift_px=(dx, dy), flux_scale=scale,
                            n_sigma_max=peak_sigma)


def build_reference_from_stack(images: list[np.ndarray]) -> np.ndarray:
    """Median-combine a list of co-registered images into a deep reference.

    Use this when no single survey reference exists -- stack 5-10 prior epochs
    of the same field. Median rejects moving objects; static field remains.

    Caller must pre-align images (e.g. via _estimate_shift_xc + _shift_image).
    """
    if not images:
        raise ValueError("no images supplied")
    if len({im.shape for im in images}) != 1:
        raise ValueError("all images must have the same shape")
    stack = np.stack(images, axis=0)
    return np.median(stack, axis=0)
