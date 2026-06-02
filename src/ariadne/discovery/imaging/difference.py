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
      residual:    science - matched_reference, same shape as science.
                   POSITIVE = flux present in science but not the reference
                   (i.e. a moving object / transient), regardless of which
                   frame was convolved to match the other.
      noise:       per-pixel sigma estimate (sqrt of combined variance).
      shift_px:    (dx, dy) the reference was shifted by to align.
      flux_scale:  the multiplicative scaling applied to the reference.
      n_sigma_max: peak |residual| in units of noise (sanity check).
      method:      "scalar" (crude flux-scaled) or "psf-matched" (Alard-Lupton).
      kernel:      the matching kernel that was solved (None for scalar method).
      n_stars:     number of static stars used to fit the kernel (0 for scalar).
    """
    residual: np.ndarray
    noise: np.ndarray
    shift_px: tuple
    flux_scale: float
    n_sigma_max: float
    method: str = "scalar"
    kernel: np.ndarray | None = None
    n_stars: int = 0


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


# ---------------------------------------------------------------------------
# PSF-matched (Alard-Lupton) difference imaging
#
# The crude `subtract_reference` above flux-scales the reference and subtracts.
# When the two epochs have DIFFERENT seeing (the normal case), every static
# star leaves a bright/dark DIPOLE residual: the narrower PSF over-subtracts in
# the core and under-subtracts in the wings. Those dipoles are false positives
# AND they raise the local noise floor, which is why crude differencing did NOT
# improve real-data recall.
#
# The fix (Alard & Lupton 1998): before subtracting, convolve the sharper image
# with a small matching kernel K so its PSF equals the broader image's PSF. We
# solve K empirically from the actual static stars (so it captures the real PSF
# shape, wings and all, not an assumed Gaussian) as a linear least-squares over
# a basis of Gaussians modulated by low-order polynomials. Static stars then
# cancel to the noise; only moving/transient flux survives.
# ---------------------------------------------------------------------------

def _kernel_basis(half: int, sigmas, max_deg: int) -> list[np.ndarray]:
    """Alard-Lupton kernel basis: Gaussians x 2-D polynomials.

    Each Gaussian of width `s` is multiplied by every monomial x^i y^j with
    i+j <= max_deg. The least-squares fit picks the linear combination that
    turns the source PSF into the target PSF (broadening, recentering, and
    mild shape changes are all representable).
    """
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = xx * xx + yy * yy
    basis = []
    for s in sigmas:
        g = np.exp(-r2 / (2.0 * s * s))
        for i in range(max_deg + 1):
            for j in range(max_deg + 1 - i):
                basis.append(g * (xx.astype(float) ** i) * (yy.astype(float) ** j))
    return basis


def _detect_static_stars(reference: np.ndarray, *, fwhm_px: float = 4.0,
                         min_snr: float = 20.0, n_stars: int = 50,
                         edge: int = 20):
    """Bright, well-separated stars from the REFERENCE (a star-only frame, so
    no moving objects contaminate the kernel fit). Returns [(x, y), ...]."""
    try:
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
    except ImportError:
        return []
    data = np.asarray(reference, float)
    data = np.where(np.isfinite(data), data, np.nanmedian(data))
    _m, med, std = sigma_clipped_stats(data, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return []
    tbl = DAOStarFinder(fwhm=fwhm_px, threshold=min_snr * std)(data - med)
    if tbl is None or len(tbl) == 0:
        return []
    xcol = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
    H, W = data.shape
    # brightest first, drop saturated / edge sources
    rows = sorted(list(tbl), key=lambda r: -float(r["flux"]))
    out = []
    for r in rows:
        x, y = float(r[xcol]), float(r[ycol])
        if x < edge or y < edge or x > W - edge or y > H - edge:
            continue
        out.append((int(round(x)), int(round(y))))
        if len(out) >= n_stars:
            break
    return out


def _solve_matching_kernel(target: np.ndarray, source: np.ndarray, stars,
                           *, half: int, sigmas, max_deg: int, stamp: int):
    """Least-squares matching kernel K so that (source (*) K) ~= target.

    Fit over background-subtracted stamps around `stars`. The valid (edge-free)
    central region of each convolved stamp is stacked into one linear system.
    Returns (K, n_used) or (None, 0) if it cannot be solved.
    """
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        return None, 0
    basis = _kernel_basis(half, sigmas, max_deg)
    crop = half  # 'same'-convolution edge contamination width
    A_blocks, b_blocks, used = [], [], 0
    for (x, y) in stars:
        ts = target[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1]
        ss = source[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1]
        if ts.shape != (2 * stamp + 1, 2 * stamp + 1) or ts.shape != ss.shape:
            continue
        if not (np.isfinite(ts).all() and np.isfinite(ss).all()):
            continue
        ts = ts - np.median(ts)
        ss = ss - np.median(ss)
        cols = [fftconvolve(ss, k, mode="same")[crop:-crop, crop:-crop].ravel()
                for k in basis]
        A_blocks.append(np.stack(cols, axis=1))
        b_blocks.append(ts[crop:-crop, crop:-crop].ravel())
        used += 1
    if used < 4:
        return None, used
    A = np.vstack(A_blocks)
    b = np.concatenate(b_blocks)
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, used
    K = np.zeros_like(basis[0])
    for c, k in zip(coef, basis):
        K = K + c * k
    return K, used


def psf_matched_difference(science: np.ndarray, reference: np.ndarray, *,
                           max_shift_px: int = 32,
                           kernel_half: int = 7,
                           sigmas=(0.7, 1.5, 3.0),
                           max_deg: int = 2,
                           fwhm_px: float = 4.0,
                           min_stars: int = 6,
                           gain_e_per_adu: float = 1.0,
                           read_noise_e: float = 5.0) -> DifferenceResult:
    """Alard-Lupton PSF-matched subtraction -- the RIGHT way to difference.

    Aligns the reference, measures which frame has the sharper PSF, convolves
    that one with an empirically-fit matching kernel so both PSFs agree, then
    subtracts. The result's POSITIVE residual is flux in the science epoch that
    is absent from the reference (a mover / transient). Static stars cancel to
    the noise -- no dipoles -- which is what crude scalar subtraction could not
    do.

    Falls back to `subtract_reference` (crude scalar) if photutils/scipy are
    missing or too few static stars are found.
    """
    if science.shape != reference.shape:
        raise ValueError(f"science {science.shape} != reference {reference.shape}")
    if science.ndim != 2:
        raise ValueError("expected 2-D images")

    # 1. ALIGN reference onto science (bulk integer+subpixel shift; the kernel
    #    mops up any residual sub-pixel offset within its footprint).
    dx, dy = _estimate_shift_xc(science, reference, max_shift_px=max_shift_px)
    aligned_ref = _shift_image(reference, dx, dy)

    # 2. Which frame is sharper? Convolve the SHARPER one (broadening is well
    #    posed; sharpening would amplify noise / ring).
    try:
        from .trailed_rate import stellar_psf_fwhm
        f_sci = stellar_psf_fwhm(science, fwhm_px=fwhm_px)
        f_ref = stellar_psf_fwhm(aligned_ref, fwhm_px=fwhm_px)
    except Exception:
        f_sci = f_ref = fwhm_px

    stamp = kernel_half + 8
    stars = _detect_static_stars(aligned_ref, fwhm_px=fwhm_px, n_stars=50,
                                 edge=stamp + 2)

    if len(stars) < min_stars:
        # not enough static structure to fit a kernel -> crude path
        res = subtract_reference(science, reference, max_shift_px=max_shift_px,
                                 gain_e_per_adu=gain_e_per_adu,
                                 read_noise_e=read_noise_e)
        return res

    if f_ref <= f_sci:
        # reference sharper -> convolve reference up to science PSF
        K, n_used = _solve_matching_kernel(science, aligned_ref, stars,
                                           half=kernel_half, sigmas=sigmas,
                                           max_deg=max_deg, stamp=stamp)
        if K is None:
            return subtract_reference(science, reference,
                                      max_shift_px=max_shift_px,
                                      gain_e_per_adu=gain_e_per_adu,
                                      read_noise_e=read_noise_e)
        from scipy.signal import fftconvolve
        matched = fftconvolve(aligned_ref, K, mode="same")
        residual = science - matched
        ref_var = np.maximum(aligned_ref, 0.0) / gain_e_per_adu + read_noise_e ** 2
        matched_var = fftconvolve(ref_var, K * K, mode="same")
        sci_var = np.maximum(science, 0.0) / gain_e_per_adu + read_noise_e ** 2
    else:
        # science sharper -> convolve science down to reference PSF, subtract ref
        K, n_used = _solve_matching_kernel(aligned_ref, science, stars,
                                           half=kernel_half, sigmas=sigmas,
                                           max_deg=max_deg, stamp=stamp)
        if K is None:
            return subtract_reference(science, reference,
                                      max_shift_px=max_shift_px,
                                      gain_e_per_adu=gain_e_per_adu,
                                      read_noise_e=read_noise_e)
        from scipy.signal import fftconvolve
        matched = fftconvolve(science, K, mode="same")
        residual = matched - aligned_ref
        sci_var0 = np.maximum(science, 0.0) / gain_e_per_adu + read_noise_e ** 2
        sci_var = fftconvolve(sci_var0, K * K, mode="same")
        matched_var = np.maximum(aligned_ref, 0.0) / gain_e_per_adu + read_noise_e ** 2

    # center the residual (differential background)
    bg = float(np.nanmedian(residual))
    residual = residual - bg
    noise = np.sqrt(np.maximum(sci_var, 0.0) + np.maximum(matched_var, 0.0))
    peak_sigma = float(np.max(np.abs(residual) / np.maximum(noise, 1e-6)))
    flux_scale = float(np.sum(K)) if K is not None else 1.0
    return DifferenceResult(residual=residual, noise=noise, shift_px=(dx, dy),
                            flux_scale=flux_scale, n_sigma_max=peak_sigma,
                            method="psf-matched", kernel=K, n_stars=n_used)
