"""Pixel-likelihood orbit refinement.

Every IOD strategy in `iod_advanced.py` produces an (x, v) state vector
that's then LM-refined by minimizing the centroid-residual RMS: the sum
of squared angular differences between the centroids the source-
extraction found and the centroids the orbit predicts.

That's the conventional approach but it throws away most of the
information in the actual image pixels. The centroid is a 2-number
summary of a 9x9 PSF pattern; the OTHER 79 numbers per detection
also constrain where the object is. A proper-uncertainty IOD would
maximize P(image_pixels | orbit) instead of P(centroid | orbit).

Pixel-likelihood scoring:

  For a candidate orbit (x, v, t_ref):
    1. Propagate to each image's epoch -> predicted pixel position (xi, yi).
    2. At that predicted position, the image SHOULD show a PSF with
       amplitude A and shape (xi, yi, sigma_psf).
    3. The likelihood is P(pixels_in_patch | A, xi, yi, sigma_psf):
         L = product over pixels of N(pix_value; bg + A * PSF(...), sigma_pix)
    4. Sum log-L over all images.

Maximizing log-L over (A, x, v) is a coupled nonlinear fit. We use
the simpler approach: hold A fixed at the per-image best-fit, optimize
(x, v) with a numerical Nelder-Mead. The fit landscape is much smoother
than centroid-RMS because every pixel contributes.

Benefit: catches IODs that converge to plausible but wrong orbits --
the centroid residual may be small while the pixel pattern shows the
object is somewhere else, OR there's NOTHING at the predicted position.

Public API:

  patch_log_likelihood(patch, A, sigma_psf, bg, noise_sigma)
        Log-likelihood of one PSF-fit patch given parameters.

  orbit_pixel_log_likelihood(x, v, t_ref, images, wcs_list, image_ets,
                              sigma_psf=1.5)
        Total log-L of a candidate orbit across all input images.

  refine_orbit_against_pixels(x_init, v_init, t_ref, images, wcs_list,
                                image_ets)
        Nelder-Mead refine of (x, v) to maximize pixel log-L. Returns
        refined (x, v) plus the improvement in log-L.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PixelRefinementResult:
    """Output of `refine_orbit_against_pixels`."""
    converged: bool
    x_refined: np.ndarray
    v_refined: np.ndarray
    log_l_initial: float
    log_l_refined: float
    log_l_improvement: float
    n_iterations: int = 0
    notes: str = ""


def _gaussian_psf(xx: np.ndarray, yy: np.ndarray,
                    x0: float, y0: float, amp: float, sigma: float,
                    bg: float = 0.0) -> np.ndarray:
    """2-D Gaussian PSF model + constant background."""
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return bg + amp * np.exp(-r2 / (2.0 * sigma * sigma))


def patch_log_likelihood(patch: np.ndarray,
                           x0: float, y0: float,
                           sigma_psf: float = 1.5,
                           bg: float | None = None,
                           noise_sigma: float | None = None,
                           ) -> float:
    """Return log P(patch | PSF at x0,y0 with shape sigma_psf).

    The PSF AMPLITUDE is profile-marginalised: we solve for the best-fit
    A analytically given the model, then plug back in to get the log-
    likelihood at the optimum amplitude.
    """
    if patch.size == 0:
        return 0.0
    ny, nx = patch.shape
    yy, xx = np.indices(patch.shape)
    # Profile = Gaussian PSF shape (unit amplitude, zero bg)
    profile = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma_psf ** 2))
    # Background
    if bg is None:
        # Use sigma-clipped median outside the PSF core
        mask = profile < 0.1
        if np.any(mask):
            bg = float(np.median(patch[mask]))
        else:
            bg = float(np.median(patch))
    # Noise: MAD of off-source pixels
    if noise_sigma is None:
        mask = profile < 0.1
        off = patch[mask]
        if off.size > 4:
            noise_sigma = float(np.median(np.abs(off - bg)) * 1.4826)
        else:
            noise_sigma = float(np.std(patch))
        noise_sigma = max(noise_sigma, 1e-6)
    # Best-fit amplitude (least-squares against the profile)
    residual = patch - bg
    num = float(np.sum(residual * profile))
    den = float(np.sum(profile * profile))
    if den <= 0:
        A_hat = 0.0
    else:
        A_hat = num / den
    # Model at A_hat
    model = bg + A_hat * profile
    res = patch - model
    # Gaussian log-likelihood (drop constants)
    n_eff = patch.size
    log_l = (-0.5 * float(np.sum(res * res)) / (noise_sigma * noise_sigma)
             - n_eff * math.log(noise_sigma * math.sqrt(2.0 * math.pi)))
    return log_l


def _crop_patch(image: np.ndarray, x_c: float, y_c: float,
                  half_size: int = 8) -> tuple[np.ndarray, float, float]:
    """Crop a patch around (x_c, y_c). Returns (patch, x_c_in_patch, y_c_in_patch)
    where the (x_c, y_c) coordinates are translated to the patch frame.
    Out-of-bounds pixels become NaN."""
    ny, nx = image.shape
    size = 2 * half_size + 1
    out = np.full((size, size), np.nan, dtype=float)
    x0 = int(round(x_c)) - half_size
    y0 = int(round(y_c)) - half_size
    for j in range(size):
        for i in range(size):
            xi = x0 + i
            yj = y0 + j
            if 0 <= xi < nx and 0 <= yj < ny:
                out[j, i] = image[yj, xi]
    # Sub-pixel offset of the true (x_c, y_c) inside the patch
    return out, x_c - x0, y_c - y0


def orbit_pixel_log_likelihood(x_t_ref: np.ndarray, v_t_ref: np.ndarray,
                                  t_ref_et: float,
                                  images: Sequence[np.ndarray],
                                  wcs_list: Sequence,
                                  image_ets: Sequence[float],
                                  *, sigma_psf: float = 1.5,
                                  half_size: int = 8,
                                  use_nbody: bool = False) -> float:
    """Total pixel log-likelihood of a candidate orbit across all images.

    The orbit is propagated to each image's epoch (kepler by default, or
    N-body if use_nbody=True), projected to pixel coordinates via the
    image WCS, and the surrounding patch is fit to a Gaussian PSF.
    """
    from ...dynamics.secular import kepler_step
    from ...data.constants import GM_SUN
    from ...data.ephemeris import body_state
    from .shift_stack_validation import _crop
    if use_nbody:
        from .nbody_chain_grow import nbody_step

    total_log_l = 0.0
    n_valid = 0
    for img, wcs, et in zip(images, wcs_list, image_ets):
        dt_s = float(et) - float(t_ref_et)
        if use_nbody:
            try:
                r_t, _ = nbody_step(x_t_ref, v_t_ref, t_ref_et, et)
            except Exception:
                continue
        else:
            r_t, _ = kepler_step(x_t_ref, v_t_ref, GM_SUN, dt_s)
        R_e = np.array(body_state("EARTH", float(et), "J2000", "SUN")[:3])
        geo = r_t - R_e
        rho = float(np.linalg.norm(geo))
        if rho < 1.0:
            continue
        ra_deg = math.degrees(math.atan2(geo[1], geo[0])) % 360.0
        dec_deg = math.degrees(math.asin(geo[2] / rho))
        try:
            x_pix, y_pix = wcs.world_to_pixel_values(ra_deg, dec_deg)
            x_pix = float(x_pix); y_pix = float(y_pix)
        except Exception:
            continue
        ny, nx = img.shape
        if not (half_size <= x_pix < nx - half_size
                and half_size <= y_pix < ny - half_size):
            continue
        patch, xc_p, yc_p = _crop_patch(img, x_pix, y_pix, half_size)
        log_l = patch_log_likelihood(patch, xc_p, yc_p, sigma_psf=sigma_psf)
        total_log_l += log_l
        n_valid += 1
    if n_valid == 0:
        return -1e9
    return total_log_l


def refine_orbit_against_pixels(x_init: np.ndarray, v_init: np.ndarray,
                                  t_ref_et: float,
                                  images: Sequence[np.ndarray],
                                  wcs_list: Sequence,
                                  image_ets: Sequence[float],
                                  *, sigma_psf: float = 1.5,
                                  half_size: int = 8,
                                  max_iter: int = 200,
                                  use_nbody: bool = False,
                                  perturbation_scale_pos: float = 1e6,
                                  perturbation_scale_vel: float = 0.5,
                                  ) -> PixelRefinementResult:
    """Nelder-Mead refine (x, v) to MAXIMIZE pixel log-likelihood.

    `perturbation_scale_*` set the initial simplex size: 1e6 km for
    position (~6e-5 AU, fits within typical IOD covariance) and 0.5 km/s
    for velocity (~1% of TNO orbital speed).

    Returns the refined state plus a PixelRefinementResult with the
    log-L improvement so callers can decide whether the refinement was
    worth keeping.
    """
    from scipy.optimize import minimize

    x6_init = np.concatenate([x_init, v_init])

    def _neg_log_l(x6):
        x = x6[:3]
        v = x6[3:]
        return -orbit_pixel_log_likelihood(
            x, v, t_ref_et, images, wcs_list, image_ets,
            sigma_psf=sigma_psf, half_size=half_size, use_nbody=use_nbody)

    log_l_initial = -_neg_log_l(x6_init)

    # Build a Nelder-Mead initial simplex
    scales = np.array([perturbation_scale_pos] * 3
                        + [perturbation_scale_vel] * 3)
    simplex = np.vstack([x6_init,
                          x6_init + np.diag(scales)])
    try:
        res = minimize(_neg_log_l, x6_init, method="Nelder-Mead",
                         options={"maxiter": max_iter, "initial_simplex": simplex,
                                    "xatol": 1e3, "fatol": 0.1})
        log_l_refined = -res.fun
        return PixelRefinementResult(
            converged=res.success,
            x_refined=np.asarray(res.x[:3]),
            v_refined=np.asarray(res.x[3:]),
            log_l_initial=log_l_initial,
            log_l_refined=log_l_refined,
            log_l_improvement=log_l_refined - log_l_initial,
            n_iterations=int(res.nit),
            notes=str(res.message)[:80] if res.message else "",
        )
    except Exception as e:
        return PixelRefinementResult(
            converged=False,
            x_refined=x_init, v_refined=v_init,
            log_l_initial=log_l_initial,
            log_l_refined=log_l_initial,
            log_l_improvement=0.0,
            n_iterations=0,
            notes=f"exception: {str(e)[:80]}",
        )
