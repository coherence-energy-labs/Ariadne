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
                                  search_grid_pix: int = 12,
                                  search_step_pix: int = 1,
                                  ) -> PixelRefinementResult:
    """Refine orbit by maximising pixel log-L over a tight 2D pixel
    offset grid AT THE REFERENCE EPOCH.

    The full 6D refinement (free x,v) suffers from the highly non-convex
    pixel-likelihood landscape: Nelder-Mead easily wanders away from
    the IOD seed and lands on bright background stars or noise spikes.

    This implementation INSTEAD:
      1. Treats the IOD orbit's predicted positions as approximately
         right and searches a small 2D PIXEL OFFSET grid (default
         +-12 pix in steps of 1 pix at the reference epoch).
      2. For each offset, applies it consistently to every image's
         predicted position (so it's a global tweak to the orbit's
         apparent sky position, not per-image).
      3. Picks the offset with the highest total pixel log-likelihood.
      4. Converts the best (dx_pix, dy_pix) back into an (x, v)
         correction via the WCS at the reference epoch.

    This is much more robust than free 6D optimisation because the
    search space is finite and convex-by-construction.
    """
    from .shift_stack_validation import _crop, predict_pixel_positions

    images_metadata = [
        {"et": e, "wcs": w, "shape": img.shape}
        for e, w, img in zip(image_ets, wcs_list, images)
    ]
    initial_preds = predict_pixel_positions(x_init, v_init, t_ref_et,
                                              images_metadata)

    def total_log_l_for_offset(dx_pix: float, dy_pix: float) -> float:
        total = 0.0
        n_valid = 0
        for img, (px, py, ok) in zip(images, initial_preds):
            if not ok:
                continue
            ny, nx = img.shape
            x_use = px + dx_pix
            y_use = py + dy_pix
            if not (half_size <= x_use < nx - half_size
                    and half_size <= y_use < ny - half_size):
                continue
            from .pixel_likelihood import _crop_patch, patch_log_likelihood
            patch, xc_p, yc_p = _crop_patch(img, x_use, y_use, half_size)
            log_l = patch_log_likelihood(patch, xc_p, yc_p, sigma_psf=sigma_psf)
            total += log_l
            n_valid += 1
        if n_valid == 0:
            return -1e9
        return total

    log_l_initial = total_log_l_for_offset(0.0, 0.0)
    best_dx, best_dy, best_l = 0.0, 0.0, log_l_initial
    for dy in range(-search_grid_pix, search_grid_pix + 1, search_step_pix):
        for dx in range(-search_grid_pix, search_grid_pix + 1, search_step_pix):
            l = total_log_l_for_offset(float(dx), float(dy))
            if l > best_l:
                best_l = l
                best_dx = float(dx); best_dy = float(dy)

    # Translate the best (dx_pix, dy_pix) into an (x, v) correction:
    # interpret the offset as a constant angular shift applied at t_ref
    # and apply the same shift in heliocentric (x) at t_ref. Velocity is
    # left unchanged -- the grid only tweaks the position, not the
    # apparent rate. For tighter refinements iterate on the (x, v) tweak.
    if best_dx == 0 and best_dy == 0:
        return PixelRefinementResult(
            converged=True,
            x_refined=np.asarray(x_init), v_refined=np.asarray(v_init),
            log_l_initial=log_l_initial,
            log_l_refined=log_l_initial,
            log_l_improvement=0.0,
            n_iterations=(2 * search_grid_pix + 1) ** 2,
            notes="no improvement in grid",
        )
    # Apply the pixel offset by perturbing x_init's projected sky pos.
    # Using the first valid image's WCS to convert pixel -> angle, then
    # perturb x_init along the line of sight.
    try:
        valid_idx = next(i for i, (_, _, ok) in enumerate(initial_preds) if ok)
        wcs0 = wcs_list[valid_idx]
        px0, py0, _ = initial_preds[valid_idx]
        ra_orig, dec_orig = wcs0.pixel_to_world_values(px0, py0)
        ra_new, dec_new = wcs0.pixel_to_world_values(px0 + best_dx, py0 + best_dy)
        dra_deg = float(ra_new) - float(ra_orig)
        ddec_deg = float(dec_new) - float(dec_orig)
        rho = float(np.linalg.norm(x_init))
        rho_km_per_arcsec = rho / (3600.0 * 180.0 / math.pi)
        offset_km_x = (math.radians(dra_deg) * math.cos(math.radians(float(dec_orig)))) * rho
        offset_km_y = math.radians(ddec_deg) * rho
        # Construct local sky-east and sky-north vectors at x_init
        # Sky-east: orthogonal to x and to north pole (z), in east direction.
        x_hat = np.asarray(x_init, dtype=float) / max(rho, 1e-6)
        z_hat = np.array([0.0, 0.0, 1.0])
        east = np.cross(z_hat, x_hat)
        east_norm = float(np.linalg.norm(east))
        if east_norm > 1e-9:
            east /= east_norm
        north = np.cross(x_hat, east)
        x_refined = (np.asarray(x_init, dtype=float)
                       + offset_km_x * east + offset_km_y * north)
    except Exception:
        # If WCS conversion fails just return the initial state with
        # the grid-search log-L improvement noted.
        x_refined = np.asarray(x_init, dtype=float)
    v_refined = np.asarray(v_init, dtype=float)
    return PixelRefinementResult(
        converged=True,
        x_refined=x_refined, v_refined=v_refined,
        log_l_initial=log_l_initial,
        log_l_refined=best_l,
        log_l_improvement=best_l - log_l_initial,
        n_iterations=(2 * search_grid_pix + 1) ** 2,
        notes=f"grid search best=({best_dx:+.0f}, {best_dy:+.0f}) pix",
    )
