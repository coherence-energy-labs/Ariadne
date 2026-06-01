"""Shift-and-stack IOD validation against actual image pixels.

After an IOD strategy produces a candidate orbit (x, v at t_ref) we
currently accept or reject it purely on the centroid-residual RMS.
But the IMAGES contain more information than just centroid positions:
the actual pixel patterns where the object should appear vs where the
fit places it.

Shift-and-stack validation:

  1. Predict where the object should be in each input image given
     the candidate orbit (x, v) at t_ref.
  2. SHIFT each image by the predicted (delta_x_pix, delta_y_pix) so
     the predicted positions all coincide at the same pixel.
  3. COADD the shifted images.
  4. Look for an SNR boost at the coincident pixel relative to the
     single-image SNR.

If the orbit is RIGHT, all N detections stack on top of each other
and the SNR grows ~ sqrt(N) over a single image. If the orbit is
WRONG, the detections spread across pixels and the stacked SNR
stays at ~the single-image level (or worse).

This catches IOD strategies that converge to plausible-looking but
WRONG orbits -- the centroid residual may be small while the actual
pixels disagree.

Public API:

  predict_pixel_positions(x_t_ref, v_t_ref, t_ref, images_metadata)
        propagate orbit through each image's MJD + WCS -> (x, y) pix

  shift_stack(images, predicted_positions, ref_pos)
        align all images to a common reference pixel and coadd

  measure_stacked_snr(coadd, position, aperture_radius)
        aperture-photometry SNR at a known position

  validate_orbit_against_images(fit, chain, images, wcs_list, mjds)
        end-to-end: predict positions -> shift -> stack -> SNR check.
        Returns a `StackValidationResult` with the SNR boost and a
        bool accept/reject verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class StackValidationResult:
    """Result of one shift-and-stack validation."""
    n_images: int
    n_visible: int                   # images where the predicted pos is in-bounds
    snr_single: float                # median single-image SNR at the predicted pos
    snr_stacked: float               # SNR at the stacked pixel
    snr_boost: float                 # snr_stacked / snr_single  (~ sqrt(N) when good)
    accepted: bool
    notes: str = ""


def _propagate_kepler(r0: np.ndarray, v0: np.ndarray,
                       mu: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper around the Ariadne dynamics.kepler_step.

    Imported lazily because the test suite shouldn't drag in the
    dynamics module if it's not needed.
    """
    from ...dynamics.secular import kepler_step
    return kepler_step(r0, v0, mu, dt_s)


def predict_pixel_positions(x_t_ref: np.ndarray, v_t_ref: np.ndarray,
                              t_ref_et: float,
                              images_metadata: Sequence[dict]) -> list[tuple]:
    """Propagate the orbit through every image's epoch and project onto
    the image WCS. Returns [(x_pix, y_pix, in_bounds)] per image.

    `images_metadata` is a list of {"et": float, "wcs": astropy.wcs.WCS,
    "shape": (ny, nx)} dicts.
    """
    from ...data.ephemeris import body_state
    from ...data.constants import GM_SUN

    results = []
    for meta in images_metadata:
        et = float(meta["et"])
        dt_s = et - t_ref_et
        r_t, _ = _propagate_kepler(x_t_ref, v_t_ref, GM_SUN, dt_s)
        # Convert heliocentric to geocentric
        R_e = np.array(body_state("EARTH", et, "J2000", "SUN")[:3])
        geo = r_t - R_e
        rho = float(np.linalg.norm(geo))
        ra_deg = math.degrees(math.atan2(geo[1], geo[0])) % 360.0
        dec_deg = math.degrees(math.asin(geo[2] / rho))
        # Project to pixel
        wcs = meta["wcs"]
        try:
            x_pix, y_pix = wcs.world_to_pixel_values(ra_deg, dec_deg)
            x_pix = float(x_pix); y_pix = float(y_pix)
        except Exception:
            x_pix, y_pix = -1.0, -1.0
        ny, nx = meta["shape"]
        in_bounds = (5 <= x_pix < nx - 5) and (5 <= y_pix < ny - 5)
        results.append((x_pix, y_pix, in_bounds))
    return results


def _crop(image: np.ndarray, x_c: float, y_c: float,
            half_size: int = 12) -> np.ndarray:
    """Crop a `half_size`-radius patch around (x_c, y_c). Out-of-bounds
    pixels become NaN. Returns shape (2*half_size+1, 2*half_size+1)."""
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
    return out


def shift_stack(images: Sequence[np.ndarray],
                  predicted_positions: Sequence[tuple],
                  *, half_size: int = 12) -> np.ndarray:
    """Crop a patch around each image's predicted (x, y) and average them.

    `predicted_positions` is a list of (x_pix, y_pix, in_bounds) tuples
    (same shape as the output of `predict_pixel_positions`). Patches
    from out-of-bounds predictions are dropped.

    Returns a coadded (2*half_size+1, 2*half_size+1) numpy array. Missing
    pixels are nan-averaged so partial coverage is allowed.
    """
    patches = []
    for img, (x_pix, y_pix, ok) in zip(images, predicted_positions):
        if not ok:
            continue
        patches.append(_crop(img, x_pix, y_pix, half_size))
    if not patches:
        return np.zeros((2 * half_size + 1, 2 * half_size + 1))
    stack = np.stack(patches, axis=0)   # (N, h, w)
    return np.nanmean(stack, axis=0)


def measure_aperture_snr(patch: np.ndarray,
                           *, aperture_radius: int = 3,
                           annulus_inner: int | None = None,
                           annulus_outer: int | None = None) -> tuple[float, float]:
    """Return (signal, snr) for a circular aperture at the patch centre.

    Background is estimated from an ANNULUS that excludes the PSF wings
    (default inner = aperture_radius + 3, outer = annulus_inner + 6).
    Without the annulus, PSF wings inflate the background sigma and the
    SNR estimate is wrong by a large factor at high signal levels.

    Background sigma uses Median Absolute Deviation (MAD * 1.4826) which
    is robust against the wing flux that would otherwise leak in.
    """
    ny, nx = patch.shape
    cx, cy = nx // 2, ny // 2
    yy, xx = np.indices(patch.shape)
    r = np.hypot(xx - cx, yy - cy)
    inside = r <= aperture_radius

    if annulus_inner is None:
        annulus_inner = aperture_radius + 3
    if annulus_outer is None:
        annulus_outer = annulus_inner + 6
    bg_ring = (r >= annulus_inner) & (r <= annulus_outer)
    bg_pix = patch[bg_ring]
    bg_pix = bg_pix[np.isfinite(bg_pix)]
    if bg_pix.size < 8:
        # Annulus too small (small patch). Fall back to outside-aperture.
        bg_pix = patch[~inside]
        bg_pix = bg_pix[np.isfinite(bg_pix)]
    if bg_pix.size == 0:
        return (0.0, 0.0)
    bg_med = float(np.median(bg_pix))
    # Robust sigma via MAD (1.4826 factor turns MAD into sigma for Gaussians)
    bg_mad = float(np.median(np.abs(bg_pix - bg_med)))
    bg_std = max(bg_mad * 1.4826, 1e-6)

    aperture_pix = patch[inside]
    aperture_pix = aperture_pix[np.isfinite(aperture_pix)]
    if aperture_pix.size == 0:
        return (0.0, 0.0)
    signal = float(np.sum(aperture_pix - bg_med))
    noise = bg_std * math.sqrt(aperture_pix.size)
    snr = signal / noise if noise > 0 else 0.0
    return (signal, snr)


def validate_orbit_against_images(x_t_ref: np.ndarray, v_t_ref: np.ndarray,
                                    t_ref_et: float,
                                    images: Sequence[np.ndarray],
                                    wcs_list: Sequence,
                                    image_ets: Sequence[float],
                                    *,
                                    aperture_radius: int = 3,
                                    half_size: int = 12,
                                    min_snr_boost: float = 1.5,
                                    ) -> StackValidationResult:
    """End-to-end shift-and-stack validation.

    For each image in `images`, propagate (x_t_ref, v_t_ref) to its
    epoch, project to pixel coordinates, then crop+stack. Compare the
    SNR at the stacked patch's centre to the median per-image SNR at
    each image's predicted position.

    Accepts the orbit iff `snr_boost >= min_snr_boost`. For a true
    orbit with N visible epochs we expect boost ~ sqrt(N); we accept
    anything above 1.5 (a 50% improvement) as evidence the orbit is
    consistent with the actual pixels.
    """
    images_metadata = [
        {"et": e, "wcs": w, "shape": img.shape}
        for e, w, img in zip(image_ets, wcs_list, images)
    ]
    preds = predict_pixel_positions(x_t_ref, v_t_ref, t_ref_et,
                                      images_metadata)
    n_visible = sum(1 for _, _, ok in preds if ok)
    if n_visible == 0:
        return StackValidationResult(
            n_images=len(images), n_visible=0,
            snr_single=0.0, snr_stacked=0.0, snr_boost=0.0,
            accepted=False, notes="no predictions in-bounds")

    # Per-image SNR at predicted position
    snrs = []
    for img, (x, y, ok) in zip(images, preds):
        if not ok:
            continue
        patch = _crop(img, x, y, half_size)
        _, snr = measure_aperture_snr(patch, aperture_radius=aperture_radius)
        snrs.append(snr)
    snr_single = float(np.median(snrs)) if snrs else 0.0

    # Stacked SNR
    coadd = shift_stack(images, preds, half_size=half_size)
    _, snr_stacked = measure_aperture_snr(coadd, aperture_radius=aperture_radius)

    snr_boost = snr_stacked / snr_single if snr_single > 0 else 0.0
    accepted = snr_boost >= min_snr_boost
    return StackValidationResult(
        n_images=len(images), n_visible=n_visible,
        snr_single=snr_single, snr_stacked=snr_stacked,
        snr_boost=snr_boost, accepted=accepted,
        notes=(f"{n_visible}/{len(images)} visible; "
                f"snr {snr_single:.1f}->{snr_stacked:.1f}"))
