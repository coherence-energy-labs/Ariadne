"""Shift-and-stack synthetic tracking -- detect objects fainter than single-frame S/N.

For a hypothesised on-sky velocity (vra, vdec) in arcsec/hr, shift each frame
to "track" the candidate trajectory, then median-stack. Real point sources
moving at that rate co-add coherently; static field stars smear out; sources
moving at other rates average down.

Sensitivity gain: a stack of N frames at the SAME rate hypothesis lowers the
detection limit by sqrt(N) magnitudes (in S/N units). For N=20 frames, that's
~1.5 mag deeper. For a faint moving object below the single-image 5-sigma
threshold, this is the difference between "invisible" and "detectable."

Algorithm (per (vra, vdec) hypothesis):

  1. For each frame i with epoch t_i, compute the required shift
     (dx_i, dy_i) so the candidate trajectory lands at the same pixel as in
     the reference frame.
  2. Sub-pixel-shift each frame by (-dx_i, -dy_i).
  3. Median-combine the shifted frames into a deep "tracked" image.
  4. Run source extraction on the tracked image.

The hypothesis grid is the same (r, rdot) -> (vra, vdec) mapping HelioLinC
uses for tracklet linking. For each grid point, you get a stacked image; bright
peaks in that stacked image are candidates moving at exactly that rate.

This is the "third weapon" of moving-object discovery after single-frame
extraction + image-differencing. It catches the FAINTEST things.

Reference: Bernstein & Khushalani 2000 (synthetic tracking for KBO surveys);
Holman et al. 2018 (HelioLinC plus tracking).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .difference import _shift_image


@dataclass(frozen=True)
class StackResult:
    """Result of one shift-and-stack hypothesis run.

    Fields:
      stacked:           the median-combined tracked image (same shape as input).
      hypothesis:        dict {vra_arcsec_hr, vdec_arcsec_hr} of the hypothesis.
      n_frames_used:     how many frames were stacked.
      sigma_floor:       noise sigma in a quiet region of the stacked image.
      peak_sigma:        max pixel / sigma_floor (sanity check / S/N proxy).
    """
    stacked: np.ndarray
    hypothesis: dict
    n_frames_used: int
    sigma_floor: float
    peak_sigma: float


def shift_stack(images: list[np.ndarray],
                epochs_mjd: list[float],
                vra_arcsec_hr: float,
                vdec_arcsec_hr: float,
                *,
                pixel_scale_arcsec: float = 0.25,
                reference_index: int = 0,
                combine: str = "median") -> StackResult:
    """Shift each frame to track (vra, vdec) and combine into a single deep image.

    Args:
      images:              list of 2-D arrays, all SAME shape, same band, already
                            roughly aligned (e.g. via difference._estimate_shift_xc)
                            to a common pixel grid.
      epochs_mjd:          list of MJD per image (one entry per image).
      vra_arcsec_hr:       on-sky RA velocity to track (positive = east).
      vdec_arcsec_hr:      on-sky Dec velocity to track (positive = north).
      pixel_scale_arcsec:  arcsec/pixel scale.
      reference_index:     which frame defines (dx, dy) = 0.
      combine:             "median" (robust) or "mean" (deeper, less robust).

    Returns:
      StackResult with the stacked image + sigma + sanity stats.
    """
    if len(images) != len(epochs_mjd):
        raise ValueError(f"images ({len(images)}) and epochs_mjd ({len(epochs_mjd)}) "
                          "must have same length")
    if not images:
        raise ValueError("no images supplied")
    if len({im.shape for im in images}) != 1:
        raise ValueError("all images must have the same shape")

    t_ref = epochs_mjd[reference_index]
    shifted_frames = []
    for img, t in zip(images, epochs_mjd):
        dt_hr = (t - t_ref) * 24.0
        # arcsec offset of the candidate from the ref-frame position
        dra_as = vra_arcsec_hr * dt_hr
        ddec_as = vdec_arcsec_hr * dt_hr
        # to track the moving object, shift the image OPPOSITE the candidate motion
        dx_px = -dra_as / pixel_scale_arcsec
        dy_px = -ddec_as / pixel_scale_arcsec
        shifted_frames.append(_shift_image(img, dx_px, dy_px))

    stack_arr = np.stack(shifted_frames, axis=0)
    if combine == "mean":
        stacked = stack_arr.mean(axis=0)
    else:
        stacked = np.median(stack_arr, axis=0)

    # noise estimate: sigma_clipped std of pixel values
    flat = stacked.ravel()
    quiet = flat[flat < np.percentile(flat, 75)]
    sigma_floor = float(np.std(quiet)) if quiet.size > 1 else 1.0
    sigma_floor = max(sigma_floor, 1e-6)
    peak_sigma = float(np.max(stacked) - np.median(stacked)) / sigma_floor

    return StackResult(stacked=stacked,
                       hypothesis={"vra_arcsec_hr": vra_arcsec_hr,
                                    "vdec_arcsec_hr": vdec_arcsec_hr},
                       n_frames_used=len(images),
                       sigma_floor=sigma_floor,
                       peak_sigma=peak_sigma)


def hypothesis_grid_search(images: list[np.ndarray],
                            epochs_mjd: list[float],
                            *,
                            vra_grid_arcsec_hr: np.ndarray | None = None,
                            vdec_grid_arcsec_hr: np.ndarray | None = None,
                            pixel_scale_arcsec: float = 0.25,
                            detection_sigma: float = 6.0,
                            top_n_hypotheses: int = 10) -> list[StackResult]:
    """Sweep (vra, vdec) hypotheses; return the top-N stack results with bright peaks.

    For each hypothesis, run shift_stack + extract the peak S/N. Keep the
    top-N most promising hypotheses for downstream source extraction.

    Args:
      images, epochs_mjd:        same as shift_stack.
      vra_grid_arcsec_hr:        velocity hypothesis grid in RA; default
                                  np.linspace(-5, 5, 21) (TNO regime).
      vdec_grid_arcsec_hr:       same for Dec.
      pixel_scale_arcsec:        arcsec/pixel.
      detection_sigma:           keep only hypotheses with peak S/N > this.
      top_n_hypotheses:          cap the returned list at this many.

    Returns:
      List of StackResult, sorted by peak_sigma (highest first), filtered to
      peak_sigma > detection_sigma.
    """
    if vra_grid_arcsec_hr is None:
        vra_grid_arcsec_hr = np.linspace(-5, 5, 21)
    if vdec_grid_arcsec_hr is None:
        vdec_grid_arcsec_hr = np.linspace(-5, 5, 21)

    results = []
    for vra in vra_grid_arcsec_hr:
        for vdec in vdec_grid_arcsec_hr:
            r = shift_stack(images, epochs_mjd, float(vra), float(vdec),
                            pixel_scale_arcsec=pixel_scale_arcsec)
            if r.peak_sigma >= detection_sigma:
                results.append(r)
    results.sort(key=lambda x: x.peak_sigma, reverse=True)
    return results[:top_n_hypotheses]
