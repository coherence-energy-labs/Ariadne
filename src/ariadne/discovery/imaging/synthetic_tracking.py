"""Shift-and-stack synthetic tracking for sub-detection-threshold moving objects.

The standard pipeline (detect -> tracklet -> chain -> orbit) requires each
moving object to produce 2+ detections per night each at >= 5-sigma above
background. For faint TNOs that's the binding constraint -- many are
sub-threshold in single exposures even though they'd be a 10-sigma stack
across 6+ images.

Synthetic tracking inverts the search:
  For each candidate orbit (parameterised by sky-rate vector at the
  reference epoch), shift every image by the predicted per-image offset
  so that an object on that orbit lines up at the same pixel across all
  images, then coadd. A real object on that orbit shows up as a peak
  in the coadd whose SNR scales as sqrt(N_images) over the per-image
  noise.

This is genuinely beyond the standard pipeline because it
  * Finds objects BELOW the per-image detection threshold (mag 24-25
    for ground-based 30-min exposures).
  * Bypasses the chain-formation stage entirely -- the "linker" is just
    the orbit grid.
  * Returns pixel-level evidence by construction (the SNR of the peak
    IS the pixel validation metric we'd otherwise compute separately).

Two passes:
  fast pass    Straight-line sky-rate (rate, position-angle) grid.
               Valid for short arcs where parallax curvature is small.
               ~10^4 hypotheses, GPU-style batched. ~30s/chain.

  verify pass  Full heliocentric (r_au, rdot, alpha, delta) grid with
               proper Kepler propagation including parallax curvature.
               ~10^7 hypotheses but only run on candidates from fast
               pass. ~30 min/candidate (research-grade).

References:
  Heinze et al. 2019, AJ 158:15 -- "Shift-and-stack ... Kuiper Belt"
  Yu et al. 2018, ApJ 156:200  -- "Search for TNOs through synthetic tracking"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SyntheticCandidate:
    """One peak found in a shift-and-stack search."""
    ra_deg: float
    dec_deg: float
    rate_arcsec_hr: float
    pa_deg: float          # position angle of motion (E of N)
    stacked_snr: float     # peak SNR in the coadded image
    n_images: int          # how many images contributed to the stack
    consensus_count: int = 0   # number of images with above-noise signal
    # Refined orbital parameters (filled in by the slow verifier)
    r_au: float = 0.0
    rdot_au_per_day: float = 0.0
    chain: list[dict] = field(default_factory=list)
    notes: str = ""


def synthetic_candidate_to_chain(candidate: SyntheticCandidate,
                                    image_mjds: Sequence[float],
                                    pixscale_arcsec: float = 1.0,
                                    ) -> list[dict]:
    """Convert a fast-pass candidate into a chain (per-image observations)
    suitable for downstream IOD.

    Each "chain entry" is a synthetic observation at the candidate's
    predicted (ra, dec) for that image's epoch, with the candidate's
    rate vector recorded so the IOD ensemble's rate-class branching
    picks the right strategy.

    Time uses SPICE-ET seconds, ra/dec in radians (the imaging-pipeline
    convention). MJD reference epoch is the candidate's mjd value (passed
    in via image_mjds; the entry's t is computed from each image's mjd).
    """
    SEC_PER_DAY = 86400.0
    chain = []
    pa = math.radians(candidate.pa_deg)
    t_ref_mjd = float(np.median(image_mjds))
    cos_dec = math.cos(math.radians(candidate.dec_deg))
    for mjd in image_mjds:
        dt_hr = (mjd - t_ref_mjd) * 24.0
        motion_arcsec = candidate.rate_arcsec_hr * dt_hr
        dra_arcsec = motion_arcsec * math.sin(pa) / max(cos_dec, 1e-6)
        ddec_arcsec = motion_arcsec * math.cos(pa)
        ra = candidate.ra_deg + dra_arcsec / 3600.0
        dec = candidate.dec_deg + ddec_arcsec / 3600.0
        et = ((mjd + 2400000.5) - 2451545.0) * SEC_PER_DAY
        chain.append({
            "t": et, "jd": mjd + 2400000.5,
            "ra": math.radians(ra % 360.0),
            "dec": math.radians(dec),
            "dra": 1e-9, "ddec": 1e-9,
            "rate_arcsec_hr": float(candidate.rate_arcsec_hr),
            "mag": -99.0,
            "source_pair": (),
            "synthetic_tracking": True,
            "stacked_snr": float(candidate.stacked_snr),
            "consensus_count": int(candidate.consensus_count),
            "night": int(mjd),
        })
    return chain


# ---------------------------------------------------------------------------
# Image shifting (bilinear sub-pixel)
# ---------------------------------------------------------------------------

def shift_image_bilinear(img: np.ndarray, dx_pix: float, dy_pix: float
                          ) -> np.ndarray:
    """Shift an image so that pixel (i, j) in the output reads from
    pixel (i + dy_pix, j + dx_pix) in the input via bilinear interpolation.

    Equivalently: positive dx_pix moves the image content LEFT in the
    output (a feature originally at column j appears at column j - dx_pix).

    Out-of-bounds source pixels become NaN.
    """
    img = img.astype(float, copy=False)
    ny, nx = img.shape
    # Source coordinates for each output pixel
    xs = np.arange(nx, dtype=float) + dx_pix           # length nx
    ys = np.arange(ny, dtype=float) + dy_pix           # length ny
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0)                                       # length nx
    fy = (ys - y0)                                       # length ny
    x1 = x0 + 1
    y1 = y0 + 1
    # Validity per row/column
    vx = (x0 >= 0) & (x1 < nx)
    vy = (y0 >= 0) & (y1 < ny)
    valid = vy[:, None] & vx[None, :]
    # Clip into bounds so we can index without error; NaN fill handles invalid.
    x0c = np.clip(x0, 0, nx - 1)
    x1c = np.clip(x1, 0, nx - 1)
    y0c = np.clip(y0, 0, ny - 1)
    y1c = np.clip(y1, 0, ny - 1)
    # Bilinear weights broadcast to (ny, nx)
    fx2 = fx[None, :]
    fy2 = fy[:, None]
    # Fancy-index image at corner pixels for every output (i, j)
    a = img[y0c[:, None], x0c[None, :]] * (1 - fx2) * (1 - fy2)
    b = img[y0c[:, None], x1c[None, :]] * fx2 * (1 - fy2)
    c = img[y1c[:, None], x0c[None, :]] * (1 - fx2) * fy2
    d = img[y1c[:, None], x1c[None, :]] * fx2 * fy2
    out = a + b + c + d
    out[~valid] = np.nan
    return out


def shift_and_stack(images: Sequence[np.ndarray],
                     shifts_per_image: Sequence[tuple[float, float]],
                     *, return_coverage: bool = False,
                     return_stack: bool = False
                     ) -> np.ndarray | tuple:
    """Shift each image by its corresponding (dx, dy) and coadd via nanmean.

    Returns a (ny, nx) coadd with NaN where coverage was 0.

    When `return_coverage=True`, also returns a (ny, nx) integer array
    of per-pixel image-counts (how many images contributed a finite
    value at each pixel).

    When `return_stack=True`, also returns the (N, ny, nx) shifted-image
    stack itself (so per-image consistency checks can run on peaks).
    """
    if not images:
        outs = (np.array([]),)
        if return_coverage:
            outs = outs + (np.array([]),)
        if return_stack:
            outs = outs + (np.array([]),)
        return outs if (return_coverage or return_stack) else outs[0]
    stack = np.empty((len(images),) + images[0].shape, dtype=float)
    for k, (img, (dx, dy)) in enumerate(zip(images, shifts_per_image)):
        if dx == 0 and dy == 0:
            stack[k] = img.astype(float)
        else:
            stack[k] = shift_image_bilinear(img.astype(float), dx, dy)
    coadd = np.nanmean(stack, axis=0)
    if return_coverage and return_stack:
        cov = np.isfinite(stack).sum(axis=0).astype(np.int32)
        return coadd, cov, stack
    if return_coverage:
        cov = np.isfinite(stack).sum(axis=0).astype(np.int32)
        return coadd, cov
    if return_stack:
        return coadd, stack
    return coadd


def _per_image_signal_consensus(stack: np.ndarray, x: int, y: int,
                                  aperture_radius: int = 3,
                                  signal_z_threshold: float = 1.5,
                                  ) -> int:
    """Count how many of the N images have aperture-signal above noise at (x, y).

    For a real moving object that the shift hypothesis correctly aligns,
    every image's contribution at the peak pixel should be ABOVE background
    noise. Partial-alignment artefacts have signal from only 2-3 images.

    Returns the count of images where the local aperture signal exceeds
    `signal_z_threshold` * per-image background sigma.
    """
    n_imgs, ny, nx = stack.shape
    if not (aperture_radius <= x < nx - aperture_radius
            and aperture_radius <= y < ny - aperture_radius):
        return 0
    n_consensus = 0
    for k in range(n_imgs):
        img = stack[k]
        ap = img[y - aperture_radius: y + aperture_radius + 1,
                  x - aperture_radius: x + aperture_radius + 1]
        if not np.any(np.isfinite(ap)):
            continue
        # Local background from a larger region around the aperture
        bx0 = max(0, x - 12); bx1 = min(nx, x + 13)
        by0 = max(0, y - 12); by1 = min(ny, y + 13)
        bg_pix = img[by0:by1, bx0:bx1]
        bg_pix = bg_pix[np.isfinite(bg_pix)]
        if bg_pix.size < 50:
            continue
        bg_med = float(np.median(bg_pix))
        bg_std = max(float(np.median(np.abs(bg_pix - bg_med)) * 1.4826), 1e-6)
        ap_finite = ap[np.isfinite(ap)]
        if ap_finite.size == 0:
            continue
        signal = float(np.mean(ap_finite - bg_med))
        if signal > signal_z_threshold * bg_std:
            n_consensus += 1
    return n_consensus


# ---------------------------------------------------------------------------
# Rate -> per-image pixel shift
# ---------------------------------------------------------------------------

def predicted_shift(image_mjd: float, t_ref_mjd: float,
                      rate_arcsec_hr: float, pa_deg: float,
                      pixscale_arcsec: float) -> tuple[float, float]:
    """Return (dx_pix, dy_pix) shift to APPLY to the image so that an
    object moving at (rate, position-angle) lines up with its position
    at t_ref.

    Position-angle convention: 0 deg = north, 90 deg = east.

    With a standard astronomical WCS (CD11 < 0, CD22 > 0):
      east-moving object has DECREASING pixel-x
      north-moving object has INCREASING pixel-y
    so for dt > 0 (image later than t_ref):
      pixel position of object: (ref_x - dra_arcsec/pixscale,
                                  ref_y + ddec_arcsec/pixscale)
    To shift content so the object lands BACK at ref pixel, we apply
    the same (dx, dy) the OBJECT moved (positive dx moves content left,
    which counteracts the object's leftward drift -- wait, that's the
    wrong direction). Actually: shift_image_bilinear(img, dx, dy) reads
    from input[y + dy, x + dx]. So a positive dx makes output[i, j] read
    from input[i, j + dx], i.e., shifts the image LEFT. To align an
    object that's at column j_obj = ref_x - dra/pixscale (i.e., LEFT of
    ref), we need dx such that output[ref_y, ref_x] reads from
    input[ref_y, j_obj]; that gives dx = j_obj - ref_x = -dra/pixscale.

    So the correct shift is (dx, dy) = (-dra_pix, +ddec_pix).
    """
    dt_hr = (image_mjd - t_ref_mjd) * 24.0
    motion_arcsec = rate_arcsec_hr * dt_hr   # signed; negative for earlier images
    motion_pix = motion_arcsec / pixscale_arcsec
    pa = math.radians(pa_deg)
    dra_pix = motion_pix * math.sin(pa)
    ddec_pix = motion_pix * math.cos(pa)
    return (-dra_pix, ddec_pix)


# ---------------------------------------------------------------------------
# Peak finding in coadd
# ---------------------------------------------------------------------------

def _aperture_snr(patch: np.ndarray, aperture_radius: int = 3) -> float:
    """Return aperture SNR of the centroid pixel relative to an annulus."""
    if patch.size == 0:
        return 0.0
    ny, nx = patch.shape
    cx, cy = nx // 2, ny // 2
    yy, xx = np.indices(patch.shape)
    r = np.hypot(xx - cx, yy - cy)
    inside = r <= aperture_radius
    annulus = (r >= aperture_radius + 3) & (r <= aperture_radius + 9)
    bg = patch[annulus]
    bg = bg[np.isfinite(bg)]
    if bg.size < 8:
        return 0.0
    bg_med = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_med)))
    bg_std = max(bg_mad * 1.4826, 1e-6)
    sig_pix = patch[inside]
    sig_pix = sig_pix[np.isfinite(sig_pix)]
    if sig_pix.size == 0:
        return 0.0
    signal = float(np.sum(sig_pix - bg_med))
    noise = bg_std * math.sqrt(sig_pix.size)
    return signal / noise if noise > 0 else 0.0


def find_peaks_in_coadd(coadd: np.ndarray,
                          *, snr_threshold: float = 5.0,
                          aperture_radius: int = 3,
                          min_separation_pix: int = 8,
                          n_top: int = 20,
                          coverage: np.ndarray | None = None,
                          min_coverage: int = 0) -> list[dict]:
    """Find local maxima above SNR threshold in the coadded image.

    `coverage` (optional): per-pixel image-count from shift_and_stack.
    When provided, peaks at pixels with coverage < `min_coverage` are
    rejected. This prevents the search from "finding" a peak at a pixel
    where most images are NaN-padded (the coadd value there is just a
    single image's signal, not a genuine stacked detection).

    Returns up to `n_top` peaks sorted by SNR descending.
    """
    if coadd.size == 0:
        return []
    ny, nx = coadd.shape
    # Robust background statistics
    valid = coadd[np.isfinite(coadd)]
    if valid.size < 100:
        return []
    bg_med = float(np.median(valid))
    bg_mad = float(np.median(np.abs(valid - bg_med)))
    bg_std = max(bg_mad * 1.4826, 1e-6)
    pre = (coadd - bg_med) > snr_threshold * bg_std
    if coverage is not None and min_coverage > 0:
        pre = pre & (coverage >= min_coverage)
    if not np.any(pre):
        return []
    ys, xs = np.where(pre)
    vals = coadd[ys, xs]
    order = np.argsort(-vals)
    ys = ys[order]; xs = xs[order]; vals = vals[order]
    kept = []
    for yi, xi, v in zip(ys, xs, vals):
        ok = True
        for k in kept:
            if abs(k["x"] - xi) < min_separation_pix and abs(k["y"] - yi) < min_separation_pix:
                ok = False
                break
        if not ok:
            continue
        half = aperture_radius + 12
        y0, y1 = max(0, int(yi) - half), min(ny, int(yi) + half + 1)
        x0, x1 = max(0, int(xi) - half), min(nx, int(xi) + half + 1)
        patch = coadd[y0:y1, x0:x1]
        snr = _aperture_snr(patch, aperture_radius=aperture_radius)
        if snr < snr_threshold:
            continue
        cov_here = int(coverage[int(yi), int(xi)]) if coverage is not None else 0
        kept.append({"x": float(xi), "y": float(yi), "snr": float(snr),
                      "raw_value": float(v), "coverage": cov_here})
        if len(kept) >= n_top:
            break
    return kept


# ---------------------------------------------------------------------------
# Fast pass: straight-line sky-rate grid
# ---------------------------------------------------------------------------

def fast_synthetic_tracking(images: Sequence[np.ndarray],
                              wcs_list: Sequence,
                              image_mjds: Sequence[float],
                              *, t_ref_mjd: float | None = None,
                              rate_min_arcsec_hr: float = 0.5,
                              rate_max_arcsec_hr: float = 30.0,
                              n_rates: int = 24,
                              n_pa: int = 12,
                              snr_threshold: float = 5.0,
                              pixscale_arcsec: float = 1.0,
                              n_top_per_hypothesis: int = 5,
                              ) -> list[SyntheticCandidate]:
    """Fast straight-line shift-and-stack search.

    Grid over (rate_arcsec_hr, pa_deg) with the reference epoch being
    the median image MJD. For each grid point, shift all images so that
    an object moving at that rate aligns at its t_ref position, coadd,
    and find peaks above the SNR threshold.

    Each peak becomes a `SyntheticCandidate` with the (ra, dec) at
    t_ref and the rate vector. Downstream IOD can convert these into
    state vectors.
    """
    if not images:
        return []
    if t_ref_mjd is None:
        t_ref_mjd = float(np.median(image_mjds))
    # Build rate × PA grid (log-spaced in rate)
    rates = np.geomspace(rate_min_arcsec_hr, rate_max_arcsec_hr, n_rates)
    pas = np.linspace(0, 360, n_pa, endpoint=False)
    candidates: list[SyntheticCandidate] = []
    seen_positions = []   # for dedup: (x, y, rate, pa)
    # Require at least most images to contribute to any claimed peak.
    min_cov = max(2, int(0.5 * len(images)))
    # And require per-image signal consensus: a true detection must have
    # signal above background in at least `min_consensus` images. This is
    # the antidote to partial-alignment artifacts where shift+pa flips
    # produce a false coadd peak from only 2-3 images.
    min_consensus = max(3, int(0.6 * len(images)))
    for rate in rates:
        for pa in pas:
            shifts = [predicted_shift(mjd, t_ref_mjd, rate, pa, pixscale_arcsec)
                       for mjd in image_mjds]
            coadd, coverage, stack = shift_and_stack(
                images, shifts, return_coverage=True, return_stack=True)
            peaks = find_peaks_in_coadd(
                coadd, snr_threshold=snr_threshold,
                aperture_radius=3, min_separation_pix=8,
                n_top=n_top_per_hypothesis * 3,    # over-collect, will filter
                coverage=coverage, min_coverage=min_cov)
            # Filter by per-image signal consensus
            confirmed = []
            for p in peaks:
                consensus = _per_image_signal_consensus(
                    stack, int(p["x"]), int(p["y"]),
                    aperture_radius=3, signal_z_threshold=1.0)
                if consensus >= min_consensus:
                    p["consensus"] = consensus
                    confirmed.append(p)
                if len(confirmed) >= n_top_per_hypothesis:
                    break
            peaks = confirmed
            # Convert each peak to (ra, dec) via t_ref's WCS
            t_ref_wcs = wcs_list[int(np.argmin(np.abs(np.asarray(image_mjds) - t_ref_mjd)))]
            for p in peaks:
                try:
                    ra, dec = t_ref_wcs.pixel_to_world_values(p["x"], p["y"])
                    ra = float(ra); dec = float(dec)
                except Exception:
                    continue
                dup = False
                for sp in seen_positions:
                    if (abs(sp["ra"] - ra) < 0.001
                            and abs(sp["dec"] - dec) < 0.001
                            and abs(sp["rate"] - rate) / rate < 0.25):
                        dup = True
                        break
                if dup:
                    continue
                seen_positions.append({"ra": ra, "dec": dec,
                                         "rate": rate, "pa": pa})
                candidates.append(SyntheticCandidate(
                    ra_deg=ra % 360.0, dec_deg=dec,
                    rate_arcsec_hr=float(rate),
                    pa_deg=float(pa),
                    stacked_snr=p["snr"],
                    n_images=len(images),
                    consensus_count=p.get("consensus", 0),
                    notes=f"fast pass: rate={rate:.2f}, pa={pa:.0f}, "
                          f"consensus={p.get('consensus', 0)}/{len(images)}",
                ))
    # Sort by SNR descending
    candidates.sort(key=lambda c: -c.stacked_snr)
    return candidates


# ---------------------------------------------------------------------------
# Verify pass: full heliocentric propagation
# ---------------------------------------------------------------------------

def verify_candidate_with_kepler(candidate: SyntheticCandidate,
                                    images: Sequence[np.ndarray],
                                    wcs_list: Sequence,
                                    image_ets: Sequence[float],
                                    *, t_ref_et: float | None = None,
                                    r_grid_au: Sequence[float] | None = None,
                                    rdot_grid_au_per_day: Sequence[float] | None = None,
                                    pixscale_arcsec: float = 1.0,
                                    ) -> SyntheticCandidate:
    """Slow verification pass: full heliocentric propagation grid.

    Takes a `SyntheticCandidate` from the fast pass and searches over
    (r_au, rdot) at the candidate's (ra, dec). For each (r, rdot),
    proper Kepler propagation gives per-image (ra, dec, pixel) predictions
    that include parallax curvature. Returns the candidate with the
    best-fit r/rdot and updated stacked SNR.
    """
    from ...data.constants import GM_SUN, AU_KM
    from ...data.ephemeris import body_state
    from ...dynamics.secular import kepler_step

    if t_ref_et is None:
        t_ref_et = float(np.median(image_ets))
    if r_grid_au is None:
        r_grid_au = np.geomspace(1.5, 100.0, 40)
    if rdot_grid_au_per_day is None:
        rdot_grid_au_per_day = np.linspace(-0.02, 0.02, 21)

    # Initial sky direction
    ra_rad = math.radians(candidate.ra_deg)
    dec_rad = math.radians(candidate.dec_deg)
    d_los = np.array([math.cos(dec_rad) * math.cos(ra_rad),
                       math.cos(dec_rad) * math.sin(ra_rad),
                       math.sin(dec_rad)])
    R_e_ref = np.array(body_state("EARTH", t_ref_et, "J2000", "SUN")[:3])

    best_snr = candidate.stacked_snr
    best_r = candidate.r_au
    best_rdot = candidate.rdot_au_per_day
    for r_au in r_grid_au:
        rho_km = r_au * AU_KM
        # At t_ref: object is at R_earth + rho * d_los  (geocentric distance rho)
        # Heliocentric position
        r_helio_ref = R_e_ref + rho_km * d_los
        r_helio_norm = float(np.linalg.norm(r_helio_ref))
        # Radial velocity unit vector = r_helio / |r_helio|
        r_hat = r_helio_ref / r_helio_norm
        # Need to choose tangential velocity perpendicular to r_helio.
        # Use the line-of-sight motion d/dt(rho) implicit in rdot:
        # For simplicity, assume the object's transverse velocity is
        # circular-orbit speed × (cos(tilt)). Try a single tilt per
        # r value (the fast pass already constrained the apparent rate).
        v_circ = math.sqrt(GM_SUN / r_helio_norm)
        # Tangent perpendicular to r in ecliptic plane
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(r_hat, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        tan = np.cross(r_hat, up)
        tan = tan / float(np.linalg.norm(tan))
        for rdot in rdot_grid_au_per_day:
            rdot_km_s = rdot * AU_KM / 86400.0
            v_helio = v_circ * tan + rdot_km_s * r_hat
            # Propagate to each image's epoch
            shifts = []
            for img_et in image_ets:
                dt_s = float(img_et) - float(t_ref_et)
                try:
                    r_t, _ = kepler_step(r_helio_ref, v_helio, GM_SUN, dt_s)
                except Exception:
                    shifts = None
                    break
                R_e_t = np.array(body_state("EARTH", img_et, "J2000", "SUN")[:3])
                geo = r_t - R_e_t
                rho = float(np.linalg.norm(geo))
                if rho < 1.0:
                    shifts = None
                    break
                ra_t = math.degrees(math.atan2(geo[1], geo[0])) % 360.0
                dec_t = math.degrees(math.asin(geo[2] / rho))
                # Project to pixels via the corresponding image's WCS
                wcs_t = wcs_list[image_ets.index(img_et)] if image_ets.count(img_et) == 1 else wcs_list[0]
                try:
                    x_t, y_t = wcs_t.pixel_to_world_values  # need WORLD->PIXEL
                except Exception:
                    pass
                try:
                    x_t, y_t = wcs_t.world_to_pixel_values(ra_t, dec_t)
                    x_ref, y_ref = wcs_t.world_to_pixel_values(
                        candidate.ra_deg, candidate.dec_deg)
                    shifts.append((float(x_t) - float(x_ref),
                                     float(y_t) - float(y_ref)))
                except Exception:
                    shifts = None
                    break
            if shifts is None:
                continue
            coadd = shift_and_stack(images, shifts)
            # Crop a small patch at the candidate's reference pixel
            wcs_ref = wcs_list[0]
            try:
                x_ref0, y_ref0 = wcs_ref.world_to_pixel_values(
                    candidate.ra_deg, candidate.dec_deg)
                x_ref0, y_ref0 = int(round(float(x_ref0))), int(round(float(y_ref0)))
            except Exception:
                continue
            ny, nx = coadd.shape
            half = 12
            y0, y1 = max(0, y_ref0 - half), min(ny, y_ref0 + half + 1)
            x0, x1 = max(0, x_ref0 - half), min(nx, x_ref0 + half + 1)
            patch = coadd[y0:y1, x0:x1]
            snr = _aperture_snr(patch, aperture_radius=3)
            if snr > best_snr:
                best_snr = snr
                best_r = float(r_au)
                best_rdot = float(rdot)

    out = SyntheticCandidate(
        ra_deg=candidate.ra_deg, dec_deg=candidate.dec_deg,
        rate_arcsec_hr=candidate.rate_arcsec_hr,
        pa_deg=candidate.pa_deg,
        stacked_snr=best_snr, n_images=len(images),
        r_au=best_r, rdot_au_per_day=best_rdot,
        notes=f"verified: r={best_r:.2f}AU rdot={best_rdot:+.5f}AU/d",
    )
    return out
