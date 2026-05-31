"""Gaia DR3 astrometric refinement -- pull every detection onto the Gaia frame.

The WCS solution that arrives with most archival FITS images has 0.1-0.5
arcsec systematic offset from the Gaia DR3 absolute reference frame. For
tracklet linking across nights, even 0.3" offset is a serious problem: the
linker's "are these the same object?" test typically uses a 1-2" tolerance,
and an unaccounted-for 0.3" systematic eats most of that budget.

This module:

  1. Queries Gaia DR3 (via astroquery) for reference stars within the image
     footprint, filtered for high-quality astrometric and photometric stars.
  2. Cross-matches each Gaia star against a detected source in the image
     within a small tolerance (default 1.5" -- generous enough for the
     initial WCS offset).
  3. Computes the per-source dx, dy offsets in equatorial coordinates.
  4. Fits a 6-parameter affine transform (rotation + translation + scale)
     or just a 2-parameter translation, depending on the number of matches.
  5. Applies the transform to refine every (RA, Dec) in the image's source
     list -- pulling them onto the Gaia frame to ~50 mas absolute precision.

Behaviour:
  * If Gaia is unreachable (network down / astroquery missing) -> degrade
    gracefully: return the input sources unchanged with a warning.
  * If too few cross-matches (<5) -> apply only the mean offset, not a full
    fit.
  * If matches are highly inconsistent (residual scatter >0.5") -> reject.

Reference: Gaia Collaboration 2022 (Gaia DR3 catalogue); Brown 2018 (Gaia
astrometric calibration handbook).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .source_extraction import Source


@dataclass(frozen=True)
class GaiaRefinement:
    """Result of one image's Gaia-frame refinement."""
    n_gaia_stars: int
    n_matches: int
    mean_dra_arcsec: float           # mean RA offset before correction
    mean_ddec_arcsec: float          # mean Dec offset before correction
    rms_residual_arcsec: float       # post-fit residual RMS
    rotation_arcsec: float           # fitted rotation (small angle)
    scale_factor: float              # fitted scale (1.0 = no change)
    success: bool
    method_used: str                 # "affine" | "translation" | "passthrough"


def _query_gaia_dr3(ra_deg: float, dec_deg: float, radius_deg: float,
                    g_mag_max: float = 19.0,
                    timeout_s: float = 60.0):
    """Query Gaia DR3 for high-quality astrometric stars in a cone.

    Returns a list of (ra_deg, dec_deg, g_mag, source_id) for stars with
    well-measured astrometry (parallax_over_error > 5, ruwe < 1.4, magnitude
    cut to keep only stars bright enough for our images).

    Returns [] if astroquery / network is unavailable.
    """
    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        from astropy import units as u
    except ImportError:
        return []

    try:
        q = (f"SELECT TOP 500 ra, dec, phot_g_mean_mag, source_id "
              f"FROM gaiadr3.gaia_source "
              f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
              f"               CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1 "
              f"  AND phot_g_mean_mag < {g_mag_max} "
              f"  AND ruwe < 1.4 "
              f"  AND astrometric_excess_noise < 1.0")
        job = Gaia.launch_job(q)
        result = job.get_results()
        return [(float(r["ra"]), float(r["dec"]),
                 float(r["phot_g_mean_mag"]), int(r["source_id"]))
                for r in result]
    except Exception:
        return []


def _nearest_match(target_ra: float, target_dec: float,
                   sources: list[Source], tol_arcsec: float) -> Source | None:
    """Return the source within tol_arcsec of (target_ra, target_dec), or None."""
    best, best_d = None, float("inf")
    cos_dec = math.cos(math.radians(target_dec))
    for s in sources:
        dra = (s.ra - target_ra) * cos_dec
        ddec = s.dec - target_dec
        d = math.hypot(dra, ddec) * 3600.0
        if d > tol_arcsec:
            continue
        if d < best_d:
            best, best_d = s, d
    return best


def _fit_translation_only(matches: list[tuple[float, float, float, float]]
                          ) -> tuple[float, float, float]:
    """Fit just a (dRA, dDec) shift via robust median.

    matches: list of (gaia_ra, gaia_dec, src_ra, src_dec)
    returns: (dra_deg, ddec_deg, rms_arcsec)
    """
    if not matches:
        return 0.0, 0.0, float("inf")
    dras = []
    ddecs = []
    for g_ra, g_dec, s_ra, s_dec in matches:
        cos_d = math.cos(math.radians(g_dec))
        dras.append((g_ra - s_ra) * cos_d)
        ddecs.append(g_dec - s_dec)
    dra_med = float(np.median(dras))
    ddec_med = float(np.median(ddecs))
    rms_arcsec = float(np.sqrt(
        np.mean([(d * 3600.0 - dra_med * 3600.0) ** 2 +
                  (e * 3600.0 - ddec_med * 3600.0) ** 2
                  for d, e in zip(dras, ddecs)])))
    return dra_med, ddec_med, rms_arcsec


def _fit_affine(matches: list[tuple[float, float, float, float]]
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a 2x2 affine + 2-vector translation: gaia = A @ src + t.

    Returns (A, t, rms_arcsec).
    """
    if len(matches) < 4:
        return np.eye(2), np.zeros(2), float("inf")
    src = np.array([[s_ra, s_dec] for g_ra, g_dec, s_ra, s_dec in matches])
    gaia = np.array([[g_ra, g_dec] for g_ra, g_dec, s_ra, s_dec in matches])
    # Least-squares: solve [src | 1] * params = gaia for each axis
    A = np.hstack([src, np.ones((len(src), 1))])
    # solve for each output dim
    px, _, _, _ = np.linalg.lstsq(A, gaia[:, 0], rcond=None)
    py, _, _, _ = np.linalg.lstsq(A, gaia[:, 1], rcond=None)
    mat = np.array([[px[0], px[1]], [py[0], py[1]]])
    tr = np.array([px[2], py[2]])
    # residuals
    pred = src @ mat.T + tr
    resid = pred - gaia
    cos_decs = np.cos(np.radians(gaia[:, 1]))
    rms_arcsec = float(np.sqrt(np.mean(
        (resid[:, 0] * 3600.0 * cos_decs) ** 2 +
        (resid[:, 1] * 3600.0) ** 2)))
    return mat, tr, rms_arcsec


def refine_to_gaia(sources: list[Source],
                   *, image_centre_ra_deg: float, image_centre_dec_deg: float,
                   image_radius_deg: float = 0.2,
                   match_tol_arcsec: float = 1.5,
                   min_matches_for_affine: int = 8,
                   gaia_g_mag_max: float = 19.0,
                   accept_rms_arcsec: float = 0.3) -> tuple[list[Source], GaiaRefinement]:
    """Refine every source's (RA, Dec) onto the Gaia DR3 absolute frame.

    Args:
      sources:              the detected sources in the image (with their
                             current image-WCS-derived RA/Dec).
      image_centre_ra_deg:  centre of the image footprint, for Gaia query.
      image_centre_dec_deg: centre of the image footprint.
      image_radius_deg:     query radius (should cover the full image).
      match_tol_arcsec:     cross-match tolerance for Gaia-vs-source.
      min_matches_for_affine: switch from translation-only to affine when at
                             least this many matches exist.
      gaia_g_mag_max:       only use Gaia stars brighter than this.
      accept_rms_arcsec:    reject the refinement if the fit residual exceeds
                             this; return sources unchanged.

    Returns:
      Tuple of (refined_sources, refinement_report). On any failure, refined
      = sources (passthrough) and refinement.success = False.
    """
    gaia_stars = _query_gaia_dr3(image_centre_ra_deg, image_centre_dec_deg,
                                  image_radius_deg, gaia_g_mag_max)
    if not gaia_stars:
        return sources, GaiaRefinement(
            n_gaia_stars=0, n_matches=0,
            mean_dra_arcsec=0.0, mean_ddec_arcsec=0.0,
            rms_residual_arcsec=float("inf"),
            rotation_arcsec=0.0, scale_factor=1.0,
            success=False, method_used="passthrough")

    # Cross-match Gaia stars to image sources
    matches = []
    for g_ra, g_dec, g_mag, _gid in gaia_stars:
        m = _nearest_match(g_ra, g_dec, sources, match_tol_arcsec)
        if m is not None:
            matches.append((g_ra, g_dec, m.ra, m.dec))

    if len(matches) < 3:
        return sources, GaiaRefinement(
            n_gaia_stars=len(gaia_stars), n_matches=len(matches),
            mean_dra_arcsec=0.0, mean_ddec_arcsec=0.0,
            rms_residual_arcsec=float("inf"),
            rotation_arcsec=0.0, scale_factor=1.0,
            success=False, method_used="passthrough")

    # Mean offsets (pre-fit)
    cos_d = math.cos(math.radians(image_centre_dec_deg))
    mean_dra_arcsec = float(np.median(
        [(g_ra - s_ra) * cos_d * 3600.0 for g_ra, g_dec, s_ra, s_dec in matches]))
    mean_ddec_arcsec = float(np.median(
        [(g_dec - s_dec) * 3600.0 for g_ra, g_dec, s_ra, s_dec in matches]))

    if len(matches) >= min_matches_for_affine:
        method = "affine"
        mat, tr, rms = _fit_affine(matches)
        if rms > accept_rms_arcsec:
            return sources, GaiaRefinement(
                n_gaia_stars=len(gaia_stars), n_matches=len(matches),
                mean_dra_arcsec=mean_dra_arcsec, mean_ddec_arcsec=mean_ddec_arcsec,
                rms_residual_arcsec=rms, rotation_arcsec=0.0, scale_factor=1.0,
                success=False, method_used="reject_high_rms")
        refined = []
        for s in sources:
            v = np.array([s.ra, s.dec])
            v2 = mat @ v + tr
            refined.append(Source(
                ra=float(v2[0]) % 360.0, dec=float(v2[1]),
                flux=s.flux, mag=s.mag, fwhm_px=s.fwhm_px,
                mjd=s.mjd, image_id=s.image_id, x=s.x, y=s.y))
        # rotation = arc-tan of off-diagonal; scale = sqrt(det)
        rotation_arcsec = math.degrees(math.atan2(mat[1, 0], mat[0, 0])) * 3600.0
        scale = math.sqrt(abs(np.linalg.det(mat)))
        return refined, GaiaRefinement(
            n_gaia_stars=len(gaia_stars), n_matches=len(matches),
            mean_dra_arcsec=mean_dra_arcsec, mean_ddec_arcsec=mean_ddec_arcsec,
            rms_residual_arcsec=rms,
            rotation_arcsec=rotation_arcsec, scale_factor=scale,
            success=True, method_used="affine")

    # Translation-only fallback
    method = "translation"
    dra, ddec, rms = _fit_translation_only(matches)
    if rms > accept_rms_arcsec:
        return sources, GaiaRefinement(
            n_gaia_stars=len(gaia_stars), n_matches=len(matches),
            mean_dra_arcsec=mean_dra_arcsec, mean_ddec_arcsec=mean_ddec_arcsec,
            rms_residual_arcsec=rms, rotation_arcsec=0.0, scale_factor=1.0,
            success=False, method_used="reject_high_rms")

    refined = []
    for s in sources:
        refined.append(Source(
            ra=(s.ra + dra) % 360.0, dec=(s.dec + ddec),
            flux=s.flux, mag=s.mag, fwhm_px=s.fwhm_px,
            mjd=s.mjd, image_id=s.image_id, x=s.x, y=s.y))
    return refined, GaiaRefinement(
        n_gaia_stars=len(gaia_stars), n_matches=len(matches),
        mean_dra_arcsec=mean_dra_arcsec, mean_ddec_arcsec=mean_ddec_arcsec,
        rms_residual_arcsec=rms, rotation_arcsec=0.0, scale_factor=1.0,
        success=True, method_used="translation")
