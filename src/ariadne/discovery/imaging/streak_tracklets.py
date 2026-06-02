"""Convert detected streaks into detections + within-night tracklets.

A streak in a single exposure is a fast-moving object dragged across the
detector during the integration. Its two endpoints are the object's sky
position at exposure-start and exposure-end. That means a SINGLE exposure
of a fast NEO already yields a tracklet (two timed positions -> a rate
vector) -- no second exposure needed.

This is the bridge that lets the streak detector feed the same
discovery pipeline (DB -> linker -> IOD -> submission) as point-source
tracklets. Without it, streaks are detected but never become candidates.

Public API:
  streak_to_endpoints(streak, wcs, exposure_start_mjd, exposure_seconds)
    -> two (mjd, ra, dec) endpoint detections
  ingest_streaks(db, streaks, wcs, image_id, exposure_start_mjd,
                   exposure_seconds, pixel_scale_arcsec, ...)
    -> (detection_ids, tracklet_ids) persisted to the DB
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .streaks import Streak, classify_streak
from .detection_db import DetectionRow, TrackletRow


@dataclass
class StreakEndpoints:
    """The two timed sky positions of a streak's endpoints."""
    mjd_start: float
    ra_start: float
    dec_start: float
    mjd_end: float
    ra_end: float
    dec_end: float
    rate_arcsec_hr: float
    pa_deg: float
    mag: float


def streak_to_endpoints(streak: Streak, wcs,
                          exposure_start_mjd: float,
                          exposure_seconds: float,
                          *, zeropoint_mag: float | None = None,
                          ) -> StreakEndpoints:
    """Project a streak's pixel endpoints through the WCS to two timed
    sky positions.

    The leading endpoint is observed at exposure start, the trailing at
    exposure end (we cannot tell direction of travel from one exposure,
    so we adopt the (x1,y1)->(x2,y2) order; the linker is direction-
    agnostic via the rate magnitude + PA modulo 180).
    """
    ra1, dec1 = wcs.pixel_to_world_values(streak.x1, streak.y1)
    ra2, dec2 = wcs.pixel_to_world_values(streak.x2, streak.y2)
    ra1 = float(ra1) % 360.0
    ra2 = float(ra2) % 360.0
    dec1 = float(dec1)
    dec2 = float(dec2)
    mjd_start = exposure_start_mjd
    mjd_end = exposure_start_mjd + exposure_seconds / 86400.0
    # Rate from the angular separation over the exposure
    cos_dec = math.cos(math.radians(0.5 * (dec1 + dec2)))
    dra = (ra2 - ra1) * cos_dec
    ddec = dec2 - dec1
    sep_deg = math.hypot(dra, ddec)
    dt_hr = exposure_seconds / 3600.0
    rate_arcsec_hr = sep_deg * 3600.0 / max(dt_hr, 1e-9)
    pa_deg = math.degrees(math.atan2(dra, ddec)) % 360.0
    # Magnitude from total flux if a zeropoint is supplied
    if zeropoint_mag is not None and streak.total_flux > 0:
        mag = zeropoint_mag - 2.5 * math.log10(streak.total_flux)
    else:
        mag = -99.0
    return StreakEndpoints(
        mjd_start=mjd_start, ra_start=ra1, dec_start=dec1,
        mjd_end=mjd_end, ra_end=ra2, dec_end=dec2,
        rate_arcsec_hr=rate_arcsec_hr, pa_deg=pa_deg, mag=mag)


def ingest_streaks(db, streaks: Sequence[Streak], wcs,
                     image_id: str,
                     exposure_start_mjd: float,
                     exposure_seconds: float,
                     *, pixel_scale_arcsec: float = 0.263,
                     psf_fwhm_px: float = 3.0,
                     frame_diagonal_px: float | None = None,
                     zeropoint_mag: float | None = None,
                     asteroid_only: bool = True,
                     ccd_id: str = "",
                     ) -> tuple[list[int], list[int]]:
    """Persist each asteroid-candidate streak as two endpoint detections
    plus a within-night tracklet linking them.

    Returns (detection_ids, tracklet_ids).

    If `asteroid_only` is True (default), streaks classified as
    satellites / cosmic rays / extended sources are skipped -- only
    `asteroid_candidate` streaks become detections.
    """
    det_ids: list[int] = []
    trk_ids: list[int] = []
    for s in streaks:
        cls = classify_streak(
            s, exposure_seconds=exposure_seconds,
            pixel_scale_arcsec=pixel_scale_arcsec,
            psf_fwhm_px=psf_fwhm_px,
            frame_diagonal_px=frame_diagonal_px)
        if asteroid_only and not cls["is_asteroid_candidate"]:
            continue
        ep = streak_to_endpoints(
            s, wcs, exposure_start_mjd, exposure_seconds,
            zeropoint_mag=zeropoint_mag)
        # Two timed detections
        rows = [
            DetectionRow(
                image_id=f"{image_id}#streak_start", mjd=ep.mjd_start,
                ra=ep.ra_start, dec=ep.dec_start, mag=ep.mag,
                flux=s.total_flux, fwhm_px=s.width_px,
                x_pix=s.x1, y_pix=s.y1,
                astrom_sigma_arcsec=pixel_scale_arcsec,
                ccd_id=ccd_id, status="streak"),
            DetectionRow(
                image_id=f"{image_id}#streak_end", mjd=ep.mjd_end,
                ra=ep.ra_end, dec=ep.dec_end, mag=ep.mag,
                flux=s.total_flux, fwhm_px=s.width_px,
                x_pix=s.x2, y_pix=s.y2,
                astrom_sigma_arcsec=pixel_scale_arcsec,
                ccd_id=ccd_id, status="streak"),
        ]
        ids = db.insert_detections(rows)
        det_ids.extend(ids)
        # The streak IS a tracklet: one rate vector from a single exposure
        tid = db.insert_tracklet(TrackletRow(
            detection_a_id=ids[0], detection_b_id=ids[1],
            mean_mjd=0.5 * (ep.mjd_start + ep.mjd_end),
            mean_ra=0.5 * (ep.ra_start + ep.ra_end),
            mean_dec=0.5 * (ep.dec_start + ep.dec_end),
            rate_arcsec_hr=ep.rate_arcsec_hr, pa_deg=ep.pa_deg,
            rate_sigma=0.0, mag=ep.mag,
            night=int(exposure_start_mjd)))
        trk_ids.append(tid)
    db.conn.commit()
    return det_ids, trk_ids
