"""Within-night tracklet builder.

Forms two-detection pairs from same-night exposures of the same field.
Distinct from `tracklets_from_images.nightly_tracklets` which works on
in-memory Source lists; this version is shaped for the DB-driven pipeline
where each detection already has a DB-assigned id.

A "tracklet" here is one pair (det_a_id, det_b_id) at distinct image_ids
within a single night, where both detections are within a configurable
positional separation. The rate vector is computed from the pair.

Public API:
  build_within_night_tracklets(detections_with_ids, ...) -> list[TrackletRow]
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .detection_db import TrackletRow


def build_within_night_tracklets(detections_with_ids: Sequence,
                                    *,
                                    max_pair_separation_arcsec: float = 200.0,
                                    min_pair_separation_arcsec: float = 0.3,
                                    max_within_night_hours: float = 24.0,
                                    ) -> list[TrackletRow]:
    """Form within-night tracklet rows from same-image-id detection pairs.

    `detections_with_ids` is a list of (db_id, Source) tuples for one
    night's detections. Returns a list of TrackletRow records ready to
    insert into the DB.

    Pairs are formed across EXPOSURES (distinct image_ids) within the
    same night (separated by < max_within_night_hours), with positional
    separation in [min, max]_pair_separation_arcsec.
    """
    by_image: dict[str, list] = {}
    for db_id, src in detections_with_ids:
        by_image.setdefault(src.image_id, []).append((db_id, src))
    out: list[TrackletRow] = []
    max_sep_deg = max_pair_separation_arcsec / 3600.0
    min_sep_deg = min_pair_separation_arcsec / 3600.0
    image_ids = sorted(by_image.keys(),
                          key=lambda iid: by_image[iid][0][1].mjd)
    for i, iid_a in enumerate(image_ids):
        for iid_b in image_ids[i + 1:]:
            srcs_a = by_image[iid_a]
            srcs_b = by_image[iid_b]
            if not srcs_a or not srcs_b:
                continue
            mjd_a = srcs_a[0][1].mjd
            mjd_b = srcs_b[0][1].mjd
            if mjd_b <= mjd_a:
                continue
            dt_hr = (mjd_b - mjd_a) * 24.0
            if dt_hr < 1e-3 or dt_hr > max_within_night_hours:
                continue
            for db_id_a, sa in srcs_a:
                cos_dec = math.cos(math.radians(sa.dec))
                for db_id_b, sb in srcs_b:
                    dra = (sb.ra - sa.ra) * cos_dec
                    ddec = sb.dec - sa.dec
                    d_deg = math.hypot(dra, ddec)
                    if d_deg < min_sep_deg or d_deg > max_sep_deg:
                        continue
                    rate_arcsec_hr = d_deg * 3600.0 / dt_hr
                    pa_deg = math.degrees(math.atan2(dra, ddec)) % 360.0
                    mean_mag = -99.0
                    if sa.mag > -50 and sb.mag > -50:
                        mean_mag = 0.5 * (sa.mag + sb.mag)
                    out.append(TrackletRow(
                        detection_a_id=db_id_a,
                        detection_b_id=db_id_b,
                        mean_mjd=0.5 * (mjd_a + mjd_b),
                        mean_ra=0.5 * (sa.ra + sb.ra),
                        mean_dec=0.5 * (sa.dec + sb.dec),
                        rate_arcsec_hr=rate_arcsec_hr,
                        pa_deg=pa_deg,
                        rate_sigma=0.0,
                        mag=mean_mag,
                        night=int(mjd_a),
                    ))
    return out
