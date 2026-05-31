"""ALeRCE broker adapter -- public ZTF alert stream, no auth required.

ALeRCE (Automatic Learning for the Rapid Classification of Events) is a Chilean
broker that ingests the ZTF alert stream and CLASSIFIES objects (SN, AGN, VS,
asteroid, ...). We default-filter on `class_name='asteroid'` to concentrate on
solar-system bodies the broker has already pre-identified.

Public REST API; the official Python client is `alerce` on PyPI.
Docs: https://api.alerce.online/
"""
from __future__ import annotations

import math
from typing import Iterator

from .base import Alert, BrokerBase, BrokerError


class AlerceZTFBroker(BrokerBase):
    """ALeRCE adapter. Returns Alert objects representing ZTF detections.

    Default `class_name='asteroid'` filters to pre-classified moving objects --
    the right starting pool for discovery work. Pass class_name=None to query
    all classes (e.g., to look for misclassified moving objects).
    """

    name = "ALeRCE-ZTF"

    def __init__(self, class_name: str | None = "asteroid"):
        try:
            from alerce.core import Alerce
        except ImportError as e:
            raise BrokerError(
                "alerce client not installed -- `pip install alerce`") from e
        self._client = Alerce()
        self.class_name = class_name

    def _query_objects(self, ra=None, dec=None, radius=None,
                       firstmjd=None, lastmjd=None, page_size=2000):
        kwargs = dict(page_size=page_size, count=False, format="pandas")
        if self.class_name:
            kwargs["class_name"] = self.class_name
        if ra is not None and dec is not None and radius is not None:
            kwargs["ra"] = ra; kwargs["dec"] = dec; kwargs["radius"] = radius
        if firstmjd is not None:
            kwargs["firstmjd"] = firstmjd
        if lastmjd is not None:
            kwargs["lastmjd"] = lastmjd
        try:
            return self._client.query_objects(**kwargs)
        except Exception as e:
            raise BrokerError(f"ALeRCE query_objects failed: {e}") from e

    def query_box(self, ra_min: float, ra_max: float,
                  dec_min: float, dec_max: float,
                  mjd_start: float, mjd_end: float,
                  max_alerts: int = 2000) -> Iterator[Alert]:
        ra_c = 0.5 * (ra_min + ra_max)
        dec_c = 0.5 * (dec_min + dec_max)
        radius_arcsec = 0.5 * max(ra_max - ra_min, dec_max - dec_min) * 3600.0
        # ALeRCE object-level firstmjd/lastmjd are exclusive (object entirely within
        # the window). Asteroids are typically tracked far longer than any one search
        # window, so we filter at the DETECTION level after pulling the object list.
        df = self._query_objects(ra=ra_c, dec=dec_c, radius=radius_arcsec,
                                  page_size=max_alerts)
        if df is None or df.empty:
            return
        for _, obj_row in df.iterrows():
            oid = str(obj_row.get("oid", ""))
            if not oid:
                continue
            try:
                dets = self._client.query_detections(oid, format="pandas")
            except Exception:
                continue
            if dets is None or dets.empty:
                continue
            cls = obj_row.get("class", "") if "class" in obj_row else ""
            ndet = int(obj_row.get("ndethist", 0)) if "ndethist" in obj_row else 0
            prob = float(obj_row.get("probability", 0.0)) if "probability" in obj_row else 0.0
            for _, d in dets.iterrows():
                mjd = float(d.get("mjd", 0.0))
                if not (mjd_start <= mjd <= mjd_end):
                    continue
                # detection rows DO have 'ra' / 'dec' (single-detection schema)
                ra = float(d.get("ra", 0.0))
                dec = float(d.get("dec", 0.0))
                if not (dec_min <= dec <= dec_max and ra_min <= ra <= ra_max):
                    continue
                fid = int(d.get("fid", 0))
                band = {1: "g", 2: "r", 3: "i"}.get(fid, "")
                yield Alert(
                    survey="ZTF",
                    alert_id=str(d.get("candid", "")),
                    obj_id=oid,
                    mjd=mjd, ra=ra, dec=dec,
                    mag=float(d.get("magpsf", -99.0)),
                    band=band,
                    meta={"alerce_class": cls,
                          "ndethist": ndet,
                          "probability": prob},
                )

    def list_objects(self, ra: float, dec: float, radius_deg: float,
                     mjd_start: float, mjd_end: float,
                     page_size: int = 2000):
        """Quick object list (one row per ALeRCE object, not per detection).

        Returns the raw pandas DataFrame with columns oid, meanra, meandec,
        firstmjd, lastmjd, ndet, class, probability. Use this for fast
        reconnaissance; use query_box / query_cone for the full per-detection
        Alert stream.
        """
        return self._query_objects(ra=ra, dec=dec, radius=radius_deg * 3600.0,
                                    firstmjd=mjd_start, lastmjd=mjd_end,
                                    page_size=page_size)
