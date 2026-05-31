"""ALeRCE broker adapter -- public ZTF alert stream, no auth required.

ALeRCE (Automatic Learning for the Rapid Classification of Events) is a Chilean broker
that ingests the ZTF alert stream + classifies. Public REST API; the official Python
client is `alerce` on PyPI (pip install alerce).

Docs: https://api.alerce.online/
"""
from __future__ import annotations

from typing import Iterator

from .base import Alert, BrokerBase, BrokerError


class AlerceZTFBroker(BrokerBase):
    """Adapter that translates ALeRCE `query_objects` / `query_detections` into Alert."""

    name = "ALeRCE-ZTF"

    def __init__(self):
        try:
            from alerce.core import Alerce
        except ImportError as e:
            raise BrokerError(
                "alerce client not installed -- `pip install alerce`") from e
        self._client = Alerce()

    def query_box(self, ra_min: float, ra_max: float,
                  dec_min: float, dec_max: float,
                  mjd_start: float, mjd_end: float,
                  max_alerts: int = 10000) -> Iterator[Alert]:
        """Query ALeRCE ZTF for objects in a (RA, Dec) box + MJD window."""
        # ALeRCE expects RA/Dec in degrees; uses `firstmjd` and `lastmjd` filters
        try:
            df = self._client.query_objects(
                ra=0.5 * (ra_min + ra_max),
                dec=0.5 * (dec_min + dec_max),
                radius=max(0.05, 0.5 * max(ra_max - ra_min,
                                            dec_max - dec_min)) * 3600.0,   # arcsec
                firstmjd=mjd_start,
                lastmjd=mjd_end,
                page_size=max_alerts,
                count=False,
                format="pandas",
            )
        except Exception as e:
            raise BrokerError(f"ALeRCE query_objects failed: {e}") from e

        if df is None or df.empty:
            return
        # Pull per-object detections (one row per detection)
        for _, obj_row in df.iterrows():
            oid = str(obj_row.get("oid", ""))
            if not oid:
                continue
            try:
                dets = self._client.query_detections(
                    oid, format="pandas",
                )
            except Exception:
                continue
            if dets is None or dets.empty:
                continue
            for _, d in dets.iterrows():
                mjd = float(d.get("mjd", 0.0))
                if not (mjd_start <= mjd <= mjd_end):
                    continue
                ra = float(d.get("ra", 0.0))
                dec = float(d.get("dec", 0.0))
                if not (dec_min <= dec <= dec_max and ra_min <= ra <= ra_max):
                    continue
                fid = int(d.get("fid", 0))    # 1=g, 2=r, 3=i
                band = {1: "g", 2: "r", 3: "i"}.get(fid, "")
                yield Alert(
                    survey="ZTF",
                    alert_id=str(d.get("candid", "")),
                    obj_id=oid,
                    mjd=mjd,
                    ra=ra,
                    dec=dec,
                    mag=float(d.get("magpsf", -99.0)),
                    band=band,
                    meta={"alerce_class": obj_row.get("class", "")
                          if "class" in obj_row else "",
                          "ndethist": int(obj_row.get("ndethist", 0))
                          if "ndethist" in obj_row else 0},
                )
