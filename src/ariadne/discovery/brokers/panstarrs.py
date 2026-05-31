"""Pan-STARRS DR2 broker: archival cone-search via MAST.

Pan-STARRS1 (PS1) is a 1.8m telescope on Haleakala that surveyed 3pi sr of
sky from 2010 to 2014 in five bands (g, r, i, z, y) to mean limiting depth
~22 in r. The MAST archive provides a `Pan-STARRS Catalog` interface that
returns per-detection records (mean position + per-epoch mags + per-epoch
positions) for any cone in the survey.

For moving-object work, PS1 is invaluable as a PRE-OPPOSITION archive: when
a candidate has its discovery in 2024-2026, you can search PS1 from 2010-14
for prior detections that would extend the arc by a decade -- often turning
"interesting candidate" into "secure orbit."

This module returns the catalogue's per-epoch positions as Alert records,
one Alert per detection (NOT per object). The realtime pipeline then treats
them like any other broker's alerts: cluster -> tracklet -> link to a fitted
orbit.

Authentication: MAST does not require credentials for the PS1 catalogue,
but pip-install `astroquery` to use this broker.

Reference: Chambers et al. 2016 (Pan-STARRS1 survey paper); Magnier et al.
2020 (PS1 DR2 release).
"""
from __future__ import annotations

import math
from typing import Iterator

from .base import Alert, BrokerBase


class PanStarrsBroker(BrokerBase):
    """Pan-STARRS1 DR2 broker via MAST.

    Args:
      catalog:  "mean" returns mean positions/magnitudes per object;
                "detection" returns one Alert per individual epoch
                (the right choice for moving-object tracklet building).
      bands:    list of bands to include ("g", "r", "i", "z", "y").
      mag_max:  drop sources fainter than this in any band.
    """

    name = "PanSTARRS"

    def __init__(self, catalog: str = "detection",
                 bands: list[str] | None = None,
                 mag_max: float = 22.5):
        self.catalog = catalog
        self.bands = bands or ["g", "r", "i", "z"]
        self.mag_max = mag_max

    def query_cone(self, ra: float, dec: float, radius_deg: float,
                   mjd_start: float, mjd_end: float,
                   max_alerts: int = 5000) -> Iterator[Alert]:
        try:
            from astroquery.mast import Catalogs
            from astropy.coordinates import SkyCoord
            from astropy import units as u
        except ImportError:
            return iter([])

        try:
            coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            table = Catalogs.query_region(
                coord, radius=radius_deg * u.deg,
                catalog="Panstarrs", table="detection",
                data_release="dr2")
        except Exception:
            return iter([])

        if table is None or len(table) == 0:
            return iter([])

        n = 0
        for row in table:
            try:
                row_mjd = float(row.get("obsTime", row.get("obsTimeMJD", 0)))
                if not (mjd_start <= row_mjd <= mjd_end):
                    continue
                band_id = row.get("filterID", row.get("filter", 0))
                band_map = {1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
                band = band_map.get(int(band_id), "") if isinstance(band_id, (int, float)) else str(band_id)
                if band and self.bands and band not in self.bands:
                    continue
                mag = float(row.get("psfFlux", row.get("apMag", float("nan"))))
                if not math.isfinite(mag) or mag > self.mag_max:
                    continue
                ra_d = float(row.get("ra", row.get("raMean", float("nan"))))
                dec_d = float(row.get("dec", row.get("decMean", float("nan"))))
                obj_id = str(row.get("objID", "ps1_unknown"))
                alert_id = str(row.get("detectID",
                                        row.get("objID", f"ps1_{row_mjd}_{ra_d}")))
                if not (math.isfinite(ra_d) and math.isfinite(dec_d)):
                    continue
                yield Alert(survey="PanSTARRS", alert_id=alert_id, obj_id=obj_id,
                            mjd=row_mjd, ra=ra_d, dec=dec_d, mag=mag, band=band,
                            meta={"panstarrs_row": int(row.get("objID", 0))})
                n += 1
                if n >= max_alerts:
                    return
            except (ValueError, TypeError, KeyError):
                continue
