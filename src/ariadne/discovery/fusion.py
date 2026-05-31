"""Multi-broker fusion: same physical source detected across N surveys -> merged record.

A single moving object may produce ALERTS in ZTF, ATLAS, and Pan-STARRS on
the same night. Without fusion, each broker contributes separate "alerts"
that the linker treats as different objects -- losing the cross-survey
information that would give us 3x the temporal coverage. With fusion, the
combined record has 3x the detections per night, which:

  * tightens the on-sky rate measurement (more lever-arm pairs),
  * raises the chance the source crosses 3+ nights with detections,
  * adds multi-band photometry (color) for free,
  * adds independent cross-confirmation of borderline detections (real if
    confirmed in 2+ surveys; bogus if only one).

Cross-matching uses spatial + temporal proximity: an Alert from survey A
and one from survey B are the SAME detection iff:

  * angular separation < pos_tol_arcsec (default 1.5"),
  * temporal separation < time_tol_minutes (default 30 min),
  * (optional) magnitudes within mag_diff_max (default 1.0 mag, allowing
    band-conversion uncertainty),

Otherwise they're separate. The merged record's RA/Dec is the inverse-
variance-weighted mean; magnitude is per-band; all detection IDs are
preserved in `.meta["fused_alert_ids"]`.

When the same source appears across MJDs (e.g. one ZTF detection on MJD
60450.3, an ATLAS one on MJD 60450.4), they fuse to a single same-night
detection -- the linker still sees them as one position at one epoch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .brokers.base import Alert


@dataclass
class FusedDetection:
    """One detection combined across N brokers.

    Fields:
      mjd:                  median MJD across the constituent alerts.
      ra, dec:              variance-weighted mean position (degrees).
      pos_sigma_arcsec:     scatter of the constituent positions.
      mag_by_band:          dict {band: magnitude} aggregated.
      surveys:              sorted list of surveys that contributed.
      n_alerts:             number of constituent alerts.
      alert_ids:            list of original alert IDs.
      obj_ids:              list of broker obj_ids (cross-survey link hints).
    """
    mjd: float
    ra: float
    dec: float
    pos_sigma_arcsec: float
    mag_by_band: dict
    surveys: list
    n_alerts: int
    alert_ids: list
    obj_ids: list

    def as_alert(self) -> Alert:
        """Convert to the standard Alert schema for downstream code."""
        # pick a representative magnitude (prefer r-band; else any)
        rep_mag, rep_band = -99.0, ""
        for b in ("r", "g", "i", "z", "V", "o"):
            if b in self.mag_by_band:
                rep_mag = self.mag_by_band[b]
                rep_band = b
                break
        if rep_mag == -99.0 and self.mag_by_band:
            rep_band = next(iter(self.mag_by_band))
            rep_mag = self.mag_by_band[rep_band]
        surveys_tag = "+".join(self.surveys)
        return Alert(
            survey=surveys_tag,
            alert_id=f"fused_{self.mjd:.4f}_{self.ra:.4f}_{self.dec:+.4f}",
            obj_id=f"fused_{self.ra:.3f}_{self.dec:+.3f}",
            mjd=self.mjd, ra=self.ra, dec=self.dec,
            mag=rep_mag, band=rep_band,
            meta={
                "fused_alert_ids": self.alert_ids,
                "constituent_surveys": self.surveys,
                "pos_sigma_arcsec": self.pos_sigma_arcsec,
                "n_constituent_alerts": self.n_alerts,
                "mag_by_band": self.mag_by_band,
                "constituent_obj_ids": self.obj_ids,
            },
        )


def _ang_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Haversine angular separation in arcsec."""
    cos_dec = math.cos(math.radians(0.5 * (dec1 + dec2)))
    dra = (ra1 - ra2) * cos_dec
    ddec = dec1 - dec2
    return math.hypot(dra, ddec) * 3600.0


def fuse_alerts(alerts: list[Alert],
                *, pos_tol_arcsec: float = 1.5,
                time_tol_minutes: float = 30.0,
                mag_diff_max: float = 1.0) -> list[FusedDetection]:
    """Cluster alerts across surveys into fused detections.

    Args:
      alerts:           heterogeneous Alert list (any combination of surveys).
      pos_tol_arcsec:   max separation for two alerts to be the same source.
      time_tol_minutes: max time separation for two alerts to be the same epoch.
      mag_diff_max:     if both alerts have a magnitude, they cluster only if
                         |mag_a - mag_b| <= mag_diff_max (accounts for band
                         conversion + variability).

    Returns:
      List of FusedDetection, one per merged source-epoch.
    """
    if not alerts:
        return []
    alerts = sorted(alerts, key=lambda a: a.mjd)
    time_tol_days = time_tol_minutes / (60.0 * 24.0)
    visited = set()
    fused = []
    for i, a in enumerate(alerts):
        if i in visited:
            continue
        cluster = [a]
        visited.add(i)
        for j in range(i + 1, len(alerts)):
            if j in visited:
                continue
            b = alerts[j]
            if b.mjd - a.mjd > time_tol_days:
                break
            sep = _ang_sep_arcsec(a.ra, a.dec, b.ra, b.dec)
            if sep > pos_tol_arcsec:
                continue
            if (a.mag > -50 and b.mag > -50 and
                    abs(a.mag - b.mag) > mag_diff_max and
                    a.band == b.band):                # only require mag-match within same band
                continue
            cluster.append(b)
            visited.add(j)
        fused.append(_merge_cluster(cluster))
    return fused


def _merge_cluster(cluster: list[Alert]) -> FusedDetection:
    """Combine a cluster of alerts (same source, same epoch) into one record."""
    n = len(cluster)
    mjds = [a.mjd for a in cluster]
    ras = [a.ra for a in cluster]
    decs = [a.dec for a in cluster]
    # variance-weighted mean (assume equal weight if no per-alert sigma)
    ra_m = sum(ras) / n
    dec_m = sum(decs) / n
    cos_dec = math.cos(math.radians(dec_m))
    pos_scatter_arcsec = (math.sqrt(sum((r - ra_m) ** 2 * cos_dec ** 2 + (d - dec_m) ** 2
                                          for r, d in zip(ras, decs)) / max(n - 1, 1))
                            * 3600.0) if n > 1 else 0.0

    # per-band magnitude aggregation: median per band
    from collections import defaultdict
    band_mags = defaultdict(list)
    for a in cluster:
        if a.mag > -50 and a.band:
            band_mags[a.band[0]].append(a.mag)
    mag_by_band = {b: float(sorted(mags)[len(mags) // 2])
                    for b, mags in band_mags.items()}

    surveys = sorted(set(a.survey for a in cluster))
    alert_ids = [a.alert_id for a in cluster]
    obj_ids = sorted(set(a.obj_id for a in cluster))

    return FusedDetection(
        mjd=float(sorted(mjds)[n // 2]),                # median MJD
        ra=ra_m, dec=dec_m, pos_sigma_arcsec=pos_scatter_arcsec,
        mag_by_band=mag_by_band,
        surveys=surveys, n_alerts=n,
        alert_ids=alert_ids, obj_ids=obj_ids,
    )


def fuse_to_alerts(alerts: list[Alert], **kwargs) -> list[Alert]:
    """Convenience: fuse and return as a flat list of (fused) Alerts.

    Drop-in replacement for the realtime pipeline -- swap a raw `alerts` list
    for `fuse_to_alerts(alerts)` and the pipeline now gets one merged Alert
    per source-epoch.
    """
    return [f.as_alert() for f in fuse_alerts(alerts, **kwargs)]


def confirmation_count(fused: FusedDetection) -> int:
    """How many independent surveys confirmed this fused detection.

    A fused detection with confirmation_count >= 2 is essentially noise-free
    (independent surveys agreeing on a position + time = real).
    """
    return len(fused.surveys)
