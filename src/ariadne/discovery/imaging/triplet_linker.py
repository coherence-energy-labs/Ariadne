"""Within-night triplet tracklet linker (3+ epochs, collinear-motion).

The discovery unit for surveys that take >=3 visits of a field per night.
A real moving object appears at 3 positions on a straight, constant-rate
track over the ~hour-scale within-night arc. Requiring a THIRD point that
agrees with the line from the first two (to a tight tolerance) is what
makes within-night linking clean where 2-night sparse pairing is hopeless:

  Validated on real DECam (3x30s, 134 min, 2024-04-28): 2-night blind
  pairing made ~89M chance pairs; the 3-point collinear linker made 288
  (tol 2") -> 72 (tol 1"), recovering all 9 known asteroids in the field.
  Tighter astrometry (Gaia-refined) allows a tighter tolerance and fewer
  chance triplets.

Public API:
  link_collinear_tracklets(epochs, ...) -> list[Tracklet3]

`epochs` is a list of >=3 (ra, dec, mjd) arrays, one per exposure, in any
time order (sorted internally). Returns triplet tracklets with their
constant-rate motion vector. Pure geometry, no DB / catalog dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Tracklet3:
    """A 3-point within-night tracklet on a constant-rate track."""
    idx: tuple                  # (i, j, k) detection indices per epoch
    mean_ra: float
    mean_dec: float
    mean_mjd: float
    rate_deg_day: float
    pa_deg: float
    collinear_resid_arcsec: float
    mag: float


def link_collinear_tracklets(epochs,
                                *, max_rate_deg_day: float = 0.8,
                                min_rate_deg_day: float = 0.02,
                                collinear_tol_arcsec: float = 1.0,
                                mags=None):
    """Link detections across >=3 same-night exposures into collinear,
    constant-rate triplet tracklets.

    epochs: list of (ra_deg_array, dec_deg_array, mjd_float). Uses the
            first three by time (after sorting). For >3 exposures, callers
            can run on consecutive triples and merge.
    mags:   optional list of per-epoch magnitude arrays (same order as the
            sorted epochs) for the reported tracklet magnitude.

    A candidate is kept iff:
      - the epoch0->1 displacement implies a rate in [min,max]_rate, and
      - the epoch-2 detection lies within collinear_tol_arcsec of the
        position predicted by constant-rate motion from epochs 0->1.
    """
    from scipy.spatial import cKDTree
    order = sorted(range(len(epochs)), key=lambda i: epochs[i][2])
    if len(order) < 3:
        raise ValueError("need >=3 epochs for triplet linking")
    e0, e1, e2 = (epochs[order[0]], epochs[order[1]], epochs[order[2]])
    ra0, dec0, t0 = e0
    ra1, dec1, t1 = e1
    ra2, dec2, t2 = e2
    if mags is not None:
        m0, m1, m2 = (mags[order[0]], mags[order[1]], mags[order[2]])
    else:
        m0 = m1 = m2 = None
    if len(ra0) == 0 or len(ra1) == 0 or len(ra2) == 0:
        return []
    cosd = math.cos(math.radians(float(np.median(dec0))))
    dt01 = t1 - t0
    dt02 = t2 - t0
    if dt01 <= 0 or dt02 <= 0:
        return []
    tree1 = cKDTree(np.column_stack([ra1 * cosd, dec1]))
    tree2 = cKDTree(np.column_stack([ra2 * cosd, dec2]))
    d01_hi = max_rate_deg_day * dt01
    d01_lo = min_rate_deg_day * dt01
    tol_deg = collinear_tol_arcsec / 3600.0
    out = []
    for i in range(len(ra0)):
        for j in tree1.query_ball_point([ra0[i] * cosd, dec0[i]], d01_hi):
            dra = (ra1[j] - ra0[i]) * cosd
            ddec = dec1[j] - dec0[i]
            d01 = math.hypot(dra, ddec)
            if d01 < d01_lo:
                continue
            pra = ra0[i] + (ra1[j] - ra0[i]) * (dt02 / dt01)
            pdec = dec0[i] + (dec1[j] - dec0[i]) * (dt02 / dt01)
            for k in tree2.query_ball_point([pra * cosd, pdec], tol_deg):
                resid = math.hypot((ra2[k] - pra) * cosd,
                                     dec2[k] - pdec) * 3600
                if resid > collinear_tol_arcsec:
                    continue
                mag = -99.0
                if m0 is not None:
                    vals = [v for v in (m0[i], m1[j], m2[k]) if v > -50]
                    if vals:
                        mag = float(np.median(vals))
                out.append(Tracklet3(
                    idx=(i, j, k),
                    mean_ra=float((ra0[i] + ra1[j] + ra2[k]) / 3),
                    mean_dec=float((dec0[i] + dec1[j] + dec2[k]) / 3),
                    mean_mjd=float((t0 + t1 + t2) / 3),
                    rate_deg_day=d01 / dt01,
                    pa_deg=math.degrees(math.atan2(dra, ddec)) % 360.0,
                    collinear_resid_arcsec=resid, mag=mag))
    return out
