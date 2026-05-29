"""Ingest the MPC Isolated Tracklet File (ITF) -- the real unlinked-detection haystack (Stage 43).

The ITF is the Minor Planet Center's flat file of observations that its own pipeline could NOT link
to any object: millions of genuinely unassociated detections. Feeding it to the Stage-39/42 linker is
the real attempt at finding a new (or mislinked) distant body.

This module parses the 80-column ITF, groups same-temporary-designation detections into tracklets,
and filters to SLOW movers -- the distant-object regime Ariadne is validated for, which also shrinks
the millions of tracklets to a tractable set. The linker (discovery.linkage) then runs per sky/time bin.

HONEST: the MPC's linking is excellent, so what remains unlinked is hard; the likely outcome is
re-linking known objects or nothing new. A genuine new find would be extraordinary and must be
cross-checked, not announced.
"""
from __future__ import annotations

import gzip
import math
from collections import defaultdict

import numpy as np


def _parse_date(s):
    """MPC date field 'YYYY MM DD.dddddd' -> Julian Date (UTC, good to ~seconds)."""
    y = int(s[0:4]); mo = int(s[5:7]); d = float(s[8:])
    # JDN at 0h of (y, mo, floor(d)) via the standard algorithm, + day fraction
    day = int(d)
    a = (14 - mo) // 12
    yy = y + 4800 - a
    mm = mo + 12 * a - 3
    jdn = day + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 0.5 + (d - day)


def _parse_radec(line):
    ra = (float(line[32:34]) + float(line[35:37]) / 60.0 + float(line[38:44]) / 3600.0) * 15.0
    sign = -1.0 if line[44] == "-" else 1.0
    dec = sign * (float(line[45:47]) + float(line[48:50]) / 60.0 + float(line[51:56]) / 3600.0)
    return math.radians(ra), math.radians(dec)


def parse_itf(path, max_lines=None):
    """Parse the ITF (.txt.gz or .txt) -> dict: temp_designation -> list of (jd, ra, dec, obscode).

    Robust to malformed lines (skipped). Returns only multi-detection tracklet groups downstream.
    """
    op = gzip.open if str(path).endswith(".gz") else open
    groups = defaultdict(list)
    n = 0
    with op(path, "rt", errors="replace") as f:
        for line in f:
            if len(line) < 56:
                continue
            desig = line[5:12].strip()
            if not desig:
                continue
            try:
                jd = _parse_date(line[15:32])
                ra, dec = _parse_radec(line)
            except (ValueError, IndexError):
                continue
            groups[desig].append((jd, ra, dec, line[77:80].strip()))
            n += 1
            if max_lines and n >= max_lines:
                break
    return groups


def build_tracklets(groups, min_obs=2, max_arc_hours=48.0):
    """Group -> tracklets: same-designation detections spanning a night -> position + on-sky rate."""
    tracks = []
    for desig, obs in groups.items():
        if len(obs) < min_obs:
            continue
        obs = sorted(obs)
        (j1, r1, d1, _), (j2, r2, d2, oc) = obs[0], obs[-1]
        dtj = j2 - j1
        if dtj <= 0 or dtj > max_arc_hours / 24.0:
            continue
        # rate from endpoints; mean position at midpoint
        dra = (r2 - r1) / (dtj * 86400.0)
        ddec = (d2 - d1) / (dtj * 86400.0)
        et = (0.5 * (j1 + j2) - 2451545.0) * 86400.0
        rate_arcsec_hr = math.hypot(dra * math.cos(0.5 * (d1 + d2)), ddec) * 206265.0 * 3600.0
        tracks.append({"desig": desig, "t": et, "jd": 0.5 * (j1 + j2),
                       "ra": 0.5 * (r1 + r2), "dec": 0.5 * (d1 + d2),
                       "dra": dra, "ddec": ddec, "rate_arcsec_hr": rate_arcsec_hr,
                       "obscode": oc, "obj": -1})
    return tracks


def filter_slow(tracks, max_rate_arcsec_hr=5.0):
    """Keep only SLOW tracklets (distant-object candidates). TNOs move ~1-3 arcsec/hr; NEOs/MBAs much faster."""
    return [t for t in tracks if t["rate_arcsec_hr"] <= max_rate_arcsec_hr]


def sky_time_bins(tracks, ra_cells=24, dec_cells=12, window_days=150):
    """Partition slow tracklets into (RA, Dec, opposition-window) bins for per-bin linking."""
    bins = defaultdict(list)
    for t in tracks:
        ira = int(t["ra"] / (2 * math.pi) * ra_cells) % ra_cells
        idec = int((t["dec"] + math.pi / 2) / math.pi * dec_cells)
        iwin = int(t["jd"] // window_days)
        bins[(ira, idec, iwin)].append(t)
    return bins


def link_bins(tracks, r_grid_au, rdot_grid, min_obs=4, min_nights=3, cluster_au=0.5,
              ra_cells=24, dec_cells=12, window_days=150, min_bin=None):
    """Run the HelioLinC linker per (sky, time) bin over the slow tracklets. Returns candidate clusters.

    Each bin is linked independently (objects live in one sky cell within one opposition). Only bins
    with enough tracklets to possibly form a >=min_obs cluster are processed.
    """
    from . import linkage as L
    if min_bin is None:
        min_bin = min_obs
    bins = sky_time_bins(tracks, ra_cells, dec_cells, window_days)
    out = []
    for key, bt in bins.items():
        if len(bt) < min_bin:
            continue
        geom = L.precompute_geometry(bt)
        t_ref = float(np.median([t["t"] for t in bt]))
        with np.errstate(all="ignore"):
            cands = L.link(geom, t_ref, r_grid_au, rdot_grid, cluster_au=cluster_au,
                           min_obs=min_obs, min_nights=min_nights)
        for c in cands:
            members = [bt[i] for i in c]
            out.append({"bin": key, "n": len(c),
                        "designations": [m["desig"] for m in members],
                        "ra_deg": float(np.degrees(np.mean([m["ra"] for m in members]))),
                        "dec_deg": float(np.degrees(np.mean([m["dec"] for m in members]))),
                        "jd": float(np.mean([m["jd"] for m in members])),
                        "tracklets": members})
    return out
