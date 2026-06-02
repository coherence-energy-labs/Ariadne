"""Build tracklets from per-image source detections (the image-based discovery path).

Given a list of `Source` detections from many images covering the same sky region
over a few nights, build:
  1. Same-night tracklets: pair sources within a few hours that move at a
     plausible distant-object rate (default 0.05-5 arcsec/hr).
  2. Multi-night candidate arcs: chain same-night tracklets whose extrapolated
     position matches the next night's tracklet centroid within tolerance.

The output is the same tracklet dict schema as discovery.itf.build_tracklets, so
the downstream HelioLinC linker + IOD + LM pipeline runs unchanged.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .source_extraction import Source

SEC_PER_DAY = 86400.0


def _angular_separation_arcsec(a: Source, b: Source) -> float:
    """Small-angle separation in arcsec between two sources."""
    dra = math.radians(b.ra - a.ra) * math.cos(math.radians(a.dec))
    ddec = math.radians(b.dec - a.dec)
    return math.degrees(math.hypot(dra, ddec)) * 3600.0


def suppress_stationary_sources(sources: list[Source],
                                radius_arcsec: float = 0.8) -> list[Source]:
    """Remove same-night fixed-source repeats before moving-object pairing.

    The raw image path sees stars and static artifacts in every exposure. If
    those fixed sources are allowed into pair formation, bright star/star
    accidents dominate the O(N^2) candidate space and can bury real moving
    objects under caps. This veto drops a source when a different exposure on
    the same night has a neighbour at the same sky position.
    """
    if radius_arcsec <= 0.0 or len(sources) < 2:
        return list(sources)
    from scipy.spatial import cKDTree

    by_night = defaultdict(list)
    for idx, src in enumerate(sources):
        by_night[int(round(src.mjd))].append((idx, src))
    keep = np.ones(len(sources), dtype=bool)
    radius_deg = radius_arcsec / 3600.0
    for rows in by_night.values():
        if len(rows) < 2:
            continue
        dec0 = float(np.median([src.dec for _, src in rows]))
        cos_dec = math.cos(math.radians(dec0))
        xy = np.array([[src.ra * cos_dec, src.dec] for _, src in rows], dtype=float)
        tree = cKDTree(xy)
        for local_i, (global_i, src) in enumerate(rows):
            near = tree.query_ball_point(xy[local_i], radius_deg)
            for local_j in near:
                if local_i == local_j:
                    continue
                _, other = rows[local_j]
                if other.image_id != src.image_id:
                    keep[global_i] = False
                    break
    return [src for i, src in enumerate(sources) if keep[i]]


def nightly_tracklets(sources: list[Source],
                      min_rate_arcsec_hr: float = 0.05,
                      max_rate_arcsec_hr: float = 5.0,
                      min_pair_dt_hours: float = 0.1,
                      max_pair_dt_hours: float = 6.0,
                      obscode: str = "807",
                      min_pair_separation_arcsec: float = 0.5,
                      max_per_night: int | None = 5000,
                      stationary_veto_arcsec: float = 0.0) -> list[dict]:
    """Pair sources within the same night into Ariadne-shaped tracklets.

    Default rate window targets the distant-object regime (TNOs / outer Centaurs /
    far-side MBA tail; faster movers are excluded). The dict schema matches
    discovery.itf.build_tracklets so the HelioLinC linker drops in unchanged.
    """
    if stationary_veto_arcsec > 0.0:
        sources = suppress_stationary_sources(sources, stationary_veto_arcsec)
    by_night = defaultdict(list)
    for s in sources:
        by_night[int(round(s.mjd))].append(s)
    tracks = []
    for night, ss in by_night.items():
        ss = sorted(ss, key=lambda s: s.mjd)
        per_night = []
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                a, b = ss[i], ss[j]
                dt_h = (b.mjd - a.mjd) * 24.0
                # SANITY: same-frame (dt=0) pairs cannot be a real tracklet
                if dt_h < min_pair_dt_hours:
                    continue
                if dt_h > max_pair_dt_hours:
                    break
                # SANITY: require minimum separation (drops noise pixels of
                # the same source being paired across frames within seconds)
                sep = _angular_separation_arcsec(a, b)
                if sep < min_pair_separation_arcsec:
                    continue
                rate = sep / dt_h
                if not (min_rate_arcsec_hr <= rate <= max_rate_arcsec_hr):
                    continue
                jd_mid = 0.5 * (a.mjd + b.mjd) + 2400000.5
                et_mid = (jd_mid - 2451545.0) * SEC_PER_DAY
                dt_s = (b.mjd - a.mjd) * SEC_PER_DAY
                per_night.append({
                    "t": et_mid, "jd": jd_mid,
                    "ra": math.radians(0.5 * (a.ra + b.ra)),
                    "dec": math.radians(0.5 * (a.dec + b.dec)),
                    "dra": math.radians(b.ra - a.ra) / dt_s,
                    "ddec": math.radians(b.dec - a.dec) / dt_s,
                    "rate_arcsec_hr": rate,
                    "desig": f"IMG_{a.image_id}_{b.image_id}",
                    "obscode": obscode, "obj": -1,
                    "source_pair": (a, b),
                    "night": night,
                })
        # CAP: explosion-protection. With N detections you can build O(N^2)
        # pairs; on 500-detection nights that's a quarter-million tracklets
        # all of which then go through HelioLinC/IOD. Cap per night to keep
        # downstream tractable. Keep the brightest-flux pairs.
        if max_per_night is not None and len(per_night) > max_per_night:
            per_night.sort(key=lambda t: (t["source_pair"][0].flux
                                            + t["source_pair"][1].flux),
                           reverse=True)
            per_night = per_night[:max_per_night]
        tracks.extend(per_night)
    return tracks


def chain_multi_night(tracklets: list[dict], max_position_gap_arcsec: float = 60.0,
                      max_rate_change_pct: float = 50.0,
                      max_nights_gap: int = 14) -> list[list[dict]]:
    """Chain multi-night tracklets into candidate orbital arcs.

    For each night-N tracklet, extrapolate its (RA, Dec) to a later night using the
    on-sky rate (linear extrapolation -- good enough for short multi-night arcs).
    Pair with night-M tracklets whose centroid is within `max_position_gap_arcsec`
    of the extrapolated position AND whose rate is within `max_rate_change_pct` of
    the night-N rate (slow objects don't accelerate much on a few-night baseline).

    Returns a list of CHAINS, each a list of tracklets in time order. Singletons
    (no chain partner) are NOT returned -- the discovery pipeline requires >= 2
    nights of arc.
    """
    by_night = defaultdict(list)
    for t in tracklets:
        by_night[t["night"]].append(t)
    nights = sorted(by_night)

    # Each tracklet starts its own chain; we merge by extrapolation matching
    chains = {id(t): [t] for tlist in by_night.values() for t in tlist}
    for k, night in enumerate(nights):
        for t in by_night[night]:
            for later in nights[k + 1: k + 1 + max_nights_gap]:
                dt_days = later - night
                # extrapolate t's centroid forward by dt_days
                dra_per_day = t["dra"] * SEC_PER_DAY
                ddec_per_day = t["ddec"] * SEC_PER_DAY
                ra_pred = math.degrees(t["ra"] + dra_per_day * dt_days)
                dec_pred = math.degrees(t["dec"] + ddec_per_day * dt_days)
                best, best_d = None, float("inf")
                for u in by_night[later]:
                    dra = math.radians(math.degrees(u["ra"]) - ra_pred) * math.cos(u["dec"])
                    ddec = math.radians(math.degrees(u["dec"]) - dec_pred)
                    gap = math.degrees(math.hypot(dra, ddec)) * 3600.0
                    if gap > max_position_gap_arcsec:
                        continue
                    # rate-consistency check
                    drate_pct = (abs(u["rate_arcsec_hr"] - t["rate_arcsec_hr"])
                                 / max(t["rate_arcsec_hr"], 1e-9) * 100.0)
                    if drate_pct > max_rate_change_pct:
                        continue
                    if gap < best_d:
                        best, best_d = u, gap
                if best is not None:
                    # merge u's chain into t's
                    src_chain = chains[id(t)]
                    dst_chain = chains[id(best)]
                    if src_chain is not dst_chain:
                        merged = src_chain + dst_chain
                        for x in merged:
                            chains[id(x)] = merged
                    break        # only one chain partner per tracklet per night
    seen = set()
    multi = []
    for chain in chains.values():
        cid = id(chain)
        if cid in seen:
            continue
        seen.add(cid)
        if len(chain) >= 2:
            unique_nights = len(set(t["night"] for t in chain))
            if unique_nights >= 2:
                multi.append(sorted(chain, key=lambda t: t["t"]))
    return multi
