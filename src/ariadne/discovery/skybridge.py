"""Real-data bridge: Ariadne localization -> Signalbook catalog cross-match (Stage 29).

Turns Ariadne's *simulation-based* "look here" into a query against REAL public observational data.
Signalbook maintains a 47-million-record cross-modality atlas from real public catalogs (Gaia DR3,
Pan-STARRS, AllWISE, 2MASS, SDSS, Chandra/XMM/Swift/eROSITA X-ray, Fermi gamma, IceCube neutrinos,
GWOSC, ...). We read it read-only and, given a Stage-28 localization sky-box, return every catalogued
celestial source in the cone -- the multi-band confirmation step (the IR/optical handoff made real).

A real gap we close: in the base atlas, celestial sources store their sky position inside
`payload_json` (ra_deg/dec_deg), NOT in the indexed lat_deg/lon_deg columns, so signalbook's own
R-tree does not index 3M+ celestial sources by sky position. `build_celestial_index` extracts those
coordinates once into an indexed `celestial_sources` table -- a genuine enhancement that makes the
catalog sky-queryable. `query_sky` then runs fast cone searches against it.

HONEST scope (critical): this CROSS-MATCHES a localization against KNOWN catalogued sources. It does
NOT detect a NEW moving body -- that needs multi-epoch imaging / proper-motion detection, which a
static source catalog does not provide. It answers "what known sources sit where we are pointing?"
(rule-out / candidate flag), not "here is an unseen planet."
"""
from __future__ import annotations

import json
import math
import os
import sqlite3

CELESTIAL_MODALITIES = (
    "optical", "x_ray", "gamma", "gamma_ray", "neutrino", "compact_object",
    "pulsar", "high_z_galaxy", "radio", "cosmic_ray",
)

_OBLIQUITY_DEG = 23.439291    # mean obliquity of the ecliptic (J2000)


def ecliptic_to_equatorial(lon_deg, lat_deg):
    """Ecliptic (lon, lat) -> equatorial (RA, Dec) in degrees (J2000 obliquity)."""
    lam, bet, eps = (math.radians(v) for v in (lon_deg, lat_deg, _OBLIQUITY_DEG))
    dec = math.asin(math.sin(bet) * math.cos(eps) + math.cos(bet) * math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.sin(lam) * math.cos(eps) - math.tan(bet) * math.sin(eps), math.cos(lam))
    return math.degrees(ra) % 360.0, math.degrees(dec)


def _angsep_deg(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def build_celestial_index(atlas_db, out_db, modalities=CELESTIAL_MODALITIES, batch=20000):
    """Extract real sky coordinates (ra_deg/dec_deg from payload_json) of celestial sources in a
    Signalbook atlas into an indexed `celestial_sources` table. One-time; returns the row count.

    This makes signalbook's celestial catalog sky-queryable (the base atlas indexes only lat/lon,
    which celestial sources leave NULL).
    """
    src = sqlite3.connect("file:" + atlas_db + "?mode=ro", uri=True)
    dst = sqlite3.connect(out_db)
    try:
        dst.execute("drop table if exists celestial_sources")
        dst.execute("create table celestial_sources "
                    "(record_id text, modality text, observatory text, ra_deg real, dec_deg real)")
        q = ("select record_id, modality, observatory, payload_json from cross_scale_records "
             "where modality in (%s)" % ",".join("?" * len(modalities)))
        rows, n = [], 0
        for rid, mod, obs, pj in src.execute(q, list(modalities)):
            if not pj:
                continue
            try:
                d = json.loads(pj)
            except (ValueError, TypeError):
                continue
            ra, dec = d.get("ra_deg"), d.get("dec_deg")
            if ra is None or dec is None:
                continue
            rows.append((rid, mod, obs, float(ra), float(dec)))
            if len(rows) >= batch:
                dst.executemany("insert into celestial_sources values (?,?,?,?,?)", rows)
                n += len(rows); rows = []
        if rows:
            dst.executemany("insert into celestial_sources values (?,?,?,?,?)", rows); n += len(rows)
        dst.execute("create index ix_cs_dec on celestial_sources(dec_deg)")
        dst.commit()
        return n
    finally:
        src.close(); dst.close()


def query_sky(index_db, ra_deg, dec_deg, radius_deg, modalities=None, limit=2000):
    """Catalogued celestial sources within `radius_deg` of (ra, dec) from a celestial sky-index.

    Bounding-box on the indexed dec, then exact angular-separation refinement (handles RA wraparound).
    Returns dicts: record_id, modality, observatory, ra_deg, dec_deg, sep_deg, sorted by separation.
    """
    if not os.path.exists(index_db):
        raise FileNotFoundError(index_db)
    lo_dec, hi_dec = dec_deg - radius_deg, dec_deg + radius_deg
    mod_set = set(modalities) if modalities else None
    con = sqlite3.connect("file:" + index_db + "?mode=ro", uri=True)
    try:
        cur = con.execute("select record_id, modality, observatory, ra_deg, dec_deg "
                          "from celestial_sources where dec_deg >= ? and dec_deg <= ?",
                          [lo_dec, hi_dec])
        out = []
        for rid, mod, obs, ra, dec in cur:
            if mod_set and mod not in mod_set:
                continue
            sep = _angsep_deg(ra_deg, dec_deg, ra, dec)
            if sep <= radius_deg:
                out.append({"record_id": rid, "modality": mod, "observatory": obs,
                            "ra_deg": ra, "dec_deg": dec, "sep_deg": sep})
        out.sort(key=lambda s: s["sep_deg"])
        return out[:limit]
    finally:
        con.close()


def crossmatch_localization(localization, index_db, ecliptic=True, radius_scale=1.0, min_radius=0.5):
    """Cross-match an Ariadne Stage-28 localization (with sky_box) against a celestial sky-index.

    Returns the search direction (RA/Dec), the sources found, and a by-modality tally --
    "what known catalogued celestial sources sit in our gravitational search-box?"
    """
    lon, lat = localization["ecliptic_lon_deg"], localization["ecliptic_lat_deg"]
    ra, dec = ecliptic_to_equatorial(lon, lat) if ecliptic else (lon, lat)
    radius = max(localization.get("angular_sigma_deg", 1.0) * radius_scale, min_radius)
    sources = query_sky(index_db, ra, dec, radius)
    tally = {}
    for s in sources:
        tally[s["modality"]] = tally.get(s["modality"], 0) + 1
    return {"ra_deg": ra, "dec_deg": dec, "radius_deg": radius,
            "n_sources": len(sources), "by_modality": tally, "sources": sources}
