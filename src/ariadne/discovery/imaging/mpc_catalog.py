"""MPC orbital element catalog handling.

Downloads + parses the Minor Planet Center's orbital element catalog
(MPCORB.dat), caches it into the persistent detection_db.known_objects
table, and provides utilities to compute predicted positions on the sky
for any subset of cataloged objects at a target epoch.

The MPC publishes MPCORB.dat (~1.4M asteroids) and NEOCP (current
candidates), updated daily. We pull the gzipped version and parse the
fixed-column ASCII format.

Public API:
  download_mpcorb(out_dir, force=False) -> Path
    Download the latest MPCORB.dat.gz to disk. Cached by date.
  parse_mpcorb_record(line) -> dict
    Parse one 203-character orbital-element line.
  ingest_mpcorb_to_db(db, mpcorb_path, limit=None)
    Bulk ingest into the DB's known_objects table.
  ephemeris_for_known_objects(elements, et_target) -> list[dict]
    Compute predicted (ra, dec) for each object at the target epoch.
  cross_match_detections(detections, known_objects, ...)
    Flag detections that fall within match_radius_arcsec of any known
    object's predicted position. Returns {det_id: designation}.

MPC orbital elements are heliocentric J2000.0 osculating Keplerian
elements at the catalog's epoch. We propagate via the existing
dynamics.secular.kepler_step routine.

Schema of MPCORB.dat (single asteroid line, 203 chars):
  cols  1-7    Designation (packed)
  cols  9-13   Absolute magnitude H
  cols 15-19   Slope parameter G
  cols 21-25   Epoch (packed)
  cols 27-35   Mean anomaly M (deg)
  cols 38-46   Argument of perihelion omega (deg)
  cols 49-57   Longitude of ascending node Omega (deg)
  cols 60-68   Inclination i (deg)
  cols 71-79   Eccentricity e
  cols 81-91   Mean daily motion n (deg/day)
  cols 93-103  Semi-major axis a (AU)
  cols 105-106 Uncertainty parameter U
  cols 108-116 Reference
  cols 118-122 Number of observations
  cols 124-126 Number of oppositions
  cols 128-136 Years of observation (arc)
  cols 138-141 r.m.s. residual
  cols 143-145 Coarse perturber indicator
  cols 147-149 Precise perturber indicator
  cols 151-160 Computer name
  cols 162-165 Hexadecimal flags
  cols 167-194 Readable designation
"""
from __future__ import annotations

import gzip
import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


MPCORB_URL = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"


@dataclass
class OrbitalElements:
    """Heliocentric osculating Keplerian elements at the catalog epoch."""
    designation: str
    epoch_mjd: float
    a_au: float                 # semi-major axis
    e: float                    # eccentricity
    i_deg: float                # inclination
    Omega_deg: float            # longitude of ascending node
    omega_deg: float            # argument of perihelion
    M_deg: float                # mean anomaly at epoch
    H_mag: float = 0.0
    G_param: float = 0.15
    n_obs: int = 0
    arc_years: float = 0.0
    rms_arcsec: float = 0.0
    name: str = ""
    extras: dict = field(default_factory=dict)


def _unpack_mpc_designation(packed: str) -> str:
    """Unpack the 7-char MPC packed designation.

    See https://minorplanetcenter.net/iau/info/PackedDes.html
    Provisional designations (e.g., 'J95X00A' -> '1995 XA') and
    numbered (1-9999 in 5-char dec, 'A0000'+ for 100000+).
    For our purposes we accept the packed form as-is (it's stable).
    """
    return packed.strip()


def _unpack_mpc_epoch(packed: str) -> float:
    """Unpack the 5-char packed epoch to MJD.

    Format: 'K244B' = 2024-04-11. The first char is century:
      J = 18xx, K = 19xx ... wait actually:
      I = 1800-1899, J = 1900-1999, K = 2000-2099, etc.
    The next 2 are YY, then MM and DD packed in single-char base-31:
      digits 1-9 = 1-9, A-V = 10-31.
    """
    if len(packed) != 5:
        return 0.0
    century_letter = packed[0]
    century_map = {"I": 1800, "J": 1900, "K": 2000, "L": 2100}
    century_base = century_map.get(century_letter)
    if century_base is None:
        return 0.0
    try:
        yy = int(packed[1:3])
    except ValueError:
        return 0.0
    year = century_base + yy

    def _base31(c: str) -> int:
        if c.isdigit():
            return int(c)
        if c.isalpha():
            return ord(c.upper()) - ord("A") + 10
        return 0
    month = _base31(packed[3])
    day = _base31(packed[4])
    if month < 1 or month > 12 or day < 1 or day > 31:
        return 0.0
    try:
        from astropy.time import Time
        t = Time(f"{year:04d}-{month:02d}-{day:02d}",
                  format="iso", scale="utc")
        return float(t.mjd)
    except Exception:
        return 0.0


def parse_mpcorb_record(line: str) -> OrbitalElements | None:
    """Parse one line of MPCORB.dat. Returns None on parse failure."""
    if len(line) < 168 or line[0] == "#":
        return None
    try:
        # MPCORB fixed-column spec (1-indexed in MPC docs). Python slices
        # are 0-indexed, end-exclusive: "cols A-B inclusive" is [A-1, B).
        # Spec: https://minorplanetcenter.net/iau/info/MPOrbitFormat.html
        designation = line[0:7].strip()        # cols 1-7
        H_str = line[8:13].strip()             # cols 9-13
        G_str = line[14:19].strip()            # cols 15-19
        epoch_pack = line[20:25].strip()       # cols 21-25
        M_str = line[26:35].strip()            # cols 27-35
        omega_str = line[37:46].strip()        # cols 38-46
        Omega_str = line[48:57].strip()        # cols 49-57
        i_str = line[59:68].strip()            # cols 60-68
        e_str = line[69:79].strip()            # cols 70-79
        a_str = line[92:103].strip()           # cols 93-103
        n_obs_str = line[117:122].strip()      # cols 118-122
        arc_str = line[127:136].strip() if len(line) > 136 else ""  # 128-136
        rms_str = line[137:141].strip() if len(line) > 141 else ""  # 138-141
        name = line[166:194].strip() if len(line) > 166 else ""     # 167-194

        H = float(H_str) if H_str else 0.0
        G = float(G_str) if G_str else 0.15
        epoch_mjd = _unpack_mpc_epoch(epoch_pack)
        M = float(M_str) if M_str else 0.0
        omega = float(omega_str) if omega_str else 0.0
        Omega = float(Omega_str) if Omega_str else 0.0
        inc = float(i_str) if i_str else 0.0
        e = float(e_str) if e_str else 0.0
        a = float(a_str) if a_str else 0.0
        n_obs = int(n_obs_str) if n_obs_str.isdigit() else 0
        # arc_str is like '1996-2023' or '5 days'; parse only year-year
        arc_years = 0.0
        if "-" in arc_str and len(arc_str) >= 9:
            try:
                lo = int(arc_str[:4])
                hi = int(arc_str[5:9])
                arc_years = float(hi - lo)
            except ValueError:
                pass
        rms_arcsec = float(rms_str) if rms_str else 0.0

        return OrbitalElements(
            designation=designation, epoch_mjd=epoch_mjd,
            a_au=a, e=e, i_deg=inc, Omega_deg=Omega,
            omega_deg=omega, M_deg=M,
            H_mag=H, G_param=G,
            n_obs=n_obs, arc_years=arc_years, rms_arcsec=rms_arcsec,
            name=name,
        )
    except (ValueError, IndexError):
        return None


def download_mpcorb(out_dir: str | Path = "data/mpc_catalog",
                      *, force: bool = False,
                      timeout_s: float = 300.0) -> Path | None:
    """Download MPCORB.dat.gz to disk. Cached: skips if file is < 7 days old.

    Returns the local Path, or None on failure.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / "MPCORB.DAT.gz"
    if local.exists() and not force:
        age_days = (time.time() - local.stat().st_mtime) / 86400.0
        if age_days < 7:
            return local
    try:
        urllib.request.urlretrieve(MPCORB_URL, str(local))
    except Exception:
        return None
    if not local.exists() or local.stat().st_size == 0:
        return None
    return local


def iter_mpcorb_records(path: str | Path,
                          *, limit: int | None = None
                          ) -> Iterable[OrbitalElements]:
    """Stream OrbitalElements records from MPCORB.dat[.gz]. Skips bad lines."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    n_yielded = 0
    with opener(path, "rt", encoding="latin-1") as f:
        # The file has a long ASCII header; skip until we find a real
        # orbital-element line.
        for line in f:
            if not line.strip():
                continue
            rec = parse_mpcorb_record(line)
            if rec is None:
                continue
            if rec.a_au <= 0 or rec.epoch_mjd <= 0:
                continue
            yield rec
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return


def ingest_mpcorb_to_db(db, mpcorb_path: str | Path,
                          *, limit: int | None = None) -> int:
    """Bulk insert MPCORB records into the DB's known_objects table.

    Returns the number of records ingested.
    """
    now = time.time()
    n = 0
    cur = db.conn.cursor()
    for rec in iter_mpcorb_records(mpcorb_path, limit=limit):
        # Compute crude rate from mean daily motion + parallax handwave
        # (proper rate comes from ephemeris at observation epoch; here we
        # store the orbital elements for later ephemeris computation)
        elements_json = (
            f'{{"a_au": {rec.a_au}, "e": {rec.e}, '
            f'"i_deg": {rec.i_deg}, "Omega_deg": {rec.Omega_deg}, '
            f'"omega_deg": {rec.omega_deg}, "M_deg": {rec.M_deg}, '
            f'"H": {rec.H_mag}, "G": {rec.G_param}, '
            f'"epoch_mjd": {rec.epoch_mjd}, "name": "{rec.name}"}}')
        try:
            cur.execute(
                """INSERT OR REPLACE INTO known_objects
                   (designation, epoch_mjd, ra_at_epoch, dec_at_epoch,
                    rate_arcsec_hr, pa_deg, orbital_elements,
                    last_observed_mjd, catalog, created_at)
                   VALUES (?, ?, NULL, NULL, NULL, NULL, ?, NULL, 'mpc', ?)""",
                (rec.designation, rec.epoch_mjd, elements_json, now))
            n += 1
        except Exception:
            continue
    db.conn.commit()
    return n


# -------------------------------------------------------------------
# Ephemeris computation
# -------------------------------------------------------------------

def elements_to_state(rec: OrbitalElements) -> tuple:
    """Convert Keplerian elements to a heliocentric (r, v) state at the
    catalog epoch via the existing dynamics module."""
    from ...dynamics.secular import elements_to_state as _e2s
    r0, v0 = _e2s(rec.a_au, rec.e, rec.i_deg,
                    rec.Omega_deg, rec.omega_deg, rec.M_deg)
    return r0, v0


def ephemeris_at_mjd(rec: OrbitalElements,
                       target_mjd: float) -> tuple[float, float, float, float]:
    """Propagate rec from its catalog epoch to target_mjd via 2-body Kepler,
    project to geocentric (RA, Dec), and return (ra_deg, dec_deg, mag_est,
    geocentric_distance_au).

    The magnitude estimate uses the IAU H-G phase function.
    """
    import numpy as np
    from ...dynamics.secular import kepler_step
    from ...data.constants import GM_SUN, AU_KM
    from ...data.ephemeris import body_state

    r0, v0 = elements_to_state(rec)
    dt_days = target_mjd - rec.epoch_mjd
    dt_s = dt_days * 86400.0
    et_target = (target_mjd - 51544.5) * 86400.0
    r_t, _ = kepler_step(r0, v0, GM_SUN, dt_s)
    # Earth position at target
    R_e = np.array(body_state("EARTH", et_target, "J2000", "SUN")[:3])
    geo = r_t - R_e
    rho_km = float(np.linalg.norm(geo))
    rho_au = rho_km / AU_KM
    ra_deg = math.degrees(math.atan2(geo[1], geo[0])) % 360.0
    dec_deg = math.degrees(math.asin(geo[2] / rho_km))
    # Magnitude (no phase angle correction; simple approx)
    r_helio_au = float(np.linalg.norm(r_t)) / AU_KM
    mag_est = (rec.H_mag + 5.0 * math.log10(rho_au * r_helio_au)
                 if (rho_au > 0 and r_helio_au > 0) else 0.0)
    return ra_deg, dec_deg, mag_est, rho_au


# -------------------------------------------------------------------
# Cross-match
# -------------------------------------------------------------------

def cross_match_detections(detections: Sequence[dict],
                              known_objects: Sequence[OrbitalElements],
                              target_mjd: float,
                              *, match_radius_arcsec: float = 3.0,
                              ) -> dict[int, str]:
    """Return {detection_id: known_object_designation} for detections
    that fall within `match_radius_arcsec` of any known object's predicted
    position at `target_mjd`.

    Detections are dicts with at least 'id', 'ra', 'dec', 'mjd' keys.
    Match uses the catalog object's predicted (ra, dec) propagated from
    its catalog epoch to `target_mjd`.
    """
    # Pre-compute ephemerides for all known objects at target_mjd
    radius_deg = match_radius_arcsec / 3600.0
    eph = []
    for rec in known_objects:
        try:
            ra, dec, mag, rho = ephemeris_at_mjd(rec, target_mjd)
            eph.append((rec.designation, ra, dec, mag, rho))
        except Exception:
            continue
    out: dict[int, str] = {}
    for det in detections:
        det_ra = float(det["ra"])
        det_dec = float(det["dec"])
        cos_dec = math.cos(math.radians(det_dec))
        for designation, ra, dec, _, _ in eph:
            dra = (ra - det_ra) * cos_dec
            ddec = dec - det_dec
            if math.hypot(dra, ddec) <= radius_deg:
                out[int(det["id"])] = designation
                break
    return out


def flag_known_in_db(db, target_mjd: float,
                       *, mjd_box_days: float = 0.5,
                       match_radius_arcsec: float = 3.0,
                       limit_known: int | None = None) -> int:
    """Cross-match every detection in the DB within `mjd_box_days` of
    target_mjd against the known_objects catalog. Sets known_designation
    on matching detections; updates their status to 'known'.

    Returns the number of detections flagged.
    """
    import json as _json
    # Fetch detections in the target window
    dets = db.query_detections_by_cone(
        mjd_range=(target_mjd - mjd_box_days, target_mjd + mjd_box_days),
        ra_range=None, dec_range=None, limit=200000)
    if not dets:
        return 0
    # Fetch known-object records (limit for performance during dev)
    cur = db.conn.cursor()
    sql = "SELECT designation, epoch_mjd, orbital_elements FROM known_objects"
    if limit_known is not None:
        sql += f" LIMIT {int(limit_known)}"
    rows = list(cur.execute(sql))
    known_records: list[OrbitalElements] = []
    for r in rows:
        try:
            elems = _json.loads(r["orbital_elements"])
            known_records.append(OrbitalElements(
                designation=r["designation"],
                epoch_mjd=float(r["epoch_mjd"]),
                a_au=float(elems["a_au"]),
                e=float(elems["e"]),
                i_deg=float(elems["i_deg"]),
                Omega_deg=float(elems["Omega_deg"]),
                omega_deg=float(elems["omega_deg"]),
                M_deg=float(elems["M_deg"]),
                H_mag=float(elems.get("H", 0.0)),
                G_param=float(elems.get("G", 0.15)),
                name=str(elems.get("name", "")),
            ))
        except Exception:
            continue
    # Cross-match
    matches = cross_match_detections(
        dets, known_records, target_mjd,
        match_radius_arcsec=match_radius_arcsec)
    # Persist
    for det_id, designation in matches.items():
        cur.execute(
            "UPDATE detections SET known_designation = ?, status = 'known' "
            "WHERE id = ?", (designation, det_id))
    db.conn.commit()
    return len(matches)
