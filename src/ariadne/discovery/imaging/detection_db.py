"""Persistent multi-night detection / tracklet / chain database.

PanSTARRS / MOPS's core advantage over a per-run pipeline is that it
accumulates detections across the survey lifetime. When tonight's
exposures land, the system DOESN'T just look at tonight's tracklets in
isolation -- it queries the running catalog for tracklets from the past
N nights and tries to extend chains forward, AND queries the MPC catalog
to skip known objects.

This module is the persistent-catalog foundation. SQLite for the schema
(works locally without a database server), R-Tree extension for fast
spatial cone queries. Can migrate to Postgres+PostGIS later for scale.

Schema:

  detections
    every per-exposure point-source detection (after Gaia refinement)
    indexed on (mjd, ra, dec) for cone-in-time queries

  tracklets
    every within-night two-detection pair (rate vector estimated)
    indexed on (mean_mjd, mean_ra, mean_dec) and chain_id

  chains
    every multi-night link of tracklets (a candidate moving object)
    plus IOD state when we have one

  known_objects
    cache of MPC ephemerides for cross-matching incoming detections

Public API:
  open_db(path) -> DetectionDB
  close_db(db)
  DetectionDB.insert_detections(rows)        bulk insert
  DetectionDB.insert_tracklet(row)
  DetectionDB.insert_chain(row)
  DetectionDB.update_chain(chain_id, fields)
  DetectionDB.query_detections_by_cone(mjd_range, ra_dec_box) -> list[dict]
  DetectionDB.query_tracklets_by_window(mjd_start, mjd_end, ra_dec_box) -> list[dict]
  DetectionDB.query_open_chains(min_arc_hours=0) -> list[dict]
  DetectionDB.stats() -> dict   diagnostic counts

The database is self-contained: a single SQLite file. Move it
between machines as a file. Backup with `sqlite3 .backup`.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = 1


SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id TEXT NOT NULL,
        ccd_id TEXT,
        mjd REAL NOT NULL,
        ra REAL NOT NULL,
        dec REAL NOT NULL,
        mag REAL,
        flux REAL,
        fwhm_px REAL,
        x_pix REAL,
        y_pix REAL,
        astrom_sigma_arcsec REAL,
        chain_id INTEGER,
        tracklet_id INTEGER,
        status TEXT DEFAULT 'raw',
        known_designation TEXT,
        created_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_det_mjd ON detections (mjd)",
    "CREATE INDEX IF NOT EXISTS idx_det_radec ON detections (ra, dec)",
    "CREATE INDEX IF NOT EXISTS idx_det_image ON detections (image_id)",
    "CREATE INDEX IF NOT EXISTS idx_det_chain ON detections (chain_id)",
    "CREATE INDEX IF NOT EXISTS idx_det_status ON detections (status)",
    """
    CREATE TABLE IF NOT EXISTS tracklets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_a_id INTEGER NOT NULL,
        detection_b_id INTEGER NOT NULL,
        mean_mjd REAL NOT NULL,
        mean_ra REAL NOT NULL,
        mean_dec REAL NOT NULL,
        rate_arcsec_hr REAL NOT NULL,
        pa_deg REAL NOT NULL,
        rate_sigma REAL,
        mag REAL,
        chain_id INTEGER,
        night INTEGER NOT NULL,
        created_at REAL NOT NULL,
        FOREIGN KEY (detection_a_id) REFERENCES detections(id),
        FOREIGN KEY (detection_b_id) REFERENCES detections(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trk_mjd ON tracklets (mean_mjd)",
    "CREATE INDEX IF NOT EXISTS idx_trk_radec ON tracklets (mean_ra, mean_dec)",
    "CREATE INDEX IF NOT EXISTS idx_trk_night ON tracklets (night)",
    "CREATE INDEX IF NOT EXISTS idx_trk_chain ON tracklets (chain_id)",
    """
    CREATE TABLE IF NOT EXISTS chains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        status TEXT DEFAULT 'open',
        n_tracklets INTEGER NOT NULL DEFAULT 0,
        n_detections INTEGER NOT NULL DEFAULT 0,
        first_mjd REAL,
        last_mjd REAL,
        arc_days REAL,
        mean_ra REAL,
        mean_dec REAL,
        mean_rate_arcsec_hr REAL,
        mean_pa_deg REAL,
        iod_strategy TEXT,
        iod_rms_arcsec REAL,
        iod_x_km_x REAL, iod_x_km_y REAL, iod_x_km_z REAL,
        iod_v_kms_x REAL, iod_v_kms_y REAL, iod_v_kms_z REAL,
        iod_t_ref REAL,
        known_designation TEXT,
        quality_grade TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chain_status ON chains (status)",
    "CREATE INDEX IF NOT EXISTS idx_chain_lastmjd ON chains (last_mjd)",
    """
    CREATE TABLE IF NOT EXISTS known_objects (
        designation TEXT PRIMARY KEY,
        epoch_mjd REAL,
        ra_at_epoch REAL,
        dec_at_epoch REAL,
        rate_arcsec_hr REAL,
        pa_deg REAL,
        orbital_elements TEXT,
        last_observed_mjd REAL,
        catalog TEXT DEFAULT 'mpc',
        created_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_known_radec ON known_objects (ra_at_epoch, dec_at_epoch)",
    "CREATE INDEX IF NOT EXISTS idx_known_epoch ON known_objects (epoch_mjd)",
]


@dataclass
class DetectionRow:
    """One detection row for bulk insert. Times in MJD UTC, angles in deg."""
    image_id: str
    mjd: float
    ra: float
    dec: float
    mag: float = -99.0
    flux: float = 0.0
    fwhm_px: float = 0.0
    x_pix: float = 0.0
    y_pix: float = 0.0
    astrom_sigma_arcsec: float = 0.1
    ccd_id: str = ""
    status: str = "raw"


@dataclass
class TrackletRow:
    """One within-night two-detection pair."""
    detection_a_id: int
    detection_b_id: int
    mean_mjd: float
    mean_ra: float
    mean_dec: float
    rate_arcsec_hr: float
    pa_deg: float
    night: int
    rate_sigma: float = 0.0
    mag: float = -99.0


@dataclass
class ChainRow:
    """A multi-night candidate moving object."""
    n_tracklets: int = 0
    n_detections: int = 0
    first_mjd: float = 0.0
    last_mjd: float = 0.0
    arc_days: float = 0.0
    mean_ra: float = 0.0
    mean_dec: float = 0.0
    mean_rate_arcsec_hr: float = 0.0
    mean_pa_deg: float = 0.0
    iod_strategy: str = ""
    iod_rms_arcsec: float = float("inf")
    iod_state: tuple[float, ...] = ()   # (x, y, z, vx, vy, vz)
    iod_t_ref: float = 0.0
    status: str = "open"
    known_designation: str = ""
    quality_grade: str = ""


class DetectionDB:
    """Wrapper around a SQLite connection with bulk-insert + query helpers."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")

    # ---------------------- DDL --------------------------------------
    def init_schema(self) -> None:
        for stmt in SCHEMA_SQL:
            self.conn.execute(stmt)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)))
        self.conn.commit()

    # ---------------------- Inserts ----------------------------------
    def insert_detections(self, rows: Iterable[DetectionRow]) -> list[int]:
        """Bulk insert detections; return their assigned ids."""
        now = time.time()
        cursor = self.conn.cursor()
        ids = []
        for r in rows:
            cursor.execute(
                """INSERT INTO detections (image_id, ccd_id, mjd, ra, dec, mag,
                    flux, fwhm_px, x_pix, y_pix, astrom_sigma_arcsec, status,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r.image_id, r.ccd_id, r.mjd, r.ra, r.dec, r.mag,
                  r.flux, r.fwhm_px, r.x_pix, r.y_pix,
                  r.astrom_sigma_arcsec, r.status, now))
            ids.append(cursor.lastrowid)
        self.conn.commit()
        return ids

    def insert_tracklet(self, row: TrackletRow) -> int:
        now = time.time()
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO tracklets (detection_a_id, detection_b_id,
                mean_mjd, mean_ra, mean_dec, rate_arcsec_hr, pa_deg,
                rate_sigma, mag, night, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.detection_a_id, row.detection_b_id,
              row.mean_mjd, row.mean_ra, row.mean_dec,
              row.rate_arcsec_hr, row.pa_deg, row.rate_sigma,
              row.mag, row.night, now))
        tid = cur.lastrowid
        # Update the detections to point at this tracklet
        cur.execute(
            "UPDATE detections SET tracklet_id = ? WHERE id IN (?, ?)",
            (tid, row.detection_a_id, row.detection_b_id))
        self.conn.commit()
        return tid

    def insert_chain(self, row: ChainRow) -> int:
        now = time.time()
        iod_state = list(row.iod_state) + [0.0] * (6 - len(row.iod_state))
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO chains (status, n_tracklets, n_detections,
                first_mjd, last_mjd, arc_days, mean_ra, mean_dec,
                mean_rate_arcsec_hr, mean_pa_deg,
                iod_strategy, iod_rms_arcsec,
                iod_x_km_x, iod_x_km_y, iod_x_km_z,
                iod_v_kms_x, iod_v_kms_y, iod_v_kms_z,
                iod_t_ref, known_designation, quality_grade,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.status, row.n_tracklets, row.n_detections,
              row.first_mjd, row.last_mjd, row.arc_days,
              row.mean_ra, row.mean_dec,
              row.mean_rate_arcsec_hr, row.mean_pa_deg,
              row.iod_strategy, row.iod_rms_arcsec,
              iod_state[0], iod_state[1], iod_state[2],
              iod_state[3], iod_state[4], iod_state[5],
              row.iod_t_ref, row.known_designation, row.quality_grade,
              now, now))
        cid = cur.lastrowid
        self.conn.commit()
        return cid

    def update_chain(self, chain_id: int, fields: dict) -> None:
        """Update fields on an existing chain. Use this when appending a
        new tracklet extends the arc, or when IOD produces a new fit."""
        if not fields:
            return
        keys = list(fields.keys())
        sets = ", ".join(f"{k} = ?" for k in keys)
        values = [fields[k] for k in keys]
        values.append(time.time())
        values.append(chain_id)
        self.conn.execute(
            f"UPDATE chains SET {sets}, updated_at = ? WHERE id = ?",
            values)
        self.conn.commit()

    def attach_tracklet_to_chain(self, tracklet_id: int, chain_id: int) -> None:
        """Mark a tracklet as belonging to a chain (and propagate to its
        member detections)."""
        cur = self.conn.cursor()
        cur.execute("UPDATE tracklets SET chain_id = ? WHERE id = ?",
                       (chain_id, tracklet_id))
        cur.execute(
            """UPDATE detections SET chain_id = ?
               WHERE tracklet_id = ?""", (chain_id, tracklet_id))
        self.conn.commit()

    # ---------------------- Queries ----------------------------------
    def query_detections_by_cone(self,
                                    mjd_range: tuple[float, float] | None,
                                    ra_range: tuple[float, float] | None,
                                    dec_range: tuple[float, float] | None,
                                    *, limit: int = 100000
                                    ) -> list[dict]:
        clauses = []
        params: list = []
        if mjd_range:
            clauses.append("mjd >= ? AND mjd <= ?")
            params.extend(mjd_range)
        if ra_range:
            clauses.append("ra >= ? AND ra <= ?")
            params.extend(ra_range)
        if dec_range:
            clauses.append("dec >= ? AND dec <= ?")
            params.extend(dec_range)
        sql = "SELECT * FROM detections"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY mjd ASC LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def query_tracklets_by_window(self,
                                      mjd_start: float, mjd_end: float,
                                      ra_range: tuple[float, float] | None = None,
                                      dec_range: tuple[float, float] | None = None,
                                      *, limit: int = 100000) -> list[dict]:
        clauses = ["mean_mjd >= ? AND mean_mjd <= ?"]
        params: list = [mjd_start, mjd_end]
        if ra_range:
            clauses.append("mean_ra >= ? AND mean_ra <= ?")
            params.extend(ra_range)
        if dec_range:
            clauses.append("mean_dec >= ? AND mean_dec <= ?")
            params.extend(dec_range)
        sql = ("SELECT * FROM tracklets WHERE " + " AND ".join(clauses)
                 + f" ORDER BY mean_mjd ASC LIMIT {int(limit)}")
        return [dict(r) for r in self.conn.execute(sql, params)]

    def query_open_chains(self, min_arc_hours: float = 0,
                            max_chains: int = 10000) -> list[dict]:
        sql = ("SELECT * FROM chains WHERE status = 'open' "
                 "AND arc_days >= ? "
                 "ORDER BY last_mjd DESC LIMIT ?")
        params = (min_arc_hours / 24.0, max_chains)
        return [dict(r) for r in self.conn.execute(sql, params)]

    def get_chain(self, chain_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM chains WHERE id = ?", (chain_id,)).fetchone()
        return dict(row) if row else None

    def get_tracklet_detections(self, tracklet_id: int) -> list[dict]:
        sql = """SELECT d.* FROM detections d
                 INNER JOIN tracklets t
                 ON d.id IN (t.detection_a_id, t.detection_b_id)
                 WHERE t.id = ?
                 ORDER BY d.mjd ASC"""
        return [dict(r) for r in self.conn.execute(sql, (tracklet_id,))]

    def get_chain_tracklets(self, chain_id: int) -> list[dict]:
        sql = ("SELECT * FROM tracklets WHERE chain_id = ? "
                 "ORDER BY mean_mjd ASC")
        return [dict(r) for r in self.conn.execute(sql, (chain_id,))]

    def get_chain_detections(self, chain_id: int) -> list[dict]:
        sql = ("SELECT * FROM detections WHERE chain_id = ? "
                 "ORDER BY mjd ASC")
        return [dict(r) for r in self.conn.execute(sql, (chain_id,))]

    # ---------------------- Stats ------------------------------------
    def stats(self) -> dict:
        out = {}
        for table in ("detections", "tracklets", "chains", "known_objects"):
            row = self.conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[f"n_{table}"] = int(row["n"]) if row else 0
        # MJD coverage
        row = self.conn.execute(
            "SELECT MIN(mjd) AS lo, MAX(mjd) AS hi FROM detections"
        ).fetchone()
        if row and row["lo"] is not None:
            out["mjd_min"] = float(row["lo"])
            out["mjd_max"] = float(row["hi"])
            out["mjd_span_days"] = out["mjd_max"] - out["mjd_min"]
        # Chain status counts
        for status in ("open", "closed", "submitted", "known"):
            r = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chains WHERE status = ?",
                (status,)).fetchone()
            out[f"n_chains_{status}"] = int(r["n"]) if r else 0
        return out


def open_db(path: str | Path,
              *, init_schema: bool = True) -> DetectionDB:
    """Open (or create) a DetectionDB at `path`. Initialises the schema
    if it's not already present.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None,
                              check_same_thread=False)
    db = DetectionDB(conn)
    if init_schema:
        db.init_schema()
    return db


def close_db(db: DetectionDB) -> None:
    db.conn.close()
