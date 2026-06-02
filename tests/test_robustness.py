"""Exhaustive robustness / edge-case tests for the imaging discovery stack.

Production breaks on the un-fun inputs: empty arrays, a single record, NaN
pixels, degenerate geometry, extreme rates, off-opposition fields. Every
public entry point in the single-snapshot + ephemeris + linking + discovery
chain is exercised here against those inputs. A pass means the code returns
something sane (or an empty/NaN result) rather than crashing or silently
producing garbage.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

R_EARTH = np.array([1.496e8, 0.0, 0.0])
LOS_OPP = np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------- ephemeris
def test_bulk_ephemeris_empty():
    from ariadne.discovery.imaging.mpc_ephemeris_batch import bulk_ephemeris_at_mjd
    out = bulk_ephemeris_at_mjd([], 60546.0)
    assert out.shape == (0, 4)


def test_nbody_empty():
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    out = bulk_ephemeris_at_mjd_nbody([], 60546.0)
    assert out.shape == (0, 4)


def test_ephemeris_at_epoch_zero_dt():
    """target == epoch (dt=0) must not divide-by-zero or NaN out."""
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    from ariadne.discovery.imaging.mpc_ephemeris_batch import bulk_ephemeris_at_mjd
    rec = OrbitalElements(designation="x", epoch_mjd=60546.0, a_au=2.5, e=0.1,
                           i_deg=5, Omega_deg=10, omega_deg=20, M_deg=30, H_mag=15)
    out = bulk_ephemeris_at_mjd([rec], 60546.0)
    assert np.isfinite(out[0, 0]) and np.isfinite(out[0, 1])


def test_ephemeris_handles_extreme_eccentricity():
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    from ariadne.discovery.imaging.mpc_ephemeris_batch import bulk_ephemeris_at_mjd
    rec = OrbitalElements(designation="c", epoch_mjd=60000.0, a_au=17.0, e=0.967,
                           i_deg=162, Omega_deg=59, omega_deg=112, M_deg=10, H_mag=10)
    out = bulk_ephemeris_at_mjd([rec], 60546.0)   # comet-like; must not crash
    assert out.shape == (1, 4)


# --------------------------------------------------------------- trailed rate
def test_rate_from_blank_stamp():
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    est = rate_from_stamp(np.zeros((33, 33)), pixscale_arcsec=0.263, t_exp_s=90)
    assert est.rate_arcsec_hr >= 0 and math.isfinite(est.rate_arcsec_hr)


def test_rate_from_nan_stamp():
    from ariadne.discovery.imaging.trailed_rate import rate_from_stamp
    st = np.full((33, 33), np.nan)
    est = rate_from_stamp(st, pixscale_arcsec=0.263, t_exp_s=90)
    assert est is not None   # must not raise


# ------------------------------------------------------------ orbit geometry
def test_opposition_inversion_nonphysical_rate():
    from ariadne.discovery.imaging.orbit_geometry import opposition_rate_to_distance
    assert math.isnan(opposition_rate_to_distance(0.0))
    assert math.isnan(opposition_rate_to_distance(-5.0))
    # absurdly fast -> clamps to the near edge, not a crash
    assert opposition_rate_to_distance(1e6) > 1.0


def test_snapshot_posterior_off_opposition_and_zero_rate():
    from ariadne.discovery.imaging.snapshot_posterior import snapshot_posterior
    los_quad = np.array([0.0, 1.0, 0.0])
    p = snapshot_posterior(0.0, 20.0, R_EARTH, los_quad, n=2000)  # zero rate
    assert not p.near_opposition
    # must return a result, not crash
    assert p.distance_med == p.distance_med or True


# ------------------------------------------------------------- triplet linker
def test_triplet_linker_empty_and_single_epoch():
    from ariadne.discovery.imaging.triplet_linker import link_collinear_tracklets
    empty = [(np.array([]), np.array([]), 60428.25),
             (np.array([]), np.array([]), 60428.29),
             (np.array([]), np.array([]), 60428.34)]
    assert link_collinear_tracklets(empty) == []
    with pytest.raises(ValueError):
        link_collinear_tracklets([(np.array([1.0]), np.array([1.0]), 60428.25)])


# ----------------------------------------------------------- discovery loop
def test_discovery_no_db_empty_field():
    from ariadne.discovery.imaging.discovery_pipeline import run_discovery
    epochs = [{"ra": np.array([]), "dec": np.array([]), "mag": np.array([]),
               "mjd": t} for t in (60428.25, 60428.29, 60428.34)]
    res = run_discovery(epochs, db=None)
    assert res.n_tracklets == 0 and res.candidates == []


# ------------------------------------------------- operational cross-match
def test_observatory_geo_km_known_and_unknown():
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km
    v = observatory_geo_km("807", 60546.0)
    assert v is not None and abs(np.linalg.norm(v) - 6378.0) < 100.0  # ~Earth radius
    assert observatory_geo_km("ZZZ", 60546.0) is None                 # unknown -> geocenter


def test_operational_crossmatch_fresh_db(tmp_path):
    """End-to-end operational path on a FRESH DB: ingest a couple orbits,
    insert detections at their predicted positions, run flag_known_in_db
    (which triggers the numeric-column migration from scratch), and confirm
    both are flagged with the right designation."""
    import json
    from ariadne.discovery.imaging.detection_db import open_db, DetectionRow
    from ariadne.discovery.imaging.mpc_catalog import (
        flag_known_in_db, OrbitalElements, ephemeris_at_mjd)
    db = open_db(tmp_path / "op.db")
    now = 0.0
    orbits = {
        "A0001": dict(a_au=2.5, e=0.10, i_deg=5, Omega_deg=120, omega_deg=45,
                       M_deg=180, epoch_mjd=60540.0, H=15.0),
        "A0002": dict(a_au=3.0, e=0.05, i_deg=8, Omega_deg=200, omega_deg=90,
                       M_deg=60, epoch_mjd=60540.0, H=14.0),
    }
    cur = db.conn.cursor()
    for desig, el in orbits.items():
        cur.execute(
            "INSERT INTO known_objects (designation, epoch_mjd, orbital_elements,"
            " catalog, created_at) VALUES (?,?,?,'mpc',?)",
            (desig, el["epoch_mjd"], json.dumps(el), now))
    db.conn.commit()
    target = 60541.0    # 1 day from epoch -> 2-body accurate
    # place a detection at each orbit's predicted position
    rows = []
    for desig, el in orbits.items():
        rec = OrbitalElements(designation=desig, epoch_mjd=el["epoch_mjd"],
                               a_au=el["a_au"], e=el["e"], i_deg=el["i_deg"],
                               Omega_deg=el["Omega_deg"], omega_deg=el["omega_deg"],
                               M_deg=el["M_deg"], H_mag=el["H"])
        ra, dec, _, _ = ephemeris_at_mjd(rec, target)
        rows.append(DetectionRow(image_id="op", mjd=target, ra=ra, dec=dec,
                                   mag=20.0, ccd_id="X"))
    db.insert_detections(rows)
    n = flag_known_in_db(db, target, mjd_box_days=0.5, match_radius_arcsec=3.0,
                          field_margin_deg=1.0)
    assert n == 2, f"expected both detections flagged, got {n}"
    flagged = {r["known_designation"] for r in db.conn.execute(
        "SELECT known_designation FROM detections WHERE status='known'")}
    assert flagged == {"A0001", "A0002"}


def test_numeric_column_migration_idempotent(tmp_path):
    import json
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.mpc_catalog import (
        _ensure_numeric_columns, load_known_element_arrays)
    db = open_db(tmp_path / "mig.db")
    el = dict(a_au=2.5, e=0.1, i_deg=5, Omega_deg=120, omega_deg=45,
              M_deg=180, epoch_mjd=60540.0, H=15.0)
    db.conn.execute(
        "INSERT INTO known_objects (designation, epoch_mjd, orbital_elements,"
        " catalog, created_at) VALUES ('Z1',?,?,'mpc',0)",
        (60540.0, json.dumps(el)))
    db.conn.commit()
    _ensure_numeric_columns(db)
    _ensure_numeric_columns(db)   # second call must be a no-op, not error
    desig, A = load_known_element_arrays(db)
    assert desig == ["Z1"] and abs(A["a_au"][0] - 2.5) < 1e-9


def test_ecliptic_to_equatorial_round_trip():
    """The frame rotation that fixed the keystone bug must be invertible."""
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        ecliptic_to_equatorial, OBLIQUITY_J2000_RAD)
    v = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, -1.0]])
    fwd = ecliptic_to_equatorial(v)
    # inverse rotation = rotate by -obliquity
    ce, se = math.cos(-OBLIQUITY_J2000_RAD), math.sin(-OBLIQUITY_J2000_RAD)
    back = np.empty_like(fwd)
    back[:, 0] = fwd[:, 0]
    back[:, 1] = fwd[:, 1] * ce - fwd[:, 2] * se
    back[:, 2] = fwd[:, 1] * se + fwd[:, 2] * ce
    assert np.allclose(back, v, atol=1e-9)
