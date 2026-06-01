"""Tests for the MPC catalog handling."""
from __future__ import annotations

import math
import pytest


def test_unpack_mpc_epoch_k244b():
    """K244B = 2024-04-11. Test the packed epoch parser."""
    from ariadne.discovery.imaging.mpc_catalog import _unpack_mpc_epoch
    mjd = _unpack_mpc_epoch("K244B")
    # 2024-04-11 -> MJD 60411
    assert 60410 < mjd < 60412


def test_unpack_mpc_epoch_j991a():
    """J991A = 1999-01-10 -> MJD 51188."""
    from ariadne.discovery.imaging.mpc_catalog import _unpack_mpc_epoch
    mjd = _unpack_mpc_epoch("J991A")
    assert 51187 < mjd < 51189


def test_unpack_mpc_epoch_bad_input_returns_zero():
    from ariadne.discovery.imaging.mpc_catalog import _unpack_mpc_epoch
    assert _unpack_mpc_epoch("XX") == 0.0
    assert _unpack_mpc_epoch("Z244B") == 0.0   # bad century letter


def _build_mpcorb_line(*, designation="00001", H=3.34, G=0.12,
                          epoch_pack="K244B", M=152.69810,
                          omega=73.43893, Omega=80.25577,
                          i=10.58688, e=0.0789912, a=2.7660452,
                          n_obs=7307, name="(1) Ceres") -> str:
    """Programmatically build a MPCORB-format line with exact column placement."""
    # Use fixed-width formatting per the spec
    s = list(" " * 220)
    def _set(start, end, text):
        # 1-indexed inclusive [start, end]; slice [start-1:end]
        n = end - start + 1
        text = text.rjust(n) if len(text) <= n else text[:n]
        for i, c in enumerate(text):
            s[start - 1 + i] = c

    _set(1, 7, designation)
    _set(9, 13, f"{H:5.2f}")
    _set(15, 19, f"{G:5.2f}")
    _set(21, 25, epoch_pack)
    _set(27, 35, f"{M:9.5f}")
    _set(38, 46, f"{omega:9.5f}")
    _set(49, 57, f"{Omega:9.5f}")
    _set(60, 68, f"{i:9.5f}")
    _set(70, 79, f"{e:9.7f}")
    _set(93, 103, f"{a:11.7f}")
    _set(118, 122, f"{n_obs:5d}")
    _set(128, 136, "1801-2024")
    _set(167, 194, name)
    return "".join(s).rstrip()


def test_parse_mpcorb_record_ceres():
    """Parse a properly-formatted MPCORB line for Ceres."""
    line = _build_mpcorb_line()
    from ariadne.discovery.imaging.mpc_catalog import parse_mpcorb_record
    rec = parse_mpcorb_record(line)
    assert rec is not None
    assert rec.designation == "00001"
    assert rec.a_au == pytest.approx(2.7660452, abs=1e-4)
    assert rec.e == pytest.approx(0.0789912, abs=1e-4)
    assert rec.i_deg == pytest.approx(10.58688, abs=1e-3)
    assert rec.H_mag == pytest.approx(3.34, abs=0.01)


def test_parse_mpcorb_record_returns_none_on_short_line():
    from ariadne.discovery.imaging.mpc_catalog import parse_mpcorb_record
    assert parse_mpcorb_record("") is None
    assert parse_mpcorb_record("# comment") is None
    assert parse_mpcorb_record("too short") is None


def test_iter_mpcorb_records_filters_bad_lines(tmp_path):
    """Write a tiny MPCORB-like file with one good + several bad lines;
    iterator should yield only the good one."""
    fake_path = tmp_path / "mpcorb.dat"
    line = _build_mpcorb_line()
    fake_path.write_text(f"# header\n\n{line}\nbad short line\n",
                            encoding="latin-1")
    from ariadne.discovery.imaging.mpc_catalog import iter_mpcorb_records
    out = list(iter_mpcorb_records(fake_path))
    assert len(out) == 1
    assert out[0].designation.startswith("00001")


def test_elements_to_state_returns_nonzero(tmp_path):
    """Test that a real-ish set of elements gives a non-trivial state."""
    from ariadne.discovery.imaging.mpc_catalog import (
        OrbitalElements, elements_to_state)
    rec = OrbitalElements(
        designation="ceres_test", epoch_mjd=60411.0,
        a_au=2.766, e=0.079, i_deg=10.6,
        Omega_deg=80.3, omega_deg=73.4, M_deg=152.7)
    r0, v0 = elements_to_state(rec)
    # r0 should be in km, magnitude ~ a_au * AU_KM = ~4.1e8 km
    import numpy as np
    r_mag_km = float(np.linalg.norm(r0))
    AU_KM = 149_597_870.7
    assert 1.5 * AU_KM < r_mag_km < 4.0 * AU_KM    # within MBA range


def test_ephemeris_at_mjd_returns_finite():
    """Test that ephemeris computation doesn't NaN or blow up."""
    from ariadne.discovery.imaging.mpc_catalog import (
        OrbitalElements, ephemeris_at_mjd)
    rec = OrbitalElements(
        designation="test", epoch_mjd=60450.0,
        a_au=2.5, e=0.1, i_deg=5.0,
        Omega_deg=120.0, omega_deg=50.0, M_deg=180.0)
    ra, dec, mag, rho = ephemeris_at_mjd(rec, 60450.0)
    assert 0 <= ra < 360
    assert -90 < dec < 90
    assert rho > 0


def test_cross_match_detections_finds_match():
    """A detection at a known object's predicted position should match."""
    from ariadne.discovery.imaging.mpc_catalog import (
        OrbitalElements, ephemeris_at_mjd, cross_match_detections)
    rec = OrbitalElements(
        designation="testobj", epoch_mjd=60450.0,
        a_au=2.5, e=0.1, i_deg=5.0,
        Omega_deg=120.0, omega_deg=50.0, M_deg=180.0)
    ra, dec, _, _ = ephemeris_at_mjd(rec, 60450.5)
    detections = [{"id": 1, "ra": ra, "dec": dec, "mjd": 60450.5}]
    matches = cross_match_detections(
        detections, [rec], 60450.5, match_radius_arcsec=5.0)
    assert matches.get(1) == "testobj"


def test_cross_match_no_match_for_distant_detection():
    """Detection far from any known should not match."""
    from ariadne.discovery.imaging.mpc_catalog import (
        OrbitalElements, cross_match_detections)
    rec = OrbitalElements(
        designation="testobj", epoch_mjd=60450.0,
        a_au=2.5, e=0.1, i_deg=5.0,
        Omega_deg=120.0, omega_deg=50.0, M_deg=180.0)
    detections = [{"id": 1, "ra": 359.0, "dec": -89.5, "mjd": 60450.5}]
    matches = cross_match_detections(
        detections, [rec], 60450.5, match_radius_arcsec=1.0)
    assert matches.get(1) is None


def test_ingest_mpcorb_to_db_persists(tmp_path):
    """Ingest a small MPCORB into the DB and verify counts."""
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.mpc_catalog import ingest_mpcorb_to_db

    fake_mpcorb = tmp_path / "fake.dat"
    line1 = _build_mpcorb_line(designation="00001", name="(1) Ceres")
    line2 = _build_mpcorb_line(designation="00002", H=4.13,
                                  M=100.1, i=34.83, e=0.23, a=2.77,
                                  name="(2) Pallas")
    fake_mpcorb.write_text(f"# header\n{line1}\n{line2}\n",
                              encoding="latin-1")

    db = open_db(tmp_path / "kn.db")
    n = ingest_mpcorb_to_db(db, fake_mpcorb)
    assert n == 2
    assert db.stats()["n_known_objects"] == 2


def test_flag_known_in_db_marks_matching_detections(tmp_path):
    """End-to-end: ingest a known object, insert a detection at its
    predicted position, and verify flag_known_in_db marks it."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow)
    from ariadne.discovery.imaging.mpc_catalog import (
        OrbitalElements, ephemeris_at_mjd, ingest_mpcorb_to_db,
        flag_known_in_db)

    db = open_db(tmp_path / "kn2.db")
    # Manually insert a known object record (skip the parse pipeline)
    import json as _json
    elements_json = _json.dumps({
        "a_au": 2.5, "e": 0.1, "i_deg": 5.0, "Omega_deg": 120,
        "omega_deg": 50, "M_deg": 180, "H": 16.0, "G": 0.15,
        "epoch_mjd": 60450.0, "name": "test",
    })
    import time
    db.conn.execute(
        """INSERT INTO known_objects
           (designation, epoch_mjd, orbital_elements, created_at,
            catalog)
           VALUES (?, ?, ?, ?, 'mpc')""",
        ("testobj", 60450.0, elements_json, time.time()))
    db.conn.commit()
    # Predict its position
    rec = OrbitalElements(
        designation="testobj", epoch_mjd=60450.0,
        a_au=2.5, e=0.1, i_deg=5.0,
        Omega_deg=120.0, omega_deg=50.0, M_deg=180.0)
    ra, dec, _, _ = ephemeris_at_mjd(rec, 60450.5)
    # Insert a detection there
    db.insert_detections([
        DetectionRow(image_id="testimg", mjd=60450.5, ra=ra, dec=dec),
    ])
    n_flagged = flag_known_in_db(
        db, target_mjd=60450.5,
        mjd_box_days=0.5, match_radius_arcsec=5.0)
    assert n_flagged == 1
    # Verify the detection's status changed
    rows = list(db.conn.execute(
        "SELECT status, known_designation FROM detections"))
    assert rows[0]["status"] == "known"
    assert rows[0]["known_designation"] == "testobj"
