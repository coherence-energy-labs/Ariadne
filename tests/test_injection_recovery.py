"""Tests for the injection-recovery validation harness."""
from __future__ import annotations

import math
import json

import pytest


def _build_db_with_known(tmp_path):
    """Build a DB with 3 catalogued orbits we control."""
    from ariadne.discovery.imaging.detection_db import open_db
    db = open_db(tmp_path / "inj.db")
    # 3 orbits at the same epoch with different elements
    orbits = [
        ("ORBIT_A", 2.5, 0.1, 5.0, 120.0, 45.0, 180.0, 15.0),
        ("ORBIT_B", 3.0, 0.05, 10.0, 200.0, 60.0, 90.0, 16.0),
        ("ORBIT_C", 2.0, 0.2, 15.0, 50.0, 100.0, 270.0, 17.0),
    ]
    cur = db.conn.cursor()
    for d, a, e, i, O, w, M, H in orbits:
        elems = json.dumps({
            "a_au": a, "e": e, "i_deg": i, "Omega_deg": O,
            "omega_deg": w, "M_deg": M, "H": H, "G": 0.15,
            "epoch_mjd": 60450.0, "name": ""})
        cur.execute(
            """INSERT INTO known_objects
               (designation, epoch_mjd, ra_at_epoch, dec_at_epoch,
                rate_arcsec_hr, pa_deg, orbital_elements,
                last_observed_mjd, catalog, created_at)
               VALUES (?, ?, NULL, NULL, NULL, NULL, ?, NULL, 'test', '')""",
            (d, 60450.0, elems))
    db.conn.commit()
    return db


def test_pick_orbits_finds_in_field(tmp_path):
    """Orbits whose predicted position falls in the requested sky box
    must be returned by pick_orbits_in_field."""
    from ariadne.discovery.imaging.injection_recovery import (
        pick_orbits_in_field)
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    db = _build_db_with_known(tmp_path)
    target_mjd = 60500.0
    # First figure out where each orbit is by predicting it ourselves
    rows = list(db.conn.execute(
        "SELECT designation, epoch_mjd, orbital_elements FROM known_objects"))
    recs = []
    for r in rows:
        elems = json.loads(r["orbital_elements"])
        recs.append(OrbitalElements(
            designation=r["designation"],
            epoch_mjd=float(r["epoch_mjd"]),
            a_au=float(elems["a_au"]), e=float(elems["e"]),
            i_deg=float(elems["i_deg"]), Omega_deg=float(elems["Omega_deg"]),
            omega_deg=float(elems["omega_deg"]), M_deg=float(elems["M_deg"]),
            H_mag=float(elems["H"])))
    eph = bulk_ephemeris_at_mjd(recs, target_mjd)
    # Pick a sky box that contains exactly orbit 0
    ra0, dec0 = float(eph[0, 0]), float(eph[0, 1])
    box = (ra0 - 0.05, ra0 + 0.05), (dec0 - 0.05, dec0 + 0.05)
    in_field = pick_orbits_in_field(db, target_mjd, box[0], box[1],
                                            max_mag=30.0)
    designations = [r[0].designation for r in in_field]
    assert recs[0].designation in designations


def test_inject_and_recover_100pct(tmp_path):
    """Inject synthetic detections at predicted positions and verify
    100% are recovered with a reasonable match radius + low noise."""
    from ariadne.discovery.imaging.injection_recovery import (
        run_full_validation)
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    db = _build_db_with_known(tmp_path)
    target_mjd = 60500.0
    # Find the spread of orbits so we can build a box containing all 3
    rows = list(db.conn.execute(
        "SELECT designation, epoch_mjd, orbital_elements FROM known_objects"))
    recs = []
    for r in rows:
        elems = json.loads(r["orbital_elements"])
        recs.append(OrbitalElements(
            designation=r["designation"],
            epoch_mjd=float(r["epoch_mjd"]),
            a_au=float(elems["a_au"]), e=float(elems["e"]),
            i_deg=float(elems["i_deg"]), Omega_deg=float(elems["Omega_deg"]),
            omega_deg=float(elems["omega_deg"]), M_deg=float(elems["M_deg"]),
            H_mag=float(elems["H"])))
    eph = bulk_ephemeris_at_mjd(recs, target_mjd)
    ra_range = (float(eph[:, 0].min()) - 0.1,
                  float(eph[:, 0].max()) + 0.1)
    dec_range = (float(eph[:, 1].min()) - 0.1,
                   float(eph[:, 1].max()) + 0.1)
    report = run_full_validation(
        db, target_mjd, ra_range, dec_range,
        max_mag=30.0,
        n_inject_target=10,
        astrom_noise_arcsec=0.15,
        match_radius_arcsec=3.0)
    assert report.n_injected == 3, f"expected 3 injectable, got {report.n_injected}"
    assert report.recall == 1.0, f"recall {report.recall:.2f} not 100%"
    assert report.precision == 1.0, f"precision {report.precision:.2f} not 100%"
    assert report.n_false_positive == 0


def test_clear_injections_removes_only_inject_prefix(tmp_path):
    """clear_injections must delete INJECT__ rows but spare real detections."""
    from ariadne.discovery.imaging.injection_recovery import (
        clear_injections, INJECT_PREFIX)
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow)
    db = open_db(tmp_path / "clr.db")
    # Mix of real and injected
    db.insert_detections([
        DetectionRow(image_id="real_001", mjd=60500, ra=180, dec=-10),
        DetectionRow(image_id=f"{INJECT_PREFIX}foo", mjd=60500, ra=181, dec=-10),
        DetectionRow(image_id=f"{INJECT_PREFIX}bar", mjd=60500, ra=182, dec=-10),
    ])
    n = clear_injections(db)
    assert n == 2
    rows = list(db.conn.execute("SELECT image_id FROM detections"))
    assert len(rows) == 1
    assert rows[0]["image_id"] == "real_001"


def test_recall_degrades_when_noise_exceeds_match_radius(tmp_path):
    """If injection noise >> match radius, recall must drop below 100%.
    This pins the operational tuning knob."""
    from ariadne.discovery.imaging.injection_recovery import (
        run_full_validation)
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    db = _build_db_with_known(tmp_path)
    target_mjd = 60500.0
    rows = list(db.conn.execute(
        "SELECT designation, epoch_mjd, orbital_elements FROM known_objects"))
    recs = []
    for r in rows:
        elems = json.loads(r["orbital_elements"])
        recs.append(OrbitalElements(
            designation=r["designation"],
            epoch_mjd=float(r["epoch_mjd"]),
            a_au=float(elems["a_au"]), e=float(elems["e"]),
            i_deg=float(elems["i_deg"]), Omega_deg=float(elems["Omega_deg"]),
            omega_deg=float(elems["omega_deg"]), M_deg=float(elems["M_deg"]),
            H_mag=float(elems["H"])))
    eph = bulk_ephemeris_at_mjd(recs, target_mjd)
    ra_range = (float(eph[:, 0].min()) - 0.1,
                  float(eph[:, 0].max()) + 0.1)
    dec_range = (float(eph[:, 1].min()) - 0.1,
                   float(eph[:, 1].max()) + 0.1)
    # Noise = 5", radius = 0.5" => many injections drift outside the
    # match radius
    report = run_full_validation(
        db, target_mjd, ra_range, dec_range,
        max_mag=30.0, n_inject_target=10,
        astrom_noise_arcsec=5.0,
        match_radius_arcsec=0.5,
        seed=42)
    # Most should miss
    assert report.recall < 0.8, (
        f"recall {report.recall:.2f} should be low when noise >> radius")
