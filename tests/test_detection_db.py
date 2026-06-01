"""Tests for the persistent detection database."""
from __future__ import annotations

import math
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def test_open_db_initialises_schema(tmp_path):
    from ariadne.discovery.imaging.detection_db import open_db
    db = open_db(tmp_path / "test.db")
    s = db.stats()
    assert s["n_detections"] == 0
    assert s["n_tracklets"] == 0
    assert s["n_chains"] == 0


def test_insert_detections_returns_ids(tmp_path):
    from ariadne.discovery.imaging.detection_db import open_db, DetectionRow
    db = open_db(tmp_path / "test.db")
    rows = [
        DetectionRow(image_id="img1", mjd=60450.0, ra=180.0, dec=-10.0,
                       mag=21.5, flux=1000.0, fwhm_px=3.0, x_pix=100, y_pix=200),
        DetectionRow(image_id="img1", mjd=60450.0, ra=180.05, dec=-10.05,
                       mag=22.0, flux=500.0),
    ]
    ids = db.insert_detections(rows)
    assert len(ids) == 2
    assert ids[0] != ids[1]
    s = db.stats()
    assert s["n_detections"] == 2


def test_insert_tracklet_links_to_detections(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow)
    db = open_db(tmp_path / "test.db")
    det_ids = db.insert_detections([
        DetectionRow(image_id="img1", mjd=60450.0, ra=180.0, dec=-10.0),
        DetectionRow(image_id="img1", mjd=60450.05, ra=180.001, dec=-10.001),
    ])
    tracklet = TrackletRow(
        detection_a_id=det_ids[0], detection_b_id=det_ids[1],
        mean_mjd=60450.025, mean_ra=180.0005, mean_dec=-10.0005,
        rate_arcsec_hr=2.0, pa_deg=45.0, night=60450)
    tid = db.insert_tracklet(tracklet)
    assert tid > 0
    # Detections should now point at this tracklet
    dets = db.get_tracklet_detections(tid)
    assert len(dets) == 2
    for d in dets:
        assert d["tracklet_id"] == tid


def test_insert_and_query_chain(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, ChainRow)
    db = open_db(tmp_path / "test.db")
    chain = ChainRow(
        n_tracklets=3, n_detections=6,
        first_mjd=60450.0, last_mjd=60456.0, arc_days=6.0,
        mean_ra=180.0, mean_dec=-10.0,
        mean_rate_arcsec_hr=2.0, mean_pa_deg=45.0,
        iod_strategy="bayesian_tno_class",
        iod_rms_arcsec=1.4,
        iod_state=(-1e10, 5e8, 2e9, -0.5, 4.5, -0.3),
        iod_t_ref=_et(60453.0),
    )
    cid = db.insert_chain(chain)
    fetched = db.get_chain(cid)
    assert fetched["n_tracklets"] == 3
    assert fetched["iod_rms_arcsec"] == pytest.approx(1.4)
    assert fetched["status"] == "open"


def test_update_chain_fields(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, ChainRow)
    db = open_db(tmp_path / "test.db")
    cid = db.insert_chain(ChainRow(
        n_tracklets=2, first_mjd=60450, last_mjd=60453,
        arc_days=3.0, mean_ra=180, mean_dec=-10,
        mean_rate_arcsec_hr=2.0, mean_pa_deg=45.0))
    db.update_chain(cid, {"n_tracklets": 4, "arc_days": 6.0,
                              "iod_rms_arcsec": 0.8,
                              "status": "closed"})
    chain = db.get_chain(cid)
    assert chain["n_tracklets"] == 4
    assert chain["arc_days"] == 6.0
    assert chain["iod_rms_arcsec"] == pytest.approx(0.8)
    assert chain["status"] == "closed"


def test_query_detections_by_cone(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow)
    db = open_db(tmp_path / "test.db")
    db.insert_detections([
        DetectionRow(image_id="i1", mjd=60450, ra=180.0, dec=-10.0),
        DetectionRow(image_id="i1", mjd=60450, ra=200.0, dec=20.0),
        DetectionRow(image_id="i2", mjd=60451, ra=180.01, dec=-10.01),
        DetectionRow(image_id="i2", mjd=60460, ra=180.0, dec=-10.0),
    ])
    # Query in (180,-10) box, MJD 60450..60452
    out = db.query_detections_by_cone(
        mjd_range=(60450, 60452),
        ra_range=(179.9, 180.1), dec_range=(-10.1, -9.9))
    assert len(out) == 2   # i1 and i2 within the box


def test_query_tracklets_by_window(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow)
    db = open_db(tmp_path / "test.db")
    det_ids = db.insert_detections([
        DetectionRow(image_id="i1", mjd=60450, ra=180, dec=-10),
        DetectionRow(image_id="i1", mjd=60450.05, ra=180.001, dec=-10),
    ])
    db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids[0], detection_b_id=det_ids[1],
        mean_mjd=60450.025, mean_ra=180.0005, mean_dec=-10.0,
        rate_arcsec_hr=3.0, pa_deg=90.0, night=60450))
    out = db.query_tracklets_by_window(60449, 60451)
    assert len(out) == 1
    assert out[0]["rate_arcsec_hr"] == pytest.approx(3.0)


def test_attach_tracklet_to_chain(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow, ChainRow)
    db = open_db(tmp_path / "test.db")
    det_ids = db.insert_detections([
        DetectionRow(image_id="i1", mjd=60450, ra=180, dec=-10),
        DetectionRow(image_id="i1", mjd=60450.05, ra=180.001, dec=-10),
    ])
    tid = db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids[0], detection_b_id=det_ids[1],
        mean_mjd=60450.025, mean_ra=180, mean_dec=-10,
        rate_arcsec_hr=2.0, pa_deg=45.0, night=60450))
    cid = db.insert_chain(ChainRow(n_tracklets=0))
    db.attach_tracklet_to_chain(tid, cid)
    tracks = db.get_chain_tracklets(cid)
    assert len(tracks) == 1
    assert tracks[0]["chain_id"] == cid
    dets = db.get_chain_detections(cid)
    assert len(dets) == 2


def test_open_chains_filtered_by_arc(tmp_path):
    from ariadne.discovery.imaging.detection_db import (
        open_db, ChainRow)
    db = open_db(tmp_path / "test.db")
    db.insert_chain(ChainRow(arc_days=2.0, first_mjd=60450, last_mjd=60452))
    db.insert_chain(ChainRow(arc_days=10.0, first_mjd=60450, last_mjd=60460))
    db.insert_chain(ChainRow(arc_days=0.5, first_mjd=60450, last_mjd=60450.5))
    # min_arc_hours=48 means arc >= 2 days -- so chain1 (2 days) AND chain2 (10 days)
    out = db.query_open_chains(min_arc_hours=48)
    assert len(out) == 2
    # min_arc_hours=120 means >=5 days -- only chain2
    out = db.query_open_chains(min_arc_hours=120)
    assert len(out) == 1
