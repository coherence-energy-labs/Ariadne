"""Tests for the multi-night linker."""
from __future__ import annotations

import math
import pytest


def _make_db_with_seed_chain(tmp_path, *, rate=2.0, pa=45.0):
    """Build a DB with one open chain spanning MJDs 60450 + 60453."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow, ChainRow)
    db = open_db(tmp_path / "test.db")

    # Two tracklets on two different nights, simulating a real moving object
    cos_dec = math.cos(math.radians(-10.0))

    # Night 1: tracklet at (180, -10), rate=2"/hr, pa=45 (NE)
    det_ids_n1 = db.insert_detections([
        DetectionRow(image_id="img1a", mjd=60450.000, ra=180.0, dec=-10.0),
        DetectionRow(image_id="img1b", mjd=60450.083, ra=180.001, dec=-9.999),
    ])
    tid1 = db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids_n1[0], detection_b_id=det_ids_n1[1],
        mean_mjd=60450.04, mean_ra=180.0005, mean_dec=-9.9995,
        rate_arcsec_hr=rate, pa_deg=pa, night=60450))

    # Night 2: at the predicted position 3 days later
    # rate=2"/hr * 72h = 144" along PA=45 -> dra=ddec=102"
    dra_deg = 144 * math.sin(math.radians(pa)) / 3600.0 / cos_dec
    ddec_deg = 144 * math.cos(math.radians(pa)) / 3600.0
    n2_ra = 180.0005 + dra_deg
    n2_dec = -9.9995 + ddec_deg
    det_ids_n2 = db.insert_detections([
        DetectionRow(image_id="img2a", mjd=60453.000, ra=n2_ra, dec=n2_dec),
        DetectionRow(image_id="img2b", mjd=60453.083, ra=n2_ra + 0.001,
                       dec=n2_dec),
    ])
    tid2 = db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids_n2[0], detection_b_id=det_ids_n2[1],
        mean_mjd=60453.04, mean_ra=n2_ra, mean_dec=n2_dec,
        rate_arcsec_hr=rate, pa_deg=pa, night=60453))

    # Bundle into one open chain
    cid = db.insert_chain(ChainRow(
        n_tracklets=2, n_detections=4,
        first_mjd=60450.04, last_mjd=60453.04, arc_days=3.0,
        mean_ra=n2_ra, mean_dec=n2_dec,
        mean_rate_arcsec_hr=rate, mean_pa_deg=pa,
        status="open"))
    db.attach_tracklet_to_chain(tid1, cid)
    db.attach_tracklet_to_chain(tid2, cid)
    return db, cid, n2_ra, n2_dec


def test_predict_chain_at_mjd_zero_dt_is_last_pos():
    from ariadne.discovery.imaging.multi_night_linker import predict_chain_at_mjd
    chain = {"last_mjd": 60450.0, "mean_ra": 180.0, "mean_dec": -10.0,
              "mean_rate_arcsec_hr": 3.0, "mean_pa_deg": 90.0}
    ra, dec = predict_chain_at_mjd(chain, 60450.0)
    assert ra == pytest.approx(180.0)
    assert dec == pytest.approx(-10.0)


def test_predict_chain_at_mjd_eastward_motion():
    """pa=90 (east) at rate 3600"/hr for 1 hour -> 1 deg east at constant dec."""
    from ariadne.discovery.imaging.multi_night_linker import predict_chain_at_mjd
    chain = {"last_mjd": 60450.0, "mean_ra": 180.0, "mean_dec": 0.0,
              "mean_rate_arcsec_hr": 3600.0, "mean_pa_deg": 90.0}
    ra, dec = predict_chain_at_mjd(chain, 60450.0 + 1.0/24.0)
    # 3600"/hr * 1hr = 3600" = 1 deg in RA (cos(0)=1)
    assert ra == pytest.approx(181.0, abs=0.001)
    assert dec == pytest.approx(0.0, abs=0.001)


def test_link_tonight_empty_tracklets_returns_zero(tmp_path):
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.multi_night_linker import link_tonight
    db = open_db(tmp_path / "empty.db")
    report = link_tonight(db, [])
    assert report.n_tonight_tracklets == 0
    assert report.n_chains_extended == 0
    assert report.n_chains_seeded == 0


def test_link_tonight_extends_existing_chain(tmp_path):
    """An open chain ending at MJD 60453 should be extended by tonight's
    tracklet at MJD 60456 sitting at the predicted position."""
    from ariadne.discovery.imaging.detection_db import (
        DetectionRow, TrackletRow)
    from ariadne.discovery.imaging.multi_night_linker import link_tonight
    db, cid, n2_ra, n2_dec = _make_db_with_seed_chain(tmp_path)

    rate = 2.0; pa = 45.0
    cos_dec = math.cos(math.radians(-10.0))
    dra_deg = 144 * math.sin(math.radians(pa)) / 3600.0 / cos_dec
    ddec_deg = 144 * math.cos(math.radians(pa)) / 3600.0
    n3_ra = n2_ra + dra_deg
    n3_dec = n2_dec + ddec_deg

    # Insert tonight's detections + tracklet (NOT attached to a chain yet)
    det_ids = db.insert_detections([
        DetectionRow(image_id="img3a", mjd=60456.0, ra=n3_ra, dec=n3_dec),
        DetectionRow(image_id="img3b", mjd=60456.083, ra=n3_ra + 0.001,
                       dec=n3_dec),
    ])
    tid = db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids[0], detection_b_id=det_ids[1],
        mean_mjd=60456.04, mean_ra=n3_ra, mean_dec=n3_dec,
        rate_arcsec_hr=rate, pa_deg=pa, night=60456))

    tonight = [{
        "tracklet_id": tid, "mean_mjd": 60456.04,
        "mean_ra": n3_ra, "mean_dec": n3_dec,
        "rate_arcsec_hr": rate, "pa_deg": pa,
    }]
    report = link_tonight(db, tonight, link_window_days=30.0)
    assert report.n_chains_extended == 1
    assert report.n_chains_seeded == 0
    # Chain should now reference 3 tracklets
    chain = db.get_chain(cid)
    assert chain["n_tracklets"] == 3
    assert chain["arc_days"] == pytest.approx(6.0, abs=0.1)


def test_link_tonight_seeds_new_chain_from_unmatched(tmp_path):
    """Insert a tracklet from a past night (NOT in any chain), then a
    tonight tracklet that is consistent with it under rate-coherence.
    The linker should seed a new chain from them."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow)
    from ariadne.discovery.imaging.multi_night_linker import link_tonight
    db = open_db(tmp_path / "seed.db")

    rate = 3.0; pa = 90.0   # eastward
    cos_dec = math.cos(math.radians(-10.0))
    # Tracklet 3 days ago at (180, -10)
    det_ids_old = db.insert_detections([
        DetectionRow(image_id="old_a", mjd=60450.0, ra=180.0, dec=-10.0),
        DetectionRow(image_id="old_b", mjd=60450.083, ra=180.001, dec=-10.0),
    ])
    tid_old = db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids_old[0], detection_b_id=det_ids_old[1],
        mean_mjd=60450.04, mean_ra=180.0005, mean_dec=-10.0,
        rate_arcsec_hr=rate, pa_deg=pa, night=60450))

    # Tonight at the predicted eastward position 3 days later
    dra_deg = (rate * 72.0) * math.sin(math.radians(pa)) / 3600.0 / cos_dec
    n_ra = 180.0005 + dra_deg
    n_dec = -10.0
    det_ids_new = db.insert_detections([
        DetectionRow(image_id="new_a", mjd=60453.0, ra=n_ra, dec=n_dec),
        DetectionRow(image_id="new_b", mjd=60453.083, ra=n_ra + 0.001,
                       dec=n_dec),
    ])
    tid_new = db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids_new[0], detection_b_id=det_ids_new[1],
        mean_mjd=60453.04, mean_ra=n_ra, mean_dec=n_dec,
        rate_arcsec_hr=rate, pa_deg=pa, night=60453))

    tonight = [{
        "tracklet_id": tid_new, "mean_mjd": 60453.04,
        "mean_ra": n_ra, "mean_dec": n_dec,
        "rate_arcsec_hr": rate, "pa_deg": pa,
    }]
    # Pre-state: 0 chains
    assert db.stats()["n_chains"] == 0
    report = link_tonight(db, tonight,
                            link_window_days=30.0,
                            seed_window_days=7.0,
                            position_tol_arcsec=120.0)
    assert report.n_chains_seeded == 1
    s = db.stats()
    assert s["n_chains"] == 1
    chains = db.query_open_chains(min_arc_hours=0)
    assert chains[0]["n_tracklets"] >= 1   # at least the new tracklet
