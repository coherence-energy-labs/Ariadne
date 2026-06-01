"""Tests for the DB-chain <-> IOD adapter."""
from __future__ import annotations

import math
import pytest


def _seed_chain_in_db(tmp_path, n_nights=3, obs_per_night=2, rate=2.0, pa=45.0):
    """Build a DB with a single chain spanning n_nights, returns (db, chain_id)."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow, ChainRow)
    db = open_db(tmp_path / "chain_iod.db")
    cid = db.insert_chain(ChainRow(n_tracklets=0, status="open"))
    ra0, dec0 = 180.0, -10.0
    cos_dec = math.cos(math.radians(dec0))
    rate_rad_hr = math.radians(rate / 3600.0)
    night_mjds = [60450.0 + 3.0 * k for k in range(n_nights)]
    for ni, mjd_base in enumerate(night_mjds):
        for j in range(obs_per_night):
            mjd = mjd_base + j * 2.0 / 24.0
            dt_hr = (mjd - 60450.0) * 24.0
            dra = rate_rad_hr * dt_hr * math.sin(math.radians(pa)) / cos_dec
            ddec = rate_rad_hr * dt_hr * math.cos(math.radians(pa))
            ra = ra0 + math.degrees(dra)
            dec = dec0 + math.degrees(ddec)
            ids = db.insert_detections([
                DetectionRow(image_id=f"n{ni}_{j}", mjd=mjd, ra=ra, dec=dec,
                               astrom_sigma_arcsec=0.1),
            ])
            db.conn.execute(
                "UPDATE detections SET chain_id = ? WHERE id = ?",
                (cid, ids[0]))
        # Build a within-night tracklet from the two obs (assumes obs_per_night=2)
        if obs_per_night >= 2:
            det_rows = list(db.conn.execute(
                "SELECT id FROM detections WHERE image_id IN (?, ?)",
                (f"n{ni}_0", f"n{ni}_1")))
            a_id, b_id = det_rows[0]["id"], det_rows[1]["id"]
            tid = db.insert_tracklet(TrackletRow(
                detection_a_id=a_id, detection_b_id=b_id,
                mean_mjd=mjd_base + 1.0 / 24.0,
                mean_ra=ra0 + math.degrees(
                    rate_rad_hr * (mjd_base - 60450.0) * 24.0
                    * math.sin(math.radians(pa)) / cos_dec),
                mean_dec=dec0 + math.degrees(
                    rate_rad_hr * (mjd_base - 60450.0) * 24.0
                    * math.cos(math.radians(pa))),
                rate_arcsec_hr=rate, pa_deg=pa,
                night=int(mjd_base)))
            db.conn.execute(
                "UPDATE tracklets SET chain_id = ? WHERE id = ?",
                (cid, tid))
    db.conn.commit()
    db.update_chain(cid, {"n_tracklets": n_nights})
    return db, cid


def test_load_chain_for_iod_returns_radians_and_et(tmp_path):
    from ariadne.discovery.imaging.chain_iod import load_chain_for_iod
    db, cid = _seed_chain_in_db(tmp_path)
    entries = load_chain_for_iod(db, cid)
    assert len(entries) == 6   # 3 nights x 2 obs
    for e in entries:
        # ra/dec in radians
        assert -math.pi <= e["ra"] <= 2 * math.pi
        assert -math.pi / 2 <= e["dec"] <= math.pi / 2
        # ET timestamp ~ 1e9 sec for MJD 60450
        assert 7e8 < e["t"] < 1e9
        # required keys for IOD
        assert "rate_arcsec_hr" in e
        assert "jd" in e


def test_load_empty_chain_returns_empty(tmp_path):
    from ariadne.discovery.imaging.chain_iod import load_chain_for_iod
    from ariadne.discovery.imaging.detection_db import open_db, ChainRow
    db = open_db(tmp_path / "empty.db")
    cid = db.insert_chain(ChainRow(n_tracklets=0, status="open"))
    entries = load_chain_for_iod(db, cid)
    assert entries == []


def test_run_iod_on_chain_persists_strategy(tmp_path):
    from ariadne.discovery.imaging.chain_iod import run_iod_on_chain
    db, cid = _seed_chain_in_db(tmp_path)
    res = run_iod_on_chain(db, cid,
                              rms_acceptance_arcsec=30.0,
                              n_draws=2, use_monte_carlo=False)
    assert isinstance(res, dict)
    assert "success" in res
    assert "wall_seconds" in res
    if res["success"]:
        # If IOD succeeded, the chain row should be updated
        chain = db.get_chain(cid)
        assert chain["iod_strategy"]
        assert chain["iod_rms_arcsec"] is not None


def test_run_iod_on_short_chain_returns_failure(tmp_path):
    from ariadne.discovery.imaging.chain_iod import run_iod_on_chain
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, ChainRow)
    db = open_db(tmp_path / "short.db")
    cid = db.insert_chain(ChainRow(n_tracklets=0, status="open"))
    # Insert just 2 detections (below 3-observation IOD minimum)
    ids = db.insert_detections([
        DetectionRow(image_id="i1", mjd=60450, ra=180, dec=-10),
        DetectionRow(image_id="i2", mjd=60450.1, ra=180.01, dec=-10),
    ])
    for did in ids:
        db.conn.execute(
            "UPDATE detections SET chain_id = ? WHERE id = ?",
            (cid, did))
    db.conn.commit()
    res = run_iod_on_chain(db, cid)
    assert res["success"] is False
    assert "insufficient" in res["reason"]


def test_run_iod_on_all_open_chains_skips_already_fit(tmp_path):
    from ariadne.discovery.imaging.chain_iod import run_iod_on_all_open_chains
    from ariadne.discovery.imaging.detection_db import (
        open_db, ChainRow)
    db, cid = _seed_chain_in_db(tmp_path)
    # Mark the chain as already IOD'd
    db.update_chain(cid, {
        "iod_strategy": "gauss",
        "iod_rms_arcsec": 0.5,
        "iod_t_ref": 0.0,
    })
    results = run_iod_on_all_open_chains(db, min_tracklets=1,
                                              n_draws=2)
    # Already-fit chain should be skipped
    assert len(results) == 0
