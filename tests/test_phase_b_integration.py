"""Phase B integration test: 3-night synthetic sequence through the DB.

Verifies:
  - Night-1 detections seed a chain (no prior chain exists -- linker
    finds no extension target, and the unmatched tracklet falls through
    to the SEED path which requires a PRIOR tracklet, so no chain forms).
  - Night-2 brings a tracklet at the predicted position. The linker
    now seeds a chain by pairing the night-2 tracklet with the
    night-1 tracklet.
  - Night-3 brings another tracklet at the next predicted position.
    The linker extends the existing chain.

End state: one chain with 3 tracklets spanning 6 days.
"""
from __future__ import annotations

import math
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def test_three_night_sequence_builds_chain_via_db(tmp_path):
    """End-to-end: feed three nights' tracklets to the multi-night linker
    via the DB and verify a chain forms + extends correctly."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow)
    from ariadne.discovery.imaging.multi_night_linker import link_tonight

    db = open_db(tmp_path / "phaseb.db")

    # Object orbit: at (180, -10), moving east at 3"/hr
    ra0, dec0 = 180.0, -10.0
    rate = 3.0   # arcsec/hr
    pa = 90.0    # due east
    cos_dec = math.cos(math.radians(dec0))

    def position_at(mjd):
        dt_hr = (mjd - 60450.0) * 24.0
        dra_deg = rate * dt_hr * math.sin(math.radians(pa)) / 3600.0 / cos_dec
        ddec_deg = rate * dt_hr * math.cos(math.radians(pa)) / 3600.0
        return (ra0 + dra_deg, dec0 + ddec_deg)

    def insert_night(mjd_base, image_prefix):
        """Insert 2 detections + 1 tracklet for one night."""
        mjd_a, mjd_b = mjd_base, mjd_base + 2.0 / 24.0
        ra_a, dec_a = position_at(mjd_a)
        ra_b, dec_b = position_at(mjd_b)
        ids = db.insert_detections([
            DetectionRow(image_id=f"{image_prefix}a", mjd=mjd_a,
                           ra=ra_a, dec=dec_a),
            DetectionRow(image_id=f"{image_prefix}b", mjd=mjd_b,
                           ra=ra_b, dec=dec_b),
        ])
        # Tracklet at the mean position
        d_ra = (ra_b - ra_a) * cos_dec
        d_dec = dec_b - dec_a
        dt_hr = (mjd_b - mjd_a) * 24.0
        actual_rate = math.hypot(d_ra, d_dec) * 3600.0 / dt_hr
        actual_pa = math.degrees(math.atan2(d_ra, d_dec)) % 360.0
        trk_row = TrackletRow(
            detection_a_id=ids[0], detection_b_id=ids[1],
            mean_mjd=0.5 * (mjd_a + mjd_b),
            mean_ra=0.5 * (ra_a + ra_b),
            mean_dec=0.5 * (dec_a + dec_b),
            rate_arcsec_hr=actual_rate, pa_deg=actual_pa,
            night=int(mjd_base))
        tid = db.insert_tracklet(trk_row)
        return [{
            "tracklet_id": tid,
            "mean_mjd": trk_row.mean_mjd,
            "mean_ra": trk_row.mean_ra,
            "mean_dec": trk_row.mean_dec,
            "rate_arcsec_hr": trk_row.rate_arcsec_hr,
            "pa_deg": trk_row.pa_deg,
        }]

    # Night 1
    night1 = insert_night(60450.0, "n1_")
    report1 = link_tonight(db, night1)
    assert report1.n_chains_extended == 0
    assert report1.n_chains_seeded == 0   # no prior tracklets to seed from
    assert db.stats()["n_chains"] == 0

    # Night 2 -- pairs with night-1's tracklet, seeds a chain
    night2 = insert_night(60453.0, "n2_")
    report2 = link_tonight(
        db, night2,
        link_window_days=30.0, seed_window_days=10.0,
        position_tol_arcsec=120.0)
    assert report2.n_chains_seeded == 1
    assert db.stats()["n_chains"] == 1
    chain = db.query_open_chains()[0]
    assert chain["n_tracklets"] == 2

    # Night 3 -- extends the existing chain
    night3 = insert_night(60456.0, "n3_")
    report3 = link_tonight(
        db, night3,
        link_window_days=30.0,
        position_tol_arcsec=120.0)
    assert report3.n_chains_extended == 1
    chain = db.query_open_chains()[0]
    assert chain["n_tracklets"] == 3
    assert chain["arc_days"] == pytest.approx(6.0, abs=0.1)


def test_disconnected_motion_does_not_link(tmp_path):
    """Two tracklets at the same position but on different nights with
    very different rate vectors should NOT link (they're different objects)."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow)
    from ariadne.discovery.imaging.multi_night_linker import link_tonight

    db = open_db(tmp_path / "disco.db")
    # Object A: slow eastward at night 1
    ids_a = db.insert_detections([
        DetectionRow(image_id="n1a", mjd=60450.0, ra=180.0, dec=-10.0),
        DetectionRow(image_id="n1b", mjd=60450.083, ra=180.001, dec=-10.0),
    ])
    db.insert_tracklet(TrackletRow(
        detection_a_id=ids_a[0], detection_b_id=ids_a[1],
        mean_mjd=60450.04, mean_ra=180.0005, mean_dec=-10.0,
        rate_arcsec_hr=3.0, pa_deg=90.0, night=60450))
    # Object B: fast NEO at night 4 at same position -- WAY off in rate
    ids_b = db.insert_detections([
        DetectionRow(image_id="n4a", mjd=60454.0, ra=180.0005, dec=-10.0),
        DetectionRow(image_id="n4b", mjd=60454.083, ra=180.06, dec=-10.0),
    ])
    tid_b = db.insert_tracklet(TrackletRow(
        detection_a_id=ids_b[0], detection_b_id=ids_b[1],
        mean_mjd=60454.04, mean_ra=180.0305, mean_dec=-10.0,
        rate_arcsec_hr=200.0, pa_deg=90.0, night=60454))
    tonight = [{"tracklet_id": tid_b, "mean_mjd": 60454.04,
                  "mean_ra": 180.0305, "mean_dec": -10.0,
                  "rate_arcsec_hr": 200.0, "pa_deg": 90.0}]
    report = link_tonight(db, tonight,
                            link_window_days=30.0, seed_window_days=10.0,
                            rate_tol_pct=50.0)
    # Rate disagrees by 6500% -- linker should not bridge
    assert report.n_chains_seeded == 0
    assert db.stats()["n_chains"] == 0


def test_db_stats_after_realistic_session(tmp_path):
    """Sanity: stats reflect the DB contents after a small session."""
    from ariadne.discovery.imaging.detection_db import (
        open_db, DetectionRow, TrackletRow, ChainRow)

    db = open_db(tmp_path / "stats.db")
    # 10 detections, 2 tracklets, 1 chain
    det_ids = db.insert_detections([
        DetectionRow(image_id=f"img{i}", mjd=60450 + i * 0.1, ra=180, dec=-10)
        for i in range(10)
    ])
    db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids[0], detection_b_id=det_ids[1],
        mean_mjd=60450.05, mean_ra=180, mean_dec=-10,
        rate_arcsec_hr=2, pa_deg=45, night=60450))
    db.insert_tracklet(TrackletRow(
        detection_a_id=det_ids[2], detection_b_id=det_ids[3],
        mean_mjd=60450.25, mean_ra=180, mean_dec=-10,
        rate_arcsec_hr=3, pa_deg=90, night=60450))
    db.insert_chain(ChainRow(n_tracklets=2, status="open"))

    s = db.stats()
    assert s["n_detections"] == 10
    assert s["n_tracklets"] == 2
    assert s["n_chains"] == 1
    assert s["n_chains_open"] == 1
    assert s["mjd_span_days"] == pytest.approx(0.9)
