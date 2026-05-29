"""Stage 29 tests: Ariadne -> Signalbook real-data cross-match bridge."""
import sqlite3

from ariadne.discovery.skybridge import (ecliptic_to_equatorial, _angsep_deg, query_sky,
                                         crossmatch_localization, build_celestial_index)


def _synthetic_index(path):
    con = sqlite3.connect(str(path))
    con.execute("create table celestial_sources "
                "(record_id text, modality text, observatory text, ra_deg real, dec_deg real)")
    con.executemany("insert into celestial_sources values (?,?,?,?,?)", [
        ("a", "optical", "gaia", 150.00, 20.00),
        ("b", "x_ray", "chandra", 150.30, 20.10),
        ("c", "optical", "sdss", 200.00, 20.00),       # far away
        ("d", "neutrino", "icecube", 150.05, 19.95),
    ])
    con.execute("create index ix on celestial_sources(dec_deg)")
    con.commit(); con.close()


def test_ecliptic_to_equatorial_anchors():
    ra, dec = ecliptic_to_equatorial(90.0, 0.0)
    assert abs(ra - 90.0) < 1e-6 and abs(dec - 23.4393) < 1e-3
    ra0, dec0 = ecliptic_to_equatorial(0.0, 0.0)
    assert abs(ra0) < 1e-6 and abs(dec0) < 1e-6


def test_angular_separation():
    assert abs(_angsep_deg(0, 0, 0, 1) - 1.0) < 1e-9
    assert abs(_angsep_deg(0, 0, 90, 0) - 90.0) < 1e-9


def test_cone_query_and_modality_filter(tmp_path):
    idx = tmp_path / "syn.db"
    _synthetic_index(idx)
    res = query_sky(str(idx), 150.0, 20.0, 1.0)
    ids = [s["record_id"] for s in res]
    assert set(ids) == {"a", "b", "d"}                 # the three in-cone sources
    assert "c" not in ids                              # out-of-cone excluded
    assert all(s["sep_deg"] <= 1.0 for s in res)
    optical = query_sky(str(idx), 150.0, 20.0, 1.0, modalities=("optical",))
    assert len(optical) == 1 and optical[0]["record_id"] == "a"


def test_crossmatch_localization(tmp_path):
    idx = tmp_path / "syn.db"
    _synthetic_index(idx)
    # an Ariadne sky-box pointed (in equatorial terms) at the synthetic cluster
    loc = {"ecliptic_lon_deg": 150.0, "ecliptic_lat_deg": 20.0, "angular_sigma_deg": 1.0}
    xm = crossmatch_localization(loc, str(idx), ecliptic=False)   # treat lon/lat as RA/Dec directly
    assert xm["n_sources"] == 3
    assert xm["by_modality"]["optical"] == 1 and xm["by_modality"]["x_ray"] == 1


def test_build_celestial_index_from_atlas(tmp_path):
    """build_celestial_index extracts ra/dec from payload_json (the signalbook gap-fix)."""
    atlas = tmp_path / "atlas.db"
    con = sqlite3.connect(str(atlas))
    con.execute("create table cross_scale_records (record_id text, time_utc_ns int, duration_ns int,"
                " lat_deg real, lon_deg real, center_freq_hz real, modality text, observatory text,"
                " phenomenon text, bucket_spatial int, bucket_temporal int, bucket_frequency int,"
                " payload_json text)")
    con.executemany("insert into cross_scale_records values (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("s1", 0, 0, None, None, 0, "optical", "gaia", "", 0, 0, 0, '{"ra_deg":10.0,"dec_deg":5.0}'),
        ("s2", 0, 0, None, None, 0, "x_ray", "xmm", "", 0, 0, 0, '{"ra_deg":11.0,"dec_deg":5.5}'),
        ("s3", 0, 0, 40.0, -100.0, 0, "rf", "fcc", "", 0, 0, 0, "{}"),   # terrestrial -> skipped
    ])
    con.commit(); con.close()
    out = tmp_path / "idx.db"
    n = build_celestial_index(str(atlas), str(out))
    assert n == 2                                      # only the two celestial sources indexed
    res = query_sky(str(out), 10.5, 5.25, 1.0)
    assert {s["record_id"] for s in res} == {"s1", "s2"}
