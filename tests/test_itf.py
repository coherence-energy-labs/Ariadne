"""Stage 43 tests: ITF parser + tracklet builder + per-bin linker (pure unit tests)."""
import math
import os
import tempfile

import numpy as np

from ariadne.discovery import itf
from ariadne.discovery import linkage as L


def _mock(path):
    # MPC 80-col format is PACKED (no separators between date/RA/Dec/etc)
    # Layout: [5 num][7 desig][3 notes][17 date=yyyy mm dd.dddddd][12 RA=hh mm ss.ddd]
    #         [12 Dec=sdd mm ss.dd][21 mag/cat][3 obs]
    lines = [
        "     T000001  C2024 03 15.50123412 30 00.000+05 30 00.00         20.5 R      500",
        "     T000001  C2024 03 15.54123412 30 00.500+05 30 01.50         20.5 R      500",
        "     T000002  C2024 03 16.50123414 45 30.000-10 15 45.00         21.0 R      500",
        "     T000002  C2024 03 16.54123414 45 30.300-10 15 46.00         21.0 R      500",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_parse_itf_handles_80col_format():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.txt"); _mock(p)
        groups = itf.parse_itf(p)
    assert set(groups) == {"T000001", "T000002"}
    assert all(len(v) == 2 for v in groups.values())
    j, ra, dec, oc = groups["T000001"][0]
    assert 2460384 < j < 2460386                          # 2024 Mar 15
    assert oc == "500"


def test_build_tracklets_makes_position_and_rate():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.txt"); _mock(p)
        tracks = itf.build_tracklets(itf.parse_itf(p))
    assert len(tracks) == 2
    for t in tracks:
        assert 0 <= math.degrees(t["ra"]) < 360
        assert -90 <= math.degrees(t["dec"]) <= 90
        assert t["rate_arcsec_hr"] >= 0


def test_filter_slow_drops_fast_movers():
    tracks = [{"rate_arcsec_hr": r} for r in (0.5, 2.0, 4.9, 5.1, 30.0)]
    assert len(itf.filter_slow(tracks, 5.0)) == 3


def test_link_bins_finds_a_synthetic_cluster():
    """A handful of synthetic distant-object tracklets in one sky/time bin link to a candidate."""
    from ariadne.dynamics.secular import elements_to_state, kepler_step, YEAR_S
    from ariadne.data.ephemeris import et, body_state
    from ariadne.data.constants import GM_SUN
    e0 = et("2026-01-01T00:00:00")
    r0, v0 = elements_to_state(120.0, 0.3, 10.0, 80.0, 60.0, 180.0)
    tracks = []
    for k, doff in enumerate([0, 8, 18, 32, 48, 70]):
        et_n = e0 + doff * 86400.0
        sub = []
        for ddt in (0.0, 4 * 3600.0):
            x, _ = kepler_step(r0, v0, GM_SUN, et_n + ddt - e0)
            g = x - body_state("EARTH", et_n + ddt, "J2000", "SUN")[:3]
            rn = np.linalg.norm(g)
            sub.append((2451545 + (et_n + ddt) / 86400, math.atan2(g[1], g[0]), math.asin(g[2] / rn)))
        (j1, r1, d1), (j2, r2, d2) = sub
        tracks.append({"desig": "T%d" % k, "t": (0.5 * (j1 + j2) - 2451545) * 86400, "jd": 0.5 * (j1 + j2),
                       "ra": 0.5 * (r1 + r2), "dec": 0.5 * (d1 + d2),
                       "dra": (r2 - r1) / ((j2 - j1) * 86400), "ddec": (d2 - d1) / ((j2 - j1) * 86400),
                       "rate_arcsec_hr": 1.0, "obscode": "500", "obj": 0})
    cands = itf.link_bins(tracks, np.linspace(40, 200, 50), np.linspace(-1.5, 1.5, 11),
                          min_obs=4, min_nights=3, cluster_au=0.5, ra_cells=12, dec_cells=6,
                          window_days=150, min_bin=4)
    assert any(c["n"] >= 4 for c in cands)
