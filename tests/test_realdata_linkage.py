"""Stage 42 test: orbit linkage on REAL MPC astrometry (skipped if no network/astroquery)."""
import numpy as np
import pytest

from ariadne.discovery import linkage as L


def _fetch_or_skip(desig):
    try:
        import astroquery.mpc  # noqa: F401
        tracks, e0 = L.tracklets_from_mpc(desig, window_days=150, min_per_night=2)
    except Exception as e:                      # network/service/astroquery unavailable
        pytest.skip(f"real MPC data unavailable: {str(e)[:80]}")
    if len(tracks) < 4:
        pytest.skip(f"too few real tracklets in season ({len(tracks)})")
    return tracks, e0


def test_recover_known_object_from_real_astrometry():
    tracks, e0 = _fetch_or_skip("90377")        # Sedna
    n_real = len(tracks)
    tracks = L.add_interlopers(tracks, 200, seed=1)
    geom = L.precompute_geometry(tracks)
    t_ref = float(np.median([tr["t"] for tr in tracks]))
    with np.errstate(all="ignore"):
        cands = L.link(geom, t_ref, np.linspace(40, 160, 100), np.linspace(-1.5, 1.5, 15),
                       cluster_au=0.5, min_obs=4, min_nights=3)
    rep = L.recovery_report(cands, geom)
    assert rep["n_recovered"] >= 1               # recovered the real object from real data
    assert rep["n_candidates"] - rep["n_pure"] <= 1
