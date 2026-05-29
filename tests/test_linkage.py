"""Stage 39 tests: HelioLinC-style orbit linkage + known-object recovery."""
import numpy as np

from ariadne.discovery import linkage as L
from ariadne.dynamics.secular import elements_to_state, kepler_step
from ariadne.data.constants import AU_KM
from ariadne.fields.hidden_mass import CLUSTERED_ETNOS

WINDOW = (0, 7, 15, 28, 42, 58)


def test_transform_collapses_at_true_distance():
    objs = CLUSTERED_ETNOS[:3]
    tracks, e0 = L.synthesize_tracklets(objs, night_offsets_days=WINDOW, pair_dt_s=4 * 3600.0,
                                        noise_arcsec=0.0, n_interlopers=0)
    geom = L.precompute_geometry(tracks)
    o = objs[0]
    r0, _ = elements_to_state(o["a_au"], o["e"], o["i"], o["Omega"], o["omega"], 180.0)
    m0 = np.where(geom.obj == 0)[0]
    x, v, _ = L.transform(geom, np.linalg.norm(r0), 0.0)
    t_ref = e0 + 30 * 86400.0
    xr, _ = kepler_step(x[m0], v[m0], L.MU, t_ref - geom.t[m0])
    spread = np.linalg.norm(xr - xr.mean(0), axis=1).max() / AU_KM
    assert spread < 0.01


def test_known_object_recovery():
    objs = CLUSTERED_ETNOS[:3]
    tracks, e0 = L.synthesize_tracklets(objs, night_offsets_days=WINDOW, pair_dt_s=4 * 3600.0,
                                        noise_arcsec=0.1, n_interlopers=150, seed=1)
    geom = L.precompute_geometry(tracks)
    t_ref = e0 + 30 * 86400.0
    with np.errstate(all="ignore"):
        cands = L.link(geom, t_ref, np.linspace(40, 1000, 120), np.linspace(-1.5, 1.5, 13),
                       cluster_au=1.0, min_obs=4, min_nights=3)
    rep = L.recovery_report(cands, geom)
    assert rep["n_recovered"] == len(objs)               # all known objects recovered
    assert rep["n_candidates"] - rep["n_pure"] <= 1      # ~no false positives


def test_interlopers_alone_produce_no_candidates():
    """A field of pure noise (no real objects) should yield ~no spurious linkages."""
    tracks, e0 = L.synthesize_tracklets([], night_offsets_days=WINDOW, pair_dt_s=4 * 3600.0,
                                        noise_arcsec=0.1, n_interlopers=200, seed=2)
    geom = L.precompute_geometry(tracks)
    with np.errstate(all="ignore"):
        cands = L.link(geom, e0 + 30 * 86400.0, np.linspace(40, 1000, 120),
                       np.linspace(-1.5, 1.5, 13), cluster_au=1.0, min_obs=4, min_nights=3)
    assert len(cands) <= 1
