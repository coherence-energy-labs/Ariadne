"""Stage 18 tests: synodic<->inertial frame + ephemeris re-targeting."""
import numpy as np
import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.data.ephemeris import et, body_state
from ariadne.dynamics.frames import synodic_frame, synodic_to_inertial, inertial_to_synodic
from ariadne.orbits.families import lyapunov_orbit_at_jacobi
from ariadne.connections.heteroclinic import find_heteroclinic
from ariadne.transfers.ephemeris_retarget import retarget_orbit, retarget_heteroclinic

MU = EARTH_MOON.mu
EPOCH = "2025-06-01T00:00:00"


def test_frame_roundtrip_is_exact():
    e0 = et(EPOCH)
    f = synodic_frame(e0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.normal(size=6)
        s2 = inertial_to_synodic(synodic_to_inertial(s, e0, MU, f), e0, MU, f)
        assert np.max(np.abs(s - s2)) < 1e-11


def test_moon_embeds_onto_real_position():
    e0 = et(EPOCH)
    embed = synodic_to_inertial(np.array([1 - MU, 0, 0, 0, 0, 0]), e0, MU)
    real = body_state("MOON", e0, "J2000", "EARTH")
    assert np.linalg.norm(embed[:3] - real[:3]) < 1e-6      # position exact


@pytest.mark.slow
def test_lyapunov_orbit_reconverges_in_ephemeris():
    e0 = et(EPOCH)
    orb = lyapunov_orbit_at_jacobi(MU, "L1", 3.16)
    r = retarget_orbit(orb, e0, MU, EARTH_MOON, n_patch=8, periodic=False)
    assert r["max_resid_km"] < 1.0                          # position continuity in DE440
    assert r["total_dv_ms"] < 1000.0                        # a sane stationkeeping budget


@pytest.mark.slow
def test_heteroclinic_reconverges_in_ephemeris():
    e0 = et(EPOCH)
    conn = find_heteroclinic(MU, 3.15, "L1", "L2", n_seeds=120)
    assert conn is not None
    r = retarget_heteroclinic(MU, EARTH_MOON, conn, e0, n_patch=12, t_leg=2.6)
    assert r is not None
    assert r["max_resid_km"] < 1.0                          # connection exists in DE440
