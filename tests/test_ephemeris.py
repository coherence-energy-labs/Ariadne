"""Real-ephemeris tests (Gate G7). Uses cached DE440 kernels (downloaded on first run)."""
import math

import numpy as np

from ariadne.data.ephemeris import body_gm, body_state, et, utc
from ariadne.dynamics.ephemeris_nbody import propagate_test_particle, propagate_nbody

EPOCH = "2025-06-01T00:00:00"


def test_utc_et_roundtrip():
    e = et(EPOCH)
    assert utc(e, "ISOC", 0).startswith("2025-06-01")


def test_earth_moon_distance_in_range():
    moon = body_state("MOON", et(EPOCH), "J2000", "EARTH")
    d = np.linalg.norm(moon[:3])
    assert 356000 < d < 407000          # perigee ~363k, apogee ~406k
    assert 0.9 < np.linalg.norm(moon[3:]) < 1.1


def test_two_body_closure_and_energy():
    mu = body_gm("EARTH")
    r = 8000.0
    r0 = np.array([r, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(mu / r), 0.0])
    T = 2 * math.pi * math.sqrt(r ** 3 / mu)
    sol = propagate_test_particle(r0, v0, et(EPOCH), (0.0, T), perturbers=())
    rf, vf = sol.y[:3, -1], sol.y[3:, -1]
    assert np.linalg.norm(rf - r0) < 1e-3        # < 1 m closure
    e0 = 0.5 * v0 @ v0 - mu / np.linalg.norm(r0)
    ef = 0.5 * vf @ vf - mu / np.linalg.norm(rf)
    assert abs(ef - e0) < 1e-6


def test_nbody_tracks_de440():
    bodies = ["SUN", "EARTH", "MOON"]
    ext = ["JUPITER BARYCENTER", "VENUS BARYCENTER", "MARS BARYCENTER", "SATURN BARYCENTER"]
    e0 = et(EPOCH)
    days = 2.0
    sol, _ = propagate_nbody(bodies, e0, (0.0, days * 86400.0), external=ext)
    for i, b in enumerate(("SUN", "EARTH", "MOON")):
        integ = sol.y[3 * i:3 * i + 3, -1]
        truth = body_state(b, e0 + days * 86400.0, "J2000", "SSB")[:3]
        assert np.linalg.norm(integ - truth) < 1.0       # sub-km over 2 days
