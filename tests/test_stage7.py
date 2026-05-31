"""Genesis (G9) + independent cross-validation (G10*) tests."""
import os

import numpy as np
import pytest

from ariadne.data.constants import SUN_EARTH
from ariadne.data.ephemeris import body_state, et
from ariadne.data.kernels import KERNEL_DIR, ensure_kernels
from ariadne.dynamics.ephemeris_nbody import propagate_test_particle
from ariadne.transfers.genesis import genesis_halo, earth_approach


@pytest.mark.slow
def test_genesis_halo_period_and_manifold_reaches_earth():
    h, _ = genesis_halo()
    period_d = h.period * SUN_EARTH.T_star / 86400.0
    assert 170.0 < period_d < 185.0                 # real SOHO/Genesis ~178 d
    r = earth_approach(h, t_max=7.0, n_seeds=70)
    # manifold carries the ride from L1 (1.49e6 km) down toward Earth's vicinity
    assert r["fraction_of_l1"] < 0.2


def test_ephemeris_libraries_agree():
    ensure_kernels()
    from jplephem.spk import SPK
    e = et("2025-06-01T00:00:00")
    jd = 2451545.0 + e / 86400.0
    k = SPK.open(os.path.join(KERNEL_DIR, "de440s.bsp"))
    moon_jpl = np.array(k[3, 301].compute(jd)) - np.array(k[3, 399].compute(jd))
    k.close()
    moon_spice = body_state("MOON", e, "J2000", "EARTH")[:3]
    assert np.linalg.norm(moon_spice - moon_jpl) < 1e-3      # < 1 m


def test_independent_integrators_agree():
    e = et("2025-06-01T00:00:00")
    r0 = np.array([7000.0, 1000.0, 500.0])
    v0 = np.array([1.0, 7.5, 0.3])
    span = (0.0, 86400.0)
    a = propagate_test_particle(r0, v0, e, span, perturbers=("SUN", "MOON"))
    b = propagate_test_particle(r0, v0, e, span, perturbers=("SUN", "MOON"),
                                method="Radau", rtol=1e-11, atol=1e-9)
    assert np.linalg.norm(a.y[:3, -1] - b.y[:3, -1]) < 0.1   # < 100 m
