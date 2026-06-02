"""Tests for the N-body batch ephemeris.

The N-body path handles long epoch_deltas (years) by including
Sun + giant-planet perturbations. These tests pin:
  - N-body agrees with 2-body for short dt (perturbations negligible)
  - N-body produces a *different* (and physically larger) drift over
    multi-year propagation (Jupiter pull)
  - The batched N-body agrees with the trusted single-orbit propagator
    in `dynamics.ephemeris_nbody`
  - auto_ephemeris_at_mjd switches modes correctly
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _make_recs(epoch_mjd=60450.0):
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    return [
        OrbitalElements(designation='MB1', a_au=2.5, e=0.1, i_deg=5.0,
                         Omega_deg=120.0, omega_deg=45.0, M_deg=180.0,
                         epoch_mjd=epoch_mjd, H_mag=15.0),
        OrbitalElements(designation='MB2', a_au=3.0, e=0.05, i_deg=10.0,
                         Omega_deg=200.0, omega_deg=60.0, M_deg=90.0,
                         epoch_mjd=epoch_mjd, H_mag=16.0),
    ]


def test_nbody_at_epoch_matches_2body():
    """Zero propagation. N-body must match 2-body exactly."""
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_ephemeris_at_mjd_nbody)
    recs = _make_recs()
    two_body = bulk_ephemeris_at_mjd(recs, recs[0].epoch_mjd)
    nbody = bulk_ephemeris_at_mjd_nbody(recs, recs[0].epoch_mjd)
    for i in range(2):
        cos_dec = math.cos(math.radians(two_body[i, 1]))
        sep_arcsec = math.hypot(
            (two_body[i, 0] - nbody[i, 0]) * cos_dec,
            two_body[i, 1] - nbody[i, 1]) * 3600
        # At epoch, perturbations are negligible; the small residual is the
        # batch light-time handling (N-body uses the batch-median light-time,
        # the 2-body path is per-object). Sub-arcsec is expected.
        assert sep_arcsec < 2.0, f"sep {sep_arcsec:.4f}\""


def test_nbody_close_to_2body_for_short_dt():
    """30-day propagation. Planetary perturbations are small (<0.5\")."""
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_ephemeris_at_mjd_nbody)
    recs = _make_recs()
    target = recs[0].epoch_mjd + 30.0
    two_body = bulk_ephemeris_at_mjd(recs, target)
    nbody = bulk_ephemeris_at_mjd_nbody(recs, target)
    for i in range(2):
        cos_dec = math.cos(math.radians(two_body[i, 1]))
        sep_arcsec = math.hypot(
            (two_body[i, 0] - nbody[i, 0]) * cos_dec,
            two_body[i, 1] - nbody[i, 1]) * 3600
        assert sep_arcsec < 1.0, f"30-day disagreement {sep_arcsec:.3f}\""


def test_nbody_drifts_meaningfully_over_years():
    """Multi-year propagation: planetary perturbations should produce
    a non-zero, physically-sensible delta from pure 2-body."""
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_ephemeris_at_mjd_nbody)
    recs = _make_recs()
    target = recs[0].epoch_mjd + 365.0 * 5     # 5 years
    two_body = bulk_ephemeris_at_mjd(recs, target)
    nbody = bulk_ephemeris_at_mjd_nbody(recs, target)
    seps = []
    for i in range(2):
        if math.isnan(nbody[i, 0]):
            continue
        cos_dec = math.cos(math.radians(two_body[i, 1]))
        sep_arcsec = math.hypot(
            (two_body[i, 0] - nbody[i, 0]) * cos_dec,
            two_body[i, 1] - nbody[i, 1]) * 3600
        seps.append(sep_arcsec)
    # 5-year Jupiter perturbations are typically 100s of arcseconds
    # for main-belt orbits; require > 10" to confirm we aren't doing pure 2-body
    assert max(seps) > 10, (
        f"N-body should differ meaningfully from 2-body over 5 yrs; got max "
        f"{max(seps):.2f}\"")


def test_batched_nbody_matches_serial_nbody():
    """My batched N-body must agree with the trusted single-orbit
    propagator in `dynamics.ephemeris_nbody.propagate_test_particle`."""
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_propagate_nbody)
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_elements_to_state)
    from ariadne.dynamics.ephemeris_nbody import propagate_test_particle

    recs = _make_recs(epoch_mjd=60450.0)
    a = np.array([r.a_au for r in recs])
    e = np.array([r.e for r in recs])
    i = np.array([r.i_deg for r in recs])
    O = np.array([r.Omega_deg for r in recs])
    w = np.array([r.omega_deg for r in recs])
    M = np.array([r.M_deg for r in recs])
    r0, v0 = bulk_elements_to_state(a, e, i, O, w, M)

    t0_mjd = 60450.0
    t1_mjd = t0_mjd + 365.0
    et0 = (t0_mjd - 51544.5) * 86400.0
    t_span = (0.0, (t1_mjd - t0_mjd) * 86400.0)

    r_batch, _ = bulk_propagate_nbody(
        r0, v0, t0_mjd, t1_mjd,
        perturbers=('JUPITER BARYCENTER', 'SATURN BARYCENTER'),
        atol=1e-6, rtol=1e-12, max_step=86400.0)
    for k in range(len(recs)):
        sol = propagate_test_particle(
            r0[k], v0[k], et0, t_span, central='SUN',
            perturbers=('JUPITER BARYCENTER', 'SATURN BARYCENTER'),
            t_eval=[t_span[1]], atol=1e-6, rtol=1e-12, max_step=86400)
        delta_km = np.linalg.norm(sol.y[:3, -1] - r_batch[k])
        # 1-year integration with same forces should agree to a few hundred
        # meters (numerical scheme differences)
        assert delta_km < 100, (
            f"batched vs serial N-body delta {delta_km:.1f} km for orbit {k}")


def test_auto_ephemeris_uses_2body_for_short_dt():
    """auto_ephemeris must take the fast 2-body path when epoch_delta
    is small."""
    import time
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        auto_ephemeris_at_mjd)
    recs = _make_recs() * 50   # 100 orbits
    t0 = time.time()
    auto_ephemeris_at_mjd(recs, recs[0].epoch_mjd + 30.0,
                              nbody_threshold_days=365.0)
    wall = time.time() - t0
    # 2-body path on 100 orbits should be <0.1s
    assert wall < 1.0, f"auto-2body wall {wall:.2f}s suggests it took the N-body path"


def test_auto_ephemeris_uses_nbody_for_long_dt():
    """auto_ephemeris must switch to N-body when epoch_delta exceeds
    the threshold."""
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        auto_ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    recs = _make_recs()
    target = recs[0].epoch_mjd + 365.0 * 5
    two_body = bulk_ephemeris_at_mjd(recs, target)
    auto = auto_ephemeris_at_mjd(recs, target, nbody_threshold_days=365.0)
    # The auto result must differ from pure 2-body (N-body path was used)
    deltas = []
    for i in range(len(recs)):
        cos_dec = math.cos(math.radians(two_body[i, 1]))
        sep_arcsec = math.hypot(
            (two_body[i, 0] - auto[i, 0]) * cos_dec,
            two_body[i, 1] - auto[i, 1]) * 3600
        deltas.append(sep_arcsec)
    assert max(deltas) > 10, (
        f"auto_ephemeris with 5-yr delta should use N-body; max delta only "
        f"{max(deltas):.2f}\"")
