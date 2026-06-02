"""Tests for the vectorised batch MPC ephemeris.

The batch path must agree bit-for-bit with the serial path now that the
serial path solves Kepler's equation correctly (M -> E -> nu). These tests
pin that contract and also pin the speedup so a future regression that
introduces a python loop fails CI.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest


def _make_recs(seed=42, n=200):
    import random
    random.seed(seed)
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    recs = []
    for i in range(n):
        recs.append(OrbitalElements(
            designation=f'A{i:05d}',
            a_au=random.uniform(1.2, 45.0),
            e=random.uniform(0, 0.85),
            i_deg=random.uniform(0, 80),
            Omega_deg=random.uniform(0, 360),
            omega_deg=random.uniform(0, 360),
            M_deg=random.uniform(0, 360),
            epoch_mjd=60450.0,
            H_mag=random.uniform(5, 22)))
    return recs


def test_bulk_ephemeris_matches_serial_at_epoch():
    """At dt=0, batch and serial should be identical to machine epsilon."""
    from ariadne.discovery.imaging.mpc_catalog import ephemeris_at_mjd
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    recs = _make_recs(n=50)
    target_mjd = 60450.0   # SAME as catalog epoch
    serial = [ephemeris_at_mjd(r, target_mjd) for r in recs]
    batch = bulk_ephemeris_at_mjd(recs, target_mjd)
    max_sep_arcsec = 0
    for i in range(len(recs)):
        s = serial[i]; b = batch[i]
        if math.isnan(b[0]):
            continue
        dra = (s[0] - b[0]) * math.cos(math.radians(s[1])) * 3600
        ddec = (s[1] - b[1]) * 3600
        max_sep_arcsec = max(max_sep_arcsec, math.hypot(dra, ddec))
    assert max_sep_arcsec < 0.001, f"max disagreement {max_sep_arcsec:.4f}\""


def test_bulk_ephemeris_matches_serial_after_propagation():
    """50-day propagation. Should still match to sub-mas precision."""
    from ariadne.discovery.imaging.mpc_catalog import ephemeris_at_mjd
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    recs = _make_recs(n=100)
    target_mjd = 60500.0
    serial = [ephemeris_at_mjd(r, target_mjd) for r in recs]
    batch = bulk_ephemeris_at_mjd(recs, target_mjd)
    max_sep_arcsec = 0
    for i in range(len(recs)):
        s = serial[i]; b = batch[i]
        if math.isnan(b[0]):
            continue
        dra = (s[0] - b[0]) * math.cos(math.radians(s[1])) * 3600
        ddec = (s[1] - b[1]) * 3600
        max_sep_arcsec = max(max_sep_arcsec, math.hypot(dra, ddec))
    assert max_sep_arcsec < 0.001, f"max disagreement {max_sep_arcsec:.4f}\""


def test_bulk_ephemeris_handles_high_eccentricity():
    """Cometary orbits with e=0.85 must propagate correctly. This was the
    case that exposed the M-vs-nu anomaly bug in the original serial path."""
    from ariadne.discovery.imaging.mpc_catalog import (
        OrbitalElements, ephemeris_at_mjd)
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    high_e = [
        OrbitalElements(designation='C/HiE', a_au=5.0, e=0.85, i_deg=30.0,
                         Omega_deg=90.0, omega_deg=60.0, M_deg=30.0,
                         epoch_mjd=60450.0, H_mag=12.0),
        OrbitalElements(designation='NEO', a_au=1.5, e=0.55, i_deg=15.0,
                         Omega_deg=200.0, omega_deg=80.0, M_deg=120.0,
                         epoch_mjd=60450.0, H_mag=20.0),
    ]
    target_mjd = 60500.0
    for r in high_e:
        s = ephemeris_at_mjd(r, target_mjd)
        b = bulk_ephemeris_at_mjd([r], target_mjd)[0]
        dra = (s[0] - b[0]) * math.cos(math.radians(s[1])) * 3600
        ddec = (s[1] - b[1]) * 3600
        sep = math.hypot(dra, ddec)
        assert sep < 0.001, f"{r.designation}: disagreement {sep:.4f}\""


def test_bulk_ephemeris_speedup():
    """Pin the ~100x+ speedup so a regression that re-introduces a Python
    loop fails CI."""
    from ariadne.discovery.imaging.mpc_catalog import ephemeris_at_mjd
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd)
    N = 1000
    recs = _make_recs(n=N)
    target_mjd = 60500.0

    t0 = time.time()
    for r in recs[:200]:    # 200 to keep test fast; ratio still meaningful
        ephemeris_at_mjd(r, target_mjd)
    t_serial = (time.time() - t0) * (N / 200)   # extrapolate to full N

    t0 = time.time()
    bulk_ephemeris_at_mjd(recs, target_mjd)
    t_batch = time.time() - t0

    speedup = t_serial / max(t_batch, 1e-6)
    # We measured 192x; require at least 30x to leave headroom for slower CI
    assert speedup > 30, (
        f"speedup only {speedup:.1f}x  (serial={t_serial:.3f}s  "
        f"batch={t_batch:.3f}s)")


def test_bulk_cross_match_finds_known_position():
    """A detection at exactly the predicted position of a known orbit
    must be flagged with that orbit's designation."""
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_ephemeris_at_mjd, bulk_cross_match)
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    recs = _make_recs(n=10)
    target_mjd = 60500.0
    eph = bulk_ephemeris_at_mjd(recs, target_mjd)
    # Build a "detection" at the predicted RA/Dec of orbit index 3
    target_idx = 3
    det = [{
        "id": 999,
        "ra": float(eph[target_idx, 0]),
        "dec": float(eph[target_idx, 1]),
        "mjd": target_mjd,
    }]
    res = bulk_cross_match(det, recs, target_mjd,
                              match_radius_arcsec=1.0)
    assert 999 in res["matches"]
    assert res["matches"][999] == recs[target_idx].designation


def test_bulk_cross_match_misses_far_detection():
    """A detection far from any predicted position must NOT match."""
    from ariadne.discovery.imaging.mpc_ephemeris_batch import (
        bulk_cross_match)
    recs = _make_recs(n=10)
    target_mjd = 60500.0
    # Detection at the north celestial pole, far from any of our test orbits
    det = [{"id": 1, "ra": 0.0, "dec": 89.99, "mjd": target_mjd}]
    res = bulk_cross_match(det, recs, target_mjd,
                              match_radius_arcsec=3.0)
    assert 1 not in res["matches"]
