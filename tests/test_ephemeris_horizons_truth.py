"""Absolute-correctness regression test: ephemeris vs baked-in JPL Horizons.

THE lesson from the ecliptic-vs-equatorial frame bug: every prior test
checked SELF-CONSISTENCY (serial vs batch, 2-body vs N-body at epoch) and
all passed while the ephemeris was ~100,000" wrong on the real sky,
because the error was common to every code path. The only thing that
catches that class of bug is an EXTERNAL ground truth.

These reference positions are JPL Horizons astrometric RA/Dec at CTIO
(observatory code 807) for mjd 60546.0643 (2024-08-24), baked in so the
test is deterministic and needs no network. They pin the full correction
chain: ecliptic->equatorial rotation, per-object light-time, topocentric
observer, and the 8-planet N-body force model.

If this test fails, the ephemeris has a systematic error -- do NOT trust
any cross-match until it passes again.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# mjd of the reference epoch (2024-08-24), ~1.2 yr from the MPCORB epoch.
REF_MJD = 60546.0643
CTIO = dict(lat=-30.169, lon=-70.806, height_m=2207.0)

# MPCORB osculating elements (epoch MJD 61000), as parsed by the pipeline.
ELEMENTS = {
    "1": dict(epoch_mjd=61000.0, a_au=2.7656157, e=0.0795764, i_deg=10.58789,
              Omega_deg=80.24964, omega_deg=73.29974, M_deg=231.53974, H_mag=3.34),
    "10": dict(epoch_mjd=61000.0, a_au=3.1475914, e=0.1082238, i_deg=3.83294,
               Omega_deg=283.12164, omega_deg=312.60583, M_deg=216.69031, H_mag=5.65),
    "100": dict(epoch_mjd=61000.0, a_au=3.0904457, e=0.1679761, i_deg=6.43092,
                Omega_deg=127.15593, omega_deg=183.55245, M_deg=323.244, H_mag=7.74),
    "74765": dict(epoch_mjd=61000.0, a_au=2.3334171, e=0.236823, i_deg=2.92605,
                  Omega_deg=157.94739, omega_deg=199.35172, M_deg=108.91273, H_mag=15.98),
    "25356": dict(epoch_mjd=61000.0, a_au=2.9467607, e=0.1055081, i_deg=11.83247,
                  Omega_deg=79.01103, omega_deg=33.06577, M_deg=315.81261, H_mag=12.9),
}

# JPL Horizons astrometric RA/Dec (deg) at CTIO, mjd 60546.0643.
HORIZONS_TRUTH = {
    "1": (278.277730, -30.949630),
    "10": (31.780750, 17.695070),
    "100": (172.891470, 7.233650),
    "74765": (330.748650, -11.000000),
    "25356": (339.354690, -26.442930),
}

TOL_ARCSEC = 2.0   # the pipeline achieves ~0.65" median; 2" is a safe gate


def _ctio_observer_geo_km(mjd):
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    import astropy.units as u
    loc = EarthLocation(lat=CTIO["lat"] * u.deg, lon=CTIO["lon"] * u.deg,
                          height=CTIO["height_m"] * u.m)
    g = loc.get_gcrs(Time(mjd, format="mjd", scale="utc")).cartesian
    return np.array([g.x.to(u.km).value, g.y.to(u.km).value,
                      g.z.to(u.km).value])


def _make_records(keys):
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    return [OrbitalElements(designation=k, **ELEMENTS[k]) for k in keys]


@pytest.mark.parametrize("desig", list(HORIZONS_TRUTH))
def test_nbody_matches_horizons_truth(desig):
    """Each numbered asteroid must land within TOL_ARCSEC of JPL Horizons."""
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_ephemeris_at_mjd_nbody)
    obs = _ctio_observer_geo_km(REF_MJD)
    eph = bulk_ephemeris_at_mjd_nbody(_make_records([desig]), REF_MJD,
                                        observer_geo_km=obs)[0]
    rah, dech = HORIZONS_TRUTH[desig]
    cd = math.cos(math.radians(dech))
    sep = math.hypot((eph[0] - rah) * cd, eph[1] - dech) * 3600
    assert sep <= TOL_ARCSEC, (
        f"{desig}: {sep:.2f}\" from Horizons truth (> {TOL_ARCSEC}\"). "
        "Ephemeris has a systematic error -- check frame rotation, "
        "light-time, topocentric observer, force model.")


def test_batch_does_not_corrupt_inner_belt_object():
    """Regression for the batch-median light-time bug: an inner-belt object
    (74765) propagated in a MIXED batch with outer objects must still match
    Horizons -- not drift tens of arcsec because the batch-median light-time
    was wrong for it."""
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_ephemeris_at_mjd_nbody)
    keys = ["1", "10", "100", "74765", "25356"]   # mixed a = 2.33..3.15 AU
    obs = _ctio_observer_geo_km(REF_MJD)
    eph = bulk_ephemeris_at_mjd_nbody(_make_records(keys), REF_MJD,
                                        observer_geo_km=obs)
    for i, k in enumerate(keys):
        rah, dech = HORIZONS_TRUTH[k]
        cd = math.cos(math.radians(dech))
        sep = math.hypot((eph[i, 0] - rah) * cd, eph[i, 1] - dech) * 3600
        assert sep <= TOL_ARCSEC, (
            f"{k} in mixed batch: {sep:.2f}\" (per-object light-time regressed?)")


# Diverse cross-class baked truth (CTIO/807, mjd 60546.0643): main belt,
# high-inclination (Pallas i=35), NEOs incl extreme-e (Icarus e=0.83),
# Trojan, Centaur, TNO, and Sedna (a=549, e=0.86). Locks in that the
# ephemeris stays sub-few-arcsec across the WHOLE orbit-class range.
_DIVERSE_ELEMENTS = {
    "1": dict(epoch_mjd=61000.0, a_au=2.7656157, e=0.0795764, i_deg=10.58789, Omega_deg=80.24964, omega_deg=73.29974, M_deg=231.53974, H_mag=3.34),
    "2": dict(epoch_mjd=61000.0, a_au=2.7699258, e=0.230643, i_deg=34.92833, Omega_deg=172.88859, omega_deg=310.9334, M_deg=211.52976, H_mag=4.12),
    "433": dict(epoch_mjd=61000.0, a_au=1.458121, e=0.222836, i_deg=10.82847, Omega_deg=304.2701, omega_deg=178.92976, M_deg=310.55432, H_mag=10.4),
    "1566": dict(epoch_mjd=61000.0, a_au=1.0780378, e=0.8270057, i_deg=22.8032, Omega_deg=87.95243, omega_deg=31.43821, M_deg=153.07893, H_mag=16.53),
    "624": dict(epoch_mjd=61000.0, a_au=5.276272, e=0.0242864, i_deg=18.14819, Omega_deg=342.80299, omega_deg=180.68365, M_deg=9.22945, H_mag=7.32),
    "2060": dict(epoch_mjd=61000.0, a_au=13.6921987, e=0.3789792, i_deg=6.926, Omega_deg=209.29853, omega_deg=339.25363, M_deg=212.83978, H_mag=5.54),
    "15760": dict(epoch_mjd=61000.0, a_au=44.1989361, e=0.0724959, i_deg=2.18703, Omega_deg=359.47464, omega_deg=6.89416, M_deg=35.05883, H_mag=7.18),
    "90377": dict(epoch_mjd=61000.0, a_au=549.5301384, e=0.8612949, i_deg=11.92592, Omega_deg=144.47863, omega_deg=311.00992, M_deg=358.60722, H_mag=1.5),
}
_DIVERSE_TRUTH = {
    "1": (278.277730, -30.949630), "2": (241.571200, 15.896130),
    "433": (178.347580, -6.649550), "1566": (248.686520, -32.338500),
    "624": (143.002010, 22.163460), "2060": (20.707460, 9.916910),
    "15760": (43.245010, 18.180430), "90377": (61.168260, 8.698330),
}


@pytest.mark.parametrize("desig", list(_DIVERSE_TRUTH))
def test_nbody_matches_horizons_across_orbit_classes(desig):
    """Sub-few-arcsec vs Horizons for the whole orbit-class range: main
    belt, hi-inclination, NEOs (incl e=0.83), Trojan, Centaur, TNO, Sedna."""
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    obs = _ctio_observer_geo_km(REF_MJD)
    rec = OrbitalElements(designation=desig, **_DIVERSE_ELEMENTS[desig])
    eph = bulk_ephemeris_at_mjd_nbody([rec], REF_MJD, observer_geo_km=obs)[0]
    rah, dech = _DIVERSE_TRUTH[desig]
    cd = math.cos(math.radians(dech))
    sep = math.hypot((eph[0] - rah) * cd, eph[1] - dech) * 3600
    # NEOs (fast/close) are the largest residual; 3.5" is a safe gate
    assert sep <= 3.5, f"{desig}: {sep:.2f}\" from Horizons (orbit-class regression)"


def test_frame_rotation_is_applied():
    """A blunt guard: WITHOUT the ecliptic->equatorial rotation the error is
    ~tens of thousands of arcsec. Confirm we are nowhere near that regime."""
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import (
        bulk_ephemeris_at_mjd_nbody)
    obs = _ctio_observer_geo_km(REF_MJD)
    recs = _make_records(list(HORIZONS_TRUTH))
    eph = bulk_ephemeris_at_mjd_nbody(recs, REF_MJD, observer_geo_km=obs)
    worst = 0.0
    for i, k in enumerate(HORIZONS_TRUTH):
        rah, dech = HORIZONS_TRUTH[k]
        cd = math.cos(math.radians(dech))
        worst = max(worst, math.hypot((eph[i, 0] - rah) * cd,
                                        eph[i, 1] - dech) * 3600)
    assert worst < 100.0, f"worst {worst:.0f}\" -- frame/units regression"
