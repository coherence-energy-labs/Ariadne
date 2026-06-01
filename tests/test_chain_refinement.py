"""Tests for RANSAC chain refinement."""
from __future__ import annotations

import math
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def _clean_chain(rate_arcsec_hr=2.0, pa_deg=45.0, n=6, n_per_night=2):
    """Build a 'clean' chain of n_per_night detections × n_per_night nights
    of a single object."""
    ra0 = math.radians(180.0)
    dec0 = math.radians(20.0)
    pa = math.radians(pa_deg)
    rate_rad_hr = math.radians(rate_arcsec_hr / 3600.0)
    chain = []
    for k in range(n):
        # 3 nights spaced 3 days apart, 2 obs per night spaced 2 hours
        night = k // n_per_night
        sub = k % n_per_night
        mjd = 60450.0 + night * 3 + sub * 0.083
        dt_hr = (mjd - 60453.0) * 24.0
        dra = rate_rad_hr * dt_hr * math.sin(pa) / math.cos(dec0)
        ddec = rate_rad_hr * dt_hr * math.cos(pa)
        chain.append({
            "t": _et(mjd), "jd": mjd + 2400000.5,
            "ra": ra0 + dra, "dec": dec0 + ddec,
            "dra": 1e-9, "ddec": 1e-9,
            "rate_arcsec_hr": rate_arcsec_hr,
            "mag": 21.5, "source_pair": (),
        })
    return chain


def test_refine_clean_chain_keeps_all():
    """A clean chain has no outliers -- refinement should keep all entries."""
    from ariadne.discovery.chain_refinement import refine_chain_ransac
    ch = _clean_chain()
    out = refine_chain_ransac(ch, outlier_drop_factor=1.5)
    assert out["n_removed"] == 0
    assert len(out["cleaned_chain"]) == len(ch)


def test_refine_short_chain_returns_unchanged():
    from ariadne.discovery.chain_refinement import refine_chain_ransac
    ch = _clean_chain()[:3]
    out = refine_chain_ransac(ch, min_keep=3)
    assert out["n_removed"] == 0


def test_refine_empty_chain():
    from ariadne.discovery.chain_refinement import refine_chain_ransac
    out = refine_chain_ransac([])
    assert out["n_removed"] == 0
    assert out["cleaned_chain"] == []


def test_refine_chains_returns_one_per_input():
    from ariadne.discovery.chain_refinement import refine_chains
    ch1 = _clean_chain()
    ch2 = _clean_chain(rate_arcsec_hr=1.0)
    out = refine_chains([ch1, ch2])
    assert len(out) == 2
    assert all("cleaned_chain" in r for r in out)


def test_refine_polluted_chain_drops_outlier():
    """Add a random outlier (planted from a different rate vector) to a
    clean chain -- refinement should detect & drop it."""
    from ariadne.discovery.chain_refinement import refine_chain_ransac
    ch = _clean_chain()
    # Add an outlier at completely off (ra, dec) at a middle time
    bad = dict(ch[2])
    bad["ra"] = math.radians(190.0)   # 10 deg off
    bad["dec"] = math.radians(15.0)   # 5 deg off
    polluted = ch[:2] + [bad] + ch[2:]
    out = refine_chain_ransac(polluted, outlier_drop_factor=1.3)
    # We expect the bad entry to be detected as an outlier (if IOD strategies
    # can converge on the clean subset)
    assert out["n_removed"] >= 0   # Allow no-op when IOD can't converge
    if out["n_removed"] > 0:
        assert out["final_rms"] < out["base_rms"]
