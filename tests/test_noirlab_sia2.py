"""Tests for NOIRLab SIA2 client."""
from __future__ import annotations

import pytest


def test_exposure_record_dataclass():
    from ariadne.discovery.imaging.noirlab_sia2 import ExposureRecord
    r = ExposureRecord(
        archive_id="abc", instrument="DECam", proc_type="instcal",
        obs_mjd=60450.0, exptime_s=90.0, band="r",
        ra_center=180.0, dec_center=-10.0)
    assert r.archive_id == "abc"
    assert r.band == "r"
    assert r.extras == {}


def test_query_returns_empty_when_pyvo_unavailable(monkeypatch):
    """Force ImportError on pyvo path; query should return [] gracefully."""
    from ariadne.discovery.imaging import noirlab_sia2

    # Simulate pyvo missing by replacing it with None via sys.modules
    import sys
    orig = sys.modules.get("pyvo")
    sys.modules["pyvo"] = None
    try:
        out = noirlab_sia2.query_decam_exposures(ra=180, dec=-10, radius_deg=0.1)
        assert out == []
    finally:
        if orig is not None:
            sys.modules["pyvo"] = orig
        else:
            sys.modules.pop("pyvo", None)


def test_download_decam_exposure_no_url_returns_none(tmp_path):
    from ariadne.discovery.imaging.noirlab_sia2 import (
        ExposureRecord, download_decam_exposure)
    r = ExposureRecord(
        archive_id="abc", instrument="DECam", proc_type="instcal",
        obs_mjd=60450.0, exptime_s=90.0, band="r",
        ra_center=180.0, dec_center=-10.0, data_url="")
    out = download_decam_exposure(r, tmp_path)
    assert out is None


def test_download_decam_exposure_bad_url_returns_none(tmp_path):
    """Truly-unreachable URL should return None, not raise."""
    from ariadne.discovery.imaging.noirlab_sia2 import (
        ExposureRecord, download_decam_exposure)
    r = ExposureRecord(
        archive_id="abc", instrument="DECam", proc_type="instcal",
        obs_mjd=60450.0, exptime_s=90.0, band="r",
        ra_center=180.0, dec_center=-10.0,
        data_url="http://does-not-exist.invalid/file.fits")
    out = download_decam_exposure(r, tmp_path, timeout_s=5.0)
    assert out is None


def test_ping_noirlab_doesnt_raise():
    """Quick reachability test should never raise, only return True/False."""
    from ariadne.discovery.imaging.noirlab_sia2 import _ping_noirlab
    result = _ping_noirlab()
    assert isinstance(result, bool)
