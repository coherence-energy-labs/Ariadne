"""Tests for the multi-extension DECam instcal handler."""
from __future__ import annotations

import math
import numpy as np
import pytest


def test_classify_extension_science():
    from ariadne.discovery.imaging.decam_instcal import _classify_extension
    assert _classify_extension("SCI", {}) == "science"
    assert _classify_extension("IMAGE", {}) == "science"
    assert _classify_extension("UNKNOWN", {"BUNIT": "ELECTRONS"}) == "science"


def test_classify_extension_dqm():
    from ariadne.discovery.imaging.decam_instcal import _classify_extension
    assert _classify_extension("DQM", {}) == "dqm"
    assert _classify_extension("DQMASK", {}) == "dqm"


def test_classify_extension_weight():
    from ariadne.discovery.imaging.decam_instcal import _classify_extension
    assert _classify_extension("WGT", {}) == "weight"
    assert _classify_extension("WEIGHT", {}) == "weight"


def test_apply_dqm_mask_replaces_bad_pixels_with_nan():
    from ariadne.discovery.imaging.decam_instcal import apply_dqm_mask
    img = np.ones((10, 10), dtype=float) * 100
    dqm = np.zeros_like(img, dtype=np.uint8)
    dqm[3, 4] = 2   # saturated bit
    dqm[7, 7] = 8   # cosmic-ray bit
    out = apply_dqm_mask(img, dqm)
    assert np.isnan(out[3, 4])
    assert np.isnan(out[7, 7])
    # Everything else preserved
    assert out[0, 0] == 100
    assert out[5, 5] == 100


def test_calibrate_magnitudes_basic():
    from ariadne.discovery.imaging.decam_instcal import calibrate_magnitudes
    # flux=1 with ZP=25 -> mag = -2.5 log10(1) + 25 = 25
    assert calibrate_magnitudes(1.0, 25.0) == pytest.approx(25.0)
    # flux=100 -> mag = -5 + 25 = 20
    assert calibrate_magnitudes(100.0, 25.0) == pytest.approx(20.0)
    # negative flux -> NaN
    result = calibrate_magnitudes(-5.0, 25.0)
    assert np.isnan(result)


def test_calibrate_magnitudes_vectorised():
    from ariadne.discovery.imaging.decam_instcal import calibrate_magnitudes
    fluxes = np.array([1.0, 10.0, 100.0, -1.0, 1000.0])
    mags = calibrate_magnitudes(fluxes, 25.0)
    assert mags[0] == pytest.approx(25.0)
    assert mags[1] == pytest.approx(22.5)
    assert mags[2] == pytest.approx(20.0)
    assert np.isnan(mags[3])
    assert mags[4] == pytest.approx(17.5)


def test_load_decam_instcal_handles_missing_file(tmp_path):
    """Loading a non-existent file should raise -- we don't silently produce
    empty results."""
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    fake_path = tmp_path / "no_such_file.fits"
    with pytest.raises(Exception):
        load_decam_instcal(fake_path)


def test_load_decam_instcal_on_synthesised_multi_ext_fits(tmp_path):
    """Synthesise a tiny multi-extension FITS that mimics a DECam mosaic
    and verify the loader parses it correctly."""
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS

    # Build a 3-CCD mosaic, each with SCI / WGT / DQM extensions.
    hdul = astrofits.HDUList([astrofits.PrimaryHDU()])
    hdul[0].header["MJD-OBS"] = 60450.0
    hdul[0].header["FILTER"] = "r"
    hdul[0].header["EXPTIME"] = 90.0
    hdul[0].header["EXPNUM"] = 123456
    hdul[0].header["MAGZERO"] = 30.5
    for ccdnum in (1, 2, 3):
        sci = astrofits.ImageHDU(
            data=np.random.normal(100, 10, size=(64, 64)).astype(np.float32))
        sci.header["EXTNAME"] = "SCI"
        sci.header["CCDNUM"] = ccdnum
        sci.header["MAGZERO"] = 30.5 + 0.01 * ccdnum  # per-CCD ZP
        w = WCS(naxis=2)
        w.wcs.crpix = [32, 32]
        w.wcs.crval = [180.0 + 0.001 * ccdnum, -10.0]
        w.wcs.cd = [[-1/3600.0, 0], [0, 1/3600.0]]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        sci.header.update(w.to_header())
        hdul.append(sci)
        wgt = astrofits.ImageHDU(
            data=np.ones((64, 64), dtype=np.float32))
        wgt.header["EXTNAME"] = "WGT"
        wgt.header["CCDNUM"] = ccdnum
        hdul.append(wgt)
        dqm = astrofits.ImageHDU(
            data=np.zeros((64, 64), dtype=np.uint8))
        dqm.header["EXTNAME"] = "DQM"
        dqm.header["CCDNUM"] = ccdnum
        hdul.append(dqm)

    path = tmp_path / "fake_decam_instcal.fits"
    hdul.writeto(path)

    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    inst = load_decam_instcal(path, read_dqm=True, read_weight=True)
    assert inst.n_ccds == 3
    assert inst.mjd == 60450.0
    assert inst.band == "r"
    assert inst.exptime_s == 90.0
    # Per-CCD ZPs override the global ZP
    for i, ccd in enumerate(inst.ccds, start=1):
        assert ccd.ccdnum == i
        assert ccd.magzero == pytest.approx(30.5 + 0.01 * i)
        assert ccd.science.shape == (64, 64)


def test_iterate_ccds_applies_dqm_mask(tmp_path):
    """Test that iterate_ccds with mask_dqm=True replaces bad pixels with NaN."""
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.decam_instcal import (
        load_decam_instcal, iterate_ccds)

    hdul = astrofits.HDUList([astrofits.PrimaryHDU()])
    hdul[0].header["MJD-OBS"] = 60450.0
    hdul[0].header["FILTER"] = "r"
    sci = astrofits.ImageHDU(
        data=np.ones((32, 32), dtype=np.float32) * 100)
    sci.header["EXTNAME"] = "SCI"
    sci.header["CCDNUM"] = 1
    sci.header["MAGZERO"] = 30.0
    hdul.append(sci)
    dqm_data = np.zeros((32, 32), dtype=np.uint8)
    dqm_data[10, 10] = 1
    dqm_data[20, 20] = 4
    dqm = astrofits.ImageHDU(data=dqm_data)
    dqm.header["EXTNAME"] = "DQM"
    dqm.header["CCDNUM"] = 1
    hdul.append(dqm)

    path = tmp_path / "fake.fits"
    hdul.writeto(path)

    inst = load_decam_instcal(path, read_dqm=True)
    assert inst.n_ccds == 1
    ccds = list(iterate_ccds(inst, mask_dqm=True))
    assert np.isnan(ccds[0].science[10, 10])
    assert np.isnan(ccds[0].science[20, 20])
    assert ccds[0].science[0, 0] == 100
