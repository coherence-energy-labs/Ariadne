"""Validation tests for the characterization engine Stages 2-4:
color taxonomy, light-curve periodicity, transient/variable detection, and the
top-level identify() fusion.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------- color
def test_color_s_vs_c_complex():
    from ariadne.discovery.imaging.color import classify_color, asteroid_a_star
    s = classify_color({"g-r": 0.70, "r-i": 0.30})      # red -> a*>0 -> S
    c = classify_color({"g-r": 0.45, "r-i": 0.12})      # neutral -> a*<0 -> C
    assert asteroid_a_star({"g-r": 0.70, "r-i": 0.30}) > 0
    assert asteroid_a_star({"g-r": 0.45, "r-i": 0.12}) < 0
    assert max(s["taxonomy"], key=s["taxonomy"].get).startswith("S-complex")
    assert max(c["taxonomy"], key=c["taxonomy"].get).startswith("C-complex")


def test_color_v_type_and_red_tno():
    from ariadne.discovery.imaging.color import classify_color
    v = classify_color({"g-r": 0.6, "r-i": 0.2, "i-z": -0.35})
    assert any("V-type" in k for k in v["taxonomy"])
    tno = classify_color({"g-r": 1.1, "r-i": 0.7}, is_distant=True)
    assert max(tno["taxonomy"], key=tno["taxonomy"].get).startswith("red TNO")


# ---------------------------------------------------------------- light curve
def test_lightcurve_recovers_injected_period():
    pytest.importorskip("astropy")
    from ariadne.discovery.imaging.light_curve import analyze_light_curve
    rng = np.random.default_rng(0)
    P = 0.5
    t = np.sort(rng.uniform(0, 6.0, 60))            # ~12 cycles
    m = 18.0 + 0.4 * np.sin(2 * math.pi * t / P) + rng.normal(0, 0.03, len(t))
    r = analyze_light_curve(t, m)
    assert r.best_period is not None
    # recovered period within 5% of truth (or its 2x alias)
    assert min(abs(r.best_period - P), abs(r.best_period - 2 * P)) < 0.05 * P
    assert r.false_alarm_prob < 0.01


def test_lightcurve_eclipse_shape():
    pytest.importorskip("astropy")
    from ariadne.discovery.imaging.light_curve import analyze_light_curve
    rng = np.random.default_rng(1)
    P = 0.8
    t = np.sort(rng.uniform(0, 8.0, 80))
    phase = (t / P) % 1.0
    m = 18.0 + rng.normal(0, 0.02, len(t))
    m[phase < 0.08] += 0.9                           # narrow eclipse
    r = analyze_light_curve(t, m)
    assert r.shape == "eclipse"
    assert max(r.var_type, key=r.var_type.get) == "eclipsing binary"


def test_lightcurve_flat_is_nonvariable():
    pytest.importorskip("astropy")
    from ariadne.discovery.imaging.light_curve import analyze_light_curve
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 5, 40)); m = 18.0 + rng.normal(0, 0.01, len(t))
    r = analyze_light_curve(t, m)
    assert r.shape == "flat"


# ---------------------------------------------------------------- transient detection
def _gauss(img, x, y, flux, sigma=2.0):
    H, W = img.shape
    h = int(math.ceil(5 * sigma))
    x0, x1 = max(0, int(x) - h), min(W, int(x) + h + 1)
    y0, y1 = max(0, int(y) - h), min(H, int(y) + h + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += flux / (2 * math.pi * sigma ** 2) * np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def _wcs(npix=240, pixscale=0.5):
    from astropy.wcs import WCS
    w = WCS(naxis=2); w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.crval = [150.0, 2.0]; w.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_transient_detection_finds_variable_and_transient_not_stars():
    pytest.importorskip("photutils"); pytest.importorskip("scipy")
    from ariadne.discovery.imaging.transient_detection import find_variability_candidates
    npix = 240; w = _wcs(npix)
    rng = np.random.default_rng(7)
    star_xy = [(40 + 30 * (i % 6), 40 + 30 * (i // 6)) for i in range(18)]  # 18 static stars
    var_xy = (180.0, 60.0); trans_xy = (60.0, 180.0)
    var_flux = [3e4, 5e4, 9e4, 6e4, 4e4]      # varies -> above median in 2 frames
    images, wcs_list, mjds = [], [], []
    for e in range(5):
        img = rng.normal(100.0, 5.0, (npix, npix))
        for (x, y) in star_xy:
            _gauss(img, x, y, 8e4)                       # constant stars
        _gauss(img, var_xy[0], var_xy[1], var_flux[e])   # variable
        if e == 1:
            _gauss(img, trans_xy[0], trans_xy[1], 1.2e5) # one-epoch transient
        images.append(img); wcs_list.append(w); mjds.append(60000.0 + e)
    cands = find_variability_candidates(images, wcs_list, mjds, fwhm_px=2.0 * 2.3548,
                                        threshold_sigma=5.0, match_arcsec=3.0)
    # locate candidates near the planted variable + transient
    def near(cx, cy):
        rax, decx = w.pixel_to_world_values(cx, cy)
        return [c for c in cands
                if math.hypot((c.ra - float(rax)) * math.cos(math.radians(c.dec)),
                              c.dec - float(decx)) * 3600 < 6]
    assert near(*var_xy), "variable star not detected"
    assert near(*trans_xy), "transient not detected"
    # the 18 static stars must NOT all become candidates (they cancel)
    assert len(cands) <= 6, f"too many candidates ({len(cands)}) -> stars leaking"


# ---------------------------------------------------------------- identify() fusion
def test_identify_mover_with_color_and_size():
    from ariadne.discovery.imaging.classifier import identify
    AU = 1.495978707e8
    ident = identify({"kind": "mover", "rate_arcsec_hr": 36.0, "v_mag": 19.0,
                      "observer_helio_km": [AU, 0, 0], "los_unit": [1, 0, 0],
                      "colors": {"g-r": 0.7, "r-i": 0.3}})
    assert "belt" in ident.label.lower()
    assert "S-type" in ident.label or "S-complex" in str(ident.posterior) or "S" in ident.label
    assert "km" in ident.label and "NEW candidate" in ident.label
    assert ident.evidence and ident.next_step


def test_identify_variable():
    pytest.importorskip("astropy")
    from ariadne.discovery.imaging.classifier import identify
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 6, 60)); m = 18 + 0.5 * np.sin(2 * math.pi * t / 0.6) + rng.normal(0, 0.03, len(t))
    ident = identify({"kind": "variable", "times": t, "mags": m})
    assert ident.posterior and ident.confidence in ("low", "moderate", "high")


# ---------------------------------------------------------------- calibrated CIs
def test_size_confidence_interval_reported():
    from ariadne.discovery.imaging.characterize import characterize_mover
    AU = 1.495978707e8
    d = characterize_mover(rate_arcsec_hr=36.0, v_mag=20.0,
                           observer_helio_km=[AU, 0, 0], los_unit=[1, 0, 0])
    p = d.properties
    assert "size_lo_km" in p and "size_hi_km" in p
    assert 0 < p["size_lo_km"] < p["size_hi_km"]      # a real interval, not a point
    assert "[" in d.headline                          # range shown to the user


# ---------------------------------------------------------------- robustness / edge cases
def test_edge_cases_dont_crash():
    from ariadne.discovery.imaging.characterize import characterize_mover, characterize_variable
    from ariadne.discovery.imaging.color import classify_color
    from ariadne.discovery.imaging.classifier import identify
    AU = 1.495978707e8; OBS = [AU, 0, 0]; LOS = [1, 0, 0]
    # zero / undetermined rate -> graceful, low confidence, no crash
    d = characterize_mover(rate_arcsec_hr=0.0, v_mag=20.0, observer_helio_km=OBS, los_unit=LOS)
    assert d.confidence == "low" and d.headline
    # too-few-point light curve
    v = characterize_variable([0, 1, 2], [18, 18, 18])
    assert "too few" in v.headline
    # empty / single-band color
    assert classify_color({})["taxonomy"] == {}
    assert classify_color({"g": 19.0})["taxonomy"] == {}    # one band -> no color
    # identify never throws on a degenerate mover
    ident = identify({"kind": "mover", "rate_arcsec_hr": 0.0, "v_mag": 25.0,
                      "observer_helio_km": OBS, "los_unit": LOS})
    assert ident.label


def test_coherence_classifier_color_breaks_degeneracy():
    """The same short-period sinusoid is RR Lyrae if BLUE, contact binary if RED
    -- color is the degeneracy-breaking axis in the coherence field. (Validated
    on real ZTF data: EW type accuracy 42% -> 100% when color is added.)"""
    from ariadne.discovery.imaging.coherence_classifier import classify_variable, most_coherent
    blue = classify_variable(period=0.30, R21=0.12, amplitude=0.5, g_r=0.25)
    red = classify_variable(period=0.30, R21=0.12, amplitude=0.5, g_r=0.80)
    assert "RR" in most_coherent(blue) or "sinusoidal" in most_coherent(blue), blue
    assert "contact" in most_coherent(red) or "EW" in most_coherent(red), red
    # eclipse shape routes to Algol-type regardless of color
    ecl = classify_variable(period=2.0, R21=0.3, amplitude=1.0, g_r=0.5, eclipse=True)
    assert "Algol" in most_coherent(ecl) or "eclipsing" in most_coherent(ecl)
    # period leads: a 5-day red object is a Cepheid, not a (red) Mira
    cep = classify_variable(period=5.0, R21=0.4, amplitude=0.6, g_r=1.5)
    assert most_coherent(cep) == "Cepheid", cep


def test_lightcurve_handles_nan_and_sparse():
    pytest.importorskip("astropy")
    from ariadne.discovery.imaging.light_curve import analyze_light_curve
    r = analyze_light_curve([0, 1, 2, 3], [18, np.nan, 18, 18])   # sparse + NaN
    assert r.best_period is None and "point" in r.notes.lower()
