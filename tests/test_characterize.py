"""Tests for the characterization spine: spot -> analyze -> identify.

Verifies the unified engine produces sensible verdicts across the object
taxonomy (main-belt, TNO, comet, NEO, interstellar) and the variability
families (eclipse/occultation dip, microlensing brightening), with honest
confidence + disambiguation hints.
"""
from __future__ import annotations

import numpy as np

AU_KM = 1.495978707e8
# opposition geometry: observer at +1 AU, line of sight anti-sunward
OBS = [AU_KM, 0.0, 0.0]
LOS = [1.0, 0.0, 0.0]


def _top(dossier):
    return max(dossier.type_probabilities, key=dossier.type_probabilities.get)


def test_main_belt_mover():
    from ariadne.discovery.imaging.characterize import characterize_mover
    # a true main-belt object at ~2.5 AU moves ~36"/hr at opposition (faster
    # than the outer belt) -- the rate->distance physics ties rate to class.
    d = characterize_mover(rate_arcsec_hr=36.0, v_mag=20.0,
                           observer_helio_km=OBS, los_unit=LOS)
    assert "belt" in _top(d).lower(), d.headline
    assert 2.0 < d.properties["helio_r_au"] < 3.3
    assert d.properties["size_km_est"] > 0


def test_tno_mover():
    from ariadne.discovery.imaging.characterize import characterize_mover
    d = characterize_mover(rate_arcsec_hr=3.0, v_mag=22.5,
                           observer_helio_km=OBS, los_unit=LOS)
    assert "TNO" in _top(d) or "Centaur" in _top(d), d.headline
    assert d.properties["helio_r_au"] > 15


def test_comet_flagged_from_morphology():
    from ariadne.discovery.imaging.characterize import characterize_mover
    d = characterize_mover(rate_arcsec_hr=20.0, v_mag=19.0,
                           observer_helio_km=OBS, los_unit=LOS,
                           morphology="extended")
    assert any("coma" in f or "comet" in f.lower() for f in d.flags), d.flags


def test_interstellar_from_hyperbolic_orbit():
    from ariadne.discovery.imaging.characterize import characterize_mover
    d = characterize_mover(rate_arcsec_hr=80.0, v_mag=20.0,
                           observer_helio_km=OBS, los_unit=LOS,
                           orbit_elements=(-3.0, 1.3, 25.0), n_nights=3)
    assert "interstellar" in _top(d).lower() or any("HYPERBOLIC" in f for f in d.flags)


def test_neo_from_orbit_elements():
    from ariadne.discovery.imaging.characterize import classify_orbit
    assert classify_orbit(1.1, 0.4, 5.0) == "NEO"          # perihelion 0.66 AU
    assert classify_orbit(2.7, 0.1, 10.0) == "main-belt"


def test_unknown_vs_known_disambiguation():
    from ariadne.discovery.imaging.characterize import characterize_mover
    d = characterize_mover(rate_arcsec_hr=25.0, v_mag=20.0,
                           observer_helio_km=OBS, los_unit=LOS, n_nights=1)
    # single night, no match -> must say what would sharpen it
    assert any("second night" in s for s in d.disambiguate)
    assert any("match" in s for s in d.disambiguate)
    assert d.confidence in ("low", "moderate")


def test_variable_dip_is_eclipse_or_occultation():
    from ariadne.discovery.imaging.characterize import characterize_variable
    t = np.linspace(0, 10, 40)
    m = np.full_like(t, 18.0)
    m[18:23] += 0.8           # a dip (fainter)
    d = characterize_variable(t, m)
    assert "eclips" in _top(d).lower() or "occult" in _top(d).lower() or "transit" in _top(d).lower()


def test_variable_brightening_is_microlensing_family():
    from ariadne.discovery.imaging.characterize import characterize_variable
    t = np.linspace(0, 30, 50)
    m = np.full_like(t, 19.0)
    m -= 1.2 * np.exp(-((t - 15) ** 2) / (2 * 3.0 ** 2))   # symmetric brightening
    d = characterize_variable(t, m)
    assert _top(d) in ("microlensing", "nova/CV-outburst", "flare")


def test_dossier_has_headline_and_props():
    from ariadne.discovery.imaging.characterize import characterize
    d = characterize({"kind": "mover", "rate_arcsec_hr": 25.0, "v_mag": 20.0,
                      "observer_helio_km": OBS, "los_unit": LOS})
    assert d.headline and d.properties and d.type_probabilities
