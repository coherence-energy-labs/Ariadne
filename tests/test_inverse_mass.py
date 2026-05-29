"""Stage 28 tests: inverse hidden-mass localizer + two-tier honing + sensitivity map."""
import numpy as np

from ariadne.fields.hidden_mass import (CLUSTERED_ETNOS, PLANET9, elements_to_position,
                                        GM_EARTH, AU_KM)
from ariadne.discovery.inverse_mass import (simulate_observations, localize, sky_box,
                                            localization_vs_n, sensitivity_skymap)

NOISE = 1e-14


def _tracked():
    return [elements_to_position(o["a_au"], o["e"], o["i"], o["Omega"], o["omega"], 180.0)
            for o in CLUSTERED_ETNOS]


def test_inverse_localizer_recovers_injected_body():
    tracked = _tracked()
    gm = PLANET9["m_earth"] * GM_EARTH
    pos = elements_to_position(PLANET9["a_au"], PLANET9["e"], PLANET9["i"],
                               PLANET9["Omega"], PLANET9["omega"], 180.0)
    obs = simulate_observations(tracked, gm, pos, NOISE, seed=1)
    rec = localize(tracked, obs, NOISE)
    assert rec["success"]
    assert 0.3 < rec["m_earth"] / PLANET9["m_earth"] < 3.0      # mass within ~2x
    assert np.linalg.norm(rec["position"] - pos) < 2.0 * rec["pos_sigma_km"]  # within ~1-2 sigma


def test_localization_region_shrinks_with_more_bodies():
    tracked = _tracked()
    gm = PLANET9["m_earth"] * GM_EARTH
    pos = elements_to_position(PLANET9["a_au"], PLANET9["e"], PLANET9["i"],
                               PLANET9["Omega"], PLANET9["omega"], 180.0)
    hone = localization_vs_n(tracked, gm, pos, NOISE, seed=1)
    assert hone[-1]["pos_sigma_au"] < hone[0]["pos_sigma_au"]   # the two-tier honing works


def test_single_body_is_degenerate():
    tracked = _tracked()
    gm = PLANET9["m_earth"] * GM_EARTH
    pos = elements_to_position(PLANET9["a_au"], PLANET9["e"], PLANET9["i"],
                               PLANET9["Omega"], PLANET9["omega"], 180.0)
    full = localize(tracked, simulate_observations(tracked, gm, pos, NOISE, 1), NOISE)
    one = localize(tracked[:1], simulate_observations(tracked[:1], gm, pos, NOISE, 1), NOISE)
    # a single tracked body cannot pin the location: far larger (or non-finite) uncertainty
    assert (not np.isfinite(one["pos_sigma_km"])) or one["pos_sigma_km"] > 5 * full["pos_sigma_km"]


def test_sky_box_and_sensitivity_map():
    tracked = _tracked()
    box = sky_box(np.array([400 * AU_KM, 100 * AU_KM, -50 * AU_KM]), 50 * AU_KM)
    assert 0 <= box["ecliptic_lon_deg"] < 360 and -90 <= box["ecliptic_lat_deg"] <= 90
    assert box["distance_au"] > 0
    sm = sensitivity_skymap(tracked, distance_au=500.0, noise_ms2=NOISE, n_lon=12, n_lat=6)
    assert sm["min_mass_earth"].shape == (6, 12)
    assert np.all(np.isfinite(sm["min_mass_earth"])) and np.all(sm["min_mass_earth"] > 0)
