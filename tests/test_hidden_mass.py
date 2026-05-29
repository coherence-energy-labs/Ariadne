"""Stage 27 tests: principled coherence field (tau_c) + trajectory-residual hidden-mass detector."""
import numpy as np
import pytest

from ariadne.data.constants import GM_SUN, GM_EARTH
from ariadne.fields.tau_c import (newtonian_accel, coherence_accel, coherence_accel_fd,
                                  tau_c, potential, C2)
from ariadne.fields.hidden_mass import (CLUSTERED_ETNOS, PLANET9, REFERENCE_BODIES,
                                        elements_to_position, residual_accel, kuiper_noise_floor,
                                        residual_magnitude, detectability_map, AU_KM, GM_EARTH as GME)


def test_detector_generalizes_to_any_body():
    """The residual method is body-agnostic: closer/more-massive bodies are more detectable."""
    # Ceres in the belt vs a distant hidden planet -- both produce computable residuals
    near = residual_magnitude(1.57e-4, 2.77)      # Ceres
    far = residual_magnitude(6.0, 500.0)          # Planet 9
    assert near > 0 and far > 0
    # a more massive body at the same distance is always more detectable
    assert residual_magnitude(1.0, 50.0) > residual_magnitude(1e-3, 50.0)
    # a closer body of the same mass is more detectable
    assert residual_magnitude(1.0, 5.0) > residual_magnitude(1.0, 50.0)
    dm = detectability_map([1e-6, 1e-3, 1.0], [5.0, 50.0, 500.0], 1e-15, 1e-13)
    assert dm["detectable"].shape == (3, 3)
    assert dm["detectable"][0, 2]                  # massive + close -> detectable


def test_coherence_field_reduces_to_newton_in_weak_field():
    masses = [(GM_SUN, np.zeros(3))]
    x = np.array([1.0 * AU_KM, 0.0, 0.0])
    gN, gC = newtonian_accel(x, masses), coherence_accel(x, masses)
    rel = np.linalg.norm(gC - gN) / np.linalg.norm(gN)
    # the deviation equals exactly |Phi|/c^2 (the framework's Newton recovery), and is tiny
    assert abs(rel - abs(potential(x, masses)) / C2) / (abs(potential(x, masses)) / C2) < 1e-3
    assert rel < 1e-7


def test_coherence_form_correct_in_strong_field():
    """-c^2 grad ln(tau_c) (finite diff) matches analytic g_N/tau_c where there is no cancellation."""
    masses = [(8.99e15, np.zeros(3))]
    x = np.array([1.0e6, 0.0, 0.0])
    assert abs(tau_c(x, masses) - 0.9) < 1e-3                  # Phi/c^2 ~ -0.1
    rel = np.linalg.norm(coherence_accel_fd(x, masses, 1.0) - coherence_accel(x, masses)) \
        / np.linalg.norm(coherence_accel(x, masses))
    assert rel < 1e-6


def test_residual_equals_model_difference():
    masses = [(GM_SUN, np.zeros(3))]
    gm_x, pos_x = 6 * GM_EARTH, np.array([500 * AU_KM, 0.0, 0.0])
    x = np.array([250 * AU_KM, 1e9, 0.0])
    res = residual_accel(x, gm_x, pos_x)
    diff = newtonian_accel(x, masses + [(gm_x, pos_x)]) - newtonian_accel(x, masses)
    assert np.linalg.norm(diff - res) < 1e-18                 # residual == with - without


def test_real_etno_dataset_is_authentic():
    names = {o["name"] for o in CLUSTERED_ETNOS}
    assert "Sedna" in names and "2012 VP113" in names         # canonical Batygin-Brown objects
    for o in CLUSTERED_ETNOS:
        assert o["a_au"] > 200 and 0.6 < o["e"] < 0.95         # extreme, detached orbits


def test_planet9_signal_rises_above_kuiper_floor():
    gm_x = PLANET9["m_earth"] * GME
    pos_x = elements_to_position(PLANET9["a_au"], PLANET9["e"], PLANET9["i"],
                                 PLANET9["Omega"], PLANET9["omega"], 180.0)
    x = elements_to_position(263.1, 0.70, 24.0, 90.8, 293.8, 180.0)   # 2012 VP113 aphelion
    sig = np.linalg.norm(residual_accel(x, gm_x, pos_x))
    floor = kuiper_noise_floor(x)
    assert sig > floor                                        # distinguishable from small-body noise
    # but tiny vs the Sun -> needs secular accumulation (the honest limit)
    solar = np.linalg.norm(newtonian_accel(x, [(GM_SUN, np.zeros(3))]))
    assert sig / solar < 1e-3
