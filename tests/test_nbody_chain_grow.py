"""Tests for the N-body chain-grow module."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def test_nbody_step_zero_dt_returns_input():
    from ariadne.discovery.imaging.nbody_chain_grow import nbody_step
    r0 = np.array([1.5e8, 0.0, 0.0])
    v0 = np.array([0.0, 29.78, 0.0])
    r, v = nbody_step(r0, v0, _et(60450.0), _et(60450.0))
    np.testing.assert_allclose(r, r0)
    np.testing.assert_allclose(v, v0)


def test_nbody_step_conserves_energy_for_simple_orbit():
    """Sun-only orbit should conserve energy to ~few-percent over months."""
    from ariadne.discovery.imaging.nbody_chain_grow import nbody_step
    from ariadne.data.constants import GM_SUN, AU_KM
    # Circular Earth-like orbit
    r0 = np.array([1.0 * AU_KM, 0.0, 0.0])
    v_circ = math.sqrt(GM_SUN / (1.0 * AU_KM))
    v0 = np.array([0.0, v_circ, 0.0])
    t0 = _et(60450.0); t1 = _et(60450.0 + 90)   # 90 days
    r, v = nbody_step(r0, v0, t0, t1,
                       include_jupiter=False, include_saturn=False,
                       integrator="leapfrog", dt_max=86400.0)
    E0 = 0.5 * np.dot(v0, v0) - GM_SUN / float(np.linalg.norm(r0))
    E1 = 0.5 * np.dot(v, v) - GM_SUN / float(np.linalg.norm(r))
    # Energy conservation: 5% over 90 days with leapfrog at 1-day step
    assert abs(E1 - E0) / abs(E0) < 0.05


def test_predict_sky_position_returns_finite():
    from ariadne.discovery.imaging.nbody_chain_grow import predict_sky_position
    from ariadne.data.constants import AU_KM
    r = np.array([40 * AU_KM, 5 * AU_KM, 2 * AU_KM])
    ra, dec = predict_sky_position(r, _et(60450.0))
    assert 0 <= ra < 360
    assert -90 <= dec <= 90


def test_propagate_orbit_returns_sky_positions_per_epoch():
    from ariadne.discovery.imaging.nbody_chain_grow import (
        propagate_orbit_with_perturbations)
    from ariadne.data.constants import GM_SUN, AU_KM
    r0 = np.array([40 * AU_KM, 0.0, 0.0])
    v_circ = math.sqrt(GM_SUN / (40 * AU_KM))
    v0 = np.array([0.0, v_circ, 0.0])
    epochs = [_et(60450.0), _et(60453.0), _et(60456.0)]
    sky = propagate_orbit_with_perturbations(r0, v0, _et(60450.0), epochs,
                                                include_jupiter=False,
                                                include_saturn=False)
    assert len(sky) == 3
    for ra, dec in sky:
        assert 0 <= ra < 360
        assert -90 <= dec <= 90


def test_nbody_grow_empty_for_too_few_tracklets():
    from ariadne.discovery.imaging.nbody_chain_grow import nbody_grow_chain
    tracklets = [
        {"t": _et(60450.0), "ra": math.radians(180.0),
         "dec": math.radians(20.0), "rate_arcsec_hr": 2.0},
        {"t": _et(60453.0), "ra": math.radians(180.05),
         "dec": math.radians(20.0), "rate_arcsec_hr": 2.0},
    ]
    chains = nbody_grow_chain(tracklets)
    assert chains == []   # need 3+ tracklets


def test_nbody_grow_links_consistent_tno_chain():
    """A 3-night chain of perfectly predicted positions should be linked."""
    from ariadne.discovery.imaging.nbody_chain_grow import (
        nbody_grow_chain, propagate_orbit_with_perturbations,
        _seed_orbit_from_pair)
    # Plant a fake TNO at 50 AU with circular orbit; let _seed_orbit predict
    # and we set up tracklets at the predicted positions.
    from ariadne.data.constants import GM_SUN, AU_KM
    r0 = np.array([50 * AU_KM, 0.0, 0.0])
    v_circ = math.sqrt(GM_SUN / (50 * AU_KM))
    v0 = np.array([0.0, v_circ, 0.0])
    epochs = [_et(60450.0), _et(60453.0), _et(60456.0)]
    sky = propagate_orbit_with_perturbations(r0, v0, _et(60450.0), epochs,
                                                include_jupiter=False,
                                                include_saturn=False)
    tracklets = []
    for et, (ra, dec) in zip(epochs, sky):
        # 2 tracklets per night so we have enough seed material
        tracklets.append({"t": et, "ra": math.radians(ra),
                            "dec": math.radians(dec),
                            "rate_arcsec_hr": 2.0})
        tracklets.append({"t": et + 7200,
                            "ra": math.radians(ra + 1e-4),
                            "dec": math.radians(dec),
                            "rate_arcsec_hr": 2.0})
    chains = nbody_grow_chain(tracklets, rms_acceptance_arcsec=120,
                                use_nbody=False)   # kepler-only is much faster
    # Should find at least one chain of length 3
    assert any(len(c) >= 3 for c in chains)
