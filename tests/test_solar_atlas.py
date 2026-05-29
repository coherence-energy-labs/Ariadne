"""Stage 26 tests: solar-system catalog + the fair coherence test (corrects Stage 25)."""
import pytest

from ariadne.data.constants import SOLAR_SYSTEM, EARTH_MOON
from ariadne.fields.solar_atlas import system_catalog, coherence_skeleton_test


def test_solar_system_registry_is_comprehensive():
    names = {s.name for s in SOLAR_SYSTEM}
    # all 8 planets present as Sun-planet systems
    for p in ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        assert f"Sun-{p}" in names
    assert "Pluto-Charon" in names            # the extreme binary
    assert len(SOLAR_SYSTEM) >= 20
    mus = [s.mu for s in SOLAR_SYSTEM]
    assert min(mus) < 1e-7 and max(mus) > 0.1  # ~7 orders of magnitude


@pytest.mark.slow
def test_catalog_gives_periodic_libration_for_every_system():
    for S in SOLAR_SYSTEM:
        c = system_catalog(S)
        assert c["L1_km"] > 0 and c["lyap_period_d"] > 0
        assert c["half_period_residual"] < 1e-9


@pytest.mark.slow
def test_fair_test_refutes_the_stage25_skeleton_claim():
    """Under a FAIR region-matched comparison, the Stage-25 'tube is MORE coherent' claim fails."""
    t = coherence_skeleton_test(EARTH_MOON)
    assert t["ok"]
    assert not t["skeleton"]                  # the key correction: Stage-25 claim does NOT hold
    # (the tube is at most as coherent as its local background -- not more; often a mild separatrix)
    assert t["man_mean"] >= t["rnd_mean"] - 1e-6
