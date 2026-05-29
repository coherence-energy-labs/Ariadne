"""Stage 22 tests: gravity-assist multi-flyby chains."""
import numpy as np
import pytest

from ariadne.data.ephemeris import et
from ariadne.interplanetary.flyby import evaluate_chain, reference_veega, _max_turn
from ariadne.data.constants import GM_EARTH, R_EARTH


def test_max_turn_decreases_with_vinf():
    """Turn authority drops as v_inf rises (why high-energy flybys can't turn much)."""
    t_lo = _max_turn(3.0, GM_EARTH, R_EARTH + 300.0)
    t_hi = _max_turn(10.0, GM_EARTH, R_EARTH + 300.0)
    assert t_lo > t_hi > 0.0


@pytest.mark.slow
def test_reference_veega_is_feasible_and_low_c3():
    v = reference_veega()
    assert v is not None
    assert v["c3"] < 25.0                              # Galileo-class, vs direct ~85
    assert all(f["feasible"] for f in v["flybys"])     # all flybys within turn authority
    assert 3.0 < v["arr_vinf_kms"] < 9.0               # sensible Jupiter arrival v_inf
    # a feasible VEEGA, though not fully ballistic (powered-flyby Delta-v is bounded)
    assert sum(f["mismatch_ms"] for f in v["flybys"]) < 4000.0


@pytest.mark.slow
def test_direct_jupiter_needs_high_c3():
    """Sanity: the DIRECT Earth->Jupiter transfer needs a large launch C3 (the GA motivation)."""
    from ariadne.data.constants import GM_SUN
    from ariadne.data.ephemeris import body_state
    from ariadne.optimize.lambert import lambert
    e0 = et("2029-06-01T00:00:00"); tof = 950 * 86400.0
    se = body_state("EARTH", e0, "J2000", "SUN")
    sj = body_state("JUPITER BARYCENTER", e0 + tof, "J2000", "SUN")
    v1, _ = lambert(se[:3], sj[:3], tof, GM_SUN)
    c3 = float(np.dot(v1 - se[3:], v1 - se[3:]))
    assert c3 > 60.0                                   # direct is expensive
