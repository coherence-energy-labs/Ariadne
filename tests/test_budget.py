"""Delta-v budget tests (Gates G8b, G8c)."""
from ariadne.optimize.budget import circular_speed, earth_moon_budget
from ariadne.data.constants import GM_EARTH, R_EARTH

B = earth_moon_budget()


def test_circular_speeds_textbook():
    assert abs(B["v_circ_leo"] - 7.784) < 0.02      # 200 km LEO
    assert abs(B["v_circ_llo"] - 1.633) < 0.02      # 100 km LLO


def test_tli_apollo_class():
    assert abs(B["dv_tli"] - 3.131) < 0.05          # classic TLI ~3.13 km/s


def test_direct_total_matches_known():
    # Apollo-class LEO->LLO direct transfer; Coimbra "previous best" ~3992 m/s
    assert 3.85 < B["total_direct"] < 4.05


def test_ballistic_capture_saves_fuel():
    assert B["capture_saving"] > 0.1                # ~145 m/s
    assert B["total_low_energy"] < B["total_direct"]


def test_vis_viva_circular_consistency():
    # vis-viva with a = r reduces to circular speed
    r = R_EARTH + 200.0
    assert abs(circular_speed(GM_EARTH, r) - (GM_EARTH / r) ** 0.5) < 1e-12
