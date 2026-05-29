"""Stage 25 tests: the coherence (FLI) field + the falsifiable transport-structure test."""
import numpy as np
import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.fields.coherence_field import fli, accessible_speed, _prograde_velocity

MU = EARTH_MOON.mu


def test_accessible_speed_respects_forbidden_region():
    # the L1 corridor is accessible at C=3.15; the L4 equilateral point (2*Omega ~ 2.99) is forbidden
    assert accessible_speed(0.9, 0.05, MU, 3.15) is not None
    assert accessible_speed(0.5 - MU, np.sqrt(3) / 2, MU, 3.15) is None   # L4 region: 2*Omega < C


def test_fli_is_finite_and_nonnegative():
    sp = accessible_speed(0.9, 0.05, MU, 3.15)
    vx, vy = _prograde_velocity(0.9, 0.05, sp)
    f = fli([0.9, 0.05, 0.0, vx, vy, 0.0], MU, t_max=2.0)
    assert np.isfinite(f) and f >= 0.0


@pytest.mark.slow
def test_naive_chaos_ridge_hypothesis_is_refuted():
    """Manifold-tube states are NOT significantly more chaotic than generic states."""
    from ariadne.validate.stage25 import check
    ok, info = check()
    assert ok                                                  # naive ridge hypothesis refuted
    # and the manifold population is not a high-FLI outlier
    assert info["man_fli"].mean() <= info["rnd_fli"].mean() * 1.05
