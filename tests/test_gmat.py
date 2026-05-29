"""Literal NASA GMAT cross-check (Gate G10). Skips if GMAT is not installed locally."""
import pytest

from ariadne.io.gmat_export import locate_gmat


def test_gmat_crosscheck_agrees():
    if locate_gmat() is None:
        pytest.skip("GMAT not installed under tools/ (literal G10 check)")
    from ariadne.validate.stage9 import gmat_crosscheck
    dpos_km, dvel_kms = gmat_crosscheck(days=3.0)
    # Ariadne's propagator vs NASA GMAT on an identical trans-lunar state
    assert dpos_km < 1.0          # < 1 km position agreement over 3 days
    assert dvel_kms < 1e-5        # < 1 cm/s velocity agreement
