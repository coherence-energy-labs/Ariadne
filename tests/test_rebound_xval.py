"""Stage 37 test: independent cross-validation against REBOUND (skipped if rebound absent)."""
import pytest

from ariadne.validate import stage37


@pytest.mark.skipif(not stage37.HAVE_REBOUND, reason="rebound not installed")
def test_ariadne_agrees_with_rebound():
    err, dEr = stage37._agreement(200.0)
    assert err < 2e-3                 # two independent WH integrators agree over 200 yr
    assert dEr < 1e-4                 # REBOUND energy bounded (symplectic)
