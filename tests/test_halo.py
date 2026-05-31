"""3D halo orbit tests (Gate G_halo)."""
import numpy as np
import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import propagate
from ariadne.orbits.halo import halo_family

MU = EARTH_MOON.mu


@pytest.mark.slow
def test_halo_family_periodic_and_3d():
    halos = halo_family(MU, "L1", n=8, dz=2e-3, fam_n=40)
    assert len(halos) >= 6
    for h in halos[::2]:
        # truly periodic 3D orbit
        res = np.max(np.abs(propagate(h.s0, (0.0, h.period), MU).y[:, -1] - h.s0))
        assert res < 1e-9
        assert h.z_amplitude > 0.0          # genuinely out of plane


@pytest.mark.slow
def test_halo_branches_at_lyapunov_bifurcation():
    halos = halo_family(MU, "L1", n=8, dz=2e-3, fam_n=40)
    # halos branch from the Lyapunov vertical bifurcation found in Stage 2 (C ~ 3.186)
    assert 3.17 < halos[0].jacobi < 3.19
    z = [h.z_amplitude for h in halos]
    assert all(b > a for a, b in zip(z, z[1:]))     # amplitude grows along the family
