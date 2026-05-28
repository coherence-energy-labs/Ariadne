"""Invariant-manifold tests (Gate G5)."""
import numpy as np

from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import jacobi_constant
from ariadne.orbits.differential_correction import correct_lyapunov, monodromy
from ariadne.orbits.linear import linear_lyapunov_guess
from ariadne.manifolds.manifold import manifold_eigenvectors, manifold_seeds
from ariadne.connections.poincare import propagate_until_section

MU = EARTH_MOON.mu


def _l1_orbit(amp=0.03):
    s0, Tg = linear_lyapunov_guess(MU, "L1", amp)
    orb = correct_lyapunov(MU, s0, Tg)
    orb.point = "L1"
    return orb


def test_eigenvectors_reciprocal_and_unstable():
    orb = _l1_orbit()
    M = monodromy(MU, orb)
    v_u, v_s, lam = manifold_eigenvectors(M)
    assert lam > 1.0
    vals = np.abs(np.linalg.eigvals(M))
    # reciprocal pair: largest * smallest ~ 1
    assert abs(vals.max() * vals.min() - 1.0) < 1e-3
    assert abs(np.linalg.norm(v_u[:3]) - 1.0) < 1e-12


def test_seeds_sit_at_orbit_energy():
    orb = _l1_orbit()
    seeds, lam = manifold_seeds(MU, orb, n_seeds=20, displacement=1e-4,
                                stable=False, branch=+1)
    dc = max(abs(jacobi_constant(s, MU) - orb.jacobi) for s in seeds)
    assert dc < 1e-2          # O(displacement)


def test_tube_reaches_moon_neck_and_conserves_jacobi():
    orb = _l1_orbit(amp=0.05)
    x_sec = 1.0 - MU
    # the Moon-ward branch reaches the neck; the eigenvector sign that points
    # Moon-ward varies per orbit, so accept whichever branch reaches it.
    best_reached, best_drifts = 0, []
    for branch in (+1, -1):
        seeds, lam = manifold_seeds(MU, orb, n_seeds=16, displacement=1e-4,
                                    stable=False, branch=branch)
        drifts, reached = [], 0
        for s in seeds:
            _, Y, hit = propagate_until_section(MU, s, x_sec, stable=False, t_max=8.0)
            if hit:
                reached += 1
                c0 = jacobi_constant(Y[:, 0], MU)
                drifts.append(max(abs(jacobi_constant(Y[:, i], MU) - c0)
                                  for i in range(Y.shape[1])))
        if reached > best_reached:
            best_reached, best_drifts = reached, drifts
    assert best_reached >= 4
    assert np.median(best_drifts) < 1e-9      # machinery sound on non-grazing arcs
