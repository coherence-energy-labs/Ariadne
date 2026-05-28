"""Lagrange-point tests (Gate G2)."""
import numpy as np

from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import omega_gradient
from ariadne.orbits.lagrange import lagrange_points

MU = EARTH_MOON.mu

PUBLISHED_EM = {"L1": 0.8369151324, "L2": 1.1556821603, "L3": -1.0050626453}


def test_gradient_zero_at_all_points():
    pts = lagrange_points(MU)
    for k, s in pts.items():
        assert np.linalg.norm(omega_gradient(s, MU)) < 1e-10, k


def test_collinear_positions_match_published():
    pts = lagrange_points(MU)
    for k, x_pub in PUBLISHED_EM.items():
        assert abs(pts[k][0] - x_pub) < 1e-3, (k, pts[k][0], x_pub)


def test_triangular_points_exact():
    pts = lagrange_points(MU)
    assert abs(pts["L4"][0] - (0.5 - MU)) < 1e-14
    assert abs(pts["L4"][1] - np.sqrt(3) / 2) < 1e-14
    assert abs(pts["L5"][1] + np.sqrt(3) / 2) < 1e-14
