"""Heteroclinic-connection tests (Gate G6)."""
import numpy as np
import pytest

from ariadne.data.constants import EARTH_MOON
from ariadne.orbits.families import lyapunov_orbit_at_jacobi
from ariadne.connections.heteroclinic import (
    _segment_intersection, loop_intersections, find_heteroclinic,
)

MU = EARTH_MOON.mu


def test_segment_intersection_geometry():
    p = _segment_intersection((0, 0), (2, 2), (0, 2), (2, 0))
    assert p is not None and abs(p[0] - 1) < 1e-12 and abs(p[1] - 1) < 1e-12
    assert _segment_intersection((0, 0), (1, 0), (0, 1), (1, 1)) is None


def test_loop_intersections_crossing_squares():
    a = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], float)
    b = np.array([[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]], float)
    assert len(loop_intersections(a, b)) >= 2


@pytest.mark.slow
def test_lyapunov_orbit_at_target_jacobi():
    for point, c in (("L1", 3.17), ("L2", 3.15)):
        orb = lyapunov_orbit_at_jacobi(MU, point, c)
        assert abs(orb.jacobi - c) < 1e-7


@pytest.mark.slow
def test_heteroclinic_l1_l2_connection_exists():
    conn = find_heteroclinic(MU, 3.15, "L1", "L2", n_seeds=140, displacement=1e-4)
    assert conn is not None
    assert len(conn["intersections"]) >= 1
    assert conn["jacobi"] == 3.15
