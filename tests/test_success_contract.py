"""Declared-domain tests, independent dense/exponential/fraction oracles."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ariadne.optimize.success_contract import (
    SuccessContract,
    goal_success,
    model_fingerprint,
    residual_certificate,
)


@pytest.mark.parametrize("seed", range(50))
def test_reduction_moments_deadlines_against_full_oracles(seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(3, 15))
    W = r.uniform(0, 1, (n, n)) * (r.random((n, n)) < 0.2)
    np.fill_diagonal(W, 0)
    failure, b = r.uniform(0.01, 0.3, n), r.uniform(0, 1, n)
    B = sorted({0, n // 2, n - 1})
    model = SuccessContract(W, failure, b, B)
    for _ in range(8):
        delta = failure[B] * r.uniform(0, 0.95, len(B))
        got = model.evaluate(
            delta, current_fingerprint=model.fingerprint, deadline=0.3, certify=True
        )
        altered = failure.copy()
        altered[B] -= delta
        A = np.diag(W.sum(1) + altered + b) - W
        h = np.linalg.solve(A, b)
        m1 = np.linalg.solve(A, h)
        m2 = 2 * np.linalg.solve(A, m1)
        assert got["success_probability"] == pytest.approx(h[0], abs=3e-13)
        assert got["mean_success_time"] == pytest.approx(m1[0] / h[0], rel=2e-11)
        assert got["variance_success_time"] == pytest.approx(
            m2[0] / h[0] - (m1[0] / h[0]) ** 2, rel=3e-10
        )
        assert got["success_before_deadline"] == pytest.approx(
            (h - expm(-0.3 * A) @ h)[0], abs=3e-13
        )
        c = got["certificate"]
        assert c["error_bound"] < 1e-10
        # Float full solve is a comparison, not an exact certificate oracle.
        assert abs(got["success_probability"] - h[0]) < c["error_bound"] + 1e-13


def test_directed_edges_do_not_become_invented_reverse_routes():
    W = [[0, 2, 0], [0, 0, 1], [0, 0, 0]]
    assert goal_success(W, [0], loss_rate=0.2, start=2)["success_probability"] == 0
    assert goal_success(W, [2], loss_rate=0.2, start=0)["success_probability"] > 0.7


def test_goal_set_is_a_boundary_not_mean_source_normalization():
    W = [[0, 1, 5], [0, 0, 0], [0, 0, 0]]
    for g in (1, 2):
        assert goal_success(W, [1, 2], loss_rate=0.4, start=g)["success_probability"] == 1
    assert goal_success(csr_matrix(W), [1, 2], loss_rate=0.4)[
        "success_probability"
    ] == pytest.approx(6 / 6.4)


def test_hidden_waiting_time_is_not_erased():
    model = SuccessContract([[0, 0.01], [0, 0]], [0.0001, 0.0001], [0, 10], [0])
    exact = model.evaluate([0], current_fingerprint=model.fingerprint, deadline=1)
    fake_clock = 1 - math.exp(-10)
    assert exact["success_before_deadline"] < 0.01
    assert fake_clock > 0.99
    assert exact["mean_success_time"] > 99
    assert exact["deadline_method"].startswith("FULL_")


def test_single_state_and_unreachable_conditional_time():
    m = SuccessContract([[0]], [2.0], [3.0], [0])
    v = m.evaluate([0], current_fingerprint=m.fingerprint, deadline=0.2)
    assert v["success_probability"] == pytest.approx(0.6)
    assert v["mean_success_time"] == pytest.approx(0.2)
    assert v["variance_success_time"] == pytest.approx(0.04)
    assert v["success_before_deadline"] == pytest.approx(0.6 * (1 - math.exp(-1)))
    m = SuccessContract([[0]], [1.0], [0.0], [0])
    v = m.evaluate([0], current_fingerprint=m.fingerprint, deadline=1)
    assert v["success_probability"] == v["success_before_deadline"] == 0
    assert v["mean_success_time"] is None


def test_exact_fraction_certificate_and_mutated_answer():
    A = np.array([[3.0, -1.0], [-2.0, 5.0]])
    b = np.array([1.0, 2.0])
    x = np.linalg.solve(A, b)
    c = residual_certificate(A, b, x, 0)
    truth = Fraction(7, 13)
    assert Fraction.from_float(c["lower"]) <= truth <= Fraction.from_float(c["upper"])
    x[0] += 0.1
    bad = residual_certificate(A, b, x, 0)
    assert bad["error_bound"] > 0.1


def test_fingerprint_and_snapshot():
    W = np.array([[0.0, 1.0], [2.0, 0.0]])
    m = SuccessContract(W, [0.1, 0.1], [1.0, 1.0], [0])
    baseline = m.evaluate([0], current_fingerprint=m.fingerprint)
    W[0, 1] = 20
    now = model_fingerprint(W, [0.1, 0.1], [1.0, 1.0])
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.evaluate([0], current_fingerprint=now)
    assert m.evaluate([0], current_fingerprint=m.fingerprint) == baseline


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), -1, 0])
def test_invalid_failure_rates(replacement):
    with pytest.raises(ValueError):
        SuccessContract([[0]], [replacement], [1], [0])


@pytest.mark.parametrize("retained", [[0, 0], [0.5], [True], [], [1]])
def test_invalid_retained(retained):
    with pytest.raises(ValueError):
        SuccessContract([[0]], [1], [1], retained)


@pytest.mark.parametrize("delta", [[-0.1], [1.0], [float("nan")], []])
def test_invalid_controls(delta):
    m = SuccessContract([[0]], [1], [1], [0])
    with pytest.raises(ValueError):
        m.evaluate(delta, current_fingerprint=m.fingerprint)


def test_deadline_refusal_and_numerical_scale():
    m = SuccessContract([[0]], [1], [1], [0])
    for t in (-1, float("nan"), 1e300):
        with pytest.raises(ValueError):
            m.evaluate([0], current_fingerprint=m.fingerprint, deadline=t)
    with pytest.raises(ValueError, match="RATE_SCALE"):
        SuccessContract([[0, 1e200], [1e200, 0]], [1, 1], [1, 1], [0])
    with pytest.raises(ValueError, match="real rates"):
        SuccessContract([[0j]], [1], [1], [0])


def test_bundle_finite_enumeration_and_budget_refusal():
    # Five nodes with independent and complementary routes; enumerate via full oracle.
    W = np.array(
        [[0, 1, 1, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], float
    )
    f, b = np.array([0.1, 2, 2, 3, 3]), np.array([0, 0, 0, 1, 1])
    m = SuccessContract(W, f, b, [0, 1, 2, 3, 4])
    result = m.select_bundle(
        [1, 2, 3, 4], [1, 1, 1, 1], 2, fraction=0.9, current_fingerprint=m.fingerprint
    )
    from itertools import combinations

    scores = []
    for k in range(3):
        for selected in combinations([1, 2, 3, 4], k):
            ff = f.copy()
            ff[list(selected)] *= 0.1
            scores.append(np.linalg.solve(np.diag(W.sum(1) + ff + b) - W, b)[0])
    assert result["success_probability"] == pytest.approx(max(scores))
    assert result["evaluated_candidates"] == 11
    with pytest.raises(ValueError, match="SEARCH_BUDGET_EXCEEDED"):
        m.select_bundle(
            [1, 2], [1, 1], 2, fraction=0.9, current_fingerprint=m.fingerprint, max_candidates=1
        )
    with pytest.raises(ValueError, match="integer cost"):
        m.select_bundle([1], [1.2], 2, fraction=0.9, current_fingerprint=m.fingerprint)


def test_transient_independent_ode():
    W, f, b = np.array([[0.0, 3.0], [1.0, 0.0]]), [0.2, 0.1], [0, 2]
    m = SuccessContract(W, f, b, [0])
    v = m.evaluate([0], current_fingerprint=m.fingerprint, deadline=2)
    A = np.diag(W.sum(1) + np.array(f) + b) - W
    ivp = solve_ivp(lambda t, x: -A @ x + b, [0, 2], [0.0, 0.0], rtol=1e-11, atol=1e-13)
    assert ivp.success
    assert v["success_before_deadline"] == pytest.approx(ivp.y[0, -1], abs=2e-12)


def test_public_package_facade_uses_compiler_and_existing_constants():
    import ariadne

    assert ariadne.system("EARTH_MOON").L_star == 384400.0
    assert ariadne.GM_JUPITER > ariadne.GM_EARTH
    m = ariadne.compile_success_contract([[0]], [2.0], [3.0], [0])
    v = m.evaluate([0], current_fingerprint=m.fingerprint, certify=True)
    assert v["certificate"]["lower"] <= 0.6 <= v["certificate"]["upper"]
    assert ariadne.goal_success([[0, 1], [0, 0]], [1], loss_rate=1)[
        "success_probability"
    ] == pytest.approx(0.5)


def test_complex_values_never_silently_discard_imaginary_parts():
    with pytest.raises(ValueError, match="real"):
        goal_success(np.array([[0, 1j], [0, 0]]), [1], loss_rate=1)
    model = SuccessContract([[0]], [1], [1], [0])
    with pytest.raises(ValueError, match="real"):
        model.evaluate([0.1j], current_fingerprint=model.fingerprint)
    with pytest.raises(ValueError, match="real"):
        residual_certificate([[1j]], [1], [1], 0)


def test_fingerprint_ignores_array_layout_endianness_and_signed_zero():
    W = np.array([[0.0, 1.0], [2.0, 0.0]])
    baseline = model_fingerprint(W, [0.1, 0.2], [1.0, 2.0])
    changed = np.array(W, dtype=">f8", order="F")
    changed[0, 0] = -0.0
    assert model_fingerprint(changed, [0.1, 0.2], [1.0, 2.0]) == baseline
    changed[0, 1] = np.nextafter(1.0, 2.0)
    assert model_fingerprint(changed, [0.1, 0.2], [1.0, 2.0]) != baseline
