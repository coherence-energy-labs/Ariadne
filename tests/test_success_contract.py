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


# --- The certificate is about the EXACT rational model, not binary64 rounding. ---


def _exact_two_state_truth(W, f, s):
    """Eventual success from state 0 of the exact rational 2-state model (Cramer)."""
    q = lambda v: v if isinstance(v, Fraction) else Fraction.from_float(float(v))
    w01, w10 = q(W[0][1]), q(W[1][0])
    a00, a11 = w01 + q(f[0]) + q(s[0]), w10 + q(f[1]) + q(s[1])
    return (q(s[0]) * a11 + w01 * q(s[1])) / (a00 * a11 - w01 * w10)


def test_certificate_contains_the_exact_model_when_float_row_sums_round():
    # Witness: binary64 rounds the generator diagonal 7 + 0.001 + 1e-06, and the old
    # certificate on that rounded matrix EXCLUDED the exact model's probability.
    W, f, s = [[0.0, 7.0], [11.0, 0.0]], [1e-3, 1e-6], [1e-6, 1e-3]
    float_diagonal = np.diag(np.array(W).sum(1) + np.array(f) + np.array(s)) - np.array(W)
    exact_diagonal = Fraction(7) + Fraction.from_float(1e-3) + Fraction.from_float(1e-6)
    assert Fraction.from_float(float(float_diagonal[0, 0])) != exact_diagonal  # premise holds
    m = SuccessContract(W, f, s, [0])
    c = m.evaluate([0], current_fingerprint=m.fingerprint, moments=False, certify=True)[
        "certificate"
    ]
    truth = _exact_two_state_truth(W, f, s)
    assert Fraction.from_float(c["lower"]) <= truth <= Fraction.from_float(c["upper"])
    assert c["family"] == "EXACT_RATIONAL_RESIDUAL_ON_EXACT_RATIONAL_GENERATOR"


def test_certificate_certifies_caller_supplied_rationals_exactly():
    f = [Fraction(1, 1000), Fraction(1, 10**6)]
    s = [Fraction(1, 10**6), Fraction(1, 1000)]
    W = np.array([[0, 7], [11, 0]], dtype=object)
    m = SuccessContract(W, f, s, [0])
    got = m.evaluate([0], current_fingerprint=m.fingerprint, moments=False, certify=True)
    c = got["certificate"]
    truth = _exact_two_state_truth(W, f, s)  # decimal model, not the binary64 one
    assert truth != _exact_two_state_truth(W, [float(v) for v in f], [float(v) for v in s])
    assert Fraction.from_float(c["lower"]) <= truth <= Fraction.from_float(c["upper"])
    assert c["exact_rational_entries"] == 4
    # 1/1000 is not a binary64 number, so the rational model has its own fingerprint;
    # integers that binary64 holds exactly keep the float fingerprint.
    assert m.fingerprint != model_fingerprint(W, [float(v) for v in f], [float(v) for v in s])
    assert model_fingerprint([[0, 7], [11, 0]], [1, 2], [3, 4]) == model_fingerprint(
        [[0.0, 7.0], [11.0, 0.0]], [1.0, 2.0], [3.0, 4.0]
    )


def test_rational_reductions_are_certified_exactly_and_bounded_exactly():
    m = SuccessContract([[0]], [Fraction(1, 3)], [Fraction(2, 3)], [0])
    got = m.evaluate([Fraction(1, 7)], current_fingerprint=m.fingerprint, certify=True)
    truth = Fraction(2, 3) / (Fraction(1, 3) - Fraction(1, 7) + Fraction(2, 3))
    c = got["certificate"]
    assert Fraction.from_float(c["lower"]) <= truth <= Fraction.from_float(c["upper"])
    with pytest.raises(ValueError, match="strictly positive"):
        # float(1/3) > 1/3, so this reduction passes a float check yet ends the failure rate.
        m.evaluate([Fraction(1, 3)], current_fingerprint=m.fingerprint)
    with pytest.raises(ValueError):
        SuccessContract([[0]], [Fraction(-1, 10**400)], [1], [0])  # negative, rounds to -0.0


def test_goal_flux_sum_is_exact_in_the_goal_certificate():
    # Flux into goals {1, 2} is 0.1 + 0.2; binary64's sum is not the exact sum.
    rates = [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert Fraction.from_float(0.1 + 0.2) != Fraction.from_float(0.1) + Fraction.from_float(0.2)
    got = goal_success(rates, [1, 2], loss_rate=Fraction(1, 10**15))
    flux = Fraction.from_float(0.1) + Fraction.from_float(0.2)
    truth = flux / (flux + Fraction(1, 10**15))
    c = got["certificate"]
    assert Fraction.from_float(c["lower"]) <= truth <= Fraction.from_float(c["upper"])


def test_public_residual_certificate_accepts_exact_rational_matrices():
    A = np.array([[Fraction(10, 3), Fraction(-1, 3)], [Fraction(-1, 7), Fraction(9, 7)]], object)
    b = [Fraction(1, 3), Fraction(2, 7)]
    x = np.linalg.solve(A.astype(float), np.array(b, dtype=float))
    c = residual_certificate(A, b, x, 0)
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    truth = (b[0] * A[1][1] - A[0][1] * b[1]) / det
    assert Fraction.from_float(c["lower"]) <= truth <= Fraction.from_float(c["upper"])


# --- Type validation happens BEFORE any comparison on raw input. ---


@pytest.mark.parametrize("deadline", ["3", b"3", True, 1j, None.__class__, [1.0]])
def test_goal_success_rejects_non_real_deadline_with_value_error(deadline):
    with pytest.raises(ValueError, match="deadline"):
        goal_success([[0, 1], [0, 0]], [1], loss_rate=1, deadline=deadline)


@pytest.mark.parametrize("deadline", ["3", True, 1j])
def test_evaluate_rejects_non_real_deadline_with_value_error(deadline):
    m = SuccessContract([[0]], [1], [1], [0])
    with pytest.raises(ValueError, match="deadline"):
        m.evaluate([0], current_fingerprint=m.fingerprint, deadline=deadline)


def test_real_deadline_types_agree():
    m = SuccessContract([[0]], [1], [1], [0])
    values = {
        m.evaluate([0], current_fingerprint=m.fingerprint, deadline=t)["success_before_deadline"]
        for t in (0.5, np.float64(0.5), Fraction(1, 2), np.array(0.5))
    }
    assert len(values) == 1
    with pytest.raises(ValueError, match="fraction"):
        m.select_bundle([0], [1], 1, fraction="0.5", current_fingerprint=m.fingerprint)


# --- Accepted solver noise is clamped into [0, 1] and recorded. ---


def test_probabilities_are_clamped_into_the_unit_interval_natural_witness():
    W = [
        [0.0, 1.9, 3.3, 0.0, 0.0],
        [3.3, 0.0, 0.1, 0.3, 0.3],
        [3.3, 3.3, 0.0, 0.7, 0.0],
        [0.1, 1.9, 0.1, 0.0, 3.3],
        [0.3, 0.3, 0.3, 1.9, 0.0],
    ]
    m = SuccessContract(W, [1e-18] * 5, [1.1, 0.7, 0.3, 0.7, 0.3], [0, 3, 4])
    got = m.evaluate([0, 0, 0], current_fingerprint=m.fingerprint, deadline=50.0)
    assert all(0.0 <= v <= 1.0 for v in got["retained_values"])
    assert 0.0 <= got["success_before_deadline"] <= got["success_probability"] <= 1.0
    assert 0.0 <= got["probability_clamp"]["max_adjustment"] <= 1e-12


@pytest.mark.parametrize("noise", [5e-13, -5e-13])
def test_clamp_records_its_adjustment_and_tolerance(monkeypatch, noise):
    m = SuccessContract([[0]], [1], [1], [0])
    target = 1.0 if noise > 0 else 0.0
    monkeypatch.setattr(np.linalg, "solve", lambda A, b: np.array([target + noise]))
    got = m.evaluate([0], current_fingerprint=m.fingerprint, moments=False)
    assert got["success_probability"] == target
    assert got["retained_values"] == [target]
    adjustment = abs((target + noise) - target)  # the representable noise, not the literal
    assert got["probability_clamp"] == {"tolerance": 1e-12, "max_adjustment": adjustment}


def test_probability_beyond_tolerance_is_refused_not_clamped(monkeypatch):
    m = SuccessContract([[0]], [1], [1], [0])
    monkeypatch.setattr(np.linalg, "solve", lambda A, b: np.array([1.0 + 2e-12]))
    with pytest.raises(ArithmeticError, match="PROBABILITY_INVALID"):
        m.evaluate([0], current_fingerprint=m.fingerprint, moments=False)


# --- Stale models are refused on EVERY entry point, for every kind of edit. ---


def _stale_base():
    W = np.array([[0.0, 1.0, 0.5], [0.2, 0.0, 1.0], [0.0, 0.3, 0.0]])
    return W, np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.4, 1.0])


def _edited(kind):
    W, f, s = _stale_base()
    if kind == "transition_rate":
        W[1, 2] = 1.5
    elif kind == "failure_rate":
        f[0] = 0.05
    elif kind == "success_rate":
        s[1] = 0.0
    elif kind == "goal_set":
        # Declaring state 1 a goal redirects its inflow into success flux.
        s = s + W[:, 1]
        W[:, 1] = 0.0
    elif kind == "state_count":
        W, f, s = np.pad(W, ((0, 1), (0, 1))), np.append(f, 0.1), np.append(s, 1.0)
    return model_fingerprint(W, f, s)


STALE_EDITS = ["transition_rate", "failure_rate", "success_rate", "goal_set", "state_count"]


@pytest.mark.parametrize("kind", STALE_EDITS)
def test_stale_fingerprint_refused_by_evaluate(kind):
    m = SuccessContract(*_stale_base(), [0, 1])
    assert _edited(kind) != m.fingerprint
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.evaluate([0, 0], current_fingerprint=_edited(kind), moments=False)


@pytest.mark.parametrize("kind", STALE_EDITS)
def test_stale_fingerprint_refused_by_deadline_and_certificate_queries(kind):
    # A deadline or certificate request must not become a side door past the tripwire.
    m = SuccessContract(*_stale_base(), [0, 1])
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.evaluate([0, 0], current_fingerprint=_edited(kind), deadline=1.0)
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.evaluate([0, 0], current_fingerprint=_edited(kind), certify=True)


@pytest.mark.parametrize("kind", STALE_EDITS)
def test_stale_fingerprint_refused_by_bundle_selection(kind):
    m = SuccessContract(*_stale_base(), [0, 1])
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.select_bundle([0, 1], [1, 1], 1, fraction=0.5, current_fingerprint=_edited(kind))


@pytest.mark.parametrize("wrong", [None, "", 0, b"", "0" * 64])
def test_non_matching_fingerprint_values_are_refused(wrong):
    m = SuccessContract(*_stale_base(), [0, 1])
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.evaluate([0, 0], current_fingerprint=wrong)


def test_stale_refusal_happens_before_any_numerical_work(monkeypatch):
    m = SuccessContract(*_stale_base(), [0, 1])

    def forbidden(*args, **kwargs):
        raise AssertionError("a stale model reached the solver")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    with pytest.raises(ValueError, match="STALE_MODEL"):
        m.evaluate([0, 0], current_fingerprint=_edited("failure_rate"))
