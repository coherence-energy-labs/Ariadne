"""Opt-in success-before-failure planning for explicitly supplied CTMC rates.

Port of the supplied ONE Frontier v4 contract compiler. Known Schur/resolvent
mathematics; not a new astrodynamics force model or an arbitrary HJB replacement.
Geometric k-NN weights are NOT calibrated transition rates by themselves.

Only constant reductions in failure rates at retained states are supported.
The prepared object is a frozen model snapshot, not a live telemetry feed.
A current fingerprint is required at evaluation to expose accidental stale use.

Numbers enter in two representations. Every solve runs in binary64. Every
certificate is checked in exact rational arithmetic against the EXACT RATIONAL
MODEL of the supplied rates: ``Fraction``/``int``/``Decimal`` entries exactly as
given, binary64 entries at their exact binary value, and every derived quantity
(generator row sums, goal-flux sums, reduced diagonals) recomputed exactly rather
than inherited from floating-point rounding.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import operator
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from types import MappingProxyType

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

#: Solver noise tolerated on a probability before it is refused rather than clamped.
PROBABILITY_TOLERANCE = 1e-12
#: Solver noise tolerated on a deadline probability above its eventual probability.
DEADLINE_TOLERANCE = 1e-10

_EMPTY = MappingProxyType({})


def _index(value, n, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not 0 <= result < n:
        raise ValueError(f"{label} out of range")
    return result


def _real_scalar(value, label):
    """A real number as binary64, validated by TYPE before any comparison.

    Strings, booleans, complex numbers and arbitrary objects are refused with
    ValueError; a comparison on raw input would otherwise leak a TypeError.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.floating, np.integer, Decimal)
    ):
        raise ValueError(f"{label} must be a real number")
    try:
        return float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc


def _exact_entry(value, label):
    """Exact rational value of one supplied entry; None when it IS its binary64 value."""
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"real {label} required")
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value) if value.is_finite() else None
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        return Fraction(*value.as_integer_ratio()) if np.isfinite(value) else None
    if isinstance(value, (float, np.floating)):
        return None
    raise ValueError(f"{label} must be numeric")


def _real_array(raw, label):
    """Binary64 working copy plus the exact value of every entry that copy rounded.

    Returns ``(values, exact)`` where ``exact`` maps a flat C-order index to the
    Fraction of each entry whose exact value differs from its binary64 copy. Plain
    float input produces an empty map at no per-entry cost.
    """
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    if np.iscomplexobj(raw):
        raise ValueError(f"real {label} required")
    arr = np.asarray(raw)
    kind = arr.dtype.kind
    if kind not in "biufO":
        raise ValueError(f"{label} must be numeric")
    scan = (
        kind == "O"
        or (kind in "iu" and arr.size and (arr.max() > 2**53 or arr.min() < -(2**53)))
        or (kind == "f" and arr.dtype.itemsize > 8)
    )
    exact_values = [_exact_entry(v, label) for v in arr.flat] if scan else ()
    try:
        values = np.array(arr, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    exact = {}
    for k, q in enumerate(exact_values):
        w = float(values.flat[k])
        if q is not None and (not math.isfinite(w) or q != Fraction.from_float(w)):
            exact[k] = q
    return values, exact


def _check_rates(W, failure, success, exact):
    if failure.ndim != 1 or not 0 < len(failure) <= 2000:
        raise ValueError("failure must contain 1..2000 rates")
    n = len(failure)
    if W.shape != (n, n) or success.shape != (n,):
        raise ValueError("rate shape mismatch")
    if not all(np.isfinite(v).all() for v in (W, failure, success)):
        raise ValueError("nonfinite rates")
    w_exact, f_exact, s_exact = exact
    off_negative = any(q < 0 for q in w_exact.values())
    diagonal = any(q != 0 and k % (n + 1) == 0 for k, q in w_exact.items())
    if np.any(W < 0) or np.any(np.diag(W) != 0) or off_negative or diagonal:
        raise ValueError("off-diagonal transition rates >=0; diagonal must be zero")
    if (
        np.any(failure <= 0)
        or np.any(success < 0)
        or any(q <= 0 for q in f_exact.values())
        or any(q < 0 for q in s_exact.values())
    ):
        raise ValueError("require positive failure and nonnegative success rates")


def _rates(W, failure, success):
    if hasattr(W, "toarray") and W.shape[0] > 2000:
        raise ValueError("DENSE_PREPARATION_LIMIT: at most 2000 states")
    (W, w_exact), (failure, f_exact), (success, s_exact) = (
        _real_array(W, "rates"),
        _real_array(failure, "rates"),
        _real_array(success, "rates"),
    )
    exact = (w_exact, f_exact, s_exact)
    _check_rates(W, failure, success, exact)
    return W, failure, success, exact


def _fingerprint_arrays(W, failure, success, exact=(_EMPTY, _EMPTY, _EMPTY)):
    # Canonical binary64 little-endian encoding avoids constructing huge nested
    # Python/JSON objects on every preparation. Shape and schema are bound too.
    digest = hashlib.sha256(b"ARIADNE_CTMCRATES_V1_BINARY64_LE\0")
    digest.update(len(failure).to_bytes(8, "big"))
    for value in (W, failure, success):
        canonical = np.array(value, dtype="<f8", order="C", copy=True)
        canonical[canonical == 0] = 0.0  # normalize signed zero
        digest.update(canonical.tobytes(order="C"))
    if any(exact):
        # Only entries binary64 could NOT hold are bound here, so a rational input
        # equal to its float copy keeps the V1 fingerprint of that float.
        digest.update(b"ARIADNE_CTMCRATES_V2_EXACT_RATIONAL_ENTRIES\0")
        for tag, entries in zip((b"W", b"F", b"S"), exact, strict=True):
            digest.update(tag + len(entries).to_bytes(8, "big"))
            for k in sorted(entries):
                q = entries[k]
                digest.update(k.to_bytes(8, "big") + f"{q.numerator}/{q.denominator};".encode())
    return digest.hexdigest()


def model_fingerprint(W, failure, success):
    return _fingerprint_arrays(*_rates(W, failure, success))


def _exact_rows_of(A_values, A_exact):
    """Sparse exact rows ``[(column, Fraction), ...]`` of a matrix and its exact map."""
    n = A_values.shape[1]
    extra = {}
    for k in A_exact:
        extra.setdefault(k // n, []).append(k % n)
    q = Fraction.from_float
    for i in range(A_values.shape[0]):
        columns = sorted(set(np.flatnonzero(A_values[i]).tolist()) | set(extra.get(i, ())))
        yield [(j, A_exact.get(i * n + j, q(float(A_values[i, j])))) for j in columns]


def _rational_certificate(rows, b, x, start):
    """||x - x*||_inf <= ||b - A x||_inf / eta for strictly row-dominant exact A.

    ``rows`` are exact sparse rows, ``b`` exact, ``x`` the floating candidate taken
    at its exact binary value. No producer factorization is used.
    """
    q = Fraction.from_float
    xx = [q(float(v)) for v in x]
    residual, eta = Fraction(0), None
    for i, row in enumerate(rows):
        diagonal, off, action = Fraction(0), Fraction(0), Fraction(0)
        for j, a in row:
            action += a * xx[j]
            if j == i:
                diagonal += a
            else:
                off += abs(a)
        margin = diagonal - off
        eta = margin if eta is None else min(eta, margin)
        residual = max(residual, abs(b[i] - action))
    if eta is None or eta <= 0:
        raise ValueError("UNSUPPORTED_CERTIFICATE: no strict row dominance")
    bound = residual / eta
    return {
        "lower": math.nextafter(float(xx[start] - bound), -math.inf),
        "upper": math.nextafter(float(xx[start] + bound), math.inf),
        "error_bound": math.nextafter(float(bound), math.inf),
        "exact_error_bound": str(bound),
        "exact_residual": str(residual),
        "scope": "eventual probability only; not time moments or physical rates",
    }


def residual_certificate(A, b, x, start):
    """Exact rational residual bound for the SUPPLIED matrix and right-hand side.

    Each entry is taken at its exact value: ``Fraction``/``int``/``Decimal``
    entries exactly, binary64 entries at their exact binary value. It certifies
    that matrix, not the physics it was built from. Strict row diagonal dominance
    gives ||error||_inf <= ||residual||_inf/eta. No producer factorization is used.
    Intended for final selections, not every candidate: rational arithmetic over
    the nonzeros can be expensive.
    """
    (A, a_exact), (b, b_exact), (x, x_exact) = (
        _real_array(A, "certificate input"),
        _real_array(b, "certificate input"),
        _real_array(x, "certificate input"),
    )
    n = len(b)
    start = _index(start, n, "start")
    if b.ndim != 1 or A.shape != (n, n) or x.shape != (n,):
        raise ValueError("certificate shape mismatch")
    if not all(np.isfinite(v).all() for v in (A, b, x)):
        raise ValueError("nonfinite certificate input")
    if x_exact:
        raise ValueError("certificate candidate must be binary64")
    bb = [b_exact.get(i, Fraction.from_float(float(b[i]))) for i in range(n)]
    result = _rational_certificate(_exact_rows_of(A, a_exact), bb, x, start)
    return {
        "family": "EXACT_RATIONAL_RESIDUAL_ON_SUPPLIED_MATRIX",
        **result,
        "matrix": "exact value of each supplied entry; binary64 entries at their binary value",
    }


class SuccessContract:
    """Compiled CTMC snapshot; B contains start and all controllable states.

    Dense preparation, boundary-sized repeated queries. Deadline queries use the
    FULL generator because the static Schur complement does not preserve clocks.
    """

    def __init__(self, W, failure, success, retained, *, start=0):
        W, failure, success, exact = _rates(W, failure, success)
        n = len(failure)
        start = _index(start, n, "start")
        retained = [_index(v, n, "retained state") for v in retained]
        if not retained or len(set(retained)) != len(retained) or start not in retained:
            raise ValueError("retained states must be unique and include start")
        self.fingerprint = _fingerprint_arrays(W, failure, success, exact)
        self.start, self.start_pos = start, retained.index(start)
        self.retained = tuple(retained)
        self._exact = tuple(MappingProxyType(dict(entries)) for entries in exact)
        self._B = np.array(retained, dtype=int)
        self._I = np.array([i for i in range(n) if i not in retained], dtype=int)
        self._failure, self._b = failure, success
        M = np.diag(W.sum(axis=1) + failure + success) - W
        if not np.isfinite(M).all() or np.any(np.diag(M) - W.sum(axis=1) <= 0):
            raise ValueError("RATE_SCALE_UNRESOLVED: rescale time units")
        self._M = M
        B, H = self._B, self._I  # retained boundary, hidden interior
        bb, bi, ib = M[np.ix_(B, B)], M[np.ix_(B, H)], M[np.ix_(H, B)]
        factor = lu_factor(M[np.ix_(H, H)]) if len(H) else None
        solve = lambda v: lu_solve(factor, v) if len(H) else np.zeros_like(v)
        X, y = solve(ib), solve(success[H])
        X2, y2 = solve(X), solve(y)
        X3, y3 = solve(X2), solve(y2)
        self._X, self._y = X, y
        self._S, self._r = bb - bi @ X, success[B] - bi @ y
        self._S1, self._r1 = np.eye(len(B)) + bi @ X2, bi @ y2
        self._S2, self._r2 = -2 * bi @ X3, -2 * bi @ y3
        # Caller-owned arrays cannot mutate a prepared model behind its hash.
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def _validate(self, reductions, current_fingerprint):
        if current_fingerprint != self.fingerprint:
            raise ValueError("STALE_MODEL: recompile the current rates")
        d, d_exact = _real_array(reductions, "reductions")
        if d.shape != (len(self.retained),) or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("one finite nonnegative reduction per retained state")
        if np.any(d >= self._failure[self._B]):
            raise ValueError("reductions must retain strictly positive failure rates")
        f_exact = self._exact[1]
        for pos, state in enumerate(self.retained):
            if pos in d_exact or state in f_exact:
                dq = d_exact.get(pos, Fraction.from_float(float(d[pos])))
                fq = f_exact.get(state, Fraction.from_float(float(self._failure[state])))
                if dq < 0 or dq >= fq:
                    raise ValueError("reductions must retain strictly positive failure rates")
        return d, d_exact

    def _exact_generator_rows(self, d, d_exact):
        """Rows of the FULL intervened generator built entirely in rational arithmetic.

        Off-diagonals are the exact supplied transition rates; each diagonal is
        the exact sum of its row's rates plus failure plus success minus the exact
        reduction. Nothing is inherited from the binary64 generator's rounding.
        """
        w_exact, f_exact, s_exact = self._exact
        q = Fraction.from_float
        reduction = {
            state: d_exact.get(pos, q(float(d[pos]))) for pos, state in enumerate(self.retained)
        }
        W = -self._M  # off-diagonal entries are exactly the supplied binary64 rates
        np.fill_diagonal(W, 0.0)
        for i, row in enumerate(_exact_rows_of(W, w_exact)):
            total = sum((a for _, a in row), Fraction(0))
            diagonal = (
                total
                + f_exact.get(i, q(float(self._failure[i])))
                + s_exact.get(i, q(float(self._b[i])))
                - reduction.get(i, Fraction(0))
            )
            yield sorted([(j, -a) for j, a in row] + [(i, diagonal)])

    def evaluate(
        self, reductions, *, current_fingerprint, moments=True, deadline=None, certify=False
    ):
        d, d_exact = self._validate(reductions, current_fingerprint)
        if deadline is not None:
            t = _real_scalar(deadline, "deadline")
            if not math.isfinite(t) or t < 0:
                raise ValueError("deadline must be finite and nonnegative")
        A = self._S - np.diag(d)
        h = np.linalg.solve(A, self._r)
        tol = PROBABILITY_TOLERANCE
        if not np.isfinite(h).all() or np.any(h < -tol) or np.any(h > 1 + tol):
            raise ArithmeticError("NUMERICAL_PROBABILITY_INVALID")
        # Accepted solver noise is CLAMPED into [0, 1] and the adjustment recorded, so
        # no returned probability sits outside its own domain.
        values = np.clip(h, 0.0, 1.0)
        p = float(values[self.start_pos])
        result = {
            "status": "NUMERICAL_MODEL_RESULT",
            "model_fingerprint": self.fingerprint,
            "success_probability": p,
            "retained": list(self.retained),
            "retained_values": values.tolist(),
            "probability_clamp": {
                "tolerance": tol,
                "max_adjustment": float(np.max(np.abs(values - h))),
            },
            "reductions": d.tolist(),
            "source_states": len(self._b),
            "solve_states": len(self.retained),
        }
        if moments:
            if p <= 0:
                result.update(mean_success_time=None, variance_success_time=None)
            else:
                p_raw = float(h[self.start_pos])
                hp = np.linalg.solve(A, self._r1 - self._S1 @ h)
                hpp = np.linalg.solve(A, self._r2 - self._S2 @ h - 2 * self._S1 @ hp)
                mean = -float(hp[self.start_pos]) / p_raw
                var = float(hpp[self.start_pos]) / p_raw - mean * mean
                if not math.isfinite(mean + var) or mean < 0 or var < -1e-9 * max(1.0, mean**2):
                    raise ArithmeticError("NUMERICAL_MOMENTS_INVALID")
                result.update(mean_success_time=mean, variance_success_time=max(0.0, var))
        if certify:
            reconstructed = np.empty(len(self._b))
            reconstructed[self._B] = h
            reconstructed[self._I] = self._y - self._X @ h
            s_exact = self._exact[2]
            b = [s_exact.get(i, Fraction.from_float(float(v))) for i, v in enumerate(self._b)]
            certificate = _rational_certificate(
                self._exact_generator_rows(d, d_exact), b, reconstructed, self.start
            )
            result["certificate"] = {
                "family": "EXACT_RATIONAL_RESIDUAL_ON_EXACT_RATIONAL_GENERATOR",
                **certificate,
                "model": (
                    "generator rebuilt in exact rational arithmetic from the exact value of "
                    "every supplied rate and reduction (binary64 entries at their binary "
                    "value); row sums are exact, not floating-point"
                ),
                "exact_rational_entries": sum(map(len, self._exact)) + len(d_exact),
            }
        if deadline is not None:
            full = self._M.copy()
            full[self._B, self._B] -= d
            # Augmentation integrates the success flux without subtracting nearly
            # equal eventual-probability vectors at short deadlines.
            if t * float(np.max(np.diag(full))) > 10000:
                raise ValueError("DEADLINE_WORK_LIMIT: use rescaled/specialized transient methods")
            n = len(self._b)
            G = np.zeros((n + 1, n + 1))
            G[:n, :n], G[:n, n] = -full, self._b
            e = np.zeros(n + 1)
            e[n] = 1.0
            cdf = float(expm_multiply(csr_matrix(G * t), e)[self.start])
            if not math.isfinite(cdf) or cdf < -tol or cdf > p + DEADLINE_TOLERANCE:
                raise ArithmeticError("NUMERICAL_DEADLINE_INVALID")
            bounded = min(max(0.0, cdf), p)
            result.update(
                deadline=t,
                success_before_deadline=bounded,
                deadline_clamp={"tolerance": DEADLINE_TOLERANCE, "adjustment": abs(bounded - cdf)},
                deadline_method="FULL_AUGMENTED_EXPONENTIAL_NOT_STATIC_REDUCTION",
            )
        return result

    def select_bundle(
        self, sites, costs, budget, *, fraction, current_fingerprint, max_candidates=10000
    ):
        """Enumerate all feasible constant protection bundles in a bounded family.

        Objective: eventual success, NOT deadline success. Numerical selection,
        not a proof that nearly tied floating-point candidates are strictly ordered.
        Costs are arbitrary INTEGER budget units, not automatically joules.
        """
        self._validate(np.zeros(len(self.retained)), current_fingerprint)
        sites = [_index(v, len(self._b), "site") for v in sites]
        if (
            len(sites) > 20
            or len(set(sites)) != len(sites)
            or any(v not in self.retained for v in sites)
        ):
            raise ValueError("at most 20 unique retained intervention sites")

        def integer(v):
            if isinstance(v, (bool, np.bool_)):
                raise ValueError("integer cost required")
            try:
                return operator.index(v)
            except TypeError as exc:
                raise ValueError("integer cost required") from exc

        costs, budget, max_candidates = (
            list(map(integer, costs)),
            integer(budget),
            integer(max_candidates),
        )
        if (
            len(costs) != len(sites)
            or budget < 0
            or max_candidates < 1
            or any(c <= 0 for c in costs)
        ):
            raise ValueError("positive integer costs and candidate cap; nonnegative budget")
        fraction = _real_scalar(fraction, "fraction")
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("fraction must be in [0,1)")
        best, count, best_sites = None, 0, []
        for k in range(len(sites) + 1):
            for subset in combinations(range(len(sites)), k):
                if sum(costs[i] for i in subset) > budget:
                    continue
                count += 1
                if count > max_candidates:
                    raise ValueError("SEARCH_BUDGET_EXCEEDED: no optimality claim")
                d = np.zeros(len(self.retained))
                for i in subset:
                    site = sites[i]
                    d[self.retained.index(site)] = fraction * self._failure[site]
                trial = self.evaluate(d, current_fingerprint=current_fingerprint, moments=False)
                if best is None or trial["success_probability"] > best["success_probability"]:
                    best, best_sites = trial, [sites[i] for i in subset]
        final = self.evaluate(
            best["reductions"], current_fingerprint=current_fingerprint, certify=True
        )
        final.update(
            selected_sites=best_sites,
            evaluated_candidates=count,
            selection_scope="EXHAUSTIVE_FINITE_FAMILY_NUMERICAL_EVENTUAL_SUCCESS",
        )
        return final


def goal_success(rates, goals, *, loss_rate, start=0, deadline=None):
    """Hit ANY declared goal before killing, with genuine multi-goal boundaries.

    Takes directed transition RATES, not distances or arbitrary heuristic scores.
    When adapted from an uncalibrated graph, the probability is for that synthetic
    random walk only; it is not an orbital mission success forecast.
    """
    if np.iscomplexobj(rates) or np.iscomplexobj(loss_rate):
        raise ValueError("real rates required")
    if hasattr(rates, "toarray") and rates.shape[0] > 2000:
        raise ValueError("DENSE_PREPARATION_LIMIT")
    rates, rates_exact = _real_array(rates, "rates")
    if rates.ndim != 2 or rates.shape[0] != rates.shape[1] or not 0 < len(rates) <= 2000:
        raise ValueError("square rate matrix of 1..2000 states required")
    n = len(rates)
    start = _index(start, n, "start")
    goals = [_index(v, n, "goal") for v in goals]
    if not goals or len(set(goals)) != len(goals):
        raise ValueError("nonempty distinct goals required")
    try:
        loss_raw = np.broadcast_to(np.asarray(loss_rate), (n,))
    except ValueError as exc:
        raise ValueError("loss_rate must be a scalar or one rate per state") from exc
    loss, loss_exact = _real_array(loss_raw, "rates")
    _check_rates(rates, loss, np.zeros(n), (rates_exact, loss_exact, {}))
    if deadline is not None:
        t = _real_scalar(deadline, "deadline")
        if not math.isfinite(t) or t < 0:
            raise ValueError("nonnegative finite deadline required")
    if start in goals:
        result = {
            "status": "AT_GOAL",
            "success_probability": 1.0,
            "mean_success_time": 0.0,
            "variance_success_time": 0.0,
        }
        if deadline is not None:
            result["success_before_deadline"] = 1.0
        return result
    goal_set = set(goals)
    transient = [i for i in range(n) if i not in goal_set]
    i = transient.index(start)
    position = {state: k for k, state in enumerate(transient)}
    W = rates[np.ix_(transient, transient)]
    success = rates[np.ix_(transient, goals)].sum(axis=1)
    loss_t = loss[transient]
    # Carry exact values through the goal reduction: an exact rate that binary64
    # rounded stays exact, and a goal-flux sum that binary64 rounded is recomputed
    # exactly (fsum of the row minus its float sum is zero iff the float sum is exact).
    exact_rows = {k // n for k in rates_exact}
    lossy = [
        k
        for k, r in enumerate(transient)
        if r in exact_rows or math.fsum([*rates[r, goals].tolist(), -float(success[k])]) != 0.0
    ]
    if lossy:
        q = Fraction.from_float
        success = success.astype(object)
        for k in lossy:
            r = transient[k]
            success[k] = sum(
                (rates_exact.get(r * n + g, q(float(rates[r, g]))) for g in goals), Fraction(0)
            )
    w_entries = {
        position[k // n] * len(transient) + position[k % n]: q
        for k, q in rates_exact.items()
        if k // n in position and k % n in position
    }
    if w_entries:
        W = W.astype(object)
        for k, q in w_entries.items():
            W.flat[k] = q
    if loss_exact:
        loss_t = loss_t.astype(object)
        for k, q in loss_exact.items():
            if k in position:
                loss_t[position[k]] = q
    compiler = SuccessContract(W, loss_t, success, [i], start=i)
    result = compiler.evaluate(
        [0.0], current_fingerprint=compiler.fingerprint, deadline=deadline, certify=True
    )
    result["transient_node_ids"] = transient
    result["goals"] = goals
    return result
