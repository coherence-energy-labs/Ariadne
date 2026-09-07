"""Opt-in success-before-failure planning for explicitly supplied CTMC rates.

Port of the supplied ONE Frontier v4 contract compiler. Known Schur/resolvent
mathematics; not a new astrodynamics force model or an arbitrary HJB replacement.
Geometric k-NN weights are NOT calibrated transition rates by themselves.

Only constant reductions in failure rates at retained states are supported.
The prepared object is a frozen model snapshot, not a live telemetry feed.
A current fingerprint is required at evaluation to expose accidental stale use.
"""
from __future__ import annotations

import hashlib
import math
import operator
from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


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


def _rates(W, failure, success):
    if hasattr(W, "toarray"):
        if W.shape[0] > 2000:
            raise ValueError("DENSE_PREPARATION_LIMIT: at most 2000 states")
        W = W.toarray()
    if np.iscomplexobj(W) or np.iscomplexobj(failure) or np.iscomplexobj(success):
        raise ValueError("real rates required")
    W = np.array(W, dtype=float, copy=True)
    failure = np.array(failure, dtype=float, copy=True)
    success = np.array(success, dtype=float, copy=True)
    if failure.ndim != 1 or not 0 < len(failure) <= 2000:
        raise ValueError("failure must contain 1..2000 rates")
    n = len(failure)
    if W.shape != (n, n) or success.shape != (n,):
        raise ValueError("rate shape mismatch")
    if not all(np.isfinite(v).all() for v in (W, failure, success)):
        raise ValueError("nonfinite rates")
    if np.any(W < 0) or np.any(np.diag(W) != 0):
        raise ValueError("off-diagonal transition rates >=0; diagonal must be zero")
    if np.any(failure <= 0) or np.any(success < 0):
        raise ValueError("require positive failure and nonnegative success rates")
    return W, failure, success


def _fingerprint_arrays(W, failure, success):
    # Canonical binary64 little-endian encoding avoids constructing huge nested
    # Python/JSON objects on every preparation. Shape and schema are bound too.
    digest = hashlib.sha256(b"ARIADNE_CTMCRATES_V1_BINARY64_LE\0")
    digest.update(len(failure).to_bytes(8, "big"))
    for value in (W, failure, success):
        canonical = np.array(value, dtype="<f8", order="C", copy=True)
        canonical[canonical == 0] = 0.0  # normalize signed zero
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def model_fingerprint(W, failure, success):
    return _fingerprint_arrays(*_rates(W, failure, success))


def residual_certificate(A, b, x, start):
    """Exact rational residual on the supplied BINARY64 matrix, not its physics.

    Strict row diagonal dominance gives ||error||_inf <= ||residual||_inf/eta.
    No producer factorization is used. Intended for final selections, not every
    candidate: O(n^2) rational arithmetic can be expensive.
    """
    if any(np.iscomplexobj(v) for v in (A, b, x)):
        raise ValueError("real certificate input required")
    A, b, x = np.asarray(A), np.asarray(b), np.asarray(x)
    n = len(b)
    start = _index(start, n, "start")
    if A.shape != (n, n) or x.shape != (n,):
        raise ValueError("certificate shape mismatch")
    if not all(np.isfinite(v).all() for v in (A, b, x)):
        raise ValueError("nonfinite certificate input")
    q = lambda v: Fraction.from_float(float(v))
    xx, bb = list(map(q, x)), list(map(q, b))
    residual, eta = Fraction(0), None
    for i in range(n):
        row = list(map(q, A[i]))
        margin = row[i] - sum(abs(row[j]) for j in range(n) if i != j)
        eta = margin if eta is None else min(eta, margin)
        residual = max(residual, abs(bb[i] - sum(v * w for v, w in zip(row, xx, strict=True))))
    if eta <= 0:
        raise ValueError("UNSUPPORTED_CERTIFICATE: no strict row dominance")
    bound = residual / eta
    return {"family": "EXACT_RATIONAL_RESIDUAL_ON_BINARY64_MATRIX",
            "lower": math.nextafter(float(xx[start] - bound), -math.inf),
            "upper": math.nextafter(float(xx[start] + bound), math.inf),
            "error_bound": math.nextafter(float(bound), math.inf),
            "exact_error_bound": str(bound), "exact_residual": str(residual),
            "scope": "eventual probability only; not time moments or physical rates"}


class SuccessContract:
    """Compiled CTMC snapshot; B contains start and all controllable states.

    Dense preparation, boundary-sized repeated queries. Deadline queries use the
    FULL generator because the static Schur complement does not preserve clocks.
    """

    def __init__(self, W, failure, success, retained, *, start=0):
        W, failure, success = _rates(W, failure, success)
        n = len(failure)
        start = _index(start, n, "start")
        retained = [_index(v, n, "retained state") for v in retained]
        if not retained or len(set(retained)) != len(retained) or start not in retained:
            raise ValueError("retained states must be unique and include start")
        self.fingerprint = _fingerprint_arrays(W, failure, success)
        self.start, self.start_pos = start, retained.index(start)
        self.retained = tuple(retained)
        self._B = np.array(retained, dtype=int)
        self._I = np.array([i for i in range(n) if i not in retained], dtype=int)
        self._failure, self._b = failure, success
        M = np.diag(W.sum(axis=1) + failure + success) - W
        if not np.isfinite(M).all() or np.any(np.diag(M) - W.sum(axis=1) <= 0):
            raise ValueError("RATE_SCALE_UNRESOLVED: rescale time units")
        self._M = M
        B, I = self._B, self._I
        bb, bi, ib = M[np.ix_(B, B)], M[np.ix_(B, I)], M[np.ix_(I, B)]
        factor = lu_factor(M[np.ix_(I, I)]) if len(I) else None
        solve = lambda v: lu_solve(factor, v) if len(I) else np.zeros_like(v)
        X, y = solve(ib), solve(success[I])
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
        if np.iscomplexobj(reductions):
            raise ValueError("real reductions required")
        d = np.asarray(reductions, dtype=float)
        if d.shape != (len(self.retained),) or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("one finite nonnegative reduction per retained state")
        if np.any(d >= self._failure[self._B]):
            raise ValueError("reductions must retain strictly positive failure rates")
        return d

    def evaluate(self, reductions, *, current_fingerprint, moments=True,
                 deadline=None, certify=False):
        d = self._validate(reductions, current_fingerprint)
        A = self._S - np.diag(d)
        h = np.linalg.solve(A, self._r)
        if not np.isfinite(h).all() or np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
            raise ArithmeticError("NUMERICAL_PROBABILITY_INVALID")
        p = float(h[self.start_pos])
        result = {"status": "NUMERICAL_MODEL_RESULT", "model_fingerprint": self.fingerprint,
                  "success_probability": p, "retained": list(self.retained),
                  "retained_values": h.tolist(), "reductions": d.tolist(),
                  "source_states": len(self._b), "solve_states": len(self.retained)}
        if moments:
            if p <= 0:
                result.update(mean_success_time=None, variance_success_time=None)
            else:
                hp = np.linalg.solve(A, self._r1 - self._S1 @ h)
                hpp = np.linalg.solve(A, self._r2 - self._S2 @ h - 2 * self._S1 @ hp)
                mean = -float(hp[self.start_pos]) / p
                var = float(hpp[self.start_pos]) / p - mean * mean
                if not math.isfinite(mean + var) or mean < 0 or var < -1e-9 * max(1., mean**2):
                    raise ArithmeticError("NUMERICAL_MOMENTS_INVALID")
                result.update(mean_success_time=mean, variance_success_time=max(0., var))
        if certify or deadline is not None:
            full = self._M.copy()
            full[self._B, self._B] -= d
        if certify:
            reconstructed = np.empty(len(self._b))
            reconstructed[self._B] = h
            reconstructed[self._I] = self._y - self._X @ h
            result["certificate"] = residual_certificate(full, self._b, reconstructed, self.start)
        if deadline is not None:
            t = float(deadline)
            if not math.isfinite(t) or t < 0:
                raise ValueError("deadline must be finite and nonnegative")
            # Augmentation integrates the success flux without subtracting nearly
            # equal eventual-probability vectors at short deadlines.
            if t * float(np.max(np.diag(full))) > 10000:
                raise ValueError("DEADLINE_WORK_LIMIT: use rescaled/specialized transient methods")
            n = len(self._b)
            G = np.zeros((n + 1, n + 1))
            G[:n, :n], G[:n, n] = -full, self._b
            e = np.zeros(n + 1)
            e[n] = 1.
            cdf = float(expm_multiply(csr_matrix(G * t), e)[self.start])
            if not math.isfinite(cdf) or cdf < -1e-12 or cdf > p + 1e-10:
                raise ArithmeticError("NUMERICAL_DEADLINE_INVALID")
            result.update(deadline=t, success_before_deadline=max(0., cdf),
                          deadline_method="FULL_AUGMENTED_EXPONENTIAL_NOT_STATIC_REDUCTION")
        return result

    def select_bundle(self, sites, costs, budget, *, fraction, current_fingerprint,
                      max_candidates=10000):
        """Enumerate all feasible constant protection bundles in a bounded family.

        Objective: eventual success, NOT deadline success. Numerical selection,
        not a proof that nearly tied floating-point candidates are strictly ordered.
        Costs are arbitrary INTEGER budget units, not automatically joules.
        """
        self._validate(np.zeros(len(self.retained)), current_fingerprint)
        sites = [_index(v, len(self._b), "site") for v in sites]
        if len(sites) > 20 or len(set(sites)) != len(sites) or any(v not in self.retained for v in sites):
            raise ValueError("at most 20 unique retained intervention sites")
        def integer(v):
            if isinstance(v, (bool, np.bool_)):
                raise ValueError("integer cost required")
            try:
                return operator.index(v)
            except TypeError as exc:
                raise ValueError("integer cost required") from exc
        costs, budget, max_candidates = list(map(integer, costs)), integer(budget), integer(max_candidates)
        if len(costs) != len(sites) or budget < 0 or max_candidates < 1 or any(c <= 0 for c in costs):
            raise ValueError("positive integer costs and candidate cap; nonnegative budget")
        fraction = float(fraction)
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
        final = self.evaluate(best["reductions"], current_fingerprint=current_fingerprint, certify=True)
        final.update(selected_sites=best_sites, evaluated_candidates=count,
                     selection_scope="EXHAUSTIVE_FINITE_FAMILY_NUMERICAL_EVENTUAL_SUCCESS")
        return final


def goal_success(rates, goals, *, loss_rate, start=0, deadline=None):
    """Hit ANY declared goal before killing, with genuine multi-goal boundaries.

    Takes directed transition RATES, not distances or arbitrary heuristic scores.
    When adapted from an uncalibrated graph, the probability is for that synthetic
    random walk only; it is not an orbital mission success forecast.
    """
    if np.iscomplexobj(rates) or np.iscomplexobj(loss_rate):
        raise ValueError("real rates required")
    if hasattr(rates, "toarray"):
        if rates.shape[0] > 2000:
            raise ValueError("DENSE_PREPARATION_LIMIT")
        rates = rates.toarray()
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2 or rates.shape[0] != rates.shape[1] or not 0 < len(rates) <= 2000:
        raise ValueError("square rate matrix of 1..2000 states required")
    n = len(rates)
    start = _index(start, n, "start")
    goals = [_index(v, n, "goal") for v in goals]
    if not goals or len(set(goals)) != len(goals):
        raise ValueError("nonempty distinct goals required")
    loss = np.broadcast_to(np.asarray(loss_rate, dtype=float), (n,)).copy()
    _rates(rates, loss, np.zeros(n))
    if deadline is not None and (not math.isfinite(float(deadline)) or deadline < 0):
        raise ValueError("nonnegative finite deadline required")
    if start in goals:
        result = {"status": "AT_GOAL", "success_probability": 1., "mean_success_time": 0.,
                  "variance_success_time": 0.}
        if deadline is not None:
            result["success_before_deadline"] = 1.
        return result
    transient = [i for i in range(n) if i not in goals]
    i = transient.index(start)
    W = rates[np.ix_(transient, transient)]
    success = rates[np.ix_(transient, goals)].sum(axis=1)
    compiler = SuccessContract(W, loss[transient], success, [i], start=i)
    result = compiler.evaluate([0.], current_fingerprint=compiler.fingerprint,
                               deadline=deadline, certify=True)
    result["transient_node_ids"] = transient
    result["goals"] = goals
    return result
