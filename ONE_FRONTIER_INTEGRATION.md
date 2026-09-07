# ONE Frontier integration: directed success contracts

Base: `88ad47819b54079d1965d050fb0d259c2d1b6d9f`. This is an opt-in API, not a replacement for the existing geometric Helmholtz/HJB heuristic.

## Public entry points

```python
import ariadne

rates = [[0, 2, 1], [0, 0, 1], [0, 0, 0]]
answer = ariadne.goal_success(rates, [2], loss_rate=0.2, deadline=3)
model = ariadne.compile_success_contract(
    [[0, 2], [0, 0]], [0.2, 0.3], [0, 1], [0, 1])
answer = model.evaluate([0.1, 0.1], current_fingerprint=model.fingerprint,
                        deadline=3, certify=True)
bundle = model.select_bundle([0, 1], [1, 1], 1, fraction=0.5,
                             current_fingerprint=model.fingerprint)
```

Inputs are **directed continuous-time transition rates**, positive failure rates, and nonnegative absorbing-success rates, all in consistent inverse-time units. Arbitrary k-NN weights are not a calibrated orbital success model. `goal_success` uses actual absorbing goal boundaries, including multiple goals; it does not average or normalize multiple source amplitudes.

`SuccessContract` compiles hidden states by Schur complement. Repeated constant failure-rate reductions use boundary-sized solves. First two resolvent derivatives preserve successful-arrival mean and variance. Explicit deadline queries deliberately use the **full augmented matrix exponential**, because a static Schur complement is not a memoryless time model. Large deadline work is refused rather than running an unbounded exponential computation.

Only failure-rate reductions at declared retained states are supported. Dense preparation is capped at 2,000 states. Budget selection enumerates a finite family of at most 20 sites; integer costs only, explicit candidate cap, and refusal rather than a false optimum on exhaustion. The objective is **eventual** success, not deadline probability. Numerical near-tie ordering is not a proof of unique optimality.

The caller must supply the current model fingerprint. This is a stale-use tripwire, not proof the caller supplied live telemetry. Hashing uses a schema-bound, canonical binary64 little-endian representation including shape and normalizing signed zero.

## What the certificate proves

The final-answer checker recomputes an exact rational residual against the supplied binary64 matrix and uses strict row diagonal dominance to bound probability error. It does not use the producer's factorization. This is a computational certificate for **that matrix**, not proof of the physical rate model or timing moments. It costs O(n^2) rational work and is deliberately charged separately from repeated-query timing.

## Measured source-level evidence

75 focused tests passed locally: 400 randomized interventions compared with independent full dense solves, successful-time moments, full exponentials, an ODE integration, exact-fraction checks, invalid/stale inputs, directed reachability, multi-goal boundaries, finite-budget exhaustion, canonical-hash invariance, and the root package facade.

An independent 100,000-trajectory event-driven simulation predicted/observed success 0.76695715/0.76911 and deadline success 0.64882641/0.65100: 1.61 and 1.44 binomial standard errors respectively.

Five local single-thread CPU repeats, one synthetic 300-state model, 11 retained states, 56 repeated queries: median preparation 2.894 ms, queries 1.334 ms, fresh full sparse re-solves 133.867 ms. A standard prepared Woodbury comparator took 1.118 ms preparation plus 0.849 ms queries and was faster than this compiler on the eventual-only contract. Maximum disagreement was 1.39e-16. No universal speed or new-algorithm claim; the compiler additionally prepares timing information and a typed interface. Final rational certification and full deadline queries are not included in these batch times.

Clock-sign, stale-model-bypass, and eventual-for-deadline mutants were rejected by assertions. Exact base root/constants source blobs were verified; original root API text is preserved before two lazy additions. The existing coherence_hjb.py is unchanged.

## Review boundary

The new Ubuntu/Windows focused workflow exercises this module and the root facade. Local execution used Git-blob-verified source dependencies, not a complete repository checkout or installed astrophysics dependencies. Full repository suites, Ruff, GPU, SPICE data, CR3BP operational mission calibration and physical experiments are NOT claimed. Independent full CI and maintainer review are required before merge/deployment. This adds no new force model, proof tier, production switch or live ACE action.

Reproduce: `python -m pytest -q -o addopts= tests/test_success_contract.py`.
