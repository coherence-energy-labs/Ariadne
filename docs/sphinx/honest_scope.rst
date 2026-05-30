Honest scope — what Ariadne is, and isn't
==========================================

What it **is**
--------------

An open, high-fidelity, Python-native CR3BP + cislunar mission-design + TNO-discovery
toolkit built on standard gravity and real ephemerides, cross-validated against NASA GMAT.

- **Validated**: every claim has an independent cross-check (GMAT, REBOUND, jplephem,
  published reference values). 45+ stages, all gates green.
- **Tutorial-driven**: five runnable examples in ``examples/``, each ~50 lines, each
  ending in a PNG you can look at.
- **Pip-installable**: ``pip install -e .`` then ``import ariadne`` works.
- **Honest**: null results are reported as null. The ITF discovery run found **0 new
  objects** — that is the *correct* outcome on a public archive the MPC has already
  picked over.

What it **is not**
------------------

- **New physics.** The dynamics are standard n-body gravity (CR3BP, BCR4BP, full DE440
  ephemeris). Coherence-field methods (Stage 45 sampled-graph Helmholtz) are a search-
  acceleration layer on top of standard gravity — never a replacement for the physics.
- **An operational mission-planning platform.** For real ops, use GMAT, Monte, or
  Copernicus. Ariadne is for design, exploration, education, and research.
- **A discovery service.** It's a toolkit you run on data — we don't operate an MPC
  alert stream.
- **A replacement for poliastro / Tudat / Heyoka.** Those tools are excellent at what
  they do (2-body + perturbations, complex OCP, high-precision propagation). Ariadne
  fills the *cislunar + manifold + TNO discovery* gap they leave open.

Honest limitations by subsystem
-------------------------------

**CR3BP core**: synodic-frame restricted three-body, mass parameter μ only. No
oblateness perturbations within the CR3BP model (added separately via the DE440 path).

**Transport graph**: discovery + ranking of low-energy heteroclinic-class connections
between libration orbits at finite Jacobi-energy resolution. The optimum *route topology*
is verified stable across manifold resolution; the *exact Δv* depends on the (n_seeds,
t_max, section position) discretization and is reported with provenance.

**Interplanetary**: patched-conic with Lambert arcs + per-leg DSMs. Gravity-assist
turn-feasibility is checked against the flyby's geometric authority at the chosen
altitude. The 4-DOF DSM model (full 3D Δv kick + position-error penalty) is implemented
but for the specific Galileo VEEGA reference the 1-DOF Lambert-determined solution is at
a global optimum — extra DOFs don't help that particular geometry.

**Coherence-HJB (Stage 45)**: sampled-graph Helmholtz value function over CR3BP phase
space. Validated in 2D analytic eikonal (rho 0.999), dimension-scaling 4D-6D synthetic
(100% greedy reach at N=30k), and planar + full 6D CR3BP (29/29 = 100% planar; 27/29 =
93% in 6D). Sub-second compute. *Not* claimed to beat GMAT/Monte for operational design;
*is* claimed to beat naive 6D grid HJB (which is intractable at any practical resolution).

**Discovery**: HelioLinC-style linker validated on synthetic (5/5 recovery) and real
(Sedna + 2001 FP185 from MPC astrometry). IOD + LM orbit fit validated on 5 real TNOs
(RMS 1.4–8.7"). The full ITF sweep is honest: 515/863 known re-links + 0 new
discoveries. A genuine new find requires Rubin/LSST-class data deeper than the public
MPC bar — never announced from a single pipeline run.

**Certification**: route certificates are proof-carrying artifacts (canonical JSON +
SHA-256 hash + explicit required-rung evidence). The CR3BP rung is the model itself;
BCR4BP survivability is a real divergence measurement; DE440 retargeting requires real
multi-shooting evidence; GMAT replay is explicit ``not_run`` unless an installed
GmatConsole is actually invoked. Missing evidence rejects fail-closed; hash tampering
breaks the certificate.

Provenance + reproducibility
----------------------------

Every published number comes from a numbered validation stage with reproducible code:

- Source: ``src/ariadne/`` (clean public API + internal modules).
- Gates: ``src/ariadne/validate/stage##.py`` (each is a runnable script with
  PASS/FAIL gates).
- Tests: ``tests/test_*.py`` (pytest suite, run on Python 3.10–3.13 in CI).
- Cross-checks: ``MASTER_PLAN.md`` records the GMAT/REBOUND/jplephem comparison values
  with their measured magnitudes (e.g., "Ariadne vs GMAT 149 m / 0.89 mm·s⁻¹ over 3 days").

Numbers are reported from the construction, never fitted to a target. When a
construction misses a published target (e.g., the Coimbra 3,925 m/s low-energy lunar
transfer), the gap is reported with its physical reason rather than papered over.
