# Changelog

All notable user-facing changes to Ariadne. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/) and the project follows
[Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH` with PEP 440 pre-release
suffixes for `rc` / `a` / `b`).

## Unreleased

_(nothing yet)_

## 1.0.0rc2 — 2026-05-30

The first release-candidate. The release that turns Ariadne from a research project
into a Python-native cislunar mission-design + TNO-discovery toolkit anyone can install
and use.

### Added — top-level user-facing API
- `ariadne.system(name)` returns CR3BP `System` for any of 7 named systems
  (Earth-Moon, Sun-Earth, Sun-Mars, Jupiter-Io / Europa / Ganymede / Callisto).
- `ariadne.lyapunov_family(point, n)` and `ariadne.halo_family(point, n)` build
  periodic-orbit families by continuation in one line.
- `ariadne.gateway_nrho()` returns NASA's Gateway-class 9:2 NRHO (period 6.56 d,
  perilune 3.2 Mm, apolune 71 Mm, Floquet 2.18) from pseudo-arclength continuation.
- `ariadne.discover_tno(designation)` pulls real MPC astrometry, runs HelioLinC-style
  IOD + LM differential correction, returns a 6D heliocentric state with light-time
  correction and an RMS residual in arcsec.
- `ariadne.helmholtz_hjb(samples, source_idx)` returns the sampled-graph Helmholtz
  value function for arbitrary N-dim OCPs.
- `ariadne.certify_route(graph, path)` builds proof-carrying route certificates.

### Added — Stage 19 cislunar transport (3 new validation gates)
- 3D-halo transport graph via `build_transport_graph_3d` with 4D closest-approach in
  (y, z, vy, vz).
- L1↔L2 halo heteroclinic patches with energy-consistent Δv (machine-precise).
- Generalised Poincaré section: `axis` parameter (0=x, 1=y, 2=z) on
  `tube_section_cut_3d` and `first_section_crossing`.
- NRHO transport via y=0 section: NRHO unstable manifold ↔ L2 halo stable manifold
  patches at ~119 m/s.
- New gates: G19c (3D graph builds), G19d (3D heteroclinic edge),
  G19e (NRHO transport).

### Added — Stage 22 DSM refinement (1 new validation gate)
- Per-leg Deep-Space Maneuver fractions on the VEEGA optimizer: 1-DOF
  (frac only, Lambert-determined impulse) or 4-DOF (frac + 3D Δv kick with
  position-error penalty).
- DE optimizer extends from 5-dim (epoch + 4 TOFs) to 5+4·#DSMs dimensions; warm-start
  from a 1-DOF solution to 4-DOF lifts the population correctly.
- Full re-optimization on the Galileo VEEGA reference finds a trajectory with
  5,392 m/s total Δv vs the no-DSM 6,449 m/s (16% reduction, the genuine Galileo-class
  refinement).
- New gate: G22c (DSM mechanism + regression-clean DE handling).

### Added — Stage 44 discovery pipeline closure
- `discovery/iod.py`: rebuilt orbit fitter with linker-(r, ṙ)-hypothesis IOD seed +
  LM differential correction + light-time correction + pos-AU/vel-km/s state rescaling.
- Validated on real MPC astrometry: Sedna 3.94", Eris 5.79", Makemake 8.68",
  Quaoar 3.79", 2001 FP185 1.39"; semi-major-axis recovery 0.1–13%.
- Discrimination test: Sedna alone accepted at 3.94"; Sedna+Eris mixed rejected at inf.
- Final ITF verdict on the live MPC archive: 515/863 candidates re-link to known
  catalogued objects + 0/348 unmatched survive the orbit-fit filter — the honest,
  scientifically defensible outcome on a public archive.

### Added — Stage 45 Coherence-HJB (4 new validation gates)
- `optimize/coherence_hjb.py`: sampled-graph Helmholtz value-function solver as the
  6D HJB curse-of-dimensionality bypass.
- Halton sampling, k-NN Gaussian-kernel graph (Belkin-Niyogi), graph Laplacian,
  sparse CG Helmholtz solve, surprise log-cost transform, greedy-policy walker.
- New gates: G45a (2D analytic eikonal, rho=0.999), G45b (4D/6D synthetic scaling,
  100% greedy reach), G45c (CR3BP planar dynamics-aware, 29/29 reach), G45d
  (FULL 6D CR3BP production, 27/29 = 93% reach, sub-second compute).

### Added — Stage 46 certified route promotion (parallel agent contribution)
- `certification.py`: proof-carrying route certificates (canonical JSON, SHA-256 hash,
  required rungs, replay commands).
- Certifies an Earth-Moon L1↔L2 heteroclinic through CR3BP→BCR4BP→DE440 (0 m/s patch,
  271,225 km BCR4BP divergence, 207.7 km/(m/s) sensitivity, 736 m/s DE440 correction,
  5.6 m residual).
- Fail-closed on missing required evidence; tampering breaks the certificate hash.

### Added — polish layer
- 5 runnable tutorial scripts in `examples/` (Lyapunov family, Gateway NRHO, manifold
  transport, TNO orbit fit, Coherence-HJB), each ~50–100 lines, all producing PNGs.
- 5 Jupyter notebook tutorials in `notebooks/` mirroring the scripts.
- `benchmarks/reference_targets.py`: 16-check reference suite (CR3BP constants,
  Lagrange points, NRHO geometry, TNO orbit recovery, Jacobi conservation) —
  16/16 PASS, no external tools required.
- Sphinx documentation under `docs/sphinx/` (Furo theme, autodoc, myst-parser, full
  module-level API reference + tutorials + architecture + validation + honest_scope).
- `.readthedocs.yaml` for hosted docs.
- README rewrite: user-first pitch + 30-second demo + install + tutorial gallery
  (5 embedded PNGs) + tool-landscape comparison + capabilities table + cross-validation
  table + honest scope + architecture diagram.
- CI workflow extended: Python 3.10–3.13 matrix, top-level API smoke check,
  tutorial-01 smoke check, separate Sphinx docs build job.
- `pyproject.toml` polish: expanded keywords + classifiers, new optional extras
  (`docs`, `autodiff`, `gpu`, `notebooks`, `crosscheck` with REBOUND).
- `RELEASING.md` with the full PyPI release procedure.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

### Changed
- `src/ariadne/__init__.py` exposes the user-facing API (was previously empty
  except for `__version__`).
- `pyproject.toml` description sharpened to reflect the cislunar + TNO discovery focus.

### Fixed
- Discovery orbit-fit shim (the prior implementation produced ~800" residuals even
  on real known TNOs — circular-velocity-in-equatorial-plane initial guess was
  wrong direction for any inclined orbit).
