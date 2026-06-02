# Changelog

All notable user-facing changes to Ariadne. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/) and the project follows
[Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH` with PEP 440 pre-release
suffixes for `rc` / `a` / `b`).

## Unreleased

### Hardening sprint — verification, honesty, polish

What was *claimed* to work; this sprint actually exercised everything end-to-end and
corrected loose numbers.

- **Discovery inference hardening**: `discovery.inference` now emits evidence audits,
  posterior-predictive checks, tamper-evident inference certificates, per-hypothesis
  evidence-term traces, temperature-scaled posterior calibration, reliability reports
  (accuracy/NLL/Brier/ECE), and fail-closed contradictory-evidence handling. This turns
  sparse-evidence "extreme guessing" into a calibrated, auditable inference artifact.
- **Discovery inference benchmark harness**: `discovery.benchmarking` now runs labelled
  known-object proxy cases, ZTF/LSST-like alert streams, adversarial false positives,
  blind holdouts, precision/recall, confusion matrices, reliability curves, calibration
  scoring, ablation studies, and deterministic benchmark certificates. Reports emit JSON
  plus CSV artifacts for independent review.
- **External labelled-corpus ingest**: `discovery.external_corpora` plus
  `scripts/build_external_inference_benchmark.py` now import live MPCORB known-object
  samples and local ZTF/Rubin alert exports into the `LabelledCase` benchmark schema.
  The first live MPC run wrote `.benchmarks/external_mpc_live_500` with 500 labelled
  external cases. After adding recovered orbital-element context for known-object
  recovery, the live MPC run reaches **0.986 accuracy** with certificate
  `dde4ddb7e218d556f15200861e3f4f264a5e8c294044c433669866635c2f7037`.
- **Inference improvement cockpit**: benchmark results now include stratified
  leaderboards, failure diagnostics, low-margin risk cases, channel-weight search,
  and label-bias calibration search. Max-mode sparse benchmark reaches **1.000
  accuracy** / **1.000 macro-F1** with certificate
  `2aa620fc88dc02f5d8689f3379f688cee240a002615c730ba4fd1c12ee60c34b`; max-mode
  live MPC recovery reaches **1.000 accuracy** / **1.000 macro-F1** on 500 labelled
  MPCORB cases with certificate
  `50d42d1e3defe0d91150ad74146a137c2cddd2334700891b058cf60e2ee7c57d`.
- **Real corpus acquisition bundle**: added `scripts/acquire_real_labelled_corpus.py`
  plus portable case/alert JSONL round-tripping for downloaded MPCORB samples,
  local ZTF/Rubin exports, and ALeRCE ZTF alert probes. The script writes
  source manifests, benchmark artifacts, replay manifests, and provenance ledgers.
  Latest acquisition-bundle MPC run: 500 downloaded labelled cases, **1.000
  accuracy**, **1.000 safe-decision accuracy**, **1.000 macro-F1**, certificate
  `da3b6fedff6aba7eac3c2b5bd6e1274ea2c7c7ba7888b1f0c18388938b8ff7ed`.
  Latest ALeRCE probe acquired 2 real ZTF broker alerts and replayed them with
  deterministic zero-candidate output.
- **Benchmark anti-overfit hardening**: benchmark reports now include safe-decision
  accuracy, frozen holdout manifests, train/eval split calibration, deterministic
  adversarial mutations, drift manifests, and reliability diagram images. Latest
  built-in adversarial stress run: 75 cases, **0.733 exact accuracy**, **1.000
  safe-decision accuracy**, macro-F1 0.804, certificate
  `22cb7bd7cfa2d9e3bc6a003ba2560ab70b7de107f625f3822e6b1759ebb8d262`.
- **Operational discovery hardening**: added chronological alert replay,
  append-only provenance ledgers with input/output hashes, pipeline provenance
  wrappers, real/bogus severity/action/explanation fields, scheduler
  hypothesis-separation metadata, and a dashboard `/api/ops` endpoint for
  operator-facing state.
- **Replay drift tooling**: replay outputs now summarize candidate keys, status
  counts, action counts, and write comparable manifests. Added
  `scripts/compare_replay_manifests.py` plus nightly `provenance_path` support
  so scheduled runs can emit the same audit trail.
- **Cislunar mission architect**: added `transfers.mission_architect`,
  `ariadne.architect_cislunar_round_trip()`, and
  `scripts/architect_cislunar_mission.py`. The architect combines direct
  full-ephemeris Earth->Moon targeting, CR3BP low-energy capture, coherence
  Pareto candidates, and analytic Moon->Earth return screening into a certified
  Earth-Moon-Moon-Earth route catalogue with explicit fidelity tags,
  assumptions, validation notes, Pareto flags, and deterministic hashes.
- **Solar-system navigator**: added `interplanetary.navigator`,
  `ariadne.navigate_solar_system()`, and `scripts/navigate_solar_system.py`.
  The navigator resolves planet/moon targets, combines direct real-ephemeris
  Lambert porkchops, optimized flyby-chain templates, and Jupiter/Saturn
  Tisserand moon-tour screening, then writes certified route cards plus PNG
  heat-map/trade-space artifacts. Giant-planet flyby turn authority now uses
  Jupiter/Saturn GM and radius instead of Earth fallback values.
- **Solar navigator benchmark hardening**: added
  `interplanetary.navigator_benchmark` and
  `scripts/benchmark_solar_navigator.py`. The benchmark runs real-SPICE Mars
  and Enceladus cases, validates route-card invariants and physical sanity
  bounds, checks generated PNG artifacts, and writes a certificate-bearing
  summary. Porkchop and launch-window sweeps now cache departure ephemeris
  states across TOF rows to avoid repeated SPICE calls while preserving exact
  Lambert outputs.
- **Solar navigator route intelligence**: reports now include representative
  direct-Pareto routes across the time/energy frontier, a human-readable
  `route_cards.md`, and `scripts/compare_solar_navigator_benchmarks.py` for
  drift comparison of benchmark summaries. This exposes fastest/cheapest/
  balanced/in-between routes instead of only the single minimum.
- **Fresh-venv install verified**: built the wheel, installed in a clean venv, ran
  `import ariadne` + `ariadne.lyapunov_family('L1', n=3)` cleanly.
- **Sphinx `-W` build clean**: fixed 9 docutils warnings (the `|...|` substitution-
  reference issue in module docstrings + RST). CI can now use `-W --keep-going`.
- **Notebooks execute end-to-end**: ran all 5 `.ipynb` via `jupyter nbconvert --execute`;
  added missing cell IDs (nbformat 5+ requirement) via `scripts/fix_notebook_ids.py`.
- **CLI sweep**: `ariadne {info, systems, lyapunov, nrho, benchmark}` all clean.
- **Heteroclinic Δv honesty correction**: the earlier "112 m/s L1↔L2" and "119 m/s
  NRHO↔L2" were velocity-mismatch ONLY. Honest totals (with 1-day correction window)
  are **L1↔L2 = 162 m/s** (112 vel + 50 rendezvous) and **NRHO↔L2 = 646 m/s** (119 vel
  + 527 rendezvous). Sanity: same-orbit-to-itself = 6.7 m/s (essentially ballistic).
  See `benchmarks/heteroclinic_honest_dv.py`.
- **Coherence-HJB Δv honesty correction**: the earlier "p50 = 2.88 km/s" was velocity-
  mismatch ONLY (suspiciously below Hohmann). Honest total is **p50 = 7.32 km/s** —
  in the Hohmann-to-Edelbaum band. See `benchmarks/stage45_honest_dv.py`.
- **Discovery synthetic-injection positive control**: planted 33 real Sedna tracklets in
  a 200-interloper haystack. **100% pure recovery** (29/29 Sedna, zero interlopers in
  the best cluster); orbit fit a=527 AU (JPL 506, 4% err), e=0.854 (JPL 0.85), RMS
  3.80″. Validates the positive direction; combined with the 0/348 negative on live ITF
  the pipeline is end-to-end-validated in both directions.
  See `benchmarks/discovery_synthetic_injection.py`.
- **Reference benchmark tolerances tightened**: NRHO period 0.05d→0.01d, perilune
  600km→100km, apolune 5000km→500km; Sedna/Quaoar a-error 15%→5%. Plus 2 new
  L2-halo-self-heteroclinic sanity checks. The loose old tolerances were hiding
  regressions.
- **README + Sphinx validation table** corrected to quote honest patch Δv totals.
- **Stage 22 DSM benefit correction (BIG honesty fix)**: the earlier "16% reduction"
  claim (5392 m/s with DSMs vs 6449 m/s stored Galileo reference) was a **seed/topology
  artefact, NOT a real DSM benefit**. The 6449 stored reference is from an older,
  less-converged DE; a matched-conditions DE (same seed, same maxiter, same popsize)
  with no DSMs finds **6207 m/s median** (5 seeds) — that's the 4% "improvement"
  from re-running the optimizer better. With DSMs allowed (9 dims): **6220 m/s median**
  — DSMs make it **0.2% WORSE**. The DSM mechanism is implemented correctly (regression-
  clean) but doesn't help on this specific trajectory geometry. See
  `benchmarks/stage22_dsm_isolation.py`. The 1-DOF DSM remains valid for *other*
  trajectories where directional kicks help; the Galileo VEEGA at this epoch is just
  not one of them.

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
