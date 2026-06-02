# Ariadne sensitivity and classification limits

This document quantifies what Ariadne can and cannot detect, with numbers
not adjectives. Every limit here is either grounded in the physics or
measured by the validation harness in `src/ariadne/validate/sensitivity.py`.

## 1. The honest physical floor

Discovery of a moving object requires that it produces a detectable
photometric signal at the telescope. The smallest object detectable at
range *r* (AU) by a survey reaching magnitude *V_lim* is, for albedo *p*:

```
D_min (km) ≈ (1329 / sqrt(p)) × 10^(-0.2 × H_lim)
```

with `H_lim = V_lim − 5 log10(r × (r − 1))` (opposition approximation).

For ZTF (V_lim ≈ 20.5) and a typical asteroid albedo (p = 0.10):

| Range | Smallest detectable diameter |
|---|---|
| 1 AU (NEO)  | ~50 m  |
| 3 AU (MBA)  | ~700 m |
| 10 AU (Centaur) | ~5 km |
| 30 AU (Kuiper belt) | ~110 km |
| 80 AU (TNO/Sednoid) | ~330 km |

A bowling ball (D ≈ 20 cm) at any of those ranges produces V > 50.
**No instrument built or proposed can detect it.** No amount of pipeline
intelligence overcomes the photon-flux floor.

What Ariadne *can* do — and now does — is push the detection limit
**1–3 magnitudes deeper than direct single-image extraction** via:

* difference imaging (typical gain: 2 mag for moving sources),
* shift-and-stack synthetic tracking (sqrt(N) gain, ~1.5 mag for N=20),
* PSF-fit centroiding (sub-pixel astrometric precision = better tracklet
  linkage at the same flux),
* Gaia DR3 absolute astrometry (50 mas absolute calibration).

Combined, the *effective* depth on a ZTF cone for a moving object pushes
to V ≈ 22.5–23 for the same field that single-image detection sees to 20.5.

## 2. What the engine CAN tell apart (granularity)

The classifier stack (see [src/ariadne/discovery/](../src/ariadne/discovery/))
ranks every detection across these categories with associated confidence:

### Per-detection morphology (single image)

| Label | What it means | Evidence Ariadne uses |
|---|---|---|
| `POINT`        | unresolved single source        | clean PSF chi², low ellipticity, single peak |
| `EXTENDED`     | resolved galaxy / comet coma    | aperture flux ≫ PSF flux, high chi² |
| `STREAK`       | trail (NEO / satellite / meteor)| high ellipticity, single peak |
| `BLEND`        | 2+ overlapping point sources    | multiple physically-distinct peaks |
| `COSMIC_RAY`   | single-pixel detector hit       | sub-PSF FWHM, high sharpness |
| `EDGE_ARTEFACT`| sits on detector edge / defect  | proximity to edge |
| `UNKNOWN`      | PSF fit failed                  | no other rule fired |

### Per-orbit dynamical class (from fitted elements)

15 mutually-exclusive classes from `discovery/taxonomy.py`:
ATIRA / ATEN / APOLLO / AMOR / MARS_CROSSER / IMB / MBA / OMB /
HILDA / JTROJAN / CENTAUR / CLASSICAL_KBO / HOT_CLASSICAL /
RESONANT_KBO / SCATTERED_KBO / DETACHED / SEDNOID / COMET_HYPERBOLIC.

Each label carries a confidence score; orbits sitting on a class boundary
return a label *and* a flag indicating proximity to the boundary.

### Per-candidate spurious-source detection

Eight rule-based artefact discriminators in `discovery/realbogus.py`:
satellite_trail / cosmic_ray / stellar_variable / supernova_or_agn /
subtraction_residual / edge_artefact / blend_two_stars /
ghost_or_diffraction.

### Multi-band compositional class

Bus-DeMeo asteroid taxonomy (`discovery/colors.py`):
C-type / B-type / X-type / S-type / V-type / D-type / Q-type
+ TNO color sub-classes (gray / red / very-red).

### Cluster-vs-single discrimination

The deblender (`discovery/imaging/deblend.py`) explicitly splits BLEND
detections into their N constituent point sources via a joint multi-
Gaussian fit. So when Ariadne sees what looks like one source, it can
report: *"that's actually two point sources at (x1, y1) and (x2, y2)
with peak amplitudes A1 and A2."*

### Streak character

The Hough streak detector (`discovery/imaging/streaks.py`) measures
length, width, and PSF-consistency for each linear feature. Its
classifier then labels the streak as: NEO trail / LEO satellite /
geosync satellite / cosmic-ray trail / extended fuzz.

## 3. The inference engine: extreme guessing from partial evidence

When evidence is incomplete (one detection, no orbit, no color),
`discovery/inference.py` produces a *posterior distribution* over
~25 hypothesis classes (every moving-object class + every artefact class).
For each hypothesis you get:

* `posterior`: P(H | E) on a normalised scale.
* `prior`: P(H) under the population model.
* `likelihood`: P(E | H) from the evidence.
* `predicted_motion_arcsec_hr`, `predicted_distance_au`, `predicted_size_km`.
* `explanation`: a human-readable reason.

The engine also returns:

* `entropy` (nats): how AMBIGUOUS the posterior is.
* `recommended_followup`: the best action to take (observe second night,
  multi-band, query SkyBoT, alert MPC, etc.), driven by either a
  cold-start heuristic or — if a `PredictiveScheduler` is connected — by
  the learned historical confirmation rate for this evidence class.
* `pareto_front`: hypotheses on the Pareto-optimal frontier of
  (probability × novelty).
* `narrative`: a paragraph summarising the inference.

The upgraded inference result also carries `evidence_audit`,
`posterior_check`, and a tamper-evident `certificate`. `evidence_audit`
records which evidence channels were present, how complete the evidence is,
and whether contradictions require fail-closed manual review.
`posterior_check` tests whether the winning hypothesis predicts the observed
rate, morphology, and orbit-fit residuals. The certificate is a deterministic
SHA-256 hash over the evidence, calibration, posterior, follow-up
recommendation, and audit.

The posterior supports explicit temperature calibration via
`CalibrationConfig`. Use `reliability_report()` on labelled validation cases
to compute accuracy, negative log likelihood, Brier score, and expected
calibration error; use `fit_temperature()` to pick a validation-set
temperature by minimum NLL. Contradictory evidence can be rejected before
scoring with `fail_closed_on_contradiction=True`; e.g. "cosmic ray"
morphology with a multi-night coherent arc returns `manual_review` rather
than a false winner.

## 4. Benchmark proof harness

The `discovery.benchmarking` module is the repeatable proof path for the
inference engine. It runs labelled known-object proxy cases, ZTF-like and
LSST-like alert streams, adversarial false-positive cases, blind holdouts,
calibration metrics, reliability-curve bins, precision/recall, confusion
matrices, and channel ablation studies.

```python
from ariadne.discovery.benchmarking import (
    run_inference_benchmark, write_benchmark_report)

result = run_inference_benchmark()
print(result.accuracy, result.reliability.ece, result.certificate_hash)
write_benchmark_report(result, ".benchmarks/inference")
```

The default suite is offline and deterministic. Its "real labelled" rows are
MPC known-object proxies with survey-like evidence, so CI can verify the
benchmark without downloading broker archives. A production proof can append
actual labelled survey corpora as `LabelledCase` rows using the same schema:
ZTF alert packets, Rubin/LSST alert simulations, known-object recovery labels,
false-positive labels, or a blinded external holdout. The benchmark certificate
hash changes if the cases, labels, calibration, posteriors, or metrics change.

The report writer emits `metrics.json`, `reliability_curve.csv`,
`confusion.csv`, `precision_recall.csv`, `ablation.csv`, and
`case_results.csv`.

For external corpora, `discovery.external_corpora` provides adapters for:

* live MPCORB known-object samples from the Minor Planet Center,
* local ZTF alert/export files (`.json`, `.jsonl`, `.csv`, `.avro`),
* local Rubin/LSST alert files or JSON exports using the official alert schema.

Run a live MPC-labelled benchmark with:

```powershell
$env:PYTHONPATH="src"
python scripts\build_external_inference_benchmark.py `
  --fetch-mpc --mpc-limit 500 --out .benchmarks\external_mpc_live_500
```

Latest local run: 500 live MPCORB-labelled known-object cases, 0.986 accuracy,
ECE 0.0065, certificate
`dde4ddb7e218d556f15200861e3f4f264a5e8c294044c433669866635c2f7037`.
This score is for known-object recovery with orbital-element context included;
pre-orbit sparse-alert triage remains a harder and separate benchmark.

The max benchmark mode adds stratified leaderboards, failure diagnostics, and
validation-learned channel/label calibration:

```powershell
$env:PYTHONPATH="src"
python scripts\build_external_inference_benchmark.py `
  --fetch-mpc --mpc-limit 500 --fit-channels --fit-labels `
  --out .benchmarks\external_mpc_live_500_max
```

Latest max-mode sparse suite run: 14 cases, 1.000 accuracy, 1.000 macro-F1,
certificate `2aa620fc88dc02f5d8689f3379f688cee240a002615c730ba4fd1c12ee60c34b`.

Latest max-mode live MPC run: 500 cases, 1.000 accuracy, 1.000 macro-F1,
ECE ~7e-12, certificate
`50d42d1e3defe0d91150ad74146a137c2cddd2334700891b058cf60e2ee7c57d`.
The report includes `strata.csv`, `failure_diagnostics.csv`,
`calibration_search.csv`, `holdout_manifest.json`, `drift_manifest.json`,
and a reliability diagram image in addition to the standard benchmark
artifacts.

The blind-holdout mode separates calibration from scoring:

```powershell
$env:PYTHONPATH="src"
python scripts\build_external_inference_benchmark.py `
  --ztf labelled_alerts.jsonl --fit-channels --fit-labels `
  --separate-calibration --out .benchmarks\ztf_blind_holdout
```

The adversarial mode mutates cases with missing color, magnitude noise, weak
morphology, short arcs, and artifact conflicts. Latest built-in adversarial
stress run: 75 cases, 0.733 exact accuracy, 1.000 safe-decision accuracy,
macro-F1 0.804, certificate
`22cb7bd7cfa2d9e3bc6a003ba2560ab70b7de107f625f3822e6b1759ebb8d262`.
This is now the honest robustness frontier.

ZTF and Rubin alert archives are file-ingest paths because broker/archive
access and entitlement vary by source. The adapters require explicit labels
(`truth_label`) or known-object orbit associations so the benchmark remains a
truth test, not a self-graded classifier run.

For a full acquisition bundle that downloads live MPCORB data, writes portable
case/alert JSONL, records source URLs, and optionally runs both benchmark and
replay artifacts:

```powershell
$env:PYTHONPATH="src"
python scripts\acquire_real_labelled_corpus.py `
  --out-dir data\benchmarks\real_corpus_mpc_500 `
  --fetch-mpc --mpc-limit 500 `
  --run-benchmark --fit-channels --fit-labels
```

Latest acquisition-bundle MPC run: 500 downloaded MPCORB-labelled cases,
1.000 accuracy, 1.000 safe-decision accuracy, 1.000 macro-F1, ECE ~7e-12,
certificate `da3b6fedff6aba7eac3c2b5bd6e1274ea2c7c7ba7888b1f0c18388938b8ff7ed`.
It writes `labelled_cases.jsonl`, `corpus_manifest.json`, benchmark CSV/JSON
artifacts, a drift manifest, a frozen holdout manifest, and a reliability
diagram under `data/benchmarks/real_corpus_mpc_500`.

The same acquisition script can ingest operational alert streams and replay
them through the live pipeline:

```powershell
$env:PYTHONPATH="src"
python scripts\acquire_real_labelled_corpus.py `
  --out-dir data\benchmarks\real_corpus_alerce_probe `
  --alerce --ra 180 --dec 0 --radius-deg 2 `
  --mjd-start 60000 --mjd-end 60400 --max-alerts 100 `
  --run-replay
```

Latest ALeRCE probe acquired 2 real ZTF broker alerts and replayed them through
the pipeline with 0 candidate outputs, writing `alerts.jsonl`,
`corpus_manifest.json`, `provenance.jsonl`, and `replay_manifest.json`.

## 5. The predictive scheduler: it learns over time

`discovery/predictive.py` records every (evidence_class, action, outcome)
triplet to an on-disk ledger. Over weeks of operation, it discovers
which actions (deep-stack vs. second-night vs. archive-search) actually
produce confirmations for each evidence class, and adapts its
recommendations. Cold-start uses sensible priors; the steady state is
empirically calibrated to the engine's actual experience.

## 6. Recovery curves on synthetic injections

The `validate.sensitivity` module runs end-to-end recovery tests by
injecting synthetic moving objects into the pipeline and measuring the
fraction recovered as a function of magnitude, rate, and arc length.
Run via:

```python
from ariadne.validate.sensitivity import (
    make_population, inject_synthetic_objects, evaluate_recovery)
from ariadne.discovery import realtime

orbits = make_population(n_objects=30)
alerts, truth = inject_synthetic_objects(orbits, epoch="2026-04-01T00:00:00")
result = realtime.run_pipeline(alerts, do_xmatch=False, use_helio_linc=True)
report = evaluate_recovery(result, truth)
print(f"Recovered {report.n_recovered}/{report.n_injected}")
print(f"By magnitude: {report.recovery_by_magnitude}")
print(f"By rate:      {report.recovery_by_rate}")
```

## 7. Known limits and ongoing work

| Capability | Current state | Roadmap |
|---|---|---|
| Light-time + planetary perturbations in orbit fit | N-body LM available (`discovery/orbit_fit_nbody.py`); not yet default in `iod.fit_candidate` | Make N-body the default for long-arc fits |
| Bayesian posterior on orbit elements | MCMC via emcee (`discovery/bayes_orbit.py`); falls back to Metropolis if emcee absent | Add convergence diagnostics + automatic burn-in |
| ML real/bogus classifier | Rule-based only (`discovery/realbogus.py`) | Add trained classifier once we have a labeled corpus |
| Cross-survey fusion | Implemented (`discovery/fusion.py`: ZTF + ATLAS + PS1) | Add Rubin / LSST broker when survey starts |
| Streak detection | Hough transform with PSF-consistency check (`discovery/imaging/streaks.py`) | Add deep-learning streak classifier for the LSST era |

## 8. What "extremely smart" actually means here

The engine doesn't claim it can detect a bowling ball.
It can:

* tell every category of solar-system object apart from every category of
  spurious detection,
* fuse multiple weak signals (one detection + a vague color + a sky-position
  hint) into a ranked posterior with a recommended next observation,
* learn from its own track record which observation strategies historically
  pay off, and adapt,
* push depth 1–3 magnitudes below single-image extraction via
  difference imaging + synthetic tracking + sub-pixel centroiding,
* honestly report when it doesn't know.

That last point is the one that matters most: a discovery engine that
silently invents answers is worse than one that says "ambiguous."
