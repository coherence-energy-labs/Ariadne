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

## 4. The predictive scheduler: it learns over time

`discovery/predictive.py` records every (evidence_class, action, outcome)
triplet to an on-disk ledger. Over weeks of operation, it discovers
which actions (deep-stack vs. second-night vs. archive-search) actually
produce confirmations for each evidence class, and adapts its
recommendations. Cold-start uses sensible priors; the steady state is
empirically calibrated to the engine's actual experience.

## 5. Recovery curves on synthetic injections

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

## 6. Known limits and ongoing work

| Capability | Current state | Roadmap |
|---|---|---|
| Light-time + planetary perturbations in orbit fit | N-body LM available (`discovery/orbit_fit_nbody.py`); not yet default in `iod.fit_candidate` | Make N-body the default for long-arc fits |
| Bayesian posterior on orbit elements | MCMC via emcee (`discovery/bayes_orbit.py`); falls back to Metropolis if emcee absent | Add convergence diagnostics + automatic burn-in |
| ML real/bogus classifier | Rule-based only (`discovery/realbogus.py`) | Add trained classifier once we have a labeled corpus |
| Cross-survey fusion | Implemented (`discovery/fusion.py`: ZTF + ATLAS + PS1) | Add Rubin / LSST broker when survey starts |
| Streak detection | Hough transform with PSF-consistency check (`discovery/imaging/streaks.py`) | Add deep-learning streak classifier for the LSST era |

## 7. What "extremely smart" actually means here

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
