# Coherence / Equation-of-ONE — Real-Data Scorecard

Every place the coherence framework is used as a **selector** in the discovery
pipeline, tested against its non-coherence baseline on **real data**. Date:
2026-06-02. Reproduce with the scripts named in each row.

The governing lesson (see `coherence_vet.py` header and the memory keystone):
the Equation-of-ONE energy is a **fast selector over candidates**, never a brute
search. Used that way it wins; used as a search it loses (the linker experiment,
below, is kept as the honest negative control).

## Scorecard

| selector | real data | baseline | baseline | **coherence** | verdict |
|---|---|---|---|---|---|
| **Mover / orbit class** (`classify_mover`) | 60k MPC orbits (`known_objects`) | distance-only cut | 91.2% | **97.9%** | **+6.7pp**; NEO recall **31%→96%**, eccentric pop **+44pp** |
| **Candidate vetting** (`coherence_vet.track_energy`) | 4k real asteroid orbits propagated across nights | hard AND-rules | F1 0.888 | **F1 0.917** | **+2.9pp**, dominates (higher prec AND +5.6pp recall) |
| **Variable typing** (`classify_variable`) | **260** VSX variables + real ZTF light curves | light-curve only (no color) | ~74% | **89%** type, 92% period | color fusion breaks RR/EW & Cep degeneracy; holds on 7× sample |
| Linker (search, NOT a selector) | 8.5k real DECam detections | pairwise tracker | **2/9, floor 4, <1s** | 1/9, floor 193, 75s | **baseline wins** — honest negative control |

## Detail and honest caveats

### Mover / orbit classifier — `scripts/validate_mover_real.py`
- 60,000 real asteroids sampled from the 1.54M-row MPC orbit DB. True dynamical
  class from the standard (a, e) definitions (q = a(1−e): NEO q<1.3, Mars-crosser
  1.3≤q<1.666, main-belt, outer-belt/Hilda/Trojan, Centaur, TNO).
- The coherence **fusion** (adding eccentricity → perihelion q) beats the
  distance-only cut **97.9% vs 91.2%** overall, **+43.9pp on eccentric objects**
  (e>0.3), and lifts **NEO recall from 31% → 95.7%**.
- Real data both *motivated and validated* the fix: the first pass keyed NEOs on
  mean distance (a) and got 30% — real orbits exposed it; re-keying the basins on
  perihelion q (only computable through the ecc fusion) fixed it. This is the
  asteroid analog of the color axis for variables.
- Honest: the headline +6.7pp is modest because main-belt is 91% of all asteroids
  and distance alone already nails it; the framework's value is concentrated in
  the high-value eccentric/NEO tail, which is exactly where it should be.
- Caveat: true class is itself defined from (a, e), so this measures whether the
  basins reproduce the taxonomy and where ecc-fusion helps at the boundaries — a
  fair fusion-vs-no-fusion comparison on the real population, not an independent
  label source.

### Candidate vetting — `scripts/validate_coherence_vet_real.py` (+ `validate_coherence_vet.py`)
- Positives: real asteroid orbits propagated to 4 nights × 2 exposures (real sky
  rates, real Keplerian curvature, real H-brightness, 0.15″ astrometric jitter).
- EoO energy selector **F1 0.917 vs hard-rules 0.888 (+2.9pp)**, with higher
  precision *and* +5.6pp recall at the rules' operating point, in the same
  millisecond budget (it scores a few linked chains — a selector, not a search).
- Synthetic linear-arc A/B gave +5.1pp; real curved motion narrows it to +2.9pp
  (real arcs carry slightly higher linear residual). Win holds either way.
- Caveat: positives are real orbital motion; the chance **negatives** are modeled
  (we lack a real dense-field linker false-positive stream). Wired into the
  pipeline as `run_discovery_benchmark.py --vet-mode coherence` (default `rules`).

### Variable typing — `scripts/validate_lightcurve.py`
- 38 known VSX variables (RRab, EA, EW, DCEP) with real ZTF light curves.
- 100% period recovery, **89% type** accuracy. The coherence classifier fuses
  period + R21 + amplitude + **color (g−r)**; color is the axis that separates the
  light-curve-degenerate classes (RR Lyrae vs contact EW). Remaining misses are
  physically degenerate (reddened Cepheids, EW/RRc overlap) — not fixable without
  more bands.

### Linker — honest negative control — `scripts/run_coherence_field_discovery.py`
- Applying coherence as a **search** (field sourced by all detection pairs) over
  8,485 real DECam detections LOST to the plain pairwise tracker on recall (1 vs
  2 of 9 knowns), precision (chance floor 193 vs 4), and speed (75s vs <1s).
- Kept deliberately as the control that proves the boundary: coherence is a
  decision/selection tool, and detection precision comes from physical
  constraints, not from a permissive all-pairs energy.

## Is it the smartest it can be? — self-tuning audit (gap23)

The original module was entirely **hand-tuned** (every basin and weight a human
guess). Added a self-tuning/calibration layer (`coherence_calibrate.py`:
`fit_basins`, `fit_weights`, `class_priors`, `reliability`/ECE) so the engine can
LEARN its parameters from labelled data — the gap23 upgrade. Validated honestly on
real orbits with a 50/50 train/test split (`scripts/calibrate_mover.py`):

| approach | overall | balanced (rare-class) |
|---|---|---|
| hand-tuned (current default) | **97.9%** | 88.0% |
| fully-learned (generative MLE basins) | 89–92% | 82–89% |
| hybrid (hand basins + self-tuned weights) | 97.1% | **91.9%** |

Findings (honest):
1. **Naive full learning LOSES** to hand-tuning — generative MLE fits each class's
   marginal, not the decision boundary (outer-belt collapses 90%→48%). So the
   hand-tuned basins are already near the discriminative optimum; "smartest" is
   NOT "auto-fit the marginals."
2. **Hybrid weight-tuning** lifts balanced/rare-class accuracy +4pp (Mars-crosser
   +16, Centaur +15) but **down-weights the perihelion axis to 0.24** — robust on
   the (a,e)-labelled test, fragile for the pipeline's real snapshot-distance
   input. Kept as a validation artifact, NOT promoted to default.
3. **Decision: keep the principled hand-tuned defaults; ship self-tuning as a
   validated capability** the module can apply to any labelled set, plus a
   reliability/ECE diagnostic. The module is now self-calibrating, and we verified
   its defaults are already robust — both are real improvements over "trust me."

## Closing the data-access gaps (2026-06-02)

- **GAP #1 CLOSED — independent mover labels.** Fetched JPL Small-Body Database
  `orbit_class` (JPL's own criteria) for a balanced 23,561-asteroid per-class sample,
  cached `data/jpl_orbit_class.json` (`fetch_and_validate_jpl.py`). `classify_mover`
  (hand-tuned default) agrees with JPL **88.8%** (NEO 93.7%, Mars-crosser 99.8%,
  outer-belt 100%, TNO 98.6%, main-belt 76.1%, Centaur 67.9%) — reproducing the
  AUTHORITATIVE taxonomy, not its own (a,e) labels. The boundary classes (main-belt
  edges, Centaur/TNO) are where my basins legitimately differ from JPL.
  **Bonus finding:** the discriminatively-refined basins do WORSE on JPL (84.2%,
  main-belt 57%) — they OVERFIT my (a,e) labels and do not generalize, which
  INDEPENDENTLY vindicates shipping refinement opt-in and keeping hand-tuned default.
- **GAP #2 / #3 CLOSED — real linker FPs + fresh-survey e2e.** End-to-end run on the
  2019 `deep_field` (never processed before; 2,015,218 real detections over 4 nights,
  a crowded r-band field). Fixed two real bugs to get here: a legacy extraction cache
  missing `flux`, and a crowded-field ordering bug (brightness-cap BEFORE the
  stationary veto cut the faint asteroids → reordered to veto-stars-first then cap).
  Results: stars vetoed 1.88M → ~131k movers; **120 recoverable knowns, 8 recovered**
  end-to-end (7% — crowded + faint single-filter, but real); **18 unknown multi-night
  chains = REAL linker false-positives** (not modeled); coherence vetting cuts them
  **18 → 7** (61% FP suppression on REAL linker output); scrambled chance floor 14
  (7 vetted < 14 → no new discovery, honest). **Hard-rules vetting also gives 18 → 7**
  on this field — the two TIE here: only 18 candidates, none borderline. The coherence
  vetting advantage (+6.6pp F1, graceful-vs-collapse under noise) is established on the
  large-sample A/B, not distinguishable on an 18-candidate real field. Logs:
  `_deepfield_stress.log`, `_deepfield_rules.log`.

## Hardening toward NASA grade (2026-06-02) — rigor, calibration, robustness

- **Repeated trials with confidence intervals** (`validate_repeated.py`): every
  selector run over many independent real-data draws.
  - mover EoO overall **97.91% ± 0.07** (95% CI ±0.04), NEO recall 95.6% ± 0.74,
    beats baseline EVERY draw (12 draws);
  - vetting EoO best-F1 **0.95 ± 0.00**, beats hard-rules EVERY draw (12 draws);
  - refinement 5-fold CV gain **+5.36pp ± 1.67**, positive EVERY fold.
  Stable, not a lucky sample.
- **Calibrated confidence** (`coherence_calibrate.fit_temperature`,
  `calibrate_confidence.py`): temperature fit on real orbits cut the mover ECE
  **0.41 → 0.09 (−78%)** on held-out data — when it says 90% it is ~right. Applied
  to `classify_mover` by default (temperature only scales confidence, never changes
  the predicted class, so it is a pure safe win).
- **Graceful degradation under noise** (`validate_robustness.py`): the EoO vetting
  selector slides smoothly under astrometric jitter (**F1 0.957 → 0.922** across
  0.1″→3.0″) while the hard AND-rules COLLAPSE (0.905 → 0.423). The mover slides
  smoothly vs distance error (no cliff). The coherence selector's smoothness is
  exactly why it beats thresholds in the field.
- **Expanded samples**: variable typing rerun on **260 real VSX+ZTF stars** (was 38)
  → **89% type / 92% period** (EA 100%, EW 95%, RRab 85%, DCEP 68%). The 89% holds
  on a 7× larger sample, tightening the CI from ~±10% to ~±4%. DCEP (Cepheids)
  remain the hard case (reddening confuses them with eclipsing binaries).

## ONE engine — unification + discriminative refinement (2026-06-02)

Collapsed the multiple coherence scorers into ONE engine and added the refinement
that legitimately beats hand-tuning:

- **Unified track scoring.** `coherence_vet.track_energy` was rebuilt to extract a
  curvature-tolerant feature vector (`track_features`: rate spread + heading,
  one-sided coverage deficits, down-weighted residual) and score it with the SAME
  `coherence_field.incoherence_energy` primitive used for classification — no more
  separate 4-sector formula. `chain_quality.chain_coherence_score` now delegates to
  it (and `filter_chains` reports/gates on it). The pipeline has ONE coherence
  implementation.
- **It got BETTER, not just unified:**
  - vetting on real orbits: F1 **0.956 vs hard-rules 0.890 (+6.6pp)** — up from the
    pre-unification +2.9pp.
  - chain real-vs-chance: unified **AUC 0.993 vs old purity 0.973** (length-different)
    and **0.998 vs 0.50** on a same-length motion-only test (it reads motion
    coherence directly from positions; purity needs precomputed rates).
- **Discriminative basin refinement** (`coherence_calibrate.refine_basins`): tunes
  basin WIDTHS/centers for held-out balanced accuracy (boundaries, not marginals),
  keeping the robust weights. On the mover (held-out real orbits) it lifts
  Mars-crosser +20pp, outer-belt +11pp, NEO to ~99% — the legitimate way learning
  beats hand-tuning. Shipped OPT-IN (`COH_MOVER_CALIBRATED=1`): it is tuned on the
  with-orbit (a,e,logq) regime, so the default stays hand-tuned to keep the
  snapshot-distance-only path robust (a regression the unit test caught).

## Pipeline coverage audit — where coherence belongs (and where it doesn't)

The goal is NOT "coherence everywhere" — it is coherence at every genuine
DECISION/FUSION point, and the right tool elsewhere. Audited every decision
module in `discovery/imaging`:

**Applied + real-data validated (selectors that WIN):**
- `coherence_classifier.classify_mover` — orbit class (real +6.7pp)
- `coherence_classifier.classify_variable` — variable typing (real 89%)
- `coherence_vet.track_energy` — candidate vetting (real +2.9pp F1)
- `coherence_field` engine + `coherence_calibrate` self-tuning

**Evaluated for coherence, but the purpose-built code WINS — correctly left as-is:**
- `chain_quality.chain_purity_score` — its curvature-tolerant geometric-mean
  beats the unified linear track-energy at real-vs-chance (AUC 0.98 vs 0.86);
  track_energy assumes linear motion, real arcs curve. Unified alternative
  exposed as `chain_coherence_score` but NOT made default
  (`validate_chain_quality_coherence.py`).

**Already Bayesian / likelihood — they EMBODY energy-minimization; a coherence
relabel would add nothing (and `snapshot_posterior` already cites the same
reparametrize-the-degeneracy lesson from the Coherence cosmology MCMC):**
- `snapshot_posterior`, `statistical_ranging` (orbit ranging posteriors)
- `bayesian_linker` (hierarchical orbit-prior linking)
- `pixel_likelihood` (P(pixels|orbit) IOD refinement)

**Coherence does NOT belong (search / physics / signal — proven by the linker
negative control):** cross-night linking, Gauss/HelioLinC IOD, n-body ephemeris,
source extraction, difference imaging, deblending, morphology, shift-and-stack.

**Applicable selector, not yet done (needs external real data to validate):**
- `color.classify_color` — taxonomy from colors uses hard thresholds; could be
  coherence basins in (g−r, r−i, i−z, a*) space, but honest validation needs the
  SDSS Moving Object Catalog (no local copy). Flagged, not bolted on unvalidated.

**Answer to "is it applied to every subsystem?": No — by design.** Coherence is
applied and validated at every decision/fusion layer where it beats the baseline;
it is deliberately NOT applied where purpose-built code wins (chain_quality), where
proper Bayesian methods already minimize a principled objective (IOD/ranging), or
where the problem is search/physics (linking/ephemeris). Forcing it everywhere
would regress the pipeline — empirically shown twice (mover MLE basins, chain
purity).

## Bottom line
On real data, the coherence/EoO framework **wins at every decision/selection
layer it was applied to** (orbit class +6.7pp, vetting +2.9pp, variable typing
fusion to 89%) and **loses as a search primitive** (kept as the control). The
rule for future work: generator (cheap, physical) + EoO selector (few candidates).
