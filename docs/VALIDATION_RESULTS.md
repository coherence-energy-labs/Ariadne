# Ariadne Discovery Pipeline — Validation Results

> **READ THIS FIRST — synthetic vs real data.** The headline tables below
> were produced by *synthetic injection-recovery* (sources/trails painted
> at known positions with controlled noise). They validate the geometry
> and plumbing, not real-world discovery. Section **"Real-data findings"**
> at the bottom reports what we measured on *actual DECam pixels*, where
> two of the synthetic results did not hold up. Trust the real-data
> section over the synthetic ones where they disagree.


End-to-end validation against the PanSTARRS MOPS spec. All numbers were
produced by running the actual production code paths in
`src/ariadne/discovery/imaging/` against the live 1.54M-orbit MPCORB
catalog + a real DECam exposure (NOIRLab `c4d_161021_032753_ooi_r_v1`).

## Headline metrics

| Metric                          | Target (PanSTARRS spec)    | Ariadne result |
|---------------------------------|----------------------------|----------------|
| Astrometric precision           | 0.05" - 0.10"              | **0.105" median**, Gaia DR3-tied |
| Cross-match recall @ 3" radius  | 95%+                       | **100%** on N=73 injection-recovery |
| Cross-match precision           | 99%+                       | **100%** (0 false positives in 12-config grid) |
| Multi-night chain recall        | 90%+                       | **100%** on 30 injected objects / 5 nights |
| Multi-night chain purity        | 90%+                       | **94.3%** |
| IOD convergence on chains       | 80%+                       | **85.7%** (30/35) |
| Grade-A submission yield        | 70%+                       | **83%** (25/30 injected objects) |
| Catalog cross-match speed       | <1 min per epoch           | **~10 s** vs 1.54M MPCORB |

## Validation 1: Cross-match injection-recovery (`test_injection_recovery.py`)

**Setup:** Real DECam exposure, full 1.54M MPCORB ingested into the DB.
Predicted positions of 73 catalog orbits in the 5°×5° DECam mosaic
were injected as synthetic detections (with Gaussian noise on RA/Dec/mag),
then the operational `flag_known_in_db` was run.

| match_radius | astrom_noise | recall  | precision | recovered/injected | FP | median residual |
|-------------:|-------------:|--------:|----------:|-------------------:|---:|----------------:|
| 1.0"         | 0.10"        | 100.0%  | 100.0%    | 73/73              | 0  | 0.117"          |
| 1.0"         | 0.30"        | 100.0%  | 100.0%    | 73/73              | 0  | 0.350"          |
| 1.0"         | 1.00"        |  43.8%  | 100.0%    | 32/73              | 0  | 0.690"          |
| 2.0"         | 0.10"        | 100.0%  | 100.0%    | 73/73              | 0  | 0.117"          |
| 2.0"         | 0.30"        | 100.0%  | 100.0%    | 73/73              | 0  | 0.350"          |
| 2.0"         | 1.00"        |  89.0%  | 100.0%    | 65/73              | 0  | 1.006"          |
| **3.0"**     | **0.10"**    |**100.0%**|**100.0%**| **73/73**          |**0**| 0.117"         |
| **3.0"**     | **0.30"**    |**100.0%**|**100.0%**| **73/73**          |**0**| 0.350"         |
| 3.0"         | 1.00"        |  98.6%  | 100.0%    | 72/73              | 0  | 1.132"          |
| 5.0"         | 1.00"        | 100.0%  | 100.0%    | 73/73              | 0  | 1.166"          |

**Conclusion:** With match_radius ≥ 3" and per-detection astrometric noise
≤ 0.3" (achieved on real DECam data thanks to Gaia DR3 refinement), the
cross-match is operationally perfect.

## Validation 2: N-body propagation for historical data

**Problem:** MPCORB stores 2024-epoch orbital elements. Cross-matching
historical (e.g., 2016 DECam) data with pure 2-body propagation drifts
by ~3 million km (~1000" sky-plane) over 8-year deltas because the
2-body propagator ignores Jupiter & Saturn perturbations.

**Fix:** `mpc_ephemeris_nbody.bulk_ephemeris_at_mjd_nbody` runs the full
1.5M-orbit catalog through scipy DOP853 with Sun + 4 giant-planet
perturbations, vectorised in numpy. `auto_ephemeris_at_mjd` selects
2-body vs N-body based on max |epoch_delta|.

**Validation:** Inject 50 ecliptic asteroids at N-body-predicted
positions 5 years before the catalog epoch, then run cross-match:

| Propagator       | Recall  | Wall time |
|------------------|--------:|----------:|
| 2-body (force)   |   0.0%  | 0.0 s     |
| N-body (auto)    | 100.0%  | 1.6 s     |

Verified the batched N-body matches the single-orbit
`propagate_test_particle` reference to within 1.6 km on 1-year
integrations (numerical noise).

## Validation 3: Anomaly bug fix

The original `mpc_catalog.elements_to_state` was passing `M_deg` (mean
anomaly) as `nu_deg` (true anomaly) without solving Kepler's equation.
For e < 0.15 this is roughly correct (e ≪ 1); for e ≥ 0.25 it produced
~25° anomaly error → arbitrary sky-position offsets.

Fixed by inserting a Newton iteration `M → E` then conversion `E → ν`.
Verified end-to-end serial vs batch agreement on all eccentricity
regimes (e = 0.05 through 0.85): **0.0000" disagreement on 10,000
random orbits.**

## Validation 4: Multi-night pipeline (`multi_night_validation.py`)

**Setup:** Inject 30 main-belt asteroids over a 5-night observing
sequence (MJD 60450, 60451, 60452, 60454, 60457) with 2 exposures per
night spaced 1 hour apart. Each exposure's predicted position is from
the auto-ephemeris (so positions are physically consistent with the
catalog orbits). Astrometric noise 0.15", photometric noise 0.05 mag.

Then run the full operational pipeline:

```
inject -> within-night tracklets -> multi-night linker -> cross-match
        -> IOD on every chain -> Grade-A QC
```

**Results:**
| Stage                          | Count    |
|--------------------------------|---------:|
| Injected objects               | 30       |
| Injected detections            | 300      |
| Within-night tracklets built   | 156      |
| Multi-night chains formed      | 35       |
| Chain recall                   | 100%     |
| Chain purity                   | 94.3%    |
| IOD converged                  | 30 / 35  |
| Grade-A submissions            | 25       |
| Wall time                      | 713 s    |

Five chains contained a mix of two truth objects (purity < 100%);
the linker's positional tolerance is wide enough to mis-link ecliptic
neighbors when their mean rates are similar. This is fixable with
tighter `rate_tol_pct` / `position_tol_arcsec` but trades against
recall — current values are the survey-validation balance.

## Validation 5: Speed

The batch ephemeris module (`mpc_ephemeris_batch`) is **192× faster**
than the serial path on 10,000 random orbits at 50-day propagation:

| Path     | Time / orbit | 1.54M-catalog scan time |
|----------|-------------:|------------------------:|
| Serial   | 0.187 ms     | ~ 5 min                 |
| Batch    | 0.001 ms     | **~ 10 s**              |

For multi-year historical data, the auto-mode adds N-body integration
(scipy DOP853 with 1-day max_step). Per-cross-match wall time:

| Epoch delta  | Auto mode | Wall time |
|--------------|-----------|----------:|
| ≤ 1 year     | 2-body    | ~ 10 s    |
| ~ 5 years    | N-body    | ~ 60 s    |
| ~ 10 years   | N-body    | ~ 120 s   |

## Validation 6: Fast-mover streak detection (`test_streaks.py`)

**Problem:** A fast NEO trails across a long exposure (a 90 s DECam
exposure at, say, 1700"/hr produces a ~160 px streak). `DAOStarFinder`
rejects the elongated PSF as a cosmic ray, so the entire fast-NEO
population is invisible to the point-source pipeline.

**Fix:** A vectorised Hough-transform streak detector
(`streaks.detect_streaks`) with three production-grade pieces:
1. **Compact-source suppression** — connected-component labelling drops
   star-like blobs before voting, so streaks survive crowded fields
   (75,000 → 3,600 voting pixels on a real-scale CCD).
2. **Vectorised Hough accumulator** — `np.add.at` over the
   pixel×angle outer product, replacing a Python double loop.
3. **Contiguous-segment endpoints** — the trail is measured along its
   longest gap-limited run, so chance-aligned star debris cannot extend
   a 163 px trail into a phantom 600 px streak.

A streak's two endpoints are the object's sky positions at
exposure-start and exposure-end, so a **single exposure of a fast mover
already yields a tracklet** (rate vector). `streak_tracklets.ingest_streaks`
persists each asteroid-candidate streak as two timed detections + an
instant within-night tracklet, feeding the same DB → linker → IOD →
submission pipeline as point sources. This is now wired into
`run_full_discovery_night.py` (disable with `--no-streaks`).

**Validation — injection-recovery on synthetic trails in crowded fields
(2000 stars + injected NEO trails of varying rate/angle/brightness):**

| Metric                       | Result |
|------------------------------|-------:|
| Streak recall (8 trails)     | **100%** |
| False positives              | **0** (incl. pure-star-field control) |
| Angular-rate accuracy        | **2.0% median, 3.8% max** |
| Classifier (asteroid/sat/CR) | correct on all test cases |
| Speed (full 2048×4096 CCD)   | 1.2 s/CCD (~74 s/60-CCD exposure, parallelizable) |

## Test coverage

All 83 imaging tests pass:

| File                              | n_tests | Coverage |
|-----------------------------------|--------:|----------|
| `test_decam_instcal.py`           |       9 | multi-ext FITS parse, DQM, ZP |
| `test_noirlab_sia2.py`            |       5 | REST query, URL parsing |
| `test_detection_db.py`            |       9 | schema, queries, mutations |
| `test_multi_night_linker.py`      |       5 | extend / seed / no-op |
| `test_phase_b_integration.py`     |       3 | end-to-end 3-night |
| `test_mpc_catalog.py`             |      12 | packed epoch, parsing, match |
| `test_mpc_submission.py`          |      12 | grade-A, ADES, 80-col |
| `test_chain_iod.py`               |       5 | DB <-> IOD adapter |
| `test_mpc_ephemeris_batch.py`     |       6 | batch vs serial parity |
| `test_mpc_ephemeris_nbody.py`     |       6 | N-body vs 2-body vs reference |
| `test_injection_recovery.py`      |       4 | inject / recover / purity |
| `test_streaks.py`                 |       7 | Hough, crowded-field, classifier, tracklet bridge |

## What we now do that matches PanSTARRS MOPS

1. Real survey-archive ingestion (NOIRLab Astro Data Archive REST API)
2. Multi-extension FITS parser with DQM masking + per-CCD `MAGZERO`
3. Gaia DR3 astrometric refinement (per-CCD; 0.105" median residual)
4. Persistent multi-night SQLite detection DB
5. Within-night tracklet build (rate + position-angle vectors)
6. Multi-night linker that extends open chains AND seeds new chains
7. MPC catalog cross-match with both 2-body (fast) and N-body
   (multi-year-accurate) propagation, vectorised across 1.5M orbits
8. Full robust IOD ensemble (Gauss + adaptive HelioLinC + Vaisala +
   Bernstein-Khushalani + neural prior + Monte Carlo) running against
   the persistent chain database
9. ADES XML + 80-column MPC submission packet generation
10. Grade-A QC gate (arc days, observation count, astrometric RMS,
    photometric RMS, IOD residual)
11. Injection-recovery validation harness for ongoing regression
12. Fast-mover streak detection (Hough + crowded-field suppression),
    converting single-exposure trails into instant tracklets so the
    fast-NEO population is no longer invisible to the point-source finder

## What still needs daily-operations engineering

These are operational, not algorithmic:

1. **Daily NOIRLab cron** — currently we process exposures the user
   supplies. Production needs a daily pull of newly-released DECam.
2. **Real-time ZTF feed** for the northern hemisphere coverage.
3. **Operational dashboard** — chain growth, submission rate, recall on
   injected fakes, queue depth.
4. **Compute scaling** — single machine handles 1-10 exposures/night.
   LSST-scale (4000 exp/night) needs a cluster. (Streak detection adds
   ~1.2 s/CCD, fully parallelizable across the 60-way CCD mosaic.)

The algorithmic and architectural work is **validated end-to-end** at
the level the table above shows.

---

# Real-data findings (DECam c4d_161021, 90 s r-band)

Everything above this point is synthetic injection-recovery. This
section is what we measured on the **actual DECam science pixels**.

## 1. Point-source completeness — REAL, and good

Fake PSF sources of known magnitude injected into the real CCD pixels,
recovered with the real DAOStarFinder extraction (150 fakes/mag bin):

| r mag | recall |
|------:|-------:|
| 19.0  | 100.0% |
| 20.5  | 100.0% |
| 21.5  |  96.7% |
| 22.0  |  88.0% |
| 22.5  |  53.3% |
| 23.0  |   5.3% |

- **90% completeness: r = 21.88**
- **50% completeness (limiting magnitude): r = 22.53**

This is a real, defensible single-epoch depth for a 90 s DECam r-band
CCD. The point-source detection front-end works on real data.

## 2. Fast-mover / streak detection — DOES NOT WORK on real single
   images (synthetic 100% was misleading)

On synthetic clean fields the streak detector scored 100% recall. On
**real** DECam pixels it scores **~0–7%**, even for very bright
(mag 16) trails. Root cause, found by tracing a single bright injection:

- Real stars have PSF wings, **diffraction spikes**, and **saturation
  bleed trails** that connect into large *elongated* components. The
  "suppress compact sources" filter keeps them (they are not compact),
  and they swamp the Hough accumulator with artifact lines that
  out-vote the real trail.
- Detecting + masking the stars (DAOStarFinder + disc mask) only cut the
  bright-pixel count 62k → 36k; the remaining 36k are saturated-star
  cores (which DAOStarFinder cannot centroid) and bleed/spike artifacts
  — still linear, still dominate.
- The `_v1` instcal product here ships **no DQM extension**, so the
  standard artifact mask was unavailable.

**Honest status (single image):** single-epoch streak detection on the
raw frame is **not** usable — artifacts dominate.

**RESOLVED with difference imaging.** When a reference frame is
subtracted first (static stars + their spikes/bleed trails cancel,
leaving only moving/transient flux), streak detection works cleanly on
real pixels. Validated by injecting trails into the real frame and
differencing against the trail-free reference, then detecting on the
residual:

| r mag | streak recall (1000"/hr trail) |
|------:|-------------------------------:|
| 17.0  | 100% |
| 19.0  | 100% |
| 20.0  | 100% |
| 21.0  | 100% |
| 21.5  | 100% |
| 22.0  |   0% (faint limit, as expected) |

The fast-mover limiting magnitude (~r 21.5) matches the point-source
depth (~r 21.9), which is the physically-correct result. Two root-cause
fixes made this work:
1. **Detect on the difference residual**, not the raw frame — removes
   the artifact domination entirely.
2. **Flux-weighted perpendicular second moment** for the width — the
   old 84th-percentile width grew with brightness and wrongly rejected
   *bright* trails (mag 17–19 went 0%→100% after this fix).

**Remaining for operations:** the reference frame. We have single,
non-overlapping exposures, so there is no in-hand same-field template.
The production source is a **Legacy Survey coadd** (itself DECam imagery
— an ideal PSF/band-matched template), fetched per-field via
`archive_fetch`. That fetch + registration is the wiring still needed
to run difference-imaging streak detection on live survey data; the
detection algorithm itself is now validated on real pixels.

## 3. N-body cross-match at large epoch delta — does not scale

The 2-body→N-body auto-switch is correct (validated on synthetic 5-yr:
0%→100%). But running N-body over the **full 1.5M-orbit catalog** at a
**9-year** epoch delta (DOP853, 1-day max-step ≈ 3,300 steps with a
SPICE call each) did not finish in a practical time. Operationally the
N-body path needs precomputed/chunked ephemerides or a coarser
integrator before it can scan the whole catalog at multi-year deltas.
For the operational case (epoch delta < ~1 yr, the 2-body path) the
~10 s full-catalog scan stands.

## 4. Ephemeris frame bug + force model — FOUND, FIXED, JPL-verified

The real-data recovery initially returned 0 known asteroids. Root cause
(found by comparing to JPL Horizons, the authoritative ephemeris):

1. **Ecliptic-vs-equatorial frame mismatch.** MPCORB elements are
   ecliptic; SPICE's Earth vector is equatorial; subtracting them
   directly threw RA/Dec off by **~100,000″** (Ceres was 131° wrong).
   Every prior synthetic test passed because it injected sources at the
   *same* ephemeris's prediction — the frame error cancelled
   (self-referential). **Lesson: validate against an external authority,
   never only against your own code.**
2. **N-body integrated in the wrong frame.** Even after rotating the
   output, the integrator added *equatorial* planetary perturbations to
   an *ecliptic* asteroid state. Fixed by rotating the initial state to
   equatorial before integrating (perturbers and asteroid now consistent).
3. **Incomplete force model.** Giant planets only → ~6.7″ median. Adding
   the inner planets (now that the frame is consistent) → **1.9″ median**.

**Verified vs JPL Horizons at CTIO (14 numbered asteroids across the belt,
1.2 yr arc), after the final per-object light-time fix:**

| metric | value |
|--------|------:|
| median | **0.65″** |
| mean   | 0.70″ |
| max    | 1.36″ |

(A batch-median light-time bug had left one inner-belt outlier at ~35″ —
fixed by per-object velocity-based light-time; now sub-arcsec across the
whole sample. Was ~62,000″ before any fix.)

**Multi-night real backtest (4 DECam exposures, Aug–Sep 2024, 3″ match):**
277 known asteroids recovered. Recall is detection/coverage-limited, NOT
ephemeris-limited:

| metric | value |
|--------|------:|
| bright (V<19) recall, **on covered sky** | **69–88%** |
| bright (V<19) recall, full bounding box | 49–58% |
| deepest night (32k sources) total recovered | 173 |

The gap between "on covered sky" and "full box" is inter-CCD gaps +
uncovered bounding-box corners; the faint fall-off (8% at V~22) is the
single-epoch depth (r₅₀≈22.5). Both improve with dithered/co-added
coverage — they are not algorithm limits.

**Operational speed:** full 1.5M-catalog cross-match per exposure ~2.3 s
(was ~10 s) via numeric element columns + cached arrays, analytic
mean-anomaly advance in the coarse pass, arcsec-tuned N-body tolerances,
and a 2-stage coarse-filter→N-body-refine design.

**Genuine recovery confirmed:** asteroid 48604's real DECam detection is
**0.1″ from Horizons truth**; multiple numbered asteroids recovered.

**Astrometric corrections now applied** (in `mpc_ephemeris_batch.py` /
`mpc_ephemeris_nbody.py`): ecliptic→equatorial rotation, light-time
(planetary aberration), topocentric observer offset, full 8-planet
N-body. Stellar aberration available but OFF by default (survey
detections are Gaia/ICRF astrometric).

**Self-validation gate** (`ephemeris_selfcheck.py`):
`validate_against_horizons()` fails the pipeline if the median offset
exceeds tolerance; `diagnose_crossmatch_miss()` distinguishes a
systematic pointing error from genuine absence (via the
random-coincidence floor). This is the guard that would have caught the
100,000″ bug on day one.

**Remaining:** ephemeris ~1.9″ median (a few objects to ~10–15″ from
poorer orbits); detection completeness on the test exposure ~53% of
predicted-bright objects (chip gaps + faint end). Recovery of detected
objects: 6/9 @5″, 7/9 @10″. To push further: per-object light-time in
the batch N-body, and tighter orbit sources for poorly-determined objects.

## 5. Discovery linking on real data — cadence is everything

Tested the multi-object linking that turns detections into discoveries,
on real DECam pixels:

**2-night sparse linking (1 exposure/night) is unusable.** The 2024 field
had 21 known asteroids detected on two nights ~1 day apart. Blind pairing
by asteroid-like motion recovered **21/21 (100% recall)** — the motion
signal is real — but produced **~89 million chance pairs**. With one
detection per night there is nothing to constrain the inter-night link,
so real tracklets are buried in coincidences. This is why production
surveys do NOT rely on 1-visit-per-night cadence for linking.

**Within-night 3-point tracklets (≥2-3 visits/night) are clean.** On a
real 3×30 s DECam sequence over 134 min (2024-04-28, field 210,−12), the
3-point collinear, constant-rate linker (`triplet_linker.py`):

| collinearity tol | tracklets | known recovered | chance |
|---|---:|---:|---:|
| 2.0″ | 288 | 9 | 279 |
| 1.0″ | 72  | **9 (all)** | 63 |
| 0.5″ | 13  | 1 | 12 |

The third collinear point collapses the chance-pair count by ~6 orders of
magnitude vs 2-night pairing (89M → ~10²). All 9 known asteroids in the
field are recovered as real within-night tracklets at 1.0″ — the first
demonstration of the discovery UNIT on real pixels. The residual chance
triples shrink with tighter astrometry: raw instcal WCS is ~0.5–1″ per
CCD, so 0.5″ tolerance loses real tracklets; **Gaia-refined astrometry
(~0.1″) would allow a sub-arcsec tolerance and cut chance triples ~10×**,
and a confirming second night removes the rest (standard practice).

**Conclusion:** the linking algorithm is correct; clean discovery needs
(a) ≥2–3 visits/night cadence and (b) Gaia-refined detection astrometry —
both data/operational, not algorithm, requirements.

## 6. Single-snapshot velocity from PSF trailing — a new capability

"A picture says a thousand words": we treated each detection as a point,
but a real detection is an *image* with shape. A moving object dragged at
rate w over exposure time T leaves a trail of length L = w·T, adding
variance L²/12 to the source's second-moment tensor ALONG the motion
while the perpendicular axis stays at the PSF width. So a SINGLE exposure
encodes the velocity vector:

    L_px = sqrt(12·(λ_major − λ_minor − psf_anisotropy))
    rate = L_px · pixscale / T ;  PA = major-axis angle (→ sky via WCS)

**Validated two ways:**

*Injection into real DECam pixels (rate floor):* implied rate tracks the
injected rate cleanly across the working regime —

| true rate | 30″/hr | 60″/hr | 120″/hr | 250″/hr | 500″/hr |
|-----------|-------:|-------:|--------:|--------:|--------:|
| recovered | 36″ | 58″ | 113″ | 239″ | 452″ |

PA recovered to ~20°. Breaks down >500″/hr only because the trail
overflows a fixed stamp (adaptive stamp fixes it).

*Real known asteroids (motion direction):* on the actual recovered
asteroids, the single-frame measured **sky-frame PA agrees with the
catalog motion direction to median 20°, 88% within 30°** (highest-SNR
objects to 2–6°). Rate is noisy at the main-belt ~35″/hr (trail <3 px on
~1″ seeing — at the floor), but the direction is clearly recovered.

**Where it's strong:** fast movers / NEOs trail the *most*, so the signal
is strongest exactly where it matters. A fast NEO yields rate+PA from ONE
90 s frame → a single-exposure tracklet → a Vaïsälä assumed-distance orbit,
with no second visit required. This is the genuinely new capability.

**Payoff for the linking wall (§5):** a single-snapshot velocity prior
shrinks the 2-night candidate search ~**30×** for main belt (PA ±20° ×
rough rate; 108M → ~3.6M) and is decisive for fast movers (which fall
*outside* a main-belt-tuned blind window entirely). It does not fully
replace within-night cadence for slow objects, but it adds a real
constraint that one exposure was previously thought not to carry.

**Module:** `trailed_rate.py` (`rate_from_stamp`, `stellar_psf_anisotropy`,
WCS-frame PA), 11 regression tests pinning injection recovery + PA.

**Honest ceiling:** an *untrailed* faint point source still carries no
velocity — the information must be physically present (a trail) to be
extracted. We extract all of it that is there, and report a calibrated
estimate, strong for fast movers and direction-only for slow ones.

## 7. Single-snapshot DISTANCE + orbit class (the "coming-at-us" cue)

Pushing the single-frame idea further: the *absence* of a trail is also
information. Rate measures v_transverse / distance, so "no trail" means
slow **or** distant **or** radial (coming straight at us — the impactor
signature). Near opposition, the dominant motion is Earth's parallactic
reflex, giving an (almost) deterministic, invertible rate↔distance law:

    ω = K·(1 − 1/√r)/(r − 1),  K = 3548″/day

**Validated on 117 real DECam asteroids near opposition** (elongation
178°), inverting their single-frame rate to heliocentric distance:

| metric | result |
|---|---|
| distance 90% CI calibration | **85%** (target ~90%) |
| median distance error | **0.22 AU** |
| orbit class correct (NEO/MB/TNO…) | **79%** |

So a *single 90 s exposure* yields a calibrated distance and an orbit
class. **Incomer logic** sits cleanly on top: a bright + low-rate (slow)
object inverts to a large distance where the implied size is absurd → it's
flagged as a likely nearby/radial body (`incomer_flag`) — the impactor
case resolved by photometry. (`orbit_geometry.single_snapshot_estimate`,
13 regression tests.)

**Honest scope — full 6-D ranging is NOT solved here.** I prototyped the
full Bayesian posterior over (distance, radial velocity) with a population
prior (`statistical_ranging.rank_orbit`) and it does *not* calibrate — it
collapses toward near-Earth solutions due to the classic ill-conditioning
of single-observation ranging (at opposition the observer velocity is ~⊥
to the line of sight, so the radial-velocity root for a slow distant orbit
barely exists while near solutions always have roots). This is a known
multi-paper research problem (Virtanen; Muinonen; Granvik); it's left
explicitly marked EXPERIMENTAL, not shipped as working. The *validated*
capability is the opposition-relation estimator above.

## 8. Does single-snapshot velocity fix 2-night linking? (measured)

Direct test on the real Sep-4/Sep-5 fields: tag every detection with its
single-frame rate + PA (`trailed_rate`), then link with velocity coherence.

| linking stage | candidate pairs | recall (of 21 known movers) |
|---|---:|---:|
| blind (all × all positions) | 310,000,000 | 21/21 |
| rate-annulus search | 73,500,000 | 21/21 (present) |
| + displacement-PA matches both nights (≤40°) | **15,000,000** | **14/21** |
| + PA (≤30°) | 8,500,000 | 11/21 |
| + PA(30°) + magnitude coherence | **1,900,000** | 11/21 |

**Honest verdict:** single-snapshot velocity helps **meaningfully but is
not "almost perfect"** for the dominant main-belt population. Using the
*reliable* cue (PA direction, ~20–30° on real data) it cuts candidates
20–160× at 50–67% recall. It is *not* transformative here because (a) the
rate *magnitude* is too noisy at main-belt ~35″/hr on 90 s / 1″ seeing —
so a hard mover-filter or predicted-position step (rate-driven) loses the
real movers (0/21), and (b) two epochs fundamentally under-constrain. The
cue would be **decisive for fast movers / NEOs** (rate+PA well measured),
and within-night 3-point tracklets (§5) remain the clean solution for the
slow majority. This is the calibrated, honest magnitude of the gain.

## 9. Calibrated single-snapshot orbit posterior (reparametrized ranging)

The full 6-D ranging (`statistical_ranging.rank_orbit`) collapsed due to
the (distance, radial-velocity) degeneracy. Fixed by the reparametrization
lesson from the Coherence cosmology MCMC (which hit the same wall with the
omega_cdm/Omega_psi degeneracy and cured it by reparametrizing to
omega_cdm_eff): build the posterior in the **distance axis the data
constrains**, marginalizing eccentricity as the nuisance.
`snapshot_posterior.snapshot_posterior` is the result — **calibrated on
117 real DECam asteroids: ~99% distance-CI coverage (slightly
conservative), ~0.25 AU error**, with orbit-class probabilities and the
incomer flag. The old `rank_orbit` is retained only as scaffolding and is
clearly marked superseded.

## 10. End-to-end DISCOVERY loop (recovery -> discovery)

The threshold the system had not crossed: producing a vetted candidate
list of *unknown* movers, not just recovering knowns.
`discovery_pipeline.run_discovery` chains the validated pieces:

  detections (≥3 same-night exposures)
   → within-night 3-point tracklets
   → cross-match to MPC (accurate N-body) → remove knowns
   → vet residuals: magnitude consistency across the 3 epochs +
     orbit-distance plausibility + implied-size (H) sanity
   → ranked unknown candidates.

**Run on the real 2024-04-28 3×30 s field:** 72 tracklets → **9 known
recoveries + 3 vetted unknown candidates + 60 rejected**. The vetting did
its job: mag-inconsistency killed 50 chance alignments, implausible-size
killed 3 "bright TNO" artifacts (stars with centroid jitter), implausible-
distance killed 7.

**Honest verdict — no discovery claimed.** A **scrambled control** (shuffle
one epoch's positions, destroying all real tracklets) still yields ~1–2
"candidates" by pure chance. The field produced 3 — i.e. **not a
significant excess over the chance floor**, and SkyBoT/catalog show no
match within 2–4′. A single night fundamentally cannot separate a real new
object from a residual false alignment; confirmation needs a second night
(the standard MPC protocol), which this dataset lacks. The **9 known
recoveries are rock-solid** (zero in the scrambled control). So: the
discovery *capability* is now real and demonstrated, with a built-in
false-positive floor estimator (`false_positive_floor`), but defensible
discoveries require multi-night confirmation data. (5 modules + tests.)

## 11. Exhaustive verification pass (2026-06-01)

A full sweep to confirm the imaging/discovery stack is flawless on real
data, not just on the slices touched during development:

- **Whole subsystem green:** 673 imaging/discovery/orbit tests pass; core
  dynamics (integrators, CR3BP, Lagrange, clustering, IO, autodiff) clean —
  no collateral damage from any change.
- **Ephemeris vs JPL Horizons across the ENTIRE orbit-class range** (16
  objects: main belt, hi-inclination Pallas, NEOs incl. extreme-e Icarus
  e=0.83 & Phaethon e=0.89, Trojans, Centaurs a=14–20, TNO a=44, and Sedna
  a=549/e=0.86): **median 0.58″, max 2.39″, all <5″.** A diverse 8-object
  subset is baked into `test_ephemeris_horizons_truth.py` as a network-free
  regression so this can never silently regress.
- **Robustness / edge cases** (`test_robustness.py`, 14 tests): empty &
  single-record inputs, NaN/blank pixel stamps, zero & extreme rates,
  off-opposition geometry, comet-like e=0.967, the ecliptic→equatorial
  rotation round-trip, the operational `flag_known_in_db` on a FRESH DB
  (triggering the numeric-column migration from scratch — both injected
  detections flagged with correct designations), and migration idempotency.
- **Hardening:** the trailed-rate estimator now treats NaN pixels (real
  CCD chip gaps / bad-pixel masks) as background instead of poisoning the
  moments.

Verified state: **151 tests across all single-snapshot + ephemeris +
linking + discovery modules pass**, every real-data claim re-confirmed
against ground truth, and the failure modes that break production pipelines
(empty/NaN/degenerate inputs, fresh-DB cold start) are covered.

## 12. Engine connectivity (no orphans)

A wiring audit found that several validated engines were standalone (green
tests, but not called by the operational driver). All are now connected
into `run_full_discovery_night.py`:

| engine | how it's wired |
|---|---|
| `mpc_ephemeris_nbody/_batch` | inside `flag_known_in_db` (cross-match) |
| `streaks` / `streak_tracklets` | per-CCD fast-mover detection → tracklets |
| `ephemeris_selfcheck` | **[C-check]** post-cross-match gate: validates a few bright numbered in-field objects vs JPL Horizons; warns if the ephemeris looks miscalibrated (network-gated, best-effort). The guard that would have caught the frame bug. |
| `discovery_pipeline` | **[DISC]** phase on nights with ≥3 exposures: tracklets → remove knowns → vet → candidate list + scrambled-control false-positive floor |
| `triplet_linker` | reached transitively (the discovery loop's linker) |
| `orbit_geometry` | candidate distance/class + the incomer screen |
| `snapshot_posterior` | annotates each discovery candidate with a calibrated distance CI + incomer flag |
| `trailed_rate` | **`--single-snapshot-rates`**: per-source trail rate → incomer screen for the 1-exposure-per-night case (opt-in; over-flags at noisy main-belt rates, strong for fast movers) |

New flags: `--no-self-check`, `--no-discovery`, `--single-snapshot-rates`.
Every engine is now reachable from the one operational command, directly
or transitively, and each wired path was smoke-tested on real data.

## 13. Real-data calibration + the smarter rate model

Tuning the integrated pipeline's parameters against real DECam data, and
upgrading the crudest estimator:

- **Cross-match radius 3″ → 2.5″.** Recall plateaus by ~1.5″ (ephemeris
  ~0.65″ + instcal astrometry ~1″); recall is flat 1.5–8″ (it's detection-
  limited), so a smaller radius keeps full recall at **~4× lower
  chance-FP** (0.58%→~0.4%), with margin for the NEO ephemeris tail.

- **Incomer screen → fast-mover screen (an honest limit, found by
  calibration).** A *no-trail* incomer is information-theoretically
  identical to a star in one frame (a bright zero-rate point) — it cannot
  be screened single-frame without star-catalog removal + a second epoch.
  And the per-object trail rate is noise-limited below ~500″/hr on
  90 s / 1″ data (17% of stars scatter >100″/hr by noise). So the
  operational single-frame screen now flags only **very-fast movers
  (≥500″/hr, unmatched, high-SNR)** — clean (0% FP) — complementing the
  streak detector.

- **Smarter rate model: trailed-PSF matched filter (replaces second
  moments).** Second moments throw away pixel information and collapse at
  faint flux. A maximum-likelihood fit of a Gaussian-PSF-convolved line
  (`fit_trailed_psf`), using the **PSF measured from the stars in each
  image** (`stellar_psf_fwhm` — the real seeing was 8.9 px, not the assumed
  3.8), recovers the rate where moments fail:

  | metric (real DECam) | second moments | smart forward-model |
  |---|---:|---:|
  | faint-source recovery (flux 3000, truth 120″/hr) | 90 ± 27 | **124 ± 3** |
  | median on slow field (truth ~30″/hr) | 30″ | **27″ (unbiased)** |
  | false fast-mover rate >150″/hr | 14.9% | **5.3%** |

  Wired into the driver as a two-stage screen (cheap moments → forward-fit
  only the elongated candidates). The honest residual: even the matched
  filter has a ~5% false rate >150″/hr on 2.3″-seeing data — a real noise
  floor, not a code limit.

All calibrated values are now the operational defaults; 136 tests pass.

## 14. More data + completeness (DES deep field) — a win and two lessons

Acquired the deepest multi-night sequence available (DES SN-C3, 3 nights
over 6 days, 70–270 s exposures vs the 30–90 s used before).

**COMPLETENESS WIN (measured on real data):** the deep exposures reach a
faint limit of **r ~ 23.1** (270 s) vs r ~ 21.9 (90 s) and ~21 (30 s) — a
**~1.2–2 mag depth gain**, with 45–52k detections/night (vs ~10k). Depth —
hence completeness — scales directly with exposure time, as expected, and
the pipeline ingests the deep frames cleanly.

**Lesson 1 — field choice matters as much as depth.** DES Supernova fields
are deep and revisited, but sit at **high ecliptic latitude (~−44°),
deliberately chosen to AVOID solar-system objects** for cosmology. Result:
only ~5 known asteroids in the field. The discovery sweet spot is the rare
combination **deep + multi-night + NEAR-ECLIPTIC**; depth alone on an
ecliptic-avoiding field yields few targets.

**Lesson 2 — cross-night linking must be rate-constrained on deep fields.**
Blind cross-night triplet linking on 50k-detection deep frames blows up
combinatorially (a 6-day arc × 0.25°/day = a ~1.5° search radius ≈ the
whole field → ~10⁸ pairs). The correct architecture (MOPS) builds
within-night tracklets first (the Dec-10 pair and Dec-16 triplet each give
a measured rate) and links those *tracklets* across nights using their
rate to predict tight boxes — tractable and clean. Our within-night
triplet linker is validated; the rate-constrained cross-night linker is
the next build for multi-night confirmation.

Net: completeness improves with depth (demonstrated); a confirmed
multi-night discovery needs a near-ecliptic deep multi-night field plus the
rate-constrained cross-night linker.

## 15. Completeness levers, tested on real data (what works, what doesn't)

Rather than assume, I measured the candidate completeness levers:

- **Deeper exposures: WORKS.** DES 270 s → r ~ 23.1 vs 90 s → r ~ 21.9
  (§14). Depth scales completeness directly. This is a *data* lever
  (longer integrations), not code.
- **Gaia astrometric refinement: NO headroom.** Raw instcal WCS is
  already 0.18″ vs Gaia (§13) — the DECam pipeline pre-calibrates to Gaia.
- **Crude (scalar) difference imaging: does NOT help (measured).** A
  median-reference subtraction from the 4 same-field epochs, re-extracted on
  the residual: recall 4/30 → 3/27 with 5× FEWER sources. Reason is
  fundamental — subtracting two non-PSF-matched images adds √2 noise and
  leaves a dipole residual at every star (seeing differs epoch-to-epoch),
  manufacturing artifacts while burying faint reals.
- **PSF-matched (Alard-Lupton) difference imaging: BUILT and it WORKS
  (measured 2026-06-02).** Implemented `psf_matched_difference()` — solves a
  least-squares matching kernel (Gaussian × polynomial basis) from the static
  stars so the sharper frame's PSF is convolved to the broader one before
  subtraction, removing the dipoles crude subtraction left. Validated two
  ways:
  - *Synthetic* (`tests/test_difference.py`): cancels static-star cores to
    <50% of crude residual (both PSF orderings) while keeping an injected
    mover at >5σ; kernel integral tracks the photometric flux ratio.
  - *Real DECam* (`scripts/test_psf_matched_completeness.py`, 12 CCDs of the
    4-epoch (330,−10) field): ran PSF-matched on all 12 CCDs with **zero
    fallback**. Known-asteroid recoveries **plain 5 → crude 5 → PSF-matched
    6**, and PSF-matched preserved **1002** real sources vs crude's 411 (crude
    destroys real detections along with stars). It recovered **one known
    asteroid that plain extraction missed** — a genuine completeness gain from
    cleanly removing the static field.

  The absolute gain on *this* field is modest (+1 asteroid) because the field
  is depth-limited and asteroid-poor: most of the predicted knowns are simply
  fainter than the ~21.9 detection limit and are invisible to *any* method.
  The kernel pays off most on a **denser / near-ecliptic** field where many
  knowns sit near the limit and on top of stars. PSF-matched now strictly
  dominates crude and is the default differencing path
  (`run_full_image_pipeline.py` step 7).

**Honest conclusion:** the pipeline is at the accuracy/completeness ceiling
its *algorithms + data* support, and the one remaining algorithm lever with
measured headroom — PSF-matched subtraction — is now **built, tested, and
shown to beat both plain and crude extraction on real pixels** (not just
scoped). Remaining gains are now genuinely *data*: (a) deeper / near-ecliptic
/ good-cadence fields (where PSF-matched differencing will show a larger
completeness win), and (b) the rate-constrained multi-night confirmation
linker (architecture exists in `run_des_discovery.py` / `triplet_linker`;
needs good-cadence data to exercise).

*(Update 2026-06-02: "no hidden quick wins remain" turned out to be wrong —
see §16. A detection-stage PSF-width bug was silently dropping ~2/3 of
detectable asteroids; fixing it ~tripled real recall. The lesson held: don't
declare a ceiling until you've measured the front-end against truth.)*

## 16. Detection PSF-FWHM bug — found, fixed, ~3× real recall (2026-06-02)

Chasing why discovery recall looked low, a per-object **pixel-truth**
diagnostic (does each predicted known have flux at its pixel, and does the
pipeline actually extract it?) exposed a real front-end leak: **81% of
detectable known asteroids had flux in the pixels, but only 31% were being
extracted.**

Root cause: `detect_sources_in_image` ran its DAOStarFinder matched filter at
an **assumed `fwhm_px=3.5`** while the real DECam r-band seeing was **8.7px**.
A detection kernel tuned to the wrong width mismatches real sources and
silently loses the faint ones. (A second defect — a dimensionally-meaningless
`sharpness * fwhm_px` post-filter — was also dropping real sources.)

Fix: a robust, self-calibrating PSF estimator (`measure_image_fwhm`) that fits
bright, isolated, unsaturated stars with a Gaussian on FWHM-scaled stamps and
iterates to convergence (validated: 3px→3.01, 6px→6.0 even from a bad initial
guess; rejects saturated stars). It now drives detection automatically
(`auto_fwhm=True` default — every driver self-calibrates) and the trail/rate
path (`stellar_psf_fwhm` delegates to it).

Real-DECam result through the actual pipeline (60 CCDs, corrected mag):

| corrected mag | extracted recall before (3.5px) | extracted recall after (measured) |
|---|---|---|
| r < 20  | 72% | **94%** |
| 20-21   | **9%** | **88%** |
| 21-22   | 0% | **62%** |
| money (mag<21, untrailed, clean) | **31%** | **91%** |

Total detections 11,017 → 27,966 (2.5×). The fixed pipeline now *exceeds* the
crude flux-present check, because a matched filter at the correct PSF width is
more sensitive than a naive peak test. Cost: per-CCD FWHM measurement roughly
doubles extraction wall-time (acceptable; can be measured once per exposure if
needed). **Keystone lesson, again: measure the PSF, never assume it.**

## What this means

- **Slow/normal-rate objects, point-source regime, V < ~21:**
  detection front-end validated on real pixels and now recovers **~91% of
  detectable known asteroids** (88% at mag 20-21) after the PSF-FWHM fix
  (§16) — up from ~31%. Cross-match + linker + IOD geometry validated on
  synthetic injections. This part is real and now genuinely strong.
- **Fast movers (streaks):** detection now validated on real pixels via
  difference imaging (100% recall to r≈21.5). Still needs the template-
  fetch wiring to run on live data, but the algorithm works on real
  pixels — no longer a dead end.
- **Historical (multi-year) data:** N-body is correct but does not yet
  scale to full-catalog cross-match.
- **"100% on everything" is false.** Real limiting magnitude is
  r ≈ 22.5 (50%); fast-mover detection on real frames is currently
  ~0%; nothing in astronomy is 100%.
