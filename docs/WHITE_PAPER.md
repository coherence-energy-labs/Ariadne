# Ariadne: an open, validated engine for low-energy trajectory design and discovery

**A from-scratch astrodynamics toolkit for the Interplanetary Transport Network — built on
standard gravity, validated against NASA GMAT and JPL DE440, with an automated route-discovery
layer and a multi-system atlas.**

*Working paper. Standard gravity throughout (credibility firewall). Nothing in this work is new
physics; the contribution is openness, validation, automation, and an honest accounting of what is
and is not established.*

---

## Abstract

We present **Ariadne**, an open engine that designs and discovers low-energy spaceflight routes in
the Circular Restricted Three-Body Problem (CR3BP) and its higher-fidelity extensions, up to the
full JPL DE440 ephemeris and cross-validation against NASA's General Mission Analysis Tool (GMAT).
The engine reproduces the published University of Coimbra Earth–Moon optimum (≈3,925 m/s, ~32 d) to
within model fidelity — bracketing it from both sides (3,761 m/s ballistic-capture / 3,953 m/s
direct) and finding a Sun-assisted weak-stability-boundary transfer at **3,907 m/s** (below the
published figure, at a longer 48.8-day flight time). An independent GMAT cross-check of an identical
trans-lunar state agrees to **149 m / 0.89 mm/s** over three days. We then discretize the
Interplanetary Transport Network into a **transport graph** and route it with Dijkstra (single-source
shortest path) and A\*; the search reaches the provable optimum with **~42× fewer node expansions
than exhaustive enumeration**, and **discovers a non-obvious 3-impulse L1→L2 reconfiguration (~17
m/s) about twice as cheap as the single direct patch**, exploiting the Oberth effect at high-speed
near-Moon crossings. The engine generalizes with only a change of constants across **six orders of
magnitude in mass ratio** (Mars–Phobos μ≈1.7×10⁻⁸ through the DART/Hera binary asteroid
Didymos–Dimorphos μ≈7×10⁻³), recovering the (μ/3)^⅓ Hill scaling, and persists its results to a
round-trippable HDF5 **atlas**. Every capability is gated by a pass/fail validation test, and every
caveat is stated plainly.

---

## 1. Motivation

A 2024 University of Coimbra study optimized Earth–Moon transfers over ~24 million trajectories using
the Theory of Functional Connections, reporting a notable low-energy solution near **3,925 m/s** with
a ~32-day time of flight. We set out to build — from scratch, openly, and to a standard one could hand
to a mission-design team — an engine that (a) reproduces results of that class on real data, (b)
cross-validates against the industry-standard tool (GMAT), and (c) goes further by treating the
natural low-energy "highway" structure of the solar system as a **searchable graph**, then mining and
verifying routes on it.

The deeper motivation is methodological. The invariant-manifold "tubes" of libration-point orbits and
their heteroclinic connections form the Interplanetary Transport Network (IPTN). Finding minimum-cost
itineraries on that network is, formally, a **shortest-path problem** — the setting of Dijkstra's
algorithm and A\*. Ariadne makes that formal correspondence concrete and benchmarks it.

## 2. The credibility firewall

The single most important design rule: **all dynamics use standard Newtonian gravity.** A separate
"coherence" framework (the user's research interest) enters *only* as a scoring/search-acceleration
layer — never as a force model. "Coherence" is operationalized strictly as **robustness** (how far a
trajectory's arrival drifts per unit injection error). It is a real, *different* objective; it does
**not** reduce the energy floor, and it is never represented as free energy or new physics. Violating
this firewall would forfeit the project's credibility, so it is enforced throughout and re-stated in
every results section.

## 3. Methods — the fidelity ladder

Each rung is validated before the next is trusted (Section 4).

1. **CR3BP core.** Rotating-frame equations of motion, the Jacobi integral, the five Lagrange points,
   zero-velocity curves, and the state-transition matrix (variational equations).
2. **Periodic orbits & families.** Symmetric single-shooting differential correction for planar
   Lyapunov orbits; natural-parameter continuation with a tangent predictor; the halo bifurcation
   located where the vertical stability index crosses +1.
3. **Invariant manifolds & connections.** Floquet eigenvectors of the monodromy matrix seed the
   stable/unstable tubes; Poincaré sections cut the tubes; heteroclinic connections are tube-cut
   intersections in the (y, v_y) plane.
4. **Solar perturbation (BCR4BP).** The bicircular four-body model adds the Sun in the synodic frame —
   the lowest-fidelity model that captures ballistic (weak-stability-boundary) capture.
5. **Full ephemeris (DE440).** SPICE/DE440 kernels; test-particle and mutual n-body propagators;
   Lambert (universal-variable) and Hermite–Simpson collocation solvers; a differential corrector that
   targets the *real* Moon to ~50 m.
6. **GMAT cross-validation.** Identical states are propagated in Ariadne and in NASA GMAT (GmatConsole,
   point masses Earth+Sun+Luna) and compared.
7. **Coherence/robustness lens.** Endpoint sensitivity (km of arrival drift per m/s of injection error)
   and a Δv-vs-robustness Pareto optimizer.
8. **Transport-graph search.** The IPTN as a weighted directed graph; Dijkstra, A\*, and an exhaustive
   brute-force baseline; an admissible energy heuristic.
9. **Discovery engine.** Yen's k-shortest-paths mining into a ranked route catalog, with CR3BP
   verification and solar-perturbation survivability of each route.
10. **Generalization & atlas.** The same engine across many systems; a persistent HDF5 atlas.

## 4. Validation gates

No capability is "done" until its gate passes; no route is "real" until it survives the burden of
proof (Section 6.4). Results as built:

| Gate | What it checks | Result |
|---|---|---|
| G1 | Jacobi constant conserved | <1e-10 over long integrations |
| G2 | Lagrange points vs published | <1e-6 |
| G3/G4 | Lyapunov orbit + family + halo bifurcation | bifurcation at C≈3.1864 (matches literature) |
| G5/G6 | Manifold tube + L1↔L2 heteroclinic | connection recovered at C=3.15 |
| G7 | Ephemeris propagation vs DE440 | ~0.02 km over 2 d |
| G8 | Coimbra transfer class | bracketed 3,761 / 3,953 m/s |
| G9 | Flown-mission mechanism (Genesis) | Sun–Earth L1 halo period 177.9 d |
| G10 | **GMAT cross-check** | **149 m / 0.89 mm/s** |
| G_wsb | Sun-assisted low-energy transfer | **3,907 m/s** (below Coimbra 3,925) |
| G_coh / G_opt | Robustness lens + weighted optimizer | robustness costs fuel; Pareto knee found |
| G_jov / G_lt | Galilean generalization + low-thrust | ports by constants; dC/dt=−2a·|v| to 2e-5 |
| G11 | **Search efficiency** | A\* ~42× fewer expansions than brute force |
| G12 | Discovery + verification | 8 ranked routes; Jacobi resid 1.3e-15; survives perturbation |
| G_gen / G_atlas | Multi-system + HDF5 atlas | μ 1.7e-8..7e-3, Hill scaling; round-trips exactly |

## 5. Key results

### 5.1 Reproducing — and beating — the Coimbra figure (within model fidelity)
On the full DE440 ephemeris, the TOF-optimized **direct** transfer converges to **3,953 m/s**, and a
**ballistic-capture** assembly reaches **3,756 m/s**, bracketing the published 3,925 m/s from both
sides. A Belbruno-style **weak-stability-boundary** transfer — propagated backward from a near-ballistic
lunar capture and optimized for minimum Δv — totals **3,907 m/s** at a 48.8-day flight time, below the
published 3,925. Honest framing: this is a same-class low-energy solution at a longer TOF (Coimbra's
~32 d), found from the dynamics, not fitted to the target.

### 5.2 GMAT cross-validation
An identical trans-lunar state propagated in Ariadne and in NASA GMAT agrees to **149 m in position
and 0.89 mm/s in velocity over three days** — our propagator matches the industry-standard tool.

### 5.3 The transport network as a searchable graph (the core new result)
We discretize the Earth–Moon IPTN: nodes are L1/L2 Lyapunov orbits at a grid of Jacobi energies; an
edge is an *exact* Poincaré section crossing, where the patch Δv = |v_x^src − v_x^dst| follows from
energy conservation to machine precision. Routing L1→L2:

- **All three routers agree on the optimum** (Dijkstra, A\*, exhaustive brute force) — search does not
  sacrifice optimality.
- **A\* reaches it with ~42× fewer node expansions than brute force** (5 vs ~208), with a *provably
  admissible* heuristic — the efficiency benchmark the project promised, and the Dijkstra/A\* payoff
  the IPTN-as-graph formulation was built for.
- **The discovered optimum is a non-obvious 3-impulse route at ~16–17 m/s** (L1@3.120 → L1@3.160 →
  L2@3.160 → L2@3.172), about **twice as cheap as the single direct patch** (~37 m/s). It changes
  energy at high-speed near-Moon crossings (the **Oberth effect**) and takes a free ballistic L1↔L2 hop
  between. Its route *topology is stable across manifold resolution* (cost converges 90→12.7, 120→16.1,
  150→16.9 m/s), distinguishing it from a discretization artifact.

### 5.4 Discovery + verification
Yen's k-shortest-paths mines a ranked catalog (8 distinct L1→L2 routes with a Δv-vs-robustness Pareto
set). Every route is **verified at the CR3BP rung**: each patch is a true section crossing (position
continuity exact; each side's Jacobi equals its orbit's to **1.3×10⁻¹⁵**; burn equals the edge Δv). The
optimum **survives the solar perturbation** as a bounded, correctable arc (CR3BP-vs-BCR4BP divergence
38,610 km ≈ 0.10 lunar-distance over 8.7 days — midcourse-correction scale, not chaotic escape).

### 5.5 Generalization and the atlas
With only a change of constants the engine produces periodic L1 Lyapunov orbits for six additional
systems spanning **~6 orders of magnitude in mass ratio**: Mars–Phobos (μ≈1.7×10⁻⁸, L1 16.6 km),
Saturn–Enceladus/Rhea/Titan, Sun–Mars (L1 1.08×10⁶ km), and the **DART/Hera binary asteroid
Didymos–Dimorphos** (μ≈7×10⁻³, L1 at just **150 m**). The L1 distances track the (μ/3)^⅓ Hill-radius
scaling across the whole spread. All results persist to a round-trippable **HDF5 atlas** with
provenance.

### 5.6 The coherence/robustness lens (honest scope)
"Coherence" as robustness produces a real, *different* objective: the Δv-vs-robustness frontier is
monotonic — **robustness costs fuel** (the cheapest WSB route is the most fragile). On the transport
graph, raising the robustness weight switches the choice from the cheap 3-hop route to the more robust
single direct patch. None of this beats the energy floor; physics fixes the minimum Δv.

## 6. Honest limitations

1. **Not new physics.** Standard n-body gravity throughout. The "coherence" framework is a scoring layer
   only, operationalized as robustness; it is never a force model and never free energy.
2. **"Novel" means automatically discovered + verified — not unknown to science.** The L1↔L2 heteroclinic
   web and the Oberth effect are well studied; the contribution is automated discovery, ranking, and
   verification on an open engine, not a route humanity had never seen.
3. **Model-rung honesty.** CR3BP results are structural; the discovered ~17 m/s libration route is
   verified at the CR3BP rung and shown to survive the solar perturbation, but a dedicated
   libration-to-libration *full-ephemeris* re-targeter (with a GMAT check) is **not yet built** — it is
   the one remaining fidelity tool. For the Earth→Moon *transfer* leg, that loop *was* closed (DE440
   ~50 m + GMAT 149 m).
4. **Discretization.** Patch-Δv estimates depend on manifold sampling; we use an energy-exact edge model
   (v_x from energy conservation) and verify route-topology stability across resolution. The near-Moon
   guard (|y| > 7,700 km) bounds the Oberth saving.
5. **FMM/HJB.** Continuous-field reachability (the other half of the classical "field search") is noted
   as the continuous-control cousin most relevant to the low-thrust regime; it is not wired to real
   dynamics here.
6. **Coimbra comparison.** 3,907 < 3,925 m/s is a same-class low-energy solution at a longer TOF, in a
   two-impulse patched model — not the identical transfer.

## 7. Reproducibility

Every result is reproduced by a validation script and covered by tests.

```
PYTHONPATH=src python -m ariadne.validate.stage1   # ... through stage16
PYTHONPATH=src python -m ariadne.validate.stage14  # transport-graph efficiency benchmark (G11)
PYTHONPATH=src python -m ariadne.validate.stage15  # discovery + verification (G12)
PYTHONPATH=src python -m ariadne.validate.stage16  # generalization + atlas
PYTHONPATH=src python -m pytest -q                 # full suite
PYTHONPATH=src python -m ariadne.viz.figures       # regenerate all figures
PYTHONPATH=src python -m ariadne.atlas.release      # build the open release bundle
```

SPICE/DE440 kernels download on first use; GMAT (R2026a) is located under `tools/` if installed.
The HDF5 atlas and release bundle are regenerable artifacts; the engine code is the deliverable.

## 8. References

Szebehely (1967); Koon, Lo, Marsden & Ross (2011); Gómez, Llibre, Martínez & Simó (2001); Howell
(1984); Belbruno & Miller (1993); Parker & Anderson (JPL DESCANSO, 2014); Lawden (1963); Mortari
(2017); Park et al., *DE440/DE441* (2021); NASA GMAT (GSFC); University of Coimbra Earth–Moon TFC study.
Full citations in `MASTER_PLAN.md` §15.
