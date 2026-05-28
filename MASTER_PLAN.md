# ARIADNE — Master Plan & Project Bible

> **Ariadne**: the thread through the interplanetary labyrinth.
> An open, high-fidelity engine and atlas for **low-energy trajectory design** and
> **discovery of the natural transport structures of the solar system** — the
> invariant-manifold "tubes," heteroclinic chains, and resonant corridors that let a
> spacecraft travel between bodies for a fraction of the usual fuel.

---

## 0. How to use this document

This is the **single source of truth** for the project. It is written so that *anyone*
— a new collaborator, a future version of us, or a NASA reviewer — can pick the project
up at any point and continue without losing context.

- **If you are new:** read §1 (Vision), §2 (Prior art / honest landscape), §3 (Science),
  then §8 (Architecture) and §10 (Roadmap). Then look at `CHANGELOG` (§16) to see where we are.
- **If you are continuing work:** go straight to §16 (Status & Changelog) and §10 (Roadmap)
  to find the current stage and the next "definition of done."
- **If you are reviewing the science:** §3 (Foundations), §5 (Data), §9 (Validation gates).
- **Naming:** "Ariadne" is a working codename. Renaming = change this title, the `src/ariadne`
  package folder, and references herein. Nothing else depends on the name.

**Document status:** `v0.1 — project kickoff, pre-code.` Update §16 every working session.

---

## 1. Vision & mission

### 1.1 The one-sentence pitch
Build the most complete, validated, open computational map of the solar system's
**natural low-energy transport network**, plus a trajectory engine that can *find and
optimize* routes on it — turning "24 million brute-force trajectory guesses" into a
principled search that returns better routes in orders of magnitude fewer evaluations.

### 1.2 Why this matters
Chemical propellant is the tyrant of spaceflight: every extra m/s of Δv costs mass,
money, and mission scope. The solar system, however, has a hidden structure of
**gravitational corridors** — pathways where the combined pull of multiple bodies does
the steering for you. Riding them, missions like **Genesis** and the rescue of **Hiten**
reached their destinations on a fuel budget that direct transfers cannot match. These
corridors are real, mathematically characterized, and *under-exploited* because mapping
and searching them at scale is hard.

If we map them comprehensively and make route-finding cheap and reliable, we lower the
energy floor for: lunar logistics (Artemis-era cargo), Mars cargo pre-positioning,
multi-moon tours (Jovian/Saturnian systems), asteroid access, and sample return. That is
the Mars/deep-space relevance: **cheaper cargo legs free up mass and money for crewed and
science payloads.**

### 1.3 What "success" looks like
1. We **reproduce** known results (Lagrange points, published low-energy transfers, flown
   missions) to within tight tolerance — earning trust.
2. We **out-search** brute-force/grid methods: same-or-better Δv in far fewer trajectory
   evaluations, with the eval-count reduction reported as a headline metric.
3. We **discover** at least one genuinely novel, non-obvious low-energy connection
   (a heteroclinic chain or resonant corridor) and validate it on full ephemeris.
4. We ship an **open atlas** + reproducible code + a clear white paper that a mission
   designer at JPL/GSFC could pick up and use.

### 1.4 What this is NOT (scope discipline / credibility firewall)
- **We do not invent new physics or new forces.** The dynamics use *standard gravity*
  (Newtonian n-body on real JPL ephemerides, plus standard perturbations). The credibility
  of everything here depends on this.
- The **coherence-field / corridor heuristics** from the broader Coherence/OneField work
  enter **only as a search-acceleration layer on top of real dynamics** (a heuristic for
  where to look), never as a replacement for the equations of motion. See §3.11 and §13.
- We are **standing on giants** (Poincaré, Conley, McGehee, Llibre–Martínez–Simó,
  Koon–Lo–Marsden–Ross, Belbruno, Howell, Lo, Ross, Parker–Anderson). Our contribution is
  **completeness, automated discovery at scale, search efficiency, and openness** — not a
  claim to have invented the tubes.

---

## 2. Prior art — the honest landscape

We must know exactly what exists so we can state precisely what is new.

| Thing | What it is | Who / when | Relevance |
|---|---|---|---|
| **Restricted 3-body problem** | The mathematical model of motion in two-primary gravity | Euler, Lagrange, Jacobi, Poincaré | Our core dynamics (CR3BP) |
| **Lagrange points L1–L5** | Equilibria of the rotating frame | Euler/Lagrange | Anchors of the network |
| **Lyapunov / Halo / Lissajous orbits** | Periodic & quasi-periodic libration-point orbits | Farquhar, Howell (halo, 1984) | The "stations" tubes attach to |
| **Invariant manifolds (tubes)** | Stable/unstable sets of libration orbits that channel transport | Conley, McGehee; Gómez–Llibre–Martínez–Simó | The "tunnels/jetstreams" |
| **Interplanetary Transport Network / Superhighway** | The global web of connected tubes | Lo, Ross; Koon–Lo–Marsden–Ross (KLMR) | Exactly our subject |
| **Genesis mission** | Flew Sun–Earth L1 halo + manifold return | JPL, Lo/Howell, 2001–2004 | Validation target |
| **Hiten / Weak Stability Boundary** | First ballistic lunar capture (fuel rescue) | Belbruno & Miller, 1991 | Validation target |
| **Low-energy lunar transfers** | Use Sun perturbation for cheap Moon capture | Parker & Anderson (JPL DESCANSO, 2014) | Earth–Moon method reference |
| **Theory of Functional Connections (TFC)** | Constraint-embedding functionals for BVP/trajectory solving | Mortari, 2017+ | The method in the article that sparked this |
| **The Coimbra study** | 24M-trajectory TFC sweep → 3,925 m/s Earth–Moon transfer via L1 + Lyapunov (66.7 m/s saving, 32 days) | Univ. Coimbra, *Astrodynamics* | The benchmark we reproduce & try to beat on **search cost** |
| **GMAT** | NASA's open-source mission analysis tool | NASA GSFC | Our independent validation oracle |
| **SPICE / DE440** | NASA ephemeris + geometry toolkit & data | NAIF/JPL (Acton); Park et al. 2021 | Our real data backbone |

**Our four genuine differentiators** (restate often, defend always):
1. **Completeness** — a unified atlas of orbit families + manifolds + connections across
   many systems, not one-off mission studies.
2. **Automated discovery** — systematic mining of tube intersections for *novel* chains.
3. **Search efficiency** — field/heuristic-guided global search vs brute sweeps.
4. **Openness & reproducibility** — cross-validated against GMAT, fully documented.

---

## 3. Scientific foundations

This section is the theory reference. Symbols are collected in §14.

### 3.1 The Circular Restricted Three-Body Problem (CR3BP)
Two **primaries** (e.g. Earth & Moon) move in circular orbits about their common
barycenter; a third body of negligible mass (the spacecraft) moves in their combined
gravity. We work in the **rotating (synodic) frame** that co-rotates with the primaries,
nondimensionalized so that:
- total mass = 1, distance between primaries = 1, angular rate = 1;
- the **mass parameter** `μ = m₂ / (m₁ + m₂)` is the only system parameter;
- larger primary `m₁ = 1−μ` sits at `(−μ, 0, 0)`; smaller `m₂ = μ` at `(1−μ, 0, 0)`.

Distances to the primaries:
```
r₁ = sqrt((x+μ)²  + y² + z²)
r₂ = sqrt((x−1+μ)² + y² + z²)
```

**Pseudo-potential** (effective potential in the rotating frame):
```
Ω(x,y,z) = ½(x² + y²) + (1−μ)/r₁ + μ/r₂ + ½ μ(1−μ)
```

**Equations of motion:**
```
ẍ − 2ẏ = ∂Ω/∂x
ÿ + 2ẋ = ∂Ω/∂y
z̈      = ∂Ω/∂z
```
The `−2ẏ` and `+2ẋ` are the **Coriolis** terms; the `x,y` in Ω are the **centrifugal** terms.

### 3.2 The Jacobi constant (the energy integral)
The single conserved quantity of the CR3BP:
```
C = 2Ω(x,y,z) − (ẋ² + ẏ² + ż²)
```
`C` is the currency of the network. Lower `C` ⇄ higher energy ⇄ more of space is
accessible. **Conservation of `C` is our first integration sanity check** (it must hold to
~1e-10 over long propagations with a good integrator).

### 3.3 Zero-velocity curves/surfaces (the walls of the maze)
Setting velocity to zero gives `2Ω = C`: surfaces the spacecraft cannot cross at that
energy. As `C` decreases, "necks" open at L1, then L2, then L3 — these necks are the
**gateways** through which all low-energy transport must pass. This is *why* the corridors
exist and where they are.

### 3.4 Lagrange points
Five equilibria where `∇Ω = 0`:
- **Collinear L1, L2, L3** (on the x-axis): solve the collinear quintic; unstable
  (saddle × center), and therefore the seats of the manifold tubes.
- **Triangular L4, L5** at `(½−μ, ±√3/2, 0)`: stable for `μ < 0.0385` (Routh).

Known anchors to validate against (Earth–Moon, `μ ≈ 0.012150585`):
`L1 x ≈ 0.8369`, `L2 x ≈ 1.1557`, `L3 x ≈ −1.0051` (nondimensional, Moon at `1−μ`).

### 3.5 Periodic & quasi-periodic libration orbits
Around the collinear points live families of bounded orbits:
- **Planar Lyapunov** orbits (in the x–y plane);
- **Halo** orbits (3D, bifurcate from Lyapunov; Howell 1984);
- **Vertical Lyapunov** and **Lissajous** (quasi-periodic) orbits.
These are the "stations." Tubes are attached to them. We generate whole **families** by
numerical continuation (§3.8).

### 3.6 The State Transition Matrix (STM) and monodromy
Linearized sensitivity of the flow: `Φ(t)` solves `Φ̇ = A(t) Φ`, `Φ(0)=I`, where `A` is
the Jacobian of the EOM along the trajectory. Over one period `T` of a periodic orbit,
`M = Φ(T)` is the **monodromy matrix**. Its eigenvalues (**Floquet multipliers**) come in
reciprocal pairs `{λ, 1/λ}`:
- `|λ| > 1` → **unstable** direction (unstable manifold);
- `|λ| < 1` → **stable** direction (stable manifold);
- `|λ| = 1` → center / neutral.

### 3.7 Differential correction (how we actually find the orbits)
Periodic orbits are computed by **shooting + Newton's method using the STM**:
guess an initial state, integrate, measure how far it is from the periodicity/symmetry
condition, and use `Φ` to correct the guess. Variants: single shooting (symmetry method
for Lyapunov/halo), multiple shooting (robustness), and full collocation (§3.10).

### 3.8 Numerical continuation / homotopy
Given one orbit, step a parameter (energy, z-amplitude, Jacobi constant) and re-correct to
trace an entire **family**. **Pseudo-arclength continuation** handles folds. This is how we
get from "one Lyapunov orbit" to "the whole L1 Lyapunov family" to "halos."

### 3.9 Invariant manifolds — the tubes
From a periodic orbit, perturb states along the stable/unstable Floquet eigenvectors by a
small ε and propagate:
- **Unstable manifold** `W^u`: integrate **forward** → where the orbit *flows out to*.
- **Stable manifold** `W^s`: integrate **backward** → where you must come *from* to fall in.
These trace tubular surfaces in phase space. **A spacecraft inside a stable tube is
ballistically captured by the orbit it leads to — no fuel required.** These are the
"jetstreams." ε scaling and number of seed points are key numerical parameters (§6).

### 3.10 Poincaré sections & heteroclinic connections (how routes form)
Choose a surface of section (e.g. `x = 1−μ`, or `y = 0`); record where tubes cross it.
Where an **unstable tube from orbit A** intersects a **stable tube into orbit B** *on the
same section at the same Jacobi constant*, there is a near-free **heteroclinic connection**
A→B. Chaining these gives **heteroclinic chains** — exactly "leave A on tube X, coast,
arrive at B." This is the mathematical form of the user's "leave point A, travel on X,
arrive at B." Connections within one orbit are **homoclinic**.

### 3.11 The fidelity ladder (models, low → high)
1. **CR3BP** — circular, autonomous. Where structure is cleanest; all validation of the
   core happens here. *No external data needed (μ only).*
2. **BCR4BP** (bicircular) — add the Sun on a circular orbit. **Unlocks low-energy lunar
   transfers** and Sun–Earth ↔ Earth–Moon corridor handoffs (the Sun's tug is what makes
   cheap Moon capture possible).
3. **ER3BP** — elliptic primaries (real eccentricity), time-periodic.
4. **Patched three-body** — stitch Sun–Earth and Earth–Moon manifolds at a handoff section
   (how real IPTN routes are assembled).
5. **Full ephemeris n-body** — real positions of all relevant bodies from **DE440**, plus
   **solar radiation pressure (SRP)**, **nonspherical gravity** (J2, etc.), and relativistic
   corrections where needed. **This is the "hand to NASA" rung**: every candidate route is
   re-converged here before we claim a number.

### 3.12 Weak Stability Boundary (WSB) / ballistic capture
The fuzzy, chaotic region where capture/escape flips — Belbruno's mechanism for
*propellant-free* capture (Hiten). We characterize it numerically (stability index maps)
and use it for capture legs.

### 3.13 Trajectory optimization (turning a corridor into a flyable plan)
- **Impulsive** (chemical) vs **low-thrust** (electric) formulations.
- **Primer vector theory** (Lawden): the costate that tells you *whether and where* an
  impulse helps — the rigorous version of "follow the gradient" for impulsive transfers.
- **Indirect methods** (Pontryagin Maximum Principle): derive costate EOM, solve the
  two-point boundary value problem; accurate, sensitive to initial guess.
- **Direct methods**: transcribe to a nonlinear program — **Hermite–Simpson collocation**
  or **pseudospectral** (Gauss–Lobatto) — robust, what we lead with.
- **Continuation** from CR3BP solution → ephemeris solution (homotopy on perturbations).

### 3.14 "The field is the algorithm" — our search layer (rigorously homed)
The intuition "solve a field once and read the path off the gradient" has a real home:
- **Eikonal / Fast Marching Method (FMM):** solves `|∇T| = 1/F` for a minimum-cost arrival
  field `T`; gives global first guesses and corridor identification cheaply. (This is the
  rigorous form of the project's `wavefront_tau` experiment.)
- **Hamilton–Jacobi–Bellman (HJB) reachability / level sets** (Mitchell): solve a PDE over
  *state space* for the reachable / minimum-fuel set; the optimal control falls out of the
  value-function gradient. This is the principled "field IS the algorithm" for
  control-affine dynamics.
- **Heuristic transport-graph search** (our forge_shootouts lineage): once connections are
  catalogued, route-finding is shortest-low-energy-path search where the **Jacobi
  constant / value function serves as an admissible heuristic** — the eval-count collapse
  (cf. A\*-Tau-Corridor: 10 ops vs 57) is the contribution. **All of this only proposes
  guesses; real dynamics + optimal control verify them.**

---

## 4. Methodology — capability by capability

For each capability: inputs → method → outputs → validation.

1. **CR3BP propagator** — high-order adaptive integrator (DOP853 / explicit RK with tight
   tol; numba-JIT hot path), with **variational equations** for the STM. *Validate:* Jacobi
   constant conserved to ~1e-10; reversibility.
2. **Lagrange-point solver** — root-find the collinear quintic + triangular closed form.
   *Validate:* match published Earth–Moon / Sun–Earth values.
3. **Periodic-orbit corrector** — symmetry single-shooting + STM Newton. *Validate:*
   reproduce a published L1 Lyapunov / halo orbit (period, Jacobi).
4. **Family continuation** — pseudo-arclength. *Validate:* family Jacobi-vs-amplitude curve
   matches literature shape; halo bifurcation appears.
5. **Manifold generator** — Floquet eigenvector seeding + propagation. *Validate:* tube
   reaches the expected neck; energy preserved.
6. **Poincaré + connection finder** — section crossing DB + intersection matcher.
   *Validate:* recover a known L1↔L2 heteroclinic connection.
7. **BCR4BP & ephemeris models** — add Sun / DE440 perturbers + SRP + J2.
   *Validate:* propagate a real object (from Horizons) and match its state.
8. **Optimizer** — collocation NLP (SciPy/IPOPT-style) + primer-vector check.
   *Validate:* reproduce the Coimbra 3,925 m/s transfer; cross-check Δv in GMAT.
9. **Field/heuristic search** — FMM + HJB + transport-graph A\*. *Validate:* find the same
   optimum as brute sweep in ≪ evaluations; report the ratio.
10. **Discovery engine** — §7. *Validate:* every "novel" route survives full-ephemeris
    re-convergence and GMAT cross-check before it is reported.

---

## 5. Real data sources & conventions

**Network access confirmed** (JPL NAIF reachable). We use real data; nothing is faked.

### 5.1 Kernels (NASA SPICE / NAIF)
- **Planetary ephemeris:** `de440.bsp` (or `de441` for long span). Park et al. 2021.
- **Satellite ephemerides:** e.g. `jup365`, `sat441` for moon systems (later stages).
- **Leap seconds (LSK):** `naif0012.tls`.
- **Orientation / constants (PCK):** `pck00011.tpc`, `gm_de440.tpc` (GM values).
- Managed by `src/ariadne/data/kernels.py` (download, checksum, cache in `data/kernels/`,
  which is **git-ignored**). A `kernels.lock` records exact files + hashes for reproducibility.

### 5.2 Independent checks
- **JPL Horizons** (state vectors) — cross-validate ephemeris propagation.
- **NASA GMAT** — independent end-to-end Δv / trajectory validation (export scripts).
- **JPL Small-Body Database (SBDB)** — asteroid/comet targets (later stages).

### 5.3 Constants, units, frames, time — the policy (READ THIS)
- **Internal compute:** nondimensional CR3BP units. **Conversions** via characteristic
  scales: length `L*`, time `T* = 1/n`, velocity `V* = L*/T*`.
  - Earth–Moon: `L* ≈ 384,400 km`, `T* ≈ 4.342 d (375,200 s)`, `V* ≈ 1.025 km/s`, `μ ≈ 0.0121506`.
  - Sun–Earth: `L* = 1 AU = 1.496e8 km`, `V* ≈ 29.78 km/s`, `μ ≈ 3.0035e-6`.
- **Dimensional I/O:** strict SI (km, km/s, s) at all boundaries. A single `units.py`
  owns every conversion; **no ad-hoc factors anywhere else.**
- **Frames:** inertial **J2000/ICRF** for ephemeris; **rotating synodic** for CR3BP.
  Frame transforms live in one module and are unit-tested both directions.
- **Time:** **TDB / ET** (SPICE ephemeris time) internally; convert to/from UTC via the LSK.
- **GM source of truth:** `gm_de440.tpc` (consistent with the ephemeris).

---

## 6. Numerical & implementation notes (gotchas that cost days if ignored)
- **Integrator:** adaptive high-order (DOP853), rtol≈1e-12/atol≈1e-12 for orbits/manifolds;
  conserve Jacobi as a live check. numba-JIT the EOM + variational RHS.
- **Manifold ε:** perturbation magnitude must be small enough to stay near the linear
  manifold but large enough to grow in finite time; standard ~1e-6 (nondim) of the orbit
  scale. **Record ε in outputs** — results depend on it.
- **Stiff necks:** near L-points, dynamics are sensitive; use event detection for section
  crossings rather than fixed steps.
- **Symmetry:** exploit the CR3BP `(x,y,z, ẋ,ẏ,ż,t) → (x,−y,z, −ẋ,ẏ,−ż,−t)` symmetry for
  corrector robustness and for halving manifold work.
- **Determinism:** fixed seeds, pinned dependency versions, logged kernel hashes — every
  result must be byte-reproducible.

---

## 7. The discovery engine (the "jetstreams," done rigorously)
1. **Catalog** libration-orbit families + their manifolds across target systems and a grid
   of Jacobi constants → store in the atlas (§8, HDF5).
2. **Build the transport graph:** nodes = `(system, orbit-family, energy level)`;
   edges = validated low-Δv manifold connections (from Poincaré intersection mining).
3. **Search:** novel-route queries become **shortest low-energy path** problems on this
   graph — our SSSP/corridor engine on a *physically real* graph. Multi-objective:
   Δv, time-of-flight, comms geometry, capture robustness.
4. **Triage & verify:** candidate routes are auto-flagged, **re-converged on full
   ephemeris**, optimized, and **cross-checked in GMAT** before being reported as real.
5. **Rank & publish** to the atlas with full provenance (model rung, ε, kernels, Δv, TOF).

Realistic discoveries: cheaper Earth–Moon routes via Sun–Earth L2; multi-moon tours in the
Jovian/Saturnian systems; resonance-hopping asteroid corridors. We report **specific new
routes**, never "new physics."

---

## 8. Software architecture
```
Ariadne/
  MASTER_PLAN.md            # ← this document (single source of truth)
  README.md                 # short orientation, points here
  requirements.txt          # pinned deps
  pyproject.toml            # (added when packaging)
  .gitignore                # ignores data/kernels, results, caches
  src/ariadne/
    dynamics/      # cr3bp.py, bcr4bp.py, er3bp.py, ephemeris_nbody.py, forces.py, variational.py
    orbits/        # lagrange.py, differential_correction.py, families.py, continuation.py
    manifolds/     # stm.py, monodromy.py, manifold.py
    connections/   # poincare.py, intersect.py, heteroclinic.py
    fields/        # eikonal.py (FMM), hjb.py (reachability), corridor_heuristic.py
    optimize/      # shooting.py, collocation.py, indirect.py, primer_vector.py
    data/          # kernels.py, horizons.py, constants.py, units.py, frames.py, time.py
    transport_graph/ # graph.py, discovery.py, ranking.py
    viz/           # rotating_frame.py, poincare_maps.py, three_d.py
    io/            # atlas.py (HDF5), trajectory_export.py, gmat_export.py
    validate/      # gates.py + one module per validation target
  tests/           # unit + validation tests (pytest)
  data/kernels/    # downloaded SPICE kernels (git-ignored)
  notebooks/       # exploration
  results/atlas/   # generated atlas + route catalogs
  docs/            # supplementary notes, derivations, figures
```
**Principles:** one job per module; `units.py`/`frames.py`/`time.py` own *all* conversions;
every capability has a validation test; configs in YAML; structured logging; pure functions
in hot paths for numba. Public API re-exported from `src/ariadne/__init__.py`.

---

## 9. Validation & falsification gates (pass/fail, with targets)

No capability is "done" until its gate passes. No route is "real" until §7.4 passes.

| # | Gate | Target / source | Model rung |
|---|---|---|---|
| G1 | Jacobi constant conserved | |ΔC| < 1e-10 over 100 periods | CR3BP |
| G2 | Lagrange points | Match published EM/SE values to 1e-6 | CR3BP |
| G3 | L1 Lyapunov orbit | Reproduce a published orbit (period, C) | CR3BP |
| G4 | Family continuation | Halo bifurcation + C-amplitude curve shape | CR3BP |
| G5 | Manifold tube | Reaches expected neck; energy preserved | CR3BP |
| G6 | Heteroclinic connection | Recover known L1↔L2 connection | CR3BP |
| G7 | Ephemeris propagation | Match a Horizons state to ~km / mm/s | DE440 |
| G8 | **Coimbra transfer** | Reproduce ~3,925 m/s, ~32 d, L1+Lyapunov | BCR4BP→eph |
| G9 | Flown mission | Reproduce Genesis manifold and/or Hiten capture | eph |
| G10 | GMAT cross-check | End-to-end Δv agree within tolerance | eph |
| G11 | **Search efficiency** | Match brute-sweep optimum in ≪ evals (report ratio) | any |
| G12 | Novel route | Survives full-eph re-convergence + GMAT | eph |

---

## 10. Roadmap (stages, deliverables, definition of done)

> Quality-first, no timelines. A stage is done when its gates pass and its docs are updated.
> Earth–Moon is nailed end-to-end **before** generalizing (per decision on 2026-05-28).

- **Stage 0 — Foundation** *(in progress)*: repo, this doc, deps, data/units/frames/time
  scaffolding, kernel manager. *DoD:* `import ariadne` works; kernels download + checksum.
- **Stage 1 — CR3BP core (Earth–Moon)**: propagator + variational eqs; Jacobi; Lagrange
  solver; zero-velocity curves. *DoD:* **G1, G2**.
- **Stage 2 — Orbits & families**: differential corrector; L1/L2 Lyapunov + halo; continuation.
  *DoD:* **G3, G4**.
- **Stage 3 — Manifolds & connections**: STM/monodromy; tube generation; Poincaré; heteroclinic
  finder. *DoD:* **G5, G6**.
- **Stage 4 — The headline reproduction**: BCR4BP; low-energy lunar transfer; optimizer
  (collocation + primer vector); reproduce the Coimbra result. *DoD:* **G8**.
- **Stage 5 — Real ephemeris**: SPICE/DE440 n-body + SRP + J2; Horizons checks; re-converge
  Stage 4 on full ephemeris; GMAT export. *DoD:* **G7, G9, G10**.
- **Stage 6 — Field/heuristic search**: FMM + HJB + transport-graph A\*; brute-sweep
  baseline; efficiency benchmark. *DoD:* **G11**.
- **Stage 7 — Discovery engine**: atlas build; transport graph; novel-route mining + verify.
  *DoD:* **G12** + first verified novel route.
- **Stage 8 — Generalize**: Sun–Earth, Mars system, Jovian/Saturnian moons, asteroids;
  scale the atlas.
- **Stage 9 — Deliverables**: white paper, open atlas release, GMAT-validated reference routes.

---

## 11. Compute & performance plan
- **Hot paths:** numba-JIT EOM + variational RHS; vectorized manifold seeding.
- **Embarrassingly parallel:** family continuation, manifold fans, Poincaré sweeps,
  discovery search → `multiprocessing`/`joblib` now; GPU/cluster later if needed.
- **Storage:** HDF5 atlas (orbits, manifolds, connections, routes) with metadata/provenance.
- **Reproducibility:** pinned versions, kernel hashes, seeds, config snapshots per run.

---

## 12. Outputs & deliverables
- **The Atlas** (HDF5 + browsable index): orbit families, manifolds, connection catalog,
  ranked route database — each with full provenance.
- **Trajectory exports:** SPK/ephemeris-friendly + CSV + **GMAT scripts**.
- **Figures:** rotating-frame orbits, manifold tubes, Poincaré maps, zero-velocity curves,
  3D fly-throughs.
- **The white paper:** methods + validation + the efficiency result + any novel routes,
  written for a mission-design audience.

---

## 13. Risks, limitations & honest caveats
- **Credibility firewall (most important):** standard gravity for dynamics; coherence-field
  ideas are *only* a heuristic search layer. Violating this sinks the project's credibility.
- **"Novel route" burden of proof:** nothing is announced until it survives full-ephemeris
  re-convergence **and** an independent GMAT check.
- **Model-rung honesty:** every reported number is tagged with its model rung; CR3BP results
  are structural, not flight-ready.
- **Optimizer fragility:** indirect methods need good guesses — that's *why* the field/graph
  layer (good global guesses) is valuable; lead with robust collocation.
- **We are not reinventing astrodynamics:** the win is scale + automation + efficiency +
  openness. Say so plainly.
- **Compute ceilings:** full-eph optimization and solar-system-wide discovery are expensive;
  scope per stage, parallelize, and report honest cost.

---

## 14. Glossary & symbols
- **μ** — mass parameter `m₂/(m₁+m₂)`.
- **Ω** — pseudo-potential (rotating frame effective potential).
- **C** — Jacobi constant, `C = 2Ω − v²`.
- **L1…L5** — Lagrange (libration) points.
- **STM (Φ)** — state transition matrix; **M = Φ(T)** monodromy matrix.
- **Floquet multipliers** — eigenvalues of M (reciprocal pairs).
- **W^s / W^u** — stable / unstable invariant manifold ("tubes").
- **Heteroclinic / homoclinic connection** — tube intersection linking two / one orbit(s).
- **CR3BP / BCR4BP / ER3BP** — circular / bicircular-4-body / elliptic restricted models.
- **WSB** — weak stability boundary (ballistic capture region).
- **Δv** — velocity change (fuel cost).
- **TOF** — time of flight.
- **FMM / HJB** — Fast Marching Method / Hamilton–Jacobi–Bellman.
- **L*, T*, V*** — characteristic length/time/velocity for nondimensionalization.
- **IPTN** — Interplanetary Transport Network.
- **SPICE / SPK / DE440** — NASA NAIF toolkit / ephemeris file format / planetary ephemeris.
- **GMAT** — NASA General Mission Analysis Tool.
- **TFC** — Theory of Functional Connections.

---

## 15. References (author/title/year — verify editions when citing formally)
- Szebehely, *Theory of Orbits: The Restricted Problem of Three Bodies*, 1967.
- Koon, Lo, Marsden, Ross, *Dynamical Systems, the Three-Body Problem and Space Mission
  Design*, 2011 (freely available).
- Gómez, Llibre, Martínez, Simó, *Dynamics and Mission Design Near Libration Points*, 2001.
- Howell, "Three-Dimensional, Periodic 'Halo' Orbits," *Celestial Mechanics*, 1984.
- Belbruno & Miller, "Sun-Perturbed Earth-to-Moon Transfers with Ballistic Capture,"
  *J. Guidance, Control, and Dynamics*, 1993; Belbruno, *Capture Dynamics…*, 2004.
- Parker & Anderson, *Low-Energy Lunar Trajectory Design*, JPL DESCANSO series, 2014 (free).
- Lo & Ross, "The Lunar L1 Gateway…"; Lo, "The InterPlanetary Superhighway," 2002.
- Lawden, *Optimal Trajectories for Space Navigation*, 1963 (primer vector).
- Mortari, "The Theory of Connections," *Mathematics*, 2017 (and TFC follow-ups).
- Mitchell, *A Toolbox of Level Set Methods* (HJB reachability), 2007.
- Acton, "Ancillary data services of NASA's NAIF" (SPICE), 1996; NAIF documentation.
- Park, Folkner, Williams, Boggs, "The JPL Planetary and Lunar Ephemerides DE440 and
  DE441," *Astronomical Journal*, 2021.
- NASA GMAT documentation (GSFC).
- Univ. Coimbra Earth–Moon TFC study, *Astrodynamics* (the article that sparked this).

---

## 16. Status & changelog (UPDATE EVERY SESSION)

**Current stage:** Stage 3 — Manifolds & connections **COMPLETE** (gates G5, G6 pass). Next: Stage 4.
**Next action:** Stage 4 — the headline reproduction. Add `dynamics/bcr4bp.py` (bicircular,
add the Sun) to unlock low-energy lunar transfers, and `optimize/collocation.py`
(Hermite–Simpson direct transcription) + `optimize/primer_vector.py`. Use the Stage 3
heteroclinic/tube machinery to build an Earth–Moon L1→Lyapunov ballistic-approach transfer
and reproduce the Coimbra result (~3,925 m/s, ~32 d). Target **G8**.
**Repo:** https://github.com/Jphilbrick10/Ariadne (private).

**Stage 1 results (Earth-Moon, mu=0.012150584):** Jacobi conserved max|dC|=1.4e-12;
STM vs finite-diff max err=3.7e-6; Lagrange points match published values to ~5e-11
(L1=0.8369151324, L2=1.1556821603, L3=-1.0050626453).

**Stage 2 results (Earth-Moon, L1):** tiny-amp Lyapunov period matches linear theory to
7.7e-7; finite-amp orbit (Ax=0.02) periodic to 2.4e-11, C=3.1659; 40-member family,
Jacobi monotonic 3.200->2.913, period 2.69->4.07; **halo bifurcation located at C=3.1864**
(matches literature ~3.18-3.19 for EM-L1). Run: `python -m pytest` (18 pass) or
`PYTHONPATH=src python -m ariadne.validate.stage2`.

**Stage 3 results (Earth-Moon):** L1 Lyapunov unstable tube (lambda=2.0e3) conserves Jacobi
to median 2.6e-12 along the tube (close approaches bounded < 1e-5), 40/40 trajectories reach
the Moon neck. **L1<->L2 heteroclinic connection found at C=3.15** on section x=1-mu, crossing
(y,vy)=(-0.0732, 0.0595). Figures in docs/figures/ (tubes + Poincaré section; L1 family +
halo bifurcation). Run: `PYTHONPATH=src python -m ariadne.validate.stage3` and
`PYTHONPATH=src python -m ariadne.viz.figures`.
NOTE on robustness: finite-amplitude orbits must be reached by CONTINUATION
(lyapunov_family / lyapunov_orbit_at_jacobi with the tangent predictor); a raw linear guess
only converges near the libration point. Manifold branch (+/-1) that points Moon-ward varies
per orbit — try both (find_heteroclinic does).

**Decisions on record:**
- 2026-05-28 — New standalone repo (credibility); codename **Ariadne**.
- 2026-05-28 — Reproduce **Earth–Moon first**, then generalize.
- 2026-05-28 — Coherence-field methods are a **search-acceleration layer only**; dynamics
  use standard gravity (credibility firewall).
- 2026-05-28 — Documentation-first: this master doc precedes code and is kept exhaustive.

**Changelog:**
- 2026-05-28 `v0.3` — Stage 3 (manifolds & connections) complete. Added manifolds/manifold.py
  (Floquet eigenvector seeding + tube propagation), connections/poincare.py (sections +
  tube cuts), connections/heteroclinic.py (loop-intersection connection finder),
  orbits.lyapunov_orbit_at_jacobi (Jacobi targeter), viz/figures.py, validate/stage3.py,
  test_manifolds.py + test_connections.py (18 tests pass). Gates G5, G6 pass; L1<->L2
  heteroclinic at C=3.15. Robustness fix: tangent predictor in continuation (raw linear
  guess + Newton step-clamp removed — the clamp caused crossing-regime oscillation).
- 2026-05-28 `v0.2` — Stage 2 (orbits & families) complete. Added orbits/linear.py
  (collinear linear modes + Lyapunov guess), orbits/differential_correction.py
  (symmetric single-shooting corrector, monodromy, stability indices),
  orbits/families.py (natural-parameter continuation + halo-bifurcation finder),
  validate/stage2.py, test_orbits.py. Gates G3, G4 pass; halo bifurcation C=3.1864.
- 2026-05-28 `v0.1.1` — Stage 1 (CR3BP core) complete and pushed to GitHub. Added
  dynamics/cr3bp.py (EOM, Jacobi, variational STM), orbits/lagrange.py (L1-L5),
  data/constants.py + units.py, validate/stage1.py, pytest suite. Gates G1, G2 pass.
- 2026-05-28 `v0.1` — Repo scaffolded; MASTER_PLAN.md written (vision, prior art, full
  science foundations, data/units policy, architecture, validation gates, roadmap,
  risks, glossary, references).
```
(Append newest entries at the top of the changelog. Bump version on each substantive update.)
```
