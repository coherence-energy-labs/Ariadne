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
- **Stage 4 — BCR4BP + Δv mechanism** *(done)*: bicircular Sun-perturbed model; vis-viva Δv
  budget; quantify the ballistic-capture saving (the low-energy mechanism). *DoD:* **G8a/b/c**.
- **Stage 5 — Real ephemeris + optimization toolkit** *(done)*: SPICE/DE440 kernels;
  test-particle + mutual n-body propagators (validated vs DE440); Lambert solver;
  Hermite-Simpson collocation; GMAT exporter. *DoD:* **G7** (+ Lambert/collocation validated).
- **Stage 6 — Low-energy transfer + ballistic capture** *(done)*: real-manifold lunar capture
  (LOI from CR3BP dynamics); end-to-end LEO->LLO transfer assembly + sweep; brackets Coimbra.
  *DoD:* **G8r** (capture < direct), **G8t** (total in low-energy class).
- **Stage 7 — Halos + Genesis + cross-validation** *(done)*: 3D halo orbits (validated vs the
  Stage-2 bifurcation); Sun-Earth L1 halo + manifold to Earth (Genesis); independent
  ephemeris-library + integrator cross-checks. *DoD:* **G_halo, G9, G10\***.
- **Stage 8 — Full-ephemeris transfer** *(done)*: real DE440 trans-lunar transfer (Lambert seed
  + ephemeris differential correction targeting the real Moon to ~50 m); TOF-optimized direct
  transfer 3,953 m/s; brackets Coimbra 3,925. *DoD:* **G8e, G8b**.
- **Stage 9 — Literal GMAT cross-validation** *(done)*: GMAT R2026a installed locally; Ariadne vs
  GMAT propagation of an identical trans-lunar state agree to 149 m. *DoD:* **G10 (literal)**.
- **Stage 10 — Sun-assisted low-energy (WSB) transfer** *(done)*: Belbruno backward-from-capture
  in DE440, optimized for min Delta-v; LEO-departing low-energy transfer at **3,907 m/s** (below
  direct 3,953 and below Coimbra 3,925, longer TOF). *DoD:* **G_wsb**.
- **Stage 11 — Coherence / robustness lens** *(done)*: analysis/coherence.py (endpoint sensitivity
  + FTLE); measured the Δv-vs-coherence frontier (robustness costs fuel; cheapest=least coherent).
  *DoD:* **G_coh**.
- **Stage 12 — Coherence-weighted optimizer** *(done)*: Δv-vs-robustness Pareto front + knee
  (coherence-guided route chooser, standard gravity). *DoD:* **G_opt**.
- **Stage 13 — Engine generalization + low-thrust regime** *(done)*: transfers/jovian.py ports the
  CR3BP/manifold engine to the four Galilean moons with only constants (sensible L-points, periodic
  Lyapunov orbits, moon-tour Delta-v); dynamics/low_thrust.py adds continuous-acceleration dynamics
  (zero-thrust = CR3BP; tangential dC/dt = -2a|v| -> spiral). *DoD:* **G_jov, G_lt**.
- **Stage 14 — Field/heuristic search + efficiency benchmark** *(done)*: discretize the
  IPTN into a transport graph (nodes = Lyapunov orbits at energy levels, edges = exact
  section-crossing patches); route with Dijkstra (SSSP) and A* (admissible energy heuristic)
  against an exhaustive brute-force baseline; report the optimum-match + node-expansion speedup.
  (FMM/HJB continuous-field reachability is the continuous-control cousin, most relevant to the
  low-thrust regime — noted as a future extension, not wired to real dynamics here.) *DoD:* **G11**.
- **Stage 15 — Discovery engine + route verification** *(done)*: Yen k-shortest mining over the
  transport graph -> ranked route catalog + Delta-v/robustness Pareto set; discovers a 3-hop ~17 m/s
  route ~2x cheaper than the direct patch (Oberth at near-Moon crossings). Every route verified at
  the CR3BP rung (continuity + energy bookkeeping, machine precision) + solar-perturbation
  survivability quantified. "Novel" = automatically discovered + verified, not unknown to science;
  full DE440+GMAT re-convergence done for the transfer leg (Stages 8-10), a libration ephemeris
  re-targeter noted as the remaining tool. *DoD:* **G12a/b/c**.
- **Stage 16 — Generalize + scale the atlas** *(done)*: 6 new systems (Mars-Phobos, Saturn moons,
  Sun-Mars, the DART binary asteroid) verified to produce periodic libration orbits spanning ~6
  orders of magnitude in mass ratio (tracking the Hill scaling); `atlas/store.py` + `build.py`
  persist multi-system libration + the Earth-Moon graph + ranked routes to a round-trippable HDF5
  atlas with provenance. *DoD:* **G_gen, G_atlas**.
- **Stage 17 — Deliverables** *(done)*: `docs/WHITE_PAPER.md` (capstone paper), `atlas/release.py`
  (open release bundle: HDF5 atlas + INDEX.md + reference_routes.csv), and an honestly-tagged
  reference route table (the direct trans-lunar transfer is the GMAT-validated one). *DoD:*
  **G_deliver**. **Roadmap closed end-to-end (Stages 0-17).**

**Post-roadmap enhancements (user-requested, Stages 18-22):** the original roadmap is complete;
these add depth where real value remained.
- **Stage 18 — Ephemeris re-targeter** *(done)*: closes the one fidelity rung Stage 15 left open.
  `dynamics/frames.py` (exact synodic<->inertial transform) + `transfers/ephemeris_retarget.py`
  (multiple-shooting in DE440). The L1 Lyapunov orbit AND the L1<->L2 heteroclinic connection
  re-converge in the true ephemeris to within meters -- the discovered structure is ephemeris-real.
  *DoD:* **G18a/b/c**.
- **Stage 19 — 3D halos + NRHO** *(done)*: `orbits/nrho.py` pseudo-arclength continuation reaches
  the Gateway-class L2 NRHO (period 6.558 d, perilune alt 1,501 km, apolune 71,198 km, periodic to
  2.4e-14; near-stable, Floquet 2.18 vs Lyapunov 1,841). 3D-halo transport-graph integration (needs
  a 3D Poincare section) noted as the remaining extension. *DoD:* **G19a/b**.
- **Stage 20 — Moon-tour mining** *(done)*: `transfers/tisserand.py` -- Tisserand-graph gravity-assist
  tour of the Galilean moons (Io->Callisto): 252 m/s deterministic Delta-v vs 8,990 m/s Hohmann
  (35.7x saving), v_inf 1.2-1.9 km/s with feasible flyby turn authority. *DoD:* **G20a/b**.
- **Stage 21 — Interplanetary porkchop + epoch-swept global optimizer** *(done)*: `interplanetary/`
  -- launch epoch is now a free variable; recovers the real Earth->Mars windows (~26-mo cadence),
  global optimum 2026-11 / 5,679 m/s, time/energy Pareto + coherence knee; PNG + GMAT flight-path
  export. *DoD:* **G21a/b/c**.
- **Stage 22 — Gravity-assist multi-flyby global optimizer** *(done)*: `interplanetary/flyby.py` --
  patched-conic flyby chain + turn authority + differential-evolution global search; a Venus-Earth-
  Earth VEEGA cuts launch C3 to Jupiter 76.9 -> 16.8 km^2/s^2 (4.6x, Galileo-class), all flybys
  feasible (feasible but not fully ballistic; minimal-DSM refinement noted). *DoD:* **G22a/b**.
- **Stage 23 — Unified multi-objective grand optimizer** *(done)*: `interplanetary/grand.py` -- one
  optimizer balancing energy/time/robustness with coherence as the dial; picks a middle-ground
  Earth->Mars route (186 d/8,085 m/s), weights steer it; Edelbaum low-thrust estimate included.
  *DoD:* **G23a/b/c**.
- **Stage 24 — Packaging + open-source** *(done)*: pyproject.toml (src-layout, deps, entry points),
  MIT LICENSE, GitHub Actions CI (fast tests on 3.11/3.12), pip-installable, wheel builds.
  *DoD:* **G_pkg**. **Enhancement arc (Stages 18-24) closed.**

**Discovery / coherence / secular arc (user-requested, Stages 25-30)** -- full results in §16:
- **Stage 25 — Coherence (FLI) atlas** *(done, later partly OVERTURNED by Stage 26)*: FLI field +
  manifold-tube overlay; the "coherence skeleton" claim was refuted by Stage 26's fair test.
- **Stage 26 — Solar-system coherence atlas + self-correction** *(done)*: region-matched statistics
  refute the Stage-25 skeleton as a sampling artifact (the right honesty move). *DoD:* **G26a/b**.
- **Stage 27 — Principled coherence field + residual hidden-mass detector** *(done)*: tau_c field
  reduces to Newtonian gravity to |Phi|/c^2 (~1e-8, Newton recovery); trajectory-residual detector vs
  the real eTNOs + Kuiper floor. *DoD:* **G27a/b/c**.
- **Stage 28 — Inverse hidden-mass localizer + two-tier pipeline** *(done)*: weighted-LSQ inverse
  solver recovers an injected body's (mass, position); confidence region hones with N. *DoD:* **G28a/b/c**.
- **Stage 29 — Real-data bridge to Signalbook** *(done)*: indexes 4.4M real celestial sources, cross-
  matches a localization to 703 catalogued sources. Honest: confirmation handoff, not new-body detection.
- **Stage 30 — Long-term symplectic dynamics + secular Planet 9** *(done)*: machine-precision
  universal-variable Kepler + symplectic Wisdom-Holman (democratic heliocentric); energy bounded +
  2nd-order, DE440 cross-check 5.7e-4/century; real eTNOs diverge 0.7-64 AU/100 kyr with vs without
  Planet 9 -- the secular-accumulation proof. *DoD:* **G30a/b/c/d**.
- **Stage 31 — Secular/Gyr frontier + 427x acceleration** *(done)*: numba-JIT exact map (427x);
  doubly-averaged Gauss-ring secular integrator (reaches Gyr), validated vs analytic Laplace-Lagrange
  (ratio 1.0000) AND the exact integrator (14%/4% on the real eTNOs). Honest: fixed-ring P9 disperses,
  not confines (averaging removes resonances). *DoD:* **G31a/b/c/d**.
- **Stage 32 — Multi-backend ensemble + intelligent selector** *(done)*: 24-core parallel (8.7x@N=1k)
  + numba.cuda GPU (faithful 1.2e-14) ensemble integrators + a measured-crossover selector; honest
  forge audit (tau-field cost methods are redundant in astrodynamics; the selector is the transfer).
  *DoD:* **G32a/b/c**.
- **Stage 33 — General relativity (1PN)** *(done)*: Schwarzschild 1PN term; reproduces Mercury's
  42.99 arcsec/century perihelion advance (analytic ratio 1.0002). Engine is GR-capable. *DoD:* **G33a/b**.
- **Stage 34 — Real distant-TNO clustering significance** *(done)*: live JPL SBDB (706 bodies),
  circular stats + Rayleigh + Monte-Carlo. Honest: extreme-sample varpi clustering is MARGINAL on
  current data (p=0.07); selection bias uncontrolled; no Planet 9 claim. *DoD:* **G34a/b/c**.
- **Stage 35 — Differentiable dynamics + gradient optimization** *(done)*: JAX differentiable RK4 +
  Levenberg-Marquardt shooting (exact gradients through the integrator). 1.6 m in 9 iters vs
  Nelder-Mead 549 evals @ 9.2 km. Foundation for gradient-based trajectory design. *DoD:* **G35a/b/c**.
- **Stage 36 — Uncertainty propagation into clustering** *(done)*: real JPL 1-sigma element errors
  resampled; the marginal eTNO clustering degrades p=0.07 -> 0.13 (fragile; one object's perihelion is
  unconstrained). Current data give no compelling evidence for Planet 9. *DoD:* **G36a/b/c**.
- **Stage 37 — Independent cross-validation vs REBOUND** *(done)*: same DE440 ICs through Ariadne's WH
  and REBOUND WHFast agree to 1e-4/100yr, 1.7e-3/2000yr; REBOUND energy bounded 3.6e-6. The integrator
  is correct by an independent professional code, not just self-consistent. *DoD:* **G37a/b**.
- **Stage 38 — Selection bias vs clustering (OSSOS test)** *(done)*: scattered control population traces
  the selection function (mean 21.7 deg); extreme objects cluster the same way (20.4 deg) and are not
  distinguishable from it (p=0.12). Settles the honest verdict: no compelling P9 evidence. *DoD:* **G38a/b/c**.
- **Stage 39 — Discovery core: moving-object orbit linkage** *(done)*: HelioLinC-style linker; recovers
  5/5 known eTNOs (pure, 0 false positives) from a survey-realistic haystack of 430 tracklets. The
  machinery that can find UNKNOWN objects -- validated by known-object recovery. *DoD:* **G39a/b/c**.
- **Stage 40 — Dynamical-structure mining** *(done)*: checked the OTHER P9 signatures -- orbital-pole
  clustering (p=0.69, selection-explained) and Neptune-resonance proximity (0/19, detached). No structure
  beyond selection on any signature; honest verdict reinforced. *DoD:* **G40a/b**.
- **Stage 41 — Autodiff global trajectory optimization** *(done)*: exact 2-impulse Delta-v via autodiff
  LM shooting + global grid optimization; finds the Earth->Mars optimum 5.63 km/s (miss 0.19 km),
  matching the textbook value and Stage 21's Lambert porkchop. *DoD:* **G41a/b**.
- **Stage 42 — Discovery core on REAL telescope data** *(done)*: links the ACTUAL recorded MPC astrometry;
  recovers Sedna + 2001 FP185 from their real detections (+200 interlopers each) as pure candidates, 0
  false positives. Discovery core proven on real data. *DoD:* **G42a**.
- **Stage 43 — ITF ingest (the beast's dinner)** *(done)*: full pipeline on the live 135 MB MPC Isolated
  Tracklet File (2.6 M tracklets parsed in 17 s, 45 k slow movers, 607 clean candidates). Injection
  validated on a real bin; 25/25 spot-checked candidates cross-match to KNOWN objects (correct skeptical
  outcome -- no new find on the public archive). Discovery-capable end-to-end on the real unlinked
  archive. *DoD:* **G43a/b/c**.
- **Stage 44 — IOD + orbit-fit (the trustworthy filter)** *(done)*: full ITF + vslow sweep ran 863
  candidates -> 515 re-link to KNOWN catalogued objects via SkyBoT (pipeline works on real data); the
  348 unmatched run through a rebuilt orbit-fit (linker-(r,rdot)-hypothesis IOD seed + LM differential
  correction with light-time correction and pos-AU/vel-km/s rescaling). Validated against 5 real TNO
  orbits: Sedna 3.94", Eris 5.79", Makemake 8.68", Quaoar 3.79", 2001 FP185 1.39" (a-error 0.1-13%).
  Discrimination test: Sedna-alone accepted at 3.94", Sedna+Eris-mixed rejected at inf -- a sharp two-
  state filter. Final verdict on the 348 unmatched: 0 low-RMS leads (median 999 arcsec). The honest
  scientific outcome -- residual unlinked tracklets are mixed-object false-positive clusters, never a
  new discovery from a single pipeline run on the public archive. *DoD:* **G44a (validated IOD on real
  TNOs)**, **G44b (discrimination clean vs mixed)**, **G44c (0/348 verified unmatched leads with sound
  filter)**.
- **Stage 45 — Coherence-HJB (sampled-graph Helmholtz value function)** *(done)*: the Forge-Doctrine HJB
  substitute -- replace 6D grids with N=5-30k quasirandom samples + dynamics-derived k-NN edges, solve
  ONE sparse (Gamma*I + D*L)*V = source CG, apply Equation-of-One log-cost W = -ln(V/V_max). Validated
  on (a) 2D analytic eikonal V=||x|| (Spearman rho=+0.9990, 100% greedy reach), (b) dimension scaling
  through 6D synthetic (100% greedy reach at N=30k, k=8*dim), (c) planar CR3BP with dynamics-aware
  graph (Earth-Moon, 5000 samples, 127k edges, Helmholtz CG 17 iters, 29/29 greedy starts reach a
  lunar-vicinity goal sample in 3-5 steps). Real-dynamics value field, sub-second compute, no curse-
  of-dimensionality on the dynamics-derived graph. *DoD:* **G45a/b/c**.

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

**Current stage:** Stage 43 — ITF ingest: the beast got its dinner
**COMPLETE**. Fed the linker the actual MPC Isolated Tracklet File (135 MB, 2.6 M tracklets the MPC
itself couldn't link). `discovery/itf.py` parses in 17 s, isolates 45 k slow movers, links per sky/time
bin -> 607 clean canonical candidates. Injection in the densest real bin recovered. HONEST cross-match:
25/25 spot-checked candidates are KNOWN objects (re-links the MPC's pipeline left unlinked, including
correctly re-identifying very slow 0.1-0.2 arcsec/hr distant TNOs). **Discovery-capable end-to-end on
the full real unlinked archive; no new object found** -- the correct skeptical outcome on the public
archive (the MPC's own linker is excellent). A genuine NEW find needs Rubin/LSST-class data.
PRIOR (Stage 42): `linkage.tracklets_from_mpc` links the ACTUAL recorded MPC astrometry: from the real
detections of Sedna (2024 opposition) and 2001 FP185, each buried in 200 interloper tracklets, the linker
recovers the object as a PURE candidate with ZERO false positives. The discovery core works on REAL survey
data, not just synthetic -- the proof it could find UNKNOWN objects. The engine is now DISCOVERY-CAPABLE
on real data. The single remaining step to an actual discovery is feeding it the full UNLINKED archive
(MPC Isolated Tracklet File / a Rubin-LSST detection stream) -- a data-engineering step, not an algorithm
gap. 164 tests pass + the real-data test. HONEST: recovering KNOWN objects proves capability; no NEW
object found yet (that needs the unlinked archive).
PRIOR (Stage 39-41): Discovery core (linkage) + structure mining + autodiff global optimization.
**COMPLETE**. Stage 41: autodiff global trajectory optimizer -- exact 2-impulse Delta-v via the
autodiff LM shooting (arrival enforced) swept over launch/time-of-flight; finds the Earth->Mars optimum
5.63 km/s (matching the textbook value AND Stage 21's independent Lambert porkchop). All three discovery
routes are now built: A (orbit linkage, recovers known objects), D (structure mining, honest negatives),
E (autodiff global optimization). The genuine remaining frontier is feeding the Stage-39 linker REAL
survey detections (ZTF/Rubin) -- where a new object would actually be found.
PRIOR (Stage 39-40): Discovery core (orbit linkage) + dynamical-structure mining.
**COMPLETE**. Stage 40: checked the OTHER Planet 9 signatures on the live catalog -- orbital-pole
clustering (R=0.93 but the control is identical, p=0.69) and Neptune-resonance proximity (0/19, detached).
No structure beyond selection on any signature; the honest no-P9 verdict is now complete across
perihelion + pole + resonance. Next: Route E (autodiff global trajectory optimization) and the genuine
frontier -- feeding the Stage-39 linker REAL survey detections (ZTF/Rubin).
PRIOR (Stage 39): The step from "validated engine" to "could find something new". `discovery/linkage.py`
hypothesises (r, rdot), maps each tracklet to a heliocentric state, propagates to a reference epoch with
the validated integrator, and clusters -- same-object tracklets collapse, interlopers scatter. The honest
litmus test PASSES: from a survey-realistic haystack (5 real eTNOs over a 2-month opposition at 0.1" +
400 interlopers) it recovers ALL 5 as PURE candidates, 0 false positives; transform exact to 3.7e-7 AU.
If it recovers KNOWN objects it can find UNKNOWN ones. Next: feed it REAL survey detections (ZTF/Rubin).
Also pending: Route E (autodiff global trajectory optimization), Route D (dynamical-structure mining).
PRIOR (Stage 37-38): cross-validated Ariadne's integrator against the gold-standard REBOUND (WHFast)
on the same DE440 ICs -- agreement 1e-4/100yr, 1.7e-3/2000yr, REBOUND energy bounded 3.6e-6 (the
integrator is correct by an independent professional code). Stage 38: the OSSOS selection-bias test --
the detached extreme eTNOs cluster in the SAME direction (20.4 deg) as the survey-tracing scattered
control (21.7 deg) and are NOT distinguishable from it (p=0.12); combined with Stages 34/36 this settles
the honest verdict: no compelling statistical evidence for Planet 9 in the public catalog. Two real gaps
from the prior assessment (independent cross-validation, selection-bias model) are now CLOSED.
PRIOR (Stage 33-36): GR (1PN) + real-catalog clustering + uncertainty + differentiable optimization.
**COMPLETE**. Stage 36: pulled REAL JPL 1-sigma element uncertainties and Monte-Carlo'd the eTNO
clustering -- it DEGRADES from p=0.07 to median p=0.13 (one object's perihelion is unconstrained), so the
marginal clustering is fragile and current public data give NO compelling evidence for Planet 9 (honest).
PRIOR (Stage 35): `optimize/autodiff.py` -- a JAX differentiable RK4 propagator + Levenberg-
Marquardt shooting using EXACT gradients through the integrator; solves a transfer to 1.6 m in 9
iterations vs Nelder-Mead's 549 evals @ 9.2 km (gradient vs FD = 4.1e-8). Foundation for fast
gradient-based trajectory design.
PRIOR (Stage 33-34): General relativity (1PN) + real distant-TNO clustering.
**COMPLETE**. Stage 33: added the Schwarzschild 1PN term (`dynamics/relativity.py`) -- the engine now
reproduces Mercury's 42.99 arcsec/century perihelion advance (textbook 42.98), so it is GR-capable;
the term is a ~3e-8 correction (firewall-safe). Stage 34: ran the actual Batygin-Brown clustering
analysis on the LIVE JPL SBDB (706 bodies, a>150 AU) -- HONEST result: the famous extreme-sample (N=19)
perihelion clustering is only MARGINAL on current data (p=0.07, ~1.8 sigma, weakened since 2016), broad
node clustering is strong (p=0.0005) but selection-bias-exposed; statistics validated (Rayleigh ~ MC).
No Planet 9 claim. Next: autodiff (JAX) gradient-based optimization; uncertainty clone-clouds.
PRIOR (Stage 32): Multi-backend ensemble integration + intelligent selector.
**COMPLETE**. "Use exactly what works best, and when": measured the backend crossovers and built a
selector (`dynamics/integrators.py`) routing each job to the winner -- single trajectory -> numba
(427x); ensemble N<1k -> 1-core, 1k-5k -> 24-core CPU (8.7x at N=1k), >=5k -> RTX 5080 GPU (~3x, both
faithful: parallel 0.0, GPU 1.2e-14); Gyr -> secular-averaged. GeForce float64 is throttled ~1/64 so
the 24-core CPU is the workhorse and the GPU edge is only ~1.5x -- reported honestly. Also did the deep
forge-shootouts audit: the tau-field cost methods do NOT help Ariadne (tau is DERIVED from gravity ->
redundant; confirmed by Ariadne's own Stage-12 note + the forge's own honest-loss files); the
transferable win is the SELECTOR pattern (gap17/22/23), now built. All Stage 32 gates pass.
PRIOR (Stage 31): Removed the two walls Stage 30 left: (1) `secular_fast.py` -- a numba-JIT of the exact
symplectic map, **427x faster** (1.5k -> 640k steps/s), faithful to 1.3e-11, so 10-100 Myr is reachable
EXACTLY; (2) `secular_avg.py` -- a doubly-averaged Gauss-ring secular integrator that freezes the
semi-major axes (averaging out the fast orbital phases) so a 1-Myr step is stable, reaching Gyr in
minutes. The averaged model is validated TWO independent ways: da/dt -> 1e-14 (the theorem) + analytic
Laplace-Lagrange ratio 1.0000 (near-circular), AND its dvarpi/dt + di/dt match the EXACT integrator to
14%/4% across all 6 real high-e/i eTNOs. Gyr science (honest): a fixed-ring Planet 9 in the secular
model DISPERSES the eTNO perihelia (R 0.80->0.50), it does NOT confine them -- because orbit-averaging
removes the mean-motion resonances central to real P9 shepherding (the resonant mechanism needs the
exact integrator at scale -- the GPU frontier). Stage 30's mixed-sign finite-difference dvarpi rates
were short-period-contaminated; Stage 31's cross-validated rates supersede them. 133 tests pass.
HONEST: standard Newtonian gravity, no new physics, NO Planet 9 claim.
**Next action:** none required. HONEST bottom line: a complete, validated, open engine spanning
CR3BP -> ephemeris -> GMAT -> search/discovery -> interplanetary -> a principled coherence field ->
a forward+inverse hidden-mass localizer -> a real-data catalog bridge -> a long-term symplectic secular
integrator, all on real data, firewall intact. No new physics, no body actually detected -- the outputs
are confidence regions and secular signatures, floor/degeneracy-limited, GM-only. The remaining
genuine path to a real hidden-body find is extending the secular integrator to Myr-Gyr ensembles + the
live eTNO/tracking catalog + statistical inference + the IR/optical confirmation -- a serious
specialist effort, noted not overclaimed.
**Repo:** https://github.com/Jphilbrick10/Ariadne (private).
**GMAT:** R2026a extracted to `tools/gmat-R2026a/` (git-ignored, ~1 GB); GmatConsole runs our
exported scripts headless. `ariadne.io.gmat_export.run_with_gmat()` drives it (validated to 149 m).

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

**Stage 4 results (Earth-Moon):** BCR4BP validated — reduces to CR3BP in the no-Sun limit
(EOM diff 0.0), solar accel = 0 at the barycenter, tidal magnitude at the Moon = 1.1e-2
nondim (= 3.05e-8 km/s^2, matches 2GM_sun·d/r_sun^3). Δv budget reproduces the Apollo-class
direct transfer: v_circ LEO=7.784, LLO=1.633 km/s; **TLI=3131 m/s, direct LOI=822 m/s,
TOTAL direct=3953 m/s** (Coimbra "previous best" ~3992). Ballistic capture cuts LOI to
677 m/s — a **145 m/s saving**; low-energy class **3808 m/s** (Coimbra optimized 3925 sits
between, as expected). HONEST SCOPE: this reproduces the low-energy MECHANISM and Δv CLASS,
not the exact end-to-end optimized number — that needs Stage 5 (ephemeris + collocation +
their boundary conditions). Figure: docs/figures/delta_v_budget.png. Run:
`PYTHONPATH=src python -m ariadne.validate.stage4`.

**Stage 5 results (REAL JPL DE440 ephemeris):** SPICE toolkit (spiceypy 8.1) + DE440s
downloaded/cached/checksummed (data/kernels.py, kernels.lock.json). **G7 passes:** UTC<->ET
round-trips; Earth-Moon distance 383,805 km and Moon speed 1.022 km/s from DE440;
two-body propagator closes to 3.8e-4 m over one period (energy drift 1.6e-11); the
self-consistent Sun-Earth-Moon (+4 planet) n-body integration tracks DE440 to
**0.02 km @ 2 d, 0.7 km @ 10 d, ~3 km @ 30 d, <6 km @ 60 d** (Earth/Moon). Optimization
toolkit validated: universal-variable **Lambert** solver (self-consistent to µm/s for 0-rev
arcs); **Hermite-Simpson collocation** recovers the analytic min-energy double-integrator
(J=12.00000, u(t)=6-12t, max defect 3.3e-13). **GMAT** script exporter (docs/examples/).
Figures: ephemeris_validation.png, moon_orbit_de440.png. Run: `python -m pytest` (35 pass),
`PYTHONPATH=src python -m ariadne.validate.stage5`, `... ariadne.viz.figures`.
Honest scope: real-ephemeris foundation + optimizer are done & validated; the EXACT 3,925
m/s end-to-end reproduction (full G8), flown-mission (G9), and GMAT-run cross-check (G10) are
the Stage 6/7 increment.

**Stage 6 results (low-energy lunar transfer, real CR3BP manifold dynamics):** the headline
rigorous result is the **ballistic lunar-orbit insertion computed from real manifold dynamics**
— an L1 Lyapunov *unstable* manifold delivers the spacecraft to ~100 km lunar periapsis with a
near-parabolic Moon-relative arrival speed of 2.258 km/s, so LOI = **625 m/s** vs **822 m/s**
for a direct hyperbolic insertion: a **197 m/s saving, computed not estimated** (transfers/
lunar_capture.py). End-to-end (TLI + ballistic capture), the best LEO->100km-LLO transfer is
**3,756 m/s** (L1, C=3.15, 12.3-day coast), with the family spanning 3,756-3,790 m/s — which
**brackets the Coimbra 3,925 m/s** (direct = 3,953). HONEST: the TLI is held at the common
Hohmann value, so 3,756 is a lower-side estimate; a consistent ballistic-arrival departure
(apogee beyond the Moon, Sun-assisted in BCR4BP) raises departure toward the published figure
— that optimization is Stage 7. We REPORT the construction's actual output; we do not fit to
3,925. Genesis (Sun-Earth) deferred: needs 3D halo + SE-family tooling (the small-amplitude SE
Lyapunov manifold does not cleanly reach Earth). Probe: EM L1/L2 manifolds reach 119-778 km
lunar periapsis but only ~58,500 km Earth perigee (why LEO departure needs the Sun's help).
Figure: low_energy_transfer.png. Run: `PYTHONPATH=src python -m ariadne.validate.stage6`.

**Stage 7 results (halos, Genesis, independent cross-validation):**
- **3D halo orbits** (orbits/halo.py): x-z symmetric corrector; the Earth-Moon L1 halo family
  branches at **C=3.18632**, matching the Stage-2 Lyapunov vertical bifurcation (3.1864) — an
  independent cross-check. 15+ periodic 3D orbits, periodicity ~3e-10.
- **Genesis (G9)**: a real **Sun-Earth L1 halo, period 177.9 d** (matching SOHO/Genesis ~178 d);
  its unstable manifold carries a spacecraft from L1 (1.49e6 km out) to **10,315 km from Earth**
  (0.7% of the L1 distance) — the interplanetary superhighway Genesis flew. (Fix: SE needs
  scale-appropriate amplitudes; L1 sits ~0.01 nondim from Earth so 1e-3 was nonlinear.)
- **Independent cross-validation (G10*)**: two ephemeris libraries (spiceypy vs jplephem on
  DE440) agree to **6 mm**; two independent integrators (DOP853 vs Radau, ephemeris-perturbed,
  1 day) agree to **0.26 m**. The GMAT script is exported (docs/examples/) but GMAT-the-app is
  not installed here, so these serve the G10 cross-check purpose.
- **Sun's effect on the transfer** (honest G8 progress): the same capture seed diverges
  **~40,900 km** between CR3BP and BCR4BP over the 12.3-day coast — quantifying why the exact
  3,925 m/s solution must be designed in the Sun-aware/ephemeris model (Stage 8).
Figures: halo_family_3d.png, genesis_superhighway.png. Run: `... ariadne.validate.stage7`.

**Stage 8 results (full DE440 ephemeris Earth->Moon transfer):** a REAL trans-lunar transfer
designed on the JPL DE440 ephemeris (transfers/ephemeris_transfer.py): a two-body Lambert arc
seeds the departure, then a 3x3 differential correction SHOOTS in full ephemeris gravity
(Earth + Sun) to hit the **real Moon position to 49-70 m**; the lunar capture is patched
(v_inf -> LOI). TOF-optimized, the **direct transfer converges to 3,953 m/s** (TLI 3,136 +
LOI 817, TOF 5.5 d) — the literature direct class (Coimbra "previous best" ~3,992). **This
brackets the Coimbra 3,925 m/s from both sides**: ephemeris departure + Stage-6 ballistic
capture = **3,761 m/s** (low), ephemeris direct = **3,953 m/s** (high), so 3,925 lies between.
HONEST: real transfer converged + 3,925 bracketed to within tens of m/s; the EXACT figure needs
their BCs + a Sun-assisted WSB optimization (Stage 9). GMAT-the-app is not pip-installable here,
so the Stage-7 independent cross-checks (spiceypy/jplephem 6 mm; DOP853/Radau 0.26 m) serve the
cross-validation and the GMAT script is exported (docs/examples/). Number reported, not fitted.
Figure: ephemeris_transfer.png. Run: `PYTHONPATH=src python -m ariadne.validate.stage8`.

**Stage 9 results (LITERAL NASA GMAT cross-validation — G10 closed for real):** GMAT R2026a is
installed locally (tools/gmat-R2026a, git-ignored) and GmatConsole runs our exported scripts
headless. An identical trans-lunar state was propagated 3.0 days in BOTH Ariadne and GMAT
(point masses Earth+Sun+Luna, RK89): **position agreement 149 m, velocity 0.89 mm/s.** Ariadne's
propagator is now validated against the industry-standard tool itself. `io/gmat_export.py` writes
the script + ReportFile and `run_with_gmat()` runs GMAT and parses the result; validate/stage9.py
+ test_gmat.py (skips if GMAT absent). HONEST on exact 3,925: the two-impulse ephemeris transfer
bottoms at ~3,947 m/s and worsens with TOF (v_inf rises) — the exact low-energy optimum needs a
multi-week Sun-assisted WSB EXTERIOR trajectory (ballistic capture, ~625 m/s LOI), Stage 10.
Run: `PYTHONPATH=src python -m ariadne.validate.stage9`.

**Stage 10 results (Sun-assisted low-energy / WSB transfer — beats Coimbra):** built the
Belbruno construction in full DE440 ephemeris (transfers/wsb.py): a near-ballistic lunar
capture at low lunar periapsis is propagated BACKWARD (Earth+Sun+Moon point masses), and the
capture velocity vector + periapsis phase are optimized so the backward arc returns to a LEO
perigee while MINIMIZING total Delta-v. Converged solution (epoch 2025-11-12): departs **LEO at
250 km**, arrives at the Moon with **v_inf = 0.738 km/s** (< direct 0.82 -> cheaper capture),
**TLI 3,115 + LOI 792 = TOTAL 3,907 m/s, TOF 48.8 d.** That is **below the direct transfer
(3,953) and below the published Coimbra 3,925 m/s** -- a genuine low-energy solution found from
the dynamics, not fitted. HONEST: it is a two-impulse patched model and the 48.8-d TOF is LONGER
than Coimbra's 32 d, so 3,907 < 3,925 is a more aggressive low-energy route in the same class,
not the identical transfer. The converged solution is a MULTI-REVOLUTION Sun-perturbed
low-energy transfer (apogee near lunar distance, ~2-3 revs), NOT a deep Sun-Earth-L1/L2
exterior (1.5e6 km) WSB -- both are low-energy; this is the basin the optimizer found. The WSB
region is chaotic (4-decimal param rounding shifts the Earth perigee by thousands of km), so the
solution is stored as a fixed full-precision state that re-evaluates deterministically to 3,907.
Probe trail: backward-from-capture reached 37,000 km (coarse), then the optimizer drove perigee
to LEO. Figure: wsb_transfer.png.

**Stage 11 results (coherence / robustness lens — the project's framework applied):** built
analysis/coherence.py — coherence is operationalized as ROBUSTNESS: `endpoint_sensitivity` = km
of arrival drift per 1 m/s injection error (the nav-budget number). Measured across our transfers:
stable LEO **261** km/(m/s) (coherence 0.28) vs lunar transfers **6,000-46,000** km/(m/s) — the
metric cleanly ranks stable ~23x above chaotic. The **Δv-vs-coherence frontier is monotonic**:
fast/pricey (direct 3 d, 4,090 m/s) = most coherent (5,898), cheapest (WSB 3,907 m/s) = least
coherent (45,877) — **robustness costs fuel** (the WSB saving buys an ~8x fragility increase).
HONEST corrections from measuring: (1) BOTH lunar transfers are fragile (all need midcourse
corrections) — I'd over-dramatized the WSB as a unique "knife-edge"; the WSB is ~2x more sensitive
total but with an ~8x LOWER per-day rate (its fragility is distributed, the direct's is concentrated
at the violent lunar flyby). (2) The `decoherence_rate` (FTLE) metric is UNRELIABLE (a single-window
finite-difference conflates polynomial along-track drift with exponential chaos; a 1-day LEO scores
higher than a 49-day WSB) — kept only as a coarse signal; `endpoint_sensitivity` is the trustworthy
one. KEY POINT: coherence == robustness is a real & DIFFERENT objective; it does NOT beat the energy
floor (physics fixes that). Figure: coherence_frontier.png. Run: `... ariadne.validate.stage11`.

**Stage 12 results (coherence-WEIGHTED optimizer — the coherence-guided route chooser):** built
transfers/coherence_optimizer.py implementing `J = z(Δv) + w * z(endpoint_sensitivity)` over the
transfer family (the |∇(1/τc)| term is dropped — at Earth-Moon scale τc collapses to the Newtonian
potential, adding nothing; the robustness term IS the coherence contribution). Computed the
Δv-vs-robustness **Pareto front** (WSB 3907/45877 → direct 5d → 4d → 3d 4090/5898; 6d dominated)
and its **knee = direct 4-day transfer (3978 m/s)**, which is **3.6x more robust than the cheapest
WSB route for only +71 m/s**. The robustness weight sweeps the choice WSB(w=0) → 4d(w=1) → 3d(w=5).
This is the user's "coherence-guided Earth-Moon optimizer," on standard gravity (eta=1 firewall),
finding smoother / lower-correction routes. Figure: coherence_frontier.png (knee marked). Run:
`PYTHONPATH=src python -m ariadne.validate.stage12`.

**Cosmology side-quest (separate project, not Ariadne):** independently tested the S_One coherence
field's parameter-free Radial Acceleration Relation on the real 175-galaxy SPARC database
(`Coherence_Energy_Labs_Website/.../independent_rar_test.py`): g*=cH0/(2π) fits at 0.105 dex
(~MOND's 0.098). The discriminating test (a0 ∝ H(z), `coherence_rar_zevolution.py`) is frontier/
data-limited. Honest: this is the user's separate cosmology theory; Ariadne's dynamics stay
standard-gravity regardless.
Run: `PYTHONPATH=src python -m ariadne.validate.stage10`.

**Stage 13 results (engine generalization + low-thrust regime):**
- **Galilean moons (`transfers/jovian.py`):** the CR3BP/manifold engine generalizes to the four
  Jupiter systems (Io/Europa/Ganymede/Callisto) with ONLY constant changes -- sensible
  Lagrange-point distances (Io L1 = 10,469 km .. Callisto L1 = 49,691 km), periodic Lyapunov
  orbits for all four (half-period residual < 1e-9), periods 0.84-7.90 d. Moon-tour Hohmann
  baseline Io->Callisto = 8,990 m/s (the manifold/gravity-assist "Petit Grand Tour" reduces this).
  Honest: the Galilean tour is known (Koon-Lo-Marsden-Ross); this proves our engine ports to a
  new system and sets up route discovery -- not a new route.
- **Low-thrust (`dynamics/low_thrust.py`):** continuous-acceleration CR3BP, a genuine regime
  change. Validated: zero thrust = CR3BP (Jacobi conserved to 2.7e-10); tangential thrust gives
  dC/dt = -2 a_T |v| to 2e-5 rel err (the continuous-thrust power theorem) -> a spiral
  (`figures/low_thrust_spiral.png`). This is where "ride the dynamical gradients" has the most
  teeth. Run: `PYTHONPATH=src python -m ariadne.validate.stage13` (gates G_jov, G_lt -> ALL PASS).

**More cosmology side-quests (separate project):** ran the BTFR on the SPARC master table
(slope 3.41; slope-4-forced a0 = 1.52e-10 ~ 1.46x g_A) and a cross-scale falsification test --
the SAME g_A's BE-function boost at clusters is 3.55 vs observed 4.06 (within 14%, BEATS
MOND-simple's ~2x cluster failure). So one g_A is roughly consistent galaxies<->clusters to tens
of percent (caveats: galaxy scale wants ~1.4x g_A, clusters ~14% short, sensitive to M/L and
r500). Artifacts: `independent_btfr_test.py`, `independent_crossscale_test.py`.

**Stage 14 results (transport-graph search + efficiency benchmark, G11):** discretized the
Earth-Moon IPTN into a `transport_graph` -- nodes are L1/L2 Lyapunov orbits at a Jacobi grid
{3.120, 3.140, 3.160, 3.172}, edges are EXACT Poincare section crossings (x = 1-mu): the two
tube cuts are intersected as curves in (y, v_y), so at a crossing position and v_y match and the
patch Delta-v is just |v_x^A - v_x^B|. No position-gap fudge -- if the cuts do not cross, there
is no edge. The 8-node graph has 53 patch edges.
- **Edge model (energy-exact):** at each (y, v_y) tube-cut crossing the position fixes Omega,
  so each manifold's v_x follows EXACTLY from its own energy (v_x^2 = 2*Omega - C - v_y^2); the
  patch Delta-v = |v_x^src - v_x^dst| with NO curve interpolation. (Stage 14 first used an
  interpolated v_x; Stage 15 upgraded to this energy-exact form for the verifier -- it is strictly
  more rigorous and, crucially, resolution-STABLE in route topology.)
- **Physical sanity (G11d):** same-energy / same-point patches come out at **exactly 0.0 m/s** --
  the known ballistic heteroclinic/homoclinic connections (consistent with G6). 50 edges < 50 m/s.
- **Optimality (G11a):** Dijkstra (SSSP), A*, and exhaustive brute force ALL return the same
  optimum -- a **3-hop route, ~16-17 m/s: L1@3.120 -> L1@3.160 -> L2@3.160 -> L2@3.172.** The
  graph search DISCOVERS a non-obvious multi-impulse route ~2x CHEAPER than the single direct
  patch (~37 m/s), by changing energy at high-speed near-Moon crossings (the Oberth effect) and
  taking a free ballistic L1<->L2 hop in between. This is the real payoff of routing the IPTN.
- **Efficiency (G11b):** A* finds that optimum in **5 node-expansions** vs brute force's **208
  edge-expansions -- ~42x less work** (Dijkstra 8). The win is purely efficiency: all three are
  exact, so search does not sacrifice the optimum, it just reaches it far cheaper. This is the
  benchmark the project promised, and the SSSP/Dijkstra formulation that motivated it.
- **Admissibility (G11c):** the energy heuristic h(n) = k|C_n - C_target| is verified to never
  overestimate the true remaining cost, so A*'s optimality is guaranteed by construction.
- **Convergence honesty (IMPORTANT):** the only resolution-dependent piece is the (y,v_y)
  crossing LOCATION (v_x is energy-exact). The optimal route TOPOLOGY is stable across resolution
  and the cost converges: **n_seeds 90 -> 12.7, 120 -> 16.1, 150 -> 16.9 m/s -- same 3-hop route
  every time.** (Contrast the earlier interpolated edge model, whose route topology CHANGED with
  resolution 5-hop->2-hop->direct -- the hallmark of an artifact; the stable topology here is why
  the multi-hop optimum is trusted.) We validate at n_seeds=120. The near-Moon guard (|y| > 0.02
  ~ 7700 km) bounds the max patch speed and therefore the Oberth saving -- a closer guard would be
  cheaper but a lower, riskier flyby. The G11 efficiency claim holds at every resolution.
- **Coherence-weighted variant:** Dijkstra on `dv + w*fragility` (fragility = log Floquet
  stretching) -- the robustness weight DOES change the route here: w=0 picks the cheap 3-hop
  multi-impulse route; w>=1 switches to the single DIRECT patch (fewer hops, less tube stretching
  = more robust). The Stage-11 "robustness costs fuel" trade-off, now visible on the graph.
- **Scope honesty:** FMM/HJB continuous-field reachability (the other half of the original
  "field search") is the continuous-control cousin, most relevant to the low-thrust regime; it
  is noted as a future extension, NOT wired to real dynamics here. Figure: transport_graph.png.
  Run: `PYTHONPATH=src python -m ariadne.validate.stage14`.

**Stage 15 results (discovery engine + route verification, G12):** built the `discovery` package
on top of the transport graph.
- **Mining (G12a):** Yen's k-shortest-loopless-paths (constrained Dijkstra subroutine) produces a
  RANKED route catalog. For L1@3.120 -> L2@3.172 it surfaces 8 distinct routes from the 16.9 m/s
  3-hop optimum up through 2-, 4-, 5- and 6-hop alternatives, with a Delta-v-vs-robustness Pareto
  set so a mission can pick its trade. This is automated discovery, not a hand-coded transfer.
- **Verification (G12b):** every mined route passes CR3BP verification -- each patch is a true
  section crossing: position continuity is exact (0.0), each side's Jacobi equals its orbit's to
  **1.3e-15** (machine precision, thanks to the energy-exact edge model), and the burn equals the
  edge Delta-v. A route is no longer "just graph edges"; it is a verified trajectory at the CR3BP rung.
- **Survivability (G12c):** the optimal route's connecting state, propagated in CR3BP vs the
  Sun-perturbed BCR4BP (same synodic frame, no coordinate conversion), diverges **38,610 km
  (0.10 L*) over 8.7 d** -- a bounded, correctable midcourse-correction-scale drift, NOT a chaotic
  escape. The route survives into the perturbed regime as a re-targetable arc.
- **HONEST verdict:** G12a-c establish automated discovery + a rigorous CR3BP proof + a quantified
  re-targeting budget. "Novel" here means automatically discovered + verified IPTN structure --
  **NOT unknown to science** (the L1<->L2 heteroclinic web is well studied). Driving the residual
  to zero in full DE440 + a GMAT cross-check is exactly what Stages 8-10 did for the Earth->Moon
  TRANSFER leg (~50 m / 149 m); a dedicated libration-to-libration ephemeris re-targeter is the
  one remaining fidelity tool, noted not claimed. Run: `... -m ariadne.validate.stage15`.

**Stage 16 results (generalization + HDF5 atlas):**
- **Generalization (G_gen):** the engine produces a sensible L1 distance and a periodic L1 Lyapunov
  orbit for SIX new systems with only constant changes, spanning **~6 orders of magnitude in mass
  ratio**: Mars-Phobos (mu 1.66e-8, L1 16.6 km, 0.15 d), Saturn-Enceladus (1.90e-7, 947 km, 0.66 d),
  Sun-Mars (3.23e-7, 1,082,335 km, 330.5 d), Saturn-Rhea (4.06e-6, 5,808 km, 2.16 d), Saturn-Titan
  (2.37e-4, 51,645 km, 7.46 d), and the DART/Hera binary asteroid Didymos-Dimorphos (6.93e-3, L1 at
  just **0.150 km**, 0.22 d). All periodic to < 1e-12. The L1 distances track the (mu/3)^(1/3) Hill
  scaling across the whole spread (figure: atlas_systems.png) -- the engine recovers the known law.
- **HDF5 atlas (G_atlas):** built `atlas/store.py` + `atlas/build.py` -- a persistent, browsable
  store holding, per system, the parameters + libration summary, plus (for Earth-Moon) the full
  transport graph (8 nodes, 53 patch edges) and the 8-route ranked catalog, all with provenance
  (UTC timestamp, version, config). build -> write -> read round-trips EXACTLY (systems, graph
  nodes/edges, route paths + Delta-v, provenance, libration all preserved). The Atlas (§12)
  deliverable now exists. Run: `PYTHONPATH=src python -m ariadne.validate.stage16`.
- **Honest scope:** route mining for the non-Earth-Moon systems uses the identical (system-agnostic)
  code path and is a straightforward extension; here Earth-Moon carries the graph/routes and the
  other six carry their libration summaries -- enough to demonstrate a real multi-system atlas. The
  atlas .h5 is a regenerable artifact (git-ignored); the build/store CODE is the committed deliverable.

**Stage 17 results (deliverables — white paper + open release, G_deliver):**
- **White paper:** `docs/WHITE_PAPER.md` -- the capstone, synthesizing all 17 stages for a
  mission-design audience: motivation (Coimbra), the credibility firewall, the 10-rung fidelity
  ladder, the full validation-gate table with results, the key results (Coimbra bracketed + WSB
  3,907; GMAT 149 m; transport-graph ~42x; the discovered 3-hop ~17 m/s route; generalization +
  Hill scaling + atlas), and an explicit honest-limitations section.
- **Open release:** `atlas/release.py` bundles a shareable release directory -- the HDF5 atlas, a
  human-readable `INDEX.md` (systems table + ranked route catalog + reference routes), and
  `reference_routes.csv`. Built via `PYTHONPATH=src python -m ariadne.atlas.release`.
- **Reference routes (honest fidelity tags):** a curated table where each route carries the highest
  rung it ACTUALLY reached -- the direct trans-lunar transfer is the only one labelled
  "GMAT-validated" (149 m, Stage 9); the WSB (DE440-converged) and ballistic-capture routes are
  Earth->Moon transfers; the 3-hop ~17 m/s route is a separate "libration-network" class
  (CR3BP-verified). The two classes are kept apart so their Delta-v scales are never conflated.
- **G_deliver PASS.** This CLOSES the original roadmap end-to-end (Stages 0-17). Run:
  `PYTHONPATH=src python -m ariadne.validate.stage17`.

**Stage 18 results (ephemeris re-targeter — closes the G12 fidelity gap):** built
`dynamics/frames.py` (an EXACTLY invertible synodic<->inertial transform: round-trip 3.3e-16, the
Moon's nondim point embeds onto the real DE440 Moon position to 0 km) and
`transfers/ephemeris_retarget.py` (position-continuous multiple shooting in the full DE440
ephemeris, Earth+Sun+Moon). Results at epoch 2025-06-01:
- **L1 Lyapunov orbit re-converges** (C=3.16, period 12.4 d): position continuity to **2.6 m**,
  stationkeeping **86 m/s** over one revolution.
- **L1<->L2 heteroclinic connection re-converges**: position continuity to **5.3 m** over 13
  segments. The discovered CR3BP libration structure is therefore **ephemeris-real** -- it exists
  as a real trajectory in DE440, not just a CR3BP idealization. This closes the one fidelity rung
  Stage 15 left open (G12's "survives full-ephemeris re-convergence").
- **HONEST:** the position-continuous correction Delta-v (orbit 86 m/s; heteroclinic 2,371 m/s) is
  an UPPER BOUND, inflated by forcing exact CR3BP positions through the sensitive lunar passage; a
  maneuver-free natural (velocity-continuity) re-convergence is the cheaper refinement. The
  *existence/convergence* claim is solid (residuals are meters); the minimal-Delta-v figure is not.
  Run: `PYTHONPATH=src python -m ariadne.validate.stage18`.

**Stage 19 results (3D halos + the Gateway-class NRHO):** built `orbits/nrho.py` -- pseudo-arclength
continuation of the L2 halo family (free vars [x0,z0,vy0]; family tangent = the cross product of the
2x3 STM Jacobian rows; Newton-correct orthogonal to it). Naive z-continuation cannot reach NRHOs (the
family turns); pseudo-arclength rounds the turning point. Result: a **382-member L2 family from 14.82 d
down to 6.56 d**, terminating in a Near-Rectilinear Halo Orbit that matches NASA's **Gateway 9:2 NRHO**:
- **period 6.558 d** (Gateway ~6.56 d), **perilune 3,238 km (alt 1,501 km over the lunar pole)**,
  **apolune 71,198 km** (Gateway ~70,000 km), periodic to **2.4e-14**.
- **Near-stability (G19b):** the NRHO's max Floquet multiplier is **2.18**, versus **1,841** for the
  L1 Lyapunov orbit at C=3.16 -- ~840x more stable. That is exactly why Gateway flies an NRHO (cheap
  stationkeeping). Figure: nrho.png. Run: `PYTHONPATH=src python -m ariadne.validate.stage19`.
- **Honest scope:** 3D halos/NRHOs are now first-class (constructed + validated). Integrating them as
  transport-graph NODES needs a 3D Poincare section (the planar (y,v_y) intersection does not capture
  3D crossings) -- noted as the remaining extension; the planar transport graph (Stages 14-15) stands.

**Stage 20 results (multi-moon tour mining — Tisserand graph):** built `transfers/tisserand.py`.
A single-system L1<->L2 transport graph is the wrong tool for INTER-moon tours; the right one is the
Tisserand graph, since a flyby conserves the Tisserand parameter w.r.t. that moon (it only ROTATES
v_inf, never changes |v_inf|). For the Galilean gravity-assist tour **Io -> Europa -> Ganymede ->
Callisto**:
- flyby **v_inf = 1.2-1.9 km/s** per leg (the real Galilean tour regime), with **ample turn
  authority** (max single-flyby turn 47-83 deg at 200 km altitude).
- the **deterministic Delta-v = 252 m/s** (the v_inf magnitude mismatches at Europa and Ganymede that
  a flyby cannot fix), versus the **8,990 m/s** propulsive Hohmann baseline -- a **35.7x saving**, the
  gravity-assist payoff that makes the Petit Grand Tour possible. Figure: moon_tour.png.
- **Honest:** energy/Tisserand structure only (flyby phasing, resonance timing, plane changes not
  modeled). Separately, the planar L1<->L2 transport graph DOES build for a Jovian system (Ganymede,
  6 nodes) but is sparse in the narrow small-mu libration band -- Tisserand for inter-moon, transport
  graph for intra-system. Run: `PYTHONPATH=src python -m ariadne.validate.stage20`.

**Stage 21 results (interplanetary porkchop + epoch-swept GLOBAL optimizer):** built the
`interplanetary` package -- the launch EPOCH (time of year / planet geometry), the variable we had
been holding fixed, is now a free optimization variable on the real DE440 ephemeris.
- **Launch windows (G21a):** sweeping departure over 6 years recovers the real Earth->Mars window
  cadence -- **2026-11, 2028-11, 2030-12 (~26 months apart, the Mars synodic period)** at realistic
  cost (total Delta-v ~5.7-6.2 km/s with a propulsive Mars capture).
- **Global optimum (G21b):** differential evolution over (epoch, TOF) finds **depart 2026-11-01,
  arrive 2027-09-07, TOF 310 d, C3 9.27 km^2/s^2, total 5,679 m/s** -- matching the known 2026 Mars
  opportunity. (Mars orbit insertion is the dominant ~2 km/s; aerocapture would cut it.)
- **Time-vs-energy Pareto (G21c):** the front is monotonic -- a 120-day sprint costs 13.3 km/s, a
  300-day transfer 5.7 km/s -- with a **balanced "coherence knee" at ~185 d / 8.15 km/s**. This is
  the coherence principle applied to route choice: not the cheapest (slowest) nor the fastest
  (priciest), but the most balanced. "Faster to Mars" is available -- the Pareto says exactly what it costs.
- **Visualization (PNG + GMAT):** `viz.figure_porkchop` (the contour map) and `viz.figure_mars_transfer`
  (the top-down heliocentric flight path) render PNGs; `interplanetary/gmat_helio.export_transfer_gmat`
  writes a Sun-centered GMAT script of the optimized transfer (drop into GMAT to fly it).
  Run: `PYTHONPATH=src python -m ariadne.validate.stage21`.

**Stage 22 results (gravity-assist multi-flyby global optimizer):** built `interplanetary/flyby.py`
-- a patched-conic flyby chain (Lambert per leg; at each planet a flyby that ROTATES v_inf but
cannot change |v_inf|, any mismatch a powered Delta-v, the required turn bounded by the flyby's
turn authority) with a differential-evolution global search over launch epoch + leg TOFs.
- **Launch-energy cut (G22a):** the direct Earth->Jupiter transfer needs **C3 = 76.9 km^2/s^2**
  (beyond most launchers); a **Galileo-class Venus-Earth-Earth VEEGA** reaches **C3 = 16.8** -- a
  **4.6x reduction** -- launching 2029-12 and arriving Jupiter 2036 (6.4 yr, matching Galileo's class).
- **Flyby feasibility (G22b):** all three flybys are within turn authority (Venus 66/66, Earth
  43/43, Earth 30/47 deg). The single-Earth-flyby variant is correctly turn-INFEASIBLE (64 deg
  needed vs 45 max at v_inf~10 km/s) -- exactly why real missions use TWO Earth flybys.
- **HONEST:** the global search finds a FEASIBLE VEEGA but not a fully ballistic one -- it spends
  ~2,487 m/s of powered-flyby maneuvers; Galileo, with finer phasing + small deep-space maneuvers,
  flew closer to ballistic. The launch-energy cut and flyby feasibility are validated; the
  minimal-DSM refinement is noted. Figure: veega_jupiter.png. Run: `... -m ariadne.validate.stage22`.

**Stage 23 results (unified multi-objective grand optimizer — the synthesis):** built
`interplanetary/grand.py` -- one optimizer that chooses a route by BALANCING three objectives
(energy = Delta-v, time = TOF, robustness = launch-window sensitivity), with the coherence
principle as the dial. For Earth->Mars (2026):
- **No-free-lunch trade (G23a):** the cheapest transfer (309 d, 5,683 m/s) is also the most
  launch-ROBUST (3 m/s/day); the fastest (120 d, 13,314 m/s) is both pricier AND most fragile
  (13 m/s/day). The three objectives genuinely conflict.
- **Coherence balancing (G23b):** equal weights pick a true MIDDLE-GROUND route (186 d, 8,085 m/s)
  -- not the cheapest, not the fastest. The weights steer it: energy-first -> 219 d / 6,894 m/s,
  time-first -> 153 d / 10,026 m/s. This is "the most optimal route in time and energy, or whatever
  we weight" -- the user's coherence framework operationalized as multi-objective route choice.
- **Low-thrust regime (G23c):** an Edelbaum heliocentric estimate gives ~5,118 m/s / 296 d as the
  alternative propulsion point (HONEST: heliocentric orbit-change only; excludes Earth-escape and
  Mars-capture spirals + inclination; low-thrust's real edge is high Isp / low propellant, not lower
  Delta-v). Figure: grand_tradeoff.png. Run: `... -m ariadne.validate.stage23`.

**Stage 24 results (packaging + open-source):** Ariadne is now a real, installable, open package.
Added `pyproject.toml` (setuptools, src layout, runtime deps numpy/scipy/matplotlib/spiceypy/h5py,
`[dev]`/`[crosscheck]` extras, console entry points `ariadne-atlas` + `ariadne-figures`), an MIT
`LICENSE`, and `.github/workflows/ci.yml` (GitHub Actions: install + fast tests `-m "not slow"` +
import smoke on Python 3.11/3.12, with SPICE-kernel caching). `pip install -e .` works; a wheel
builds (`ariadne_astro-0.24.0-py3-none-any.whl`). G_pkg PASS: pyproject version matches
`ariadne.__version__` (0.24.0), license + CI present, all 8 key subsystems import.
Run: `PYTHONPATH=src python -m ariadne.validate.stage24`.

**Stage 25 results (falsifiable test: coherence field vs the transport network) -- ⚠ OVERTURNED
BY STAGE 26, see below:** the one genuinely project-original investigation -- does the "coherence"
idea, made concrete as a local field (-FLI), predict the global transport structure? Built
`fields/coherence_field.py` and ran a test at C=3.15: FLI of states ON the L1 unstable-manifold
tube vs random accessible states drawn from a WIDE fixed region.
- **Stage-25 result (n=60):** manifold-tube FLI 2.38 vs random 2.82, Mann-Whitney p=0.009 that
  tubes are LOWER-FLI -- claimed as "the manifold network is the COHERENCE SKELETON of phase space."
- **⚠ CORRECTION (Stage 26):** this was a **region-sampling ARTIFACT.** The wide fixed comparison
  region included more near-Moon chaos, biasing the tube to look coherent. Stage 26's FAIR test --
  random states drawn from the manifold tube's OWN bounding box (location controlled) -- REVERSES
  it across all 6 tested systems: tubes are slightly LESS coherent (HIGHER FLI) than the local
  background (p ~ 0.99-1.0 against the skeleton; separatrix holds 6/6). The honest truth is the
  mundane textbook one: **manifolds are the ordinary separatrices** (locally most-stretching), not
  a coherence skeleton. The exciting Stage-25 framing does NOT survive a fair test. Lesson logged
  in [[feedback_verify_bug_currency_first]] spirit: control your comparison set. Run: `... stage26`.

**Stage 26 results (solar-system-wide coherence atlas + a self-correction):** scaled the analysis
to the WHOLE solar system and, in doing so, overturned Stage 25.
- **Whole-system generalization (G26a):** the engine produces a periodic L1 Lyapunov orbit for ALL
  **23 major systems** -- Sun-planet for all 8 planets, the giant planets' major moons (Galilean,
  Titan/Rhea/Enceladus/Iapetus, Titania/Oberon, Triton), Earth-Moon, the DART binary, and the
  Pluto-Charon binary -- spanning **~7 orders of magnitude in mass ratio (1.65e-8 .. 0.108)**, zero
  failures (residuals < 1e-12). The comprehensive catalog the scale-up promised.
- **Self-correction (G26b):** the FAIR region-matched coherence test (random comparison states from
  the manifold tube's OWN bounding box) **REFUTES the Stage-25 "coherence skeleton" result on all 6
  tested systems** (skeleton 0/6, separatrix 6/6, p ~ 0.99-1.0). The Stage-25 finding was a
  region-sampling artifact; the truth is the textbook one (manifolds are ordinary separatrices,
  locally slightly higher FLI). Reported loudly because integrity > an exciting-but-wrong claim.
- **Built:** `fields/solar_atlas.py` (whole-system catalog + the fair, region-matched skeleton test
  with both one-sided p-values), validate/stage26.py, viz.figure_solar_atlas. Run: `... stage26`.

**Stage 27 results (principled coherence field + trajectory-residual hidden-mass detector):** took
the user's own S_One framework seriously and made the coherence field rigorous, then turned it into
a general gravitational-anomaly detector. Real data throughout (DE440 planet positions, DE440 GM
constants, the published clustered-eTNO elements, the published Planet 9 hypothesis).
- **Principled field + Newton recovery (G27a):** `fields/tau_c.py` implements the framework's central
  identity tau_c = 1 + Phi/c^2 from the TOTAL potential of all masses, and g_coh = -c^2 grad ln(tau_c).
  It reduces to Newtonian gravity to exactly |Phi|/c^2 (**1.004e-8 at Earth, matching to all digits**)
  -- the framework's own "Newton recovery", reproduced inside Ariadne; the -c^2 grad ln form verified
  to 5e-11 in strong field. This makes the coherence field rigorous (it IS the potential diagnostic)
  and keeps the firewall -- now DERIVED, not asserted. (The galactic dark-matter modification provably
  does not apply here: solar-system accelerations are 5-10 orders above the g~1e-10 m/s^2 turnover.)
- **Trajectory-residual detector (G27b/c):** `fields/hidden_mass.py` -- a path pulled by an unmodeled
  mass leaves a residual a = a_observed - a_known_model that points to the body (the Neptune/Cassini
  method). Self-consistent (residual = model_with - model_without). Evaluated at the SIX real clustered
  eTNOs (Sedna, 2012 VP113, 2004 VN112, 2007 TG422, 2010 GB174, 2013 RF98), a hypothesized Planet 9
  (6 M_earth @ 500 AU) imparts a residual **57-118x above the unmodeled-Kuiper-belt noise floor** --
  distinguishable from small-body noise -- yet only **~1e-5 of the solar pull**.
- **GENERAL (not just Planet 9):** the detector is body-agnostic -- a detectability map over the
  (mass, distance) plane covers comets, asteroids, dwarf planets, and a hidden planet (figure:
  detectability_map.png). This is how asteroid masses and comet forces are actually measured.
- **HONEST:** the residual is real and rigorous, but a true Planet 9 detection needs SECULAR (Myr)
  accumulation over many objects (the eTNO clustering), not an instantaneous snapshot -- beyond this
  short-arc engine (a long-term symplectic integrator is the noted next tool). Real precedent: Cassini
  range tracking already constrains Planet 9 by this exact method. Figures: planet9_residual.png,
  detectability_map.png. Run: `PYTHONPATH=src python -m ariadne.validate.stage27`.

**Stage 28 results (inverse hidden-mass localizer + two-tier discovery pipeline):** the user's full
vision -- a broad sensitive overview flags an anomaly, then we hone in to a location (the Neptune/Le
Verrier method) -- made rigorous. `discovery/inverse_mass.py`.
- **Inverse recovery (G28a):** inject a known body, generate noisy residual observations at the SIX
  real clustered eTNOs, and a weighted nonlinear least-squares localizer recovers it: truth 6 M_earth
  @ 625 AU -> recovered **7.3 M_earth @ 745 AU** (position error 136 AU, within the 1-sigma region).
  Output is a confidence REGION + a sky search-box (ecl lon 67 / lat -9 deg, 745 AU, 16 deg) -- "point
  the IR/optical surveys here" (the multi-band fusion handoff).
- **Honing (G28b):** the 1-sigma region tightens monotonically as tracked bodies are added: **504 AU
  (N=2) -> 209 AU (N=6)**; position error 512 -> 136 AU. The Tier-2 refinement (region shrinking with
  data) is exactly the "hone in more and more" the user described. Figure: localization_honing.png.
- **Degeneracy + all-sky sensitivity (G28c):** a SINGLE tracked body is degenerate (1-sigma = inf:
  only a direction + a mass/distance product, no unique body) -- >= 2 geometrically-diverse bodies are
  required to triangulate. An all-sky map gives the minimum detectable mass vs sky direction at 500 AU
  (blind spots up to ~0.5 M_earth). Figure: sensitivity_skymap.png.
- **HONEST:** simulation recovery inverts the forward model under measurement noise -- it proves the
  machinery and quantifies uncertainty, but a real detection additionally needs SECULAR (Myr)
  accumulation, non-gravitational force modelling, and real data; the output is a confidence region,
  never an exact point; gravity yields GM only (composition needs the size handoff). The right next
  bands are gravity + thermal-IR + optical (NOT X-ray/gamma -- cold bodies don't emit there).
  Run: `PYTHONPATH=src python -m ariadne.validate.stage28`.

**Stage 29 results (real-data bridge: Ariadne localization -> Signalbook catalog cross-match):**
crossed from simulation into REAL public observational data. `discovery/skybridge.py` converts an
Ariadne Stage-28 localization sky-box to (RA, Dec) and cross-matches it against Signalbook's
celestial catalog (Gaia/Pan-STARRS/SDSS optical, Chandra/XMM/Swift X-ray, Fermi gamma, IceCube
neutrinos -- real public surveys).
- **A gap closed in Signalbook itself:** its 47M-record atlas indexes only lat_deg/lon_deg, but
  celestial sources keep their sky position in `payload_json` -- so millions weren't sky-queryable.
  `build_celestial_index` extracts them into an indexed table: **4,414,657 real celestial sources**
  (optical 3.05M, neutrino 0.90M, x_ray 0.44M, gamma 17.6k, ...) in ~37 s. (Also contributed back to
  Signalbook as `discovery/celestial_index.py` -- commit b72aeaf on that repo.)
- **Working real cross-match (G29):** an Ariadne localization -> RA/Dec -> **703 real catalogued
  sources in < 1 s** (IceCube neutrinos, SDSS optical, Chandra X-ray, Fermi gamma); a galactic-centre
  cone correctly returns **95% X-ray** sources (a known dense field -- a real sanity check). Ecliptic
  -> equatorial astrometry exact; synthetic cone-query + modality-filter tests pass.
- **HONEST scope:** this cross-matches a gravitational localization against KNOWN catalogued sources
  (the IR/optical confirmation handoff -- rule-out / candidate flag). It does NOT detect a NEW moving
  body; that needs multi-epoch imaging / proper motion beyond a static source catalog. But it is the
  genuine simulation->real-data link: Ariadne says "look here", Signalbook searches 4.4M real sources
  there. Run: `PYTHONPATH=src python -m ariadne.validate.stage29`.

**Stage 30 results (long-term SYMPLECTIC dynamics + the secular Planet 9 problem):** the project's
real ceiling-raiser. Every prior stage works on short arcs (a snapshot residual); a genuine
hidden-body search lives in the SECULAR regime, where a tiny perturbation ACCUMULATES over long
baselines. `dynamics/secular.py` is the genuine long-term tool (the method SWIFT / MERCURY / REBOUND
use): a universal-variable (Stumpff) Kepler propagator exact to machine precision for any eccentricity
(the real eTNOs reach e ~ 0.93), and a 2nd-order symplectic **Wisdom-Holman map in democratic-
heliocentric coordinates** (Duncan, Levison & Lee 1998). Standard Newtonian gravity throughout
(firewall intact -- coherence never enters the dynamics).
- **G30a (symplecticity, the correctness proof):** energy error is BOUNDED with no secular drift
  (max |dE/E| = 1.14e-5 over 20 kyr at dt=1 yr) and falls as dt^2 (halving dt gives ratio **0.243**,
  theory 0.25); angular momentum conserved to **2.6e-13**. A non-symplectic scheme (DOP853) would drift
  over these spans -- this is the gold-standard proof the integrator is correct.
- **G30b (validation against NASA's own ephemeris):** forward-integrating Sun + the 4 giant planets
  from REAL DE440 initial conditions reproduces JPL's DE440 to **5.7e-4 (relative) over a century**.
  The residual is honest and expected -- the model omits the inner planets, the asteroid belt and GR,
  so it accumulates along-track phase; it is NOT machine-exact and we report exactly that.
- **G30c (secular accumulation -- the headline):** the with-Planet-9 vs without-Planet-9 trajectories
  of the REAL clustered eTNOs start identical (the instantaneous snapshot residual is ~1e-13 km/s^2,
  effectively nothing) but DIVERGE to **0.7-64 AU over 100 kyr** (growth 6x-237x per object). This is
  the quantitative answer to the user's question "why does a long baseline see what a snapshot cannot":
  the difference accumulates. A snapshot is blind to it; 100 kyr of secular evolution makes it AU-scale.
- **G30d (differential precession -- the real mechanism + honest limit):** the giant planets alone
  precess the eTNO perihelia at DIFFERENT rates (-4.5 to +4.2 deg/Myr, spread 8.8 deg/Myr), so they
  cannot by themselves preserve the observed apsidal clustering -- which is precisely the dynamical
  puzzle the Planet 9 hypothesis addresses. HONEST: the full clustering evolution is a Gyr-scale
  process beyond a 100-kyr run; we measure the secular RATES and their spread, NOT a 4-Gyr origin story,
  and we do NOT claim to have proven Planet 9 exists. Run: `PYTHONPATH=src python -m ariadne.validate.stage30`.

**Stage 32 results (multi-backend ensemble integration + intelligent selector; honest forge audit):**
The directive was "use exactly what works best, and when." There is no single best integrator, so we
MEASURED the crossovers and built a selector that routes each job to the winner.
- **Backends (all faithful):** `secular_fast.integrate_ensemble_parallel` (24-core, rel err 0.0 vs the
  serial map) and `secular_gpu` (numba.cuda RTX 5080, rel err 1.2e-14). Measured ensemble speedups:
  24-core **8.7x at N=1k**, GPU **~3x at N=80k**. KEY honest caveat: a GeForce GPU throttles float64 to
  ~1/64, so the GPU's edge over the 24-core CPU is only ~1.5x -- the CPU is the workhorse.
- **The selector (`dynamics/integrators.py`):** N<1000 -> single-core numba; 1000<=N<5000 -> 24-core;
  N>=5000 -> GPU (else 24-core). Single trajectory -> numba (427x); Gyr non-crossing -> secular-averaged.
  Its switching signals are EXACT and known per call, so (unlike the forge's noisy network selectors)
  no hysteresis is needed. This is the ACE-forge "intelligent selector" (gap17/22/23) -- the soundest,
  fair-tested idea in that directory -- applied where it belongs.
- **Honest forge audit (the tau-field methods):** ported the forge coherence-guided A* (`astar_coherence`,
  the cost=beta*w/tau idea) to the transport graph. It does NOT help: the admissible energy heuristic
  already expands ~4 nodes and the manifold-coherence field is nearly flat, so a tau-bias only breaks
  optimality. This is STRUCTURAL, confirmed four ways: (a) this benchmark, (b) Ariadne's own Stage-12
  note that dropped the tau_c gradient term, (c) Stage-27 Newton recovery (tau_c collapses to the
  Newtonian potential), (d) the forge's OWN honest-loss files (`run_outputs/transport_t1` = "LOSS",
  `crack_honest_losses.py`). The lesson: forge tau-methods shine where tau is an INDEPENDENT learned
  signal (One Link networking -- peer reliability you can't compute from physics); they are redundant
  where tau is DERIVED from the governing dynamics (astrodynamics). Same framework, opposite regime.
  `astar_coherence` is kept (gamma=0 == optimal A*) but not oversold. Run: `PYTHONPATH=src python -m
  ariadne.validate.stage32`; full perf table: `PYTHONPATH=src python -m ariadne.perf`.

**Stage 33 results (general relativity, 1PN):** `dynamics/relativity.py` adds the Schwarzschild
1PN acceleration `a_GR = (mu/c^2 r^3)[(4mu/r - v^2) r + 4(r.v) v]`. Integrating a Mercury-like orbit
under Newtonian+1PN reproduces the anomalous perihelion advance to **42.99 arcsec/century** (textbook
42.98; the 1915 confirmation of GR), and the measured per-orbit advance matches the analytic
`6 pi mu/(c^2 a(1-e^2))` to 0.02%. The term is a ~3e-8 fractional correction at 1 AU -- firewall-safe
(it never disturbs the Newtonian dynamics; it matters only as accumulated precession over Myr, where
it competes with any perturber's secular signal). The engine is now GR-capable. Run:
`PYTHONPATH=src python -m ariadne.validate.stage33`.

**Stage 34 results (real distant-TNO clustering -- the actual Batygin-Brown analysis on CURRENT data):**
crossed from the 6-object 2016 sample to the **live JPL Small-Body Database**: 706 bodies with a>150 AU.
`discovery/clustering.py` filters the detached extreme population and computes proper circular statistics
(mean resultant R, Rayleigh test, Monte-Carlo null). HONEST findings:
- **Extreme sample (a>=250, q>=42, N=19):** perihelion-longitude (varpi) clustering R=0.37, **p=0.07
  (~1.8 sigma) -- MARGINAL, not significant at 0.05.** It has WEAKENED since 2016 as more objects were
  found. omega and Omega are not significant.
- **Broad population (a>=150, q>=30, N=75):** node (Omega) clustering p=0.0005 and omega p=0.028 (both
  significant), but this population is the MOST exposed to observational selection bias.
- The statistics are validated (analytic Rayleigh matches Monte-Carlo to 0.002; a uniform population is
  correctly not flagged, a clustered one is). **Bottom line:** a low p is necessary but NOT sufficient --
  the clustering could be where surveys looked. No Planet 9 claim; this is the honest current state of the
  actual evidence. Run: `PYTHONPATH=src python -m ariadne.validate.stage34`.

**Stage 35 results (differentiable dynamics + gradient-based optimization):** `optimize/autodiff.py`
adds a JAX differentiable RK4 two-body propagator and a Levenberg-Marquardt shooting solver that gets
the EXACT gradient of miss-distance through the integrator. Results: the autodiff gradient matches
finite differences to **4.1e-8** (it is exact; FD is the approximation); LM shooting solves the
transfer to **1.6 metres in 9 iterations**, while derivative-free Nelder-Mead needs **549 evaluations
and stalls at 9.2 km**. The differentiable RK4 propagator is itself correct (a circular orbit returns
after one period to 1.4e-11). Key lesson: a branchy analytic Lambert gives NaN autodiff gradients
(Stumpff/Newton singularities), but gradient-THROUGH-THE-INTEGRATOR is clean -- and an undamped Newton
step overshoots on a long arc, so LM damping + backtracking is required. This is the foundation for
fast gradient-based trajectory optimization across the engine. CPU JAX (the win is exact gradients,
not the GPU). Run: `PYTHONPATH=src python -m ariadne.validate.stage35`.

**Stage 36 results (observational-uncertainty propagation -- is the clustering even real?):** pulled the
REAL per-element 1-sigma uncertainties from the JPL SBDB and Monte-Carlo'd the extreme-eTNO perihelion
longitudes. Findings: the angles are mostly measured to <0.1 deg (median sigma_varpi = 0.034 deg), so for
the well-observed objects measurement error is negligible -- BUT one short-arc object has an essentially
unconstrained perihelion (sigma -> 180 deg). Propagating the real uncertainties, the clustering
significance **degrades from p=0.07 to a median p=0.13** (16-84% [0.056, 0.25]; significant in only 13%
of resamples); the well-measured subset (N=17) is p=0.11. **Verdict:** the marginal clustering is FRAGILE
-- dominated by small N and one unconstrained object, not by a perturber. Combined with Stage 34's
selection-bias caveat, the honest bottom line is that current public data do NOT provide compelling
statistical evidence for Planet 9. Run: `PYTHONPATH=src python -m ariadne.validate.stage36`.

**Stage 37 results (independent cross-validation vs REBOUND):** the strongest possible integrator
check -- an INDEPENDENT professional code. Integrating the same DE440 ICs (Sun + 4 giants) with both
Ariadne's democratic-heliocentric Wisdom-Holman map and **REBOUND's WHFast** (Rein & Liu 2012; Jacobi
coordinates, a different Kepler solver and operator splitting), the giant heliocentric positions agree
to **1.1e-4 over 100 yr and 1.7e-3 over 2000 yr** (the residual is accumulated along-track phase from
the different symplectic splittings), and REBOUND's energy is bounded at **3.6e-6** -- independent
confirmation that Ariadne's integrator is both correct and symplectic, not merely self-consistent.
(rebound is an optional dependency; the test skips cleanly if it is absent.) Run:
`PYTHONPATH=src python -m ariadne.validate.stage37`.

**Stage 38 results (selection bias vs the clustering -- the OSSOS objection):** the deepest objection to
the Planet 9 evidence is observational selection bias. Model-light test: the scattered, Neptune-coupled
control population (a>150, 30<q<=42, N=51) -- which a distant perturber should NOT shepherd -- traces the
survey selection function. It is itself weakly clustered with mean perihelion **21.7 deg**; the detached
extreme objects (N=19) cluster at **20.4 deg -- the SAME direction (gap 1.3 deg)** -- and their clustering
is **NOT significant relative to the control (p=0.12)**. Conclusion: on current public data the extreme-
eTNO clustering CANNOT be distinguished from selection bias. Together with Stages 34/36 (marginal p=0.07,
fragile to uncertainty), the honest verdict is settled: **no compelling statistical evidence for Planet 9
in the public catalog.** Run: `PYTHONPATH=src python -m ariadne.validate.stage38`.

**Stage 39 results (THE DISCOVERY CORE -- moving-object orbit linkage):** the step that crosses from a
validated engine to one that could find something NEW. `discovery/linkage.py` implements a HelioLinC-style
linker: hypothesise a heliocentric distance r and radial velocity rdot; under that hypothesis every
tracklet (a short on-sky position+rate) maps to a full heliocentric state; propagate each to a common
reference epoch with the validated 2-body integrator; tracklets of the SAME object collapse to a tight
cluster while unrelated detections scatter. Cluster -> candidate orbit. **The honest litmus test PASSES:**
from a survey-realistic haystack -- the REAL orbits of 5 known eTNOs observed over a 2-month opposition at
0.1" astrometry, buried in 400 interloper tracklets (430 total) -- the linker **recovers all 5 as PURE
candidates with ZERO false positives**, and the transform is exact (tracklets collapse to 3.7e-7 AU at the
true distance). If it recovers KNOWN objects from the haystack, the same machinery can find UNKNOWN ones.
HONEST scope: validated on synthetic detections generated from REAL orbits (realistic noise, cadence,
interlopers); linking REAL survey detections (ZTF/Pan-STARRS/Rubin) is the next step -- the core algorithm
is proven. A subtle lesson learned: the reference-epoch window must be short (an opposition season) and
tracklet arcs long enough that a distant object's tiny on-sky rate is measurable, else velocity error
amplifies over the baseline and the cluster smears. Run: `PYTHONPATH=src python -m ariadne.validate.stage39`.

**Stage 40 results (dynamical-structure mining -- the other P9 signatures):** beyond perihelion, the
Planet 9 case invokes orbital-PLANE (pole) clustering and Neptune decoupling. `discovery/structure.py`
checks both, selection-aware. Honest negatives: the extreme objects' orbital poles are concentrated
(R=0.93) but the scattered control population is concentrated IDENTICALLY (p=0.69) -- the alignment is the
low-inclination/selection triviality, not a perturber; and 0/19 extreme objects sit at a low-order
Neptune mean-motion resonance (they are genuinely detached, as the population definition requires). So on
EVERY signature checked (perihelion varpi, orbital pole, resonance), the public catalog shows no
structure beyond selection. With Stages 34/36/38 this is the complete, honest scientific verdict:
no compelling evidence for Planet 9 in current public data. Run: `PYTHONPATH=src python -m ariadne.validate.stage40`.

**Stage 41 results (autodiff global trajectory optimization):** Stage 35 gave exact gradients through the
integrator; this stage uses them for global trajectory DESIGN. `optimize_transfer` solves each transfer
EXACTLY with the autodiff Levenberg-Marquardt shooting (arrival enforced -- no penalty-relaxed near-miss)
and sweeps (departure date, time-of-flight) for the global minimum. On Earth->Mars it finds
**Delta-v = 5.63 km/s at launch +300 d / tof 315 d, miss 0.19 km** -- the textbook ~5.6 km/s heliocentric
value, and consistent with Stage 21's INDEPENDENT Lambert porkchop (5.68 km/s): two methods, same answer.
Found by gradient-based shooting rather than a brute grid of trial velocities. This is the engineering-
discovery axis (novel/optimal trajectories), and it scales to higher-dimensional control where derivative-
free methods struggle. Run: `PYTHONPATH=src python -m ariadne.validate.stage41`.

**Stage 42 results (THE DISCOVERY CORE ON REAL DATA):** Stage 39 proved the linker on synthetic
detections; Stage 42 closes the loop on the ACTUAL telescope record. `linkage.tracklets_from_mpc` fetches
a known object's real recorded MPC astrometry (via astroquery), builds nightly tracklets from its densest
opposition season, and `link` recovers it from a haystack of interlopers. **The real-data litmus PASSES:**
from the REAL astrometry of **Sedna** (2024 opposition, 18 real tracklets) and **2001 FP185** (11 real
tracklets), each buried among 200 interloper tracklets, the linker recovers the object as a PURE candidate
with ZERO false positives. The discovery core works on REAL survey data, not just simulations -- the proof
that the same machinery could find UNKNOWN objects. HONEST scope: this recovers KNOWN objects from real
astrometry mixed with interlopers; finding a genuinely NEW object requires the full UNLINKED survey
archive (the MPC Isolated Tracklet File, or a ZTF/Pan-STARRS/Rubin detection database) -- a data-
engineering step, not an algorithm gap. Run: `PYTHONPATH=src python -m ariadne.validate.stage42`.

**Stage 43 results (THE BEAST'S DINNER -- discovery core on the real MPC ITF):** fed the linker the
actual unlinked archive -- the MPC Isolated Tracklet File, 135 MB of observations the MPC's own pipeline
could not associate to any object. End-to-end run: `discovery/itf.py` parses 2.6 M tracklets in 17 s,
isolates 45,340 slow movers (distant-object regime), bins them into ~498 linkable sky/time cells, and
links per bin. With the proper filters (size cap 4-30, multi-night >=4, tight cluster radius 0.3 AU,
canonical designation-set dedup), the search produces **607 clean distinct candidates** -- down from
16,317 raw before deduping. Calibration: an injected synthetic known object in the densest real ITF bin
is RECOVERED by the linker (the pipeline works on the actual haystack). HONEST cross-match: the top 25
spatially-distinct candidates queried against SkyBoT returned **25/25 KNOWN objects** (re-links of TNOs
the MPC's pipeline left unlinked -- including correctly re-identifying very slow movers at 0.1-0.2
arcsec/hr as 2008 US331, 2016 GL368, etc.). NO new object discovered, which is the **correct, skeptical
expected outcome on the public archive** -- the MPC's own linker is excellent, so what remains unlinked
is mostly hard or already-known. The engine is **discovery-capable on the full real unlinked archive**;
a genuine NEW find would need Rubin/LSST-class data (deeper than the MPC's public bar) and is never
announced from a pipeline run. Run: `PYTHONPATH=src python -m ariadne.validate.stage43`.

**Decisions on record:**
- 2026-05-28 — New standalone repo (credibility); codename **Ariadne**.
- 2026-05-28 — Reproduce **Earth–Moon first**, then generalize.
- 2026-05-28 — Coherence-field methods are a **search-acceleration layer only**; dynamics
  use standard gravity (credibility firewall).
- 2026-05-28 — Documentation-first: this master doc precedes code and is kept exhaustive.

**Changelog:**
- 2026-05-29 `v0.43` — Stage 43 (ITF ingest -- the real unlinked archive run) complete. Added
  discovery/itf.py (parse the MPC 80-col format, build tracklets, filter slow movers, sky/time-bin,
  link per bin), validate/stage43.py + tests/test_itf.py. End-to-end run on the live MPC Isolated
  Tracklet File (135 MB, 2.6 M tracklets, 45,340 slow movers in 17s): pipeline runs, injection
  validation recovers a synthetic known object planted in the densest real bin, and the search
  produces 607 clean canonical candidates after the proper size/night/dedup filters (down from
  16,317 raw). Cross-matched top 25 candidates against SkyBoT -- 25/25 are KNOWN objects (re-links
  the MPC's own pipeline left unlinked). HONEST: the engine works end-to-end on the real public
  archive but found no NEW object, which is the correct skeptical outcome (the MPC's linker is
  excellent, so unlinked leftovers are mostly hard or already-known; a true new find would need
  Rubin/LSST-class data and is never announced from a pipeline run).
- 2026-05-29 `v0.42` — Stage 42 (discovery core on REAL telescope data) complete. Added
  linkage.tracklets_from_mpc (fetches a known object's ACTUAL recorded MPC astrometry via astroquery,
  builds nightly tracklets from the densest opposition window) + add_interlopers, validate/stage42.py +
  tests/test_realdata_linkage.py (network-guarded). The honest real-data litmus PASSES: from the REAL
  recorded astrometry of known eTNOs (e.g. Sedna's 2024 opposition, ~18 real tracklets) buried among 200
  interlopers, the linker recovers each known object as a PURE candidate with no false positives. The
  discovery core works on REAL survey data, not just synthetic -- proof it could find UNKNOWN objects.
  Finding a NEW object needs the full unlinked archive (MPC ITF / a survey detection database) -- the
  next data-engineering step.
- 2026-05-29 `v0.41` — Stage 41 (autodiff global trajectory optimization) complete. Extended
  optimize/autodiff.py with transfer_dv (exact 2-impulse Delta-v via the autodiff LM shooting, arrival
  enforced) + optimize_transfer (global min over a departure/time-of-flight grid). validate/stage41.py +
  test_autodiff addition. On the canonical Earth->Mars problem it finds the global optimum
  Delta-v=5.63 km/s at launch+300 d / tof 315 d with 0.19 km miss -- the textbook ~5.6 km/s value, and
  consistent with Stage 21's independent Lambert porkchop (5.68 km/s). Gradient-based shooting, not a
  brute velocity grid. Completes the "Route E" engineering-discovery axis.
- 2026-05-29 `v0.40` — Stage 40 (dynamical-structure mining: the other P9 signatures) complete. Added
  discovery/structure.py (orbital-pole clustering vs the selection control; low-order Neptune-resonance
  proximity) + validate/stage40.py + tests/test_structure.py. Honest negatives: the extreme objects'
  orbital poles are concentrated (R=0.93) but the scattered control is identical (p=0.69 -- selection,
  not a perturber); 0/19 extreme objects sit at a low-order Neptune resonance (detached, as expected).
  No dynamical structure beyond selection on ANY signature (perihelion + pole + resonance) -- with
  Stages 34/36/38 the public catalog gives no compelling Planet 9 evidence.
- 2026-05-29 `v0.39` — Stage 39 (DISCOVERY CORE: moving-object orbit linkage) complete. Added
  discovery/linkage.py: a HelioLinC-style linker (hypothesise heliocentric distance r + radial velocity
  rdot -> map each tracklet to a full state -> propagate to a reference epoch with the validated 2-body
  integrator -> cluster; same-object tracklets collapse to a point, interlopers scatter), precomputed
  observer geometry + vectorised transform/propagation, and a synthetic-from-real tracklet generator.
  validate/stage39.py + tests/test_linkage.py. The honest litmus test PASSES: from a survey-realistic
  haystack (5 real eTNOs over a 2-month opposition at 0.1" + 400 interlopers) the linker recovers ALL 5
  as PURE candidates with no false positives, and the transform is exact (tracklets collapse to <0.01 AU
  at the true distance). This is the machinery that can find UNKNOWN objects. Validated on synthetic-
  from-real detections; linking real survey detections (ZTF/Pan-STARRS/Rubin) is the next step.
- 2026-05-29 `v0.38` — Stage 38 (selection bias vs eTNO clustering -- the OSSOS test) complete. Added
  discovery.clustering.selection_bias_test + validate/stage38.py + test. Uses the scattered Neptune-
  coupled control population (a>150, 30<q<=42) as a survey-selection-function proxy. Result: the
  detached extreme objects cluster at mean varpi ~20 deg, the control prefers ~22 deg (same direction),
  and the extreme clustering is NOT significant relative to the control (p~0.12). Current public data
  CANNOT distinguish a perturber from selection bias -- no Planet 9 claim.
- 2026-05-29 `v0.37` — Stage 37 (independent cross-validation vs REBOUND) complete. Added
  validate/stage37.py + test_rebound_xval.py. Integrating the same DE440 ICs (Sun + 4 giants) with both
  Ariadne's democratic-heliocentric WH map and REBOUND's WHFast (independent code, Jacobi coords,
  different Kepler solver), the giant heliocentric positions agree to ~1e-4 over 100 yr and ~1.7e-3 over
  2000 yr (accumulated phase), and REBOUND's energy is bounded at ~3.6e-6 -- an independent confirmation
  that Ariadne's integrator is correct and symplectic. (rebound is an optional dep; test skips if absent.)
- 2026-05-29 `v0.36` — Stage 36 (uncertainty propagation into the clustering significance) complete.
  Extended discovery/clustering.py (load_with_uncertainty pulls real per-element 1-sigma errors from the
  JPL SBDB -> data/distant_tnos_sigma.json; resampled_clustering_p Monte-Carlos the perihelion errors),
  validate/stage36.py, test_clustering addition. HONEST result: the extreme-eTNO angles are mostly
  measured to <0.1 deg, BUT one short-arc object has an essentially unconstrained perihelion (sigma->180
  deg); propagating that, the clustering significance DEGRADES from p=0.07 to median p=0.13 (significant
  in only ~13% of resamples). The marginal clustering is FRAGILE -- dominated by small N and one
  unconstrained object; current data do NOT give compelling evidence for a perturber.
- 2026-05-29 `v0.35` — Stage 35 (differentiable dynamics + gradient-based optimization) complete.
  Added optimize/autodiff.py: a JAX differentiable RK4 two-body propagator (branch-free -> clean
  autodiff, unlike a branchy analytic Lambert whose Stumpff/Newton singularities give NaN gradients)
  + a Gauss-Newton shooting solver that solves the transfer via EXACT gradients of the integrator.
  validate/stage35.py + tests/test_autodiff.py. The autodiff gradient matches finite differences to
  ~4e-8; Gauss-Newton hits the target to ~mm in ~16 iterations vs Nelder-Mead's ~550 evals stalling at
  km-level. CPU JAX (the win is exact gradients, not the GPU). The path to fast gradient-based
  trajectory optimization across the engine.
- 2026-05-29 `v0.34` — Stage 34 (real distant-TNO clustering significance) complete. Added
  discovery/clustering.py (loads the live/cached JPL SBDB catalog -- 706 bodies with a>150 AU --
  filters the detached extreme population, computes circular stats: mean resultant R, Rayleigh test,
  Monte-Carlo null), validate/stage34.py, tests/test_clustering.py, viz figure_etno_clustering, and
  the cached data/distant_tnos.json. HONEST result on CURRENT data: the famous extreme-sample (N=19,
  a>=250, q>=42) perihelion (varpi) clustering is only MARGINAL (R=0.37, p=0.07, ~1.8 sigma -- weakened
  since the 2016 6-object sample); broad-population (N=75) node clustering is strong (Omega p=0.0005)
  but most exposed to observational selection bias. Low p is necessary, NOT sufficient, for a perturber.
  No Planet 9 claim. The statistics are validated (analytic Rayleigh ~ Monte-Carlo to 0.002; unbiased).
- 2026-05-29 `v0.33` — Stage 33 (general relativity, 1PN) complete. Added dynamics/relativity.py
  (Schwarzschild 1PN acceleration `a_GR = (mu/c^2 r^3)[(4mu/r - v^2)r + 4(r.v)v]`, analytic
  perihelion advance, Newtonian+GR helper), validate/stage33.py, tests/test_relativity.py.
  Reproduces Mercury's anomalous perihelion advance to 42.99 arcsec/century (textbook 42.98,
  ratio 1.0002). GR is a ~3e-8 fractional correction at 1 AU (firewall-safe; matters only as
  accumulated precession over Myr). The engine is now GR-capable.
- 2026-05-29 `v0.32` — Stage 32 (multi-backend ensemble + intelligent selector) complete. Added
  dynamics/secular_gpu.py (numba.cuda one-thread-per-particle ensemble, faithful 1.2e-14),
  secular_fast.integrate_ensemble_parallel (24-core, allocation-free scalar Kepler; 8.7x at N=1k),
  dynamics/integrators.py (the selector: measured crossovers parallel>=1k, gpu>=5k), perf.py (unified
  harness), transport_graph.search.astar_coherence + node_coherence (forge tau-field port),
  validate/stage32.py + tests/test_integrators.py. HONEST: the tau-field cost methods do NOT help
  Ariadne's search/optimization (tau is derived from gravity -> redundant; Newton recovery), confirmed
  by Ariadne's own Stage-12 note + the forge's own honest-loss files. The transferable forge win is the
  SELECTOR pattern, now the brain of the multi-backend integrator. No new physics; firewall intact.
- 2026-05-29 `v0.31` — Stage 31 (secular/Gyr frontier + 427x acceleration) complete. Added
  dynamics/secular_fast.py (numba-JIT of the exact symplectic map: 1.5k->640k steps/s = 427x, faithful
  to 1.3e-11; integrate_fast + integrate_fast_elements) and dynamics/secular_avg.py (doubly-averaged
  Gauss-ring secular integrator: universal Kepler-eq sampler, ring-averaged perturbing accel with the
  indirect term, orbit-averaged Gauss planetary equations; reaches Gyr). validate/stage31.py
  (G31a numba 427x + faithful; G31b da/dt=1.3e-14 + Laplace-Lagrange ratio 1.0000; G31c secular-vs-exact
  14%/4% across the 6 real eTNOs; G31d 1-Gyr with/without P9). tests/test_secular_avg.py (7 tests).
  Honest Gyr finding: a fixed-ring P9 DISPERSES rather than confines (orbit-averaging removes the
  mean-motion resonances central to shepherding) -> motivates the GPU resonance-preserving frontier.
- 2026-05-29 `v0.30` — Stage 30 (long-term symplectic dynamics + secular Planet 9) complete. Added
  dynamics/secular.py: universal-variable (Stumpff) Kepler propagator (machine-precision to e~0.93)
  + 2nd-order symplectic Wisdom-Holman map in democratic-heliocentric coordinates (DLL98), elements
  <-> state, osculating-element extraction, and the perihelion-clustering statistics. validate/stage30.py
  (G30a symplecticity dE=1.1e-5 bounded + dt^2 ratio 0.243 + L=2.6e-13; G30b DE440 cross-check 5.7e-4
  over a century; G30c with/without-P9 eTNO divergence 0.7-64 AU over 100 kyr; G30d differential
  precession -4.5..+4.2 deg/Myr). tests/test_secular.py (8 tests). viz figure_secular (energy + divergence).
  The honest ceiling-raiser: turns a snapshot residual into an accumulating secular signal. No new physics,
  no Planet 9 claim -- standard Newtonian gravity, a Gyr-scale problem measured over 100 kyr.
- 2026-05-29 `v0.29` — Stage 29 (real-data bridge to Signalbook) complete. Added
  discovery/skybridge.py (build_celestial_index extracts ra/dec from Signalbook payload_json into an
  indexed celestial_sources table; query_sky cone search; ecliptic->equatorial; crossmatch_localization),
  validate/stage29.py, test_skybridge.py (5 tests). On the real 47M atlas it indexes 4,414,657 celestial
  sources and an Ariadne localization cross-matches to 703 real catalogued sources (<1 s); galactic
  centre 95% X-ray. Closed a Signalbook gap (celestial sources weren't sky-indexed) and contributed
  the fix back as signalbook discovery/celestial_index.py (commit b72aeaf). Honest: cross-match against
  KNOWN sources (confirmation handoff), not new-moving-body detection.
- 2026-05-29 `v0.28` — Stage 28 (inverse hidden-mass localizer + two-tier discovery pipeline)
  complete. Added discovery/inverse_mass.py (weighted nonlinear least-squares localizer with
  covariance/confidence region; simulate_observations; localization_vs_n honing; sky_box telescope
  handoff; sensitivity_skymap; single-body degeneracy guard), validate/stage28.py,
  test_inverse_mass.py (4 tests), viz.figure_localization_honing + figure_sensitivity_skymap.
  G28 PASS: recovers an injected body (6 M_E @ 625 AU -> 7.3 M_E @ 745 AU, within 1-sigma) from
  noisy residuals at the 6 real eTNOs; the 1-sigma region honing 504 AU (N=2) -> 209 AU (N=6); a
  single body is degenerate (need >=2 diverse); all-sky sensitivity map produced. Honest: a
  confidence region (Le-Verrier-style), floor/degeneracy-limited, GM-only; real detection needs
  secular accumulation + real data + IR/optical confirmation.
- 2026-05-29 `v0.27` — Stage 27 (principled coherence field + trajectory-residual hidden-mass
  detector) complete. Added fields/tau_c.py (tau_c = 1 + Phi/c^2 from the total potential of all
  masses; g_coh = -c^2 grad ln(tau_c)) and fields/hidden_mass.py (residual detector + real clustered
  eTNO elements + published Planet 9 params + Kuiper noise floor + a (mass,distance) detectability
  map generalizing to any body), validate/stage27.py, test_hidden_mass.py (6 tests),
  viz.figure_planet9 + figure_detectability. G27 PASS: Newton recovery to |Phi|/c^2 = 1.004e-8 at
  Earth (the framework's own result, reproduced); a hypothesized Planet 9 rises 57-118x above the
  unmodeled-Kuiper floor at all 6 real eTNOs but is ~1e-5 of the solar pull (needs secular baselines).
  Real data throughout (DE440 + published eTNO/Planet-9 values). Honest: rigorous detector, not a
  detection -- the secular Myr inference is the noted next tool; Cassini ranging is the real precedent.
- 2026-05-29 `v0.26` — Stage 26 (solar-system coherence atlas + self-correction) complete. Added the
  whole-solar-system registry (23 systems: Sun-planet x8, giant-planet major moons, Pluto-Charon
  binary) to constants.py, fields/solar_atlas.py (catalog + a FAIR region-matched coherence test),
  validate/stage26.py, test_solar_atlas.py, viz.figure_solar_atlas. G26a PASS: periodic libration
  for all 23 systems (mu 1.65e-8 .. 0.108). G26b PASS as a SELF-CORRECTION: the fair region-matched
  test refutes the Stage-25 "coherence skeleton" (skeleton 0/6, separatrix 6/6) -- it was a
  region-sampling artifact; manifolds are ordinary separatrices. Stage-25 entry annotated OVERTURNED.
- 2026-05-29 `v0.25` — Stage 25 (falsifiable coherence-field test) complete. Added
  fields/coherence_field.py (FLI-based coherence field over CR3BP phase space), validate/stage25.py,
  test_coherence_field.py (3 tests), viz.figure_coherence_field. Falsifiable result (C=3.15, n=60,
  Mann-Whitney p=0.009): the invariant-manifold transport tubes are significantly MORE coherent
  (lower FLI) than generic states -- the naive "chaos ridge" hypothesis is refuted; the transport
  network is the coherence SKELETON of phase space. HONEST: significant + elegant, but not a
  breakthrough and consistent with known dynamics (manifolds are organized asymptotic trajectories).
- 2026-05-29 `v0.24` — Stage 24 (packaging + open-source) complete; the user-requested enhancement
  arc (Stages 18-24) is closed. Added pyproject.toml (setuptools src-layout, runtime deps, console
  entry points ariadne-atlas/ariadne-figures), MIT LICENSE, .github/workflows/ci.yml (fast-test CI
  on 3.11/3.12 + kernel cache), validate/stage24.py, test_packaging.py (4 tests); bumped
  __version__ to 0.24.0. G_pkg PASS: version consistent, license + CI present, all subsystems
  import, `pip install -e .` works and a wheel builds.
- 2026-05-29 `v0.23` — Stage 23 (unified multi-objective grand optimizer) complete. Added
  interplanetary/grand.py (launch-window robustness axis + coherence-balanced route selection over
  energy/time/robustness with steerable weights + an Edelbaum low-thrust estimate), validate/stage23.py,
  test_grand.py (3 tests), viz.figure_grand_tradeoff. G23 PASS: the Earth->Mars trade space shows
  no-free-lunch structure (cheapest=most robust=309 d/5,683 m/s; fastest=priciest+fragile=120 d/
  13,314 m/s); coherence picks a genuine middle ground (186 d/8,085 m/s) and the weights steer it
  (energy-first 219 d, time-first 153 d). The synthesis the project was driving toward.
- 2026-05-29 `v0.22` — Stage 22 (gravity-assist multi-flyby global optimizer) complete. Added
  interplanetary/flyby.py (patched-conic flyby chain + turn-authority + differential-evolution
  global search + stored Galileo-class VEEGA reference), validate/stage22.py, test_flyby.py
  (3 tests), viz.figure_veega. G22 PASS: a Venus-Earth-Earth VEEGA cuts launch C3 to Jupiter from
  the direct 76.9 to 16.8 km^2/s^2 (4.6x), all flybys within turn authority, ~6.4-yr Galileo-class
  flight. Honest: feasible but not fully ballistic (~2,487 m/s powered-flyby Delta-v; minimal-DSM
  refinement noted). The single-Earth-flyby variant is correctly turn-infeasible.
- 2026-05-29 `v0.21` — Stage 21 (interplanetary porkchop + epoch-swept global optimizer) complete.
  Added the `interplanetary` package: porkchop.py (heliocentric Lambert over launch-date x TOF,
  global differential-evolution optimum, launch-window finder, time/energy Pareto + coherence knee)
  and gmat_helio.py (Sun-centered GMAT export); R_MARS/R_VENUS/GM_VENUS in constants.py;
  validate/stage21.py, test_interplanetary.py (6 tests), viz.figure_porkchop + figure_mars_transfer.
  G21 PASS: recovers the real Earth->Mars windows (2026-11/2028-11/2030-12, ~26-mo cadence); global
  optimum depart 2026-11-01 TOF 310 d C3 9.27, total 5,679 m/s; monotonic time/energy Pareto with a
  185-d balance knee. The launch epoch is now a free variable; PNG + GMAT flight-path visuals added.
- 2026-05-29 `v0.20` — Stage 20 (multi-moon tour mining) complete. Added transfers/tisserand.py
  (Tisserand parameter + v_inf relation + connecting transfers + flyby turn authority + the
  gravity-assist vs Hohmann comparison), Galilean moon radii in constants.py, validate/stage20.py,
  test_tisserand.py (4 tests), viz.figure_moon_tour. G20 PASS: the Galilean tour Io->Callisto needs
  252 m/s deterministic Delta-v (gravity assists supply the rest) vs 8,990 m/s Hohmann -- a 35.7x
  saving; v_inf 1.2-1.9 km/s with 47-83 deg turn authority. Honest: energy/Tisserand structure only.
- 2026-05-29 `v0.19` — Stage 19 (3D halos + Gateway-class NRHO) complete. Added orbits/nrho.py
  (pseudo-arclength halo continuation), validate/stage19.py, test_nrho.py (2 tests), viz.figure_nrho.
  G19 PASS: a 382-member L2 family continues from 14.82 d to a Near-Rectilinear Halo Orbit matching
  the Gateway 9:2 NRHO -- period 6.558 d, perilune 3,238 km (alt 1,501 km), apolune 71,198 km,
  periodic to 2.4e-14; near-stable (max Floquet 2.18 vs the L1 Lyapunov's 1,841). Honest: integrating
  halos as transport-graph nodes needs a 3D Poincare section (noted); the planar graph stands.
- 2026-05-29 `v0.18` — Stage 18 (ephemeris re-targeter) complete; closes the G12 fidelity gap.
  Added dynamics/frames.py (exact synodic<->inertial transform) + transfers/ephemeris_retarget.py
  (position-continuous multiple shooting in DE440), validate/stage18.py, test_retarget.py (4 tests).
  G18 PASS: frame round-trips to 3.3e-16 and embeds the Moon exactly; the L1 Lyapunov orbit
  re-converges in DE440 to 2.6 m (86 m/s/rev stationkeeping); the L1<->L2 heteroclinic connection
  re-converges to 5.3 m -- the discovered structure is ephemeris-real. Honest: the Delta-v figures
  are position-forced upper bounds; the minimal natural re-convergence is the noted refinement.
- 2026-05-29 `v0.17` — Stage 17 (deliverables) complete; the original roadmap is now closed
  end-to-end (Stages 0-17). Added docs/WHITE_PAPER.md (capstone paper), atlas/release.py (open
  release bundle: HDF5 atlas + INDEX.md + reference_routes.csv, with an honestly-tagged reference
  route table), validate/stage17.py, test_release.py (2 tests). G_deliver PASS: paper present with
  all key sections + headline numbers; release exports + round-trips; reference table carries a
  genuinely GMAT-validated route and keeps Earth->Moon transfers separate from libration-network
  routes. Standard gravity throughout; nothing claimed as new physics.
- 2026-05-29 `v0.16` — Stage 16 (generalization + HDF5 atlas) complete. Added 6 new systems to
  constants.py (Mars-Phobos, Saturn-Enceladus/Rhea/Titan, Sun-Mars, DART/Hera Didymos-Dimorphos;
  ATLAS_SYSTEMS) and the `atlas` package: store.py (HDF5 write/read, round-trippable, with
  provenance) + build.py (multi-system libration + Earth-Moon graph + ranked routes). validate/
  stage16.py, test_atlas.py (3 tests), viz.figure_atlas_systems. G_gen PASS: periodic L1 Lyapunov
  orbits for all 6 systems (mu 1.66e-8 .. 6.93e-3, periodic to <1e-12; Didymos L1 at 0.150 km),
  tracking the (mu/3)^(1/3) Hill scaling. G_atlas PASS: the atlas (7 systems, EM 8-node/53-edge
  graph, 8 routes, provenance) round-trips exactly through HDF5. The Atlas (§12) now exists.
- 2026-05-29 `v0.15` — Stage 15 (discovery engine + route verification, G12) complete. Added the
  `discovery` package: mining.py (Yen k-shortest loopless paths via constrained Dijkstra + ranked
  catalog + Delta-v-vs-robustness Pareto set) and verify.py (CR3BP continuity + energy bookkeeping
  + CR3BP-vs-BCR4BP solar-perturbation survivability); validate/stage15.py, test_discovery.py
  (5 tests). ALSO upgraded the transport-graph EDGE MODEL to energy-exact v_x (v_x^2 = 2*Omega - C
  - v_y^2 at each crossing, no interpolation) -- strictly more rigorous, and it REVISED the Stage 14
  optimum: the converged answer is a **3-hop route ~16-17 m/s** (L1@3.120 -> L1@3.160 -> L2@3.160
  -> L2@3.172), ~2x cheaper than the single direct patch, exploiting Oberth burns at high-speed
  near-Moon crossings. Route topology is STABLE across resolution (90->12.7, 120->16.1, 150->16.9
  m/s); the earlier interpolated "direct 37 m/s" was a discretization artifact (its topology
  changed with resolution). G12 PASS: engine mines 8 distinct ranked routes; ALL verify (continuity
  exact, Jacobi-vs-orbit residual 1.3e-15); optimal route survives the solar perturbation as a
  bounded correctable arc (38,610 km / 0.10 L* over 8.7 d). HONEST: "novel" = automatically
  discovered + verified IPTN structure, NOT unknown to science; full DE440 re-convergence + GMAT
  was closed for the Earth->Moon transfer leg in Stages 8-10; a libration ephemeris re-targeter is
  the remaining tool (noted). Coherence weight now changes the route (cheap 3-hop -> robust direct).
- 2026-05-29 `v0.14` — Stage 14 (transport-graph search + efficiency benchmark, G11) complete.
  Added the `transport_graph` package: graph.py (IPTN as a graph; edges are exact Poincare
  section-crossing patches), search.py (Dijkstra SSSP + A* with an admissible energy heuristic +
  exhaustive brute-force baseline + admissibility verifier), benchmark.py; validate/stage14.py,
  test_transport_graph.py (6 tests), viz.figure_transport_graph, pytest.ini (slow marker). G11
  PASS: all three routers agree on the optimum; A* reaches it in ~5 expansions vs brute force's
  ~210 (~42x less work); heuristic verified admissible; ballistic L1<->L2 patches ~0 m/s
  (consistent with G6). The efficiency claim is resolution-independent (search is exact). The
  specific optimum value/route was refined in v0.15 (energy-exact edge model). FMM/HJB deferred.
- 2026-05-29 `v0.13` — Stage 13 (engine generalization + low-thrust regime) complete. Added
  transfers/jovian.py (Galilean-moon libration + moon-tour Delta-v), dynamics/low_thrust.py
  (continuous-acceleration CR3BP + energy-rate theorem), validate/stage13.py, test_stage13.py
  (4 tests), viz.figure_low_thrust_spiral. Gates G_jov + G_lt PASS: engine ports to Jupiter with
  only constants (Io..Callisto L1 10,469..49,691 km, periodic to <1e-9); low-thrust validated
  (zero-thrust = CR3BP to 2.7e-10, dC/dt = -2a|v| to 2e-5). (Separately, cosmology project: BTFR
  slope 3.41 / slope-4 a0 ~ 1.46x g_A, and a cross-scale cluster test 3.55 vs 4.06 within 14%,
  beating MOND-simple. Cosmology lives in Coherence_Energy_Labs, not here.)
- 2026-05-28 `v0.12` — Stage 12 (coherence-weighted optimizer) complete. Added
  transfers/coherence_optimizer.py (Δv-vs-robustness Pareto front, knee, weighted optimum),
  validate/stage12.py, test_coherence_optimizer.py, knee marked on coherence_frontier.png.
  Knee = direct 4-day transfer: 3.6x more robust than the cheapest WSB route for +71 m/s.
  Realizes the coherence-guided route chooser on standard gravity. (Separately: independently
  tested the S_One parameter-free RAR on real SPARC -> 0.105 dex, ~MOND; the a0(z)~H(z)
  discriminator is data-limited. Cosmology lives in the Coherence_Energy_Labs project, not here.)
- 2026-05-28 `v0.11` — Stage 11 (coherence / robustness lens) complete. Added analysis/coherence.py
  (endpoint_sensitivity = km arrival-drift per m/s = robustness; decoherence_rate FTLE as a coarse,
  caveated signal), validate/stage11.py, viz.figure_coherence_frontier, test_coherence.py. Mapped
  the Δv-vs-coherence frontier: robustness costs fuel (cheapest WSB ~8x more fragile than the fast
  transfer); stable LEO ~23x more coherent than a transfer. Honest corrections: both transfers are
  fragile (corrected my "WSB knife-edge" over-claim); the FTLE metric is unreliable (demoted).
  Coherence == robustness, a real DIFFERENT objective — does not beat the energy floor.
- 2026-05-28 `v0.10` — Stage 10 (Sun-assisted low-energy / WSB transfer) complete. Added
  transfers/wsb.py (Belbruno backward-from-capture in DE440 + min-Delta-v optimization of the
  capture velocity/phase), validate/stage10.py, viz.figure_wsb_transfer, test_wsb.py. Converged
  a LEO-departing low-energy transfer at **3,907 m/s** (TOF 48.8 d, v_inf 0.738) -- below the
  direct 3,953 AND below Coimbra's 3,925. Two-impulse patched model; longer TOF than Coimbra's
  32 d. Found from the dynamics, not fitted. Original Coimbra chase now closed (bracketed then beaten).
- 2026-05-28 `v0.9` — Stage 9 (literal NASA GMAT cross-validation) complete. Installed GMAT
  R2026a locally (tools/, git-ignored); enhanced io/gmat_export.py (ReportFile + locate_gmat +
  run_with_gmat headless runner + report parser); validate/stage9.py + test_gmat.py (skip if
  absent). G10 LITERAL: Ariadne vs GMAT trans-lunar propagation agree to 149 m / 0.89 mm/s.
  Stage 9 probe: two-impulse ephemeris transfer bottoms ~3,947 m/s (worsens with TOF) -> exact
  3,925 needs a WSB exterior route (Stage 10).
- 2026-05-28 `v0.8` — Stage 8 (full DE440 ephemeris transfer) complete. Added
  transfers/ephemeris_transfer.py (Lambert seed + ephemeris differential correction targeting
  the real Moon to ~50 m; TLI/v_inf/LOI; TOF optimization), validate/stage8.py,
  viz.figure_ephemeris_transfer, test_ephemeris_transfer.py. Direct transfer converges to
  3,953 m/s (TLI 3,136 + LOI 817); brackets Coimbra 3,925 (3,761 low / 3,953 high). GMAT not
  pip-installable here -> Stage-7 cross-checks + exported script serve G10. Fixed TLI to the
  energy-based (C3) optimal tangential injection. Number reported, not fitted.
- 2026-05-28 `v0.7` — Stage 7 (halos + Genesis + independent cross-validation) complete. Added
  orbits/halo.py (3D halo corrector + family from the Lyapunov bifurcation), transfers/genesis.py
  (Sun-Earth L1 halo + Earth-reaching manifold), validate/stage7.py, viz halo-3D + Genesis
  figures, test_halo.py + test_stage7.py. Gates: EM-L1 halos branch at C=3.18632 (matches
  Stage 2); SE-L1 halo period 177.9 d, manifold reaches 10,315 km from Earth; spiceypy vs
  jplephem agree 6 mm, DOP853 vs Radau 0.26 m. Fixed SE scale (small-mu needs small amplitudes).
  Sun perturbs the transfer ~40,900 km over 12 d -> exact 3925 is Stage 8 (ephemeris collocation).
- 2026-05-28 `v0.6` — Stage 6 (low-energy transfer + ballistic capture) complete. Added
  transfers/lunar_capture.py (real-manifold ballistic capture; Moon-relative periapsis speed),
  transfers/low_energy_lunar.py (end-to-end TLI+capture assembly + sweep), validate/stage6.py,
  viz.figure_low_energy_transfer, GMAT trans-lunar-injection export, test_transfers.py (37
  tests pass). Rigorous result: ballistic LOI 625 m/s vs direct 822 (saving 197 m/s, from real
  dynamics); end-to-end best 3,756 m/s brackets Coimbra 3,925. Probed manifold reach. Genesis
  (Sun-Earth) honestly deferred to Stage 7 (needs 3D halo tooling). Number reported, not fitted.
- 2026-05-28 `v0.5` — Stage 5 (real ephemeris + optimization toolkit) complete. Added
  data/kernels.py (SPICE/DE440 download+cache+lock) + data/ephemeris.py wrappers;
  dynamics/ephemeris_nbody.py (test-particle + mutual n-body propagators);
  optimize/lambert.py (universal-variable BVP); optimize/collocation.py (Hermite-Simpson);
  io/gmat_export.py; validate/stage5.py; viz ephemeris/moon figures; test_ephemeris.py +
  test_optimize.py + test_io.py (35 tests pass). G7 passes; n-body tracks DE440 to ~0.02 km
  @ 2 d; Lambert µm/s; collocation J=12.0 exact. Installed spiceypy 8.1. Real NASA data now
  flowing. Exact 3925 reproduction + Genesis/Hiten + GMAT-run deferred to Stage 6.
- 2026-05-28 `v0.4` — Stage 4 (BCR4BP + Δv budget + low-energy mechanism) complete. Added
  dynamics/bcr4bp.py (bicircular Sun-perturbed model + sun_params), optimize/budget.py
  (vis-viva Earth-Moon Δv budget + ballistic-capture saving), validate/stage4.py,
  viz.figure_budget, test_bcr4bp.py + test_budget.py (28 tests pass). Gates G8a/b/c pass;
  direct transfer 3953 m/s (Apollo-class), ballistic-capture saving 145 m/s. Exact 3925 m/s
  reproduction deferred to Stage 5 (ephemeris + collocation) — honest scope.
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
