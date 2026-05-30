Architecture
============

Ariadne is organised as a layered set of subsystems, each with a clear scope and tested
boundary:

.. code-block:: text

                  discovery/        validate/          examples/
                    ▲    ▲              ▲                 ▲
                    │    │              │                 │
                    │    └──────────────┼─────────────────┤
    astroquery → linkage   iod        gates           tutorials
                    │        │           │                 │
                    └────┬───┘           │                 │
                         │               │                 │
                         └──────────┐    │    ┌────────────┘
                                    ▼    ▼    ▼
                            orbits/   dynamics/   manifolds/
                                │  (CR3BP, eom)       │
                                └────────┬────────────┘
                                         │
                            connections/ + transport_graph/
                                │ (Poincaré 2D & 3D)
                                ├─────────────────────────────┐
                                ▼                             ▼
                         interplanetary/                   optimize/
                          (Lambert, VEEGA,            (autodiff, LM, Lambert,
                           per-leg DSMs)               coherence_hjb, Helmholtz)
                                │
                                ▼
                       io/  +  certification/
                    (GMAT export, atlas, proof-carrying routes)

Layers from bottom to top
-------------------------

**data/** — physical constants (GM, R), CR3BP system definitions for 7 systems, and NASA
SPICE/DE440 ephemeris adapter. The ground truth.

**dynamics/cr3bp.py** — synodic-frame equations of motion, pseudo-potential, Jacobi
constant, state-transition matrix. The numerical heart.

**orbits/** — Lyapunov, halo, and NRHO periodic orbit families via differential correction
and pseudo-arclength continuation.

**manifolds/manifold.py** — stable/unstable manifold tubes seeded by Floquet eigenvectors
and transported around the orbit via the STM.

**connections/** — Poincaré sections in 2D ((y, vy)) and 3D ((y, z, vy, vz)) for
heteroclinic-class transport patches.

**transport_graph/** — the network: nodes are orbits, edges are exact section-crossing
patches with energy-consistent Δv. Routing via Dijkstra / A* / Yen's k-shortest.

**interplanetary/** — Lambert porkchop, gravity-assist chains (VEEGA + per-leg DSMs at
1-DOF or 4-DOF), Earth-Mars / Earth-Jupiter transfer optimization.

**optimize/** — Lambert universal-variable solver, autodiff (JAX) LM shooting, and the
Coherence-HJB sampled-graph Helmholtz value-function solver.

**discovery/** — HelioLinC-style tracklet linker for distant moving objects, (r, rdot)
hypothesis-based initial orbit determination, LM differential correction with light-time
correction, MPC Isolated Tracklet File ingest.

**certification/** — proof-carrying route certificates (CR3BP patch + BCR4BP survivability
+ DE440 retargeting + GMAT replay status), canonical JSON with SHA-256 payload hash.

**io/** — GMAT script export, HDF5 atlas persistence.

**viz/** — matplotlib helpers for orbit/manifold/route figures.

**validate/** — per-stage validation gates (45+ stages, every check is a PASS/FAIL gate
that exits 0/1; the validation suite IS the testbed).

Validation philosophy
---------------------

Every claim is cross-checked against an independent tool:

- Periodicity: each orbit is propagated for its period; residual is reported.
- Jacobi conservation: ``|C(t) - C(0)|`` measured along the orbit, must be < 1e-12.
- Ephemeris: two independent libraries (spiceypy, jplephem) must agree to mm.
- Integrators: DOP853 vs Radau on the same problem must agree to m.
- Transfer Δv: every computed transfer is compared to GMAT propagating the same state.
- TNO orbits: fit RMS reported on real MPC astrometry; (a, e, i) compared to JPL.

The validation gates are not optional. Every stage's check() returns (ok, info) and the
main() driver exits non-zero on failure.

Honesty firewall
----------------

The dynamics are standard n-body gravity. The CR3BP / BCR4BP / DE440 ladder is the
fidelity progression. Coherence-field methods (Stage 45 sampled-graph Helmholtz) are a
search-acceleration layer ON TOP OF standard gravity — never a replacement for the
physics. This was the original 2026-05-28 design decision; it remains the credibility
firewall.
