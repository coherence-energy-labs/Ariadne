# Ariadne

**Low-energy trajectory design & discovery of the solar system's natural transport network**
— the invariant-manifold "tubes," heteroclinic chains, and resonant corridors that move
spacecraft between bodies for a fraction of the usual fuel.

> Working codename. The thread through the interplanetary labyrinth.

## Start here

**[`MASTER_PLAN.md`](MASTER_PLAN.md) is the single source of truth.** It contains the
vision, the prior art, the full scientific foundations (CR3BP, Jacobi constant, Lagrange
points, periodic orbits, invariant manifolds, heteroclinic connections, the fidelity
ladder), the real-data plan (NASA SPICE / DE440), the software architecture, the
validation gates, and the staged roadmap. Read it before writing code.

**Current status & next action:** see `MASTER_PLAN.md` §16 (Status & Changelog).

## Where we are (Stages 1–3 complete, validated)

The CR3BP core, libration-point orbit families, and invariant-manifold transport tubes are
built and validated against published values (Lagrange points to ~1e-11, Jacobi conserved to
~1e-12, halo bifurcation at C≈3.186 matching literature). We can find the natural low-energy
"highways" — including a verified **L1↔L2 heteroclinic connection**.

![L1↔L2 transport tubes and the heteroclinic connection](docs/figures/heteroclinic_L1_L2.png)

The Earth–Moon L1 (green) and L2 (red) Lyapunov orbits and their invariant-manifold tubes;
where the tube cuts cross on the Poincaré section (★) is a near-ballistic L1↔L2 connection.

![L1 planar Lyapunov family](docs/figures/L1_lyapunov_family.png)

The L1 planar Lyapunov family colored by Jacobi constant, with the halo bifurcation located.

We also model the Sun's perturbation (bicircular model) and the Δv economics of getting to
the Moon. A direct Apollo-class transfer costs ~3953 m/s LEO→LLO; arriving via ballistic
capture cuts the lunar-insertion burn, saving ~145 m/s — the mechanism the Coimbra
3925 m/s result optimizes.

![Earth–Moon Δv budget](docs/figures/delta_v_budget.png)

We now run on **real JPL DE440 ephemeris** (via SPICE). A self-consistent Sun–Earth–Moon
(+4 planet) n-body integration tracks DE440 to ~0.02 km over 2 days and stays under a few km
for a month — and we have validated Lambert and Hermite–Simpson collocation solvers plus a
GMAT export for independent cross-checking.

![n-body propagator vs JPL DE440](docs/figures/ephemeris_validation.png)

And we can now build a **low-energy lunar transfer**: an L1 Lyapunov *unstable manifold*
delivers the spacecraft to ~100 km lunar periapsis ballistically (near-parabolic arrival), so
the lunar-orbit insertion costs **625 m/s** (computed from real CR3BP dynamics) versus **822
m/s** for a direct hyperbolic capture — a **197 m/s saving**. End-to-end, the best LEO→LLO
transfer is **~3,756 m/s**, bracketing the Coimbra **3,925 m/s** result (direct = 3,953).

![Ballistic lunar capture via manifold](docs/figures/low_energy_transfer.png)

> Honest scope: the rigorous, validated result is the *ballistic-capture saving* from real
> manifold dynamics. The exact 3,925 m/s figure depends on the paper's boundary conditions and
> a Sun-assisted (BCR4BP) departure optimization — that, the Genesis reproduction, and running
> the exported GMAT script are Stage 7. Numbers here are reported from the construction, not
> fitted to the target.

We also build **3D halo orbits** (the family branches exactly at the Stage-2 vertical
bifurcation, C≈3.186) and reproduce the **Genesis mechanism**: a real Sun–Earth L1 halo
(period **177.9 days**, matching SOHO/Genesis) whose invariant manifold carries a spacecraft
from L1 (1.49M km out) down to **10,315 km from Earth** — the interplanetary superhighway.

![Genesis Sun–Earth superhighway](docs/figures/genesis_superhighway.png)

Everything is cross-validated against independent tools: two ephemeris libraries (spiceypy vs
jplephem on DE440) agree to **6 mm**, and two independent integrators (DOP853 vs Radau) agree
to **0.26 m**.

Finally, we design transfers on the **real DE440 ephemeris**: a Lambert seed plus a
differential correction that shoots in full Sun+Earth gravity to hit the **actual Moon
position to ~50 m**. The TOF-optimized direct transfer converges to **3,953 m/s** — and this
**brackets the Coimbra 3,925 m/s from both sides** (3,761 m/s with ballistic capture, 3,953 m/s
direct), pinning the published result to within tens of m/s with real data.

![Full-ephemeris Earth→Moon transfer](docs/figures/ephemeris_transfer.png)

> The *exact* 3,925 m/s is a low-energy (Sun-assisted ballistic-capture / WSB) optimum; reaching
> it precisely needs the paper's boundary conditions and a multi-week trajectory optimization
> (Stage 9). Every number is reported from the optimizer, never fitted.

Regenerate: `PYTHONPATH=src python -m ariadne.viz.figures`. Validate:
`PYTHONPATH=src python -m ariadne.validate.stage8` (and `stage1`–`stage7`). The SPICE
kernels download automatically on first use (DE440s ~33 MB).

## What this is (and is not)
- It **is**: an open, high-fidelity, validated engine + atlas for low-energy spaceflight,
  built on standard gravity and real ephemerides, cross-validated against NASA GMAT.
- It **is not**: new physics. The dynamics are standard n-body gravity. (See §1.4 / §13.)

## Quick setup (once code exists)
```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout
See `MASTER_PLAN.md` §8. Source in `src/ariadne/`, tests in `tests/`, generated atlas in
`results/atlas/`, downloaded SPICE kernels in `data/kernels/` (git-ignored).
