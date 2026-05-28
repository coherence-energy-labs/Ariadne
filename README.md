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
