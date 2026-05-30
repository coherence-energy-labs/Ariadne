# Ariadne runnable tutorials

Seven tight, runnable scripts demonstrating the headline capabilities. Each one is intentionally
short — read top to bottom in a minute, run in seconds, look at the PNG it produces.

Run from the project root with the package on the Python path:

```bash
PYTHONPATH=src python examples/01_lyapunov_family.py
```

(once `pip install -e .` is run you can drop the `PYTHONPATH=src` prefix.)

| Script | Runtime | Demonstrates |
|---|---|---|
| [`01_lyapunov_family.py`](01_lyapunov_family.py) | ~3 s | L1 Lyapunov orbit family (30 members, amplitude continuation) → PNG of the family colored by Jacobi constant |
| [`02_gateway_nrho.py`](02_gateway_nrho.py) | ~50 s | NASA Gateway 9:2 NRHO construction via pseudo-arclength continuation; period 6.56 d, perilune 3,238 km, apolune 71,198 km, Floquet 2.18 → 3D orbit plot + Moon-distance plot |
| [`03_manifold_transport.py`](03_manifold_transport.py) | ~30 s | Cislunar transport graph: L1↔L2 halo (~112 m/s, x = 1-μ section) and NRHO↔L2 halo (~119 m/s, y = 0 section) — real Gateway-class transfers via natural manifold structure |
| [`04_tno_orbit_fit.py`](04_tno_orbit_fit.py) | ~1 min | Pull real MPC astrometry for Sedna / Eris / Quaoar; fit each orbit to 1.4–8.7″ RMS, recover a, e, i within a few percent — the discovery-engine filter that decides whether candidate tracklets are real |
| [`05_helmholtz_hjb.py`](05_helmholtz_hjb.py) | ~5 s | Coherence-HJB: 20,000 quasi-random samples + dynamics-derived k-NN graph + sparse Helmholtz CG; full 6D CR3BP value function with ~84% greedy reach in sub-second compute. No grid, no curse of dimensionality. |
| [`06_veega_jupiter.py`](06_veega_jupiter.py) | ~5 s | Galileo-class Venus-Earth-Earth gravity assist to Jupiter on real DE440 ephemeris; C3 = 16.79 km²/s² (vs direct Earth→Jupiter ~85, a 4.6× energy reduction). Plots the heliocentric trajectory. |
| [`07_lambert_porkchop.py`](07_lambert_porkchop.py) | ~30 s | Earth→Mars Lambert porkchop over a 1.5-year launch window × 120-360 d TOF grid (728 Lambert solves); finds the Oct 2026 window at C3 = 9.27 km²/s² (near Hohmann theoretical 8.9). Plots contours + the global minimum. |

Output PNGs are written to `examples_out/`. The data folder for the discovery example
(`signalbook-data/itf/`) is git-ignored — it auto-downloads on first use.
