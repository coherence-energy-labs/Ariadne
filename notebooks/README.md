# Ariadne Jupyter notebooks

Five interactive notebooks that mirror the runnable scripts in [`../examples/`](../examples/),
suitable for opening in JupyterLab / VS Code / Colab.

| Notebook | What it demonstrates |
|---|---|
| [`01_lyapunov_family.ipynb`](01_lyapunov_family.ipynb) | Build the Earth-Moon L1 Lyapunov family by amplitude continuation |
| [`02_gateway_nrho.ipynb`](02_gateway_nrho.ipynb) | Construct NASA's Gateway 9:2 NRHO + verify period / perilune / apolune / Floquet |
| [`03_manifold_transport.ipynb`](03_manifold_transport.ipynb) | L1↔L2 halo (~112 m/s) + NRHO↔L2 halo (~119 m/s) transport patches |
| [`04_tno_orbit_fit.ipynb`](04_tno_orbit_fit.ipynb) | TNO orbit fit from real MPC astrometry (Sedna / Eris / Quaoar) |
| [`05_helmholtz_hjb.ipynb`](05_helmholtz_hjb.ipynb) | Coherence-HJB sampled-graph Helmholtz on full 6D CR3BP |

## Install Jupyter

```bash
pip install -e ".[notebooks,dev]"
jupyter lab  # or: jupyter notebook
```

Then open any of the notebooks. They share the same code as the `.py` examples — just
with markdown narration between cells.
