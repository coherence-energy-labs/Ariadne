Tutorials
=========

Five tight, runnable scripts demonstrating Ariadne's headline capabilities. Each is intentionally
short — read top to bottom in a minute, run in seconds, look at the PNG it produces.

Run from the project root::

   PYTHONPATH=src python examples/01_lyapunov_family.py

(After ``pip install -e .`` the ``PYTHONPATH=src`` prefix is unnecessary.)

01 — L1 Lyapunov family
-----------------------

Build the Earth-Moon L1 Lyapunov orbit family in ~30 lines via amplitude continuation; plot
30 family members colored by Jacobi constant.

.. literalinclude:: ../../examples/01_lyapunov_family.py
   :language: python
   :linenos:

02 — Gateway NRHO
-----------------

Construct NASA's Gateway-class 9:2 Near-Rectilinear Halo Orbit via pseudo-arclength
continuation; verify period 6.56 d, perilune 3,238 km, apolune 71,198 km, Floquet 2.18
(843× more stable than a deep L1 Lyapunov — why Gateway flies an NRHO).

.. literalinclude:: ../../examples/02_gateway_nrho.py
   :language: python
   :linenos:

03 — Cislunar manifold transport
--------------------------------

L1↔L2 halo heteroclinic at ~112 m/s (x = 1-μ section); NRHO↔L2 halo at ~119 m/s (y = 0
section). Real Gateway-class cislunar transfers via natural manifold structure.

.. literalinclude:: ../../examples/03_manifold_transport.py
   :language: python
   :linenos:

04 — TNO orbit fit from MPC astrometry
--------------------------------------

Discovery-engine filter: pull real MPC astrometry for Sedna / Eris / Quaoar, fit orbits to
1.4–8.7" RMS, recover (a, e, i) within a few percent.

.. literalinclude:: ../../examples/04_tno_orbit_fit.py
   :language: python
   :linenos:

05 — Coherence-HJB on full 6D CR3BP
------------------------------------

Sampled-graph Helmholtz value function: 20,000 quasi-random samples + dynamics-derived k-NN
graph + sparse Helmholtz CG. Full 6D CR3BP with ~84% greedy reach to lunar goal in
sub-second compute. No grid, no curse of dimensionality on the dynamics-derived graph.

.. literalinclude:: ../../examples/05_helmholtz_hjb.py
   :language: python
   :linenos:
