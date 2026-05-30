Ariadne documentation
=====================

**Python-native cislunar mission design + TNO discovery — validated, tutorial-driven,
open source.**

Ariadne fills the gap between heavyweight astrodynamics platforms (GMAT, Monte,
Copernicus) and the existing Python libraries (poliastro, Tudat, Heyoka) for the
things they don't do: CR3BP-aware cislunar mission design, invariant-manifold
transport routing, and end-to-end TNO discovery from real survey astrometry.

Quick start
-----------

.. code-block:: bash

   pip install -e ".[dev]"

.. code-block:: python

   import ariadne

   # CR3BP system constants for 7 named systems
   em = ariadne.system("EARTH_MOON")

   # Periodic orbit families in one line
   family = ariadne.lyapunov_family("L1", n=30)

   # NASA Gateway 9:2 NRHO from scratch
   nrho = ariadne.gateway_nrho()

   # Fit a TNO orbit from real MPC astrometry
   fit = ariadne.discover_tno("90377")     # Sedna

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   api
   architecture
   validation
   honest_scope

Capabilities at a glance
------------------------

============================  ===========================================  =============================================
Subsystem                     Capability                                   Validated against
============================  ===========================================  =============================================
CR3BP core                    Synodic dynamics + STM + Jacobi constant     Conservation to 1e-12
Libration points              L1–L5 to ~1e-11 nondim, 7 systems            Earth-Moon, Sun-Earth, Jovian moons, Sun-Mars
Lyapunov / halo / NRHO        Continuation families, periodicity 1e-12     NASA Gateway 9:2 NRHO (6.56 d / 3.2 / 71 Mm)
Manifold tubes                Eigenvector-seeded, STM-transported          Jacobi-conserved to 1e-12
Heteroclinic                  Energy-consistent Poincaré crossings         0.0 m/s same-energy ballistic
Transport graph               Planar + 3D + NRHO via y=0                   L1↔L2 112 m/s, NRHO↔L2 119 m/s
Interplanetary                Lambert porkchop, VEEGA + per-leg DSMs       Galileo C3 16.8, Earth→Mars 5.63 km/s
Coherence-HJB                 Sampled-graph Helmholtz value function       6D CR3BP: ~84% greedy reach
TNO discovery                 HelioLinC + (r, rdot)-IOD + LM correction    Sedna/Eris/Makemake/Quaoar/2001 FP185 1.4–8.7"
Real-data                     MPC ITF (135 MB, 2.6 M tracklets)            515/863 known re-links + 0 new
Cross-validation              DE440 + GMAT + REBOUND                       149 m vs GMAT over 3 days
Proof-carrying routes         CR3BP → BCR4BP → DE440 certificates          (see :mod:`ariadne.certification`)
============================  ===========================================  =============================================

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
