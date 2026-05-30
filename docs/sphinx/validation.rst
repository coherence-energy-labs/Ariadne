Validation & cross-checks
=========================

Every claim Ariadne makes is checked against an independent tool. The validation gates are
*not optional* — every stage's ``check()`` returns ``(ok, info)`` and the ``main()`` driver
exits non-zero on failure. The full validation suite IS the testbed.

Cross-validation matrix
-----------------------

================================  ========================================================
Validation                        Result
================================  ========================================================
Lagrange points (L1–L5)           Residual < 1e-11 nondim, all 7 systems
Jacobi constant conservation      ``|C(t) - C(0)|`` < 1e-12 over many orbital periods
SPICE DE440 ephemeris             spiceypy vs jplephem agree to **6 mm**
Integrator                        DOP853 vs Radau on the same problem agree to **0.26 m**
Earth–Moon transfer               Ariadne vs NASA GMAT agree to **149 m / 0.89 mm·s⁻¹**
TNO orbit fit (Sedna)             a-error **2.0%**, RMS **3.94"** on real MPC data
NRHO geometry                     period **6.558 d**, perilune **3,238 km**, apolune **71,198 km**
NRHO near-stability               Floquet **2.18** (vs L1 Lyapunov **1,841** — 843× more stable)
6D Coherence-HJB                  **84%** greedy reach to lunar goal, sub-second compute
Heteroclinic L1↔L2                **162 m/s** honest total (112 m/s velocity + 50 m/s 1-day rendezvous)
NRHO ↔ L2 halo                    **646 m/s** honest total (119 m/s velocity + 527 m/s 1-day rendezvous)
Same-orbit sanity check           **6.7 m/s** L2 halo → itself (essentially ballistic — float-level)
Discovery synthetic injection     **100% pure** Sedna recovery from 200-interloper haystack
HJB greedy trajectory             p50 = **7.3 km/s** honest total (in Hohmann-to-Edelbaum band)
================================  ========================================================

Discovery pipeline ground truth
-------------------------------

The HelioLinC linker + IOD orbit fit was validated on real MPC astrometry of five known TNOs:

============  ==========  ===============  =================
Object        a (AU)      RMS residual     a-error
============  ==========  ===============  =================
Sedna         506.0       3.94"            2.0%
Eris          67.7        5.79"            12.7%
Makemake      45.6        8.68"            1.1%
Quaoar        43.2        3.79"            0.5%
2001 FP185    226.5       1.39"            0.1%
============  ==========  ===============  =================

These are real residuals on real MPC astrometry — single-digit arcseconds, consistent with
the astrometric noise floor of ground-based optical surveys. The discrimination test
confirms the fitter as a sharp two-state filter:

- Sedna's tracklets alone fit to **3.94" — ACCEPTED**.
- Sedna's + Eris's tracklets interleaved fit to **inf — REJECTED**.

Honest negative result on the public archive
--------------------------------------------

When the full Stage 43 / 44 pipeline ran on the live MPC Isolated Tracklet File (135 MB,
2.6 million tracklets), it produced 863 candidates → 515 re-link to KNOWN catalogued objects
via SkyBoT cross-match + **0 of 348 unmatched survive the orbit-fit filter**.

This is the *correct, scientifically defensible* outcome on a public archive that the MPC's
own (excellent) linker has already processed: the residual unlinked tracklets are mixed-
object false-positive clusters (median fit residual 999"), not real undiscovered orbits. A
genuine new TNO find would require Rubin/LSST-class data deeper than the public MPC bar.

The pipeline is validated to FIND objects when they're there (synthetic injection + 5/5
real known TNOs); it is also validated to REJECT clusters that aren't real orbits (the
discrimination test). Both directions matter.
