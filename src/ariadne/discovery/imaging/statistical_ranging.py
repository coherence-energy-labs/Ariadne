"""Single-snapshot statistical orbit ranging (Bayesian, population-prior).

*** EXPERIMENTAL / NOT YET CALIBRATED. ***
`rank_orbit` is a research prototype of full 6-D statistical ranging. On
real-asteroid validation it does NOT yet produce calibrated distance
posteriors -- it collapses toward near-Earth solutions because of the
classic ill-conditioning of single-observation ranging (at opposition the
observer's velocity is ~perpendicular to the line of sight, so the radial-
velocity root that reconciles a slow distant orbit barely exists, while
near solutions always admit roots). Getting this calibrated is a known
hard problem (Virtanen 2001; Muinonen & Bowell 1993; Granvik).

==> USE `snapshot_posterior.snapshot_posterior` INSTEAD. It is the
CALIBRATED replacement: it reparametrizes onto the distance axis the data
actually constrains (the cure for this exact degeneracy -- the same
reparametrization lesson the Coherence cosmology MCMC learned when a
perfect omega_cdm/Omega_psi degeneracy wrecked convergence), and is
validated on real DECam asteroids (~99% distance-CI coverage, ~0.25 AU
error). `orbit_geometry.single_snapshot_estimate` gives a fast point
estimate. This module is kept only as the scaffold for a future full 6-D
posterior.

The rigorous engine behind "a picture says a thousand words." A single
exposure fixes the line of sight (RA, Dec) and -- via PSF trailing -- the
TRANSVERSE angular velocity (rate, PA). What it cannot give directly is the
geocentric distance Delta and the line-of-sight (radial) velocity v_r.
Those two unknowns span the family of orbits consistent with the image.

We range over (Delta, v_r): for each sample, build the full heliocentric
state, derive (a, e), the implied absolute magnitude H, and weight by the
real asteroid POPULATION prior over (a, e) and over H. The weighted
samples are a calibrated posterior over distance and orbit class.

This fuses every single-snapshot cue automatically:
  - rate + PA      -> transverse velocity = rate * Delta (ties v to Delta)
  - apparent mag   -> H_implied(V, r, Delta); the population H prior makes a
                      "very distant + very bright" solution implausible
                      (it would need an enormous object), breaking the
                      distant-vs-near degeneracy.
  - population a,e -> bound, plausibly-eccentric orbits are favoured.

For an UNTRAILED object (rate ~ 0) the posterior is naturally BIMODAL: a
"distant & slow" lobe and a "nearby & radial (incoming/receding)" lobe,
weighted by brightness -- exactly the impactor-vs-TNO ambiguity. The
`needs_followup` flag fires when the nearby/incoming lobe carries
significant probability.

Public API:
  RangingObservation(...)            -- one snapshot's observables
  rank_orbit(obs, prior, n=200_000)  -> RangingPosterior
  PopulationPrior.from_db(db)        -- build (a,e) + H priors from MPCORB
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GM_SUN = 1.32712440018e11    # km^3/s^2
AU_KM = 149597870.7


@dataclass
class RangingObservation:
    ra_deg: float
    dec_deg: float
    mjd: float
    rate_arcsec_hr: float
    rate_sigma_arcsec_hr: float
    pa_deg: float
    pa_sigma_deg: float
    v_mag: float
    observer_helio_km: np.ndarray      # (3,) equatorial heliocentric, km
    observer_vel_km_s: np.ndarray      # (3,) equatorial heliocentric, km/s


@dataclass
class PopulationPrior:
    a_edges: np.ndarray
    e_edges: np.ndarray
    ae_logpdf: np.ndarray              # log P(a, e) on the grid
    h_edges: np.ndarray
    h_logpdf: np.ndarray

    @classmethod
    def from_db(cls, db, *, a_range=(0.4, 60.0), n_a=60, n_e=25,
                 h_range=(2.0, 26.0), n_h=48):
        from .mpc_catalog import load_known_element_arrays
        _, A = load_known_element_arrays(db)
        a = A["a_au"]; e = A["e"]; h = A["H_mag"]
        good = np.isfinite(a) & np.isfinite(e) & (a > 0) & (a < a_range[1]) & (e < 1)
        a_edges = np.linspace(a_range[0], a_range[1], n_a + 1)
        e_edges = np.linspace(0, 1, n_e + 1)
        H, _, _ = np.histogram2d(a[good], e[good], bins=[a_edges, e_edges])
        H = H + 0.5                     # Laplace smoothing
        ae_logpdf = np.log(H / H.sum())
        hg = np.isfinite(h) & (h > h_range[0]) & (h < h_range[1])
        h_edges = np.linspace(h_range[0], h_range[1], n_h + 1)
        hist, _ = np.histogram(h[hg], bins=h_edges)
        hist = hist + 0.5
        h_logpdf = np.log(hist / hist.sum())
        return cls(a_edges, e_edges, ae_logpdf, h_edges, h_logpdf)

    def logp_ae(self, a, e):
        ai = np.clip(np.digitize(a, self.a_edges) - 1, 0, len(self.a_edges) - 2)
        ei = np.clip(np.digitize(e, self.e_edges) - 1, 0, len(self.e_edges) - 2)
        return self.ae_logpdf[ai, ei]

    def logp_h(self, h):
        hi = np.clip(np.digitize(h, self.h_edges) - 1, 0, len(self.h_edges) - 2)
        lp = self.h_logpdf[hi]
        lp[(h < self.h_edges[0]) | (h > self.h_edges[-1])] = -50.0
        return lp


@dataclass
class RangingPosterior:
    distance_au_samples: np.ndarray    # weighted geocentric distance samples
    weights: np.ndarray
    a_samples: np.ndarray
    e_samples: np.ndarray
    h_samples: np.ndarray
    n_eff: float
    # summaries
    distance_med: float
    distance_lo: float                 # 5th pct
    distance_hi: float                 # 95th pct
    p_neo: float                       # P(a < 1.3 AU or q < 1.3)
    p_distant: float                   # P(r > 6 AU)
    needs_followup: bool
    bimodal: bool

    def class_probabilities(self):
        from .orbit_geometry import classify_by_distance
        # heliocentric r ~ distance + 1 (rough); use a samples for class
        w = self.weights
        classes = {}
        for a, wt in zip(self.a_samples, w):
            c = classify_by_distance(a)
            classes[c] = classes.get(c, 0.0) + wt
        tot = sum(classes.values()) or 1.0
        return {k: v / tot for k, v in sorted(classes.items(),
                                                key=lambda x: -x[1])}


def _sky_frame(ra_deg, dec_deg):
    a = math.radians(ra_deg); d = math.radians(dec_deg)
    los = np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a),
                     math.sin(d)])
    north = np.array([-math.sin(d) * math.cos(a), -math.sin(d) * math.sin(a),
                       math.cos(d)])
    east = np.array([-math.sin(a), math.cos(a), 0.0])
    return los, north, east


def rank_orbit(obs: RangingObservation, prior: PopulationPrior,
                *, n: int = 200_000, delta_range_au=(0.02, 200.0),
                vr_range_km_s=(-70.0, 70.0), seed_offset: int = 0
                ) -> RangingPosterior:
    """Posterior over geocentric distance + orbit class from one snapshot.

    Principled ranging that avoids the radial-velocity volume bias: instead
    of sampling v_r freely (which floods near-Earth solutions), we sample
    the semi-major axis `a` from the population prior, let VIS-VIVA fix the
    heliocentric speed at the trial distance, and SOLVE for the radial
    velocity that is consistent with the measured transverse motion. Each
    valid (Delta, a) pair yields a concrete orbit whose eccentricity is then
    weighted by the population p(a, e) and a gentle brightness term.

    Deterministic given inputs (fixed-seed RNG) so it is testable.
    """
    rng = np.random.default_rng(12345 + seed_offset)
    los, north, east = _sky_frame(obs.ra_deg, obs.dec_deg)

    # proposal: log-uniform Delta, a ~ population a-marginal, Gaussian rate/PA,
    # random radial-velocity SIGN (the two vis-viva roots).
    logd = rng.uniform(math.log(delta_range_au[0]), math.log(delta_range_au[1]), n)
    delta_au = np.exp(logd)
    delta_km = delta_au * AU_KM
    # sample a from the population marginal (bin centers weighted by counts)
    a_centers = 0.5 * (prior.a_edges[:-1] + prior.a_edges[1:])
    a_marg = np.exp(prior.ae_logpdf).sum(axis=1)
    a_marg = a_marg / a_marg.sum()
    a_au = rng.choice(a_centers, size=n, p=a_marg)
    a_au = a_au + rng.uniform(-0.5, 0.5, n) * (prior.a_edges[1] - prior.a_edges[0])
    a_au = np.clip(a_au, 0.5, prior.a_edges[-1])
    a_km = a_au * AU_KM
    sign = rng.integers(0, 2, n) * 2 - 1     # +/- radial root
    rate = np.abs(rng.normal(obs.rate_arcsec_hr,
                              max(obs.rate_sigma_arcsec_hr, 1e-3), n))
    pa = rng.normal(obs.pa_deg, max(obs.pa_sigma_deg, 1e-3), n)

    # heliocentric position at the trial distance
    r_vec = obs.observer_helio_km[None, :] + delta_km[:, None] * los[None, :]
    r_norm = np.linalg.norm(r_vec, axis=1)
    # vis-viva heliocentric speed for the sampled a at this r
    vv = GM_SUN * (2.0 / r_norm - 1.0 / a_km)
    speed_ok = vv > 0
    v_mag = np.sqrt(np.where(speed_ok, vv, 0.0))

    # transverse velocity (relative to observer): v_t = rate * Delta
    rate_rad_s = np.radians(rate / 3600.0) / 3600.0
    v_t = rate_rad_s * delta_km
    pa_r = np.radians(pa)
    t_hat = (np.cos(pa_r)[:, None] * north[None, :]
             + np.sin(pa_r)[:, None] * east[None, :])
    # A = V_obs + v_t*t_hat ; solve |A + v_r*los|^2 = v_mag^2 for v_r
    A = obs.observer_vel_km_s[None, :] + v_t[:, None] * t_hat
    A2 = np.einsum("ij,ij->i", A, A)
    Adotl = A @ los
    disc = Adotl ** 2 - (A2 - v_mag ** 2)
    has_root = speed_ok & (disc >= 0)
    v_r = -Adotl + sign * np.sqrt(np.where(disc >= 0, disc, 0.0))
    v_helio = A + v_r[:, None] * los[None, :]
    v_norm = np.linalg.norm(v_helio, axis=1)

    # eccentricity from the resulting state
    rv = np.einsum("ij,ij->i", r_vec, v_helio)
    e_vec = ((v_norm ** 2 - GM_SUN / r_norm)[:, None] * r_vec
             - rv[:, None] * v_helio) / GM_SUN
    e = np.linalg.norm(e_vec, axis=1)

    r_au = r_norm / AU_KM
    H_impl = obs.v_mag - 5.0 * np.log10(np.maximum(r_au * delta_au, 1e-6))

    # weights: population p(a,e) (a already from prior, so this adds p(e|a))
    # + gentle brightness term. The rate is encoded in the v_r solve, so a
    # missing real root (has_root False) means this (Delta,a) cannot produce
    # the observed transverse motion -> excluded.
    # Physically-correct weighting. Three competing terms:
    #  (1) population p(a,e)            -- favours real orbit shapes
    #  (2) geometric VOLUME prior Δ^3   -- objects-per-distance ∝ Δ^2 for a
    #      uniform space density, times the log-uniform-proposal Jacobian Δ
    #      -> Δ^3. Pushes the posterior OUTWARD (more volume far away).
    #  (3) size-distribution prior on the implied H, dN/dH ∝ 10^(βH) with
    #      β≈0.5: distant solutions need an intrinsically bright (rare big)
    #      object, which this penalises. Pulls the posterior INWARD.
    # (2) and (3) balance at the true distance (a Malmquist-type balance);
    # (1) selects the orbit class. This is what makes the rate+brightness
    # fusion calibrated rather than collapsing to the nearest solution.
    BETA = 0.5
    H_MIN, H_MAX = 4.0, 24.0
    valid = has_root & np.isfinite(a_au) & (e < 1.0) & (a_au > 0.4) \
        & (H_impl > H_MIN) & (H_impl < H_MAX)
    logw = np.full(n, -1e9)
    if valid.any():
        lp_ae = prior.logp_ae(a_au[valid], e[valid])
        lp_vol = 3.0 * np.log(delta_au[valid])
        lp_size = BETA * math.log(10.0) * H_impl[valid]
        logw[valid] = lp_ae + lp_vol + lp_size
    logw -= np.nanmax(logw)
    w = np.exp(logw)
    w[~np.isfinite(w)] = 0.0
    wsum = w.sum()
    if wsum <= 0:
        w = np.ones(n) / n
    else:
        w = w / wsum
    n_eff = 1.0 / np.sum(w ** 2)

    # summaries (weighted)
    order = np.argsort(delta_au)
    da_s = delta_au[order]; w_s = w[order]
    cw = np.cumsum(w_s)
    def wq(q):
        return float(np.interp(q, cw, da_s))
    dist_med = wq(0.5); dist_lo = wq(0.05); dist_hi = wq(0.95)
    # heliocentric r ~ delta + ~1 near opposition; classify by a instead
    p_neo = float(np.sum(w[(a_au < 1.3) | (a_au * (1 - e) < 1.3)]))
    p_distant = float(np.sum(w[delta_au > 6.0]))
    # bimodality: significant mass both near (<1.5 AU) and far (>2 AU)
    p_near = float(np.sum(w[delta_au < 1.5]))
    p_far = float(np.sum(w[delta_au > 2.0]))
    bimodal = (p_near > 0.15 and p_far > 0.15)
    needs_followup = (p_neo > 0.1) or (p_near > 0.2 and obs.v_mag < 21.5)

    return RangingPosterior(
        distance_au_samples=delta_au, weights=w, a_samples=a_au,
        e_samples=e, h_samples=H_impl, n_eff=float(n_eff),
        distance_med=dist_med, distance_lo=dist_lo, distance_hi=dist_hi,
        p_neo=p_neo, p_distant=p_distant, needs_followup=needs_followup,
        bimodal=bimodal)
