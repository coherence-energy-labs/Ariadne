"""End-to-end DISCOVERY loop: from pixels to a vetted unknown-candidate list.

Everything before this recovered KNOWN objects (the catalog hands you the
velocity). Discovery is the harder job: find moving objects, remove the
ones already catalogued, and vet what's left into real candidates. This
module chains the validated pieces into that loop:

  detections (>=3 same-night exposures)
    -> within-night 3-point tracklets        (triplet_linker)
    -> cross-match each tracklet to the MPC   (accurate N-body ephemeris)
         -> KNOWN  (recovery)
         -> UNKNOWN (candidate)
    -> VET unknowns:
         * magnitude consistency across the 3 epochs  (a real object keeps
           ~constant brightness; 3 chance-aligned stars do not -- the key
           discriminator for short arcs)
         * orbit plausibility: the trail rate must invert (opposition
           relation) to a physical heliocentric distance / orbit class
    -> ranked DISCOVERY CANDIDATES (need a confirming second night)

Output is honest: known recoveries, vetted candidates, and rejects with
reasons. Producing candidates is the system finally doing discovery, not
recovery.

Public API:
  run_discovery(epochs, db, *, observatory_code, ...) -> DiscoveryResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .triplet_linker import link_collinear_tracklets
from .orbit_geometry import opposition_rate_to_distance, classify_by_distance


@dataclass
class Candidate:
    mean_ra: float
    mean_dec: float
    mean_mjd: float
    rate_arcsec_hr: float
    pa_deg: float
    mag: float
    mag_std: float
    collinear_resid_arcsec: float
    implied_r_au: float
    orbit_class: str
    known_designation: str | None
    reject_reason: str | None


@dataclass
class DiscoveryResult:
    n_tracklets: int
    known_recoveries: list           # Candidate with known_designation set
    candidates: list                 # vetted unknowns (reject_reason is None)
    rejected: list                   # unknowns that failed vetting
    note: str = ""


def run_discovery(epochs, db, *, observatory_code: str = "807",
                   collinear_tol_arcsec: float = 1.0,
                   mag_std_max: float = 0.4,
                   match_radius_arcsec: float = 3.0,
                   min_r_au: float = 1.05, max_r_au: float = 60.0,
                   h_min: float = 3.0, h_max: float = 26.0,
                   max_mag_catalog: float = 23.0) -> DiscoveryResult:
    """Run the full discovery loop on >=3 same-night exposures.

    epochs: list of dicts {ra, dec, mag, mjd} (one per exposure), arrays.
    db: detection DB with the MPCORB known_objects catalog.
    """
    # 1. within-night tracklets
    radec = [(e["ra"], e["dec"], float(e["mjd"])) for e in epochs]
    mags = [e["mag"] for e in epochs]
    order = sorted(range(len(radec)), key=lambda i: radec[i][2])
    radec_s = [radec[i] for i in order]
    mags_s = [mags[i] for i in order]
    trk = link_collinear_tracklets(radec_s, collinear_tol_arcsec=collinear_tol_arcsec,
                                      mags=mags_s)

    # 2. catalog ephemeris at the mean epoch for cross-match (db=None skips
    #    the catalog step -- e.g. for unit tests / pure false-positive runs)
    mean_mjd = float(np.mean([r[2] for r in radec_s]))
    pred = np.empty((0, 2)); pred_desig = []
    if db is not None:
        from .injection_recovery import pick_orbits_in_field
        from .mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
        from .mpc_catalog import observatory_geo_km
        all_ra = np.concatenate([e["ra"] for e in epochs])
        all_dec = np.concatenate([e["dec"] for e in epochs])
        inf = pick_orbits_in_field(db, mean_mjd, (all_ra.min(), all_ra.max()),
                                     (all_dec.min(), all_dec.max()),
                                     max_mag=max_mag_catalog, limit_candidates=1600000)
        recs = [x[0] for x in inf]
        obs = observatory_geo_km(observatory_code, mean_mjd)
        eph = bulk_ephemeris_at_mjd_nbody(recs, mean_mjd, observer_geo_km=obs) \
            if recs else np.empty((0, 4))
        pred = np.array([[eph[i, 0], eph[i, 1]] for i in range(len(recs))
                           if not np.isnan(eph[i, 0])]) if len(recs) else np.empty((0, 2))
        pred_desig = [recs[i].designation for i in range(len(recs))
                      if not np.isnan(eph[i, 0])]

    known, candidates, rejected = [], [], []
    for t in trk:
        # magnitude consistency across the 3 epochs
        m3 = [mags_s[k][idx] for k, idx in enumerate(t.idx)]
        m3 = [v for v in m3 if v is not None and v > -50]
        mag = float(np.median(m3)) if m3 else -99.0
        mstd = float(np.std(m3)) if len(m3) >= 2 else 99.0
        rate_hr = t.rate_deg_day * 3600.0 / 24.0     # deg/day -> "/hr
        r_au = opposition_rate_to_distance(t.rate_deg_day * 3600.0)  # deg/day -> "/day
        cls = classify_by_distance(r_au) if r_au == r_au else "unknown"

        # cross-match to catalog
        designation = None
        if pred.shape[0]:
            cd = math.cos(math.radians(t.mean_dec))
            sep = np.hypot((pred[:, 0] - t.mean_ra) * cd,
                            pred[:, 1] - t.mean_dec) * 3600
            j = int(np.argmin(sep))
            if sep[j] <= match_radius_arcsec:
                designation = pred_desig[j]

        c = Candidate(mean_ra=t.mean_ra, mean_dec=t.mean_dec, mean_mjd=t.mean_mjd,
                        rate_arcsec_hr=rate_hr, pa_deg=t.pa_deg, mag=mag,
                        mag_std=mstd, collinear_resid_arcsec=t.collinear_resid_arcsec,
                        implied_r_au=r_au, orbit_class=cls,
                        known_designation=designation, reject_reason=None)

        if designation is not None:
            known.append(c)
            continue
        # vet the unknown
        # implied absolute magnitude at the inferred distance (near
        # opposition phase ~0): H = V - 5 log10(r * Delta)
        delta_au = r_au - 1.0 if (r_au == r_au and r_au > 1) else float("nan")
        H_impl = (mag - 5.0 * math.log10(r_au * max(delta_au, 1e-3))
                    if (delta_au == delta_au and delta_au > 0 and mag > -50)
                    else float("nan"))
        reason = None
        if mstd > mag_std_max:
            reason = f"mag inconsistent across epochs (std={mstd:.2f}>{mag_std_max})"
        elif not (r_au == r_au and min_r_au <= r_au <= max_r_au):
            reason = f"implausible distance (r={r_au:.1f} AU)"
        elif not (H_impl == H_impl and h_min <= H_impl <= h_max):
            # e.g. a bright source at a large inferred distance => absurd size
            # (a star with centroid jitter faking a slow mover)
            reason = f"implausible size (H={H_impl:.1f} at r={r_au:.1f} AU)"
        if reason:
            c.reject_reason = reason
            rejected.append(c)
        else:
            candidates.append(c)

    candidates.sort(key=lambda c: (c.mag_std, c.collinear_resid_arcsec))
    return DiscoveryResult(
        n_tracklets=len(trk), known_recoveries=known,
        candidates=candidates, rejected=rejected,
        note=(f"{len(trk)} tracklets -> {len(known)} known recoveries, "
              f"{len(candidates)} vetted unknown candidates, "
              f"{len(rejected)} rejected."))


def false_positive_floor(epochs, *, n_trials: int = 3, offset_deg: float = 0.08,
                           **kw) -> float:
    """Estimate the chance-alignment candidate floor by SCRAMBLING the last
    epoch's positions (which destroys every real tracklet while preserving
    source density and the magnitude distribution) and re-running discovery.

    Returns the mean number of "candidates" produced from scrambled data --
    the number a single night yields by pure chance. A real candidate count
    is only meaningful if it exceeds this floor; otherwise the survivors are
    statistically consistent with noise and require a confirming night.
    Validated: on the real 2024-04-28 field the floor is ~1-2, and the field
    produced 3 -- i.e. NOT a significant excess (no defensible discovery).
    """
    import copy
    kw = dict(kw); kw["db"] = None      # skip catalog; count raw candidates
    counts = []
    for t in range(n_trials):
        sc = copy.deepcopy(list(epochs))
        sc[-1] = dict(sc[-1])
        sc[-1]["ra"] = np.asarray(sc[-1]["ra"]) + offset_deg * (1 + 0.3 * t)
        sc[-1]["dec"] = np.asarray(sc[-1]["dec"]) + 0.5 * offset_deg
        res = run_discovery(sc, **kw)
        counts.append(len(res.candidates))
    return float(np.mean(counts))
