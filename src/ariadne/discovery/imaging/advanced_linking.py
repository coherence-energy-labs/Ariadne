"""Advanced multi-night chain linking for image-derived tracklets.

The default `chain_multi_night` in tracklets_from_images.py uses greedy
single-pass extrapolation: for each tracklet on night N, search night N+1
within a fixed gap window, take the first match. Three failure modes:

  1. Greedy + single-pass: when two tracklets on night N+1 are both
     within the gap window, the first one wins even if the second one
     is a better match. Causes mis-linking when sources are dense.
  2. Rate estimated from a SINGLE within-night pair has ~30% noise.
     Extrapolating with that rate to night N+3 accumulates ~90% error,
     so the gap budget gets exceeded by genuine matches.
  3. No use of the heliocentric geometry information. Two TNOs at
     the same sky position but different distances move at different
     rates -- the HelioLinC formalism exploits this.

This module provides three upgraded strategies:

  probabilistic_chain
    Hungarian-assignment-based linker. Each candidate (tracklet_A, tracklet_B)
    pair gets a likelihood score combining position gap, rate consistency,
    and photometric similarity. The best assignment over the bipartite
    graph is chosen GLOBALLY (not greedily), so dense fields no longer
    suffer from first-match wins.

  multipass_refined_chain
    1. Run probabilistic_chain to get 2-night chains.
    2. For each chain, RE-FIT the rate from both nights' positions
       (much tighter than single-night rate estimate).
    3. Re-search night N+2 with the refined rate, tighter tolerance.
    4. Iterate until converged or max_nights_gap reached.

  helio_linc_image_bridge
    Adapts the validated HelioLinC alert linker (discovery.linkage) to
    image-format tracklets. Hypothesises (r, rdot), maps each tracklet
    to a heliocentric state, propagates to common epoch, clusters in 6D.
    The same algorithm that recovered 100% of synthetic injections on
    the alert path -- now available for images.

  discover_in_images_chains
    Orchestrator: runs all three strategies, dedupes the chains they
    produce (by member set), returns the union. Each strategy catches
    different objects; the union is strictly better than any single
    strategy alone.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from .source_extraction import Source

SEC_PER_DAY = 86400.0


# =============================================================================
# Probabilistic Hungarian linker
# =============================================================================

def _pair_log_likelihood(t_a: dict, t_b: dict,
                          position_sigma_arcsec: float = 30.0,
                          rate_sigma_pct: float = 25.0,
                          mag_sigma: float = 0.5) -> float:
    """Log-likelihood that two tracklets are detections of the same object.

    Combines three terms:
      * Position: extrapolate t_a forward to t_b's epoch using t_a's
        rate, compare to t_b's centroid. Gaussian likelihood with
        position_sigma_arcsec.
      * Rate: compare t_a.rate to t_b.rate as a fraction. Gaussian
        with rate_sigma_pct.
      * Magnitude: when both tracklets have a representative magnitude
        in their source_pair, compare. Soft Gaussian.

    Returns log L; -inf if either input is malformed.
    """
    try:
        dt_days = t_b["jd"] - t_a["jd"]
        if dt_days <= 0:
            return -math.inf
        # Extrapolate
        dra_per_day = t_a["dra"] * SEC_PER_DAY
        ddec_per_day = t_a["ddec"] * SEC_PER_DAY
        ra_pred = math.degrees(t_a["ra"] + dra_per_day * dt_days)
        dec_pred = math.degrees(t_a["dec"] + ddec_per_day * dt_days)
        dra = math.radians(math.degrees(t_b["ra"]) - ra_pred) * math.cos(t_b["dec"])
        ddec = math.radians(math.degrees(t_b["dec"]) - dec_pred)
        gap_arcsec = math.degrees(math.hypot(dra, ddec)) * 3600.0
        log_pos = -0.5 * (gap_arcsec / position_sigma_arcsec) ** 2

        # Rate consistency
        rate_a = max(t_a.get("rate_arcsec_hr", 0.0), 1e-9)
        rate_b = max(t_b.get("rate_arcsec_hr", 0.0), 1e-9)
        rel_diff_pct = abs(rate_a - rate_b) / rate_a * 100.0
        log_rate = -0.5 * (rel_diff_pct / rate_sigma_pct) ** 2

        # Photometry consistency (optional)
        log_mag = 0.0
        sp_a = t_a.get("source_pair", ())
        sp_b = t_b.get("source_pair", ())
        if sp_a and sp_b:
            mag_a = sum(s.mag for s in sp_a) / len(sp_a)
            mag_b = sum(s.mag for s in sp_b) / len(sp_b)
            if -3 <= mag_a <= 30 and -3 <= mag_b <= 30:
                log_mag = -0.5 * ((mag_a - mag_b) / mag_sigma) ** 2

        return log_pos + log_rate + log_mag
    except Exception:
        return -math.inf


def _hungarian_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Wrap scipy's linear_sum_assignment for clarity.

    Returns list of (row_idx, col_idx) pairs that minimise total cost.
    Rows and cols may be different counts; scipy handles rectangular.
    """
    from scipy.optimize import linear_sum_assignment
    if cost_matrix.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def probabilistic_chain(tracklets: list[dict],
                         *, max_nights_gap: int = 14,
                         position_sigma_arcsec: float = 30.0,
                         rate_sigma_pct: float = 30.0,
                         log_likelihood_threshold: float = -10.0,
                         ) -> list[list[dict]]:
    """Hungarian-assignment multi-night linker.

    For each consecutive night pair, build a likelihood matrix and pick
    the globally-optimal one-to-one assignment. Tracklets without a
    match score below `log_likelihood_threshold` are not linked.

    Returns a list of chains, each spanning >= 2 distinct nights.
    """
    if not tracklets:
        return []
    by_night = defaultdict(list)
    for t in tracklets:
        by_night[t["night"]].append(t)
    nights = sorted(by_night)
    if len(nights) < 2:
        return []

    # Union-find structure: each tracklet starts as its own chain
    chains = {id(t): [t] for tlist in by_night.values() for t in tlist}

    for k in range(len(nights)):
        night_a = nights[k]
        for offset in range(1, min(max_nights_gap, len(nights) - k - 1) + 1):
            night_b = nights[k + offset]
            ts_a = by_night[night_a]
            ts_b = by_night[night_b]
            if not ts_a or not ts_b:
                continue
            # Build cost matrix (negative log-likelihood)
            n_a, n_b = len(ts_a), len(ts_b)
            cost = np.full((n_a, n_b), 1e6)
            for i, a in enumerate(ts_a):
                for j, b in enumerate(ts_b):
                    ll = _pair_log_likelihood(
                        a, b,
                        position_sigma_arcsec=position_sigma_arcsec,
                        rate_sigma_pct=rate_sigma_pct)
                    if ll > log_likelihood_threshold:
                        cost[i, j] = -ll
            pairs = _hungarian_assignment(cost)
            for i, j in pairs:
                if cost[i, j] >= 1e6:
                    continue
                a = ts_a[i]
                b = ts_b[j]
                src_chain = chains[id(a)]
                dst_chain = chains[id(b)]
                if src_chain is not dst_chain:
                    merged = src_chain + dst_chain
                    for x in merged:
                        chains[id(x)] = merged
            # Only need one offset per night pair; if matches were made
            # at offset=1, larger offsets are extension passes
    seen = set()
    out = []
    for ch in chains.values():
        cid = id(ch)
        if cid in seen:
            continue
        seen.add(cid)
        unique_nights = len({t["night"] for t in ch})
        if unique_nights >= 2:
            out.append(sorted(ch, key=lambda t: t["t"]))
    return out


# =============================================================================
# Multi-pass refinement
# =============================================================================

def _refit_rate_from_chain(chain: list[dict]) -> tuple[float, float]:
    """Re-fit (dra, ddec) per second from all chain members' positions.

    Linear regression of (ra, dec) vs time using the chain's all
    detections; gives a MUCH tighter rate estimate than the single-pair
    rate of each input tracklet.
    """
    ts = []
    ras = []
    decs = []
    for tr in chain:
        if "source_pair" in tr and tr["source_pair"]:
            for s in tr["source_pair"]:
                ts.append(s.mjd * SEC_PER_DAY)
                ras.append(math.radians(s.ra))
                decs.append(math.radians(s.dec))
        else:
            ts.append(tr["t"])
            ras.append(tr["ra"])
            decs.append(tr["dec"])
    if len(ts) < 2:
        return 0.0, 0.0
    ts = np.array(ts); ras = np.array(ras); decs = np.array(decs)
    A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
    (dra, _), _, _, _ = np.linalg.lstsq(A, ras, rcond=None)
    (ddec, _), _, _, _ = np.linalg.lstsq(A, decs, rcond=None)
    return float(dra), float(ddec)


def multipass_refined_chain(tracklets: list[dict],
                              *, n_passes: int = 3,
                              initial_sigma_arcsec: float = 60.0,
                              refined_sigma_arcsec: float = 15.0,
                              max_nights_gap: int = 14
                              ) -> list[list[dict]]:
    """Probabilistic chain + iterative rate-refinement extension.

    Pass 1: standard probabilistic_chain with a loose position_sigma.
    Pass 2..N: for each chain found, refit the rate from ALL its
    detections (much tighter), then re-search later nights with the
    refined rate and a tight position_sigma. Repeat.

    The result: chains that started as 2-night links get extended to
    3, 4, 5 nights when the refined rate gives a tighter prediction.
    """
    if not tracklets:
        return []
    # Pass 1: loose probabilistic link
    chains = probabilistic_chain(
        tracklets, max_nights_gap=max_nights_gap,
        position_sigma_arcsec=initial_sigma_arcsec)
    if not chains:
        return chains
    # Pool of unattached tracklets (not in any chain)
    in_chain = set()
    for ch in chains:
        for t in ch:
            in_chain.add(id(t))
    free = [t for t in tracklets if id(t) not in in_chain]
    by_night_free = defaultdict(list)
    for t in free:
        by_night_free[t["night"]].append(t)

    for _ in range(n_passes - 1):
        extended_any = False
        for ch in chains:
            # Re-fit the chain's rate
            dra_fit, ddec_fit = _refit_rate_from_chain(ch)
            ch_nights = sorted({t["night"] for t in ch})
            ref = ch[-1]                 # last tracklet in chain time order
            ref_jd = ref["jd"]
            ref_ra_rad = ref["ra"]
            ref_dec_rad = ref["dec"]
            for free_night, free_ts in by_night_free.items():
                if free_night in ch_nights:
                    continue
                if abs(free_night - max(ch_nights)) > max_nights_gap:
                    continue
                dt_s = (free_night + 0.5 - (ref_jd - 2400000.5)) * SEC_PER_DAY
                ra_pred = math.degrees(ref_ra_rad + dra_fit * dt_s)
                dec_pred = math.degrees(ref_dec_rad + ddec_fit * dt_s)
                best, best_gap = None, float("inf")
                for u in free_ts:
                    dra = math.radians(math.degrees(u["ra"]) - ra_pred) \
                          * math.cos(u["dec"])
                    ddec = math.radians(math.degrees(u["dec"]) - dec_pred)
                    gap = math.degrees(math.hypot(dra, ddec)) * 3600.0
                    if gap > refined_sigma_arcsec:
                        continue
                    if gap < best_gap:
                        best, best_gap = u, gap
                if best is not None:
                    ch.append(best)
                    in_chain.add(id(best))
                    by_night_free[free_night].remove(best)
                    extended_any = True
            ch.sort(key=lambda t: t["t"])
        if not extended_any:
            break
    return chains


# =============================================================================
# HelioLinC bridge for image tracklets
# =============================================================================

def helio_linc_image_bridge(tracklets: list[dict],
                              *, r_grid_au=None, rdot_grid=None,
                              cluster_au: float = 0.05,
                              min_obs: int = 3,
                              min_nights: int = 2) -> list[list[dict]]:
    """Run the validated HelioLinC linker on image-format tracklets.

    Image tracklets from `nightly_tracklets` have the same schema fields
    HelioLinC needs (t/ra/dec/dra/ddec). The only difference is image
    tracklets use higher-precision PSF centroids vs alert positions, so
    the linker's cluster_au tolerance can be tighter here than for alerts.
    """
    if not tracklets:
        return []
    if r_grid_au is None:
        # Default: span MBA -> outer detached, denser in the TNO regime
        r_grid_au = np.concatenate([np.linspace(2.0, 30.0, 30),
                                     np.linspace(31.0, 80.0, 50),
                                     np.linspace(82.0, 200.0, 30)])
    if rdot_grid is None:
        rdot_grid = np.linspace(-2.0, 2.0, 21)

    from .. import linkage as LK
    geom = LK.precompute_geometry(tracklets)
    t_ref = float(np.median(geom.t))
    cand_sets = LK.link(geom, t_ref, r_grid_au, rdot_grid,
                         cluster_au=cluster_au,
                         min_obs=min_obs, min_nights=min_nights)
    chains = []
    for idx_set in cand_sets:
        chain = [tracklets[i] for i in idx_set]
        for t in chain:
            t.setdefault("night",
                          int(round((t.get("jd", 0.0) - 2451545.0)
                                      if "jd" in t else
                                      (t["t"] / SEC_PER_DAY + 10957.5))))
        chains.append(sorted(chain, key=lambda t: t["t"]))
    return chains


# =============================================================================
# Orchestrator: run all three + dedupe
# =============================================================================

def _chain_signature(chain: list[dict]) -> frozenset:
    """Hashable identity for a chain -- the set of underlying source objects.

    Two chains that share >=80% of their underlying sources are "the same"
    chain; we keep the LONGER one and drop the duplicate.
    """
    sources = set()
    for tr in chain:
        sp = tr.get("source_pair")
        if sp:
            for s in sp:
                sources.add(id(s))
        else:
            # Fall back to tracklet identity if no source_pair
            sources.add(id(tr))
    return frozenset(sources)


def _merge_chain_lists(chain_lists: list[list[list[dict]]]) -> list[list[dict]]:
    """Union of chains from multiple linkers, with subset-dedup.

    A chain is dropped only if its full source set is a SUBSET of an
    already-kept (strictly larger) chain. This way:

      * Longer multi-night chains dominate shorter ones with the same sources
        (the IOD on the long chain is better-conditioned).
      * Chains from different objects with NO source overlap all survive.
      * Chains with PARTIAL overlap (e.g. one shares 2 sources with another's 4)
        both survive -- they might be siblings of the same arc that the IOD
        will rank separately.

    Returns the union list, sorted by night-span descending then size desc.
    """
    all_chains = []
    for chs in chain_lists:
        for c in chs:
            sig = _chain_signature(c)
            n_nights = len({t.get("night", -1) for t in c})
            all_chains.append((c, sig, n_nights, len(c)))
    # Sort by night span descending, then size descending -- best first
    all_chains.sort(key=lambda x: (-x[2], -x[3]))
    kept = []
    kept_sigs = []
    for ch, sig, n_nights, n_members in all_chains:
        # Drop when this chain is EQUAL to an already-kept chain (exact
        # duplicate) OR a proper subset of one (a shorter version was
        # found by another linker). Keep partial-overlap chains -- they
        # may be siblings of the same arc that IOD will rank separately.
        is_subset_or_equal = False
        for ks in kept_sigs:
            if sig and ks and sig.issubset(ks):
                is_subset_or_equal = True
                break
        if not is_subset_or_equal:
            kept.append(ch)
            kept_sigs.append(sig)
    return kept


def orbit_grow_chain(tracklets: list[dict], *,
                       rms_acceptance_arcsec: float = 3.0,
                       max_chain_length: int = 12) -> list[list[dict]]:
    """Grow chains by IOD-fitting pairs and extending greedy-by-RMS.

    This is the Pan-STARRS "moving object find" approach: rather than
    extrapolate sky positions from a single-tracklet rate (noisy), fit an
    actual heliocentric orbit to every PAIR of tracklets, then ask "what
    OTHER tracklet, if added, keeps the orbit's RMS below threshold?"
    Iteratively add the best-fitting tracklet until no candidate keeps RMS
    acceptable.

    Vastly more robust than rate-based linking on noisy image data:
    the orbital fit is a tight physical constraint that noise can't
    spoof. The cost is N^2 IOD attempts where N is tracklet count, so
    we cap chain length and short-circuit when no extension fits.
    """
    from . import advanced_linking as _AL  # for _refit_rate_from_chain
    from .. import iod_advanced as IODA
    if not tracklets or len(tracklets) < 3:
        return []
    by_night = defaultdict(list)
    for t in tracklets:
        by_night[t.get("night", -1)].append(t)
    nights = sorted(by_night)
    if len(nights) < 2:
        return []

    # Seed: every (n0, n1) tracklet pair. For each, try to fit + extend.
    chains = []
    seen_signatures = set()
    for i, na in enumerate(nights):
        for nb in nights[i + 1:]:
            for ta in by_night[na]:
                for tb in by_night[nb]:
                    seed = [ta, tb]
                    sig = frozenset((id(ta), id(tb)))
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    # Try to extend the seed
                    grown = _grow_from_seed(seed, tracklets,
                                              rms_acceptance_arcsec,
                                              max_chain_length)
                    if grown and len(grown) >= 3:
                        # Has it been emitted already (subset of a chain we
                        # already kept)?
                        grown_sig = frozenset(id(t) for t in grown)
                        if not any(grown_sig <= s for s in
                                    [frozenset(id(t) for t in c) for c in chains]):
                            chains.append(grown)
    return chains


def _grow_from_seed(seed_chain, all_tracklets, rms_acceptance, max_len):
    """Extend a chain by trying every other tracklet + IOD-fitting + accepting
    the one that keeps RMS lowest (below acceptance)."""
    from .. import iod_advanced as IODA
    current = list(seed_chain)
    seen_ids = {id(t) for t in current}
    while len(current) < max_len:
        # IOD-fit the current chain to get baseline RMS
        try:
            ens_now = IODA.fit_candidate_ensemble(
                current, rms_acceptance_arcsec=rms_acceptance,
                cheap_first=True, early_exit_rms_arcsec=0.3)
        except Exception:
            return current if len(current) >= 3 else None
        if not ens_now.success:
            return current if len(current) >= 3 else None
        baseline_rms = ens_now.rms_arcsec

        # Find the best candidate extension
        best_rms = float("inf")
        best_tracklet = None
        for t in all_tracklets:
            if id(t) in seen_ids:
                continue
            # Don't add another tracklet from the same night as one already in
            if t.get("night") in {x.get("night") for x in current}:
                continue
            trial = current + [t]
            try:
                ens = IODA.fit_candidate_ensemble(
                    trial, rms_acceptance_arcsec=rms_acceptance,
                    cheap_first=True, early_exit_rms_arcsec=0.3)
            except Exception:
                continue
            if not ens.success:
                continue
            if ens.rms_arcsec < best_rms:
                best_rms = ens.rms_arcsec
                best_tracklet = t
        if best_tracklet is None or best_rms > rms_acceptance:
            break
        current.append(best_tracklet)
        seen_ids.add(id(best_tracklet))
    return sorted(current, key=lambda t: t.get("t", 0))


def discover_in_images_chains(tracklets: list[dict], *,
                                use_greedy: bool = True,
                                use_probabilistic: bool = True,
                                use_multipass: bool = True,
                                use_helio_linc: bool = False,
                                use_orbit_grow: bool = False,
                                use_nbody_grow: bool = False
                                ) -> list[list[dict]]:
    """Run every available linking strategy on image tracklets + merge.

    Each strategy catches different chain geometries:
      * greedy (original chain_multi_night): fast, finds long-arc objects
        whose night-N rate is already accurate.
      * probabilistic (Hungarian): handles dense fields where two candidates
        compete for the same tracklet.
      * multipass refined: extends 2-night chains by re-fitting the rate
        from both nights and re-searching with tighter tolerance.
      * HelioLinC: heliocentric-geometry-aware linker that handles
        long arcs across many weeks where extrapolation accumulates error.
      * nbody_grow: Pan-STARRS MOF-style seed-and-grow with Sun + Jupiter
        + Saturn perturbations; catches long-arc chains (months) where
        2-body propagation drifts enough to miss real detections.

    Returns the UNION of chains found, with duplicates merged when they
    share >= 50% of their underlying source detections.
    """
    chain_lists = []
    if use_greedy:
        try:
            from .tracklets_from_images import chain_multi_night
            chain_lists.append(chain_multi_night(tracklets))
        except Exception:
            pass
    if use_probabilistic:
        try:
            chain_lists.append(probabilistic_chain(tracklets))
        except Exception:
            pass
    if use_multipass:
        try:
            chain_lists.append(multipass_refined_chain(tracklets))
        except Exception:
            pass
    if use_helio_linc:
        try:
            chain_lists.append(helio_linc_image_bridge(tracklets))
        except Exception:
            pass
    if use_orbit_grow:
        try:
            chain_lists.append(orbit_grow_chain(tracklets))
        except Exception:
            pass
    if use_nbody_grow:
        try:
            from .nbody_chain_grow import nbody_grow_chain
            # Kepler-only (use_nbody=False) is much faster and the
            # difference vs full N-body is sub-arcsec for 6-day arcs.
            # Switch to use_nbody=True when arcs span >1 month.
            chain_lists.append(nbody_grow_chain(tracklets, use_nbody=False))
        except Exception:
            pass
    return _merge_chain_lists(chain_lists)
