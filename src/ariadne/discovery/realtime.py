"""Realtime discovery pipeline: broker alerts -> tracklets -> IOD+LM -> SkyBoT.

The end-to-end filter that turns ~1M nightly broker alerts into a handful of real
moving-object candidates worth following up. Stages:

  1. PULL: subscribe to a broker (ALeRCE / Lasair / ANTARES) over a sky+time window.
  2. CLUSTER same-night nearby alerts into single-night detections (group by ~1
     arcsec position match within ~1 night).
  3. PAIR multi-night detections into tracklets when the apparent motion is
     consistent with a moving solar-system object (rate between 0.05 and 5 arcsec/hr
     for distant TNOs to inner-Main-Belt; faster for NEOs).
  4. IOD + LM fit a Keplerian orbit; reject high-RMS clusters (false positives).
  5. CROSS-MATCH each surviving candidate against SkyBoT to identify whether it's
     a known object. Anything not in SkyBoT is a discovery candidate.

This is intentionally a SHARP, CONSERVATIVE filter -- most alerts are real
sub-transients (supernovae, AGN flares, variable stars), not solar-system objects.
The pipeline rejects MOST input and surfaces only candidates with low-RMS orbit
fits + no known cross-match.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Iterable

import numpy as np

from .brokers.base import Alert
from . import iod as IOD
from . import linkage as LK
from .operations.replay import stable_hash


# ---------- stage 1: pull (broker-dependent; caller drives the broker query) ----


# ---------- stage 2: cluster same-night same-position alerts ------------------
def cluster_same_night(alerts: Iterable[Alert], pos_tol_arcsec: float = 1.0,
                       time_tol_days: float = 0.5):
    """Group alerts that fall within `pos_tol_arcsec` and `time_tol_days` of each other.

    Returns a list of clusters (each a list of Alert). Greedy single-link clustering
    in (RA, Dec, MJD) -- O(N^2) but N is small per query.
    """
    alerts = list(alerts)
    clusters = []
    visited = set()
    for i, a in enumerate(alerts):
        if i in visited:
            continue
        cluster = [a]
        visited.add(i)
        for j in range(i + 1, len(alerts)):
            if j in visited:
                continue
            b = alerts[j]
            if abs(b.mjd - a.mjd) > time_tol_days:
                continue
            dra = math.radians(b.ra - a.ra) * math.cos(math.radians(a.dec))
            ddec = math.radians(b.dec - a.dec)
            sep_arcsec = math.degrees(math.hypot(dra, ddec)) * 3600.0
            if sep_arcsec <= pos_tol_arcsec:
                cluster.append(b)
                visited.add(j)
        clusters.append(cluster)
    return clusters


def cluster_centroid(cluster: list[Alert]) -> Alert:
    """Average position / time of a same-night cluster as a single Alert.

    Preserves the first member's meta so downstream code (recovery harness
    truth_id matching, sensitivity validation) can still trace the centroid
    back to its underlying source.
    """
    n = len(cluster)
    mjd = sum(a.mjd for a in cluster) / n
    ra = sum(a.ra for a in cluster) / n
    dec = sum(a.dec for a in cluster) / n
    mag = sum(a.mag for a in cluster if a.mag > -50) / max(1,
            sum(1 for a in cluster if a.mag > -50))
    # Inherit the first member's meta + record the cluster size
    meta = dict(cluster[0].meta) if cluster[0].meta else {}
    meta["n_alerts"] = n
    return Alert(survey=cluster[0].survey, alert_id=cluster[0].alert_id,
                 obj_id=cluster[0].obj_id, mjd=mjd, ra=ra, dec=dec, mag=mag,
                 band=cluster[0].band, meta=meta)


# ---------- stage 3: pair multi-night detections into tracklets ---------------
def build_tracklets(detections: list[Alert],
                    min_rate_arcsec_hr: float = 0.05,
                    max_rate_arcsec_hr: float = 5.0,
                    min_pair_dt_hours: float = 0.5,
                    max_pair_dt_hours: float = 6.0) -> list[dict]:
    """Pair detections separated by < a night into tracklets with a derived on-sky rate.

    Default rate window [0.05, 5] arcsec/hr targets the distant-object regime
    (TNOs ~1-3 arcsec/hr; outer-MBA ~5; faster movers excluded for the TNO use case).
    The pair must be from the same band-class survey to avoid colour-drift artefacts.

    Returns list of tracklet dicts with t (seconds past J2000), jd, ra, dec, dra, ddec,
    rate_arcsec_hr, desig (synthetic).
    """
    SEC_PER_DAY = 86400.0
    tracks = []
    detections = sorted(detections, key=lambda d: d.mjd)
    n = len(detections)
    for i in range(n):
        a = detections[i]
        for j in range(i + 1, n):
            b = detections[j]
            dt_h = (b.mjd - a.mjd) * 24.0
            if dt_h < min_pair_dt_hours:
                continue
            if dt_h > max_pair_dt_hours:
                break
            if b.survey != a.survey:
                continue
            dra = math.radians(b.ra - a.ra) * math.cos(math.radians(a.dec))
            ddec = math.radians(b.dec - a.dec)
            rate_arcsec_hr = math.degrees(math.hypot(dra, ddec)) * 3600.0 / dt_h
            if not (min_rate_arcsec_hr <= rate_arcsec_hr <= max_rate_arcsec_hr):
                continue
            jd_mid = 0.5 * (a.jd + b.jd)
            et_mid = (jd_mid - 2451545.0) * SEC_PER_DAY
            dt_s = (b.jd - a.jd) * SEC_PER_DAY
            tracks.append({
                "t": et_mid, "jd": jd_mid,
                "ra": math.radians(0.5 * (a.ra + b.ra)),
                "dec": math.radians(0.5 * (a.dec + b.dec)),
                "dra": (math.radians(b.ra - a.ra)) / dt_s,
                "ddec": (math.radians(b.dec - a.dec)) / dt_s,
                "rate_arcsec_hr": rate_arcsec_hr,
                "desig": f"{a.survey}_{a.alert_id}_{b.alert_id}",
                "obscode": "I41",        # Palomar (ZTF); pick by survey if extending
                "obj": -1,
                "members": [a, b],
            })
    return tracks


# ---------- stage 3b: chain multi-night tracklets ----------------------------
def chain_tracklets(tracklets: list[dict],
                    max_position_gap_arcsec: float = 60.0,
                    max_rate_change_pct: float = 50.0,
                    max_nights_gap: int = 14) -> list[list[dict]]:
    """Group tracklets that belong to the same multi-night candidate object."""
    SEC_PER_DAY = 86400.0
    if not tracklets:
        return []
    for t in tracklets:
        t["night"] = int(round((t["jd"] - 2451545.0)))
    by_night = defaultdict(list)
    for t in tracklets:
        by_night[t["night"]].append(t)
    nights = sorted(by_night)
    chains = {id(t): [t] for tlist in by_night.values() for t in tlist}
    for k, night in enumerate(nights):
        for t in by_night[night]:
            for later in nights[k + 1: k + 1 + max_nights_gap]:
                dt_days = later - night
                dra_per_day = t["dra"] * SEC_PER_DAY
                ddec_per_day = t["ddec"] * SEC_PER_DAY
                ra_pred = math.degrees(t["ra"] + dra_per_day * dt_days)
                dec_pred = math.degrees(t["dec"] + ddec_per_day * dt_days)
                best, best_d = None, float("inf")
                for u in by_night[later]:
                    dra = math.radians(math.degrees(u["ra"]) - ra_pred) * math.cos(u["dec"])
                    ddec = math.radians(math.degrees(u["dec"]) - dec_pred)
                    gap = math.degrees(math.hypot(dra, ddec)) * 3600.0
                    if gap > max_position_gap_arcsec:
                        continue
                    drate_pct = (abs(u["rate_arcsec_hr"] - t["rate_arcsec_hr"])
                                 / max(t["rate_arcsec_hr"], 1e-9) * 100.0)
                    if drate_pct > max_rate_change_pct:
                        continue
                    if gap < best_d:
                        best, best_d = u, gap
                if best is not None:
                    src, dst = chains[id(t)], chains[id(best)]
                    if src is not dst:
                        merged = src + dst
                        for x in merged:
                            chains[id(x)] = merged
                    break
    seen = set(); out = []
    for chain in chains.values():
        cid = id(chain)
        if cid in seen:
            continue
        seen.add(cid)
        if len({t["night"] for t in chain}) >= 2:
            out.append(sorted(chain, key=lambda t: t["t"]))
    return out


# ---------- stage 4: IOD + LM filter ------------------------------------------
def fit_filter(tracklets: list[dict], rms_threshold_arcsec: float = 10.0):
    """Run IOD+LM on each tracklet group; keep only those with fit RMS below threshold.

    IOD.fit_candidate expects TRACKLET DICTS (with t/ra/dec/dra/ddec) -- it
    runs the linker's geometry precompute over them. The Alert `members` are
    only useful for n-detection counting and arc-day estimation.

    For multi-night chains: feed the contained tracklets (NOT the flattened
    Alert list) to IOD. For single 2-point tracklets without a chain, tag
    as 'unfittable_single_arc' (4 observations < 6 unknowns).
    """
    out = []
    for tr in tracklets:
        # Build the tracklet list for IOD. If we have a chain, USE IT.
        if isinstance(tr.get("chain"), list) and len(tr["chain"]) >= 3:
            iod_input = tr["chain"]
        else:
            tr_out = dict(tr); tr_out["status"] = "unfittable_single_arc"
            tr_out["rms_arcsec"] = None
            out.append(tr_out)
            continue
        # Make sure the flattened members are still attached for downstream
        # (smart_annotate / nightly use them for n_detections + arc_days).
        members = [m for sub in tr["chain"] for m in sub.get("members", [])]
        tr["members"] = members

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = IOD.fit_candidate(iod_input)
        except Exception as e:
            tr_out = dict(tr); tr_out["status"] = f"fit_error: {str(e)[:60]}"
            tr_out["rms_arcsec"] = None
            out.append(tr_out); continue
        if fit is None:
            tr_out = dict(tr); tr_out["status"] = "iod_failed"
            tr_out["rms_arcsec"] = None
            out.append(tr_out); continue
        tr_out = dict(tr)
        tr_out["rms_arcsec"] = fit["rms_arcsec"]
        tr_out["x_fit_km"] = fit["x_fit"].tolist()
        tr_out["v_fit_kms"] = fit["v_fit"].tolist()
        tr_out["status"] = ("accepted" if fit["rms_arcsec"] < rms_threshold_arcsec
                            else "high_rms_rejected")
        out.append(tr_out)
    return out


# ---------- stage 5: SkyBoT cross-match ---------------------------------------
def skybot_xmatch(tracklets: list[dict], cone_arcmin: float = 4.0,
                  rate_limit_s: float = 0.35):
    """Annotate each tracklet with SkyBoT cross-match results.

    Cross-match each accepted tracklet's mean (RA, Dec, JD) against the SkyBoT
    known-object catalogue. Anything with `n_known == 0` is a candidate that's NOT
    a known object -- the discovery-candidate flag.
    """
    import time
    try:
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        from astropy.time import Time
        from astroquery.imcce import Skybot
    except ImportError as e:
        raise RuntimeError("astroquery + astropy required for SkyBoT cross-match") from e

    annotated = []
    for tr in tracklets:
        if tr.get("status") != "accepted":
            tr_a = dict(tr); tr_a["xmatch"] = {"skipped": tr["status"]}
            annotated.append(tr_a); continue
        try:
            coord = SkyCoord(math.degrees(tr["ra"]) * u.deg,
                             math.degrees(tr["dec"]) * u.deg, frame="icrs")
            epoch = Time(tr["jd"], format="jd")
            r = Skybot.cone_search(coord, cone_arcmin * u.arcmin, epoch)
            n_known = len(r) if r is not None else 0
            names = list(r["Name"])[:3] if n_known > 0 else []
        except Exception as e:
            n_known = -1; names = [str(e)[:60]]
        tr_a = dict(tr)
        tr_a["xmatch"] = {"n_known": int(n_known), "names": names}
        annotated.append(tr_a)
        time.sleep(rate_limit_s)
    return annotated


# ---------- stage 3c: HelioLinC linker (alternative to extrapolation chainer) -
def link_helio_linc(tracklets: list[dict],
                    r_grid_au=None, rdot_grid=None,
                    cluster_au: float = 0.2, min_obs: int = 3,
                    min_nights: int = 2) -> list[list[dict]]:
    """Run the full HelioLinC (r, rdot) hypothesis sweep to group tracklets.

    This is the validated linker from Stage 39: it hypothesises a heliocentric
    distance + radial velocity, maps every tracklet to a heliocentric state,
    propagates to a common reference epoch, and clusters in 6D. The clusters ARE
    the candidate multi-tracklet arcs.

    Compared to `chain_tracklets` (greedy extrapolation): HelioLinC is more
    expensive but recovers more genuine long-arc associations (especially when
    nights are >2 weeks apart) and uses real Kepler propagation, not linear
    extrapolation. Use for the high-budget pipeline; use chain_tracklets for
    quick-look nightly runs.

    Returns: list of "chains" (list of tracklet dicts) in the same format as
    chain_tracklets so the downstream fit_filter can consume either.
    """
    import numpy as np
    if not tracklets:
        return []
    if r_grid_au is None:
        # default: a coarse grid spanning MBA -> TNO. For deep TNO-only sweeps,
        # override with r_grid_au=np.linspace(30, 100, 30).
        r_grid_au = np.linspace(2.0, 80.0, 30)
    if rdot_grid is None:
        rdot_grid = np.linspace(-2.0, 2.0, 17)

    geom = LK.precompute_geometry(tracklets)
    t_ref = float(np.median(geom.t))
    cluster_km = cluster_au * 149597870.7
    cand_sets = LK.link(geom, t_ref, r_grid_au, rdot_grid,
                        cluster_au=cluster_km / 149597870.7,
                        min_obs=min_obs, min_nights=min_nights)
    # rebuild as list-of-tracklets (chains), matching chain_tracklets' output
    chains = []
    SEC_PER_DAY = 86400.0
    for idx_set in cand_sets:
        chain = [tracklets[i] for i in idx_set]
        # tag the chain members with the night they sit in (downstream uses)
        for t in chain:
            t.setdefault("night", int(round((t["jd"] - 2451545.0)
                                              if "jd" in t else
                                              (t["t"] / SEC_PER_DAY + 10957.5))))
        chains.append(sorted(chain, key=lambda t: t["t"]))
    return chains


# ---------- stage 5b: smart-layer annotations --------------------------------
def smart_annotate(tracklets: list[dict]) -> list[dict]:
    """Run real-bogus filter + inference engine + quality scorer on accepted tracklets.

    Adds these fields to each tracklet:
      _realbogus       RealBogusVerdict with bogus_score + rules_fired + is_real
      _inference       InferenceResult.best.label + posterior + entropy + grade
      _quality_grade   A/B/C/D quality grade (from scoring module)
      _quality_score   0..1 total quality score
      _taxonomy        OrbitClass label if the fitted orbit is a moving object

    Only runs on tracklets with status='accepted'. Cheap (sub-millisecond per
    candidate); always safe to call on every nightly run.
    """
    from . import realbogus
    from . import inference
    from . import scoring
    from . import taxonomy
    from .operations.candidate_store import Candidate
    import numpy as np

    out = []
    for tr in tracklets:
        if tr.get("status") != "accepted":
            out.append(tr); continue

        # 1. Real-bogus rule filter
        rb = realbogus.score_realbogus(tr)
        tr["_realbogus"] = rb

        # If the rules say it's bogus, demote it before the inference engine
        if not rb.is_real:
            tr["status"] = "rejected_by_realbogus"
            out.append(tr); continue

        # 2. Orbital taxonomy from the fitted orbit
        orbital_class = None
        if "x_fit_km" in tr and "v_fit_kms" in tr:
            try:
                t = taxonomy.classify_state(
                    np.asarray(tr["x_fit_km"]), np.asarray(tr["v_fit_kms"]))
                orbital_class = t.label
                tr["_taxonomy"] = {"label": t.label, "confidence": t.confidence}
            except Exception:
                pass

        # 3. Inference engine -- cognitive fusion over the evidence
        ev = inference.Evidence(
            mjd=float(tr["jd"] - 2400000.5) if "jd" in tr else None,
            ra_deg=math.degrees(tr["ra"]),
            dec_deg=math.degrees(tr["dec"]),
            rate_arcsec_hr=float(tr.get("rate_arcsec_hr", 0.0)),
            apparent_mag=None,
            n_detections=len(tr.get("members", [])),
            arc_days=(max(m.mjd for m in tr.get("members", []))
                       - min(m.mjd for m in tr.get("members", [])))
                       if tr.get("members") else 0.0,
            rms_arcsec=float(tr.get("rms_arcsec") or 0.0),
            orbit_state=(tr.get("x_fit_km", []) + tr.get("v_fit_kms", []))
                         if "x_fit_km" in tr else None,
            skybot_match_names=tr.get("xmatch", {}).get("names", []),
        )
        try:
            res = inference.infer(ev)
            tr["_inference"] = {
                "best_label": res.best.label,
                "best_class": res.best.class_,
                "orbital_class": res.best.orbital_class,
                "posterior": res.best.posterior,
                "entropy": res.entropy,
                "recommended_action": res.recommended_followup.get("action"),
            }
        except Exception as e:
            tr["_inference"] = {"error": str(e)[:120]}

        # 4. Quality scoring (build a transient Candidate just for scoring)
        try:
            arc = ev.arc_days
            cand = Candidate(
                key="transient", ra=ev.ra_deg, dec=ev.dec_deg,
                rate_arcsec_hr=ev.rate_arcsec_hr,
                first_seen_mjd=ev.mjd - arc if ev.mjd else 0.0,
                last_seen_mjd=ev.mjd or 0.0,
                n_runs=1,
                rms_history=[[ev.mjd or 0, ev.rms_arcsec]],
                skybot_names=ev.skybot_match_names or [],
            )
            qs = scoring.score_candidate(cand)
            tr["_quality_grade"] = qs.grade()
            tr["_quality_score"] = qs.total
        except Exception as e:
            tr["_quality_grade"] = "?"
            tr["_quality_score"] = 0.0

        out.append(tr)
    return out


# ---------- end-to-end ---------------------------------------------------------
def run_pipeline(alerts: Iterable[Alert],
                 cluster_pos_tol_arcsec: float = 1.0,
                 cluster_time_tol_days: float = 0.5,
                 rate_window_arcsec_hr: tuple[float, float] = (0.05, 5.0),
                 pair_dt_hours: tuple[float, float] = (0.5, 6.0),
                 rms_threshold_arcsec: float = 10.0,
                 do_xmatch: bool = True,
                 use_helio_linc: bool = False,
                 smart_layer: bool = True):
    """Run all 5 pipeline stages end-to-end. Returns the annotated tracklet list.

    Discovery candidates: `[t for t in result if t['status']=='accepted'
                            and t.get('xmatch', {}).get('n_known', 0) == 0]`

    When `smart_layer=True` (default), accepted candidates are annotated with
    real-bogus verdict, inference engine posterior, orbital taxonomy, and
    quality grade. The fields are stored on each tracklet under `_realbogus`,
    `_inference`, `_taxonomy`, and `_quality_grade` for downstream pipelines.
    """
    print(f"[1/5] pulled {len(list(alerts) if not isinstance(alerts, list) else alerts)} alerts")
    alerts = list(alerts)
    print(f"[2/5] clustering same-night alerts (pos {cluster_pos_tol_arcsec}\", "
          f"time {cluster_time_tol_days}d)...")
    clusters = cluster_same_night(alerts, cluster_pos_tol_arcsec, cluster_time_tol_days)
    centroids = [cluster_centroid(c) for c in clusters]
    print(f"      -> {len(centroids)} single-night detections")
    print(f"[3/5] building tracklets (rate {rate_window_arcsec_hr} as/hr, "
          f"pair-dt {pair_dt_hours}h)...")
    tracklets = build_tracklets(centroids, *rate_window_arcsec_hr, *pair_dt_hours)
    print(f"      -> {len(tracklets)} candidate single-night tracklets")
    # Chain multi-night tracklets so the IOD has 6+ detections (3+ tracklets x 2 each).
    if use_helio_linc:
        print(f"      using HelioLinC linker (full (r, rdot) hypothesis sweep)...")
        chains = link_helio_linc(tracklets)
    else:
        chains = chain_tracklets(tracklets)
    print(f"      -> {len(chains)} multi-night candidate arcs")
    # Convert chains into the tracklet-cluster format the fit_filter expects.
    chain_clusters = []
    for ch in chains:
        members = [m for sub in ch for m in sub.get("members", [])]
        # use the centroid of the chain as the cluster's nominal tracklet
        centroid = dict(ch[len(ch) // 2])
        centroid["members"] = members
        centroid["chain"] = ch
        chain_clusters.append(centroid)
    print(f"[4/5] IOD+LM filter (RMS threshold {rms_threshold_arcsec}\")...")
    fitted = fit_filter(chain_clusters, rms_threshold_arcsec)
    n_accepted = sum(1 for t in fitted if t.get("status") == "accepted")
    print(f"      -> {n_accepted} pass the orbit-fit filter")
    if do_xmatch and n_accepted > 0:
        print(f"[5/5] SkyBoT cross-match ({n_accepted} accepted candidates)...")
        annotated = skybot_xmatch(fitted)
    else:
        annotated = fitted
        if do_xmatch:
            print(f"[5/5] no candidates passed filter; skipping cross-match")
    # Smart-layer enrichment: real-bogus, inference, taxonomy, quality
    if smart_layer:
        print(f"[6/6] smart-layer annotation (realbogus + inference + taxonomy + scoring)...")
        annotated = smart_annotate(annotated)
        n_after_smart = sum(1 for t in annotated if t.get("status") == "accepted")
        n_grade_a = sum(1 for t in annotated if t.get("_quality_grade") == "A")
        print(f"      -> {n_after_smart} survive smart filter; {n_grade_a} grade-A")
    n_discovery = sum(1 for t in annotated
                      if t.get("status") == "accepted"
                      and t.get("xmatch", {}).get("n_known") == 0)
    print(f"\n  RESULT: {n_discovery} candidate(s) not matching any SkyBoT known object")
    return annotated


def run_pipeline_with_provenance(alerts: Iterable[Alert], *, ledger=None,
                                 source: str = "realtime", **kwargs):
    """Run `run_pipeline` and record input/output hashes in a provenance ledger."""
    alerts = list(alerts)
    if ledger is not None:
        ledger.record(event="pipeline_start", source=source, alerts=alerts,
                      parameters=kwargs)
    result = run_pipeline(alerts, **kwargs)
    if ledger is not None:
        ledger.record(event="pipeline_complete", source=source, alerts=alerts,
                      outputs=result, parameters={
                          **kwargs,
                          "input_hash": stable_hash([a.__dict__ for a in alerts]),
                      })
    return result
