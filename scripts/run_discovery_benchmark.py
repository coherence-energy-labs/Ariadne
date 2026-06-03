"""Truthed END-TO-END discovery benchmark on a good-cadence near-ecliptic field.

This is the instrument that's been missing: run the FULL discovery chain on
real multi-night pixels where we know the truth, and measure it.

  per exposure : extract sources (fixed auto-PSF detection)
  per night    : pool sources -> within-night tracklets (nightly_tracklets)
  cross-match  : label tracklets that match a KNOWN asteroid (validated N-body
                 ephemeris) -> the truth set is knowns visible+detectable on >=2 nights
  link         : discover_in_images_chains across nights (HelioLinC + multipass
                 + n-body grow) on ALL tracklets
  measure      :
    - KNOWN RECOVERY: of recoverable knowns (>=2 nights), how many are recovered
      as a multi-night chain? (does the discovery machinery actually work?)
    - UNKNOWN CANDIDATES: chains not matching any known, vetted (>=2 nights,
      rate/mag consistency) -> potential discoveries
    - FALSE FLOOR: scrambled control (offset one night) -> chance chains
    - IOD (optional): fit an orbit on the best candidates

Extraction is cached per exposure (.npz) so re-runs are fast. PERF: the PSF
FWHM is measured ONCE per exposure (median of a few CCDs) and reused for all
CCDs -- per-CCD auto-measurement is more accurate but ~2x slower.
"""
from __future__ import annotations

import os
import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from astropy.utils.exceptions import AstropyWarning
    warnings.simplefilter("ignore", AstropyWarning)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OBS = "807"
DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def extract_exposure(path, sigma, n_ccd, cache_dir):
    """Extract sources from one exposure (all CCDs). PSF measured once/exposure.
    Cached to .npz. Returns (ra, dec, mag, flux, mjd)."""
    cache = Path(cache_dir) / (Path(path).stem + f".s{sigma:.0f}.npz")
    if cache.exists():
        c = np.load(cache)
        # older caches omit 'flux'; derive it from magnitude (relative scale only,
        # used for brightest-N capping) so legacy caches stay usable.
        flux = c["flux"] if "flux" in c.files else np.power(10.0, -0.4 * np.asarray(c["mag"], float))
        return c["ra"], c["dec"], c["mag"], flux, float(c["mjd"])
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.trailed_rate import measure_image_fwhm
    inst = load_decam_instcal(str(path), read_dqm=False)
    mjd = inst.mjd
    ccds = [c for c in inst.ccds if c.wcs is not None and c.magzero > 0][:n_ccd]
    # measure PSF FWHM once (median over up to 4 central CCDs) -> reuse
    fwhms = []
    for c in ccds[:4]:
        m = measure_image_fwhm(np.asarray(c.science, float), fwhm_guess=4.0)
        if m:
            fwhms.append(m)
    fwhm = float(np.median(fwhms)) if fwhms else 4.0
    ra, dec, mag, flux = [], [], [], []
    for c in ccds:
        try:
            srcs = detect_sources_in_image(
                np.asarray(c.science, float), c.wcs, mjd=mjd, image_id=Path(path).stem,
                fwhm_px=fwhm, threshold_sigma=sigma, zeropoint_mag=c.magzero,
                auto_fwhm=False)
        except Exception:
            continue
        for s in srcs:
            ra.append(s.ra); dec.append(s.dec); mag.append(s.mag); flux.append(s.flux)
    ra = np.array(ra); dec = np.array(dec); mag = np.array(mag); flux = np.array(flux)
    np.savez(cache, ra=ra, dec=dec, mag=mag, flux=flux, mjd=mjd, fwhm=fwhm)
    return ra, dec, mag, flux, mjd


def _source_key(s):
    return (s.image_id, round(float(s.ra), 7), round(float(s.dec), 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/decam_discovery_field")
    ap.add_argument("--n-ccd", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--min-rate", type=float, default=2.0, help="arcsec/hr")
    ap.add_argument("--max-rate", type=float, default=120.0)
    ap.add_argument("--match-arcsec", type=float, default=2.5)
    ap.add_argument("--max-per-night", type=int, default=1200,
                    help="cap same-night tracklets per night before cross-night linking")
    ap.add_argument("--max-sources-per-exposure", type=int, default=1800,
                    help="keep brightest detections per exposure before pair search; 0 disables")
    ap.add_argument("--max-detections-per-exposure", type=int, default=0,
                    help="skip whole exposures with more detections than this (noise-inflated / "
                         "poor-quality frames); 0 disables")
    ap.add_argument("--stationary-veto-arcsec", type=float, default=0.8,
                    help="remove repeated same-night fixed sources before tracklet pairing; 0 disables")
    ap.add_argument("--truth-limit", type=int, default=300000,
                    help="catalog rows to scan for known-object truth labels")
    ap.add_argument("--truth-mode", choices=("fast", "nbody"), default="fast",
                    help="fast uses vectorized 2-body truth after field prefilter; nbody is slower")
    ap.add_argument("--linker-mode", choices=("fast", "full", "rate", "rate-coherence"), default="fast",
                    help="fast uses greedy/probabilistic/multipass; full adds HelioLinC/nbody "
                         "grow; rate uses the scalable rate-constrained linker (>=3-pt "
                         "within-night tracks + rate-extrapolated tight-box cross-night)")
    ap.add_argument("--rate-min-points", type=int, default=3,
                    help="min detections per within-night rate track (rate linker)")
    ap.add_argument("--rate-pos-tol", type=float, default=2.5,
                    help="within-night track growth tolerance, arcsec (rate linker)")
    ap.add_argument("--vet-mode", choices=("rules", "coherence"), default="rules",
                    help="rules = hard AND-thresholds; coherence = Equation-of-ONE "
                         "energy selector (gap22 pattern, dominates the rules in A/B)")
    ap.add_argument("--vet-eoo-tau", type=float, default=2.0,
                    help="EoO vetting energy threshold (lower = stricter)")
    ap.add_argument("--vet-min-nights", type=int, default=3)
    ap.add_argument("--vet-max-rate-cv", type=float, default=0.25)
    ap.add_argument("--vet-max-heading-scatter-deg", type=float, default=25.0)
    ap.add_argument("--vet-max-linear-resid-arcsec", type=float, default=45.0)
    ap.add_argument("--iod", action="store_true", help="run IOD on candidates (slow)")
    ap.add_argument("--out-dir", default="data/benchmarks/real_decam_discovery")
    args = ap.parse_args()

    from ariadne.discovery.imaging.source_extraction import Source
    from ariadne.discovery.imaging.tracklets_from_images import (
        nightly_tracklets,
        suppress_stationary_sources,
    )
    from ariadne.discovery.imaging.advanced_linking import discover_in_images_chains
    from ariadne.discovery.imaging.rate_constrained_linker import (
        build_within_night_tracks, link_rate_constrained, link_coherence)
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    files = sorted(Path(args.data_dir).glob("c4d_*_ooi_*_v*.fits.fz"))
    if not files:
        print(f"no exposures in {args.data_dir}"); return 1
    cache_dir = Path(args.data_dir) / "_cache"; cache_dir.mkdir(exist_ok=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} exposures", flush=True)

    # extract all exposures -> Source list, grouped by night (round mjd).
    # Truth matching uses every extracted source. Link building may use a
    # capped brightest subset for tractable pair search.
    raw_link_sources = []
    all_truth_sources = []
    by_night_truth_src = defaultdict(list)
    by_exposure_truth_src = defaultdict(list)
    skipped_exposures = []
    t0 = time.time()
    extraction_t = time.time()
    for f in files:
        try:
            ra, dec, mag, flux, mjd = extract_exposure(f, args.sigma, args.n_ccd, cache_dir)
        except Exception as exc:
            skipped_exposures.append({
                "path": str(f),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  SKIP {f.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if args.max_detections_per_exposure and len(ra) > args.max_detections_per_exposure:
            skipped_exposures.append({
                "path": str(f),
                "error": f"anomalous {len(ra)} detections > {args.max_detections_per_exposure} "
                         f"(noise-inflated / poor-quality frame)"})
            print(f"  SKIP {f.name}: anomalous {len(ra)} detections (poor frame)", flush=True)
            continue
        night = int(round(mjd))
        truth_ra, truth_dec, truth_mag, truth_flux = ra, dec, mag, flux
        for i in range(len(truth_ra)):
            s = Source(ra=float(truth_ra[i]), dec=float(truth_dec[i]), flux=float(truth_flux[i]),
                       mag=float(truth_mag[i]), fwhm_px=4.0, mjd=mjd,
                       image_id=f.stem, x=0.0, y=0.0)
            all_truth_sources.append(s); by_night_truth_src[night].append(s)
            by_exposure_truth_src[f.stem].append(s)
        # NOTE: do NOT brightness-cap here -- in crowded fields the brightest
        # detections are stars; capping before the stationary veto cuts the faint
        # asteroids. Build uncapped, veto stars, THEN cap the movers (below).
        for i in range(len(ra)):
            s = Source(ra=float(ra[i]), dec=float(dec[i]), flux=float(flux[i]),
                       mag=float(mag[i]), fwhm_px=4.0, mjd=mjd,
                       image_id=f.stem, x=0.0, y=0.0)
            raw_link_sources.append(s)
        print(f"  {f.name}: {len(ra)} det, night {night} ({time.time()-t0:.0f}s)", flush=True)
    all_sources = suppress_stationary_sources(raw_link_sources, args.stationary_veto_arcsec)
    # cap the NON-STATIONARY set per exposure (tractable linking) -- after the veto
    # so faint movers survive the crowded-field star bulk.
    if args.max_sources_per_exposure:
        _by_exp = defaultdict(list)
        for s in all_sources:
            _by_exp[s.image_id].append(s)
        capped = []
        for ss in _by_exp.values():
            if len(ss) > args.max_sources_per_exposure:
                ss = sorted(ss, key=lambda s: -s.flux)[:args.max_sources_per_exposure]
            capped.extend(ss)
        all_sources = capped
    by_night_src = defaultdict(list)
    for s in all_sources:
        by_night_src[int(round(s.mjd))].append(s)
    nights = sorted(by_night_src)
    extraction_s = time.time() - extraction_t
    if len(nights) < 2:
        summary = {
            "schema": "ariadne.real_decam_discovery_benchmark.v1",
            "status": "fail",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "fewer than two usable nights after exposure extraction",
            "usable_exposures": len(files) - len(skipped_exposures),
            "skipped_exposures": skipped_exposures,
        }
        (out_dir / "discovery_benchmark_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 2
    print(f"  {len(all_sources)} detections over {len(nights)} nights: {nights}", flush=True)
    print(f"  stationary veto removed {len(raw_link_sources) - len(all_sources)} linking detections", flush=True)
    print(f"  {len(all_truth_sources)} uncapped detections retained for truth matching", flush=True)

    # within-night tracklets
    stage_t = time.time()
    rate_track_objs = None          # populated only in --linker-mode rate
    track2dict = {}                 # id(Track) -> tracklet dict
    if args.linker_mode in ("rate", "rate-coherence"):
        rate_track_objs = []
        for n in nights:
            rate_track_objs += build_within_night_tracks(
                by_night_src[n], min_rate_arcsec_hr=args.min_rate,
                max_rate_arcsec_hr=args.max_rate, pos_tol_arcsec=args.rate_pos_tol,
                min_points=args.rate_min_points)
        tracks = []
        for tr in rate_track_objs:
            d = {"night": tr.night, "t": (tr.jd_mid - 2451545.0) * 86400.0,
                 "jd": tr.jd_mid, "ra": math.radians(tr.ra_mid),
                 "dec": math.radians(tr.dec_mid),
                 "dra": math.radians(tr.vra / 3600.0) / 3600.0,
                 "ddec": math.radians(tr.vdec / 3600.0) / 3600.0,
                 "rate_arcsec_hr": tr.rate_arcsec_hr, "source_pair": tr.sources,
                 "n_points": tr.n_points}
            track2dict[id(tr)] = d
            tracks.append(d)
    else:
        tracks = nightly_tracklets(all_sources, min_rate_arcsec_hr=args.min_rate,
                                    max_rate_arcsec_hr=args.max_rate,
                                    min_pair_dt_hours=0.03, max_pair_dt_hours=3.0,
                                    min_pair_separation_arcsec=0.5,
                                    max_per_night=args.max_per_night)
    tracklet_build_s = time.time() - stage_t
    print(f"  {len(tracks)} within-night tracklets"
          f"{' (>=3-pt rate tracks)' if args.linker_mode in ('rate','rate-coherence') else ''}", flush=True)
    print(f"  tracklet build: {tracklet_build_s:.1f}s", flush=True)

    # KNOWN cross-match: label each tracklet by the known it matches (if any),
    # and build the recoverable-known truth set (detectable on >=2 nights).
    db = open_db(DB)
    known_seen = defaultdict(set)     # designation -> set of nights it's detectable on
    source_known = {}
    truth_t = time.time()
    for image_id, ss in sorted(by_exposure_truth_src.items()):
        stage_t = time.time()
        ra = np.array([s.ra for s in ss]); dec = np.array([s.dec for s in ss])
        mjd = float(np.median([s.mjd for s in ss]))
        night = int(round(mjd))
        inf = pick_orbits_in_field(db, mjd, (ra.min(), ra.max()), (dec.min(), dec.max()),
                                     max_mag=22.0, limit_candidates=args.truth_limit)
        recs = [x[0] for x in inf]
        if args.truth_mode == "nbody":
            eph = bulk_ephemeris_at_mjd_nbody(
                recs, mjd, observer_geo_km=observatory_geo_km(OBS, mjd))
        else:
            from ariadne.discovery.imaging.mpc_ephemeris_batch import bulk_ephemeris_at_mjd
            eph = bulk_ephemeris_at_mjd(recs, mjd)
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            cd = math.cos(math.radians(eph[i, 1]))
            sep = np.hypot((ra - eph[i, 0]) * cd, dec - eph[i, 1]) * 3600
            if sep.size:
                j = int(np.argmin(sep))
                if float(sep[j]) <= args.match_arcsec:
                    designation = str(recs[i].designation)
                    source_known[_source_key(ss[j])] = designation
                    known_seen[designation].add(night)
        print(f"  truth exposure {image_id}: {len(recs)} catalog objects in field "
              f"({time.time() - stage_t:.1f}s)", flush=True)
    truth_s = time.time() - truth_t
    recoverable = {k for k, ns in known_seen.items() if len(ns) >= 2}
    print(f"  knowns detected on >=2 nights (recoverable truth set): {len(recoverable)}", flush=True)

    # label tracklets by known endpoint detections. This is stricter than a
    # night-median centroid match and avoids time-smearing moving targets.
    def label_tracklet(t):
        labels = [source_known.get(_source_key(s)) for s in t.get("source_pair", ())]
        labels = [x for x in labels if x is not None]
        if not labels:
            return None
        counts = defaultdict(int)
        for x in labels:
            counts[x] += 1
        return max(counts.items(), key=lambda kv: kv[1])[0]
    for t in tracks:
        t["known_id"] = label_tracklet(t)
    if args.linker_mode in ("rate", "rate-coherence"):
        for tr in rate_track_objs:        # carry label onto the Track for scrambling
            tr._known_id = track2dict[id(tr)]["known_id"]

    # cross-night linking
    stage_t = time.time()
    if args.linker_mode in ("rate", "rate-coherence"):
        _link = link_coherence if args.linker_mode == "rate-coherence" else link_rate_constrained
        track_chains = _link(rate_track_objs, min_nights=2)
        chains = [[track2dict[id(tr)] for tr in ch] for ch in track_chains]
    else:
        chains = discover_in_images_chains(
            tracks,
            use_helio_linc=args.linker_mode == "full",
            use_nbody_grow=args.linker_mode == "full",
            use_orbit_grow=args.linker_mode == "full",
            use_multi_hypothesis=args.linker_mode == "full",
        )
    multinight = [c for c in chains if len({t["night"] for t in c}) >= 2]
    linking_s = time.time() - stage_t
    print(f"  {len(chains)} chains, {len(multinight)} span >=2 nights", flush=True)
    print(f"  cross-night linking: {linking_s:.1f}s", flush=True)

    # KNOWN RECOVERY: a chain recovers known K if >=2 of its tracklets on >=2
    # nights are labelled K.
    recovered = set()
    unknown_chains = []
    for c in multinight:
        labels = defaultdict(set)
        for t in c:
            if t.get("known_id") is not None:
                labels[t["known_id"]].add(t["night"])
        hit = [k for k, ns in labels.items() if len(ns) >= 2]
        if hit:
            recovered.update(hit)
        elif all(t.get("known_id") is None for t in c):
            unknown_chains.append(c)
    rec_known = len(recovered & recoverable)
    print(f"\n=== END-TO-END DISCOVERY BENCHMARK (real {len(nights)}-night field) ===")
    print(f"  KNOWN multi-night recovery: {rec_known}/{len(recoverable)} "
          f"= {rec_known/max(len(recoverable),1)*100:.0f}%")
    print(f"  UNKNOWN multi-night candidates (no known match): {len(unknown_chains)}")

    # vet unknown candidates: rate consistency across nights + magnitude
    def _linear_residual_arcsec(c):
        ts = np.array([t["t"] / 86400.0 for t in c], dtype=float)
        ra = np.array([math.degrees(t["ra"]) for t in c], dtype=float)
        dec = np.array([math.degrees(t["dec"]) for t in c], dtype=float)
        dec0 = float(np.median(dec))
        x = (ra - float(np.median(ra))) * math.cos(math.radians(dec0)) * 3600.0
        y = (dec - float(np.median(dec))) * 3600.0
        tt = ts - float(np.mean(ts))
        if len(c) < 3 or float(np.var(tt)) == 0.0:
            return 0.0
        ax, bx = np.polyfit(tt, x, 1)
        ay, by = np.polyfit(tt, y, 1)
        resid = np.hypot(x - (ax * tt + bx), y - (ay * tt + by))
        return float(np.sqrt(np.mean(resid * resid)))

    def _heading_scatter_deg(c):
        heads = np.unwrap(np.array([math.atan2(t["ddec"], t["dra"]) for t in c], dtype=float))
        if len(heads) < 2:
            return 0.0
        return float(np.degrees(np.std(heads)))

    def vet_rules(c):
        if len({t["night"] for t in c}) < args.vet_min_nights:
            return False
        rates = np.array([t["rate_arcsec_hr"] for t in c], dtype=float)
        mean_rate = float(np.mean(rates))
        if mean_rate <= 0:
            return False
        if float(np.std(rates) / mean_rate) > args.vet_max_rate_cv:
            return False
        if _heading_scatter_deg(c) > args.vet_max_heading_scatter_deg:
            return False
        if _linear_residual_arcsec(c) > args.vet_max_linear_resid_arcsec:
            return False
        return True

    def vet_coherence(c):
        # Equation-of-ONE energy selector over the chain's per-night centroids
        # (gap22 pattern). One unified energy replaces the AND-thresholds.
        if len({t["night"] for t in c}) < args.vet_min_nights:
            return False
        from ariadne.discovery.imaging.coherence_vet import track_energy
        ra = np.array([math.degrees(t["ra"]) for t in c], dtype=float)
        dec = np.array([math.degrees(t["dec"]) for t in c], dtype=float)
        mjd = np.array([t["t"] / 86400.0 for t in c], dtype=float)
        return track_energy(ra, dec, mjd) <= args.vet_eoo_tau

    vet = vet_coherence if args.vet_mode == "coherence" else vet_rules
    vetted = [c for c in unknown_chains if vet(c)]
    print(f"  vetted unknown candidates ({args.vet_mode}): {len(vetted)}")

    # FALSE FLOOR: scramble one night's positions, re-link. A chance excess
    # of real candidates over this floor is the discovery signal.
    import copy
    off_night = nights[len(nights) // 2]
    stage_t = time.time()
    if args.linker_mode in ("rate", "rate-coherence"):
        scr_objs = []
        for tr in rate_track_objs:
            tr2 = copy.copy(tr)            # shallow copy carries _known_id
            if tr2.night == off_night:
                tr2.ra_mid = tr2.ra_mid + 0.08
                tr2.dec_mid = tr2.dec_mid + 0.05
            scr_objs.append(tr2)
        _link = link_coherence if args.linker_mode == "rate-coherence" else link_rate_constrained
        scr_track_chains = _link(scr_objs, min_nights=2)
        scr_chains = [[{"night": tr.night,
                        "known_id": getattr(tr, "_known_id", None)} for tr in ch]
                      for ch in scr_track_chains]
    else:
        scr_tracks = []
        for t in tracks:
            t2 = dict(t)
            if t2["night"] == off_night:
                t2["ra"] = t2["ra"] + math.radians(0.08)
                t2["dec"] = t2["dec"] + math.radians(0.05)
            scr_tracks.append(t2)
        scr_chains = discover_in_images_chains(
            scr_tracks,
            use_helio_linc=args.linker_mode == "full",
            use_nbody_grow=args.linker_mode == "full",
            use_orbit_grow=args.linker_mode == "full",
            use_multi_hypothesis=args.linker_mode == "full",
        )
    scrambled_linking_s = time.time() - stage_t
    scr_mn = [c for c in scr_chains if len({t["night"] for t in c}) >= 2
              and all(t.get("known_id") is None for t in c)]
    print(f"  scrambled-control false floor (unknown multi-night chains): {len(scr_mn)}")
    print(f"  scrambled-control linking: {scrambled_linking_s:.1f}s", flush=True)
    print(f"  => {len(vetted)} candidates vs {len(scr_mn)} chance floor")

    for c in sorted(vetted, key=lambda c: -len(c))[:8]:
        ns = sorted({t['night'] for t in c})
        ra0 = math.degrees(c[0]['ra']); dec0 = math.degrees(c[0]['dec'])
        rate = np.mean([t['rate_arcsec_hr'] for t in c])
        print(f"    cand: ({ra0:.4f},{dec0:.4f}) nights={ns} rate={rate:.0f}\"/hr "
              f"tracklets={len(c)}")

    if args.iod and vetted:
        print(f"\n  IOD on top {min(5,len(vetted))} candidates:")
        from ariadne.discovery.iod_robust import robust_iod
        for c in sorted(vetted, key=lambda c: -len(c))[:5]:
            try:
                ens = robust_iod(c, n_draws=2, rms_acceptance_arcsec=30.0,
                                  use_monte_carlo=True, use_rate_class=True)
                print(f"    chain({len(c)}): success={ens.success} "
                      f"RMS={getattr(ens,'rms_arcsec',-1):.2f}\"")
            except Exception as e:
                print(f"    chain IOD error: {str(e)[:70]}")
    def coherence_vet_energy(c):
        from ariadne.discovery.imaging.coherence_vet import track_energy
        ra = [math.degrees(t["ra"]) for t in c]; dec = [math.degrees(t["dec"]) for t in c]
        mjd = [t["t"] / 86400.0 for t in c]
        return track_energy(ra, dec, mjd)

    summary = {
        "schema": "ariadne.real_decam_discovery_benchmark.v1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "n_files_seen": len(files),
        "n_usable_exposures": len(files) - len(skipped_exposures),
        "skipped_exposures": skipped_exposures,
        "n_ccd": args.n_ccd,
        "sigma": args.sigma,
        "min_rate_arcsec_hr": args.min_rate,
        "max_rate_arcsec_hr": args.max_rate,
        "match_arcsec": args.match_arcsec,
        "max_sources_per_exposure": args.max_sources_per_exposure,
        "stationary_veto_arcsec": args.stationary_veto_arcsec,
        "max_per_night": args.max_per_night,
        "truth_limit": args.truth_limit,
        "truth_mode": args.truth_mode,
        "linker_mode": args.linker_mode,
        "vet_min_nights": args.vet_min_nights,
        "vet_max_rate_cv": args.vet_max_rate_cv,
        "vet_max_heading_scatter_deg": args.vet_max_heading_scatter_deg,
        "vet_max_linear_resid_arcsec": args.vet_max_linear_resid_arcsec,
        "n_nights": len(nights),
        "n_detections": len(all_sources),
        "n_raw_link_detections": len(raw_link_sources),
        "n_stationary_vetoed_detections": len(raw_link_sources) - len(all_sources),
        "n_uncapped_truth_detections": len(all_truth_sources),
        "n_tracklets": len(tracks),
        "n_chains": len(chains),
        "n_multinight_chains": len(multinight),
        "n_recoverable_knowns": len(recoverable),
        "n_recovered_knowns": rec_known,
        "known_recovery": rec_known / max(len(recoverable), 1),
        "n_unknown_multinight_candidates": len(unknown_chains),
        "n_vetted_unknown_candidates": len(vetted),
        "n_scrambled_unknown_multinight_chains": len(scr_mn),
        "candidate_to_false_floor_ratio": (
            len(vetted) / len(scr_mn) if len(scr_mn) else None
        ),
        "above_floor": len(vetted) > len(scr_mn),   # potential real-discovery signal
        "candidates": [
            {
                "ra_deg": float(math.degrees(c[0]["ra"])),
                "dec_deg": float(math.degrees(c[0]["dec"])),
                "rate_arcsec_hr": float(np.mean([t["rate_arcsec_hr"] for t in c])),
                "nights": sorted({int(t["night"]) for t in c}),
                "n_tracklets": len(c),
                "coherence": float(math.exp(-0.5 * coherence_vet_energy(c))),
            }
            for c in sorted(vetted, key=lambda c: -len(c))
        ],
        "runtime_s": time.time() - t0,
        "extraction_s": extraction_s,
        "tracklet_build_s": tracklet_build_s,
        "truth_match_s": truth_s,
        "linking_s": linking_s,
        "scrambled_linking_s": scrambled_linking_s,
    }
    (out_dir / "discovery_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
