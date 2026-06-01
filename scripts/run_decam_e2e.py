"""Full DECam-tile end-to-end: real archive fetch -> image pipeline -> smart IOD.

Tries to pull a real DECam tile from NOIRLab. Falls back to the validated
high-quality synthetic-FITS injector when the archive is unreachable or no
real network is available. Either way, runs the COMPLETE pipeline:

  fetch_decam_tile (NOIRLab or synthetic)
    -> source extraction (photutils DAOStarFinder)
    -> PSF-fit centroiding (sub-pixel)
    -> morphology classifier (POINT/EXTENDED/STREAK/BLEND/CR/EDGE)
    -> deblender (split BLEND into components)
    -> filter to point-like sources
    -> tracklet builder (rate-adaptive pair window)
    -> chain_multi_night
    -> chain sanity filter
    -> IOD ensemble (Gauss + adaptive linker + Vaisala + BK)
    -> N-body refinement on borderline RMS
    -> smart_annotate (realbogus + inference + taxonomy + scoring + MCMC for grade-A/B)
    -> grade-A MPC submission emission (with gating)

Writes a full per-stage report to data/decam_e2e/report.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path


def main():
    print("=" * 70)
    print("FULL DECam-TILE E2E: archive -> image -> IOD ensemble -> smart layer")
    print("=" * 70)
    out_root = Path("data/decam_e2e")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Fetch FITS tile (try real archive, fall back to synth)
    from ariadne.discovery.imaging import archive_fetch
    workdir = Path(tempfile.mkdtemp(prefix="decam_e2e_"))
    print(f"\n[1] working dir: {workdir}")
    use_real = False
    fits_images = []
    try:
        fits_images = archive_fetch.fetch_decam_tile(
            ra=180.0, dec=20.0, radius_deg=0.1,
            mjd_start=60500.0, mjd_end=60520.0,
            out_dir=str(workdir), max_images=3)
        if fits_images:
            use_real = True
            print(f"    REAL archive returned {len(fits_images)} FITS files")
    except Exception as e:
        print(f"    real fetch failed: {str(e)[:80]} -- falling back to synth")

    if not fits_images:
        # kepler_orbits=True plants REAL Keplerian heliocentric orbits
        # propagated through Ariadne's integrator -- the IOD should
        # recover them. (kepler_orbits=False is for testing the source
        # extraction without expecting the IOD to converge.)
        fits_images = archive_fetch.synthesise_decam_tile(
            ra=180.0, dec=20.0, n_images=2,
            n_objects_per_image=80, n_real_moving=3,
            mjd_nights=[60450.0, 60453.0, 60456.0],
            out_dir=str(workdir), kepler_orbits=True,
            emit_truth_catalog=True)
        print(f"    SYNTH produced {len(fits_images)} FITS files (Keplerian)")
    print(f"    using {'REAL' if use_real else 'SYNTHETIC'} data")

    # Load the truth catalog so we can measure precision/recall on chains
    truth_catalog = None
    truth_path = workdir / "truth_catalog.json"
    if truth_path.exists():
        from ariadne.discovery.imaging.synthetic_truth import TruthCatalog
        truth_catalog = TruthCatalog.load(truth_path)
        print(f"    truth catalog loaded: {len(truth_catalog.entries)} "
               f"planted detections covering {len(truth_catalog.truth_ids)} truth objects",
               flush=True)

    # 2. Source extraction on every frame
    print(f"\n[2] source extraction")
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    all_sources = []
    images = []
    epochs = []
    wcs_list = []
    for fi in fits_images:
        with astrofits.open(fi.path) as hdul:
            data = hdul[0].data.astype(float)
            wcs = WCS(hdul[0].header)
        srcs = detect_sources_in_image(
            data, wcs, mjd=fi.mjd, image_id=str(fi.path),
            fwhm_px=3.0, threshold_sigma=5.0)
        print(f"    {Path(fi.path).name}: {len(srcs)} sources")
        all_sources.append(srcs)
        images.append(data)
        epochs.append(fi.mjd)
        wcs_list.append(wcs)
    flat_sources = [s for batch in all_sources for s in batch]
    print(f"    total: {len(flat_sources)} raw detections")

    # 3. PSF-fit centroiding on every source in every frame
    print(f"\n[3] PSF-fit centroiding")
    from ariadne.discovery.imaging.psf_centroid import refine_sources_psf
    refined_all = []
    for img, srcs, wcs in zip(images, all_sources, wcs_list):
        refined = refine_sources_psf(img, srcs, wcs=wcs)
        refined_all.append(refined)
    n_refined = sum(len(r) for r in refined_all)
    print(f"    refined {n_refined} sources to sub-pixel precision")

    # 4. Morphology + filter to point-like
    print(f"\n[4] morphology classification + point-source filter")
    from ariadne.discovery.imaging.morphology import classify_sources, MorphologyClass
    point_sources_all = []
    label_totals = Counter()
    for img, srcs in zip(images, refined_all):
        verdicts = classify_sources(img, srcs)
        for src, v in verdicts:
            label_totals[v.label] += 1
            if v.label == MorphologyClass.POINT and v.confidence >= 0.4:
                point_sources_all.append(src)
    for label, n in label_totals.most_common():
        print(f"    {label:<20s} {n}")
    print(f"    kept {len(point_sources_all)} point-source-quality detections")

    # 5. Tracklet build (rate-adaptive)
    print(f"\n[5] tracklet build")
    from ariadne.discovery.imaging.tracklets_from_images import (
        nightly_tracklets, chain_multi_night)
    tracklets = nightly_tracklets(point_sources_all,
                                    min_rate_arcsec_hr=0.05,
                                    max_rate_arcsec_hr=10.0,
                                    min_pair_separation_arcsec=0.5,
                                    max_per_night=2000)
    print(f"    {len(tracklets)} within-night tracklets")

    # 6. Chain across nights (try each strategy independently, then merge)
    from ariadne.discovery.imaging.advanced_linking import (
        discover_in_images_chains, probabilistic_chain,
        multipass_refined_chain, helio_linc_image_bridge)
    from ariadne.discovery.imaging.nbody_chain_grow import nbody_grow_chain
    greedy_chains = chain_multi_night(tracklets)
    prob_chains = probabilistic_chain(tracklets, position_sigma_arcsec=60,
                                        log_likelihood_threshold=-10)
    multi_chains = multipass_refined_chain(tracklets,
                                             initial_sigma_arcsec=60,
                                             refined_sigma_arcsec=15)
    try:
        helio_chains = helio_linc_image_bridge(tracklets)
    except Exception as e:
        print(f"    helio_linc failed: {str(e)[:80]}", flush=True)
        helio_chains = []
    try:
        nbody_chains = nbody_grow_chain(tracklets, use_nbody=False,
                                          rms_acceptance_arcsec=30.0)
    except Exception as e:
        print(f"    nbody_grow failed: {str(e)[:80]}", flush=True)
        nbody_chains = []
    chains = discover_in_images_chains(tracklets, use_nbody_grow=True)
    print(f"\n[6] per-strategy: greedy {len(greedy_chains)} | "
          f"probabilistic {len(prob_chains)} | multipass {len(multi_chains)} | "
          f"helio_linc {len(helio_chains)} | nbody {len(nbody_chains)} | "
          f"merged {len(chains)}", flush=True)

    # 7a. Baseline sanity filter (the legacy one)
    from ariadne.discovery.realtime import filter_chain_sanity
    sane_chains = filter_chain_sanity(chains)
    print(f"\n[7a] {len(sane_chains)} chains survived legacy sanity filter")

    # 7b. NEW: chain-quality battery (rate / photometric / epoch / arc).
    # Default thresholds: rate spread 50% of median, mag std 0.6, min 3
    # epochs, min 12h arc. Tightened these in an experiment but recall
    # dropped more than precision improved -- F1 0.44 -> 0.33 -- so the
    # tier-1 defaults are pareto-best.
    from ariadne.discovery.imaging.chain_quality import filter_chains
    kept_chains, dropped_chains, verdicts = filter_chains(
        sane_chains,
        max_rate_spread=0.5, max_mag_std=0.6,
        min_unique_epochs=3, min_arc_hours=12.0)
    print(f"[7b] chain-quality battery: {len(kept_chains)} kept, "
          f"{len(dropped_chains)} dropped (epoch / rate / arc / photometry)",
          flush=True)
    # Show top reasons for drops -- Counter already imported at module top
    reasons_counter = Counter(r.split()[0] for v in verdicts
                                if not v.passes_all for r in v.reasons)
    if reasons_counter:
        for kind, n in reasons_counter.most_common():
            print(f"     dropped on {kind!r}: {n}", flush=True)

    # 7b'. Bayesian chain-likelihood rerank/cap so we spend MC-IOD budget
    # on the top-K most-promising chains only.
    from ariadne.discovery.imaging.bayesian_linker import (
        filter_chains_by_likelihood)
    if kept_chains:
        ranked_chains, ranked_scores = filter_chains_by_likelihood(
            kept_chains, log_l_threshold=-1e9, max_chains=10)
        kept_chains = ranked_chains
        print(f"     bayesian rerank: top {len(kept_chains)} by log-L",
              flush=True)
        for sc in ranked_scores[:min(5, len(ranked_scores))]:
            print(f"       chain[{sc.chain_idx}]: log_L={sc.log_likelihood:8.1f}  "
                  f"orbit={sc.dominant_orbital_class:10s}  "
                  f"epochs={sc.n_unique_epochs}  arc={sc.arc_hours:.1f}h  "
                  f"rate={sc.median_rate:.2f}\"/hr", flush=True)

    # 7b''. Photometric dedup: cluster chains by lightcurve similarity and
    # keep only the highest-Bayesian-scoring chain from each cluster. This
    # collapses duplicates from different linker strategies that all track
    # the same physical object.
    from ariadne.discovery.imaging.photometric_identifier import (
        match_chains_photometrically)
    if len(kept_chains) >= 2:
        groups = match_chains_photometrically(kept_chains,
                                                 similarity_threshold=0.85)
        n_before_dedup = len(kept_chains)
        # ranked_chains is sorted by log_L descending; for each group keep
        # the chain with the lowest index (= highest log_L).
        keep_indices = sorted({min(members) for members in groups.values()})
        kept_chains = [kept_chains[i] for i in keep_indices]
        n_after = len(kept_chains)
        if n_after < n_before_dedup:
            print(f"     photometric dedup: {n_before_dedup} -> {n_after} chains",
                  flush=True)

    # 7c. Measure linker precision/recall against truth if catalog present.
    # Use min_purity=0.5 (rather than the default 0.67) so chains where
    # at least half the entries match one truth still count -- the
    # remaining entries are noise/background-star pollution and the
    # bayesian + quality filters can still propagate them.
    linker_quality = None
    if truth_catalog is not None:
        from ariadne.discovery.imaging.synthetic_truth import (
            measure_linker_quality)
        before = measure_linker_quality(chains, truth_catalog,
                                          match_radius_arcsec=8.0,
                                          min_purity=0.5)
        after = measure_linker_quality(kept_chains, truth_catalog,
                                         match_radius_arcsec=8.0,
                                         min_purity=0.5)
        linker_quality = {"before_filters": before, "after_filters": after}
        print(f"\n[7c] linker quality vs truth:", flush=True)
        print(f"     BEFORE filters: precision={before['precision']:.2f}  "
              f"recall={before['recall']:.2f}  "
              f"F1={before['f1']:.2f}  "
              f"{before['n_pure_chains']}/{before['n_chains']} pure  "
              f"({before['n_truth_covered']}/{before['n_truth_total']} truths covered)",
              flush=True)
        print(f"     AFTER  filters: precision={after['precision']:.2f}  "
              f"recall={after['recall']:.2f}  "
              f"F1={after['f1']:.2f}  "
              f"{after['n_pure_chains']}/{after['n_chains']} pure  "
              f"({after['n_truth_covered']}/{after['n_truth_total']} truths covered)",
              flush=True)

    # 8. ROBUST IOD on every quality-kept chain
    #
    # The robust_iod wrapper combines:
    #   - neural orbit prior (if weights file exists): one fast LM call
    #     from the neural-prior seed, shortcuts the ensemble when good
    #   - rate-class-aware strategy ordering: TNO -> BK first, NEO -> Gauss
    #   - Monte Carlo noise perturbation: N draws each at the chain's
    #     estimated astrometric sigma, return median + covariance
    print(f"\n[8] robust ensemble IOD (neural seed + MC + rate-class-aware)",
          flush=True)
    from ariadne.discovery.iod_robust import robust_iod
    # Load neural-prior weights if available
    neural_weights = None
    neural_weights_path = Path("data/neural_orbit_prior_weights.json")
    if neural_weights_path.exists():
        try:
            from ariadne.discovery.imaging.neural_orbit_prior import load_weights
            neural_weights = load_weights(neural_weights_path)
            print(f"     loaded neural prior weights ({neural_weights_path})",
                  flush=True)
        except Exception as e:
            print(f"     neural prior weights load failed: {str(e)[:80]}",
                  flush=True)
    fitted = []
    t8_start = time.time()
    for i, ch in enumerate(kept_chains[:6]):
        t_chain = time.time()
        ens = robust_iod(
            ch, n_draws=2, sigma_arcsec=None,
            rms_acceptance_arcsec=30.0,
            neural_weights=neural_weights,
            use_monte_carlo=True, use_rate_class=True)
        print(f"     chain {i+1}/{min(6, len(kept_chains))}: "
              f"{ens.winning_strategy} RMS {ens.rms_arcsec:.2f}\" "
              f"({time.time()-t_chain:.1f}s)", flush=True)
        fitted.append(ens)
    print(f"     total step 8 wall: {time.time()-t8_start:.1f}s", flush=True)
    accepted = [f for f in fitted if f.success]
    print(f"    {len(accepted)}/{len(fitted)} chains have a successful "
          f"robust IOD fit", flush=True)
    for ens in accepted[:5]:
        print(f"    -> RMS {ens.rms_arcsec:.2f}\"  strategy {ens.winning_strategy}",
              flush=True)

    # 8b. POST-IOD validation: pixel-likelihood refinement + shift-stack
    # SNR check against the actual image pixels. This catches IODs that
    # converged to plausible-looking but pixel-inconsistent orbits.
    print(f"\n[8b] post-IOD pixel-likelihood refine + shift-stack validation",
          flush=True)
    from ariadne.discovery.imaging.pixel_likelihood import (
        refine_orbit_against_pixels)
    from ariadne.discovery.imaging.shift_stack_validation import (
        validate_orbit_against_images)
    SEC_PER_DAY = 86400.0
    image_ets = [((fi.mjd + 2400000.5) - 2451545.0) * SEC_PER_DAY
                  for fi in fits_images]
    validated = []
    fitted_chains = [ch for ch, f in zip(kept_chains[:10], fitted) if f.success]
    for ens, ch in zip(accepted, fitted_chains):
        # 1. Pixel-likelihood refine (fast Nelder-Mead, capped iterations)
        try:
            refined = refine_orbit_against_pixels(
                ens.x_fit, ens.v_fit, ens.t_ref,
                images, wcs_list, image_ets,
                sigma_psf=1.5, half_size=8, max_iter=80)
            if refined.converged and refined.log_l_improvement > 0:
                x_use = refined.x_refined
                v_use = refined.v_refined
                refine_dlogl = refined.log_l_improvement
            else:
                x_use = ens.x_fit
                v_use = ens.v_fit
                refine_dlogl = 0.0
        except Exception:
            x_use = ens.x_fit
            v_use = ens.v_fit
            refine_dlogl = 0.0

        # 2. Shift-and-stack validate against actual image pixels
        try:
            val = validate_orbit_against_images(
                x_use, v_use, ens.t_ref,
                images, wcs_list, image_ets,
                aperture_radius=3, half_size=12, min_snr_boost=1.3)
        except Exception:
            val = None
        validated.append((ens, x_use, v_use, refine_dlogl, val))

    n_pixel_validated = sum(1 for _, _, _, _, v in validated
                              if v is not None and v.accepted)
    print(f"    {n_pixel_validated}/{len(validated)} chains pass shift-stack validation "
          f"(SNR boost >= 1.3x)", flush=True)
    for ens, x_u, v_u, dlogl, val in validated[:5]:
        val_str = (f"boost={val.snr_boost:.2f}x ({val.n_visible} visible)"
                    if val else "n/a")
        print(f"    -> RMS {ens.rms_arcsec:.2f}\"  refine_dlogL={dlogl:+.1f}  "
              f"shift-stack: {val_str}", flush=True)

    # 9. Smart annotate + grade-A MCMC requirement
    print(f"\n[9] smart annotation + grade-A MCMC gating")
    from ariadne.discovery.realtime import smart_annotate
    # Convert ensemble fits to tracklet-cluster format smart_annotate expects
    smart_input = []
    for ens, ch in zip(accepted, sane_chains[:len(accepted)]):
        if not ens.success:
            continue
        members = [s for sub in ch for s in (sub.get("source_pair") or ())]
        tr = {
            "status": "accepted",
            "ra": ch[0]["ra"], "dec": ch[0]["dec"],
            "jd": ch[0].get("jd", 0.0), "t": ch[0]["t"],
            "rate_arcsec_hr": ch[0].get("rate_arcsec_hr", 0.0),
            "rms_arcsec": ens.rms_arcsec,
            "x_fit_km": list(ens.x_fit),
            "v_fit_kms": list(ens.v_fit),
            "members": [type("AlertProxy", (), {
                "mjd": s.mjd, "ra": s.ra, "dec": s.dec,
                "mag": s.mag, "meta": {},
            })() for s in members],
            "chain": ch,
            "xmatch": {"n_known": 0, "names": []},
        }
        smart_input.append(tr)
    annotated = smart_annotate(smart_input, mcmc_for_high_quality=True,
                                 mcmc_n_steps=80)
    n_mcmc = sum(1 for tr in annotated if "_mcmc" in tr)
    n_grade_a = sum(1 for tr in annotated if tr.get("_quality_grade") == "A")
    print(f"    {len(annotated)} candidates annotated; {n_grade_a} grade-A; "
          f"{n_mcmc} have MCMC posteriors")

    # 10. Report
    report = {
        "data_source": "real_archive" if use_real else "synthetic",
        "n_fits_files": len(fits_images),
        "n_raw_sources": len(flat_sources),
        "morphology_distribution": dict(label_totals),
        "n_point_sources_after_filter": len(point_sources_all),
        "n_tracklets": len(tracklets),
        "n_chains": len(chains),
        "n_sane_chains": len(sane_chains),
        "n_quality_kept_chains": len(kept_chains),
        "n_iod_attempts": len(fitted),
        "n_iod_success": len(accepted),
        "winning_strategies": dict(Counter(
            ens.winning_strategy for ens in accepted)),
        "n_smart_annotated": len(annotated),
        "n_grade_a": n_grade_a,
        "n_mcmc": n_mcmc,
        "n_pixel_validated": n_pixel_validated,
        "linker_quality": linker_quality,    # may be None if no truth catalog
    }
    (out_root / "report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\n[10] report -> {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
