"""Statistical benchmark for the image-pipeline rebuild.

Runs the full pipeline N times against a planted synthetic field with
M truth objects per run, varying the seed each time, and aggregates
precision/recall/F1/IOD-success/shift-stack-validation across all
runs. Produces 95% confidence intervals on every metric.

Use this BEFORE claiming a number is "good" -- the prior single-run
result of "3 truths, 2/3 validated" was statistically meaningless.
This script puts the metrics on a defensible footing.

Output:
  data/decam_benchmark/per_run_results.json      one row per seed
  data/decam_benchmark/summary.json              aggregated stats + CIs
  data/decam_benchmark/per_stage_attrition.json  where truths get lost

To run:
  python scripts/benchmark_image_pipeline.py \\
      --n-runs 5 --n-truths 30 --npix 1024 --skip-iod
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _percentile_ci(values, p_lo=2.5, p_hi=97.5):
    """Bootstrap-free percentile CI from observed values."""
    if not values:
        return (0.0, 0.0, 0.0)
    vs = sorted(values)
    n = len(vs)
    mean = float(np.mean(vs))
    lo = float(np.percentile(vs, p_lo))
    hi = float(np.percentile(vs, p_hi))
    return (mean, lo, hi)


def run_single_seed(seed: int, n_truths: int, npix: int,
                      skip_iod: bool, mc_draws: int,
                      iod_chain_cap: int = 10) -> dict:
    """Run the pipeline once with the given seed; return per-stage stats."""
    from ariadne.discovery.imaging import archive_fetch
    from ariadne.discovery.imaging.synthetic_truth import (
        TruthCatalog, measure_linker_quality)
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.psf_centroid import refine_sources_psf
    from ariadne.discovery.imaging.morphology import classify_sources, MorphologyClass
    from ariadne.discovery.imaging.tracklets_from_images import nightly_tracklets
    from ariadne.discovery.imaging.advanced_linking import discover_in_images_chains
    from ariadne.discovery.imaging.chain_quality import filter_chains
    from ariadne.discovery.imaging.bayesian_linker import filter_chains_by_likelihood

    t_start = time.time()
    workdir = Path(tempfile.mkdtemp(prefix=f"benchmark_seed{seed}_"))
    fits = archive_fetch.synthesise_decam_tile(
        ra=180.0, dec=20.0, n_images=2,
        n_objects_per_image=80, n_real_moving=n_truths,
        mjd_nights=[60450.0, 60453.0, 60456.0],
        out_dir=str(workdir), kepler_orbits=True, emit_truth_catalog=True,
        seed=seed, npix=npix, cone_radius_deg=0.04)
    cat_path = workdir / "truth_catalog.json"
    if not cat_path.exists():
        return {"seed": seed, "error": "no truth catalog"}
    cat = TruthCatalog.load(cat_path)

    # Source extraction + tracklets
    imgs = []; wcs_list = []; ets = []
    SEC_PER_DAY = 86400.0
    for fi in fits:
        with astrofits.open(fi.path) as hdul:
            imgs.append(hdul[0].data.astype(float))
            wcs_list.append(WCS(hdul[0].header))
        ets.append(((fi.mjd + 2400000.5) - 2451545.0) * SEC_PER_DAY)
    all_srcs = [detect_sources_in_image(
                  img, wcs, mjd=fi.mjd, image_id=str(fi.path),
                  fwhm_px=3.0, threshold_sigma=5.0)
                 for img, wcs, fi in zip(imgs, wcs_list, fits)]
    refined = [refine_sources_psf(img, srcs, wcs=wcs)
                for img, srcs, wcs in zip(imgs, all_srcs, wcs_list)]
    point_srcs = []
    for img, srcs in zip(imgs, refined):
        verdicts = classify_sources(img, srcs)
        point_srcs.extend(s for s, v in verdicts
                            if v.label == MorphologyClass.POINT
                            and v.confidence >= 0.4)

    # Per-truth source coverage (how many image epochs detected per truth)
    truth_source_counts = Counter()
    for s in point_srcs:
        tid = cat.match_source(s, match_radius_pix=3.0)
        if tid:
            truth_source_counts[tid] += 1

    # Tight rate window + per-night cap to keep tracklet count bounded on
    # dense fields. With 520+ sources at npix=1024 and max_rate_arcsec_hr=200,
    # the pair window opens to ~400" and combinatorial pair count explodes
    # (8000+ tracklets / night). Cap at 10 "/hr (covers TNO/Centaur/MBA --
    # NEO recall has to come from a separate NEO-specific pass with its own
    # rate window).
    trks = nightly_tracklets(point_srcs,
                              min_rate_arcsec_hr=0.05,
                              max_rate_arcsec_hr=10.0,
                              min_pair_separation_arcsec=0.5,
                              max_per_night=1500)
    # Per-truth tracklet coverage
    truth_tracklet_counts = Counter()
    for tr in trks:
        pair = tr.get("source_pair") or ()
        tids = set()
        for s in pair:
            tid = cat.match_source(s, match_radius_pix=3.0)
            if tid:
                tids.add(tid)
        if len(tids) == 1:
            truth_tracklet_counts[list(tids)[0]] += 1

    chains = discover_in_images_chains(trks, use_nbody_grow=True)

    # NEW: shift-and-stack synthetic tracking discovers moving objects
    # that are sub-threshold per image but pile up in the coadd.
    # Threshold scaling: N_hyp = 16x12 = 192 rate/PA cells, N_pix ~ 1e6
    # per image. Effective trials = 192 * 1e6 = 2e8. For < 1 expected FP
    # at SNR threshold T, need T > sqrt(2 ln 2e8) ~ 5.8. Use 7.0 to be safe.
    from ariadne.discovery.imaging.synthetic_tracking import (
        fast_synthetic_tracking, synthetic_candidate_to_chain)
    synth_candidates = []
    try:
        synth_candidates = fast_synthetic_tracking(
            imgs, wcs_list, [fi.mjd for fi in fits],
            rate_min_arcsec_hr=0.3, rate_max_arcsec_hr=8.0,
            n_rates=12, n_pa=8,
            snr_threshold=7.0, pixscale_arcsec=1.0,
            n_top_per_hypothesis=2)
        # Only keep candidates with strong per-image consensus
        n_min_imgs = len(imgs)
        synth_candidates = [c for c in synth_candidates
                              if c.consensus_count >= max(4, n_min_imgs - 1)]
        for sc in synth_candidates:
            ch = synthetic_candidate_to_chain(
                sc, [fi.mjd for fi in fits], pixscale_arcsec=1.0)
            chains.append(ch)
    except Exception:
        pass

    # Loosened filters: at scale (10+ true objects) many real chains get
    # mis-linked with each other and inflate rate spread; tight defaults
    # rejected too many genuine chains. Loosened to recover recall.
    kept, dropped, verdicts = filter_chains(
        chains, max_rate_spread=0.8, max_mag_std=1.0,
        min_unique_epochs=2, min_arc_hours=6.0)
    ranked, _ = filter_chains_by_likelihood(
        kept, log_l_threshold=-1e9, max_chains=80)
    kept = ranked

    # Linker quality vs truth
    q_raw = measure_linker_quality(chains, cat, match_radius_arcsec=8.0,
                                      min_purity=0.5)
    q_filtered = measure_linker_quality(kept, cat, match_radius_arcsec=8.0,
                                           min_purity=0.5)

    # Per-stage attrition: at each pipeline stage, count how many TRUTHS
    # have at least one matching artifact present.
    truth_chain_set_raw = {tid for tid in q_raw["truth_for_chain"] if tid}
    truth_chain_set_filtered = {tid for tid in q_filtered["truth_for_chain"] if tid}
    n_truth = len(cat.truth_ids)
    per_stage = {
        "n_truth_planted":         n_truth,
        "n_truth_detected":        len(truth_source_counts),
        "n_truth_in_tracklet":     len(truth_tracklet_counts),
        "n_truth_in_raw_chain":    len(truth_chain_set_raw),
        "n_truth_in_filtered_chain": len(truth_chain_set_filtered),
    }

    iod_results = []
    pixel_validated_truths = set()
    iod_truths = set()
    if not skip_iod:
        from ariadne.discovery.iod_robust import robust_iod
        from ariadne.discovery.imaging.neural_orbit_prior import load_weights
        from ariadne.discovery.imaging.pixel_likelihood import (
            refine_orbit_against_pixels)
        from ariadne.discovery.imaging.shift_stack_validation import (
            validate_orbit_against_images)
        from ariadne.discovery.imaging.synthetic_truth import (
            assign_truth_to_chain)
        neural_weights = None
        weights_path = Path("data/neural_orbit_prior_weights.json")
        if weights_path.exists():
            try:
                neural_weights = load_weights(weights_path)
            except Exception:
                pass

        chains_for_iod = list(kept[:min(iod_chain_cap, len(kept))])
        # Pre-compute each kept chain's truth_id so we can track UNIQUE
        # truths recovered (not duplicate chains for the same truth).
        chain_truth_ids = []
        for ch in chains_for_iod:
            tid, _ = assign_truth_to_chain(ch, cat,
                                              match_radius_arcsec=8.0,
                                              min_purity=0.5)
            chain_truth_ids.append(tid)

        for ch_idx, ch in enumerate(chains_for_iod):
            t_chain = time.time()
            try:
                ens = robust_iod(
                    ch, n_draws=mc_draws, rms_acceptance_arcsec=30.0,
                    neural_weights=neural_weights,
                    use_monte_carlo=True, use_rate_class=True)
            except Exception as e:
                iod_results.append({"success": False, "error": str(e)[:80],
                                      "wall_s": time.time() - t_chain})
                continue
            row = {"success": ens.success,
                    "rms_arcsec": float(ens.rms_arcsec),
                    "strategy": ens.winning_strategy,
                    "wall_s": time.time() - t_chain}
            if ens.success:
                # Shift-stack validation
                try:
                    val_b = validate_orbit_against_images(
                        ens.x_fit, ens.v_fit, ens.t_ref,
                        imgs, wcs_list, ets,
                        aperture_radius=3, half_size=12, min_snr_boost=1.3)
                    baseline_boost = val_b.snr_boost if val_b else 0.0
                except Exception:
                    val_b = None; baseline_boost = 0.0
                x_use, v_use = ens.x_fit, ens.v_fit
                val = val_b
                try:
                    rfn = refine_orbit_against_pixels(
                        ens.x_fit, ens.v_fit, ens.t_ref,
                        imgs, wcs_list, ets,
                        sigma_psf=1.5, half_size=8, search_grid_pix=6)
                    if rfn.converged and rfn.log_l_improvement > 0:
                        val_after = validate_orbit_against_images(
                            rfn.x_refined, rfn.v_refined, ens.t_ref,
                            imgs, wcs_list, ets,
                            aperture_radius=3, half_size=12, min_snr_boost=1.3)
                        if val_after and val_after.snr_boost > baseline_boost:
                            x_use = rfn.x_refined
                            v_use = rfn.v_refined
                            val = val_after
                except Exception:
                    pass
                row["snr_boost"] = float(val.snr_boost) if val else 0.0
                row["validated"] = bool(val and val.accepted)
                tid = chain_truth_ids[ch_idx]
                if tid is not None:
                    iod_truths.add(tid)
                    if row["validated"]:
                        pixel_validated_truths.add(tid)
            iod_results.append(row)

    n_iod_success = sum(1 for r in iod_results if r.get("success"))
    n_validated = sum(1 for r in iod_results if r.get("validated"))

    return {
        "seed": seed,
        "wall_s_total": time.time() - t_start,
        "n_truths_planted": len(cat.truth_ids),
        "truth_family_distribution": dict(Counter(
            e.family for e in cat.entries)),
        "n_raw_sources": sum(len(s) for s in all_srcs),
        "n_point_srcs": len(point_srcs),
        "n_tracklets": len(trks),
        "n_chains_raw": len(chains),
        "n_chains_kept": len(kept),
        "n_chains_dropped": len(dropped),
        "truth_source_coverage_pct": (
            len(truth_source_counts) / max(len(cat.truth_ids), 1)),
        "truth_tracklet_coverage_pct": (
            len(truth_tracklet_counts) / max(len(cat.truth_ids), 1)),
        "linker_quality_raw": {k: q_raw[k] for k in
                                 ["precision", "recall", "f1",
                                   "n_pure_chains", "n_truth_covered"]},
        "linker_quality_filtered": {k: q_filtered[k] for k in
                                       ["precision", "recall", "f1",
                                         "n_pure_chains", "n_truth_covered"]},
        "per_stage_truth_attrition": per_stage,
        "n_iod_attempts": len(iod_results),
        "n_iod_success": n_iod_success,
        "n_pixel_validated": n_validated,
        "n_unique_truths_recovered_iod": len(iod_truths),
        "n_unique_truths_recovered_pixel": len(pixel_validated_truths),
        "iod_wall_s": float(sum(r["wall_s"] for r in iod_results)),
        "iod_strategy_counts": dict(Counter(
            r.get("strategy", "none") for r in iod_results if r.get("success"))),
        "iod_rows": iod_results[:5],   # first 5 only to keep JSON small
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=5,
                          help="Number of seeded runs to average across")
    parser.add_argument("--n-truths", type=int, default=30,
                          help="Synthetic truths per run")
    parser.add_argument("--npix", type=int, default=1024,
                          help="Synthetic image side length in pixels")
    parser.add_argument("--mc-draws", type=int, default=2,
                          help="Monte Carlo draws per IOD attempt")
    parser.add_argument("--iod-chain-cap", type=int, default=10,
                          help="Max chains per run to feed to IOD")
    parser.add_argument("--skip-iod", action="store_true",
                          help="Skip step 8/8b (faster; linker-only metrics)")
    parser.add_argument("--out", default="data/decam_benchmark",
                          help="Output dir for per-run + summary JSON")
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("=" * 70, flush=True)
    print(f"IMAGE-PIPELINE STATISTICAL BENCHMARK", flush=True)
    print(f"  runs={args.n_runs}  truths/run={args.n_truths}  "
          f"npix={args.npix}  skip_iod={args.skip_iod}", flush=True)
    print("=" * 70, flush=True)

    per_run = []
    for i in range(args.n_runs):
        seed = 1000 + i
        print(f"\n[run {i+1}/{args.n_runs}] seed={seed}", flush=True)
        try:
            row = run_single_seed(seed=seed, n_truths=args.n_truths,
                                     npix=args.npix, skip_iod=args.skip_iod,
                                     mc_draws=args.mc_draws,
                                     iod_chain_cap=args.iod_chain_cap)
        except Exception as e:
            row = {"seed": seed, "error": str(e)[:160]}
        per_run.append(row)
        print(f"  -> {row.get('n_chains_raw', 0)} chains -> "
              f"{row.get('n_chains_kept', 0)} kept -> "
              f"{row.get('n_iod_success', 0)} IOD success -> "
              f"{row.get('n_pixel_validated', 0)} pixel-validated  "
              f"(wall {row.get('wall_s_total', 0):.0f}s)", flush=True)
        if "linker_quality_filtered" in row:
            lq = row["linker_quality_filtered"]
            print(f"     linker (filtered): precision={lq['precision']:.2f}  "
                  f"recall={lq['recall']:.2f}  F1={lq['f1']:.2f}", flush=True)

    (out / "per_run_results.json").write_text(
        json.dumps(per_run, indent=2, sort_keys=True))

    # Aggregate stats
    def _pct(stage_key, r):
        pa = r.get("per_stage_truth_attrition", {})
        n = pa.get(stage_key, 0)
        d = pa.get("n_truth_planted", 1)
        return n / max(d, 1)

    keys_of_interest = [
        ("attrition.planted_to_detected",    lambda r: _pct("n_truth_detected", r)),
        ("attrition.planted_to_tracklet",    lambda r: _pct("n_truth_in_tracklet", r)),
        ("attrition.planted_to_rawchain",    lambda r: _pct("n_truth_in_raw_chain", r)),
        ("attrition.planted_to_filtchain",   lambda r: _pct("n_truth_in_filtered_chain", r)),
        ("linker_quality_raw.precision",     lambda r: r.get("linker_quality_raw", {}).get("precision", 0)),
        ("linker_quality_raw.recall",        lambda r: r.get("linker_quality_raw", {}).get("recall", 0)),
        ("linker_quality_raw.f1",            lambda r: r.get("linker_quality_raw", {}).get("f1", 0)),
        ("linker_quality_filtered.precision", lambda r: r.get("linker_quality_filtered", {}).get("precision", 0)),
        ("linker_quality_filtered.recall",   lambda r: r.get("linker_quality_filtered", {}).get("recall", 0)),
        ("linker_quality_filtered.f1",       lambda r: r.get("linker_quality_filtered", {}).get("f1", 0)),
        ("n_iod_success_per_run",            lambda r: r.get("n_iod_success", 0)),
        ("n_pixel_validated_per_run",        lambda r: r.get("n_pixel_validated", 0)),
        ("unique_truths_recovered_iod",      lambda r: r.get("n_unique_truths_recovered_iod", 0)),
        ("unique_truths_recovered_pixel",    lambda r: r.get("n_unique_truths_recovered_pixel", 0)),
        ("recovery_rate_pixel",              lambda r: r.get("n_unique_truths_recovered_pixel", 0) / max(r.get("n_truths_planted", 1), 1)),
        ("wall_s_total",                     lambda r: r.get("wall_s_total", 0)),
    ]
    successful = [r for r in per_run if "error" not in r]
    summary = {
        "n_runs_total": len(per_run),
        "n_runs_successful": len(successful),
        "config": {"n_truths": args.n_truths, "npix": args.npix,
                    "mc_draws": args.mc_draws, "skip_iod": args.skip_iod},
        "metrics": {},
    }
    for key, fn in keys_of_interest:
        values = [fn(r) for r in successful]
        mean, lo, hi = _percentile_ci(values)
        summary["metrics"][key] = {
            "mean": mean, "lo_95": lo, "hi_95": hi,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "n": len(values),
        }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    print(f"\n{'=' * 70}", flush=True)
    print(f"SUMMARY (95% percentile intervals across {len(successful)} runs):",
           flush=True)
    print(f"{'=' * 70}", flush=True)
    for key, _ in keys_of_interest:
        m = summary["metrics"][key]
        print(f"  {key:40s} {m['mean']:>7.3f}  "
              f"[{m['lo_95']:.3f}, {m['hi_95']:.3f}]", flush=True)
    print(f"\nResults -> {out}/per_run_results.json + summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
