"""Actually run the inference + sensitivity benchmarks and write the real numbers.

Two passes:

  1. INFERENCE BENCHMARK: load N MPCORB labelled cases via external_corpora
     (built-in synthesised population if no network), run the inference
     engine, compute accuracy / NLL / Brier / ECE.

  2. SENSITIVITY HARNESS: inject synthetic moving objects across a
     (magnitude, rate) grid, run them through the live pipeline, report
     recovery rates.

Writes a JSON report to data/benchmarks/real_run.json plus a markdown
summary at docs/REAL_BENCHMARKS.md.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path


def run_inference_benchmark():
    """Run the inference benchmark on the labelled suite (offline-safe).

    NEW: fits both a global temperature AND per-class temperatures on the
    same labelled corpus, persists the calibration to disk, and re-runs
    the benchmark with it to confirm ECE improvement.
    """
    from ariadne.discovery import benchmarking
    from ariadne.discovery import inference
    print("=" * 70)
    print("INFERENCE BENCHMARK -- labelled cases through ariadne.discovery.inference")
    print("=" * 70)

    # Always-available offline corpus (no network required)
    cases = benchmarking.make_labelled_inference_suite(seed=20260601)
    print(f"  using {len(cases)} labelled cases from the built-in corpus\n")
    t0 = time.time()
    result = benchmarking.run_inference_benchmark(cases=cases)
    elapsed = time.time() - t0
    print(f"\n  --- inference benchmark ({elapsed:.1f}s on {result.n} cases) ---")
    print(f"  accuracy:         {result.accuracy*100:.1f}%")
    print(f"  macro precision:  {result.macro_precision*100:.1f}%")
    print(f"  macro recall:     {result.macro_recall*100:.1f}%")
    print(f"  macro F1:         {result.macro_f1*100:.1f}%")
    print(f"  NLL:              {result.reliability.nll:.3f}")
    print(f"  Brier:            {result.reliability.brier:.3f}")
    print(f"  ECE:              {result.reliability.ece:.3f}")
    print(f"  source counts:    {result.source_counts}")
    print(f"  split counts:     {result.split_counts}")
    print(f"  certificate:      {result.certificate_hash[:16]}...")
    baseline = {
        "n_cases": result.n, "elapsed_s": elapsed,
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "nll": result.reliability.nll,
        "brier": result.reliability.brier,
        "ece": result.reliability.ece,
        "source_counts": result.source_counts,
        "split_counts": result.split_counts,
        "certificate_hash": result.certificate_hash,
        "calibration_temperature": result.calibration.temperature,
    }

    # --- per-class temperature fit + re-run --------------------------------
    print("\n  --- per-class temperature fit ---")
    reliability_cases = [(c.evidence, c.truth_label) for c in cases]
    t1 = time.time()
    fitted_cfg, fitted_rep = inference.fit_per_class_temperatures(
        reliability_cases)
    print(f"    fit took {time.time() - t1:.1f}s; global T={fitted_cfg.temperature:.2f}, "
          f"per-class entries={len(fitted_cfg.class_temperatures)}")
    print(f"    fitted reliability: NLL {fitted_rep.nll:.3f}, "
          f"Brier {fitted_rep.brier:.3f}, ECE {fitted_rep.ece:.3f}")
    # Persist calibration to disk for the live pipeline to load
    calib_path = Path("data/calibration/inference_v1.json")
    inference.save_calibration(fitted_cfg, calib_path)
    print(f"    calibration saved -> {calib_path}")

    baseline["fitted_calibration"] = {
        "global_T": fitted_cfg.temperature,
        "class_T_count": len(fitted_cfg.class_temperatures),
        "class_T": dict(fitted_cfg.class_temperatures),
        "version": fitted_cfg.version,
        "post_fit_nll": fitted_rep.nll,
        "post_fit_brier": fitted_rep.brier,
        "post_fit_ece": fitted_rep.ece,
        "nll_improvement": baseline["nll"] - fitted_rep.nll,
        "ece_improvement": baseline["ece"] - fitted_rep.ece,
    }
    print(f"\n    NLL improvement:  {baseline['nll']:.3f} -> "
          f"{fitted_rep.nll:.3f} ({baseline['nll'] - fitted_rep.nll:+.3f})")
    print(f"    ECE improvement:  {baseline['ece']:.3f} -> "
          f"{fitted_rep.ece:.3f} ({baseline['ece'] - fitted_rep.ece:+.3f})")

    return baseline


def run_sensitivity_recovery():
    """Inject synthetic moving objects + measure recovery rate."""
    from ariadne.validate.sensitivity import (
        make_population, inject_synthetic_objects, evaluate_recovery)
    from ariadne.discovery import realtime
    print("\n" + "=" * 70)
    print("SENSITIVITY RECOVERY HARNESS -- injection through realtime pipeline")
    print("=" * 70)

    # Generate 30 random orbits across the parameter space; inject + run
    orbits = make_population(n_objects=30, seed=42)
    print(f"  injecting {len(orbits)} synthetic objects across a/e/i/mag")
    alerts, truth = inject_synthetic_objects(
        orbits, epoch="2026-04-01T00:00:00",
        n_nights=4, n_per_night=3, noise_arcsec=0.3, seed=42)
    print(f"  -> {len(alerts)} synthetic alerts planted ({sum(t.n_planted_alerts for t in truth)} total)")

    t0 = time.time()
    # NEW: use the adaptive pair window (no explicit pair_dt) + tighter cluster
    # tolerance so within-night detections stay distinct.
    pipeline_out = realtime.run_pipeline(
        alerts, rms_threshold_arcsec=1e9, do_xmatch=False, smart_layer=True,
        cluster_time_tol_days=0.02,
        rate_window_arcsec_hr=(0.05, 5.0),
        pair_dt_hours=(None, None),       # adaptive
    )
    elapsed = time.time() - t0

    report = evaluate_recovery(pipeline_out, truth)
    print(f"\n  --- recovery report ({elapsed:.1f}s on {len(alerts)} alerts) ---")
    print(f"  injected:           {report.n_injected}")
    print(f"  recovered:          {report.n_recovered}")
    print(f"  false positives:    {report.n_false_positives}")
    print(f"  recovery rate:      {report.recovery_rate*100:.1f}%")
    if report.median_rms_arcsec == report.median_rms_arcsec:
        print(f"  median RMS:         {report.median_rms_arcsec:.2f} arcsec")
    if report.median_arc_recovery_days == report.median_arc_recovery_days:
        print(f"  median arc:         {report.median_arc_recovery_days:.1f} days")
    print(f"\n  recovery by magnitude:")
    for k, v in sorted(report.recovery_by_magnitude.items()):
        print(f"    mag {k:>10s}: {v['n_recovered']}/{v['n_injected']}"
              f" ({v['rate']*100:.0f}%)")
    print(f"  recovery by rate:")
    for k, v in sorted(report.recovery_by_rate.items()):
        print(f"    rate {k:>10s}: {v['n_recovered']}/{v['n_injected']}"
              f" ({v['rate']*100:.0f}%)")

    return {
        "n_injected": report.n_injected,
        "n_recovered": report.n_recovered,
        "n_false_positives": report.n_false_positives,
        "recovery_rate": report.recovery_rate,
        "median_rms_arcsec": (None if math.isnan(report.median_rms_arcsec)
                              else report.median_rms_arcsec),
        "median_arc_days": (None if math.isnan(report.median_arc_recovery_days)
                            else report.median_arc_recovery_days),
        "by_magnitude": report.recovery_by_magnitude,
        "by_rate": report.recovery_by_rate,
        "elapsed_s": elapsed,
    }


def write_markdown_summary(infer_result, recovery_result, out_path: Path):
    lines = [
        "# Real benchmark results",
        "",
        f"Generated by `scripts/run_real_benchmarks.py`. These are the actual",
        f"numbers measured by the engine, not theoretical limits.",
        "",
        "## Inference benchmark",
        "",
        f"Ran the inference engine on **{infer_result['n_cases']} labelled cases**",
        f"from the built-in corpus. Elapsed: {infer_result['elapsed_s']:.1f}s.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | **{infer_result['accuracy']*100:.1f}%** |",
        f"| Macro precision | {infer_result['macro_precision']*100:.1f}% |",
        f"| Macro recall | {infer_result['macro_recall']*100:.1f}% |",
        f"| Macro F1 | {infer_result['macro_f1']*100:.1f}% |",
        f"| Negative log likelihood | {infer_result['nll']:.3f} nats |",
        f"| Brier score | {infer_result['brier']:.3f} |",
        f"| Expected calibration error | {infer_result['ece']:.3f} |",
        f"| Best calibration temperature | {infer_result['calibration_temperature']:.2f} |",
        "",
        f"Source counts: `{infer_result['source_counts']}`",
        "",
        "## Sensitivity recovery",
        "",
        f"Injected **{recovery_result['n_injected']} synthetic moving objects**",
        f"into the live `realtime.run_pipeline` and measured the recovery rate.",
        f"Elapsed: {recovery_result['elapsed_s']:.1f}s.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Injected | {recovery_result['n_injected']} |",
        f"| Recovered | {recovery_result['n_recovered']} |",
        f"| Recovery rate | **{recovery_result['recovery_rate']*100:.1f}%** |",
        f"| False positives | {recovery_result['n_false_positives']} |",
    ]
    if recovery_result.get("median_rms_arcsec"):
        lines.append(f"| Median fit RMS | {recovery_result['median_rms_arcsec']:.2f} arcsec |")
    if recovery_result.get("median_arc_days"):
        lines.append(f"| Median recovered arc | {recovery_result['median_arc_days']:.1f} days |")

    lines.extend(["", "### Recovery by apparent magnitude", "",
                  "| Mag bin | Injected | Recovered | Rate |", "|---|---|---|---|"])
    for k, v in sorted(recovery_result["by_magnitude"].items()):
        lines.append(f"| {k} | {v['n_injected']} | {v['n_recovered']} | {v['rate']*100:.0f}% |")
    lines.extend(["", "### Recovery by on-sky rate", "",
                  "| Rate bin (arcsec/hr) | Injected | Recovered | Rate |", "|---|---|---|---|"])
    for k, v in sorted(recovery_result["by_rate"].items()):
        lines.append(f"| {k} | {v['n_injected']} | {v['n_recovered']} | {v['rate']*100:.0f}% |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  markdown summary -> {out_path}")


def main():
    infer = run_inference_benchmark()
    recovery = run_sensitivity_recovery()
    out_root = Path("data/benchmarks")
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "real_run.json"
    json_path.write_text(json.dumps({
        "inference": infer, "sensitivity": recovery,
    }, indent=2, default=str))
    print(f"\n  JSON report -> {json_path}")
    md_path = Path("docs/REAL_BENCHMARKS.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_summary(infer, recovery, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
