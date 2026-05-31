"""Fit the inference engine's per-class temperature on the large synthetic corpus.

Generates 300+ LabelledCase rows from large_corpus, then fits both global
and per-class temperatures by minimum-NLL. Persists the resulting
CalibrationConfig so the live pipeline can load it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def main():
    from ariadne.discovery import inference
    from ariadne.discovery.large_corpus import make_large_corpus

    print("=" * 70)
    print("FIT per-class temperatures on the LARGE synthetic corpus")
    print("=" * 70)

    cases = make_large_corpus(n_per_class=18, n_artefacts_per_kind=8, seed=20260601)
    reliability_cases = [(c.evidence, c.truth_label) for c in cases]
    print(f"  corpus size: {len(cases)} labelled cases")

    # Baseline: T=1.0 (no calibration)
    baseline_rep = inference.reliability_report(reliability_cases)
    print(f"\n  baseline (T=1.0):")
    print(f"    accuracy:  {baseline_rep.accuracy*100:.1f}%")
    print(f"    NLL:       {baseline_rep.nll:.3f}")
    print(f"    Brier:     {baseline_rep.brier:.3f}")
    print(f"    ECE:       {baseline_rep.ece:.3f}")

    # Global T fit
    t0 = time.time()
    global_cfg, global_rep = inference.fit_temperature(reliability_cases)
    print(f"\n  global-T fit ({time.time() - t0:.1f}s):")
    print(f"    best global T:  {global_cfg.temperature}")
    print(f"    NLL:            {global_rep.nll:.3f}")
    print(f"    Brier:          {global_rep.brier:.3f}")
    print(f"    ECE:            {global_rep.ece:.3f}")

    # Per-class T fit
    t0 = time.time()
    per_class_cfg, per_class_rep = inference.fit_per_class_temperatures(
        reliability_cases)
    print(f"\n  per-class T fit ({time.time() - t0:.1f}s):")
    print(f"    global T:       {per_class_cfg.temperature}")
    print(f"    per-class:      {len(per_class_cfg.class_temperatures)} classes fitted")
    for label, t in sorted(per_class_cfg.class_temperatures.items()):
        print(f"      {label:<25s} T={t}")
    print(f"    NLL:            {per_class_rep.nll:.3f}")
    print(f"    Brier:          {per_class_rep.brier:.3f}")
    print(f"    ECE:            {per_class_rep.ece:.3f}")

    # Persist
    out_path = Path("data/calibration/inference_v2_large_corpus.json")
    inference.save_calibration(per_class_cfg, out_path)
    print(f"\n  calibration written -> {out_path}")

    # Summary table
    print(f"\n  ============= IMPROVEMENT TABLE =============")
    print(f"    Metric    | Baseline | Global-T | Per-class")
    print(f"    ----------+----------+----------+----------")
    print(f"    Accuracy  | {baseline_rep.accuracy*100:>6.1f}%  | "
          f"{global_rep.accuracy*100:>6.1f}%  | {per_class_rep.accuracy*100:>6.1f}%")
    print(f"    NLL       | {baseline_rep.nll:>7.3f}  | "
          f"{global_rep.nll:>7.3f}  | {per_class_rep.nll:>7.3f}")
    print(f"    Brier     | {baseline_rep.brier:>7.3f}  | "
          f"{global_rep.brier:>7.3f}  | {per_class_rep.brier:>7.3f}")
    print(f"    ECE       | {baseline_rep.ece:>7.3f}  | "
          f"{global_rep.ece:>7.3f}  | {per_class_rep.ece:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
