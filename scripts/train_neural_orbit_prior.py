"""Train and persist the neural orbit prior.

Reads no input; runs a single training pass and writes weights to
data/neural_orbit_prior_weights.json. The orchestrator
(scripts/run_decam_e2e.py) loads them at runtime to seed robust_iod.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def main():
    from ariadne.discovery.imaging.neural_orbit_prior import (
        train_orbit_prior, save_weights, build_chain_features,
        _generate_training_example, _forward)
    out_path = Path("data/neural_orbit_prior_weights.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("TRAINING NEURAL ORBIT PRIOR (synthetic Keplerian inverse-mapping)",
          flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    weights = train_orbit_prior(n_examples=2000, n_epochs=200,
                                 lr=3e-3, seed=0, verbose=True)
    dt = time.time() - t0
    print(f"\nTraining wall time: {dt:.1f}s", flush=True)

    # Held-out evaluation
    rng = np.random.default_rng(99999)
    X_ho, Y_ho = [], []
    for _ in range(200):
        f, t = _generate_training_example(rng)
        X_ho.append(f); Y_ho.append(t)
    X_ho = np.stack(X_ho); Y_ho = np.stack(Y_ho)
    feats_n = (X_ho - weights["_feature_mean"]) / weights["_feature_sigma"]
    pure_w = {k: v for k, v in weights.items() if not k.startswith("_")}
    y_ho, _ = _forward(feats_n, pure_w)
    # Targets/predictions are in BALANCED units: x in AU/50, v in km/s/5.
    # Unscale before reporting RMS in physical units.
    rms_pos_au = float(np.sqrt(np.mean(((y_ho[:, :3] - Y_ho[:, :3]) * 50.0) ** 2)))
    rms_vel_kms = float(np.sqrt(np.mean(((y_ho[:, 3:] - Y_ho[:, 3:]) * 5.0) ** 2)))
    print(f"\nHELD-OUT EVAL (200 examples):", flush=True)
    print(f"  position RMS:  {rms_pos_au:.3f} AU", flush=True)
    print(f"  velocity RMS:  {rms_vel_kms:.3f} km/s", flush=True)

    save_weights(weights, out_path)
    print(f"\nWeights persisted -> {out_path}", flush=True)
    print(f"  size: {out_path.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
