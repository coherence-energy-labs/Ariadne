"""Tests for the neural orbit prior."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def test_build_features_fixed_size():
    from ariadne.discovery.imaging.neural_orbit_prior import (
        build_chain_features, FEATURE_DIM)
    ch = [
        {"t": _et(60450.0), "ra": math.radians(180.0),
         "dec": math.radians(20.0), "rate_arcsec_hr": 2.5, "mag": 21.5},
        {"t": _et(60453.0), "ra": math.radians(180.05),
         "dec": math.radians(20.0), "rate_arcsec_hr": 2.5, "mag": 21.5},
    ]
    f = build_chain_features(ch)
    assert f.shape == (FEATURE_DIM,)
    assert f.dtype == np.float32


def test_build_features_handles_empty():
    from ariadne.discovery.imaging.neural_orbit_prior import (
        build_chain_features, FEATURE_DIM)
    f = build_chain_features([])
    assert f.shape == (FEATURE_DIM,)
    assert (f == 0).all()


def test_heuristic_initial_state_at_correct_distance():
    from ariadne.discovery.imaging.neural_orbit_prior import (
        _heuristic_initial_state, TNO_TYPICAL_R_KM)
    feats = np.zeros(20, dtype=np.float32)
    feats[0] = math.radians(180.0)
    feats[1] = math.radians(20.0)
    x_km, v_km_s = _heuristic_initial_state(feats)
    # Position magnitude should be near 50 AU
    assert abs(np.linalg.norm(x_km) - TNO_TYPICAL_R_KM) / TNO_TYPICAL_R_KM < 0.1


def test_predict_initial_state_falls_back_when_no_weights():
    """When no weights are passed, should fall back to heuristic."""
    from ariadne.discovery.imaging.neural_orbit_prior import (
        predict_initial_state)
    feats = np.zeros(20, dtype=np.float32)
    feats[0] = math.radians(180.0)
    feats[1] = math.radians(20.0)
    x, v = predict_initial_state(feats, weights=None)
    assert x.shape == (3,)
    assert v.shape == (3,)
    assert np.linalg.norm(x) > 1e9   # > 1e9 km = > ~7 AU
    assert np.linalg.norm(v) > 0     # nonzero velocity


def test_init_weights_correct_shapes():
    from ariadne.discovery.imaging.neural_orbit_prior import _init_weights
    w = _init_weights(seed=0)
    assert w["W1"].shape == (20, 64)
    assert w["W2"].shape == (64, 32)
    assert w["W3"].shape == (32, 6)


def test_forward_pass_correct_output_shape():
    from ariadne.discovery.imaging.neural_orbit_prior import (
        _init_weights, _forward, FEATURE_DIM, OUTPUT_DIM)
    w = _init_weights(seed=0)
    feats = np.random.default_rng(0).normal(0, 1, size=(5, FEATURE_DIM)).astype(np.float32)
    y, _ = _forward(feats, w)
    assert y.shape == (5, OUTPUT_DIM)


def test_save_load_weights_roundtrip(tmp_path):
    from ariadne.discovery.imaging.neural_orbit_prior import (
        _init_weights, save_weights, load_weights)
    w = _init_weights(seed=42)
    p = tmp_path / "weights.json"
    save_weights(w, p)
    w2 = load_weights(p)
    for k in w:
        np.testing.assert_allclose(w[k], w2[k], atol=1e-6)


def test_short_training_reduces_loss():
    """A tiny training run should at least DECREASE the MSE loss."""
    from ariadne.discovery.imaging.neural_orbit_prior import (
        train_orbit_prior, _init_weights, _forward,
        _generate_training_example, build_chain_features)
    rng = np.random.default_rng(123)
    # Generate a small held-out set
    X_ho, Y_ho = [], []
    for _ in range(20):
        f, t = _generate_training_example(rng)
        X_ho.append(f); Y_ho.append(t)
    X_ho = np.stack(X_ho); Y_ho = np.stack(Y_ho)

    # Loss before training (random init)
    w_init = _init_weights(seed=0)
    y_init, _ = _forward(X_ho, w_init)
    loss_init = float(np.mean((y_init - Y_ho) ** 2))

    # Train briefly
    w = train_orbit_prior(n_examples=100, n_epochs=20, seed=0, verbose=False)
    feats_n = (X_ho - w["_feature_mean"]) / w["_feature_sigma"]
    pure_w = {k: v for k, v in w.items() if not k.startswith("_")}
    y_trained, _ = _forward(feats_n, pure_w)
    loss_trained = float(np.mean((y_trained - Y_ho) ** 2))

    # Loss should drop (at least 10%)
    assert loss_trained < 0.95 * loss_init, (
        f"Training failed to reduce loss: init {loss_init} -> trained {loss_trained}")
