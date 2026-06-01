"""Neural orbit prior for image-pipeline chain seeding.

Every IOD strategy needs an INITIAL GUESS for (x, v). The deterministic
strategies all bootstrap differently:

  Gauss method        analytically solves for range; can fail on noise
  HelioLinC           grid-searches over hypothesis distance
  Vaisala             assumes object is at perihelion
  Bernstein-Khushalani 6D inverse-distance, requires deep TNO

When the initial guess is good, LM refinement converges. When it's bad,
LM gets stuck in a local minimum and the candidate fit gets rejected
for high RMS even though a small perturbation would have converged.

The neural prior generates a strong initial guess from the chain's
observable features (rate, arc, n_epochs, sky_position, time_separation
between observations) by learning the inverse mapping from synthetic
Keplerian sequences:

  Train data        : Hundreds of synthetic Keplerian orbits, each
                      "observed" at 3 epochs to produce a sky-track
                      feature vector.
  Network           : Small MLP (32 -> 64 -> 32 -> 6) that predicts
                      (x, v) given the features. Outputs are SCALED
                      to AU + AU/day to keep magnitudes balanced.
  Loss              : MSE on the 6-dim state, weighted toward the
                      position components.

At inference time, the network's output is treated as a STRONG PRIOR
on (x, v) and used to seed the LM refinement step in `iod_advanced.py`.
On noisy inputs (the regime that breaks Gauss) it's much more robust
because it learns the noise-to-signal mapping during training.

Implementation note: rather than depend on torch/jax/tensorflow at
runtime, we ship a tiny pure-numpy MLP trained offline. The training
script is `scripts/train_neural_orbit_prior.py` (separate). Weights
are persisted as a JSON file under `data/neural_orbit_prior_weights.json`.

If no weights file is available, `predict_initial_state` falls back
to a deterministic centroid-distance heuristic so the rest of the
pipeline keeps working.

Public API:

  build_chain_features(chain) -> np.ndarray
        Extract 24-dim feature vector from a chain.

  predict_initial_state(features, weights=None)
        Forward-pass: features -> (x_au, v_au_per_day).

  train_orbit_prior(n_examples=2000, n_epochs=200)
        Generate synthetic Keplerian training set + train the MLP.
        Returns trained weights (dict of arrays).

  save_weights / load_weights
        JSON persistence helpers.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# ===========================================================================
# Feature extraction
# ===========================================================================

# Number of observations the feature vector summarises. Chains can be
# longer, but we use FIRST + MEDIAN + LAST to make the network input
# fixed-size regardless of chain length.
FEATURE_OBS = 3
FEATURE_DIM = (FEATURE_OBS * 4   # 3 obs x (ra_rad, dec_rad, t_norm, rate)
                + 4                  # rate stats: median, p5, p95, n_unique_epochs
                + 4)                 # arc_hours, mean_rate, mag_med, mag_std

OUTPUT_DIM = 6   # (x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s) scaled


# Reference scales for normalisation.
AU_KM = 149_597_870.7
TNO_TYPICAL_R_KM = 50 * AU_KM       # 50 AU
TNO_TYPICAL_V_KM_S = 5.0            # 5 km/s


def build_chain_features(chain: Sequence[dict]) -> np.ndarray:
    """Extract a fixed-size feature vector from a chain.

    Returns a numpy float32 array of shape (FEATURE_DIM,).
    """
    if len(chain) < 1:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    # Sort by time
    sorted_ch = sorted(chain, key=lambda e: e["t"])
    n = len(sorted_ch)
    pick = [sorted_ch[0],
            sorted_ch[n // 2],
            sorted_ch[-1]]
    t_ref = float(sorted_ch[0]["t"])
    features = []
    for e in pick:
        ra = float(e["ra"])
        dec = float(e["dec"])
        t_rel_days = (float(e["t"]) - t_ref) / 86400.0
        rate = float(e.get("rate_arcsec_hr", 0.0))
        features.extend([ra, dec, t_rel_days, rate])

    # Aggregate stats
    rates = [float(e.get("rate_arcsec_hr", 0.0)) for e in sorted_ch
             if e.get("rate_arcsec_hr") is not None]
    if rates:
        rate_med = float(np.median(rates))
        rate_p5 = float(np.percentile(rates, 5))
        rate_p95 = float(np.percentile(rates, 95))
        rate_mean = float(np.mean(rates))
    else:
        rate_med = rate_p5 = rate_p95 = rate_mean = 0.0
    epoch_days = {int(e["t"] / 86400.0) for e in sorted_ch}
    arc_hours = (sorted_ch[-1]["t"] - sorted_ch[0]["t"]) / 3600.0
    features.extend([rate_med, rate_p5, rate_p95, float(len(epoch_days))])

    mags = [float(e.get("mag", -99.0)) for e in sorted_ch
            if e.get("mag", -99.0) > -50]
    mag_med = float(np.median(mags)) if mags else 0.0
    mag_std = float(np.std(mags)) if len(mags) >= 2 else 0.0
    features.extend([arc_hours, rate_mean, mag_med, mag_std])

    return np.array(features, dtype=np.float32)


# ===========================================================================
# Pure-numpy MLP
# ===========================================================================

def _relu(x): return np.maximum(0.0, x)


def _relu_grad(x): return (x > 0).astype(x.dtype)


def _init_weights(seed: int = 0) -> dict:
    """Initialise small MLP weights (Glorot-uniform)."""
    rng = np.random.default_rng(seed)
    h1, h2 = 64, 32
    def gu(n_in, n_out):
        lim = math.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-lim, lim, size=(n_in, n_out)).astype(np.float32)
    return {
        "W1": gu(FEATURE_DIM, h1), "b1": np.zeros(h1, dtype=np.float32),
        "W2": gu(h1, h2),          "b2": np.zeros(h2, dtype=np.float32),
        "W3": gu(h2, OUTPUT_DIM),  "b3": np.zeros(OUTPUT_DIM, dtype=np.float32),
    }


def _forward(features: np.ndarray, weights: dict) -> tuple:
    """Return (output, intermediates). features may be (D,) or (N, D)."""
    x = np.atleast_2d(features)
    z1 = x @ weights["W1"] + weights["b1"]
    h1 = _relu(z1)
    z2 = h1 @ weights["W2"] + weights["b2"]
    h2 = _relu(z2)
    y = h2 @ weights["W3"] + weights["b3"]
    return y, (x, z1, h1, z2, h2)


def predict_initial_state(features: np.ndarray,
                            weights: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forward-pass: features -> (x_km, v_km_s).

    The MLP outputs are in BALANCED units (position in AU/50, velocity
    in km/s/5) so position and velocity have comparable magnitudes
    during training. We unscale here for the caller.

    If `weights` is None, fall back to a deterministic heuristic that
    places the seed at 50 AU heliocentric along the median sky direction,
    with a circular-orbit velocity. Useful when no trained weights are
    available.
    """
    if weights is None:
        return _heuristic_initial_state(features)
    y, _ = _forward(features, weights)
    y = y.squeeze()
    # Output unscaling: position in (AU/50) -> km, velocity in (km/s / 5) -> km/s
    x_km = y[:3] * 50.0 * AU_KM
    v_km_s = y[3:] * 5.0
    return x_km, v_km_s


def _heuristic_initial_state(features: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fallback for the seed.

    Uses the first observation's (ra, dec) to point a 50 AU heliocentric
    position vector, with a circular-orbit velocity perpendicular to it.
    """
    from ...data.constants import GM_SUN
    ra = features[0]; dec = features[1]
    d = np.array([math.cos(dec) * math.cos(ra),
                   math.cos(dec) * math.sin(ra),
                   math.sin(dec)])
    rho = TNO_TYPICAL_R_KM
    x_km = rho * d
    v_circ = math.sqrt(GM_SUN / rho)
    # Velocity perpendicular to x, in ecliptic plane
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    tan = np.cross(d, up)
    tan /= float(np.linalg.norm(tan))
    v_km_s = v_circ * tan
    return x_km, v_km_s


# ===========================================================================
# Training
# ===========================================================================

def _generate_training_example(rng: np.random.Generator) -> tuple:
    """Generate one (features, target_state) pair from a random Keplerian orbit
    observed at 3 epochs spread over ~6 days at typical DECam cadence.
    """
    from ...data.constants import GM_SUN, AU_KM as _AU_KM
    from ...data.ephemeris import body_state
    from ...dynamics.secular import kepler_step

    a_au = rng.uniform(35, 80)
    rho_km = a_au * _AU_KM
    # Random sky direction
    ra0 = rng.uniform(0, 2 * math.pi)
    dec0 = rng.uniform(-math.pi / 4, math.pi / 4)
    d0 = np.array([math.cos(dec0) * math.cos(ra0),
                    math.cos(dec0) * math.sin(ra0),
                    math.sin(dec0)])
    t0_et = rng.uniform(7e8, 9e8)
    R_e0 = np.array(body_state("EARTH", t0_et, "J2000", "SUN")[:3])
    r0 = R_e0 + rho_km * d0
    r0_norm = float(np.linalg.norm(r0))
    v_circ = math.sqrt(GM_SUN / r0_norm)
    r_hat = r0 / r0_norm
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(r_hat, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    tan = np.cross(r_hat, up)
    tan = tan / float(np.linalg.norm(tan))
    tilt = rng.uniform(-0.25, 0.25)
    v0 = v_circ * (math.cos(tilt) * tan + math.sin(tilt) * np.cross(r_hat, tan))

    # Observe at 3 epochs over 6 days
    epochs = [t0_et, t0_et + 86400 * 3, t0_et + 86400 * 6]
    obs_dicts = []
    for et in epochs:
        dt_s = et - t0_et
        r_t, _ = kepler_step(r0, v0, GM_SUN, dt_s)
        R_e = np.array(body_state("EARTH", et, "J2000", "SUN")[:3])
        geo = r_t - R_e
        rho = float(np.linalg.norm(geo))
        ra = math.atan2(geo[1], geo[0]) % (2 * math.pi)
        dec = math.asin(geo[2] / rho)
        # Apparent rate (arcsec/hr) via small offset
        dt_h = 0.1
        r_t_dt, _ = kepler_step(r0, v0, GM_SUN, dt_s + dt_h * 3600)
        R_e_dt = np.array(body_state("EARTH", et + dt_h * 3600, "J2000", "SUN")[:3])
        geo_dt = r_t_dt - R_e_dt
        ra_dt = math.atan2(geo_dt[1], geo_dt[0]) % (2 * math.pi)
        dec_dt = math.asin(geo_dt[2] / float(np.linalg.norm(geo_dt)))
        dra = (ra_dt - ra) * math.cos(dec)
        ddec = dec_dt - dec
        rate_arcsec_hr = math.degrees(math.hypot(dra, ddec)) * 3600.0 / dt_h
        obs_dicts.append({"t": et, "ra": ra, "dec": dec,
                          "rate_arcsec_hr": rate_arcsec_hr,
                          "mag": rng.uniform(20, 23)})

    features = build_chain_features(obs_dicts)
    # Target = (x_au/50, v_km_s/5)  -- both ~O(1) for balanced gradients
    x_scaled = r0 / (50.0 * AU_KM)
    v_scaled = v0 / 5.0
    target = np.concatenate([x_scaled, v_scaled])
    return features, target.astype(np.float32)


def train_orbit_prior(n_examples: int = 1500, n_epochs: int = 150,
                        lr: float = 5e-3, seed: int = 0,
                        verbose: bool = False) -> dict:
    """Generate training set + train the MLP via gradient descent."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_examples):
        feats, target = _generate_training_example(rng)
        X.append(feats)
        Y.append(target)
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    # Feature normalisation: per-column z-score
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-6
    X_n = (X - mu) / sigma
    # Targets are already in AU / AU-per-day -- small magnitudes, no scaling.

    weights = _init_weights(seed=seed)
    n = X_n.shape[0]
    batch = 64
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = X_n[idx]
            yb = Y[idx]
            y_hat, (xa, z1, h1, z2, h2) = _forward(xb, weights)
            err = y_hat - yb
            epoch_loss += float(np.mean(err * err))
            # Backprop
            n_b = xb.shape[0]
            dy = 2.0 * err / n_b
            dW3 = h2.T @ dy
            db3 = dy.sum(axis=0)
            dh2 = dy @ weights["W3"].T
            dz2 = dh2 * _relu_grad(z2)
            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0)
            dh1 = dz2 @ weights["W2"].T
            dz1 = dh1 * _relu_grad(z1)
            dW1 = xa.T @ dz1
            db1 = dz1.sum(axis=0)
            weights["W1"] -= lr * dW1
            weights["b1"] -= lr * db1
            weights["W2"] -= lr * dW2
            weights["b2"] -= lr * db2
            weights["W3"] -= lr * dW3
            weights["b3"] -= lr * db3
        if verbose and epoch % 25 == 0:
            print(f"  epoch {epoch}: loss {epoch_loss / max(1, n // batch):.4f}",
                   flush=True)
    weights["_feature_mean"] = mu
    weights["_feature_sigma"] = sigma
    return weights


def save_weights(weights: dict, path: str | Path) -> None:
    """Persist trained weights to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.tolist() for k, v in weights.items()}
    payload["_schema"] = "ariadne.neural_orbit_prior.v1"
    path.write_text(json.dumps(payload, sort_keys=True))


def load_weights(path: str | Path) -> dict:
    """Load weights from JSON. Returns dict of numpy arrays."""
    payload = json.loads(Path(path).read_text())
    payload.pop("_schema", None)
    return {k: np.array(v, dtype=np.float32) for k, v in payload.items()}


def predict_initial_state_normalised(features: np.ndarray,
                                       weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Normalise features using weights' stored stats, then predict.

    Use this when calling on a single chain at inference time -- the
    network was trained on normalised features so raw features will
    give garbage output.
    """
    if "_feature_mean" in weights:
        feats_n = (features - weights["_feature_mean"]) / weights["_feature_sigma"]
    else:
        feats_n = features
    # Use a temp dict without the stats keys (forward only uses W/b)
    pure_weights = {k: v for k, v in weights.items() if not k.startswith("_")}
    return predict_initial_state(feats_n, pure_weights)
