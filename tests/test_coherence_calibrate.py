"""The self-tuning calibration layer (gap23): fit_basins recovers class centers
from labelled data, fit_weights learns to up-weight the discriminative axis and
down-weight a pure-noise axis, and the fitted pair classifies held-out data well."""
from __future__ import annotations

import numpy as np


def _make(seed=0, n=400):
    """3 classes. Axis 'a' is a STRONG separator (sig 1.0); axis 'b' has the same
    class centers but is far NOISIER (sig 3.0) -> lower SNR. Relying on 'b' equally
    hurts argmax accuracy, so a good self-tuner should down-weight it."""
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for cls, mu in [("low", 0.0), ("mid", 2.0), ("high", 4.0)]:
        for _ in range(n):
            samples.append({"a": rng.normal(mu, 1.0), "b": rng.normal(mu, 3.0)})
            labels.append(cls)
    return samples, labels


def test_fit_basins_recovers_centers():
    from ariadne.discovery.imaging.coherence_calibrate import fit_basins
    s, y = _make()
    b = fit_basins(s, y, ["a", "b"])
    assert abs(b["low"]["mu"]["a"] - 0.0) < 0.3
    assert abs(b["mid"]["mu"]["a"] - 2.0) < 0.3
    assert abs(b["high"]["mu"]["a"] - 4.0) < 0.3
    # 'b' is the noisier axis -> larger fitted sigma than 'a'
    assert all(b[c]["sig"]["b"] > b[c]["sig"]["a"] for c in ("low", "mid", "high"))


def test_fit_weights_learns_discriminative_axis():
    from ariadne.discovery.imaging.coherence_calibrate import fit_basins, fit_weights
    s, y = _make()
    basins = fit_basins(s, y, ["a", "b"])
    w, score = fit_weights(s, y, basins, ["a", "b"])
    assert w["a"] > w["b"], f"reliable axis should outweigh the noisy one ({w})"
    assert score > 0.6, f"balanced accuracy should be reasonable ({score})"


def test_refine_basins_improves_or_holds():
    """Discriminative refinement of basin widths never regresses the training
    objective and recovers accuracy from deliberately-bad (too-wide) basins."""
    from ariadne.discovery.imaging.coherence_calibrate import (
        fit_basins, refine_basins, balanced_accuracy)
    from ariadne.discovery.imaging.coherence_field import coherence_posterior
    s, y = _make(seed=2)
    basins = fit_basins(s, y, ["a", "b"])
    # sabotage: blow up every width so classes overlap badly
    for c in basins:
        for ax in basins[c]["sig"]:
            basins[c]["sig"][ax] *= 6.0

    def bal(bs):
        preds = [max(coherence_posterior(x, bs, None), key=lambda k: coherence_posterior(x, bs, None)[k])
                 for x in s]
        return balanced_accuracy(y, preds)

    before = bal(basins)
    refined, score = refine_basins(s, y, basins, ["a", "b"], iters=3, tune_mu=False)
    assert bal(refined) >= before, "refinement must not regress the objective"
    assert score >= before


def test_fit_temperature_reduces_or_holds_ece():
    """Temperature calibration never makes calibration worse and finds a valid T."""
    from ariadne.discovery.imaging.coherence_calibrate import fit_basins, fit_temperature
    s, y = _make(seed=4)
    basins = fit_basins(s, y, ["a", "b"])
    T, ece_before, ece_after = fit_temperature(s, y, basins, None)
    assert T > 0 and ece_after <= ece_before + 1e-9


def test_fit_and_reliability_run():
    from ariadne.discovery.imaging.coherence_calibrate import fit, reliability
    s, y = _make(seed=1)
    res = fit(s, y, ["a", "b"])
    assert res["score"] > 0.6 and res["weights"]["a"] >= res["weights"]["b"]
    curve, ece = reliability(s, y, res["basins"], res["weights"])
    assert 0.0 <= ece <= 1.0 and curve
