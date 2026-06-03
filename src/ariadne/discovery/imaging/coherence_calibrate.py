"""Self-tuning / calibration for the coherence engine — the gap23 upgrade.

The hand-set basins and weights in coherence_classifier / coherence_vet are a
human's guess. This module makes the engine LEARN them from labelled data, which
is what separates "validated" from "the smartest it can be":

  fit_basins   -> MLE/robust basin centers+widths per class per axis (replaces
                  hand-typed Gaussians).
  fit_weights  -> derivative-free coordinate-ascent on (balanced) accuracy —
                  the gap23 self-tuning, learning each axis's importance.
  class_priors / priors_to_costs -> turns the posterior into a proper Bayesian
                  one (E_q sector = prior implausibility), optional per use case.
  reliability  -> calibration diagnostic (do predicted probabilities match
                  empirical frequencies?).

All operate on the same {axis: value} feature dicts the coherence engine consumes,
so a fitted (basins, weights) pair drops straight into coherence_posterior.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .coherence_field import coherence_posterior


def fit_basins(samples, labels, axes, *, robust=True, min_sig=1e-3):
    """Fit {class: {'mu':{axis:..}, 'sig':{axis:..}}} from labelled feature dicts.
    robust=True uses median + 1.4826*MAD (outlier-resistant); else mean + std."""
    basins = {}
    for cls in sorted(set(labels)):
        mu, sig = {}, {}
        for ax in axes:
            vals = np.array([s[ax] for s, y in zip(samples, labels)
                             if y == cls and ax in s and s[ax] is not None and s[ax] == s[ax]],
                            dtype=float)
            if vals.size == 0:
                continue
            if robust:
                m = float(np.median(vals))
                sd = float(1.4826 * np.median(np.abs(vals - m)))
            else:
                m = float(np.mean(vals)); sd = float(np.std(vals))
            mu[ax] = m; sig[ax] = max(sd, min_sig)
        basins[cls] = {"mu": mu, "sig": sig}
    return basins


def class_priors(labels):
    c = Counter(labels); n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def priors_to_costs(priors, *, weight=1.0):
    """costs = -weight*ln(prior); feeding these as `costs` to coherence_posterior
    makes it a proper Bayesian posterior (likelihood x prior)."""
    return {k: -weight * math.log(max(p, 1e-12)) for k, p in priors.items()}


def _argmax(post):
    return max(post, key=post.get) if post else None


def balanced_accuracy(labels, preds):
    classes = sorted(set(labels)); recs = []
    for c in classes:
        idx = [i for i, y in enumerate(labels) if y == c]
        if idx:
            recs.append(float(np.mean([preds[i] == c for i in idx])))
    return float(np.mean(recs)) if recs else 0.0


def fit_weights(samples, labels, basins, axes, *, init=None, costs=None,
                iters=5, factors=(0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0),
                objective="balanced", label_of=_argmax):
    """Coordinate-ascent self-tuning of per-axis weights to maximize accuracy on
    labelled data (gap23, derivative-free). Returns (weights, score)."""
    def acc(w):
        preds = [label_of(coherence_posterior(s, basins, w, costs)) for s in samples]
        if objective == "balanced":
            return balanced_accuracy(labels, preds)
        return float(np.mean([p == y for p, y in zip(preds, labels)]))

    w = dict(init or {ax: 1.0 for ax in axes})
    best = acc(w)
    for _ in range(iters):
        improved = False
        for ax in axes:
            base = w[ax]
            for f in factors:
                cand = dict(w); cand[ax] = base * f
                s = acc(cand)
                if s > best + 1e-9:
                    best, w, improved = s, cand, True
        if not improved:
            break
    return w, best


def refine_basins(samples, labels, basins, axes, *, weights=None, costs=None,
                  iters=3, sig_factors=(0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0),
                  mu_steps=(-0.5, -0.25, 0.0, 0.25, 0.5), tune_mu=True,
                  objective="balanced", label_of=_argmax):
    """DISCRIMINATIVE refinement: adjust basin WIDTHS (and optionally nudge centers)
    to maximize (balanced) accuracy on labelled data -- i.e. tune the DECISION
    BOUNDARIES, not the per-class marginals. This is the principled way to beat
    hand-tuning: it starts from good (hand or MLE) basins and only moves a
    parameter when held-out accuracy improves, so it cannot regress on the
    training objective. Returns (refined_basins, score)."""
    import copy

    def acc(bs):
        preds = [label_of(coherence_posterior(s, bs, weights, costs)) for s in samples]
        if objective == "balanced":
            return balanced_accuracy(labels, preds)
        return float(np.mean([p == y for p, y in zip(preds, labels)]))

    bs = copy.deepcopy(basins)
    best = acc(bs)
    for _ in range(iters):
        improved = False
        for cls in list(bs):
            for ax in axes:
                sig = bs[cls].get("sig", {})
                if ax in sig:                       # widen/narrow this basin's axis
                    base = sig[ax]
                    for f in sig_factors:
                        cand = copy.deepcopy(bs); cand[cls]["sig"][ax] = max(base * f, 1e-3)
                        s = acc(cand)
                        if s > best + 1e-9:
                            best, bs, improved = s, cand, True
                mu = bs[cls].get("mu", {})
                if tune_mu and ax in mu and ax in bs[cls].get("sig", {}):
                    base_mu = mu[ax]; step = bs[cls]["sig"][ax]
                    for d in mu_steps:
                        if d == 0.0:
                            continue
                        cand = copy.deepcopy(bs); cand[cls]["mu"][ax] = base_mu + d * step
                        s = acc(cand)
                        if s > best + 1e-9:
                            best, bs, improved = s, cand, True
        if not improved:
            break
    return bs, best


def fit(samples, labels, axes, *, robust=True, use_priors=False, **wkw):
    """One-shot: fit basins + self-tune weights (+ optional priors->costs).
    Returns dict(basins=, weights=, costs=, score=)."""
    basins = fit_basins(samples, labels, axes, robust=robust)
    costs = priors_to_costs(class_priors(labels)) if use_priors else None
    weights, score = fit_weights(samples, labels, basins, axes, costs=costs, **wkw)
    return {"basins": basins, "weights": weights, "costs": costs, "score": score}


def reliability(samples, labels, basins, weights, costs=None, n_bins=10, temperature=1.0):
    """Calibration curve: for predictions binned by confidence, the empirical
    accuracy. A well-calibrated selector has accuracy ~= confidence per bin.
    Returns (curve, ECE) where ECE is the expected calibration error (lower=better)."""
    conf, correct = [], []
    for s, y in zip(samples, labels):
        post = coherence_posterior(s, basins, weights, costs, temperature=temperature)
        if not post:
            continue
        pred = _argmax(post)
        conf.append(post[pred]); correct.append(pred == y)
    conf = np.array(conf); correct = np.array(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= 1.0)
        if m.sum():
            out.append((float(np.mean(conf[m])), float(np.mean(correct[m])), int(m.sum())))
    ece = sum(n * abs(c - a) for c, a, n in out) / max(len(conf), 1)
    return out, float(ece)


def fit_temperature(samples, labels, basins, weights, costs=None,
                    grid=None, n_bins=10):
    """Find the temperature T that best CALIBRATES the posterior (minimizes ECE):
    so that when the model says p%, it is right ~p% of the time -- a NASA-grade
    requirement for trustworthy confidence. Returns (T, ece_before, ece_after)."""
    if grid is None:
        grid = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    _, ece0 = reliability(samples, labels, basins, weights, costs, n_bins, 1.0)
    bestT, bestE = 1.0, ece0
    for T in grid:
        _, e = reliability(samples, labels, basins, weights, costs, n_bins, T)
        if e < bestE:
            bestE, bestT = e, T
    return bestT, float(ece0), float(bestE)
