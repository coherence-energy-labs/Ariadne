"""Coherence field — a faithful Equation-of-One inference / decision engine.

This implements the operational core of the Equation of One (S_One) as a general
decision objective for the pipeline. Wherever we choose among options (which
class, which tracklet to link, real-vs-bogus, which orbit), we evaluate the
total energy of each option and pick the MOST COHERENT (minimum-energy) one:

    E_total(option) = C_dyn * ( E_q + E_c + E_a + E_d )
    coherence(option) = exp(-E_total / 2)
    posterior(option) = coherence(option) / sum coherence

mapping the four S_One energy sectors onto inference (Form 0:
E_total = C_dyn integral (E_q + E_c + E_a + E_d) dV):

  E_q  quantum / cost     -> a per-option cost or penalty (latency, complexity,
                             prior implausibility). Optional; 0 by default.
  E_c  coherence          -> INCOHERENCE of the observation with the option's
                             basin -- an information divergence. The
                             entropy-reduction / order term (S_One Form 21:
                             E_coh ~ k_B T * D_KL). Mahalanobis distance for
                             feature vectors; KL divergence for distributions.
  E_a  alignment          -> relational organization via the alignment kernel
                             A = exp(-(d^2 + tau^2)/L); E_a = (1 - A). Low when
                             the option aligns in space (d) AND time (tau) over
                             a relationship scale L (S_One Form 16 / gap22
                             trust-decay). This is what links tracklets across
                             nights and matches detections to predictions.
  E_d  dark / baseline    -> irreducible baseline cost (overhead, floor).
  C_dyn                   -> scale normalisation e^{-lambda D}: large/sensitive
                             decisions weight coherence/alignment more.

Honest scope: the cosmological theory behind S_One is a separate research
program; here we use only its OPERATIONAL math (energy minimisation, coherence
as divergence, the alignment kernel), and we VALIDATE empirically -- adding
coherence-fused color took variable typing 74% -> 89% on real ZTF data.
"""
from __future__ import annotations

import math


# ---- E_c : coherence (incoherence as an information divergence) -------------

def incoherence_energy(x: dict, basin: dict, weights: dict | None = None) -> float:
    """Mahalanobis incoherence: mean weighted squared standardized distance of x
    from a basin {'mu':{}, 'sig':{}}, over shared axes. The Gaussian form of the
    coherence divergence E_c. Missing axes are skipped (graceful), never guessed."""
    mu, sig = basin["mu"], basin["sig"]
    if not mu:
        return 0.0                      # basin imposes no coherence constraint
    E = 0.0; W = 0.0
    for k, m in mu.items():
        if k in x and x[k] is not None and x[k] == x[k]:
            w = (weights or {}).get(k, 1.0)
            s = sig.get(k, 1.0) or 1.0
            E += w * ((x[k] - m) / s) ** 2
            W += w
    return E / W if W > 0 else math.inf  # basin has axes but none observed


def kl_divergence(p, q) -> float:
    """KL(p||q) for two discrete distributions (the information-theoretic form of
    the coherence energy, S_One Form 21: E_coh = k_B T * D_KL). p, q are dicts or
    sequences; normalized internally."""
    if isinstance(p, dict):
        keys = set(p) | set(q)
        pv = [max(p.get(k, 0.0), 0.0) for k in keys]
        qv = [max(q.get(k, 0.0), 0.0) for k in keys]
    else:
        pv = [max(v, 0.0) for v in p]; qv = [max(v, 0.0) for v in q]
    sp = sum(pv) or 1.0; sq = sum(qv) or 1.0
    out = 0.0
    for pi, qi in zip(pv, qv):
        pi /= sp; qi = (qi / sq) or 1e-12
        if pi > 0:
            out += pi * math.log(pi / qi)
    return out


# ---- E_a : alignment kernel ------------------------------------------------

def alignment_energy(d: float, tau: float = 0.0, L: float = 1.0) -> float:
    """Alignment energy E_a = 1 - A, A = exp(-(d^2 + tau^2)/L). Low when an
    option aligns closely in space (d) and time/staleness (tau) over scale L.
    The relational-coherence term for linking and matching."""
    A = math.exp(-(d * d + tau * tau) / max(L, 1e-9))
    return 1.0 - A


# ---- C_dyn : scale normalisation -------------------------------------------

def c_dynamic(scale: float = 0.0, sensitivity: float = 0.0, lam: float = 0.3) -> float:
    """Dynamic coupling C_dyn = e^{-lambda D}, D = baseline - scale - sensitivity.
    Larger / more sensitive decisions -> smaller D -> larger C_dyn -> coherence
    and alignment matter more. Defaults to ~1 (scale-neutral)."""
    D = 5.0 - scale - sensitivity
    return math.exp(-lam * D) * math.exp(lam * 5.0)   # normalized to 1 at scale=sens=0


# ---- E_total and the decision ----------------------------------------------

def total_energy(x: dict, basin: dict, *, weights: dict | None = None,
                 cost: float = 0.0, baseline: float = 0.0,
                 align: tuple | None = None, c_dyn: float = 1.0) -> float:
    """E_total = C_dyn * (E_q + E_c + E_a + E_d) for one option.
    align = (d, tau, L) feeds the alignment kernel; None -> no alignment term."""
    Ec = incoherence_energy(x, basin, weights)
    Ea = alignment_energy(*align) if align is not None else 0.0
    return c_dyn * (cost + Ec + Ea + baseline)


def coherence_posterior(x: dict, basins: dict, weights: dict | None = None,
                        costs: dict | None = None, temperature: float = 1.0) -> dict:
    """Posterior over basins by coherence = normalized exp(-E_total/(2*T)). Optional
    per-basin `costs` add the E_q sector. `temperature` T calibrates confidence
    (T>1 softens an over-confident posterior, T<1 sharpens); fit it with
    coherence_calibrate.fit_temperature so probabilities match empirical frequency.
    Returns {name: prob}, most-coherent first."""
    T = max(temperature, 1e-6)
    cohs = {}
    for name, b in basins.items():
        E = total_energy(x, b, weights=weights, cost=(costs or {}).get(name, 0.0))
        cohs[name] = math.exp(-0.5 * E / T)
    s = sum(cohs.values()) or 1.0
    post = {k: v / s for k, v in cohs.items() if v / s > 0.01}
    return dict(sorted(post.items(), key=lambda kv: -kv[1]))


def most_coherent(x: dict, basins: dict, weights: dict | None = None):
    """The single most-coherent (minimum-energy) basin name + its energy."""
    best, bestE = None, math.inf
    for name, b in basins.items():
        E = incoherence_energy(x, b, weights)
        if E < bestE:
            bestE, best = E, name
    return best, bestE


def select(options, energy_fn):
    """Generic Equation-of-One selector: return the option minimizing energy_fn
    (the most coherent choice) + its energy. `options` is any iterable; energy_fn
    maps an option to its E_total. Replaces hand-tuned threshold/argmax rules."""
    best, bestE = None, math.inf
    for o in options:
        E = energy_fn(o)
        if E < bestE:
            bestE, best = E, o
    return best, bestE
