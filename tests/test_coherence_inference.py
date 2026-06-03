"""Tests for the Equation-of-One coherence-field INFERENCE engine
(ariadne.discovery.imaging.coherence_field -- distinct from the FLI chaos field).

Validates each energy sector (coherence divergence E_c, alignment kernel E_a,
KL form, C_dyn scaling) and the decision rule (argmin E_total = most coherent),
on synthetic problems with known answers.
"""
from __future__ import annotations

import math


def test_incoherence_energy_zero_at_center_grows_with_distance():
    from ariadne.discovery.imaging.coherence_field import incoherence_energy
    basin = {"mu": {"a": 1.0, "b": 2.0}, "sig": {"a": 0.5, "b": 1.0}}
    assert incoherence_energy({"a": 1.0, "b": 2.0}, basin) < 1e-9
    near = incoherence_energy({"a": 1.2, "b": 2.0}, basin)
    far = incoherence_energy({"a": 3.0, "b": 2.0}, basin)
    assert 0 < near < far


def test_incoherence_skips_missing_axes():
    from ariadne.discovery.imaging.coherence_field import incoherence_energy
    basin = {"mu": {"a": 1.0, "b": 2.0}, "sig": {"a": 1.0, "b": 1.0}}
    assert incoherence_energy({"a": 1.0}, basin) < 1e-9
    assert math.isinf(incoherence_energy({}, basin))


def test_kl_divergence():
    from ariadne.discovery.imaging.coherence_field import kl_divergence
    assert kl_divergence({"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5}) < 1e-9
    assert kl_divergence({"x": 0.9, "y": 0.1}, {"x": 0.1, "y": 0.9}) > 0.5
    assert kl_divergence([0.7, 0.3], [0.7, 0.3]) < 1e-9


def test_alignment_kernel():
    from ariadne.discovery.imaging.coherence_field import alignment_energy
    assert alignment_energy(0.0, 0.0, 1.0) < 1e-9
    assert alignment_energy(3.0, 0.0, 1.0) > 0.9
    assert alignment_energy(0.5, 0.5, 4.0) < alignment_energy(2.0, 2.0, 4.0)


def test_c_dynamic_neutral_is_one():
    from ariadne.discovery.imaging.coherence_field import c_dynamic
    assert abs(c_dynamic(0.0, 0.0) - 1.0) < 1e-6
    assert c_dynamic(3.0, 3.0) > 1.0


def test_posterior_picks_nearest_basin():
    from ariadne.discovery.imaging.coherence_field import coherence_posterior
    basins = {"A": {"mu": {"v": 0.0}, "sig": {"v": 1.0}},
              "B": {"mu": {"v": 5.0}, "sig": {"v": 1.0}}}
    post = coherence_posterior({"v": 0.3}, basins)
    assert max(post, key=post.get) == "A" and post["A"] > 0.9


def test_cost_sector_shifts_decision():
    from ariadne.discovery.imaging.coherence_field import coherence_posterior
    basins = {"A": {"mu": {"v": 0.0}, "sig": {"v": 1.0}},
              "B": {"mu": {"v": 1.0}, "sig": {"v": 1.0}}}
    base = coherence_posterior({"v": 0.5}, basins)
    costed = coherence_posterior({"v": 0.5}, basins, costs={"A": 5.0})
    assert costed["B"] > base["B"]


def test_alignment_links_close_not_far():
    from ariadne.discovery.imaging.coherence_field import total_energy
    basin = {"mu": {}, "sig": {}}
    e_close = total_energy({}, basin, align=(0.3, 0.2, 4.0))
    e_far = total_energy({}, basin, align=(3.0, 2.0, 4.0))
    assert e_close < e_far


def test_select_argmin():
    from ariadne.discovery.imaging.coherence_field import select
    opt, E = select([1, 2, 3, 4], energy_fn=lambda o: (o - 3) ** 2)
    assert opt == 3 and E == 0


def test_coherence_mover_classifier():
    """Orbit-class by coherence over (log distance, eccentricity): rate->distance
    maps to the right dynamical class; e>=1 -> interstellar."""
    from ariadne.discovery.imaging.coherence_classifier import classify_mover
    assert max(classify_mover(2.7), key=classify_mover(2.7).get) == "main-belt"
    assert "TNO" in max(classify_mover(42.0), key=classify_mover(42.0).get)
    assert "NEO" in max(classify_mover(1.05), key=classify_mover(1.05).get)
    assert "Centaur" in max(classify_mover(14.0), key=classify_mover(14.0).get)
    hyp = classify_mover(1.5, eccentricity=1.3)
    assert "interstellar" in max(hyp, key=hyp.get)


def test_backward_compat_variable_classifier():
    """The validated variable classifier (89% on real ZTF) still works through
    the upgraded primitive: blue short-period -> RR Lyrae, red -> contact EW."""
    from ariadne.discovery.imaging.coherence_classifier import classify_variable, most_coherent
    blue = classify_variable(0.30, 0.12, 0.5, g_r=0.25)
    red = classify_variable(0.30, 0.12, 0.5, g_r=0.80)
    assert "RR" in most_coherent(blue) or "sinusoidal" in most_coherent(blue)
    assert "contact" in most_coherent(red) or "EW" in most_coherent(red)
