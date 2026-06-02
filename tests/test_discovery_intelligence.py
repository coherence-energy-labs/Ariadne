"""Tests for the discovery intelligence stack: classifiers, fusion, inference.

Covers: PSF centroiding, morphology classifier, deblender, orbital taxonomy,
real-vs-bogus, color taxonomy, difference imaging, shift-and-stack, inference
engine, predictive scheduler.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------- helpers ---------------------------------------

def _make_synthetic_star_image(size=64, x=32.0, y=32.0, flux=1000.0,
                                fwhm=3.0, bg=100.0, noise=5.0, seed=0):
    """Create a simple synthetic image with one Gaussian star + Poisson noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    sigma = fwhm / 2.355
    amplitude = flux / (2 * math.pi * sigma ** 2)
    img = bg + amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    img += rng.normal(0, noise, img.shape)
    return img


def _make_two_star_image(size=64, p1=(28.0, 32.0), p2=(36.0, 32.0),
                          flux=1000.0, fwhm=3.0, bg=100.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    sigma = fwhm / 2.355
    amp = flux / (2 * math.pi * sigma ** 2)
    img = (bg
           + amp * np.exp(-((xx - p1[0]) ** 2 + (yy - p1[1]) ** 2) / (2 * sigma ** 2))
           + amp * np.exp(-((xx - p2[0]) ** 2 + (yy - p2[1]) ** 2) / (2 * sigma ** 2)))
    img += rng.normal(0, noise, img.shape)
    return img


def _make_cosmic_ray_image(size=64, x=32, y=32, spike=5000, bg=100.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    img = bg + rng.normal(0, noise, (size, size))
    img[y, x] = spike                                     # single-pixel spike
    return img


def _make_streak_image(size=128, x=64, y=64, length=20.0, theta=0.4,
                       flux=2000.0, fwhm=3.0, bg=100.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    sigma = fwhm / 2.355
    img = np.full((size, size), bg, dtype=float)
    n_pts = int(length * 3)
    for i in range(n_pts):
        t = -length / 2 + i * length / max(1, n_pts - 1)
        cx = x + t * math.cos(theta)
        cy = y + t * math.sin(theta)
        img += (flux / n_pts / (2 * math.pi * sigma ** 2)) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    img += rng.normal(0, noise, img.shape)
    return img


# ---------------------------- PSF centroid ---------------------------------

def test_psf_fit_recovers_sub_pixel_centroid():
    from ariadne.discovery.imaging.psf_centroid import fit_psf_postage_stamp
    img = _make_synthetic_star_image(size=64, x=32.37, y=31.62,
                                      flux=5000.0, fwhm=3.0, bg=100.0, noise=3.0, seed=1)
    fit = fit_psf_postage_stamp(img, x_seed=32.0, y_seed=32.0,
                                  half_size=6, fwhm_guess_px=3.0)
    assert fit.success, "PSF fit should converge on bright synthetic star"
    assert abs(fit.x_sub - 32.37) < 0.10, f"x sub-pixel off: {fit.x_sub}"
    assert abs(fit.y_sub - 31.62) < 0.10, f"y sub-pixel off: {fit.y_sub}"
    assert 2.7 < fit.fwhm_px < 3.3, f"FWHM mismeasured: {fit.fwhm_px}"


def test_psf_fit_returns_failure_on_empty_stamp():
    from ariadne.discovery.imaging.psf_centroid import fit_psf_postage_stamp
    img = np.full((20, 20), 100.0)                       # flat, no source
    fit = fit_psf_postage_stamp(img, x_seed=10, y_seed=10, half_size=5)
    # may or may not "succeed" mathematically, but flux should be ~0
    assert fit.flux < 50.0 or not fit.success


# ---------------------------- Morphology ----------------------------------

def test_morphology_classifies_clean_psf_as_point():
    from ariadne.discovery.imaging.morphology import (
        classify_source, MorphologyClass)
    from ariadne.discovery.imaging.source_extraction import Source
    img = _make_synthetic_star_image(size=64, x=32, y=32,
                                      flux=3000.0, fwhm=3.0, seed=2)
    src = Source(ra=0, dec=0, flux=3000, mag=20, fwhm_px=3.0, mjd=60000,
                 image_id="t", x=32.0, y=32.0)
    v = classify_source(img, src, half_size=8)
    assert v.label == MorphologyClass.POINT, f"expected POINT, got {v.label}"
    assert v.confidence > 0.4


def test_morphology_classifies_cosmic_ray():
    from ariadne.discovery.imaging.morphology import (
        classify_source, MorphologyClass)
    from ariadne.discovery.imaging.source_extraction import Source
    img = _make_cosmic_ray_image(size=64, x=32, y=32, spike=8000, noise=3.0, seed=3)
    src = Source(ra=0, dec=0, flux=8000, mag=15, fwhm_px=0.5, mjd=60000,
                 image_id="t", x=32.0, y=32.0)
    v = classify_source(img, src, half_size=7)
    assert v.label in ("COSMIC_RAY", "POINT"), f"got {v.label}"


def test_morphology_classifies_streak():
    from ariadne.discovery.imaging.morphology import (
        classify_source, MorphologyClass)
    from ariadne.discovery.imaging.source_extraction import Source
    img = _make_streak_image(size=128, x=64, y=64, length=25.0,
                              theta=0.5, flux=4000.0, noise=3.0, seed=4)
    src = Source(ra=0, dec=0, flux=4000, mag=18, fwhm_px=4.0, mjd=60000,
                 image_id="t", x=64.0, y=64.0)
    v = classify_source(img, src, half_size=15)
    # An extended streak is plausibly STREAK, EXTENDED, or BLEND (multiple
    # peaks along the trail). Any of these is correct -- the important thing
    # is the classifier did NOT label it as a clean POINT source.
    assert v.label != "POINT", f"streak should not be classified POINT, got {v.label}"
    assert v.ellipticity > 0.3


def test_morphology_flags_edge_artefacts():
    from ariadne.discovery.imaging.morphology import (
        classify_source, MorphologyClass)
    from ariadne.discovery.imaging.source_extraction import Source
    img = _make_synthetic_star_image(size=64, x=2.0, y=32.0, seed=5)
    src = Source(ra=0, dec=0, flux=1000, mag=20, fwhm_px=3.0, mjd=60000,
                 image_id="t", x=2.0, y=32.0)
    v = classify_source(img, src, half_size=6, edge_margin_px=4)
    assert v.label == MorphologyClass.EDGE_ARTEFACT


# ---------------------------- Deblender ------------------------------------

def test_deblender_splits_two_close_stars():
    from ariadne.discovery.imaging.deblend import deblend_source
    from ariadne.discovery.imaging.source_extraction import Source
    img = _make_two_star_image(size=64, p1=(28.0, 32.0), p2=(36.0, 32.0),
                                flux=3000.0, fwhm=3.0, noise=3.0, seed=6)
    src = Source(ra=0, dec=0, flux=6000, mag=20, fwhm_px=4.0, mjd=60000,
                 image_id="t", x=32.0, y=32.0)
    components = deblend_source(img, src, half_size=10, min_sep_px=3)
    assert len(components) == 2, f"expected 2 components, got {len(components)}"
    xs = sorted([c.x for c in components])
    assert abs(xs[0] - 28.0) < 1.0
    assert abs(xs[1] - 36.0) < 1.0


def test_deblender_returns_single_when_no_blend():
    from ariadne.discovery.imaging.deblend import deblend_source
    from ariadne.discovery.imaging.source_extraction import Source
    img = _make_synthetic_star_image(size=64, x=32, y=32, seed=7)
    src = Source(ra=0, dec=0, flux=1000, mag=20, fwhm_px=3.0, mjd=60000,
                 image_id="t", x=32.0, y=32.0)
    components = deblend_source(img, src, half_size=7)
    assert len(components) == 1


# ---------------------------- Orbital taxonomy -----------------------------

def test_taxonomy_classifies_mba():
    from ariadne.discovery.taxonomy import classify_orbit, OrbitClass
    t = classify_orbit(a_au=2.77, e=0.08, i_deg=10.6)        # Ceres
    assert t.label == OrbitClass.MBA, f"expected MBA, got {t.label}"
    assert t.confidence > 0.5


def test_taxonomy_classifies_tno():
    from ariadne.discovery.taxonomy import classify_orbit, OrbitClass
    t = classify_orbit(a_au=44.0, e=0.05, i_deg=3.0)         # cold classical
    assert t.label == OrbitClass.CLASSICAL_KBO
    assert t.confidence > 0.7


def test_taxonomy_classifies_apollo_neo():
    from ariadne.discovery.taxonomy import classify_orbit, OrbitClass
    t = classify_orbit(a_au=1.5, e=0.5, i_deg=10.0)
    assert t.label == OrbitClass.APOLLO


def test_taxonomy_classifies_sednoid():
    from ariadne.discovery.taxonomy import classify_orbit, OrbitClass
    t = classify_orbit(a_au=500.0, e=0.85, i_deg=12.0)       # Sedna
    assert t.label == OrbitClass.SEDNOID
    assert t.confidence > 0.7


def test_taxonomy_classifies_hyperbolic_comet():
    from ariadne.discovery.taxonomy import classify_orbit, OrbitClass
    t = classify_orbit(a_au=-100.0, e=1.5, i_deg=45.0)
    assert t.label == OrbitClass.COMET_HYPERBOLIC


def test_taxonomy_classify_state_round_trip():
    from ariadne.discovery.taxonomy import classify_state, OrbitClass
    from ariadne.dynamics.secular import elements_to_state
    p, v = elements_to_state(2.77, 0.08, 10.6, 30.0, 50.0, 180.0)
    t = classify_state(np.asarray(p), np.asarray(v))
    assert t.label == OrbitClass.MBA


# ---------------------------- Real/bogus -----------------------------------

def test_realbogus_drops_zero_motion_tracklet():
    from ariadne.discovery.realbogus import score_realbogus
    tr = {"rate_arcsec_hr": 0.001, "rms_arcsec": 1.0, "members": []}
    v = score_realbogus(tr)
    assert not v.is_real
    assert any(name == "zero_motion" for name, _ in v.rules_fired)


def test_realbogus_drops_satellite_trail():
    from ariadne.discovery.realbogus import score_realbogus
    tr = {"rate_arcsec_hr": 8000.0, "rms_arcsec": 5.0, "members": []}
    v = score_realbogus(tr)
    assert not v.is_real
    assert any(name == "implausible_rate" for name, _ in v.rules_fired)


def test_realbogus_keeps_genuine_tno_tracklet():
    from ariadne.discovery.realbogus import score_realbogus
    tr = {"rate_arcsec_hr": 1.5, "rms_arcsec": 1.0, "members": []}
    v = score_realbogus(tr)
    assert v.is_real


# ---------------------------- Color taxonomy -------------------------------

def test_color_taxonomy_classifies_s_type():
    from ariadne.discovery.colors import classify_colors
    # S-type centroid: g-r=0.55, r-i=0.20, i-z=0.13
    mags = {"g": 21.55, "r": 21.00, "i": 20.80, "z": 20.67}
    t = classify_colors(mags)
    assert t.label == "S", f"expected S-type, got {t.label}"
    assert t.confidence > 0.5


def test_color_taxonomy_returns_unknown_with_one_band():
    from ariadne.discovery.colors import classify_colors
    mags = {"g": 21.0}
    t = classify_colors(mags, min_bands=2)
    assert t.label == "UNKNOWN"


def test_color_taxonomy_distinguishes_tno_red_vs_gray():
    from ariadne.discovery.colors import classify_colors
    red = {"g": 22.0, "r": 21.15, "i": 20.75, "z": 20.5}      # g-r=0.85, r-i=0.40, i-z=0.25
    gray = {"g": 22.0, "r": 21.55, "i": 21.40, "z": 21.32}     # g-r=0.45, r-i=0.15, i-z=0.08
    tr = classify_colors(red, centroid_set="tno")
    tg = classify_colors(gray, centroid_set="tno")
    assert tr.label in ("TNO_RED", "TNO_VERY_RED")
    assert tg.label == "TNO_GRAY"


# ---------------------------- Difference imaging ---------------------------

def test_difference_imaging_finds_moving_source():
    sci = _make_synthetic_star_image(size=64, x=20.0, y=32.0, flux=3000, seed=10)
    ref = _make_synthetic_star_image(size=64, x=32.0, y=32.0, flux=3000, seed=11)
    # Put a "moving object" at (45, 32) in science only
    yy, xx = np.mgrid[:64, :64]
    sigma = 3.0 / 2.355
    sci += 1500 / (2 * math.pi * sigma ** 2) * np.exp(
        -((xx - 45) ** 2 + (yy - 32) ** 2) / (2 * sigma ** 2))
    from ariadne.discovery.imaging.difference import subtract_reference
    res = subtract_reference(sci, ref, max_shift_px=20)
    # peak should be near (45, 32) -- the moving source -- not the matched star
    py, px = np.unravel_index(np.argmax(res.residual), res.residual.shape)
    assert 40 <= px <= 50, f"peak x={px} should be near 45"
    assert 28 <= py <= 36, f"peak y={py} should be near 32"


# ---------------------------- Shift-and-stack ------------------------------

def test_shift_stack_recovers_known_rate():
    """Make a moving object at known rate; verify the matching hypothesis
    produces the brightest peak."""
    from ariadne.discovery.imaging.shift_stack import shift_stack
    size = 64
    fwhm = 3.0; sigma = fwhm / 2.355
    yy, xx = np.mgrid[:size, :size]
    bg = 100.0; noise = 4.0
    rng = np.random.default_rng(20)
    images, epochs = [], []
    vra = 4.0; vdec = 0.0                                # 4 arcsec/hr in RA
    pixel_scale = 0.25
    x0, y0 = 32.0, 32.0
    for k in range(5):
        t_hr = k * 1.0
        dt_days = t_hr / 24.0
        dx_px = vra * t_hr / pixel_scale
        dy_px = vdec * t_hr / pixel_scale
        xc = x0 + dx_px
        yc = y0 + dy_px
        img = bg + 1500 / (2 * math.pi * sigma ** 2) * np.exp(
            -((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma ** 2))
        img += rng.normal(0, noise, img.shape)
        images.append(img)
        epochs.append(60000.0 + dt_days)
    correct = shift_stack(images, epochs, vra, vdec,
                           pixel_scale_arcsec=pixel_scale)
    wrong = shift_stack(images, epochs, 0.0, 0.0,         # zero-rate stack
                         pixel_scale_arcsec=pixel_scale)
    assert correct.peak_sigma > wrong.peak_sigma, \
        (f"correct-rate stack ({correct.peak_sigma:.1f}) should beat zero-rate "
         f"({wrong.peak_sigma:.1f})")


# ---------------------------- Inference engine -----------------------------

def test_inference_recovers_tno_from_evidence():
    from ariadne.discovery.inference import Evidence, infer
    ev = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        rate_arcsec_hr=1.0, apparent_mag=22.5, band="r",
        morphology_label="POINT", morphology_confidence=0.85,
        n_detections=6, arc_days=10.0, rms_arcsec=1.2,
        skybot_match_names=[],
    )
    res = infer(ev)
    assert res.best is not None
    # Top hypothesis should be moving_object, in the TNO family
    assert res.best.class_ == "moving_object"
    assert res.best.orbital_class in (
        "CLASSICAL_KBO", "HOT_CLASSICAL", "RESONANT_KBO",
        "SCATTERED_KBO", "CENTAUR",
    ), f"expected TNO-family, got {res.best.orbital_class}"


def test_inference_flags_cosmic_ray():
    from ariadne.discovery.inference import Evidence, infer
    ev = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        apparent_mag=18.0, band="r",
        morphology_label="COSMIC_RAY", morphology_confidence=0.95,
        n_detections=1,
    )
    res = infer(ev)
    # cosmic_ray artefact should be at or near the top
    top3 = [h.label for h in res.hypotheses[:3]]
    assert "cosmic_ray" in top3, f"cosmic_ray missing from top 3: {top3}"


def test_inference_flags_satellite_streak():
    from ariadne.discovery.inference import Evidence, infer
    ev = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        rate_arcsec_hr=5000.0,
        morphology_label="STREAK", morphology_confidence=0.9,
        apparent_mag=18.0,
    )
    res = infer(ev)
    # satellite_trail should be highly probable
    sat_post = next((h.posterior for h in res.hypotheses if h.label == "satellite_trail"), 0.0)
    assert sat_post > 0.10, f"satellite trail posterior too low: {sat_post}"


def test_inference_entropy_reflects_ambiguity():
    """Distinctive evidence -> lower posterior entropy than no-evidence."""
    from ariadne.discovery.inference import Evidence, infer
    # NEO-like: very fast (200"/hr is well inside the Apollo range, well outside
    # MBA/TNO ranges). The posterior should peak sharply on Apollo / Aten.
    distinctive = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        rate_arcsec_hr=200.0, apparent_mag=18.0,
        morphology_label="POINT", morphology_confidence=0.95,
        n_detections=6, arc_days=2, rms_arcsec=1.0,
        skybot_match_names=[],
    )
    poor_ev = Evidence(mjd=60450, ra_deg=180, dec_deg=15)        # almost nothing
    rich = infer(distinctive)
    poor = infer(poor_ev)
    assert poor.entropy > rich.entropy + 0.1, \
        f"distinctive evidence (entropy {rich.entropy:.2f}) should be MUCH " \
        f"more confident than no-evidence (entropy {poor.entropy:.2f})"


def test_inference_narrative_is_informative():
    from ariadne.discovery.inference import Evidence, infer
    ev = Evidence(mjd=60450, ra_deg=180, dec_deg=15,
                  rate_arcsec_hr=2.0, apparent_mag=21.0,
                  morphology_label="POINT", n_detections=4)
    res = infer(ev)
    assert res.narrative
    assert ":" in res.narrative


def test_inference_certificate_and_audit_are_tamper_evident():
    from ariadne.discovery.inference import (
        Evidence, infer, validate_inference_certificate)
    ev = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        rate_arcsec_hr=1.0, apparent_mag=22.5,
        morphology_label="POINT", morphology_confidence=0.9,
        n_detections=5, arc_days=8.0, rms_arcsec=1.0,
        skybot_match_names=[],
    )
    res = infer(ev)
    assert res.evidence_audit is not None
    assert res.evidence_audit.n_channels >= 6
    assert res.posterior_check is not None
    assert res.certificate["payload_hash"]
    assert validate_inference_certificate(ev, res)
    res.best.posterior *= 0.5
    assert not validate_inference_certificate(ev, res)


def test_inference_can_fail_closed_on_contradictory_evidence():
    from ariadne.discovery.inference import Evidence, infer
    ev = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        rate_arcsec_hr=2.0,
        morphology_label="COSMIC_RAY", morphology_confidence=0.95,
        n_detections=5, arc_days=4.0,
    )
    res = infer(ev, fail_closed_on_contradiction=True)
    assert res.best is None
    assert res.evidence_audit.fail_closed
    assert res.recommended_followup["action"] == "manual_review"


def test_temperature_calibration_softens_posterior():
    from ariadne.discovery.inference import CalibrationConfig, Evidence, infer
    ev = Evidence(
        mjd=60450, ra_deg=180, dec_deg=15,
        rate_arcsec_hr=200.0, apparent_mag=18.0,
        morphology_label="POINT", morphology_confidence=0.95,
        n_detections=6, arc_days=2, rms_arcsec=1.0,
        skybot_match_names=[],
    )
    sharp = infer(ev, calibration=CalibrationConfig(temperature=1.0))
    soft = infer(ev, calibration=CalibrationConfig(temperature=3.0))
    assert soft.best.label == sharp.best.label
    assert soft.best.posterior < sharp.best.posterior
    assert soft.entropy > sharp.entropy


def test_reliability_report_and_temperature_fit():
    from ariadne.discovery.inference import (
        Evidence, fit_temperature, reliability_report)
    cases = [
        (Evidence(rate_arcsec_hr=1.0, apparent_mag=22.5, morphology_label="POINT",
                  morphology_confidence=0.9, n_detections=6, arc_days=10,
                  rms_arcsec=1.0, skybot_match_names=[]), "CLASSICAL_KBO"),
        (Evidence(rate_arcsec_hr=5000.0, morphology_label="STREAK",
                  morphology_confidence=0.9, apparent_mag=18.0), "satellite_trail"),
        (Evidence(morphology_label="COSMIC_RAY", morphology_confidence=0.95,
                  apparent_mag=18.0, n_detections=1), "cosmic_ray"),
    ]
    rep = reliability_report(cases)
    assert rep.n == 3
    assert 0.0 <= rep.ece <= 1.0
    cfg, rep2 = fit_temperature(cases, grid=(1.0, 2.0))
    assert cfg.temperature in (1.0, 2.0)
    assert rep2.n == 3


# ---------------------------- Predictive scheduler -------------------------

def test_scheduler_learns_from_history(tmp_path):
    """The scheduler's historical-confirmation lookup must reflect ledger writes."""
    from ariadne.discovery.predictive import PredictiveScheduler
    sched = PredictiveScheduler(tmp_path / "ledger.json")
    # Cold start: confirmation rate is None (no samples)
    assert sched.historical_confirmation_rate(
        "confirmed_orbit_no_match", "alert_and_submit_mpc") is None
    # Record failures of one action, successes of another
    for _ in range(5):
        sched.record_outcome(evidence_class="confirmed_orbit_no_match",
                              action="observe_second_night",
                              outcome="refuted")
    for _ in range(5):
        sched.record_outcome(evidence_class="confirmed_orbit_no_match",
                              action="alert_and_submit_mpc",
                              outcome="confirmed")
    sched.save()
    # Reload + verify the ledger was persisted + lookup works
    sched2 = PredictiveScheduler(tmp_path / "ledger.json")
    summary = sched2.summary()
    assert summary["n_records"] == 10
    assert sched2.historical_confirmation_rate(
        "confirmed_orbit_no_match", "alert_and_submit_mpc") == 1.0
    assert sched2.historical_confirmation_rate(
        "confirmed_orbit_no_match", "observe_second_night") == 0.0


def test_scheduler_prefers_known_winner_after_training(tmp_path):
    """After training, the scheduler should recommend a confirming action
    over a refuting one when both are equally discriminative."""
    from ariadne.discovery.predictive import PredictiveScheduler
    sched = PredictiveScheduler(tmp_path / "ledger.json")
    for _ in range(8):
        sched.record_outcome(evidence_class="single_detection_no_rate",
                              action="observe_second_night", outcome="confirmed")
        sched.record_outcome(evidence_class="single_detection_no_rate",
                              action="archive_search", outcome="refuted")
    # Force a head-to-head: exclude all OTHER actions so the scheduler must
    # choose between the trained-confirm action and the trained-refute action.
    all_actions = ("observe_second_night", "observe_multi_band",
                   "observe_deep_stack", "query_skybot", "query_horizons",
                   "alert_and_submit_mpc", "archive_search", "monitor_only", "discard")
    exclude = [a for a in all_actions if a not in
               ("observe_second_night", "archive_search")]
    reco = sched.recommend(evidence_class="single_detection_no_rate",
                            hypothesis_posterior=0.5, exclude=exclude)
    assert reco.action == "observe_second_night", \
        f"expected confirming action, got {reco.action!r}"


def test_scheduler_classify_evidence_buckets():
    from ariadne.discovery.predictive import classify_evidence
    from ariadne.discovery.inference import Evidence
    e_streak = Evidence(rate_arcsec_hr=500.0, morphology_label="STREAK")
    assert classify_evidence(e_streak) == "fast_mover_streak"
    e_cr = Evidence(morphology_label="COSMIC_RAY", morphology_confidence=0.9)
    assert classify_evidence(e_cr) == "likely_artefact"
    e_orb = Evidence(n_detections=5, rms_arcsec=1.0, skybot_match_names=[])
    assert classify_evidence(e_orb) == "confirmed_orbit_no_match"


def test_inference_uses_scheduler_when_provided(tmp_path):
    from ariadne.discovery.inference import Evidence, infer
    from ariadne.discovery.predictive import PredictiveScheduler
    sched = PredictiveScheduler(tmp_path / "ledger.json")
    ev = Evidence(mjd=60450, ra_deg=180, dec_deg=15,
                  rate_arcsec_hr=1.0, apparent_mag=22.5,
                  morphology_label="POINT", n_detections=6, arc_days=10,
                  rms_arcsec=1.2, skybot_match_names=[])
    res = infer(ev, scheduler=sched)
    assert res.recommended_followup.get("source") == "learned_scheduler"
    assert "action" in res.recommended_followup
    res2 = infer(ev)
    assert res2.recommended_followup.get("source") == "heuristic"
