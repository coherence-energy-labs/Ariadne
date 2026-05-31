"""Tests for the sensitivity layer: streaks, gaia refine, N-body fit, MCMC orbit,
multi-broker fusion, recovery validation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ----------------------------- helpers -------------------------------------

def _make_streak_image(size=128, x=64, y=64, length=24.0, theta=0.4,
                       flux=4000.0, fwhm=3.0, bg=100.0, noise=3.0, seed=0):
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


def _make_blank_image(size=64, bg=100.0, noise=3.0, seed=0):
    return _make_streak_image(size=size, x=size // 2, y=size // 2,
                                length=0.0, flux=0.0, bg=bg, noise=noise, seed=seed)


# ----------------------------- Streak detector -----------------------------

def test_streaks_detects_diagonal_trail():
    from ariadne.discovery.imaging.streaks import detect_streaks
    img = _make_streak_image(size=128, x=64, y=64, length=30.0,
                              theta=0.4, flux=8000.0, noise=2.0, seed=1)
    streaks = detect_streaks(img, sigma_threshold=4.0, min_length_px=10.0)
    assert len(streaks) >= 1, "should detect at least one streak"
    s = streaks[0]
    assert 20 < s.length_px < 50, f"length {s.length_px} not in expected range"
    # Hough returns the NORMAL angle to the line. The line direction is
    # normal + pi/2 (mod pi). We made the line along angle 0.4 rad, so the
    # Hough-normal should be 0.4 - pi/2 ~= -1.17 rad. Accept any angle that,
    # rotated by pi/2 mod pi, sits near 0.4 rad.
    line_angle = (s.theta_rad + math.pi / 2) % math.pi
    line_angle = min(line_angle, math.pi - line_angle)         # wrap to [0, pi/2]
    expected = 0.4
    assert abs(line_angle - expected) < 0.2 or abs((math.pi - line_angle) - expected) < 0.2, \
        f"line-direction {line_angle:.3f} rad should be near {expected} rad"


def test_streaks_returns_empty_on_flat_image():
    from ariadne.discovery.imaging.streaks import detect_streaks
    img = _make_blank_image(size=64, seed=2)
    streaks = detect_streaks(img, sigma_threshold=4.0, min_length_px=10.0)
    assert len(streaks) == 0


def test_streak_classifier_returns_a_label():
    from ariadne.discovery.imaging.streaks import detect_streaks, classify_streak
    img = _make_streak_image(size=128, length=30.0, theta=0.5,
                               flux=8000.0, noise=2.0, seed=3)
    streaks = detect_streaks(img, min_length_px=10.0)
    if streaks:
        c = classify_streak(streaks[0], exposure_seconds=30.0,
                            pixel_scale_arcsec=0.25)
        assert "label" in c
        assert "rate_arcsec_hr" in c
        assert isinstance(c["confidence"], float)


# ----------------------------- Gaia refinement (no-network path) -----------

def test_gaia_refinement_passthrough_without_network():
    from ariadne.discovery.imaging.gaia_refine import refine_to_gaia
    from ariadne.discovery.imaging.source_extraction import Source
    sources = [Source(ra=180.0, dec=20.0, flux=1000, mag=20, fwhm_px=3,
                       mjd=60000, image_id="t", x=0, y=0)]
    # Force passthrough by giving a query position far from real data + tight tolerance
    # (real network call will return [] for an obscure RA)
    refined, report = refine_to_gaia(
        sources, image_centre_ra_deg=180.0, image_centre_dec_deg=20.0,
        image_radius_deg=0.001,                            # tiny -> no Gaia stars in box
        match_tol_arcsec=0.5)
    # Either passthrough (no network/no Gaia stars) or successful refinement;
    # both are valid -- the point is the code returns SAME number of sources.
    assert len(refined) == len(sources)


# ----------------------------- N-body orbit fit ----------------------------

@pytest.mark.slow
def test_nbody_fit_runs_end_to_end():
    """N-body LM fit produces a well-formed result. SLOW (~30s of integrations).

    Recovery accuracy on a short arc is bounded by the 2-body seed; we test
    that the N-body wrapper completes, returns the expected dict shape, and
    has the perturbers_used list populated.
    """
    from ariadne.discovery import linkage
    from ariadne.discovery.orbit_fit_nbody import fit_orbit_nbody
    from ariadne.discovery import iod
    tracklets, e0 = linkage.synthesize_tracklets(
        orbits=[{"a_au": 45.0, "e": 0.1, "i": 10.0,
                  "Omega": 30.0, "omega": 50.0}],
        epoch="2026-01-01T00:00:00",
        night_offsets_days=(0, 5, 14, 30),                # SHORT 30-day arc
        n_interlopers=0,
    )
    t_ref = float(np.median([t["t"] for t in tracklets]))
    seed = iod.iod_hypothesis_search(tracklets, t_ref=t_ref)
    assert seed is not None, "IOD should converge on synthetic"
    fit_2body = iod.fit_orbit_lm(tracklets, t_ref, seed["x_init"], seed["v_init"])
    assert fit_2body["success"]
    fit_nbody = fit_orbit_nbody(
        tracklets, t_ref, fit_2body["x_fit"], fit_2body["v_fit"],
        perturbers=("JUPITER", "NEPTUNE"), max_nfev=80)
    assert "x_fit" in fit_nbody and "v_fit" in fit_nbody
    assert "perturbers_used" in fit_nbody
    assert fit_nbody["perturbers_used"] == ["JUPITER", "NEPTUNE"]
    # Either the N-body fit converged (low RMS) or it returned the seed
    # values with a fail flag; both are acceptable for the test.
    assert math.isfinite(fit_nbody["rms_arcsec"]) or not fit_nbody["success"]


# ----------------------------- MCMC orbit posterior ------------------------

def test_mcmc_posterior_returns_well_formed_result():
    """MCMC orbit posterior on a short arc; check the output shape + that
    the best-sample is near the seed."""
    from ariadne.discovery import linkage, iod
    from ariadne.discovery.bayes_orbit import sample_posterior
    tracklets, e0 = linkage.synthesize_tracklets(
        orbits=[{"a_au": 40.0, "e": 0.05, "i": 8.0,
                  "Omega": 30.0, "omega": 50.0}],
        epoch="2026-01-01T00:00:00",
        night_offsets_days=(0, 10, 30, 90),
        n_interlopers=0,
    )
    seed = iod.iod_hypothesis_search(tracklets)
    assert seed is not None
    fit = iod.fit_orbit_lm(tracklets, t_ref=float(np.median([t["t"] for t in tracklets])),
                            x_init=seed["x_init"], v_init=seed["v_init"])
    post = sample_posterior(
        tracklets,
        t_ref=float(np.median([t["t"] for t in tracklets])),
        x_seed_km=fit["x_fit"], v_seed_kms=fit["v_fit"],
        n_walkers=12, n_steps=120, burn_in=40, thin=2,
    )
    assert post.chain.shape[2] == 6, "chain should have 6 state dims"
    assert post.sampler_used in ("emcee", "metropolis")
    # Quantile triplets are populated
    assert len(post.x_quantiles_au) == 3
    # Best sample should land somewhere reasonable for an a=40 AU orbit
    x_au = np.linalg.norm(post.best_sample[:3]) / 149597870.7
    assert 5 < x_au < 100, f"best-sample distance {x_au} AU out of range"


# ----------------------------- Multi-broker fusion -------------------------

def test_fusion_merges_same_source_across_surveys():
    from ariadne.discovery.fusion import fuse_alerts
    from ariadne.discovery.brokers.base import Alert
    alerts = [
        Alert(survey="ZTF", alert_id="z1", obj_id="z_obj", mjd=60450.0,
              ra=180.0001, dec=20.0001, mag=21.0, band="r"),
        Alert(survey="ATLAS", alert_id="a1", obj_id="a_obj", mjd=60450.005,
              ra=180.0000, dec=20.0000, mag=21.1, band="o"),
        Alert(survey="PanSTARRS", alert_id="p1", obj_id="p_obj", mjd=60450.01,
              ra=179.9999, dec=20.0001, mag=21.05, band="r"),
        # an unrelated detection elsewhere
        Alert(survey="ZTF", alert_id="z2", obj_id="z_other", mjd=60450.0,
              ra=170.0, dec=10.0, mag=20.5, band="g"),
    ]
    fused = fuse_alerts(alerts, pos_tol_arcsec=1.5, time_tol_minutes=30.0)
    assert len(fused) == 2, f"expected 2 fused detections, got {len(fused)}"
    # The first fused should have 3 surveys
    fused_3 = max(fused, key=lambda f: f.n_alerts)
    assert fused_3.n_alerts == 3
    assert set(fused_3.surveys) == {"ZTF", "ATLAS", "PanSTARRS"}


def test_fusion_keeps_distant_sources_separate():
    from ariadne.discovery.fusion import fuse_alerts
    from ariadne.discovery.brokers.base import Alert
    alerts = [
        Alert(survey="A", alert_id="1", obj_id="o1", mjd=60450.0,
              ra=10.0, dec=20.0, mag=21.0, band="r"),
        Alert(survey="B", alert_id="2", obj_id="o2", mjd=60450.001,
              ra=10.1, dec=20.0, mag=21.0, band="r"),                # 360" away
    ]
    fused = fuse_alerts(alerts, pos_tol_arcsec=2.0)
    assert len(fused) == 2


def test_fusion_to_alerts_returns_alert_list():
    from ariadne.discovery.fusion import fuse_to_alerts
    from ariadne.discovery.brokers.base import Alert
    alerts = [Alert(survey="ZTF", alert_id="z1", obj_id="o1", mjd=60450.0,
                     ra=180.0, dec=20.0, mag=21.0, band="r")]
    fused = fuse_to_alerts(alerts)
    assert len(fused) == 1
    assert hasattr(fused[0], "ra")
    assert hasattr(fused[0], "meta")


# ----------------------------- Sensitivity validation ----------------------

def test_inject_synthetic_objects_returns_alerts_and_truth():
    from ariadne.validate.sensitivity import inject_synthetic_objects, make_population
    pop = make_population(n_objects=4, seed=11)
    alerts, truth = inject_synthetic_objects(
        pop, epoch="2026-03-01T00:00:00", n_nights=3, n_per_night=2)
    assert len(truth) == 4
    assert all(a.survey == "synth" for a in alerts)
    # 4 objects x 3 nights x 2 detections = 24 alerts
    assert len(alerts) == 24
    # truth IDs are 0..3
    assert {t.truth_id for t in truth} == {0, 1, 2, 3}


def test_evaluate_recovery_handles_empty_pipeline_output():
    from ariadne.validate.sensitivity import evaluate_recovery, make_population
    from ariadne.validate.sensitivity import inject_synthetic_objects
    pop = make_population(n_objects=3)
    alerts, truth = inject_synthetic_objects(pop, epoch="2026-03-01T00:00:00")
    rep = evaluate_recovery([], truth)
    assert rep.n_injected == 3
    assert rep.n_recovered == 0
    assert rep.recovery_rate == 0.0


def test_evaluate_recovery_credits_matched_candidates():
    from ariadne.validate.sensitivity import evaluate_recovery, InjectionRecord
    from ariadne.discovery.brokers.base import Alert
    members = [Alert(survey="synth", alert_id=f"a{i}",
                      obj_id="truth_0", mjd=60450 + i, ra=180, dec=20,
                      mag=20, band="r", meta={"truth_id": 0})
               for i in range(6)]
    cand = {"status": "accepted", "members": members, "rms_arcsec": 1.5}
    truth = [InjectionRecord(truth_id=0, a_au=45, e=0.1, i_deg=10,
                              apparent_mag=22, rate_arcsec_hr=1.0,
                              n_planted_alerts=6)]
    rep = evaluate_recovery([cand], truth)
    assert rep.n_recovered == 1
    assert rep.recovery_rate == 1.0


def test_evaluate_recovery_counts_false_positives():
    from ariadne.validate.sensitivity import evaluate_recovery, InjectionRecord
    from ariadne.discovery.brokers.base import Alert
    bogus_members = [Alert(survey="ZTF", alert_id=f"b{i}",
                            obj_id="bogus", mjd=60450 + i, ra=180, dec=20,
                            mag=20, band="r", meta={})
                     for i in range(4)]
    cand = {"status": "accepted", "members": bogus_members, "rms_arcsec": 2.0}
    truth = [InjectionRecord(truth_id=0, a_au=45, e=0.1, i_deg=10,
                              apparent_mag=22, rate_arcsec_hr=1.0,
                              n_planted_alerts=6)]
    rep = evaluate_recovery([cand], truth)
    assert rep.n_false_positives == 1
    assert rep.n_recovered == 0
