from pathlib import Path


def test_labelled_inference_suite_covers_required_benchmark_sources():
    from ariadne.discovery.benchmarking import make_labelled_inference_suite

    cases = make_labelled_inference_suite()
    sources = {c.source for c in cases}
    assert "real_mpc_known_object_proxy" in sources
    assert "ztf_like_synthetic" in sources
    assert "lsst_like_synthetic" in sources
    assert "adversarial_false_positive" in sources
    assert "blind_holdout" in sources
    assert any(c.adversarial for c in cases)
    assert any(c.truth_label == "manual_review" for c in cases)


def test_inference_benchmark_computes_proof_metrics_and_ablations():
    from ariadne.discovery.benchmarking import (
        CHANNEL_FIELDS,
        make_labelled_inference_suite,
        run_inference_benchmark,
    )

    result = run_inference_benchmark(make_labelled_inference_suite())
    assert result.n >= 10
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_precision <= 1.0
    assert 0.0 <= result.macro_recall <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.reliability.n > 0
    assert 0.0 <= result.reliability.ece <= 1.0
    assert result.reliability.bins
    assert {row.channel for row in result.ablations} == set(CHANNEL_FIELDS)
    assert result.strata
    assert any(row.stratum == "artifact_adversarial" for row in result.strata)
    assert result.failures
    assert all(diag.recommendation for diag in result.failures)
    assert len(result.certificate_hash) == 64
    assert len(result.blind_hash) == 64
    assert result.source_counts["adversarial_false_positive"] >= 4


def test_inference_benchmark_fails_closed_on_adversarial_contradictions():
    from ariadne.discovery.benchmarking import (
        make_labelled_inference_suite,
        run_inference_benchmark,
    )

    result = run_inference_benchmark(make_labelled_inference_suite(), ablations=False)
    manual_rows = [r for r in result.rows if r.truth_label == "manual_review"]
    assert manual_rows
    assert all(r.predicted_label == "manual_review" for r in manual_rows)
    assert all(r.action == "manual_review" for r in manual_rows)


def test_inference_benchmark_certificate_is_deterministic():
    from ariadne.discovery.benchmarking import run_inference_benchmark

    first = run_inference_benchmark(ablations=False)
    second = run_inference_benchmark(ablations=False)
    assert first.certificate_hash == second.certificate_hash
    assert first.blind_hash == second.blind_hash


def test_write_benchmark_report_emits_audit_artifacts(tmp_path: Path):
    from ariadne.discovery.benchmarking import (
        run_inference_benchmark,
        write_benchmark_report,
    )

    result = run_inference_benchmark()
    paths = write_benchmark_report(result, tmp_path)
    expected = {
        "metrics",
        "drift_manifest",
        "reliability_curve",
        "reliability_diagram",
        "confusion",
        "precision_recall",
        "ablation",
        "case_results",
        "strata",
        "failure_diagnostics",
        "calibration_search",
        "holdout_manifest",
    }
    assert set(paths) == expected
    for path in paths.values():
        p = Path(path)
        assert p.exists()
        assert p.stat().st_size > 0


def test_channel_weight_search_returns_usable_calibration():
    from ariadne.discovery.benchmarking import (
        fit_label_biases,
        fit_channel_weights,
        make_labelled_inference_suite,
        run_inference_benchmark,
    )

    cases = make_labelled_inference_suite()
    cfg, rows = fit_channel_weights(cases, channels=("rate", "morphology"),
                                    grid=(0.75, 1.0, 1.25))
    cfg, label_rows = fit_label_biases(cases, base=cfg, grid=(0.0, 0.75))
    assert cfg.channel_weights
    assert rows
    assert label_rows
    result = run_inference_benchmark(cases, calibration=cfg, fit_calibration=False,
                                     fit_channels=False, ablations=False)
    assert result.n == len(cases)


def test_failure_diagnostics_explain_misses_and_low_margin_cases():
    from ariadne.discovery.benchmarking import (
        failure_diagnostics,
        make_labelled_inference_suite,
    )

    diagnostics = failure_diagnostics(make_labelled_inference_suite())
    assert diagnostics
    first = diagnostics[0]
    assert first.case_id
    assert first.stratum
    assert isinstance(first.top_hypotheses, list)
    assert first.missing_channels


def test_max_sparse_suite_reaches_safe_decision_closure():
    from ariadne.discovery.benchmarking import (
        make_labelled_inference_suite,
        run_inference_benchmark,
    )

    result = run_inference_benchmark(
        make_labelled_inference_suite(), fit_channels=True, fit_labels=True,
        ablations=False)
    assert result.accuracy == 1.0
    assert result.safe_accuracy == 1.0
    assert all(row.correct for row in result.rows)


def test_holdout_freeze_and_separate_calibration():
    from ariadne.discovery.benchmarking import (
        freeze_holdout,
        make_labelled_inference_suite,
        run_inference_benchmark,
    )

    cases = make_labelled_inference_suite()
    result = run_inference_benchmark(cases, separate_calibration=True,
                                     fit_channels=True, fit_labels=True,
                                     ablations=False)
    assert result.holdout_manifest.n == 2
    assert result.split_counts == {"blind": 2}
    assert len(result.holdout_manifest.holdout_hash) == 64
    path = Path("holdout_test.json")
    try:
        manifest = freeze_holdout([c for c in cases if c.split == "blind"], path)
        assert path.exists()
        assert manifest.n == 2
    finally:
        if path.exists():
            path.unlink()


def test_adversarial_mutation_suite_and_safe_accuracy():
    from ariadne.discovery.benchmarking import (
        adversarial_mutations,
        make_labelled_inference_suite,
        run_inference_benchmark,
    )

    cases = make_labelled_inference_suite()
    mutated = adversarial_mutations(cases)
    assert len(mutated) > len(cases)
    assert all("__" in c.case_id for c in mutated)
    result = run_inference_benchmark(cases, adversarial=True, ablations=False)
    assert 0.0 <= result.safe_accuracy <= 1.0
    assert any(row.source.endswith(":mutated") for row in result.rows)
    assert result.safe_accuracy >= 0.9


def test_weak_cosmic_ray_flag_does_not_erase_coherent_moving_arc():
    from ariadne.discovery.benchmarking import LabelledCase, run_inference_benchmark
    from ariadne.discovery.inference import Evidence

    case = LabelledCase(
        "weak_cosmic_flag_real_arc",
        Evidence(
            rate_arcsec_hr=18.5,
            apparent_mag=19.2,
            morphology_label="COSMIC_RAY",
            morphology_confidence=0.70,
            n_detections=12,
            arc_days=10.0,
            rms_arcsec=0.8,
            skybot_match_names=[],
            sky_context={"a_au": 2.6, "e": 0.1},
        ),
        truth_label="MBA",
        source="regression",
    )

    result = run_inference_benchmark([case], fit_calibration=False, ablations=False)

    assert result.accuracy == 1.0
    assert result.rows[0].predicted_label == "MBA"


def test_short_arc_point_motion_beats_generic_subtraction_residual():
    from ariadne.discovery.benchmarking import LabelledCase, run_inference_benchmark
    from ariadne.discovery.inference import Evidence

    case = LabelledCase(
        "short_arc_inner_belt_motion",
        Evidence(
            rate_arcsec_hr=19.0,
            apparent_mag=19.5,
            morphology_label="POINT",
            morphology_confidence=0.9,
            n_detections=2,
            arc_days=0.05,
            skybot_match_names=[],
            sky_context={"a_au": 2.0, "e": 0.08},
        ),
        truth_label="IMB",
        source="regression",
    )

    result = run_inference_benchmark([case], fit_calibration=False, ablations=False)

    assert result.accuracy == 1.0
    assert result.rows[0].predicted_label == "IMB"
