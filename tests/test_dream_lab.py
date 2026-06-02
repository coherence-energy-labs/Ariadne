import copy

from ariadne.proof.dream import build_dream_run, write_dream_run


def _closure():
    return {
        "certificate_hash": "closurehash",
        "residuals": [
            {
                "residual_id": "route_multifidelity_promotion_ladder",
                "subsystem": "trajectory_certification",
                "category": "flight_grade_proof",
                "severity": 0.95,
                "confidence": 0.95,
                "description": "route ladder",
                "recommended_action": "promote routes",
            },
            {
                "residual_id": "reviewer_grade_visual_contract",
                "subsystem": "visuals",
                "category": "explainability",
                "severity": 0.75,
                "confidence": 1.0,
                "description": "visual contract",
            },
        ],
    }


def test_dream_run_builds_ranked_experiment_queue_and_hash_validates(tmp_path):
    run = build_dream_run(_closure())
    outputs = write_dream_run(run, tmp_path)

    assert run.status == "pass"
    assert run.experiment_count == 2
    assert run.experiments[0].experiment_id == "dream_route_full_strict_promotion"
    assert run.validate_hash()
    assert "dream_run.json" in outputs["json"]


def test_dream_run_hash_detects_tamper():
    run = build_dream_run(_closure())
    tampered = copy.deepcopy(run)
    object.__setattr__(tampered, "status", "tampered")

    assert run.validate_hash()
    assert not tampered.validate_hash()
