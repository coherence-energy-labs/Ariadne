"""Inference benchmark harness for discovery claims.

This module turns "the inference engine feels good" into repeatable evidence:
labelled survey-like cases, LSST/ZTF-style alert streams, adversarial false
positives, calibration curves, precision/recall, ablations, and blind hashes.
It is intentionally deterministic so benchmark certificates can be compared
across releases.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .inference import (
    CalibrationConfig,
    Evidence,
    InferenceResult,
    ReliabilityReport,
    fit_temperature,
    infer,
    reliability_report,
)


CHANNEL_FIELDS = {
    "rate": ("rate_arcsec_hr",),
    "magnitude": ("apparent_mag", "band", "band_magnitudes"),
    "morphology": ("morphology_label", "morphology_confidence"),
    "orbit_fit": ("orbit_state", "rms_arcsec", "n_detections", "arc_days"),
    "xmatch": ("skybot_match_names",),
    "color": ("band_magnitudes",),
    "sky_context": ("sky_context",),
}

ARTIFACT_LABELS = {
    "artefact", "artifact", "satellite_trail", "cosmic_ray",
    "stellar_variable", "supernova_or_agn", "subtraction_residual",
    "edge_artefact", "blend_two_stars", "ghost_or_diffraction",
    "manual_review",
}


@dataclass
class LabelledCase:
    """One labelled inference benchmark example."""
    case_id: str
    evidence: Evidence
    truth_label: str
    source: str
    split: str = "validation"
    adversarial: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ClassificationRow:
    """Per-case benchmark outcome."""
    case_id: str
    source: str
    split: str
    truth_label: str
    predicted_label: str
    confidence: float
    correct: bool
    action: str
    certificate_hash: str
    safe_correct: bool = False


@dataclass
class PrecisionRecallRow:
    label: str
    support: int
    predicted: int
    true_positive: int
    precision: float
    recall: float
    f1: float


@dataclass
class AblationRow:
    channel: str
    accuracy: float
    nll: float
    ece: float
    delta_accuracy: float
    delta_nll: float


@dataclass
class FailureDiagnosis:
    """Why a benchmark case failed or looked risky."""
    case_id: str
    stratum: str
    truth_label: str
    predicted_label: str
    confidence: float
    margin: float
    entropy: float
    action: str
    evidence_channels: list
    missing_channels: list
    top_hypotheses: list
    strongest_terms: dict
    warnings: list
    contradictions: list
    posterior_warnings: list
    recommendation: str


@dataclass
class StratumReport:
    """Leaderboard row for one benchmark difficulty stratum."""
    stratum: str
    n: int
    accuracy: float
    macro_f1: float
    mean_confidence: float
    mean_margin: float
    manual_review_rate: float
    safe_accuracy: float = 0.0


@dataclass
class CalibrationSearchRow:
    parameter: str
    target: str
    weight: float
    accuracy: float
    nll: float
    score: float


@dataclass
class HoldoutManifest:
    """Tamper-evident blind holdout identity without exposing labels in summaries."""
    n: int
    holdout_hash: str
    case_hash: str
    label_hash: str
    split_counts: dict


@dataclass
class BenchmarkResult:
    """Full proof bundle for an inference benchmark run."""
    n: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion: dict
    precision_recall: list[PrecisionRecallRow]
    reliability: ReliabilityReport
    ablations: list[AblationRow]
    rows: list[ClassificationRow]
    calibration: CalibrationConfig
    blind_hash: str
    certificate_hash: str
    source_counts: dict
    split_counts: dict
    safe_accuracy: float = 0.0
    holdout_manifest: HoldoutManifest | None = None
    strata: list[StratumReport] = field(default_factory=list)
    failures: list[FailureDiagnosis] = field(default_factory=list)
    calibration_search: list[CalibrationSearchRow] = field(default_factory=list)


def _canonical_json(value) -> str:
    def scrub(x):
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        if hasattr(x, "__dataclass_fields__"):
            return scrub(asdict(x))
        if isinstance(x, dict):
            return {str(k): scrub(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [scrub(v) for v in x]
        return x

    return json.dumps(scrub(value), sort_keys=True, separators=(",", ":"))


def _sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def make_labelled_inference_suite(seed: int = 20260531) -> list[LabelledCase]:
    """Build a deterministic labelled benchmark suite.

    The default suite is offline and CI-friendly. The "real" cases are
    known-object proxies with MPC-style labels and survey-like measurements;
    projects with licensed or downloaded alert corpora can append actual rows
    using the same LabelledCase schema.
    """
    base_mjd = 60450.0 + (seed % 17) * 0.01
    return [
        LabelledCase(
            "mpc_proxy_sedna_detached",
            Evidence(mjd=base_mjd, rate_arcsec_hr=0.18, apparent_mag=20.8,
                     band="r", morphology_label="POINT", morphology_confidence=0.93,
                     n_detections=8, arc_days=21.0, rms_arcsec=0.8,
                     skybot_match_names=[]),
            "DETACHED",
            "real_mpc_known_object_proxy",
            metadata={"known_object": "90377 Sedna", "survey_style": "MPC"},
        ),
        LabelledCase(
            "mpc_proxy_classical_kbo",
            Evidence(mjd=base_mjd, rate_arcsec_hr=1.15, apparent_mag=22.6,
                     band="r", morphology_label="POINT", morphology_confidence=0.91,
                     n_detections=7, arc_days=12.0, rms_arcsec=0.9,
                     skybot_match_names=[]),
            "CLASSICAL_KBO",
            "real_mpc_known_object_proxy",
            metadata={"known_object": "cold classical KBO proxy"},
        ),
        LabelledCase(
            "mpc_proxy_centaur",
            Evidence(mjd=base_mjd, rate_arcsec_hr=3.2, apparent_mag=20.7,
                     band="r", morphology_label="POINT", morphology_confidence=0.88,
                     n_detections=6, arc_days=8.0, rms_arcsec=1.1,
                     skybot_match_names=[]),
            "CENTAUR",
            "real_mpc_known_object_proxy",
        ),
        LabelledCase(
            "ztf_like_main_belt",
            Evidence(mjd=base_mjd, rate_arcsec_hr=16.0, apparent_mag=19.7,
                     band="r", morphology_label="POINT", morphology_confidence=0.95,
                     n_detections=5, arc_days=1.8, rms_arcsec=0.7,
                     skybot_match_names=[],
                     sky_context={"survey": "ZTF", "cadence_minutes": 30}),
            "MBA",
            "ztf_like_synthetic",
        ),
        LabelledCase(
            "ztf_like_fast_neo",
            Evidence(mjd=base_mjd, rate_arcsec_hr=190.0, apparent_mag=18.4,
                     band="r", morphology_label="POINT", morphology_confidence=0.90,
                     n_detections=5, arc_days=0.7, rms_arcsec=1.3,
                     skybot_match_names=[],
                     sky_context={"survey": "ZTF", "cadence_minutes": 15}),
            "APOLLO",
            "ztf_like_synthetic",
        ),
        LabelledCase(
            "lsst_like_faint_kbo",
            Evidence(mjd=base_mjd, rate_arcsec_hr=0.9, apparent_mag=24.1,
                     band="r", band_magnitudes={"g": 24.8, "r": 24.1, "i": 23.8},
                     morphology_label="POINT", morphology_confidence=0.82,
                     n_detections=6, arc_days=14.0, rms_arcsec=1.6,
                     skybot_match_names=[],
                     sky_context={"survey": "LSST", "visit_pair_minutes": 33}),
            "CLASSICAL_KBO",
            "lsst_like_synthetic",
        ),
        LabelledCase(
            "lsst_like_trojan",
            Evidence(mjd=base_mjd, rate_arcsec_hr=4.8, apparent_mag=21.3,
                     band="i", band_magnitudes={"g": 22.2, "r": 21.6, "i": 21.3},
                     morphology_label="POINT", morphology_confidence=0.89,
                     n_detections=6, arc_days=5.0, rms_arcsec=0.9,
                     skybot_match_names=[],
                     sky_context={"survey": "LSST"}),
            "JTROJAN",
            "lsst_like_synthetic",
        ),
        LabelledCase(
            "false_positive_satellite_streak",
            Evidence(mjd=base_mjd, rate_arcsec_hr=5200.0, apparent_mag=17.8,
                     morphology_label="STREAK", morphology_confidence=0.96,
                     n_detections=1, arc_days=0.0),
            "satellite_trail",
            "adversarial_false_positive",
            adversarial=True,
        ),
        LabelledCase(
            "false_positive_cosmic_ray",
            Evidence(mjd=base_mjd, apparent_mag=18.0,
                     morphology_label="COSMIC_RAY", morphology_confidence=0.98,
                     n_detections=1, arc_days=0.0),
            "cosmic_ray",
            "adversarial_false_positive",
            adversarial=True,
        ),
        LabelledCase(
            "false_positive_variable_star",
            Evidence(mjd=base_mjd, apparent_mag=19.1,
                     morphology_label="POINT", morphology_confidence=0.78,
                     n_detections=3, arc_days=5.0,
                     sky_context={"near_known_star": True, "stationary": True}),
            "stellar_variable",
            "adversarial_false_positive",
            adversarial=True,
        ),
        LabelledCase(
            "contradiction_cosmic_ray_arc",
            Evidence(mjd=base_mjd, rate_arcsec_hr=1.1, apparent_mag=22.0,
                     morphology_label="COSMIC_RAY", morphology_confidence=0.97,
                     n_detections=5, arc_days=7.0, rms_arcsec=0.8),
            "manual_review",
            "adversarial_false_positive",
            adversarial=True,
        ),
        LabelledCase(
            "contradiction_satellite_multiday",
            Evidence(mjd=base_mjd, rate_arcsec_hr=3200.0, apparent_mag=18.2,
                     morphology_label="STREAK", morphology_confidence=0.93,
                     n_detections=8, arc_days=6.0, rms_arcsec=1.2),
            "manual_review",
            "adversarial_false_positive",
            adversarial=True,
        ),
        LabelledCase(
            "blind_holdout_slow_clean",
            Evidence(mjd=base_mjd, rate_arcsec_hr=0.65, apparent_mag=23.2,
                     morphology_label="POINT", morphology_confidence=0.86,
                     n_detections=5, arc_days=9.5, rms_arcsec=1.0,
                     skybot_match_names=[]),
            "RESONANT_KBO",
            "blind_holdout",
            split="blind",
        ),
        LabelledCase(
            "blind_holdout_streak",
            Evidence(mjd=base_mjd, rate_arcsec_hr=4200.0, apparent_mag=16.8,
                     morphology_label="STREAK", morphology_confidence=0.94,
                     n_detections=1),
            "satellite_trail",
            "blind_holdout",
            split="blind",
            adversarial=True,
        ),
    ]


def _prediction(result: InferenceResult) -> tuple[str, float]:
    action = result.recommended_followup.get("action") if result.recommended_followup else None
    if result.best is None or action == "manual_review":
        return "manual_review", 0.0
    label = result.best.orbital_class or result.best.label or result.best.class_
    return label, float(result.best.posterior)


def _case_matches(predicted: str, truth: str) -> bool:
    if predicted == truth:
        return True
    tno_family = {
        "CLASSICAL_KBO", "HOT_CLASSICAL", "RESONANT_KBO",
        "SCATTERED_KBO", "DETACHED", "SEDNOID",
    }
    if predicted in tno_family and truth in tno_family:
        return True
    if predicted in {"AMOR", "APOLLO", "ATEN", "ATIRA"} and truth in {"NEO", "AMOR", "APOLLO", "ATEN", "ATIRA"}:
        return True
    return False


def _safe_decision_matches(predicted: str, truth: str, action: str) -> bool:
    """Safe decision = correct class/family OR correct abstention/follow-up."""
    if _case_matches(predicted, truth):
        return True
    if truth in ARTIFACT_LABELS and action == "discard" and predicted in ARTIFACT_LABELS:
        return True
    if truth == "manual_review":
        return action == "manual_review" or predicted == "manual_review"
    if action in {"manual_review", "observe_second_night", "observe_multiband"}:
        return True
    return False


def case_stratum(case: LabelledCase) -> str:
    """Assign a benchmark difficulty/meaning stratum."""
    ev = case.evidence
    if {"a_au", "e"}.issubset(ev.sky_context) or ev.orbit_state is not None:
        return "known_orbit_recovery"
    if case.truth_label == "manual_review":
        return "contradiction_fail_closed"
    if case.adversarial or case.truth_label in ARTIFACT_LABELS:
        return "artifact_adversarial"
    if ev.n_detections <= 1:
        return "single_detection"
    if ev.n_detections == 2:
        return "two_detection_alert"
    if ev.arc_days >= 1.0 and ev.n_detections >= 3:
        return "multi_night_tracklet"
    return "sparse_alert"


def _margin(result: InferenceResult) -> float:
    if len(result.hypotheses) < 2:
        return result.best.posterior if result.best else 0.0
    return max(0.0, result.hypotheses[0].posterior - result.hypotheses[1].posterior)


def _present_channels(ev: Evidence) -> list:
    present = []
    for channel, fields in CHANNEL_FIELDS.items():
        for field_name in fields:
            value = getattr(ev, field_name)
            if value not in (None, {}, [], ""):
                present.append(channel)
                break
    return sorted(set(present))


def diagnose_case(case: LabelledCase,
                  *,
                  calibration: CalibrationConfig | None = None) -> FailureDiagnosis:
    """Run one case and return a high-signal failure/risk diagnosis."""
    calibration = calibration or CalibrationConfig()
    result = infer(
        case.evidence,
        calibration=calibration,
        fail_closed_on_contradiction=True,
    )
    predicted, confidence = _prediction(result)
    top = [
        {
            "label": h.label,
            "orbital_class": h.orbital_class,
            "class": h.class_,
            "posterior": h.posterior,
            "free_energy": h.free_energy,
        }
        for h in result.hypotheses[:5]
    ]
    strongest = {}
    if result.best is not None:
        terms = result.best.evidence_terms or {}
        strongest = dict(sorted(
            terms.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:8])
    present = _present_channels(case.evidence)
    warnings = list(result.evidence_audit.warnings) if result.evidence_audit else []
    contradictions = (
        list(result.evidence_audit.contradictions) if result.evidence_audit else [])
    posterior_warnings = (
        list(result.posterior_check.warnings) if result.posterior_check else [])
    action = result.recommended_followup.get("action", "")
    return FailureDiagnosis(
        case_id=case.case_id,
        stratum=case_stratum(case),
        truth_label=case.truth_label,
        predicted_label=predicted,
        confidence=confidence,
        margin=_margin(result),
        entropy=result.entropy,
        action=action,
        evidence_channels=present,
        missing_channels=sorted(set(CHANNEL_FIELDS) - set(present)),
        top_hypotheses=top,
        strongest_terms=strongest,
        warnings=warnings,
        contradictions=contradictions,
        posterior_warnings=posterior_warnings,
        recommendation=_diagnostic_recommendation(case, result, predicted),
    )


def _diagnostic_recommendation(case: LabelledCase,
                               result: InferenceResult,
                               predicted: str) -> str:
    if case.truth_label == "manual_review" and predicted != "manual_review":
        return "strengthen contradiction audit and fail-closed triggers"
    if result.best is None:
        return "manual review path behaved correctly"
    if result.entropy > 1.5 or _margin(result) < 0.10:
        return "collect decisive follow-up; posterior margin is too small"
    missing = set(CHANNEL_FIELDS) - set(_present_channels(case.evidence))
    if "rate" in missing:
        return "add same-night pair or tracklet rate before classifying"
    if "morphology" in missing and case.truth_label in ARTIFACT_LABELS:
        return "add morphology/cutout evidence for false-positive rejection"
    if "orbit_fit" in missing and case.evidence.n_detections >= 3:
        return "attempt preliminary orbit fit or admissible-region constraint"
    return "inspect top evidence terms; likely prior/likelihood calibration miss"


def failure_diagnostics(cases: list[LabelledCase],
                        *,
                        calibration: CalibrationConfig | None = None,
                        include_correct_low_margin: bool = True,
                        margin_threshold: float = 0.08
                        ) -> list[FailureDiagnosis]:
    """Return misses plus risky correct cases."""
    out = []
    for case in cases:
        diag = diagnose_case(case, calibration=calibration)
        ok = _case_matches(diag.predicted_label, diag.truth_label)
        if not ok or (include_correct_low_margin and diag.margin < margin_threshold):
            out.append(diag)
    return out


def _confusion(rows: list[ClassificationRow]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        out.setdefault(row.truth_label, {})
        out[row.truth_label][row.predicted_label] = out[row.truth_label].get(row.predicted_label, 0) + 1
    return out


def _precision_recall(rows: list[ClassificationRow]) -> list[PrecisionRecallRow]:
    labels = sorted({r.truth_label for r in rows} | {r.predicted_label for r in rows})
    out = []
    for label in labels:
        tp = sum(1 for r in rows if r.truth_label == label and r.predicted_label == label)
        support = sum(1 for r in rows if r.truth_label == label)
        predicted = sum(1 for r in rows if r.predicted_label == label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(PrecisionRecallRow(label, support, predicted, tp, precision, recall, f1))
    return out


def _erase_channel(evidence: Evidence, channel: str) -> Evidence:
    updates = {}
    for field_name in CHANNEL_FIELDS[channel]:
        if field_name in {"band_magnitudes", "sky_context"}:
            updates[field_name] = {}
        elif field_name == "n_detections":
            updates[field_name] = 1
        elif field_name == "arc_days":
            updates[field_name] = 0.0
        else:
            updates[field_name] = None
    return replace(evidence, **updates)


def _reliability_cases(cases: list[LabelledCase]) -> list[tuple[Evidence, str]]:
    return [
        (case.evidence, case.truth_label)
        for case in cases
        if case.truth_label != "manual_review"
    ]


def make_holdout_manifest(cases: list[LabelledCase]) -> HoldoutManifest:
    """Build hashes for frozen/blind holdout bookkeeping."""
    payload_cases = [
        {"case_id": c.case_id, "source": c.source, "split": c.split}
        for c in cases
    ]
    payload_labels = [
        {"case_id": c.case_id, "truth_label": c.truth_label}
        for c in cases
    ]
    split_counts = {}
    for case in cases:
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
    return HoldoutManifest(
        n=len(cases),
        holdout_hash=_sha256({"cases": payload_cases, "labels": payload_labels}),
        case_hash=_sha256(payload_cases),
        label_hash=_sha256(payload_labels),
        split_counts=split_counts,
    )


def freeze_holdout(cases: list[LabelledCase], path: str | Path) -> HoldoutManifest:
    """Write a tamper-evident holdout manifest to disk."""
    manifest = make_holdout_manifest(cases)
    Path(path).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def split_train_eval(cases: list[LabelledCase],
                     *,
                     train_splits: tuple[str, ...] = ("train", "validation"),
                     eval_splits: tuple[str, ...] = ("blind",)
                     ) -> tuple[list[LabelledCase], list[LabelledCase]]:
    train = [c for c in cases if c.split in train_splits]
    eval_cases = [c for c in cases if c.split in eval_splits]
    if not eval_cases:
        eval_cases = list(cases)
    if not train:
        train = [c for c in cases if c not in eval_cases] or list(cases)
    return train, eval_cases


def _run_rows(cases: list[LabelledCase],
              calibration: CalibrationConfig,
              *,
              fail_closed_on_contradiction: bool) -> list[ClassificationRow]:
    rows = []
    for case in cases:
        result = infer(
            case.evidence,
            calibration=calibration,
            fail_closed_on_contradiction=fail_closed_on_contradiction,
        )
        predicted, confidence = _prediction(result)
        action = result.recommended_followup.get("action", "")
        rows.append(ClassificationRow(
            case_id=case.case_id,
            source=case.source,
            split=case.split,
            truth_label=case.truth_label,
            predicted_label=predicted,
            confidence=confidence,
            correct=_case_matches(predicted, case.truth_label),
            action=action,
            certificate_hash=result.certificate.get("payload_hash", ""),
            safe_correct=_safe_decision_matches(predicted, case.truth_label, action),
        ))
    return rows


def stratified_leaderboard(cases: list[LabelledCase],
                           rows: list[ClassificationRow],
                           *,
                           calibration: CalibrationConfig | None = None
                           ) -> list[StratumReport]:
    """Compute per-difficulty leaderboard rows."""
    calibration = calibration or CalibrationConfig()
    case_by_id = {c.case_id: c for c in cases}
    by_stratum: dict[str, list[ClassificationRow]] = {}
    for row in rows:
        stratum = case_stratum(case_by_id[row.case_id])
        by_stratum.setdefault(stratum, []).append(row)
    reports = []
    for stratum, srows in sorted(by_stratum.items()):
        n = len(srows)
        pr = _precision_recall(srows)
        supported = [r for r in pr if r.support > 0]
        macro_f1 = sum(r.f1 for r in supported) / len(supported) if supported else 0.0
        margins = []
        for row in srows:
            diag = diagnose_case(case_by_id[row.case_id], calibration=calibration)
            margins.append(diag.margin)
        reports.append(StratumReport(
            stratum=stratum,
            n=n,
            accuracy=sum(1 for r in srows if r.correct) / n if n else 0.0,
            macro_f1=macro_f1,
            mean_confidence=sum(r.confidence for r in srows) / n if n else 0.0,
            mean_margin=sum(margins) / len(margins) if margins else 0.0,
            manual_review_rate=sum(1 for r in srows if r.action == "manual_review") / n
            if n else 0.0,
            safe_accuracy=sum(1 for r in srows if r.safe_correct) / n if n else 0.0,
        ))
    return reports


def fit_channel_weights(cases: list[LabelledCase],
                        *,
                        base: CalibrationConfig | None = None,
                        channels: tuple[str, ...] = (
                            "rate", "magnitude", "morphology",
                            "orbit_fit", "artifact_context"),
                        grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
                        ) -> tuple[CalibrationConfig, list[CalibrationSearchRow]]:
    """Coordinate-search channel weights against labelled validation cases."""
    cfg = base or CalibrationConfig()
    weights = dict(cfg.channel_weights)
    search_rows: list[CalibrationSearchRow] = []

    def evaluate(candidate_weights: dict) -> tuple[float, float, float]:
        candidate = CalibrationConfig(
            temperature=cfg.temperature,
            label_bias=dict(cfg.label_bias),
            channel_weights=dict(candidate_weights),
            version="channel-weight-search",
        )
        rows = _run_rows(cases, candidate, fail_closed_on_contradiction=True)
        accuracy = sum(1 for r in rows if r.correct) / len(rows) if rows else 0.0
        rel = reliability_report(_reliability_cases(cases), calibration=candidate)
        score = accuracy - 0.03 * (rel.nll if math.isfinite(rel.nll) else 100.0)
        return score, accuracy, rel.nll

    for channel in channels:
        best_score = -float("inf")
        best_weight = weights.get(channel, 1.0)
        for weight in grid:
            trial = dict(weights)
            trial[channel] = weight
            score, accuracy, nll = evaluate(trial)
            search_rows.append(CalibrationSearchRow(
                parameter="channel_weight", target=channel,
                weight=weight, accuracy=accuracy,
                nll=nll, score=score))
            if score > best_score:
                best_score = score
                best_weight = weight
        weights[channel] = best_weight

    tuned = CalibrationConfig(
        temperature=cfg.temperature,
        label_bias=dict(cfg.label_bias),
        channel_weights=weights,
        version="channel-weight-search-v1",
    )
    return tuned, search_rows


def fit_label_biases(cases: list[LabelledCase],
                     *,
                     base: CalibrationConfig | None = None,
                     grid: tuple[float, ...] = (
                         -3.0, -2.25, -1.5, -0.75, 0.0,
                         0.75, 1.5, 2.25, 3.0, 4.0)
                     ) -> tuple[CalibrationConfig, list[CalibrationSearchRow]]:
    """Coordinate-search class prior corrections from labelled cases."""
    cfg = base or CalibrationConfig()
    biases = dict(cfg.label_bias)
    search_rows: list[CalibrationSearchRow] = []
    labels = sorted({
        c.truth_label for c in cases
        if c.truth_label not in ARTIFACT_LABELS
    })

    def evaluate(candidate_biases: dict) -> tuple[float, float, float]:
        candidate = CalibrationConfig(
            temperature=cfg.temperature,
            label_bias=dict(candidate_biases),
            channel_weights=dict(cfg.channel_weights),
            version="label-bias-search",
        )
        rows = _run_rows(cases, candidate, fail_closed_on_contradiction=True)
        accuracy = sum(1 for r in rows if r.correct) / len(rows) if rows else 0.0
        rel = reliability_report(_reliability_cases(cases), calibration=candidate)
        score = accuracy - 0.03 * (rel.nll if math.isfinite(rel.nll) else 100.0)
        return score, accuracy, rel.nll

    for label in labels:
        best_score = -float("inf")
        best_bias = biases.get(label, 0.0)
        for bias in grid:
            trial = dict(biases)
            trial[label] = bias
            score, accuracy, nll = evaluate(trial)
            search_rows.append(CalibrationSearchRow(
                parameter="label_bias", target=label, weight=bias,
                accuracy=accuracy, nll=nll, score=score))
            if score > best_score:
                best_score = score
                best_bias = bias
        biases[label] = best_bias

    tuned = CalibrationConfig(
        temperature=cfg.temperature,
        label_bias=biases,
        channel_weights=dict(cfg.channel_weights),
        version="label-bias-search-v1",
    )
    return tuned, search_rows


def adversarial_mutations(cases: list[LabelledCase],
                          *,
                          include_original: bool = False) -> list[LabelledCase]:
    """Deterministically stress cases with missing/noisy/conflicting evidence."""
    out = list(cases) if include_original else []
    for case in cases:
        ev = case.evidence
        out.append(replace(
            case,
            case_id=f"{case.case_id}__missing_color",
            evidence=replace(ev, band_magnitudes={}, band=None),
            source=f"{case.source}:mutated",
            adversarial=True,
            metadata={**case.metadata, "mutation": "missing_color"},
        ))
        out.append(replace(
            case,
            case_id=f"{case.case_id}__mag_noise",
            evidence=replace(
                ev,
                apparent_mag=(ev.apparent_mag + 0.7)
                if ev.apparent_mag is not None else None),
            source=f"{case.source}:mutated",
            adversarial=True,
            metadata={**case.metadata, "mutation": "mag_noise"},
        ))
        out.append(replace(
            case,
            case_id=f"{case.case_id}__weak_morphology",
            evidence=replace(ev, morphology_confidence=0.35),
            source=f"{case.source}:mutated",
            adversarial=True,
            metadata={**case.metadata, "mutation": "weak_morphology"},
        ))
        if ev.n_detections >= 3:
            out.append(replace(
                case,
                case_id=f"{case.case_id}__short_arc",
                evidence=replace(ev, n_detections=2, arc_days=0.05, rms_arcsec=None),
                source=f"{case.source}:mutated",
                adversarial=True,
                metadata={**case.metadata, "mutation": "short_arc"},
            ))
        if case.truth_label not in ARTIFACT_LABELS:
            out.append(replace(
                case,
                case_id=f"{case.case_id}__artifact_conflict",
                evidence=replace(
                    ev, morphology_label="COSMIC_RAY",
                    morphology_confidence=0.70),
                source=f"{case.source}:mutated",
                adversarial=True,
                metadata={**case.metadata, "mutation": "artifact_conflict"},
            ))
    return out


def run_inference_benchmark(cases: list[LabelledCase] | None = None,
                            *,
                            calibration: CalibrationConfig | None = None,
                            fit_calibration: bool = True,
                            fit_channels: bool = False,
                            fit_labels: bool = False,
                            separate_calibration: bool = False,
                            adversarial: bool = False,
                            include_blind: bool = True,
                            ablations: bool = True,
                            n_bins: int = 10) -> BenchmarkResult:
    """Run the inference engine against labelled, adversarial, and blind cases."""
    if cases is None:
        cases = make_labelled_inference_suite()
    if adversarial:
        cases = adversarial_mutations(cases, include_original=True)
    eval_cases = list(cases) if include_blind else [c for c in cases if c.split != "blind"]
    train_cases, score_cases = split_train_eval(eval_cases) if separate_calibration else (eval_cases, eval_cases)

    if calibration is None:
        if fit_calibration:
            calibration, _ = fit_temperature(_reliability_cases(train_cases))
        else:
            calibration = CalibrationConfig()
    calibration_search = []
    if fit_channels:
        calibration, calibration_search = fit_channel_weights(train_cases, base=calibration)
    if fit_labels:
        calibration, label_search = fit_label_biases(train_cases, base=calibration)
        calibration_search.extend(label_search)

    rows = _run_rows(score_cases, calibration, fail_closed_on_contradiction=True)
    n = len(rows)
    accuracy = sum(1 for r in rows if r.correct) / n if n else 0.0
    pr = _precision_recall(rows)
    labels_with_support = [r for r in pr if r.support > 0]
    macro_precision = sum(r.precision for r in labels_with_support) / len(labels_with_support) if labels_with_support else 0.0
    macro_recall = sum(r.recall for r in labels_with_support) / len(labels_with_support) if labels_with_support else 0.0
    macro_f1 = sum(r.f1 for r in labels_with_support) / len(labels_with_support) if labels_with_support else 0.0
    reliability = reliability_report(
        _reliability_cases(score_cases), calibration=calibration, n_bins=n_bins)

    ablation_rows = []
    if ablations:
        baseline_nll = reliability.nll
        for channel in CHANNEL_FIELDS:
            ablated_cases = [
                replace(case, evidence=_erase_channel(case.evidence, channel))
                for case in score_cases
            ]
            ablated_rows = _run_rows(
                ablated_cases, calibration, fail_closed_on_contradiction=True)
            ablated_acc = (
                sum(1 for r in ablated_rows if r.correct) / len(ablated_rows)
                if ablated_rows else 0.0
            )
            ablated_rel = reliability_report(
                _reliability_cases(ablated_cases), calibration=calibration, n_bins=n_bins)
            ablation_rows.append(AblationRow(
                channel=channel,
                accuracy=ablated_acc,
                nll=ablated_rel.nll,
                ece=ablated_rel.ece,
                delta_accuracy=ablated_acc - accuracy,
                delta_nll=ablated_rel.nll - baseline_nll,
            ))

    blind_payload = [
        {"case_id": c.case_id, "truth_label": c.truth_label, "source": c.source}
        for c in score_cases if c.split == "blind"
    ]
    source_counts = {}
    split_counts = {}
    for case in score_cases:
        source_counts[case.source] = source_counts.get(case.source, 0) + 1
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
    safe_accuracy = sum(1 for r in rows if r.safe_correct) / n if n else 0.0
    holdout_manifest = make_holdout_manifest(score_cases)
    strata = stratified_leaderboard(score_cases, rows, calibration=calibration)
    failures = failure_diagnostics(score_cases, calibration=calibration)

    payload_for_certificate = {
        "schema": "ariadne.discovery.inference_benchmark.v1",
        "n": n,
        "accuracy": accuracy,
        "safe_accuracy": safe_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion": _confusion(rows),
        "rows": rows,
        "reliability": reliability,
        "ablations": ablation_rows,
        "calibration": calibration,
        "strata": strata,
        "failures": failures,
        "calibration_search": calibration_search,
        "holdout_manifest": holdout_manifest,
        "source_counts": source_counts,
        "split_counts": split_counts,
    }
    return BenchmarkResult(
        n=n,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        confusion=_confusion(rows),
        precision_recall=pr,
        reliability=reliability,
        ablations=ablation_rows,
        rows=rows,
        calibration=calibration,
        blind_hash=_sha256(blind_payload),
        certificate_hash=_sha256(payload_for_certificate),
        source_counts=source_counts,
        split_counts=split_counts,
        safe_accuracy=safe_accuracy,
        holdout_manifest=holdout_manifest,
        strata=strata,
        failures=failures,
        calibration_search=calibration_search,
    )


def write_benchmark_report(result: BenchmarkResult, outdir: str | Path) -> dict:
    """Write metrics JSON and CSV artifacts for audit/review."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "metrics.json"
    metrics_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")

    manifest_path = out / "drift_manifest.json"
    manifest = {
        "schema": "ariadne.discovery.benchmark_drift_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n": result.n,
        "accuracy": result.accuracy,
        "safe_accuracy": result.safe_accuracy,
        "macro_f1": result.macro_f1,
        "ece": result.reliability.ece,
        "certificate_hash": result.certificate_hash,
        "holdout_manifest": result.holdout_manifest,
        "calibration": result.calibration,
    }
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")

    reliability_path = out / "reliability_curve.csv"
    with reliability_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lo", "hi", "n", "avg_confidence", "accuracy"])
        w.writeheader()
        w.writerows(result.reliability.bins)

    confusion_path = out / "confusion.csv"
    labels = sorted(set(result.confusion) | {p for row in result.confusion.values() for p in row})
    with confusion_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["truth_label", *labels])
        for truth in labels:
            w.writerow([truth, *[result.confusion.get(truth, {}).get(pred, 0) for pred in labels]])

    pr_path = out / "precision_recall.csv"
    with pr_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "support", "predicted", "true_positive",
            "precision", "recall", "f1",
        ])
        w.writeheader()
        for row in result.precision_recall:
            w.writerow(asdict(row))

    ablation_path = out / "ablation.csv"
    with ablation_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "channel", "accuracy", "nll", "ece",
            "delta_accuracy", "delta_nll",
        ])
        w.writeheader()
        for row in result.ablations:
            w.writerow(asdict(row))

    rows_path = out / "case_results.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id", "source", "split", "truth_label", "predicted_label",
            "confidence", "correct", "safe_correct", "action", "certificate_hash",
        ])
        w.writeheader()
        for row in result.rows:
            w.writerow(asdict(row))

    strata_path = out / "strata.csv"
    with strata_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stratum", "n", "accuracy", "macro_f1", "mean_confidence",
            "mean_margin", "manual_review_rate", "safe_accuracy",
        ])
        w.writeheader()
        for row in result.strata:
            w.writerow(asdict(row))

    failures_path = out / "failure_diagnostics.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id", "stratum", "truth_label", "predicted_label",
            "confidence", "margin", "entropy", "action", "evidence_channels",
            "missing_channels", "top_hypotheses", "strongest_terms", "warnings",
            "contradictions", "posterior_warnings", "recommendation",
        ])
        w.writeheader()
        for row in result.failures:
            d = asdict(row)
            for key in ("evidence_channels", "missing_channels", "top_hypotheses",
                        "strongest_terms", "warnings", "contradictions",
                        "posterior_warnings"):
                d[key] = _canonical_json(d[key])
            w.writerow(d)

    calibration_path = out / "calibration_search.csv"
    with calibration_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "parameter", "target", "weight", "accuracy", "nll", "score",
        ])
        w.writeheader()
        for row in result.calibration_search:
            w.writerow(asdict(row))

    holdout_path = out / "holdout_manifest.json"
    holdout_path.write_text(
        _canonical_json(result.holdout_manifest) + "\n", encoding="utf-8")

    reliability_plot_path = out / "reliability_diagram.png"
    _write_reliability_plot(result, reliability_plot_path)

    return {
        "metrics": str(metrics_path),
        "drift_manifest": str(manifest_path),
        "reliability_curve": str(reliability_path),
        "reliability_diagram": str(reliability_plot_path),
        "confusion": str(confusion_path),
        "precision_recall": str(pr_path),
        "ablation": str(ablation_path),
        "case_results": str(rows_path),
        "strata": str(strata_path),
        "failure_diagnostics": str(failures_path),
        "calibration_search": str(calibration_path),
        "holdout_manifest": str(holdout_path),
    }


def _write_reliability_plot(result: BenchmarkResult, path: Path):
    """Write a PNG reliability diagram when matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        path.with_suffix(".txt").write_text(
            "matplotlib unavailable; reliability_curve.csv contains plot data\n",
            encoding="utf-8")
        return
    try:
        xs = [b["avg_confidence"] for b in result.reliability.bins]
        ys = [b["accuracy"] for b in result.reliability.bins]
        ns = [b["n"] for b in result.reliability.bins]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        if xs:
            sizes = [max(30, 20 + 12 * n) for n in ns]
            ax.scatter(xs, ys, s=sizes, color="#1f77b4", alpha=0.85)
            ax.plot(xs, ys, color="#1f77b4", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Reliability diagram (ECE={result.reliability.ece:.3g})")
        ax.grid(True, alpha=0.25)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        path.with_suffix(".txt").write_text(
            f"matplotlib plot failed ({type(e).__name__}); "
            "reliability_curve.csv contains plot data\n",
            encoding="utf-8")
