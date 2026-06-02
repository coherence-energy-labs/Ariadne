#!/usr/bin/env python
"""Paired inference-engine improvement benchmark.

This compares the same engine on the same cases under two configurations:

* baseline: default, uncalibrated inference settings
* tuned: calibration, channel weights, and label biases learned from a
  deterministic train split, then scored on a held-out eval split

The report intentionally excludes closure gates and proof-ledger status. It is
only about model/engine scoring behavior: accuracy, safe accuracy, macro
precision/recall/F1, NLL, Brier, ECE, and failure counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ariadne.discovery.benchmarking import (
    LabelledCase,
    adversarial_mutations,
    run_inference_benchmark,
)
from ariadne.discovery.external_corpora import read_labelled_cases_jsonl
from ariadne.discovery.inference import CalibrationConfig


def _case_bucket(case: LabelledCase, *, salt: str = "ariadne-engine-split-v1") -> int:
    payload = f"{salt}:{case.case_id}:{case.truth_label}:{case.source}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100


def deterministic_train_eval(
    cases: list[LabelledCase],
    *,
    train_percent: int = 70,
    salt: str = "ariadne-engine-split-v1",
) -> tuple[list[LabelledCase], list[LabelledCase]]:
    if not 10 <= train_percent <= 90:
        raise ValueError("train_percent must be between 10 and 90")
    train = [case for case in cases if _case_bucket(case, salt=salt) < train_percent]
    eval_cases = [case for case in cases if _case_bucket(case, salt=salt) >= train_percent]
    if not train or not eval_cases:
        raise ValueError("deterministic split produced an empty train or eval set")
    return train, eval_cases


def _summary(result) -> dict:
    return {
        "n": result.n,
        "accuracy": result.accuracy,
        "safe_accuracy": result.safe_accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "nll": result.reliability.nll,
        "brier": result.reliability.brier,
        "ece": result.reliability.ece,
        "failures": len(result.failures),
        "certificate_hash": result.certificate_hash,
        "calibration": asdict(result.calibration),
    }


def _delta(before: dict, after: dict) -> dict:
    out = {}
    for key in (
        "accuracy", "safe_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "nll", "brier", "ece", "failures",
    ):
        out[key] = after[key] - before[key]
    return out


def _guard_score(summary: dict) -> float:
    """Score calibration candidates without letting confidence beat correctness."""
    return (
        1000.0 * summary["accuracy"]
        + 100.0 * summary["safe_accuracy"]
        + 10.0 * summary["macro_f1"]
        - summary["nll"]
        - summary["ece"]
        - 0.001 * summary["failures"]
    )


def _select_guarded_calibration(train: list[LabelledCase]) -> tuple[CalibrationConfig, dict]:
    """Fit several calibration candidates and keep only the best guard performer."""
    if len(train) < 20:
        return CalibrationConfig(), {"strategy": "baseline_only", "reason": "too_few_train_cases"}
    fit_cases, guard_cases = deterministic_train_eval(
        train, train_percent=80, salt="ariadne-engine-guard-split-v1")
    guard_score_cases = adversarial_mutations(guard_cases, include_original=True)
    candidates: list[tuple[str, CalibrationConfig]] = [("baseline", CalibrationConfig())]

    temp = run_inference_benchmark(
        fit_cases,
        fit_calibration=True,
        fit_channels=False,
        fit_labels=False,
        ablations=False,
    )
    candidates.append(("temperature", temp.calibration))

    channel = run_inference_benchmark(
        fit_cases,
        fit_calibration=True,
        fit_channels=True,
        fit_labels=False,
        ablations=False,
    )
    candidates.append(("temperature_channel_weights", channel.calibration))

    full = run_inference_benchmark(
        fit_cases,
        fit_calibration=True,
        fit_channels=True,
        fit_labels=True,
        ablations=False,
    )
    candidates.append(("temperature_channel_weights_label_bias", full.calibration))

    scored = []
    for name, cfg in candidates:
        guard = run_inference_benchmark(
            guard_score_cases,
            calibration=cfg,
            fit_calibration=False,
            fit_channels=False,
            fit_labels=False,
            ablations=False,
        )
        summary = _summary(guard)
        scored.append({
            "name": name,
            "score": _guard_score(summary),
            "summary": summary,
            "calibration": asdict(cfg),
        })
    scored.sort(key=lambda row: row["score"], reverse=True)
    best = scored[0]
    return CalibrationConfig(**{
        key: best["calibration"][key]
        for key in ("temperature", "label_bias", "channel_weights",
                    "class_temperatures", "version")
    }), {
        "strategy": "guarded_selection",
        "fit_cases": len(fit_cases),
        "guard_cases": len(guard_cases),
        "guard_score_cases": len(guard_score_cases),
        "selected": best["name"],
        "candidates": scored,
    }


def _run_pair(train: list[LabelledCase], eval_cases: list[LabelledCase]) -> dict:
    baseline = run_inference_benchmark(
        eval_cases,
        calibration=CalibrationConfig(),
        fit_calibration=False,
        fit_channels=False,
        fit_labels=False,
        ablations=False,
    )
    tuned_calibration, selection = _select_guarded_calibration(train)
    tuned = run_inference_benchmark(
        eval_cases,
        calibration=tuned_calibration,
        fit_calibration=False,
        fit_channels=False,
        fit_labels=False,
        ablations=False,
    )
    before = _summary(baseline)
    after = _summary(tuned)
    return {
        "baseline": before,
        "tuned": after,
        "delta": _delta(before, after),
        "trained_calibration": asdict(tuned_calibration),
        "selection": selection,
    }


def _fmt(value) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) < 1e-5 and value != 0:
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


def _markdown(report: dict) -> str:
    lines = [
        "# Ariadne Engine Improvement Benchmark",
        "",
        "This report compares engine settings on identical held-out cases.",
        "It does not include closure gates.",
        "",
        f"- corpus: `{report['corpus_path']}`",
        f"- total cases: `{report['n_cases']}`",
        f"- train cases: `{report['n_train']}`",
        f"- eval cases: `{report['n_eval']}`",
        f"- adversarial eval cases: `{report['n_adversarial_eval']}`",
        "",
    ]
    for section, title in (
        ("heldout_real_corpus", "Held-Out Real Corpus"),
        ("heldout_adversarial_mutations", "Held-Out Adversarial Mutations"),
    ):
        lines.extend([
            f"## {title}",
            "",
            "| metric | baseline engine | tuned engine | delta |",
            "|---|---:|---:|---:|",
        ])
        block = report[section]
        for key in (
            "accuracy", "safe_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "nll", "brier", "ece", "failures",
        ):
            lines.append(
                f"| {key} | {_fmt(block['baseline'][key])} | "
                f"{_fmt(block['tuned'][key])} | {_fmt(block['delta'][key])} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "- Accuracy deltas measure class decisions only.",
        "- Safe accuracy gives credit for correct abstain/follow-up behavior.",
        "- Lower NLL, Brier, and ECE mean better calibrated confidence.",
        "- The tuned engine is trained only on the deterministic train split and scored on held-out eval cases.",
    ])
    return "\n".join(lines) + "\n"


def build_report(args) -> dict:
    cases = read_labelled_cases_jsonl(args.corpus)
    train, eval_cases = deterministic_train_eval(cases, train_percent=args.train_percent)
    adversarial_eval = adversarial_mutations(eval_cases, include_original=True)
    real_pair = _run_pair(train, eval_cases)
    adversarial_pair = _run_pair(train, adversarial_eval)
    return {
        "schema": "ariadne.engine_improvement_benchmark.v1",
        "corpus_path": str(args.corpus),
        "n_cases": len(cases),
        "n_train": len(train),
        "n_eval": len(eval_cases),
        "n_adversarial_eval": len(adversarial_eval),
        "train_percent": args.train_percent,
        "heldout_real_corpus": real_pair,
        "heldout_adversarial_mutations": adversarial_pair,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/benchmarks/real_corpus_mpc_500/labelled_cases.jsonl"),
    )
    parser.add_argument("--train-percent", type=int, default=70)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/benchmarks/engine_improvement"),
    )
    args = parser.parse_args(argv)

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "engine_improvement_report.json"
    md_path = args.out_dir / "engine_improvement_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "heldout_accuracy_delta": report["heldout_real_corpus"]["delta"]["accuracy"],
        "heldout_macro_f1_delta": report["heldout_real_corpus"]["delta"]["macro_f1"],
        "heldout_ece_delta": report["heldout_real_corpus"]["delta"]["ece"],
        "adversarial_macro_f1_delta": report["heldout_adversarial_mutations"]["delta"]["macro_f1"],
        "adversarial_ece_delta": report["heldout_adversarial_mutations"]["delta"]["ece"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
