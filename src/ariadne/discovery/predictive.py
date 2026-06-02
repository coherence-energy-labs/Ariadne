"""Predictive observation scheduler -- learns which actions confirm which hypotheses.

Sits one layer above the inference engine. When the inference engine produces a
posterior over hypotheses, this scheduler decides WHICH observation action to
take to maximise information gain, using a learned model of past outcomes.

Pattern (inspired by adaptive backend-selection engines in production
compilers): for each (evidence_signature, action) pair, record whether the
action subsequently CONFIRMED or REJECTED the leading hypothesis. Over time,
score each action by its historical confirmation rate for that evidence class,
and pick the action with the highest expected information gain per unit
"observation cost."

Available actions (every action is a real follow-up the operator could take):

  observe_second_night          take one more detection ~24h later
  observe_multi_band            take photometry in g/r/i/z bands
  observe_deep_stack            shift-and-stack 10+ exposures along trajectory
  query_skybot                  cross-match against the IMCCE SkyBoT catalog
  query_horizons                fetch JPL Horizons predictions for a hypothesis
  alert_and_submit_mpc          submit candidate to MPC for confirmation
  archive_search                search archival surveys (ZTF DR / PS1 / DECam)
  monitor_only                  store; no action this run

The scheduler:
  * Records each (evidence_class, action) -> outcome mapping in JSON-on-disk
    (a tiny ledger that grows with the engine's experience).
  * On a new evidence sample, looks up the historical confirmation rates,
    computes expected info gain per action, and recommends the best.
  * Falls back to a sensible hand-coded prior when no history exists for the
    evidence class (cold-start behaviour).

This is deliberately NOT a deep learning model -- transparent rules + a tiny
ledger give better behaviour and accountability for an engine that operates
on real candidates that might become real publications.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# --- canonical action set ------------------------------------------------

ACTIONS = (
    "observe_second_night",
    "observe_multi_band",
    "observe_deep_stack",
    "query_skybot",
    "query_horizons",
    "alert_and_submit_mpc",
    "archive_search",
    "monitor_only",
    "discard",
)

# Sensible cold-start prior: which action is best given some evidence class
# (used before the ledger has data on this class). The number = belief that
# this action will produce a confirmation for evidence of this class.
COLD_START_PRIORS = {
    # bright single detection, no rate yet
    "single_detection_no_rate": {
        "query_skybot": 0.6,
        "observe_second_night": 0.4,
        "archive_search": 0.3,
        "monitor_only": 0.1,
    },
    # multi-night arc, low RMS, no skybot match
    "confirmed_orbit_no_match": {
        "alert_and_submit_mpc": 0.7,
        "observe_multi_band": 0.6,
        "observe_second_night": 0.5,
        "query_skybot": 0.2,
    },
    # high-rate streak
    "fast_mover_streak": {
        "observe_second_night": 0.7,
        "query_skybot": 0.5,
        "alert_and_submit_mpc": 0.3,
        "discard": 0.2,
    },
    # low confidence, low rate, single detection
    "low_confidence_low_rate": {
        "monitor_only": 0.5,
        "query_skybot": 0.3,
        "discard": 0.4,
    },
    # high-novelty (Sednoid / Detached / Comet) candidate
    "high_novelty_distant": {
        "alert_and_submit_mpc": 0.8,
        "observe_deep_stack": 0.7,
        "observe_multi_band": 0.6,
        "observe_second_night": 0.6,
    },
    # likely artefact (high bogus score)
    "likely_artefact": {
        "discard": 0.9,
        "monitor_only": 0.2,
    },
    # generic fallback
    "default": {
        "observe_second_night": 0.5,
        "query_skybot": 0.5,
        "monitor_only": 0.3,
    },
}

# Observation costs (relative; lower is cheaper). Used to penalise expensive
# actions when the cheap one gives comparable info gain.
ACTION_COSTS = {
    "observe_second_night": 1.0,
    "observe_multi_band": 1.8,
    "observe_deep_stack": 3.0,
    "query_skybot": 0.1,
    "query_horizons": 0.1,
    "alert_and_submit_mpc": 0.3,
    "archive_search": 0.5,
    "monitor_only": 0.0,
    "discard": 0.0,
}


@dataclass
class ActionOutcome:
    """One historical (evidence_class, action) -> outcome record."""
    evidence_class: str
    action: str
    outcome: str            # "confirmed" | "refuted" | "inconclusive"
    ts_unix: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class ActionRecommendation:
    """Output: the action the scheduler thinks the operator should take."""
    action: str
    expected_confirmation_prob: float
    expected_info_gain_nats: float
    cost: float
    score: float            # info_gain / max(cost, 0.1)
    rationale: str
    separates: list = field(default_factory=list)


class PredictiveScheduler:
    """Adaptive observation-action scheduler with on-disk experience ledger.

    Usage:
      sched = PredictiveScheduler(ledger_path="C:/ariadne-runs/strategy.json")
      reco = sched.recommend(evidence_class="confirmed_orbit_no_match",
                              hypothesis_posterior=0.72)
      # after acting + observing:
      sched.record_outcome(evidence_class=..., action=reco.action,
                            outcome="confirmed")
      sched.save()
    """

    def __init__(self, ledger_path: str | Path | None = None):
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self._history: list[ActionOutcome] = []
        if self.ledger_path and self.ledger_path.exists():
            self._load()

    def _load(self):
        with open(self.ledger_path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.get("history", []):
            self._history.append(ActionOutcome(**rec))

    def save(self):
        if not self.ledger_path:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        payload = {
            "version": 1,
            "saved_at_unix": time.time(),
            "n_records": len(self._history),
            "history": [asdict(h) for h in self._history],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.ledger_path)

    def record_outcome(self, *, evidence_class: str, action: str,
                       outcome: str, notes: str = ""):
        """Append a new (evidence_class, action) -> outcome record.

        outcome must be one of "confirmed" | "refuted" | "inconclusive".
        """
        if outcome not in ("confirmed", "refuted", "inconclusive"):
            raise ValueError(f"outcome must be confirmed/refuted/inconclusive, got {outcome!r}")
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS!r}, got {action!r}")
        self._history.append(ActionOutcome(
            evidence_class=evidence_class, action=action,
            outcome=outcome, notes=notes,
        ))

    def historical_confirmation_rate(self, evidence_class: str,
                                     action: str,
                                     min_samples: int = 3) -> Optional[float]:
        """Return confirmed / (confirmed + refuted) for this (class, action).

        Returns None if fewer than `min_samples` past records exist.
        """
        confirmed = 0; refuted = 0
        for h in self._history:
            if h.evidence_class == evidence_class and h.action == action:
                if h.outcome == "confirmed":
                    confirmed += 1
                elif h.outcome == "refuted":
                    refuted += 1
        n = confirmed + refuted
        if n < min_samples:
            return None
        return confirmed / n

    def expected_info_gain(self, evidence_class: str, action: str,
                            hypothesis_posterior: float) -> float:
        """How many nats of entropy would this action expect to reduce?

        Use binary entropy on (posterior, 1-posterior) and the action's
        historical confirmation probability to compute expected info gain.
        """
        if hypothesis_posterior <= 0 or hypothesis_posterior >= 1:
            return 0.0

        prior_entropy = -(hypothesis_posterior * math.log(hypothesis_posterior) +
                          (1 - hypothesis_posterior) * math.log(1 - hypothesis_posterior))

        p_confirm = self.historical_confirmation_rate(evidence_class, action)
        if p_confirm is None:
            # cold-start: use the COLD_START_PRIORS as a soft signal
            base = COLD_START_PRIORS.get(evidence_class, COLD_START_PRIORS["default"])
            p_confirm = base.get(action, 0.3)

        # Information gain = H(prior) - E[H(posterior | action_outcome)].
        # If the action is DETERMINISTIC (p_confirm=1.0 always confirms real
        # hypotheses, or p=0.0 always refutes them), the outcome perfectly
        # discriminates and info gain = full prior entropy. If p_confirm=0.5
        # the outcome is random and gives zero information.
        # Discriminativeness = (2*p_confirm - 1)^2: high at p=0 or p=1, zero at 0.5.
        discriminative = (2.0 * p_confirm - 1.0) ** 2
        return prior_entropy * max(0.05, discriminative)

    def recommend(self, *, evidence_class: str,
                  hypothesis_posterior: float,
                  alternatives: list | None = None,
                  exclude: list[str] | None = None,
                  confirmation_weight: float = 1.5) -> ActionRecommendation:
        """Score every action; recommend the one with the highest info-gain-per-cost.

        Info-gain is symmetric (confirmations and refutations are equally
        informative), but in practice the OPERATOR wants confirmations -- so
        we weight the score by `(1 + confirmation_weight * p_confirm)` to break
        the tie in favour of actions that historically lead to confirmations.
        """
        exclude = exclude or []
        alternatives = alternatives or []
        best = None
        best_score = -float("inf")
        rationale = ""
        for action in ACTIONS:
            if action in exclude:
                continue
            info_gain = self.expected_info_gain(evidence_class, action,
                                                hypothesis_posterior)
            cost = ACTION_COSTS.get(action, 1.0)
            p_conf = self.historical_confirmation_rate(evidence_class, action)
            if p_conf is None:
                p_conf = COLD_START_PRIORS.get(evidence_class,
                                                COLD_START_PRIORS["default"]
                                                ).get(action, 0.3)
            score = info_gain * (1.0 + confirmation_weight * p_conf) / max(cost, 0.1)
            separates = discriminating_actions(evidence_class, alternatives).get(action, [])
            if separates:
                score *= 1.0 + 0.15 * len(separates)
            if score > best_score:
                best_score = score
                source = "history" if self.historical_confirmation_rate(
                    evidence_class, action) is not None else "cold-start prior"
                best = ActionRecommendation(
                    action=action,
                    expected_confirmation_prob=p_conf,
                    expected_info_gain_nats=info_gain,
                    cost=cost,
                    score=score,
                    rationale=(f"Best info-gain/cost (confirm-weighted): "
                                f"P(confirm)={p_conf:.2f} ({source}), info gain "
                                f"{info_gain:.3f} nats, cost {cost:.2f}."),
                    separates=separates,
                )
        if best is None:
            best = ActionRecommendation(
                action="monitor_only", expected_confirmation_prob=0.0,
                expected_info_gain_nats=0.0, cost=0.0, score=0.0,
                rationale="no actions available")
        return best

    def summary(self) -> dict:
        """Quick statistics over the experience ledger."""
        from collections import Counter
        per_class = Counter(h.evidence_class for h in self._history)
        per_action = Counter(h.action for h in self._history)
        outcomes = Counter(h.outcome for h in self._history)
        return {
            "n_records": len(self._history),
            "by_evidence_class": dict(per_class),
            "by_action": dict(per_action),
            "outcomes": dict(outcomes),
        }


def classify_evidence(evidence) -> str:
    """Map an Evidence object to one of the COLD_START_PRIORS keys.

    Heuristic; returns 'default' when no rule matches. Use to look up the
    historical confirmation rates for THIS kind of evidence.
    """
    # late-import to avoid cycle (predictive can be imported standalone)
    rate = getattr(evidence, "rate_arcsec_hr", None)
    n_det = getattr(evidence, "n_detections", 1)
    rms = getattr(evidence, "rms_arcsec", None)
    morphology = getattr(evidence, "morphology_label", None)
    skybot = getattr(evidence, "skybot_match_names", None)
    orbit_state = getattr(evidence, "orbit_state", None)

    if morphology in ("COSMIC_RAY", "EDGE_ARTEFACT"):
        return "likely_artefact"
    if morphology == "STREAK" or (rate is not None and rate > 60):
        return "fast_mover_streak"
    if n_det >= 4 and rms is not None and rms < 5 and skybot is not None and not skybot:
        if orbit_state is not None:
            # check if the orbit suggests a high-novelty class
            try:
                from . import taxonomy
                import numpy as np
                r = np.asarray(orbit_state[:3]); v = np.asarray(orbit_state[3:])
                tax = taxonomy.classify_state(r, v)
                if tax.label in ("SEDNOID", "DETACHED", "RESONANT_KBO",
                                 "COMET_HYPERBOLIC", "CLASSICAL_KBO",
                                 "SCATTERED_KBO"):
                    return "high_novelty_distant"
            except Exception:
                pass
        return "confirmed_orbit_no_match"
    if n_det == 1 and rate is None:
        return "single_detection_no_rate"
    if (rate is not None and rate < 0.5) and (n_det <= 2):
        return "low_confidence_low_rate"
    return "default"


def discriminating_actions(evidence_class: str, alternatives: list) -> dict[str, list]:
    """Map actions to hypothesis labels they can help separate.

    `alternatives` can be Hypothesis objects or dicts with label/orbital_class.
    This is intentionally transparent and rule-based.
    """
    labels = []
    for h in alternatives:
        if isinstance(h, dict):
            label = h.get("orbital_class") or h.get("label") or h.get("class")
        else:
            label = getattr(h, "orbital_class", None) or getattr(h, "label", None)
        if label:
            labels.append(label)
    labels = list(dict.fromkeys(labels))
    out = {a: [] for a in ACTIONS}
    outer = {"CENTAUR", "JTROJAN", "HILDA", "THULE", "CLASSICAL_KBO",
             "HOT_CLASSICAL", "RESONANT_KBO", "SCATTERED_KBO", "DETACHED", "SEDNOID"}
    if any(l in outer for l in labels):
        out["observe_multi_band"] = labels
        out["observe_deep_stack"] = labels
        out["archive_search"] = labels
    if evidence_class in {"single_detection_no_rate", "low_confidence_low_rate", "default"}:
        out["observe_second_night"] = labels
    if evidence_class in {"confirmed_orbit_no_match", "high_novelty_distant"}:
        out["query_skybot"] = labels
        out["query_horizons"] = labels
    if evidence_class == "likely_artefact":
        out["discard"] = labels
    return {k: v for k, v in out.items() if v}
