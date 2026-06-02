"""System-wide proof, closure, and residual intelligence for Ariadne."""
from .closure import (
    ArtifactEvidence,
    ClosureLedger,
    ClosureReport,
    GateResult,
    ResidualSignal,
    SubsystemContract,
    build_closure_report,
    load_json_artifact,
    stable_hash,
    write_closure_report,
)
from .defaults import (
    audit_png_directory,
    build_default_ariadne_closure,
    collect_default_evidence,
    default_contracts,
    known_residuals,
)
from .dream import DreamExperiment, DreamRun, build_dream_run, write_dream_run
from .artifact_manifest import build_artifact_manifest, write_artifact_manifest
from .high_fidelity import (
    covariance_envelope_evidence,
    independent_crosscheck_evidence,
    nbody_replay_evidence,
)
from .promotion import (
    PromotionEvidence,
    PromotionReport,
    PromotionRung,
    PromotionThresholds,
    RoutePromotionCertificate,
    load_routes_from_navigator_report,
    promote_route,
    promote_routes,
    write_promotion_report,
)
from .visuals import navigator_visual_contract_evidence

__all__ = [
    "ArtifactEvidence",
    "ClosureLedger",
    "ClosureReport",
    "GateResult",
    "ResidualSignal",
    "SubsystemContract",
    "build_closure_report",
    "load_json_artifact",
    "stable_hash",
    "write_closure_report",
    "audit_png_directory",
    "build_default_ariadne_closure",
    "collect_default_evidence",
    "default_contracts",
    "known_residuals",
    "DreamExperiment",
    "DreamRun",
    "build_dream_run",
    "write_dream_run",
    "build_artifact_manifest",
    "write_artifact_manifest",
    "nbody_replay_evidence",
    "covariance_envelope_evidence",
    "independent_crosscheck_evidence",
    "PromotionEvidence",
    "PromotionReport",
    "PromotionRung",
    "PromotionThresholds",
    "RoutePromotionCertificate",
    "load_routes_from_navigator_report",
    "promote_route",
    "promote_routes",
    "write_promotion_report",
    "navigator_visual_contract_evidence",
]
