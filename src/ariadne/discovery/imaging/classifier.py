"""Top-level identification (Stage 4): fuse all analyzers into one verdict.

`identify(candidate)` is the single entry point for the whole "spot -> analyze
-> identify" engine. It runs the characterization spine (which itself invokes
rate/distance/size/color/morphology for movers, or the light-curve analyzer for
variables), then assembles a composite plain-language identification, a ranked
posterior over the taxonomy, an overall confidence, the evidence that fired,
and the single most useful next observation.

Calibrated, not absolute: the confidence reflects how much the available data
actually constrains the answer, and `next_step` says how to sharpen it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .characterize import characterize, ObjectDossier


@dataclass
class Identification:
    label: str                 # composite, human-readable
    posterior: dict            # {class: prob}
    confidence: str            # low | moderate | high
    properties: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    next_step: str = ""


def identify(candidate: dict) -> Identification:
    d: ObjectDossier = characterize(candidate)
    posterior = dict(d.type_probabilities)
    top = max(posterior, key=posterior.get) if posterior else "unknown"
    evidence, label = [], top

    if d.kind == "mover":
        p = d.properties
        evidence.append(f"rate {p.get('rate_arcsec_hr', '?')}\"/hr -> distance {p.get('helio_r_au', float('nan')):.1f} AU")
        bits = [top]
        surf = p.get("surface_type")
        if surf:
            bits.append(max(surf, key=surf.get).split(" ")[0] + "-type")
            evidence.append("color -> surface taxonomy")
        if any("coma" in f or "comet" in f.lower() for f in d.flags):
            bits.append("(active/cometary)"); evidence.append("morphology -> coma/tail")
        if p.get("size_km_est") == p.get("size_km_est"):
            bits.append(f"~{p['size_km_est']:.1f} km"); evidence.append("H + albedo -> size")
        bits.append(d.known_id if d.known_id else "NEW candidate")
        label = " ".join(str(b) for b in bits)
        if any("HYPERBOLIC" in f for f in d.flags):
            label = "INTERSTELLAR object candidate (" + label + ")"
    else:
        p = d.properties
        if "period_days" in p:
            evidence.append(f"light curve: P={p['period_days']:.3g} d, shape={p.get('shape')}")
            label = f"{top} (P={p['period_days']:.3g} d)"
        else:
            evidence.append("light-curve shape (aperiodic / sparse)")

    return Identification(label=label, posterior=posterior, confidence=d.confidence,
                          properties=d.properties, evidence=evidence, flags=d.flags,
                          next_step=(d.disambiguate[0] if d.disambiguate else ""))
