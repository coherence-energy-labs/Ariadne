"""Synthetic-truth catalog for the imaging pipeline.

When `archive_fetch.synthesise_decam_tile` plants moving objects into a stack
of synthetic DECam-style FITS images, we lose the link between the pixel that
gets stamped and the orbit that generated it. Without that link we can count
how many chains the linker produces, but we can't ask the question we
actually care about: are those chains REAL or SPURIOUS?

This module provides the side-channel ground-truth carrier:

  * `TruthEntry`         one planted (image_id, mjd, ra, dec, x_pix, y_pix,
                         mag, family, truth_id) record
  * `TruthCatalog`       searchable container indexed by (image_id, x, y)
  * `match_source`       returns the truth_id closest to a Source within
                         `match_radius_pix` pixels, or None
  * `label_sources`      bulk labelling helper: returns {idx: truth_id|None}
  * `assign_truth_to_chain`
                         returns the dominant truth_id for a chain if at
                         least `min_purity` of its entries share that ID
  * `measure_linker_quality`
                         precision / recall / F1 / per-truth coverage

The catalog itself is plain JSON for easy round-tripping into reports.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TruthEntry:
    """One planted source in one image."""
    truth_id: str            # stable across images (e.g. "synth_obj_002")
    image_id: str            # FITS filename or exposure id
    mjd: float               # MJD-OBS (mid-exposure)
    ra: float                # injected RA in degrees (true position)
    dec: float               # injected Dec in degrees
    x_pix: float             # injected pixel column in this image
    y_pix: float             # injected pixel row in this image
    mag: float = -99.0       # injected magnitude (-99 if unknown)
    family: str = "unknown"  # orbit family: kepler_tno, kepler_neo, ...
    extras: dict = field(default_factory=dict)   # arbitrary metadata


class TruthCatalog:
    """Container of TruthEntries with per-image spatial index."""

    def __init__(self, entries: Iterable[TruthEntry] = ()):
        self.entries: list[TruthEntry] = list(entries)
        self._by_image: dict[str, list[TruthEntry]] = defaultdict(list)
        for e in self.entries:
            self._by_image[e.image_id].append(e)

    @property
    def truth_ids(self) -> set[str]:
        return {e.truth_id for e in self.entries}

    def entries_for_image(self, image_id: str) -> list[TruthEntry]:
        return self._by_image.get(image_id, [])

    def match_source(self, src, *, match_radius_pix: float = 2.5) -> str | None:
        """Return the truth_id whose stamped (x, y) lies within
        `match_radius_pix` pixels of `src.x, src.y` in the same image_id.
        Returns the CLOSEST match if multiple are within range, or None.
        """
        candidates = self._by_image.get(src.image_id, ())
        best_id, best_d = None, float("inf")
        for e in candidates:
            d = math.hypot(e.x_pix - src.x, e.y_pix - src.y)
            if d <= match_radius_pix and d < best_d:
                best_d = d
                best_id = e.truth_id
        return best_id

    def label_sources(self, sources, *,
                       match_radius_pix: float = 2.5) -> dict[int, str | None]:
        """Bulk-label a list of sources: returns {source_idx: truth_id|None}."""
        return {i: self.match_source(s, match_radius_pix=match_radius_pix)
                for i, s in enumerate(sources)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "ariadne.synthetic_truth.v1",
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "TruthCatalog":
        payload = json.loads(Path(path).read_text())
        return cls(TruthEntry(**e) for e in payload["entries"])


# ---------------------------------------------------------------------------
# Tracklet / chain truth propagation
# ---------------------------------------------------------------------------

def _chain_entry_to_mjd(entry: dict) -> float:
    """Chain entries carry `t` in SPICE-ET seconds (J2000 TDB epoch).
    Convert to MJD so we can match against TruthEntry.mjd which is in UTC MJD.
    """
    # MJD = ET_sec / 86400 + 51544.5  (J2000 epoch in MJD)
    if "jd" in entry and entry["jd"] not in (None, 0.0):
        return float(entry["jd"]) - 2400000.5
    return float(entry["t"]) / 86400.0 + 51544.5


def assign_truth_to_chain(chain: Sequence[dict],
                            catalog: TruthCatalog,
                            *, match_radius_arcsec: float = 4.0,
                            min_purity: float = 0.67,
                            time_window_days: float = 0.5,
                            ) -> tuple[str | None, float]:
    """Assign the dominant truth_id to a chain via TIME-WINDOWED spatial
    matching of each chain entry against the catalog.

    A true single-object chain entry at time t should match a TruthEntry
    whose `mjd` is within `time_window_days` AND whose (ra, dec) lies
    within `match_radius_arcsec` of the entry's position. Without the
    time filter we would match across nights, miscounting purity.

    Chain RA/Dec come in radians (the imaging-pipeline convention; see
    iod_advanced.py `los()`). We convert to degrees for matching.

    A chain is assigned to a truth_id iff at least `min_purity` fraction
    of its entries vote for that ID. Otherwise returns (None, purity).
    """
    radius_deg = match_radius_arcsec / 3600.0
    votes = Counter()
    for entry in chain:
        ra_deg = math.degrees(entry["ra"]) % 360.0
        dec_deg = math.degrees(entry["dec"])
        cos_dec = math.cos(math.radians(dec_deg))
        entry_mjd = _chain_entry_to_mjd(entry)

        best_id, best_d = None, radius_deg
        for e in catalog.entries:
            if abs(e.mjd - entry_mjd) > time_window_days:
                continue
            dra = (e.ra - ra_deg) * cos_dec
            ddec = e.dec - dec_deg
            d = math.hypot(dra, ddec)
            if d <= best_d:
                best_d = d
                best_id = e.truth_id
        if best_id is not None:
            votes[best_id] += 1
        else:
            votes["_NOMATCH_"] += 1
    if not votes:
        return None, 0.0
    top_id, top_count = votes.most_common(1)[0]
    purity = top_count / len(chain)
    if top_id == "_NOMATCH_":
        return None, purity
    if purity < min_purity:
        return None, purity
    return top_id, purity


def measure_linker_quality(chains: Sequence,
                            catalog: TruthCatalog,
                            *, match_radius_arcsec: float = 4.0,
                            min_purity: float = 0.67) -> dict:
    """Compute precision/recall/F1 + per-truth coverage for a set of chains.

    Precision = (chains assigned to a real truth_id) / (total chains)
    Recall    = (unique truth_ids covered) / (truth_ids in catalog)
    F1        = harmonic mean

    Per-truth coverage = {truth_id: how_many_chains_resolved_to_it}.
    """
    truth_for_chain = []
    purities = []
    for ch in chains:
        tid, pur = assign_truth_to_chain(ch, catalog,
                                           match_radius_arcsec=match_radius_arcsec,
                                           min_purity=min_purity)
        truth_for_chain.append(tid)
        purities.append(pur)

    n_total = len(chains)
    n_pure = sum(1 for t in truth_for_chain if t is not None)
    precision = (n_pure / n_total) if n_total else 0.0

    covered = {t for t in truth_for_chain if t is not None}
    total_truth = len(catalog.truth_ids)
    recall = (len(covered) / total_truth) if total_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    coverage = Counter(t for t in truth_for_chain if t is not None)
    return {
        "n_chains": n_total,
        "n_pure_chains": n_pure,
        "n_spurious_chains": n_total - n_pure,
        "n_truth_total": total_truth,
        "n_truth_covered": len(covered),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_purity": float(sum(purities) / max(len(purities), 1)),
        "coverage_per_truth": dict(coverage),
        "truth_for_chain": truth_for_chain,
        "purity_per_chain": purities,
    }
