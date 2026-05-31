"""Multi-band photometric taxonomy: asteroid surface composition from colors.

When a candidate has detections in two or more bands (e.g. ZTF g, r, i), the
color indices reveal its surface composition. The dominant asteroid taxonomic
classes have well-measured photometric signatures:

  C-type   (carbonaceous)   neutral, flat reflectance; g-r ~ 0.40, r-i ~ 0.12
  S-type   (silicaceous)    reddish; g-r ~ 0.55, r-i ~ 0.20
  V-type   (Vesta-like)     deep absorption near 0.5 um -> blue/purple slope
  X-type   (metallic / E)   slight red, often featureless
  D-type   (Trojans)        very red; g-r ~ 0.65, r-i ~ 0.25
  B-type   (sub-C)          slightly blue; g-r ~ 0.35

For TNOs the color-composition mapping is different (gray vs red TNOs);
nevertheless g-r and r-i are still useful population labels.

This module is a transparent nearest-class classifier in (g-r, r-i, i-z) color
space. Confidence reflects distance to the class centroid (smaller = better).
No ML training; the centroids are literature values from DeMeo et al. 2009
(asteroids) and Tegler et al. 2003 (TNOs).

When color data is insufficient (<2 bands), returns "UNKNOWN" with confidence 0.

Reference: DeMeo+ 2009 (Bus-DeMeo taxonomy); Tegler+ 2003 (TNO color bimodality).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# Bus-DeMeo class centroids in (g-r, r-i, i-z). Values are approximate from
# pan-STARRS asteroid color compilations (Veres+ 2015, Sergeyev & Carry 2021).
ASTEROID_CLASS_CENTROIDS = {
    "C":  (0.40, 0.12, 0.05),
    "B":  (0.35, 0.10, 0.04),
    "X":  (0.45, 0.16, 0.10),
    "S":  (0.55, 0.20, 0.13),
    "V":  (0.50, 0.05, -0.10),
    "D":  (0.65, 0.25, 0.15),
    "Q":  (0.50, 0.15, 0.05),
}

# TNO color classes (Tegler & Romanishin 1998 bimodality)
TNO_CLASS_CENTROIDS = {
    "TNO_GRAY":  (0.45, 0.15, 0.08),
    "TNO_RED":   (0.85, 0.40, 0.25),
    "TNO_VERY_RED": (1.05, 0.55, 0.35),
}


@dataclass(frozen=True)
class ColorTaxonomy:
    """Color-based classification result."""
    label: str
    confidence: float
    color_indices: dict
    distance_to_centroid: float
    all_class_distances: dict
    n_bands_used: int


def _color_distance(c1: tuple, c2: tuple, sigma: float = 0.10) -> float:
    """Mahalanobis-style distance between two color tuples, normalised by sigma."""
    total = 0.0
    for a, b in zip(c1, c2):
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            continue
        total += ((a - b) / sigma) ** 2
    return math.sqrt(total)


def classify_colors(band_magnitudes: dict,
                    *, centroid_set: str = "asteroid",
                    sigma: float = 0.10,
                    min_bands: int = 2) -> ColorTaxonomy:
    """Map a candidate's per-band magnitudes to the nearest taxonomic class.

    Args:
      band_magnitudes:   dict of {'g': 21.3, 'r': 20.9, 'i': 20.7, 'z': 20.6, ...}
                          missing bands are simply not used for the color index.
      centroid_set:      "asteroid" (Bus-DeMeo) or "tno" (Tegler bimodality).
      sigma:             per-color noise sigma for distance normalisation; smaller
                          -> more discriminative but more sensitive to mag errors.
      min_bands:         require at least this many bands; return UNKNOWN if not met.

    Returns:
      ColorTaxonomy with label, confidence, the colors used, and the full
      distance-to-each-class table for diagnostics.
    """
    bands_present = [b for b in band_magnitudes if math.isfinite(band_magnitudes.get(b, float("nan")))]
    if len(bands_present) < min_bands:
        return ColorTaxonomy("UNKNOWN", 0.0, {}, float("inf"), {}, len(bands_present))

    g_r = band_magnitudes.get("g", float("nan")) - band_magnitudes.get("r", float("nan"))
    r_i = band_magnitudes.get("r", float("nan")) - band_magnitudes.get("i", float("nan"))
    i_z = band_magnitudes.get("i", float("nan")) - band_magnitudes.get("z", float("nan"))
    colors = (g_r if math.isfinite(g_r) else None,
              r_i if math.isfinite(r_i) else None,
              i_z if math.isfinite(i_z) else None)
    color_indices_d = {"g-r": colors[0], "r-i": colors[1], "i-z": colors[2]}

    centroids = ASTEROID_CLASS_CENTROIDS if centroid_set == "asteroid" else TNO_CLASS_CENTROIDS
    distances = {label: _color_distance(colors, c, sigma) for label, c in centroids.items()}
    best_label = min(distances, key=lambda k: distances[k])
    best_dist = distances[best_label]

    # confidence: 1.0 at d=0, 0.5 at d=1 sigma, smooth decay
    confidence = math.exp(-0.5 * best_dist ** 2)

    return ColorTaxonomy(
        label=best_label, confidence=confidence,
        color_indices=color_indices_d,
        distance_to_centroid=best_dist,
        all_class_distances=distances,
        n_bands_used=len(bands_present),
    )


def merge_observations_to_band_dict(observations) -> dict:
    """Helper: convert a list of (band, mag) pairs into a band: mag dict via
    median-over-band (rejecting outliers).
    """
    from collections import defaultdict
    import statistics
    per_band = defaultdict(list)
    for item in observations:
        if hasattr(item, "band") and hasattr(item, "mag"):
            band, mag = item.band, item.mag
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            band, mag = item[0], item[1]
        else:
            continue
        if band and math.isfinite(mag) and -3 <= mag <= 30:
            per_band[band[0]].append(mag)
    out = {}
    for band, mags in per_band.items():
        if len(mags) >= 3:
            out[band] = float(statistics.median(mags))
        else:
            out[band] = float(sum(mags) / len(mags))
    return out
