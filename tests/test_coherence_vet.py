"""The Equation-of-ONE vetting selector (gap22 pattern) ranks a real linear arc as
far more coherent (lower energy) than a chance alignment, and as a thresholded
selector keeps the real one while dropping chance."""
from __future__ import annotations

import numpy as np


def _arc(rate=25.0, nights=(60000.0, 60001.0, 60002.0), per_night=2, jitter=0.0,
         mag=20.0, mag_jit=0.0, seed=0):
    """A constant-rate linear arc; jitter>0 perturbs positions (loss of coherence)."""
    rng = np.random.default_rng(seed)
    ra, dec, mjd, mg = [], [], [], []
    t0 = nights[0]
    for base in nights:
        for k in range(per_night):
            t = base + k * 0.01
            dt = (t - t0) * 24.0
            ra.append(180.0 + rate * dt / 3600.0 + rng.normal(0, jitter) / 3600.0)
            dec.append(0.0 + rng.normal(0, jitter) / 3600.0)
            mjd.append(t); mg.append(mag + rng.normal(0, mag_jit))
    return np.array(ra), np.array(dec), np.array(mjd), np.array(mg)


def test_real_arc_is_more_coherent_than_chance():
    from ariadne.discovery.imaging.coherence_vet import track_energy
    ra, dec, mjd, mag = _arc(jitter=0.0, mag_jit=0.02)
    E_real = track_energy(ra, dec, mjd, mag)
    # chance: random positions on the same nights, erratic brightness
    rng = np.random.default_rng(3)
    cra = 180.0 + rng.uniform(-0.05, 0.05, len(ra))
    cdec = rng.uniform(-0.05, 0.05, len(ra))
    cmag = 20.0 + rng.uniform(-1.5, 1.5, len(ra))
    E_chance = track_energy(cra, cdec, mjd, cmag)
    assert E_real < E_chance, f"real {E_real:.3f} should beat chance {E_chance:.3f}"


def test_vet_selector_keeps_real_drops_chance():
    from ariadne.discovery.imaging.coherence_vet import track_energy, vet_coherence
    real = _arc(jitter=0.0, mag_jit=0.02)
    rng = np.random.default_rng(5)
    n = len(real[0])
    chance = (180.0 + rng.uniform(-0.05, 0.05, n), rng.uniform(-0.05, 0.05, n),
              real[2], 20.0 + rng.uniform(-1.5, 1.5, n))
    tau = 0.5 * (track_energy(*real) + track_energy(*chance))
    kept = vet_coherence([real, chance], tau=tau)
    assert any(k is real for k in kept) and all(k is not chance for k in kept)
