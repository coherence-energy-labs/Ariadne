"""SPICE kernel management (MASTER_PLAN.md §5.1).

Downloads, caches, checksums, and furnishes the real NASA/NAIF kernels:
  - naif0012.tls : leap seconds (time conversion)
  - de440s.bsp   : JPL DE440 planetary+lunar ephemeris (1849-2150)
  - gm_de440.tpc : GM constants consistent with DE440
  - pck00011.tpc : body orientation / radii

Kernels live in data/kernels/ (git-ignored); kernels.lock.json records the exact
URLs, sizes, and SHA-256 hashes for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
KERNEL_DIR = os.path.join(REPO_ROOT, "data", "kernels")
LOCK = os.path.join(KERNEL_DIR, "kernels.lock.json")

_NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
KERNELS = [
    ("naif0012.tls", f"{_NAIF}/lsk/naif0012.tls"),
    ("gm_de440.tpc", f"{_NAIF}/pck/gm_de440.tpc"),
    ("pck00011.tpc", f"{_NAIF}/pck/pck00011.tpc"),
    ("de440s.bsp", f"{_NAIF}/spk/planets/de440s.bsp"),
]

_furnished = False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_kernels(force: bool = False) -> dict:
    """Download any missing kernels and (re)write the lock file. Returns the lock."""
    os.makedirs(KERNEL_DIR, exist_ok=True)
    lock = {}
    for name, url in KERNELS:
        dest = os.path.join(KERNEL_DIR, name)
        if force or not os.path.exists(dest):
            req = urllib.request.Request(url, headers={"User-Agent": "ariadne/0.5"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        lock[name] = {"url": url, "sha256": _sha256(dest),
                      "bytes": os.path.getsize(dest)}
    with open(LOCK, "w") as f:
        json.dump(lock, f, indent=2)
    return lock


def ensure_kernels() -> None:
    missing = [n for n, _ in KERNELS
               if not os.path.exists(os.path.join(KERNEL_DIR, n))]
    if missing:
        download_kernels()


def furnish() -> None:
    """Idempotently furnish all kernels into the SPICE kernel pool."""
    global _furnished
    if _furnished:
        return
    ensure_kernels()
    import spiceypy as sp
    for name, _ in KERNELS:
        sp.furnsh(os.path.join(KERNEL_DIR, name))
    _furnished = True


if __name__ == "__main__":
    lk = download_kernels()
    for name, meta in lk.items():
        print(f"{name:16s} {meta['bytes']:>12,d} bytes  sha256={meta['sha256'][:16]}...")
