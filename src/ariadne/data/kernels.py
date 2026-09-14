"""SPICE kernel management (MASTER_PLAN.md §5.1).

Downloads, caches, checksums, and furnishes the real NASA/NAIF kernels:
  - naif0012.tls : leap seconds (time conversion)
  - de440s.bsp   : JPL DE440 planetary+lunar ephemeris (1849-2150)
  - gm_de440.tpc : GM constants consistent with DE440
  - pck00011.tpc : body orientation / radii

Kernels live in data/kernels/ (git-ignored); kernels.lock.json records the exact
URLs, sizes, and SHA-256 hashes for reproducibility.

CONCURRENCY. Several processes (pytest-xdist workers, parallel scripts) may find
the cache empty at the same moment. A kernel therefore becomes visible under its
final name only by an atomic rename of a fully downloaded, checksum-verified
temporary file, and downloads are serialised by a cross-process lock file. A
reader can never furnish a partially written kernel: before this, a worker that
found another worker's half-written de440s.bsp furnished it and failed with
SPICE(DAFBEGGTEND) or SPICE(NOLOADEDFILES).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
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

#: A download lock whose file has not been touched for this long belongs to a dead process.
STALE_LOCK_SECONDS = 120.0
#: Give up waiting for another process's download after this long.
LOCK_WAIT_SECONDS = 1800.0

_furnished = False
_verified: set[tuple[str, int, float]] = set()


class KernelIntegrityError(RuntimeError):
    """A downloaded kernel does not match its pinned size and SHA-256."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned() -> dict:
    try:
        with open(LOCK, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _is_valid(path: str, pin: dict | None) -> bool:
    """True when the kernel exists and matches its pin (size, then SHA-256)."""
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return False
    if pin is None:
        return stat.st_size > 0
    if stat.st_size != pin["bytes"]:
        return False
    key = (path, stat.st_size, stat.st_mtime)
    if key in _verified:
        return True
    if _sha256(path) != pin["sha256"]:
        return False
    _verified.add(key)
    return True


def _acquire_download_lock() -> str:
    path = os.path.join(KERNEL_DIR, ".download.lock")
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = True
        else:
            held = False
        if not held:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return path
        try:
            age = time.time() - os.stat(path).st_mtime
        except FileNotFoundError:
            continue  # released between the two calls
        if age > STALE_LOCK_SECONDS:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)  # its holder stopped heartbeating
            continue
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for the kernel download lock {path}")
        time.sleep(0.1)


def _download(name: str, url: str, pin: dict | None, heartbeat: str) -> None:
    """Stream to a private temporary file, verify, then atomically install."""
    dest = os.path.join(KERNEL_DIR, name)
    tmp = f"{dest}.part-{os.getpid()}-{time.monotonic_ns()}"
    digest, size = hashlib.sha256(), 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ariadne/0.5"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block)
                digest.update(block)
                size += len(block)
                os.utime(heartbeat)  # keep the lock fresh for the whole transfer
        if pin is not None and (size != pin["bytes"] or digest.hexdigest() != pin["sha256"]):
            raise KernelIntegrityError(
                f"{name}: downloaded {size} bytes sha256={digest.hexdigest()}, "
                f"pinned {pin['bytes']} bytes sha256={pin['sha256']}"
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_lock(lock: dict) -> None:
    tmp = f"{LOCK}.part-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(lock, f, indent=2)
    os.replace(tmp, LOCK)


def download_kernels(force: bool = False) -> dict:
    """Download any missing or invalid kernels and return the lock.

    Kernels are verified against the committed ``kernels.lock.json`` pin. A pin is
    written only for a kernel that has none; an existing pin is never overwritten
    by whatever bytes happened to arrive. ``force`` re-downloads every kernel.
    """
    os.makedirs(KERNEL_DIR, exist_ok=True)
    heartbeat = _acquire_download_lock()
    try:
        pinned = _pinned()
        lock = dict(pinned)
        for name, url in KERNELS:
            dest = os.path.join(KERNEL_DIR, name)
            pin = pinned.get(name)
            if force or not _is_valid(dest, pin):
                _download(name, url, pin, heartbeat)
            if pin is None:
                lock[name] = {"url": url, "sha256": _sha256(dest), "bytes": os.path.getsize(dest)}
        if lock != pinned:
            _write_lock(lock)
        return lock
    finally:
        os.unlink(heartbeat)


def ensure_kernels() -> None:
    pinned = _pinned()
    if not all(_is_valid(os.path.join(KERNEL_DIR, n), pinned.get(n)) for n, _ in KERNELS):
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
