"""Kernel cache integrity under concurrency, without the network.

On a CI cache miss every pytest-xdist worker found data/kernels empty at once. The
old loader wrote each download straight to its final name and treated "the file
exists" as "the kernel is ready", so workers furnished another worker's half-written
de440s.bsp: SPICE(DAFBEGGTEND) and SPICE(NOLOADEDFILES). These tests serve a fake
kernel through a slow fake URL so the race window is wide and deterministic.
"""

import hashlib
import json
import os
import threading
import time
import urllib.request

import pytest

from ariadne.data import kernels as K

PAYLOAD = bytes(range(256)) * 2048  # 512 KiB
PIN = {"bytes": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}


class _SlowResponse:
    def __init__(self, payload, delay):
        self._payload, self._pos, self._delay = payload, 0, delay

    def read(self, size=-1):
        time.sleep(self._delay)
        size = 64 * 1024 if size is None or size < 0 else min(size, 64 * 1024)
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    kdir = tmp_path / "kernels"
    kdir.mkdir()
    lock = kdir / "kernels.lock.json"
    url = "https://example.invalid/fake.bsp"
    lock.write_text(json.dumps({"fake.bsp": {"url": url, **PIN}}, indent=2))
    monkeypatch.setattr(K, "KERNEL_DIR", str(kdir))
    monkeypatch.setattr(K, "LOCK", str(lock))
    monkeypatch.setattr(K, "KERNELS", [("fake.bsp", url)])
    served = {"calls": 0, "payload": PAYLOAD}

    def fake_urlopen(req, timeout=None):
        served["calls"] += 1
        return _SlowResponse(served["payload"], delay=0.01)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return kdir, lock, served


def test_concurrent_cold_cache_never_exposes_a_partial_kernel(fake_cache):
    kdir, _, served = fake_cache
    dest = kdir / "fake.bsp"
    seen, errors = [], []

    def worker():
        try:
            K.ensure_kernels()
            seen.append(hashlib.sha256(dest.read_bytes()).hexdigest())
        except Exception as exc:  # recorded, asserted below
            errors.append(exc)

    partial_sightings = []

    def watcher():
        # A reader must never observe the final name holding fewer bytes than pinned.
        while len(seen) + len(errors) < 4:
            if dest.exists() and dest.stat().st_size != PIN["bytes"]:
                partial_sightings.append(dest.stat().st_size)
            time.sleep(0.001)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    spy = threading.Thread(target=watcher)
    spy.start()
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join(60)
    spy.join(60)
    assert errors == []
    assert seen == [PIN["sha256"]] * 4
    assert partial_sightings == []
    assert served["calls"] == 1, "workers must share one download, not race four"
    assert sorted(p.name for p in kdir.iterdir()) == ["fake.bsp", "kernels.lock.json"]


def test_truncated_kernel_left_by_a_dead_writer_is_replaced(fake_cache):
    kdir, _, served = fake_cache
    (kdir / "fake.bsp").write_bytes(PAYLOAD[:1000])
    K.ensure_kernels()
    assert (kdir / "fake.bsp").read_bytes() == PAYLOAD
    assert served["calls"] == 1


def test_same_size_corrupt_kernel_is_detected_by_checksum(fake_cache):
    kdir, _, served = fake_cache
    (kdir / "fake.bsp").write_bytes(bytes(len(PAYLOAD)))
    K.ensure_kernels()
    assert (kdir / "fake.bsp").read_bytes() == PAYLOAD
    assert served["calls"] == 1


def test_download_that_does_not_match_its_pin_is_refused_and_not_installed(fake_cache):
    kdir, lock, served = fake_cache
    served["payload"] = PAYLOAD[:-1] + b"\x00"
    before = lock.read_text()
    with pytest.raises(K.KernelIntegrityError):
        K.ensure_kernels()
    assert not (kdir / "fake.bsp").exists()
    assert lock.read_text() == before, "a bad download must never re-pin the lock"
    assert sorted(p.name for p in kdir.iterdir()) == ["kernels.lock.json"]


def test_valid_cache_makes_no_request(fake_cache):
    kdir, _, served = fake_cache
    (kdir / "fake.bsp").write_bytes(PAYLOAD)
    K.ensure_kernels()
    K.ensure_kernels()
    assert served["calls"] == 0


def test_stale_download_lock_from_a_dead_process_is_broken(fake_cache):
    kdir, _, _ = fake_cache
    stale = kdir / ".download.lock"
    stale.write_text("99999")
    old = time.time() - 10 * K.STALE_LOCK_SECONDS
    os.utime(stale, (old, old))
    K.ensure_kernels()
    assert (kdir / "fake.bsp").read_bytes() == PAYLOAD
    assert not stale.exists()
