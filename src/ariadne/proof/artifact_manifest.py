"""Artifact integrity manifest for reproducible Ariadne proof bundles."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .closure import stable_hash


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_artifact_manifest(
    root: str | Path,
    *,
    include_patterns: Iterable[str] = ("*.json", "*.csv", "*.md", "*.png"),
) -> dict:
    base = Path(root)
    entries = []
    seen = set()
    for pattern in include_patterns:
        for path in sorted(base.rglob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            entries.append(ManifestEntry(
                path=str(path.relative_to(base)).replace("\\", "/"),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
            ))
    payload = {
        "schema": "ariadne.artifact_integrity_manifest.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(base),
        "file_count": len(entries),
        "total_size_bytes": sum(e.size_bytes for e in entries),
        "entries": [asdict(e) for e in entries],
    }
    payload["certificate_hash"] = stable_hash({k: v for k, v in payload.items() if k != "created_utc"})
    return payload


def write_artifact_manifest(manifest: dict, outdir: str | Path) -> dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "artifact_manifest.json"
    md_path = out / "artifact_manifest.md"
    json_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Ariadne Artifact Integrity Manifest",
        "",
        f"- files: {manifest['file_count']}",
        f"- total_size_bytes: {manifest['total_size_bytes']}",
        f"- certificate_hash: `{manifest['certificate_hash']}`",
        "",
        "| path | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for entry in manifest["entries"][:200]:
        lines.append(f"| `{entry['path']}` | {entry['size_bytes']} | `{entry['sha256'][:16]}` |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
