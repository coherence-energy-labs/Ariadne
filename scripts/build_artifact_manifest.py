"""Build a hash manifest for Ariadne benchmark/proof artifacts."""
from __future__ import annotations

import argparse

from ariadne.proof import build_artifact_manifest, write_artifact_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/benchmarks")
    parser.add_argument("--out-dir", default="data/benchmarks/artifact_integrity")
    args = parser.parse_args()
    manifest = build_artifact_manifest(args.root)
    outputs = write_artifact_manifest(manifest, args.out_dir)
    print(f"files={manifest['file_count']}")
    print(f"certificate_hash={manifest['certificate_hash']}")
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
