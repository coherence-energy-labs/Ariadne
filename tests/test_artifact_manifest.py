import json

from ariadne.proof.artifact_manifest import build_artifact_manifest, write_artifact_manifest


def test_artifact_manifest_hashes_files_and_writes_outputs(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "a.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (root / "b.md").write_text("hello", encoding="utf-8")

    manifest = build_artifact_manifest(root)
    outputs = write_artifact_manifest(manifest, tmp_path / "out")

    assert manifest["file_count"] == 2
    assert all(len(entry["sha256"]) == 64 for entry in manifest["entries"])
    assert manifest["certificate_hash"]
    assert "artifact_manifest.json" in outputs["json"]
