import gzip
import hashlib
import json

import pytest

from tools.restore_snapshot import restore


def snapshot(tmp_path):
    content = b'{"clinical_recommendations": []}\n'
    folder = tmp_path / "data/snapshots"
    folder.mkdir(parents=True)
    archive = folder / "input.json.gz"
    archive.write_bytes(gzip.compress(content))
    item = {"path": "data/input/input.json", "archive": "data/snapshots/input.json.gz",
            "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps({"files": [item]}))
    return content, archive, tmp_path / item["path"]


def test_restore_and_check_without_overwriting(tmp_path):
    content, _, target = snapshot(tmp_path)
    restore(tmp_path)
    assert target.read_bytes() == content
    assert restore(tmp_path, check=True) == ["verified data/input/input.json"]
    target.write_bytes(b"manual change")
    with pytest.raises(ValueError, match="preserved"):
        restore(tmp_path)
    assert target.read_bytes() == b"manual change"


def test_corrupt_archive_is_rejected_before_writing(tmp_path):
    _, archive, target = snapshot(tmp_path)
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Archive checksum"):
        restore(tmp_path)
    assert not target.exists()


def test_manifest_cannot_write_outside_project(tmp_path):
    snapshot(tmp_path)
    manifest = tmp_path / "data/snapshots/manifest.json"
    data = json.loads(manifest.read_text())
    data["files"][0]["path"] = "../outside.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside project"):
        restore(tmp_path)
