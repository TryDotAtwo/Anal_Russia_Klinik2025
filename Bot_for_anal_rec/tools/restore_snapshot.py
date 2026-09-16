"""Restore the versioned input snapshot without network access or paid API calls."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside project: {relative}")
    return path


def restore(root: Path = ROOT, *, check: bool = False) -> list[str]:
    manifest = json.loads((root / "data/snapshots/manifest.json").read_text(encoding="utf-8"))
    messages = []
    for item in manifest["files"]:
        target = contained(root, item["path"])
        archive = contained(root, item["archive"])
        if target.exists():
            if target.stat().st_size != item["size"] or digest(target) != item["sha256"]:
                raise ValueError(f"Existing file differs; preserved without changes: {target}")
            messages.append(f"verified {item['path']}")
            continue
        if check:
            raise FileNotFoundError(f"Run py tools/restore_snapshot.py first: {target}")
        if digest(archive) != item["archive_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                checksum = hashlib.sha256()
                size = 0
                with gzip.open(archive, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > item["size"]:
                            raise ValueError(f"Expanded size exceeds manifest: {archive}")
                        checksum.update(chunk)
                        output.write(chunk)
            if size != item["size"] or checksum.hexdigest() != item["sha256"]:
                raise ValueError(f"Restored file checksum mismatch: {target}")
            # Do not overwrite a file created by another process during decompression.
            with target.open("xb") as output, temporary.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            messages.append(f"restored {item['path']}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify restored files without writing")
    args = parser.parse_args()
    for message in restore(check=args.check):
        print(message)


if __name__ == "__main__":
    main()
