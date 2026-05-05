from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .jsonio import read_json, write_json

DEFAULT_SHARD_SIZE = 500


def normalize_host_word(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def default_parts_dir(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f"{target.stem}_parts")


def _payload_words(data: Any) -> list[str]:
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        return [str(item) for item in data.get("host_words", [])]
    return []


def _read_json_first(path: Path) -> Any:
    try:
        return read_json(path)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(path.read_text(encoding="utf-8-sig"))
        return data


def read_merged_filter_words(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    data = _read_json_first(target)
    return {word for word in (normalize_host_word(item) for item in _payload_words(data)) if word}


def read_filter_words(path: str | Path, *, parts_dir: str | Path | None = None) -> set[str]:
    parts = Path(parts_dir) if parts_dir else default_parts_dir(path)
    shard_files = sorted(parts.glob("host_words_*.json")) if parts.exists() else []
    if not shard_files:
        return read_merged_filter_words(path)
    words: set[str] = set()
    for shard in shard_files:
        words.update(word for word in (normalize_host_word(item) for item in _payload_words(_read_json_first(shard))) if word)
    return words


def shard_filter_words(
    path: str | Path,
    words: Iterable[str],
    *,
    parts_dir: str | Path | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    normalized = sorted({word for word in (normalize_host_word(item) for item in words) if word})
    parts = Path(parts_dir) if parts_dir else default_parts_dir(path)
    parts.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    shard_count = 0
    for index in range(0, len(normalized), shard_size):
        shard_words = normalized[index : index + shard_size]
        shard_path = parts / f"host_words_{shard_count:04d}.json"
        write_json(
            shard_path,
            {
                "schema_version": 1,
                "updated_at": now,
                "shard_index": shard_count,
                "shard_size": shard_size,
                "host_word_count": len(shard_words),
                "host_words": shard_words,
            },
            indent=2,
        )
        shard_count += 1
    for stale in sorted(parts.glob("host_words_*.json")):
        suffix = stale.stem.rsplit("_", 1)[-1]
        if suffix.isdigit() and int(suffix) >= shard_count:
            stale.unlink()
    index_payload = {
        "schema_version": 1,
        "updated_at": now,
        "shard_size": shard_size,
        "shard_count": shard_count,
        "host_word_count": len(normalized),
        "parts_dir": str(parts),
    }
    write_json(parts / "index.json", index_payload, indent=2)
    return index_payload


def merge_filter_words(
    path: str | Path,
    *,
    parts_dir: str | Path | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    words = sorted(read_filter_words(path, parts_dir=parts_dir))
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "shard_size": shard_size,
        "host_word_count": len(words),
        "host_words": words,
    }
    write_json(path, payload, indent=2)
    return payload


def update_filter_words(
    path: str | Path,
    values: Iterable[str],
    *,
    enabled: bool,
    parts_dir: str | Path | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    words = read_filter_words(path, parts_dir=parts_dir)
    values_set = {word for word in (normalize_host_word(item) for item in values) if word}
    before = len(words)
    if enabled:
        words.update(values_set)
    else:
        words.difference_update(values_set)
    if len(words) == before:
        parts = Path(parts_dir) if parts_dir else default_parts_dir(path)
        return {
            "schema_version": 1,
            "shard_size": shard_size,
            "shard_count": len(list(parts.glob("host_words_*.json"))) if parts.exists() else 0,
            "host_word_count": len(words),
            "before_count": before,
            "after_count": len(words),
            "changed_count": 0,
        }
    shard_info = shard_filter_words(path, words, parts_dir=parts_dir, shard_size=shard_size)
    return {
        **shard_info,
        "before_count": before,
        "after_count": len(words),
        "changed_count": abs(len(words) - before),
    }


def page_filter_words(path: str | Path, *, q: str = "", offset: int = 0, limit: int = 200) -> dict[str, Any]:
    query = normalize_host_word(q)
    words = sorted(read_filter_words(path))
    if query:
        words = [word for word in words if query in word]
    page = words[offset : offset + limit]
    return {"total": len(words), "offset": offset, "limit": limit, "host_words": page}
