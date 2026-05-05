from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

from .aho_filter_store import read_filter_words as read_filter_store_words
from .jsonio import read_json, write_json

LOGGER = logging.getLogger("anal_russia_klinik.aho_compact")
MARKER = '"by_search_word"'


def normalize_host_word(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def searchable_length(value: str) -> int:
    normalized = normalize_host_word(value)
    return sum(1 for char in normalized if char.isalnum())


def source_match_count(entry: dict[str, Any]) -> int:
    if "match_count" in entry:
        return int(entry.get("match_count", 0) or 0)
    return sum(int(host.get("count", 0) or 0) for host in entry.get("host_words", []))


def keep_existing_search_word(entry: dict[str, Any]) -> bool:
    match_count = source_match_count(entry)
    if match_count <= 0:
        return False
    if str(entry.get("source", "")) == "mediq":
        canonical = normalize_host_word(entry.get("canonical", ""))
        search_word = normalize_host_word(entry.get("search_word", ""))
        if canonical and search_word and canonical != search_word:
            return False
    search_len = searchable_length(str(entry.get("search_word", "")))
    if search_len >= 3:
        return True
    canonical_len = searchable_length(str(entry.get("canonical", "")))
    return search_len == 2 and canonical_len == 2 and normalize_host_word(entry.get("search_word", "")) == normalize_host_word(entry.get("canonical", ""))


def configure_logging(verbose: bool = True) -> None:
    if logging.getLogger().handlers:
        return
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _read_until_marker(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    buffer = ""
    total = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"marker_not_found path={path} marker={MARKER}")
            buffer += chunk
            total += len(chunk)
            position = buffer.find(MARKER)
            if position >= 0:
                return buffer[:position], total - len(buffer) + position
            if len(buffer) > len(MARKER) + chunk_size:
                buffer = buffer[-(len(MARKER) + chunk_size) :]


def read_source_header(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    prefix, _ = _read_until_marker(source)
    compact_prefix = prefix.rstrip()
    if compact_prefix.endswith(","):
        compact_prefix = compact_prefix[:-1]
    try:
        return json.loads(compact_prefix + ', "by_search_word": []}')
    except JSONDecodeError as exc:
        LOGGER.warning("source_header_parse_failed path=%s error=%s", source, exc)
        return {}


def iter_search_word_entries(path: str | Path, chunk_size: int = 1024 * 1024) -> Iterable[dict[str, Any]]:
    source = Path(path)
    decoder = json.JSONDecoder()
    with source.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        eof = False
        found_array = False

        def fill() -> bool:
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        while not found_array:
            if MARKER in buffer:
                marker_index = buffer.find(MARKER)
                bracket_index = buffer.find("[", marker_index)
                while bracket_index < 0 and fill():
                    bracket_index = buffer.find("[", marker_index)
                if bracket_index < 0:
                    raise ValueError(f"array_start_not_found path={source} marker={MARKER}")
                buffer = buffer[bracket_index + 1 :]
                found_array = True
                break
            if not fill():
                raise ValueError(f"marker_not_found path={source} marker={MARKER}")
            if len(buffer) > len(MARKER) + chunk_size:
                buffer = buffer[-(len(MARKER) + chunk_size) :]

        while True:
            buffer = buffer.lstrip()
            while buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                item, end_index = decoder.raw_decode(buffer)
            except JSONDecodeError:
                if eof:
                    raise
                fill()
                continue
            if not isinstance(item, dict):
                raise ValueError(f"invalid_by_search_word_item type={type(item).__name__}")
            yield item
            buffer = buffer[end_index:]


def _compact_host_words(host_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for host in host_words:
        host_word = str(host.get("host_word", ""))
        count = int(host.get("count", 0) or 0)
        rows.append(
            {
                "host_word": host_word,
                "inside_word": bool(host.get("inside_word", False)),
                "count": count,
            }
        )
    rows.sort(key=lambda row: (-row["count"], normalize_host_word(row["host_word"])))
    return rows


def compact_search_word_entry(entry: dict[str, Any]) -> dict[str, Any]:
    host_words = _compact_host_words(entry.get("host_words", []))
    match_count = sum(host["count"] for host in host_words)
    inside_count = sum(host["count"] for host in host_words if host["inside_word"])
    return {
        "source": entry.get("source", ""),
        "term_id": entry.get("term_id", ""),
        "canonical": entry.get("canonical", ""),
        "search_word": entry.get("search_word", ""),
        "normalized_search_word": entry.get("normalized_search_word", ""),
        "match_count": match_count,
        "inside_word_match_count": inside_count,
        "host_words_count": len(host_words),
        "inside_host_words_count": sum(1 for host in host_words if host["inside_word"]),
        "host_words": host_words,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "search_word_count": len(rows),
        "matched_search_word_count": sum(1 for row in rows if row["match_count"]),
        "inside_matched_search_word_count": sum(1 for row in rows if row["inside_word_match_count"]),
        "total_match_count": sum(row["match_count"] for row in rows),
        "total_inside_word_match_count": sum(row["inside_word_match_count"] for row in rows),
        "total_host_word_count": sum(row["host_words_count"] for row in rows),
        "total_inside_host_word_count": sum(row["inside_host_words_count"] for row in rows),
    }


def _merge_host_words(host_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for host in host_rows:
        key = normalize_host_word(host.get("host_word", ""))
        if not key:
            continue
        count = int(host.get("count", 0) or 0)
        current = merged.get(key)
        candidate = {
            "host_word": str(host.get("host_word", "")),
            "inside_word": bool(host.get("inside_word", False)),
            "count": count,
        }
        if current is None or count > int(current.get("count", 0) or 0):
            merged[key] = candidate
        elif current is not None and candidate["inside_word"]:
            current["inside_word"] = True
    rows = list(merged.values())
    rows.sort(key=lambda row: (-row["count"], normalize_host_word(row["host_word"])))
    return rows


def _recompute_counts(row: dict[str, Any]) -> dict[str, Any]:
    hosts = list(row.get("host_words", []))
    row["match_count"] = sum(int(host.get("count", 0) or 0) for host in hosts)
    row["inside_word_match_count"] = sum(int(host.get("count", 0) or 0) for host in hosts if host.get("inside_word"))
    row["host_words_count"] = len(hosts)
    row["inside_host_words_count"] = sum(1 for host in hosts if host.get("inside_word"))
    return row


def _prefer_display_row(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    current_exact = normalize_host_word(current.get("canonical", "")) == normalize_host_word(current.get("search_word", ""))
    candidate_exact = normalize_host_word(candidate.get("canonical", "")) == normalize_host_word(candidate.get("search_word", ""))
    if candidate_exact and not current_exact:
        return candidate
    if str(candidate.get("search_word", "")) == str(candidate.get("canonical", "")):
        return candidate
    return current


def deduplicate_compact_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("source", "")),
            str(row.get("term_id", "")),
            normalize_host_word(row.get("normalized_search_word") or row.get("search_word", "")),
        )
        if key not in grouped:
            grouped[key] = dict(row)
            grouped[key]["host_words"] = _merge_host_words(grouped[key].get("host_words", []))
            _recompute_counts(grouped[key])
            continue
        display = _prefer_display_row(grouped[key], row)
        merged_hosts = _merge_host_words([*grouped[key].get("host_words", []), *row.get("host_words", [])])
        grouped[key] = {**grouped[key], **{field: display.get(field) for field in ("source", "term_id", "canonical", "search_word", "normalized_search_word")}}
        grouped[key]["host_words"] = merged_hosts
        _recompute_counts(grouped[key])
    return list(grouped.values())


def build_compact_report(source_path: str | Path) -> dict[str, Any]:
    source = Path(source_path)
    header = read_source_header(source)
    rows = []
    LOGGER.info("compact_build_start source=%s bytes=%s", source, source.stat().st_size)
    for index, entry in enumerate(iter_search_word_entries(source), start=1):
        if not keep_existing_search_word(entry):
            if index % 1000 == 0:
                LOGGER.info("compact_progress entries=%s kept=%s", index, len(rows))
            continue
        rows.append(compact_search_word_entry(entry))
        if index % 1000 == 0:
            LOGGER.info("compact_progress entries=%s kept=%s", index, len(rows))
    rows = deduplicate_compact_rows(rows)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_schema_version": header.get("schema_version"),
        "source_generated_at": header.get("generated_at"),
        "source_document_count": header.get("document_count"),
        "source_workers": header.get("workers"),
        "source_summary": header.get("summary", {}),
        "summary": _summary(rows),
        "by_search_word": rows,
    }
    LOGGER.info("compact_build_done entries=%s matches=%s", len(rows), report["summary"]["total_match_count"])
    return report


def write_compact_report(source_path: str | Path, output_path: str | Path, *, indent: int = 2) -> dict[str, Any]:
    report = build_compact_report(source_path)
    output = Path(output_path)
    LOGGER.info("compact_write_start output=%s", output)
    write_json(output, report, indent=indent)
    LOGGER.info("compact_write_done output=%s bytes=%s", output, output.stat().st_size)
    return report


def read_filter_words(path: str | Path) -> set[str]:
    return read_filter_store_words(path)


def write_filter_words(path: str | Path, words: Iterable[str]) -> dict[str, Any]:
    normalized = sorted({normalize_host_word(word) for word in words if normalize_host_word(word)})
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "host_word_count": len(normalized),
        "host_words": normalized,
    }
    write_json(path, payload, indent=2)
    return payload


def apply_host_word_filters(compact_report: dict[str, Any], filter_words: Iterable[str]) -> dict[str, Any]:
    filters = {normalize_host_word(word) for word in filter_words if normalize_host_word(word)}
    rows = []
    removed_match_count = 0
    removed_host_word_count = 0
    for entry in compact_report.get("by_search_word", []):
        kept_hosts = []
        for host in entry.get("host_words", []):
            if normalize_host_word(host.get("host_word", "")) in filters:
                removed_match_count += int(host.get("count", 0) or 0)
                removed_host_word_count += 1
            else:
                kept_hosts.append(dict(host))
        row = dict(entry)
        row["host_words"] = kept_hosts
        row["match_count"] = sum(int(host["count"]) for host in kept_hosts)
        row["inside_word_match_count"] = sum(int(host["count"]) for host in kept_hosts if host["inside_word"])
        row["host_words_count"] = len(kept_hosts)
        row["inside_host_words_count"] = sum(1 for host in kept_hosts if host["inside_word"])
        rows.append(row)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_compact_report": compact_report.get("source_report", compact_report.get("generated_at", "")),
        "filter_host_word_count": len(filters),
        "filter_host_words": sorted(filters),
        "removed_match_count": removed_match_count,
        "removed_host_word_count": removed_host_word_count,
        "summary": _summary(rows),
        "by_search_word": rows,
    }
    return report


def write_filtered_report(
    compact_path: str | Path,
    filter_path: str | Path,
    output_path: str | Path,
    *,
    indent: int = 2,
) -> dict[str, Any]:
    compact_report = read_json(compact_path)
    filter_words = read_filter_words(filter_path)
    report = apply_host_word_filters(compact_report, filter_words)
    output = Path(output_path)
    LOGGER.info(
        "filtered_write_start output=%s filters=%s removed_matches=%s",
        output,
        len(filter_words),
        report["removed_match_count"],
    )
    write_json(output, report, indent=indent)
    LOGGER.info("filtered_write_done output=%s bytes=%s", output, output.stat().st_size)
    return report
