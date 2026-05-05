from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aho_compact import iter_search_word_entries, keep_existing_search_word, normalize_host_word, read_source_header
from .aho_filter_store import merge_filter_words, read_filter_words

LOGGER = logging.getLogger("anal_russia_klinik.aho_detailed_filter")


def _empty_summary() -> dict[str, int]:
    return {
        "search_word_count": 0,
        "matched_search_word_count": 0,
        "inside_matched_search_word_count": 0,
        "total_match_count": 0,
        "total_inside_word_match_count": 0,
        "total_host_word_count": 0,
        "total_inside_host_word_count": 0,
    }


def _add_row_to_summary(summary: dict[str, int], row: dict[str, Any]) -> None:
    summary["search_word_count"] += 1
    summary["matched_search_word_count"] += 1 if row["match_count"] else 0
    summary["inside_matched_search_word_count"] += 1 if row["inside_word_match_count"] else 0
    summary["total_match_count"] += row["match_count"]
    summary["total_inside_word_match_count"] += row["inside_word_match_count"]
    summary["total_host_word_count"] += row["host_words_count"]
    summary["total_inside_host_word_count"] += row["inside_host_words_count"]


def _filter_hosts(hosts: list[dict[str, Any]], filters: set[str]) -> tuple[list[dict[str, Any]], int, int]:
    kept: list[dict[str, Any]] = []
    removed_count = 0
    removed_matches = 0
    for host in hosts:
        if normalize_host_word(host.get("host_word", "")) in filters:
            removed_count += 1
            removed_matches += int(host.get("count", 0) or 0)
            continue
        kept.append(host)
    return kept, removed_count, removed_matches


def filter_detailed_entry(entry: dict[str, Any], filters: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    kept_hosts, removed_hosts, removed_matches = _filter_hosts(list(entry.get("host_words", [])), filters)
    kept_inside, removed_inside_hosts, removed_inside_matches = _filter_hosts(list(entry.get("inside_host_words", [])), filters)
    row = dict(entry)
    row["host_words"] = kept_hosts
    row["inside_host_words"] = kept_inside
    row["match_count"] = sum(int(host.get("count", 0) or 0) for host in kept_hosts)
    row["inside_word_match_count"] = sum(int(host.get("count", 0) or 0) for host in kept_inside)
    row["host_words_count"] = len(kept_hosts)
    row["inside_host_words_count"] = len(kept_inside)
    stats = {
        "removed_host_word_count": removed_hosts,
        "removed_inside_host_word_count": removed_inside_hosts,
        "removed_match_count": removed_matches,
        "removed_inside_word_match_count": removed_inside_matches,
    }
    return row, stats


def _write_json_row(handle: Any, row: dict[str, Any], *, first: bool, indent: int) -> None:
    if not first:
        handle.write(",\n")
    encoded = json.dumps(row, ensure_ascii=False, indent=indent)
    handle.write("    " + encoded.replace("\n", "\n    "))


def _write_final_report(output: Path, header: dict[str, Any], rows_path: Path) -> None:
    with output.open("w", encoding="utf-8") as target:
        target.write("{\n")
        fields = list(header.items())
        for key, value in fields:
            target.write(f"  {json.dumps(key, ensure_ascii=False)}: ")
            target.write(json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\n  "))
            target.write(",\n")
        target.write('  "by_search_word": [\n')
        if rows_path.exists() and rows_path.stat().st_size:
            with rows_path.open("r", encoding="utf-8") as rows_file:
                for chunk in iter(lambda: rows_file.read(1024 * 1024), ""):
                    target.write(chunk)
            target.write("\n")
        target.write("  ]\n")
        target.write("}\n")


def write_filtered_detailed_report(
    source_path: str | Path,
    filter_path: str | Path,
    output_path: str | Path,
    *,
    drop_empty: bool = True,
    indent: int = 2,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    source = Path(source_path)
    filters_file = Path(filter_path)
    output = Path(output_path)
    rows_tmp = output.with_suffix(output.suffix + ".rows.tmp")
    final_tmp = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_tmp.unlink(missing_ok=True)
    final_tmp.unlink(missing_ok=True)

    merge_filter_words(filters_file)
    filters = read_filter_words(filters_file)
    header = read_source_header(source)
    summary = _empty_summary()
    removed = {
        "removed_host_word_count": 0,
        "removed_inside_host_word_count": 0,
        "removed_match_count": 0,
        "removed_inside_word_match_count": 0,
        "skipped_policy_search_word_count": 0,
        "dropped_empty_search_word_count": 0,
        "duplicate_search_word_count": 0,
    }
    seen_keys: set[tuple[str, str, str]] = set()
    written = 0
    scanned = 0
    LOGGER.info("detailed_filter_start source=%s output=%s filters=%s bytes=%s", source, output, len(filters), source.stat().st_size)
    with rows_tmp.open("w", encoding="utf-8") as rows_file:
        first = True
        for scanned, entry in enumerate(iter_search_word_entries(source), start=1):
            if not keep_existing_search_word(entry):
                removed["skipped_policy_search_word_count"] += 1
                continue
            key = (
                str(entry.get("source", "")),
                str(entry.get("term_id", "")),
                normalize_host_word(entry.get("normalized_search_word") or entry.get("search_word", "")),
            )
            if key in seen_keys:
                removed["duplicate_search_word_count"] += 1
                continue
            seen_keys.add(key)
            row, row_removed = filter_detailed_entry(entry, filters)
            for stat_key, value in row_removed.items():
                removed[stat_key] += value
            if drop_empty and row["match_count"] <= 0:
                removed["dropped_empty_search_word_count"] += 1
                continue
            _add_row_to_summary(summary, row)
            _write_json_row(rows_file, row, first=first, indent=indent)
            first = False
            written += 1
            if scanned % progress_interval == 0:
                LOGGER.info("detailed_filter_progress scanned=%s written=%s kept_matches=%s removed_matches=%s", scanned, written, summary["total_match_count"], removed["removed_match_count"])

    result_header = {
        "schema_version": header.get("schema_version", 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_generated_at": header.get("generated_at"),
        "source_search_word_count": header.get("search_word_count"),
        "source_summary": header.get("summary", {}),
        "filter_report": str(filters_file),
        "filter_host_word_count": len(filters),
        "drop_empty": drop_empty,
        "search_word_count": summary["search_word_count"],
        "partial_rows_merged": summary["total_match_count"],
        **removed,
        "chunks": header.get("chunks", []),
        "summary": summary,
    }
    _write_final_report(final_tmp, result_header, rows_tmp)
    final_tmp.replace(output)
    rows_tmp.unlink(missing_ok=True)
    LOGGER.info("detailed_filter_done scanned=%s written=%s output=%s bytes=%s kept_matches=%s removed_matches=%s", scanned, written, output, output.stat().st_size, summary["total_match_count"], removed["removed_match_count"])
    return result_header
