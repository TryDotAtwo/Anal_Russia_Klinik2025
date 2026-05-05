from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json


LOCATION_FIELDS = (
    "document_id",
    "document_title",
    "document_link",
    "char_start",
    "char_end",
    "word_start",
    "word_end",
    "matched_text",
    "page",
    "section",
)


def location_key(occurrence: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(occurrence.get("document_id", "")),
        int(occurrence.get("char_start", 0) or 0),
        int(occurrence.get("char_end", 0) or 0),
    )


def location_id(key: tuple[str, int, int]) -> str:
    digest = hashlib.sha1(f"{key[0]}|{key[1]}|{key[2]}".encode("utf-8")).hexdigest()[:16]
    return f"loc:{digest}"


def _location_payload(occurrence: dict[str, Any]) -> dict[str, Any]:
    return {field: occurrence.get(field) for field in LOCATION_FIELDS}


def _match_payload(row: dict[str, Any], host: dict[str, Any], occurrence: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row.get("source", ""),
        "term_id": row.get("term_id", ""),
        "canonical": row.get("canonical", ""),
        "search_word": row.get("search_word", ""),
        "normalized_search_word": row.get("normalized_search_word", ""),
        "host_word": host.get("host_word", ""),
        "inside_word": bool(host.get("inside_word", False)),
        "occurrence": dict(occurrence),
    }


def _unique_sorted(values: list[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _finalize_group(key: tuple[str, int, int], group: dict[str, Any]) -> dict[str, Any]:
    matches = sorted(
        group["matches"],
        key=lambda item: (
            str(item.get("source", "")),
            str(item.get("canonical", "")).casefold(),
            str(item.get("search_word", "")).casefold(),
            str(item.get("host_word", "")).casefold(),
        ),
    )
    return {
        "location_id": location_id(key),
        **group["location"],
        "match_count": len(matches),
        "duplicate_match_count": max(0, len(matches) - 1),
        "source_count": len({item["source"] for item in matches}),
        "term_count": len({(item["source"], item["term_id"], item["normalized_search_word"]) for item in matches}),
        "inside_word": any(item["inside_word"] for item in matches),
        "sources": _unique_sorted([item["source"] for item in matches]),
        "canonicals": _unique_sorted([item["canonical"] for item in matches]),
        "search_words": _unique_sorted([item["search_word"] for item in matches]),
        "host_words": _unique_sorted([item["host_word"] for item in matches]),
        "matches": matches,
    }


def build_location_groups(report: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in report.get("by_search_word", []):
        for host in row.get("host_words", []):
            for occurrence in host.get("occurrences", []):
                key = location_key(occurrence)
                group = groups.setdefault(key, {"location": _location_payload(occurrence), "matches": []})
                group["matches"].append(_match_payload(row, host, occurrence))
    rows = [_finalize_group(key, group) for key, group in groups.items()]
    return sorted(rows, key=lambda item: (str(item.get("document_id", "")), int(item.get("char_start", 0) or 0), int(item.get("char_end", 0) or 0)))


def _summary(groups: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "location_group_count": len(groups),
        "total_grouped_match_count": sum(int(group.get("match_count", 0) or 0) for group in groups),
        "duplicate_location_group_count": sum(1 for group in groups if int(group.get("match_count", 0) or 0) > 1),
        "duplicate_match_count": sum(int(group.get("duplicate_match_count", 0) or 0) for group in groups),
        "inside_location_group_count": sum(1 for group in groups if group.get("inside_word")),
    }


def build_location_group_report(source_path: str | Path) -> dict[str, Any]:
    source = Path(source_path)
    report = read_json(source)
    groups = build_location_groups(report)
    summary = _summary(groups)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source),
        "source_generated_at": report.get("generated_at"),
        "source_summary": report.get("summary", {}),
        "dedup_key": ["document_id", "char_start", "char_end"],
        "llm_review_unit": "by_location[*]",
        "summary": summary,
        "by_location": groups,
    }


def write_location_group_report(source_path: str | Path, output_path: str | Path, *, indent: int = 2) -> dict[str, Any]:
    report = build_location_group_report(source_path)
    write_json(output_path, report, indent=indent)
    return report
