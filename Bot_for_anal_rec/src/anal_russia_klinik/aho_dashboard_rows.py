from __future__ import annotations

from typing import Any

from .aho_compact import normalize_host_word


def row_payload(row: dict[str, Any], hosts: list[dict[str, Any]], host_limit: int) -> dict[str, Any]:
    item = {key: row.get(key) for key in ("source", "term_id", "canonical", "search_word", "normalized_search_word")}
    item["group_type"] = "drug"
    item["match_count"] = sum(int(host.get("count", 0) or 0) for host in hosts)
    item["inside_word_match_count"] = sum(int(host.get("count", 0) or 0) for host in hosts if host.get("inside_word"))
    item["host_words_count"] = len(hosts)
    item["shown_host_words_count"] = min(len(hosts), host_limit)
    item["matching_host_words_count"] = len(hosts)
    item["host_words"] = hosts[:host_limit]
    return item


def row_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    return (-int(item.get("match_count", 0) or 0), normalize_host_word(item.get("search_word", "")))
