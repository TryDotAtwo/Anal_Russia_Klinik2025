from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .jsonio import read_json
from .models import FilterDecision
from .text import normalize_query

ALLOWED_ACTIONS = {"keep", "reject", "review"}


def _confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decision_from_dict(row: dict[str, Any]) -> FilterDecision:
    action = str(row.get("action", "review")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        action = "review"
    return FilterDecision(
        term=str(row.get("term", "")).strip(),
        host_word=str(row.get("host_word", "")).strip(),
        action=action,
        reason=str(row.get("reason", "")).strip(),
        confidence=_confidence(row.get("confidence")),
        created_at=str(row.get("created_at", "")).strip(),
    )


def load_filter_dictionary(path: str | Path | None) -> list[FilterDecision]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return [decision_from_dict(row) for row in csv.DictReader(handle)]
    data = read_json(source)
    rows = data.get("filters", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [decision_from_dict(row) for row in rows if isinstance(row, dict)]


def build_filter_index(decisions: list[FilterDecision]) -> dict[tuple[str, str], FilterDecision]:
    index: dict[tuple[str, str], FilterDecision] = {}
    for decision in decisions:
        index[(normalize_query(decision.term), normalize_query(decision.host_word))] = decision
    return index


def match_decision(match: dict[str, Any], index: dict[tuple[str, str], FilterDecision]) -> FilterDecision:
    term = normalize_query(str(match.get("canonical", "")))
    host = normalize_query(str(match.get("word_text", "")))
    matched = normalize_query(str(match.get("matched_text", "")))
    source = normalize_query(str(match.get("source", "")))
    return (
        index.get((term, host))
        or index.get((term, matched))
        or index.get((term, "*"))
        or index.get((source, host))
        or index.get(("*", host))
        or FilterDecision(str(match.get("canonical", "")), str(match.get("word_text", "")), "review", "no_manual_rule")
    )


def apply_manual_filters(
    matches: list[dict[str, Any]],
    decisions: list[FilterDecision],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = build_filter_index(decisions)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for match in matches:
        decision = match_decision(match, index)
        item = {
            "raw_match_id": match["raw_match_id"],
            "filter_decision": decision.to_dict(),
            "match": match,
        }
        if decision.action == "reject":
            rejected.append(item)
        else:
            kept.append(item)
    return kept, rejected
