from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .jsonio import write_json


def export_classified_csv(items: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "document_id",
        "document_title",
        "source",
        "canonical",
        "variant",
        "matched_text",
        "char_start",
        "char_end",
        "word_text",
        "word_start",
        "word_end",
        "inside_word",
        "section",
        "filter_action",
        "label",
        "confidence",
        "udd",
        "uur",
        "evidence_level",
        "recommendation_strength",
        "reason_short",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            match = item["match"]
            llm = item.get("llm", {})
            decision = item.get("filter_decision", {})
            writer.writerow(
                {
                    "document_id": match.get("document_id", ""),
                    "document_title": match.get("document_title", ""),
                    "source": match.get("source", ""),
                    "canonical": match.get("canonical", ""),
                    "variant": match.get("variant", ""),
                    "matched_text": match.get("matched_text", ""),
                    "char_start": match.get("char_start", ""),
                    "char_end": match.get("char_end", ""),
                    "word_text": match.get("word_text", ""),
                    "word_start": match.get("word_start", ""),
                    "word_end": match.get("word_end", ""),
                    "inside_word": match.get("inside_word", ""),
                    "section": match.get("section", ""),
                    "filter_action": decision.get("action", ""),
                    "label": llm.get("label", ""),
                    "confidence": llm.get("confidence", ""),
                    "udd": llm.get("udd", ""),
                    "uur": llm.get("uur", ""),
                    "evidence_level": llm.get("evidence_level", ""),
                    "recommendation_strength": llm.get("recommendation_strength", ""),
                    "reason_short": llm.get("reason_short", ""),
                }
            )


def build_legacy_results(classified: list[dict[str, Any]]) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in classified:
        match = item["match"]
        doc_id = str(match.get("document_id", ""))
        documents.setdefault(
            doc_id,
            {
                "kr_name": match.get("document_title", doc_id),
                "kr_link": match.get("document_link", ""),
                "drugs_mentioned": [],
                "summary": {
                    "total_drugs_mentioned": 0,
                    "drugs_in_blacklist": 0,
                    "total_mentions": 0,
                    "recommended_mentions": 0,
                    "contraindicated_mentions": 0,
                },
            },
        )
        grouped[(doc_id, str(match.get("canonical", "")))].append(item)

    for (doc_id, canonical), items in grouped.items():
        doc = documents[doc_id]
        first_match = items[0]["match"]
        in_blacklist = any(item["match"].get("source") == "blacklist" for item in items)
        blacklist_description = ""
        for item in items:
            metadata = item["match"].get("metadata", {})
            blacklist_description = metadata.get("description") or blacklist_description
        drug_info = {
            "drug": canonical,
            "in_blacklist": in_blacklist,
            "blacklist_description": blacklist_description or None,
            "blacklist_entry": first_match.get("metadata", {}).get("entry"),
            "mentions": [],
        }
        for item in items:
            match = item["match"]
            llm = item.get("llm", {})
            mention = {
                "context": f"{match.get('context_before', '')}{match.get('matched_text', '')}{match.get('context_after', '')}",
                "analysis": llm,
                "raw_match_id": item["raw_match_id"],
                "source": match.get("source"),
                "matched_text": match.get("matched_text"),
                "host_word": match.get("word_text"),
                "char_start": match.get("char_start"),
                "char_end": match.get("char_end"),
                "word_start": match.get("word_start"),
                "word_end": match.get("word_end"),
                "inside_word": match.get("inside_word"),
                "page": match.get("page"),
                "section": match.get("section"),
                "filter_decision": item.get("filter_decision", {}),
                "label": llm.get("label"),
                "udd": llm.get("udd"),
                "uur": llm.get("uur"),
            }
            drug_info["mentions"].append(mention)
            doc["summary"]["total_mentions"] += 1
            if llm.get("label") == "recommended":
                doc["summary"]["recommended_mentions"] += 1
            if llm.get("label") == "contraindication":
                doc["summary"]["contraindicated_mentions"] += 1
        doc["drugs_mentioned"].append(drug_info)

    for doc in documents.values():
        doc["summary"]["total_drugs_mentioned"] = len(doc["drugs_mentioned"])
        doc["summary"]["drugs_in_blacklist"] = sum(1 for item in doc["drugs_mentioned"] if item.get("in_blacklist"))
    return {"clinical_recommendations": list(documents.values())}


def write_legacy_results(classified: list[dict[str, Any]], output_path: str | Path, indent: int = 2) -> None:
    write_json(output_path, build_legacy_results(classified), indent=indent)
