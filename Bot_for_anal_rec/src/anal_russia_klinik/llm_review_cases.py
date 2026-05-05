from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json
from .models import Document
from .text import load_legacy_clinical_json

LABELS = ["recommendation", "contraindication", "literature_mention", "error", "unclear"]
TARGET_KINDS = ["drug", "method", "marker", "other"]
EVIDENCE_LETTERS = ["A", "B", "C"]
EVIDENCE_NUMBERS = ["1", "2", "3", "4", "5"]

EVIDENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("evidence_level_number", re.compile(r"(?:УДД|уровень достоверности доказательств)\s*[:=-]?\s*([1-5])", re.I)),
    ("recommendation_strength_letter", re.compile(r"(?:УУР|уровень убедительности рекомендаций)\s*[:=-]?\s*([A-CА-В])", re.I)),
    ("combined_evidence_level", re.compile(r"\b([A-CА-В][1-5])\b", re.I)),
)

def case_id(document_id: str, char_start: int, char_end: int) -> str:
    digest = hashlib.sha1(f"{document_id}|{char_start}|{char_end}".encode("utf-8")).hexdigest()[:16]
    return f"case:{digest}"

def block_id(document_id: str, char_start: int, char_end: int) -> str:
    digest = hashlib.sha1(f"{document_id}|block|{char_start}|{char_end}".encode("utf-8")).hexdigest()[:16]
    return f"block:{digest}"

def _normalize_evidence_value(value: str) -> str:
    table = str.maketrans({"А": "A", "а": "A", "В": "B", "в": "B", "С": "C", "с": "C"})
    return value.translate(table).upper()

def _evidence_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for kind, pattern in EVIDENCE_PATTERNS:
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), kind)
            if key in seen:
                continue
            seen.add(key)
            value = match.group(1) if match.groups() else match.group(0)
            spans.append(
                {
                    "type": "evidence_level",
                    "kind": kind,
                    "value": value,
                    "normalized_value": _normalize_evidence_value(value),
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "text": match.group(0),
                }
            )
    return sorted(spans, key=lambda item: (item["span_start"], item["span_end"], item["kind"]))

def _primary_terms(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms: dict[tuple[str, str, str, str, str, bool], dict[str, Any]] = {}
    for item in matches:
        key = (
            str(item.get("source", "")),
            str(item.get("term_id", "")),
            str(item.get("canonical", "")),
            str(item.get("search_word", "")),
            str(item.get("host_word", "")),
            bool(item.get("inside_word", False)),
        )
        terms[key] = {
            "source": key[0],
            "term_id": key[1],
            "canonical": key[2],
            "search_word": key[3],
            "host_word": key[4],
            "inside_word": key[5],
        }
    return sorted(terms.values(), key=lambda item: (item["source"], item["canonical"].casefold(), item["search_word"].casefold(), item["host_word"].casefold()))

def _match_rows(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in matches:
        rows.append(
            {
                "source": item.get("source", ""),
                "term_id": item.get("term_id", ""),
                "canonical": item.get("canonical", ""),
                "search_word": item.get("search_word", ""),
                "normalized_search_word": item.get("normalized_search_word", ""),
                "host_word": item.get("host_word", ""),
                "inside_word": bool(item.get("inside_word", False)),
            }
        )
    return rows

def _context_payload(document: Document, group: dict[str, Any], window_chars: int) -> dict[str, Any]:
    start = int(group.get("char_start", 0) or 0)
    end = int(group.get("char_end", 0) or 0)
    context_start = max(0, start - window_chars)
    context_end = min(len(document.text), end + window_chars)
    text = document.text[context_start:context_end]
    span_start = start - context_start
    span_end = end - context_start
    evidence_spans = _evidence_spans(text)
    highlight_spans = [
        {
            "type": "match",
            "span_start": span_start,
            "span_end": span_end,
            "text": text[span_start:span_end],
        },
        *evidence_spans,
    ]
    return {
        "context_start": context_start,
        "context_end": context_end,
        "span_start": span_start,
        "span_end": span_end,
        "text": text,
        "before": text[:span_start],
        "match": text[span_start:span_end],
        "after": text[span_end:],
        "highlight_spans": highlight_spans,
        "evidence_level_candidates": evidence_spans,
    }

def _llm_result_stub() -> dict[str, Any]:
    return {
        "status": "pending",
        "label": None,
        "target_kind": None,
        "evidence_quote": None,
        "reason": None,
        "recommendation_strength": None,
        "evidence_level": None,
        "confidence": None,
    }

def _validation_stub() -> dict[str, Any]:
    return {
        "status": "unreviewed",
        "human_label": None,
        "human_evidence_level": None,
        "human_recommendation_strength": None,
        "llm_label_is_correct": None,
        "llm_evidence_level_is_correct": None,
        "false_positive_label": None,
        "false_negative_labels": [],
        "comment": None,
    }

def _case_payload(group: dict[str, Any], document: Document, window_chars: int) -> dict[str, Any]:
    start = int(group.get("char_start", 0) or 0)
    end = int(group.get("char_end", 0) or 0)
    matches = _match_rows(list(group.get("matches", [])))
    return {
        "case_id": case_id(document.document_id, start, end),
        "location_id": group.get("location_id"),
        "document_id": document.document_id,
        "document_title": document.title,
        "document_link": document.link,
        "location": {
            "char_start": start,
            "char_end": end,
            "word_start": group.get("word_start"),
            "word_end": group.get("word_end"),
            "matched_text": group.get("matched_text"),
            "page": group.get("page"),
            "section": group.get("section"),
        },
        "context": _context_payload(document, group, window_chars),
        "primary_terms": _primary_terms(matches),
        "matches": matches,
        "llm_payload": {
            "task": "classify_clinical_recommendation_mention",
            "labels": LABELS,
            "target_kinds": TARGET_KINDS,
            "document_title": document.title,
            "section": group.get("section"),
            "found_terms": _primary_terms(matches),
            "context_ref": "context.text",
        },
        "llm_result": _llm_result_stub(),
        "human_validation": _validation_stub(),
    }

def _slim_case_for_block(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "location_id": case["location_id"],
        "location": case["location"],
        "primary_terms": case["primary_terms"],
        "matches": case["matches"],
    }

def _block_context_payload(document: Document, cases: list[dict[str, Any]], window_chars: int) -> dict[str, Any]:
    start = min(int(case["location"]["char_start"]) for case in cases)
    end = max(int(case["location"]["char_end"]) for case in cases)
    context_start = max(0, start - window_chars)
    context_end = min(len(document.text), end + window_chars)
    text = document.text[context_start:context_end]
    case_spans = []
    for case in cases:
        case_start = int(case["location"]["char_start"]) - context_start
        case_end = int(case["location"]["char_end"]) - context_start
        case_spans.append(
            {
                "type": "match",
                "case_id": case["case_id"],
                "span_start": case_start,
                "span_end": case_end,
                "text": text[case_start:case_end],
            }
        )
    evidence_spans = _evidence_spans(text)
    return {
        "context_start": context_start,
        "context_end": context_end,
        "block_span_start": start - context_start,
        "block_span_end": end - context_start,
        "text": text,
        "case_spans": case_spans,
        "highlight_spans": [*case_spans, *evidence_spans],
        "evidence_level_candidates": evidence_spans,
    }

def _merge_terms(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _primary_terms([match for case in cases for match in case.get("matches", [])])

def _build_case_blocks(cases: list[dict[str, Any]], *, max_gap_chars: int) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    for case in cases:
        if not blocks:
            blocks.append([case])
            continue
        previous_end = int(blocks[-1][-1]["location"]["char_end"])
        current_start = int(case["location"]["char_start"])
        if current_start - previous_end <= max_gap_chars:
            blocks[-1].append(case)
        else:
            blocks.append([case])
    return blocks

def _block_payload(document: Document, cases: list[dict[str, Any]], window_chars: int, max_gap_chars: int) -> dict[str, Any]:
    start = min(int(case["location"]["char_start"]) for case in cases)
    end = max(int(case["location"]["char_end"]) for case in cases)
    block_cases = [_slim_case_for_block(case) for case in cases]
    return {
        "block_id": block_id(document.document_id, start, end),
        "document_id": document.document_id,
        "document_title": document.title,
        "document_link": document.link,
        "location": {"char_start": start, "char_end": end},
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "max_gap_chars": max_gap_chars,
        "context": _block_context_payload(document, cases, window_chars),
        "primary_terms": _merge_terms(cases),
        "cases": block_cases,
        "llm_payload": {
            "task": "classify_clinical_recommendation_mention_block",
            "labels": LABELS,
            "target_kinds": TARGET_KINDS,
            "document_title": document.title,
            "found_terms": _merge_terms(cases),
            "case_count": len(cases),
            "context_ref": "context.text",
        },
        "llm_result": _llm_result_stub(),
        "human_validation": _validation_stub(),
    }


def prompt_contract() -> dict[str, Any]:
    return {
        "prompt_version": "llm-review-v3-multi",
        "system": (
            "You classify highlighted Aho-Corasick mention blocks inside a Russian clinical guideline. "
            "Classify each highlighted found term itself, not the surrounding paragraph in general. "
            "Use only context_text, case_spans, evidence candidates, and found_terms. "
            "Return a JSON object with 'predictions' array - one entry per found term (case_id), in the same order as found_terms. "
            "Output ONLY: label, recommendation_strength, evidence_level, and reason. Do NOT classify or output target_kind."
        ),
        "user_task": (
            "Labels: recommendation, contraindication, literature_mention, error, unclear. "
            "CRITICAL: Classify based on the LOCATION and CONTEXT of the highlighted host_word itself, not the general paragraph topic. "
            "Use recommendation ONLY when the highlighted host_word is itself directly prescribed/advised/used as a treatment, regimen, method, or positive therapeutic action. "
            "Component rule: if the highlighted host_word is a named component/substance inside a recommended product group or complex (e.g., vitamins, mineral complex, antioxidant complex, lutein, zeaxanthin), classify it as recommendation when the surrounding sentence recommends that group/complex for patients. "
            "Use contraindication ONLY when the highlighted host_word is itself prohibited, avoided, or explicitly stated as NOT recommended. "
            "Use literature_mention only when the highlighted host_word is itself the drug/method/entity being mentioned in literature, not when search_word only matched part of a person name, author surname, journal title, book title, publisher, DOI, citation noise, or bibliography metadata. "
            "Use error for: glossary entries, abbreviation lists, contents/ToC, OCR noise, generic common words out of context, unrelated filler text, bibliography author surnames, reference-list metadata, or when the highlighted host_word is KNOWN to be a different drug/entity than search_word. "
            "INSIDE_WORD rule: If inside_word=true (host_word contains search_word as substring): Classify the HOST_WORD context, not search_word. If host_word is generic/common (e.g., 'витамины' containing 'мин'), classify as error UNLESS the paragraph explicitly treats this specific host_word as the actionable item. If host_word is a morphological variant or stem of the same drug (e.g., 'рисдиплам' containing 'диплам'), classify as the drug's context. "
            "Do NOT override classification based on paragraph mood alone. A recommendation paragraph containing an error-classified mention should stay error. A literature section containing a drug name stays literature_mention. "
            "Use unclear ONLY when essential clinical/contextual information is genuinely missing, not for cases of known noise or false matches. "
            "For recommendation/contraindication: Extract UUR (recommendation_strength: A|B|C) and UDD (evidence_level: 1|2|3|4|5) ONLY if they are directly tied to the same recommendation/row/table entry that governs the host_word. If evidence is absent or tied to a different recommendation, return null. Never guess or infer evidence values. "
            "UUR levels: A = strong recommendation, B = moderate, C = weak. UDD levels: 1 = very high, 2 = high, 3 = moderate, 4 = low, 5 = very low evidence quality. "
            "Table/component rules: A regimen table row inherits the nearest explicit evidence statement that introduces or governs that table, until the next recommendation or new table. Alternative drug lists and component lists after a recommendation inherit the parent recommendation's evidence. "
            "Strict inside-word examples: Metacin inside indomethacin means different drugs => error; inside_word=true is not automatic error; check Децит and рисдиплам examples by host_word identity. "
            "Examples: (a) 'Оковит' or 'Оковитов' inside author surname 'Оковитов В.В.' in a numbered bibliography/reference list => error, not literature_mention; (b) real drug name in an article title/study discussion without clinical action => literature_mention; (c) 'Метацин' inside 'индометацин' with no indication they share context => error; (d) 'рисдиплам' inside larger context => classify risdiplam; (e) supporting therapy in regimen table => recommendation with inherited evidence; (f) lutein/zeaxanthin/antioxidant inside recommended vitamin-mineral or antioxidant complex => recommendation with parent evidence if present; (g) pathophysiology mention of a drug => error; (h) 'not recommended metamisole' => contraindication; (i) 'potential contraindication' with unclear clinical link => unclear or error depending on evidence."
        ),
        "output_schema": {
            "predictions": [
                {
                    "case_id": "case:xxxxx",
                    "label": "recommendation|contraindication|literature_mention|error|unclear",
                    "reason": "short reason in Russian",
                    "recommendation_strength": "A|B|C|null",
                    "evidence_level": "1|2|3|4|5|null",
                }
            ]
        },
    }


def dashboard_validation_schema() -> dict[str, Any]:
    return {
        "labels": LABELS,
        "evidence_level_letters": EVIDENCE_LETTERS,
        "evidence_level_numbers": EVIDENCE_NUMBERS,
        "metrics": [
            "label_confusion_matrix",
            "false_positive_count_by_label",
            "false_negative_count_by_label",
            "precision_recall_f1_by_label",
            "evidence_level_confusion_matrix",
            "evidence_level_precision_recall_f1_by_value",
        ],
    }


def build_review_cases_report(
    location_report_path: str | Path,
    clinical_json_path: str | Path,
    *,
    window_chars: int = 2500,
    block_gap_chars: int = 100,
) -> dict[str, Any]:
    location_report = read_json(location_report_path)
    documents = {document.document_id: document for document in load_legacy_clinical_json(clinical_json_path)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    missing_document_ids: set[str] = set()
    for group in location_report.get("by_location", []):
        document_id = str(group.get("document_id", ""))
        document = documents.get(document_id)
        if document is None:
            missing_document_ids.add(document_id)
            continue
        grouped.setdefault(document_id, []).append(_case_payload(group, document, window_chars))
    clinical_rows = []
    for document_id, document in sorted(documents.items(), key=lambda item: item[0]):
        cases = sorted(grouped.get(document_id, []), key=lambda item: int(item["location"]["char_start"]))
        if not cases:
            continue
        llm_blocks = [
            _block_payload(document, block_cases, window_chars, block_gap_chars)
            for block_cases in _build_case_blocks(cases, max_gap_chars=block_gap_chars)
        ]
        clinical_rows.append(
            {
                "document_id": document.document_id,
                "document_title": document.title,
                "document_link": document.link,
                "text_length": len(document.text),
                "case_count": len(cases),
                "llm_block_count": len(llm_blocks),
                "llm_blocks": llm_blocks,
                "cases": cases,
            }
        )
    case_count = sum(row["case_count"] for row in clinical_rows)
    block_count = sum(row["llm_block_count"] for row in clinical_rows)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_location_report": str(location_report_path),
        "source_clinical_recommendations": str(clinical_json_path),
        "context_window_chars_each_side": window_chars,
        "nearby_block_gap_chars": block_gap_chars,
        "case_grouping": "clinical_recommendations[*].cases[*]",
        "llm_review_unit": "clinical_recommendations[*].llm_blocks[*]",
        "prompt_contract": prompt_contract(),
        "dashboard_validation_schema": dashboard_validation_schema(),
        "summary": {
            "clinical_recommendation_count": len(clinical_rows),
            "case_count": case_count,
            "llm_block_count": block_count,
            "multi_case_block_count": sum(1 for row in clinical_rows for block in row["llm_blocks"] if block["case_count"] > 1),
            "missing_document_count": len(missing_document_ids),
            "missing_document_ids": sorted(missing_document_ids),
            "evidence_level_candidate_count": sum(len(case["context"]["evidence_level_candidates"]) for row in clinical_rows for case in row["cases"]),
        },
        "clinical_recommendations": clinical_rows,
    }


def write_review_cases_report(
    location_report_path: str | Path,
    clinical_json_path: str | Path,
    output_path: str | Path,
    *,
    window_chars: int = 2500,
    block_gap_chars: int = 100,
    indent: int = 2,
) -> dict[str, Any]:
    report = build_review_cases_report(location_report_path, clinical_json_path, window_chars=window_chars, block_gap_chars=block_gap_chars)
    write_json(output_path, report, indent=indent)
    return report
