from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .jsonio import read_json, write_json
from .llm_audit_dashboard_html import LLM_AUDIT_DASHBOARD_HTML
from .llm_review_cases import EVIDENCE_LETTERS, EVIDENCE_NUMBERS, LABELS, TARGET_KINDS, prompt_contract
from .llm_review_openrouter import (
    build_block_messages,
    collect_excluded_preparation_keys,
    filter_blocks_for_llm,
    iter_blocks,
    load_env_file,
    max_tokens_for_block,
    openrouter_key_status,
    terms_are_excluded,
)

LOGGER = logging.getLogger("anal_russia_klinik.llm_audit_dashboard")


def default_paths(
    base_dir: str | Path,
    *,
    review_path: str | Path | None = None,
    gold_path: str | Path | None = None,
    result_path: str | Path | None = None,
    analysis_path: str | Path | None = None,
) -> dict[str, Path]:
    base = Path(base_dir).resolve()
    return {
        "base": base,
        "review": Path(review_path).resolve() if review_path else base / "llm_review_cases.json",
        "gold": Path(gold_path).resolve() if gold_path else base / "llm_gold_40.json",
        "result": Path(result_path).resolve() if result_path else base / "openrouter_gold40_results.json",
        "analysis": Path(analysis_path).resolve() if analysis_path else base / "openrouter_gold40_partial_analysis.json",
        "excluded_preparations": base / "excluded_preparations.json",
    }


class AuditState:
    def __init__(
        self,
        base_dir: str | Path,
        env_file: str | Path | None = None,
        *,
        review_path: str | Path | None = None,
        gold_path: str | Path | None = None,
        result_path: str | Path | None = None,
        analysis_path: str | Path | None = None,
    ) -> None:
        self.paths = default_paths(base_dir, review_path=review_path, gold_path=gold_path, result_path=result_path, analysis_path=analysis_path)
        self.env_file = Path(env_file).resolve() if env_file else None
        self._blocks: dict[str, dict[str, Any]] | None = None
        self._filtered_blocks: dict[str, dict[str, Any]] | None = None
        self._excluded_filter_stats: dict[str, int] | None = None
        self._dashboard_items: list[dict[str, Any]] | None = None
        self._predictions: dict[str, dict[str, Any]] | None = None

    def gold_items(self) -> list[dict[str, Any]]:
        if not self.paths["gold"].exists():
            return []
        return read_json(self.paths["gold"]).get("items", [])

    def result(self) -> dict[str, Any]:
        return read_json(self.paths["result"]) if self.paths["result"].exists() else {}

    def predictions(self) -> dict[str, dict[str, Any]]:
        if self._predictions is None:
            self._predictions = self._load_prediction_sources()
        return self._predictions

    def excluded_preparations(self) -> dict[str, Any]:
        if not self.paths["excluded_preparations"].exists():
            return {"schema_version": 1, "items": []}
        return read_json(self.paths["excluded_preparations"])

    def excluded_preparation_keys(self) -> set[tuple[str, str, str]]:
        return collect_excluded_preparation_keys(self.excluded_preparations())

    def blocks(self) -> dict[str, dict[str, Any]]:
        if self._blocks is None:
            self._blocks = iter_blocks(read_json(self.paths["review"]))
        return self._blocks

    def filtered_blocks(self) -> dict[str, dict[str, Any]]:
        if not self.paths["review"].exists():
            return {}
        if self._filtered_blocks is None or self._excluded_filter_stats is None:
            self._filtered_blocks, self._excluded_filter_stats = filter_blocks_for_llm(self.blocks(), self.excluded_preparation_keys())
        return self._filtered_blocks

    def excluded_filter_stats(self) -> dict[str, int]:
        if not self.paths["review"].exists():
            return {"excluded_cases": 0, "excluded_blocks": 0}
        if self._filtered_blocks is None or self._excluded_filter_stats is None:
            self._filtered_blocks, self._excluded_filter_stats = filter_blocks_for_llm(self.blocks(), self.excluded_preparation_keys())
        return self._excluded_filter_stats

    def dashboard_items(self) -> list[dict[str, Any]]:
        if not self.paths["review"].exists():
            return self.gold_items()
        if self._dashboard_items is None:
            self._dashboard_items = [_dashboard_item_from_block(block) for block in self.filtered_blocks().values()]
        gold_by_block = {str(item.get("block_id", "")): item for item in self.gold_items()}
        return [_merge_gold(item, gold_by_block.get(item["block_id"])) for item in self._dashboard_items]

    def state_payload(self, *, include_key: bool = False) -> dict[str, Any]:
        result = self.result()
        predictions = self.predictions()
        gold_items = self.gold_items()
        dashboard_items = self.dashboard_items()
        eligible_block_ids = {item["block_id"] for item in dashboard_items}
        predicted_gold = [item for item in dashboard_items if item.get("block_id") in predictions]
        completed_all = len(predictions)
        completed_visible = sum(1 for block_id in predictions if block_id in eligible_block_ids)
        key_payload = self.key_status() if include_key else {}
        payload = {
            "paths": {key: str(path) for key, path in self.paths.items() if key != "base"},
            "gold_count": len(gold_items),
            "review_block_count": len(dashboard_items),
            "review_case_count": sum(len(_case_ids(item)) for item in dashboard_items),
            "reviewed_case_count": sum(_reviewed_case_count(item) for item in gold_items),
            "completed": completed_visible,
            "completed_all": completed_all,
            "completed_visible": completed_visible,
            **self.excluded_filter_stats(),
            "excluded_preparation_count": len(self.excluded_preparations().get("items", [])),
            "model": result.get("model"),
            "result_prompt_version": result.get("prompt_version"),
            "current_prompt_version": prompt_contract().get("prompt_version"),
            "score": _score_gold(predicted_gold, predictions) if predicted_gold else {},
            "key_status": key_payload if include_key else {},
        }
        return payload

    def key_status(self) -> dict[str, Any]:
        load_env_file(self.env_file, override=True)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return self._fallback_key_status("OPENROUTER_API_KEY is missing")
        try:
            return {"data": openrouter_key_status(api_key)}
        except Exception as exc:  # pragma: no cover - network diagnostics only
            return self._fallback_key_status(str(exc))

    def cases_page(self, query: dict[str, list[str]]) -> dict[str, Any]:
        q = _one(query, "q", "").strip().lower()
        status = _one(query, "status", "")
        offset = max(0, int(_one(query, "offset", "0") or 0))
        limit = min(500, max(1, int(_one(query, "limit", "80") or 80)))
        predictions = self.predictions()
        rows = [self._case_summary(index, item, predictions) for index, item in enumerate(self.dashboard_items(), start=1)]
        rows = [row for row in rows if self._matches_filters(row, q, status)]
        rows.sort(key=lambda row: (int(row.get("predicted_count", 0) or 0) == 0, int(row.get("index", 0) or 0)))
        return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset : offset + limit]}

    def case_payload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        items = self.dashboard_items()
        block_id = _one(query, "block_id", "")
        index = int(_one(query, "index", "0") or 0)
        if block_id:
            item = next(item for item in items if item["block_id"] == block_id)
            index = items.index(item) + 1
        else:
            item = items[index - 1]

        block = self.filtered_blocks().get(item["block_id"], item) if self.paths["review"].exists() else item
        prediction_block = self.predictions().get(item["block_id"], {})
        model = self.result().get("model") or os.getenv("OPENROUTER_MODEL")
        max_tokens = max_tokens_for_block(block) if self.paths["review"].exists() else int(os.getenv("OPENROUTER_MAX_TOKENS") or "600")
        context = block.get("context", {})
        case_spans = context.get("case_spans") or item.get("case_spans", [])
        case_golds = _case_golds(item)
        terms_by_case = _terms_by_case(block)
        excluded = self.excluded_preparation_keys()
        cases_data = []
        fallback_spans = [{"case_id": "default", "span_start": 0, "span_end": 0, "text": ""}]
        for span in case_spans or fallback_spans:
            case_id = str(span.get("case_id") or "default")
            cases_data.append(
                {
                    "case_id": case_id,
                    "span": span,
                    "terms": terms_by_case.get(case_id, item.get("primary_terms", [])),
                    "gold": case_golds.get(case_id, {}),
                    "prediction": _prediction_for_case(prediction_block, case_id, item["block_id"]) or {},
                    "preparation_excluded": terms_are_excluded(terms_by_case.get(case_id, item.get("primary_terms", [])), excluded),
                }
            )

        request_body = {}
        if self.paths["review"].exists():
            request_body = {
                "model": model,
                "messages": build_block_messages(block),
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": max_tokens,
            }

        return {
            "index": index,
            "item": item,
            "context": {
                "text": context.get("text", item.get("preview", "")),
                "context_start": context.get("context_start", item.get("preview_context_start", 0)),
                "context_end": context.get("context_end"),
                "block_span_start": context.get("block_span_start", context.get("span_start", 0)),
                "block_span_end": context.get("block_span_end", context.get("span_end", 0)),
                "case_spans": case_spans,
                "highlight_spans": context.get("highlight_spans", case_spans),
                "evidence_level_candidates": context.get("evidence_level_candidates", item.get("evidence_level_candidates", [])),
            },
            "cases": cases_data,
            "request_body": request_body,
        }

    def update_gold(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = read_json(self.paths["gold"]) if self.paths["gold"].exists() else _empty_gold_report(self.paths["review"])
        items = data.setdefault("items", [])
        block_id = str(payload.get("block_id") or "")
        case_id = str(payload.get("case_id") or "")
        index = int(payload.get("index") or 0)

        if block_id:
            item = next((item for item in items if item.get("block_id") == block_id), None)
            if item is None:
                item = _persisted_item_from_block(self.blocks()[block_id]) if self.paths["review"].exists() and block_id in self.blocks() else {"block_id": block_id}
                items.append(item)
            index = items.index(item) + 1
        elif index:
            dashboard_item = self.dashboard_items()[index - 1]
            block_id = dashboard_item["block_id"]
            item = next((item for item in items if item.get("block_id") == block_id), None)
            if item is None:
                item = _persisted_item_from_block(self.blocks()[block_id]) if self.paths["review"].exists() and block_id in self.blocks() else dict(dashboard_item)
                items.append(item)
        else:
            raise ValueError("index_or_block_id_required")

        case_key = case_id or (_case_ids(item)[0] if _case_ids(item) else "default")
        item.setdefault("case_golds", {})
        fallback_gold = item.get("gold", {}) if len(_case_ids(item)) <= 1 else {}
        gold = dict(item.get("case_golds", {}).get(case_key, fallback_gold))
        _set_checked(payload, gold, "label", LABELS, "label")
        _set_checked(payload, gold, "target_kind", TARGET_KINDS, "target_kind")
        _set_checked(payload, gold, "recommendation_strength", EVIDENCE_LETTERS, "recommendation_strength", optional=True)
        _set_checked(payload, gold, "evidence_level", EVIDENCE_NUMBERS, "evidence_level", optional=True)
        _set_text(payload, gold, "evidence_quote")
        _set_text(payload, gold, "reason")
        _set_text(payload, gold, "comment")
        if "exclude_from_model_stats" in payload:
            gold["exclude_from_model_stats"] = bool(payload.get("exclude_from_model_stats"))
        gold["reviewer"] = _optional_text(payload.get("reviewer")) or "dashboard_manual"
        gold["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        item["case_golds"][case_key] = gold
        if len(_case_ids(item)) <= 1 or not item.get("gold"):
            item["gold"] = gold

        write_json(self.paths["gold"], data)
        return {"index": index, "item": item, "state": self.state_payload()}

    def _case_summary(self, index: int, item: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
        block_id = item["block_id"]
        prediction_block = predictions.get(block_id, {})
        case_ids = _case_ids(item)
        case_golds = _case_golds(item)
        all_ok = True
        predicted_count = 0
        reviewed_count = 0
        pred_labels = []
        gold_labels = []
        for case_id in case_ids:
            pred = _prediction_for_case(prediction_block, case_id, block_id)
            gold = case_golds.get(case_id, {})
            if pred:
                predicted_count += 1
                pred_labels.append(str(pred.get("label") or ""))
                if gold.get("label"):
                    all_ok = all_ok and pred.get("label") == gold.get("label")
                else:
                    all_ok = False
            if gold.get("label"):
                reviewed_count += 1
                gold_labels.append(str(gold.get("label") or ""))

        if predicted_count == 0:
            status = "pending"
        elif reviewed_count == 0:
            status = "llm"
        elif reviewed_count < len(case_ids):
            status = "partial"
        else:
            status = "ok" if all_ok else "bad"

        terms = [f"{term.get('canonical')}[{term.get('host_word')}]" for term in item.get("primary_terms", [])]
        meta_sample = _prediction_meta(prediction_block)
        return {
            "index": index,
            "block_id": block_id,
            "document_title": item.get("document_title"),
            "document_link": item.get("document_link"),
            "case_count": len(case_ids),
            "predicted_count": predicted_count,
            "reviewed_count": reviewed_count,
            "terms": terms,
            "gold_label": _compact_labels(gold_labels),
            "pred_label": _compact_labels(pred_labels),
            "status": status,
            "usage_delta": meta_sample.get("key_delta", {}).get("usage_delta"),
            "total_tokens": meta_sample.get("response", {}).get("usage", {}).get("total_tokens"),
        }

    @staticmethod
    def _matches_filters(row: dict[str, Any], q: str, status: str) -> bool:
        haystack = " ".join(str(row.get(key, "")) for key in ("block_id", "document_title", "gold_label", "pred_label", "status")).lower()
        haystack += " " + " ".join(row.get("terms", [])).lower()
        return (not q or q in haystack) and (not status or row["status"] == status)

    def _load_prediction_sources(self) -> dict[str, dict[str, Any]]:
        predictions: dict[str, dict[str, Any]] = {}
        candidates = sorted(self.paths["base"].glob("openrouter*_results.json"))
        if self.paths["result"].exists() and self.paths["result"] not in candidates:
            candidates.append(self.paths["result"])
        for path in candidates:
            try:
                value = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            source_predictions = value.get("predictions", {})
            if not isinstance(source_predictions, dict):
                continue
            for block_id, prediction in source_predictions.items():
                if isinstance(prediction, dict):
                    predictions[str(block_id)] = prediction
        return predictions

    def exclude_preparation(self, payload: dict[str, Any]) -> dict[str, Any]:
        term = payload.get("term")
        if not isinstance(term, dict):
            raise ValueError("term_object_required")
        data = self.excluded_preparations()
        items = data.setdefault("items", [])
        key = (str(term.get("source", "")), str(term.get("term_id", "")), str(term.get("canonical", "")).casefold())
        if not key[2]:
            raise ValueError("canonical_required")
        existing = {
            (str(item.get("source", "")), str(item.get("term_id", "")), str(item.get("canonical", "")).casefold())
            for item in items
        }
        if key not in existing:
            items.append(
                {
                    "source": term.get("source"),
                    "term_id": term.get("term_id"),
                    "canonical": term.get("canonical"),
                    "search_word": term.get("search_word"),
                    "host_word": term.get("host_word"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "reason": _optional_text(payload.get("reason")) or "dashboard_exclude_preparation",
                }
            )
            write_json(self.paths["excluded_preparations"], data)
        self._filtered_blocks = None
        self._excluded_filter_stats = None
        self._dashboard_items = None
        return {"excluded_preparations": data, "state": self.state_payload()}

    def remove_excluded_preparation(self, payload: dict[str, Any]) -> dict[str, Any]:
        term = payload.get("term")
        if not isinstance(term, dict):
            raise ValueError("term_object_required")
        data = self.excluded_preparations()
        items = data.setdefault("items", [])
        key = (str(term.get("source", "")), str(term.get("term_id", "")), str(term.get("canonical", "")).casefold())
        data["items"] = [item for item in items if (
            str(item.get("source", "")),
            str(item.get("term_id", "")),
            str(item.get("canonical", "")).casefold()
        ) != key]
        write_json(self.paths["excluded_preparations"], data)
        self._filtered_blocks = None
        self._excluded_filter_stats = None
        self._dashboard_items = None
        return {"excluded_preparations": data, "state": self.state_payload()}

    def excluded_stats_list(self) -> list[dict[str, Any]]:
        """Get list of all case golds marked with exclude_from_model_stats."""
        items = []
        for block_id, block in self.blocks().items():
            gold_data = self.gold_items()
            for gold_item in gold_data:
                if str(gold_item.get("block_id")) == block_id:
                    case_golds = gold_item.get("case_golds", {})
                    for case_id, gold in case_golds.items():
                        if gold.get("exclude_from_model_stats"):
                            items.append({
                                "block_id": block_id,
                                "case_id": case_id,
                                "label": gold.get("label"),
                                "target_kind": gold.get("target_kind"),
                                "canonical": _canonical_from_block_and_case(self.blocks().get(block_id, {}), case_id),
                                "reviewed_at": gold.get("reviewed_at"),
                                "reason": gold.get("reason", "exclude_from_model_stats"),
                            })
                    break
        return items

    def remove_exclude_from_stats(self, payload: dict[str, Any]) -> dict[str, Any]:
        block_id = str(payload.get("block_id") or "")
        case_id = str(payload.get("case_id") or "")
        data = read_json(self.paths["gold"]) if self.paths["gold"].exists() else {}
        items = data.setdefault("items", [])
        
        for item in items:
            if str(item.get("block_id")) == block_id:
                case_golds = item.get("case_golds", {})
                if case_id in case_golds:
                    case_golds[case_id].pop("exclude_from_model_stats", None)
                break
        
        write_json(self.paths["gold"], data)
        self._dashboard_items = None
        return {"state": self.state_payload()}

    def _fallback_key_status(self, error: str) -> dict[str, Any]:
        for prediction in reversed(list(self.predictions().values())):
            meta = _prediction_meta(prediction)
            key_after = meta.get("key_after")
            if isinstance(key_after, dict) and "limit_remaining" in key_after:
                return {"data": {**key_after, "stale": True, "error": error}}
        return {"error": error}


class Handler(BaseHTTPRequestHandler):
    state: AuditState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_html(LLM_AUDIT_DASHBOARD_HTML)
            elif parsed.path == "/api/state":
                self._send_json(self.state.state_payload(include_key=_one(query, "key", "") == "1"))
            elif parsed.path == "/api/key":
                self._send_json(self.state.key_status())
            elif parsed.path == "/api/cases":
                self._send_json(self.state.cases_page(query))
            elif parsed.path == "/api/case":
                self._send_json(self.state.case_payload(query))
            elif parsed.path == "/api/excluded-preparations":
                self._send_json({"items": self.state.excluded_preparations().get("items", [])})
            elif parsed.path == "/api/excluded-stats":
                self._send_json({"items": self.state.excluded_stats_list()})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            LOGGER.exception("request_failed path=%s", self.path)
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/gold":
                self._send_json(self.state.update_gold(payload))
            elif parsed.path == "/api/exclude-preparation":
                self._send_json(self.state.exclude_preparation(payload))
            elif parsed.path == "/api/remove-excluded-preparation":
                self._send_json(self.state.remove_excluded_preparation(payload))
            elif parsed.path == "/api/remove-exclude-from-stats":
                self._send_json(self.state.remove_exclude_from_stats(payload))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            LOGGER.exception("request_failed path=%s", self.path)
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info(format, *args)

    def _send_html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        value = json.loads(raw or "{}")
        if not isinstance(value, dict):
            raise ValueError("json_object_expected")
        return value


def _canonical_from_block_and_case(block: dict[str, Any], case_id: str) -> str:
    """Extract canonical term name from block for a given case."""
    case_id_str = str(case_id)
    for case in block.get("cases", []):
        if str(case.get("case_id") or "") == case_id_str:
            terms = case.get("primary_terms", [])
            if terms:
                return str(terms[0].get("canonical", ""))
    # Fallback to primary_terms
    terms = block.get("primary_terms", [])
    if terms:
        return str(terms[0].get("canonical", ""))
    return ""


def _dashboard_item_from_block(block: dict[str, Any]) -> dict[str, Any]:
    context = block.get("context", {})
    return {
        "block_id": block["block_id"],
        "document_id": block.get("document_id"),
        "document_title": block.get("document_title"),
        "document_link": block.get("document_link"),
        "case_count": block.get("case_count", len(context.get("case_spans", []))),
        "case_ids": block.get("case_ids", [span.get("case_id") for span in context.get("case_spans", [])]),
        "case_spans": context.get("case_spans", []),
        "primary_terms": block.get("primary_terms", []),
        "cases": block.get("cases", []),
    }


def _persisted_item_from_block(block: dict[str, Any]) -> dict[str, Any]:
    context = block.get("context", {})
    return {
        **_dashboard_item_from_block(block),
        "evidence_level_candidates": context.get("evidence_level_candidates", []),
        "preview_context_start": context.get("context_start"),
        "preview": context.get("text", ""),
        "case_golds": {},
    }


def _merge_gold(item: dict[str, Any], gold_item: dict[str, Any] | None) -> dict[str, Any]:
    if not gold_item:
        return dict(item)
    merged = dict(item)
    for key in ("gold", "case_golds", "evidence_level_candidates", "preview_context_start", "preview"):
        if key in gold_item:
            merged[key] = gold_item[key]
    return merged


def _empty_gold_report(review_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "unit": "llm_blocks",
        "source_review_cases": str(review_path) if review_path.exists() else None,
        "items": [],
    }


def _case_ids(item: dict[str, Any]) -> list[str]:
    ids = [str(span.get("case_id")) for span in item.get("case_spans", []) if span.get("case_id")]
    if not ids:
        ids = [str(case_id) for case_id in item.get("case_ids", []) if case_id]
    if not ids and item.get("case_golds"):
        ids = [str(case_id) for case_id in item.get("case_golds", {})]
    if not ids:
        ids = ["default"]
    return ids


def _case_golds(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ids = _case_ids(item)
    golds: dict[str, dict[str, Any]] = {}
    if isinstance(item.get("gold"), dict):
        for case_id in ids:
            golds[case_id] = dict(item["gold"])
    if isinstance(item.get("case_golds"), dict):
        for case_id, gold in item["case_golds"].items():
            if isinstance(gold, dict):
                golds[str(case_id)] = dict(gold)
    return golds


def _reviewed_case_count(item: dict[str, Any]) -> int:
    return sum(1 for gold in _case_golds(item).values() if gold.get("label"))


def _terms_by_case(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for case in block.get("cases", []):
        case_id = str(case.get("case_id") or "")
        if case_id:
            output[case_id] = case.get("primary_terms", [])
    return output


def _prediction_for_case(prediction_block: Any, case_id: str, block_id: str) -> dict[str, Any]:
    if not isinstance(prediction_block, dict):
        return {}
    nested = prediction_block.get("predictions")
    if isinstance(nested, dict):
        for key in (case_id, block_id, "default"):
            value = nested.get(key)
            if isinstance(value, dict):
                return value
        if len(nested) == 1:
            value = next(iter(nested.values()))
            return value if isinstance(value, dict) else {}
    for key in (case_id, block_id):
        value = prediction_block.get(key)
        if isinstance(value, dict):
            return value
    if isinstance(prediction_block.get("label"), str):
        return prediction_block
    return {}


def _prediction_meta(prediction_block: Any) -> dict[str, Any]:
    if not isinstance(prediction_block, dict):
        return {}
    meta = prediction_block.get("_openrouter")
    return meta if isinstance(meta, dict) else {}


def _score_gold(gold_items: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    label_ok = 0
    evidence_ok = 0
    strength_ok = 0
    for item in gold_items:
        block_id = str(item.get("block_id") or "")
        prediction_block = predictions.get(block_id, {})
        for case_id, gold in _case_golds(item).items():
            if not gold.get("label") or gold.get("exclude_from_model_stats"):
                continue
            pred = _prediction_for_case(prediction_block, case_id, block_id)
            row = {
                "block_id": block_id,
                "case_id": case_id,
                "gold_label": gold.get("label"),
                "pred_label": pred.get("label"),
                "label_ok": gold.get("label") == pred.get("label"),
                "gold_recommendation_strength": gold.get("recommendation_strength"),
                "pred_recommendation_strength": pred.get("recommendation_strength"),
                "recommendation_strength_ok": (gold.get("recommendation_strength") or None) == (pred.get("recommendation_strength") or None),
                "gold_evidence_level": gold.get("evidence_level"),
                "pred_evidence_level": pred.get("evidence_level"),
                "evidence_ok": (gold.get("evidence_level") or None) == (pred.get("evidence_level") or None),
            }
            label_ok += int(row["label_ok"])
            evidence_ok += int(row["evidence_ok"])
            strength_ok += int(row["recommendation_strength_ok"])
            rows.append(row)
    total = len(rows)
    return {
        "total": total,
        "label_accuracy": label_ok / total if total else 0.0,
        "recommendation_strength_accuracy": strength_ok / total if total else 0.0,
        "evidence_level_accuracy": evidence_ok / total if total else 0.0,
        "rows": rows,
    }


def _compact_labels(values: list[str]) -> str:
    labels = sorted({value for value in values if value})
    if not labels:
        return "-"
    return ",".join(labels[:2]) + (f"+{len(labels) - 2}" if len(labels) > 2 else "")


def _set_checked(payload: dict[str, Any], target: dict[str, Any], key: str, allowed: list[str], name: str, *, optional: bool = False) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if value in {None, ""}:
        if optional:
            target[key] = None
        return
    target[key] = _checked(value, allowed, name)


def _set_text(payload: dict[str, Any], target: dict[str, Any], key: str) -> None:
    if key in payload:
        target[key] = _optional_text(payload.get(key))


def _one(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key)
    return values[0] if values else default


def _checked(value: Any, allowed: list[str], name: str) -> str:
    text = str(value or "")
    if text not in allowed:
        raise ValueError(f"{name}_invalid")
    return text


def _optional_text(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-openrouter-dashboard")
    parser.add_argument("--base-dir", default=str(Path("reports") / "llm"))
    parser.add_argument("--env-file", default=str(Path("config") / "openrouter.env"))
    parser.add_argument("--review")
    parser.add_argument("--gold")
    parser.add_argument("--result")
    parser.add_argument("--analysis")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    Handler.state = AuditState(args.base_dir, args.env_file, review_path=args.review, gold_path=args.gold, result_path=args.result, analysis_path=args.analysis)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("llm_audit_dashboard_started url=http://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
