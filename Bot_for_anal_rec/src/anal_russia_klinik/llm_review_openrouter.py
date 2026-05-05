from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError

from .jsonio import read_json, write_json
from .llm_review_cases import LABELS, TARGET_KINDS, prompt_contract

ExcludedPreparationKey = tuple[str, str, str]


def load_env_file(env_file: str | Path | None, *, override: bool = False) -> int:
    if not env_file:
        return 0
    path = Path(env_file)
    if not path.exists():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and (override or os.getenv(key) in {None, ""}):
            os.environ[key] = value
            loaded += 1
    return loaded


def iter_blocks(review_cases: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        block["block_id"]: block
        for row in review_cases.get("clinical_recommendations", [])
        for block in row.get("llm_blocks", [])
    }


def collect_excluded_preparation_keys(data: dict[str, Any]) -> set[ExcludedPreparationKey]:
    keys: set[ExcludedPreparationKey] = set()
    for item in data.get("items", []):
        canonical = str(item.get("canonical", "")).casefold()
        if canonical:
            keys.add((str(item.get("source", "")), str(item.get("term_id", "")), canonical))
    return keys


def load_excluded_preparation_keys(path: str | Path | None) -> set[ExcludedPreparationKey]:
    if not path:
        return set()
    blacklist_path = Path(path)
    if not blacklist_path.exists():
        return set()
    return collect_excluded_preparation_keys(read_json(blacklist_path))


def term_is_excluded(term: dict[str, Any], excluded: set[ExcludedPreparationKey]) -> bool:
    source = str(term.get("source", ""))
    term_id = str(term.get("term_id", ""))
    canonical = str(term.get("canonical", "")).casefold()
    if not canonical:
        return False
    if (source, term_id, canonical) in excluded:
        return True
    return canonical in {key[2] for key in excluded}


def terms_are_excluded(terms: list[dict[str, Any]], excluded: set[ExcludedPreparationKey]) -> bool:
    return bool(terms) and all(term_is_excluded(term, excluded) for term in terms)


def _block_case_ids(block: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    sources = [
        block.get("case_ids", []),
        [case.get("case_id") for case in block.get("cases", [])],
        [span.get("case_id") for span in block.get("context", {}).get("case_spans", [])],
    ]
    for source in sources:
        for raw_case_id in source or []:
            case_id = str(raw_case_id or "")
            if case_id and case_id not in seen:
                seen.add(case_id)
                ids.append(case_id)
    return ids


def _block_filter_case_count(block: dict[str, Any]) -> int:
    case_ids = _block_case_ids(block)
    if case_ids:
        return len(case_ids)
    return 1 if block.get("primary_terms") else 0


def _case_terms_from_block(block: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    case_id = str(case_id)
    for case in block.get("cases", []):
        if str(case.get("case_id") or "") == case_id:
            terms = case.get("primary_terms") or case.get("matches") or []
            return [term for term in terms if isinstance(term, dict)]
    return [term for term in block.get("primary_terms", []) if isinstance(term, dict)]


def _dedupe_terms(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for term in terms:
        key = (
            term.get("source"),
            term.get("term_id"),
            term.get("canonical"),
            term.get("search_word"),
            term.get("host_word"),
            bool(term.get("inside_word", False)),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(term)
    return output


def _terms_from_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for case in cases:
        case_terms = case.get("primary_terms") or case.get("matches") or []
        terms.extend(term for term in case_terms if isinstance(term, dict))
    return _dedupe_terms(terms)


def filter_block_for_llm(block: dict[str, Any], excluded: set[ExcludedPreparationKey]) -> dict[str, Any] | None:
    if not excluded:
        return block
    case_ids = _block_case_ids(block)
    if not case_ids:
        return None if terms_are_excluded(block.get("primary_terms", []), excluded) else block

    kept_ids = [case_id for case_id in case_ids if not terms_are_excluded(_case_terms_from_block(block, case_id), excluded)]
    if not kept_ids:
        return None

    kept_id_set = set(kept_ids)
    context = dict(block.get("context", {}))
    case_spans = [span for span in context.get("case_spans", []) if str(span.get("case_id") or "") in kept_id_set]
    if "case_spans" in context:
        context["case_spans"] = case_spans
    if "highlight_spans" in context:
        context["highlight_spans"] = [
            span
            for span in context.get("highlight_spans", [])
            if not span.get("case_id") or str(span.get("case_id") or "") in kept_id_set
        ]
    if case_spans:
        context["block_span_start"] = min(int(span.get("span_start", 0) or 0) for span in case_spans)
        context["block_span_end"] = max(int(span.get("span_end", 0) or 0) for span in case_spans)

    cases = [case for case in block.get("cases", []) if str(case.get("case_id") or "") in kept_id_set]
    primary_terms = _terms_from_cases(cases) if cases else _dedupe_terms([term for case_id in kept_ids for term in _case_terms_from_block(block, case_id)])
    llm_payload = dict(block.get("llm_payload", {}))
    llm_payload["found_terms"] = primary_terms
    llm_payload["case_count"] = len(kept_ids)

    filtered = dict(block)
    filtered["case_ids"] = kept_ids
    filtered["case_count"] = len(kept_ids)
    filtered["primary_terms"] = primary_terms
    filtered["context"] = context
    filtered["cases"] = cases
    filtered["llm_payload"] = llm_payload
    return filtered


def filter_blocks_for_llm(
    blocks: dict[str, dict[str, Any]],
    excluded: set[ExcludedPreparationKey],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    filtered: dict[str, dict[str, Any]] = {}
    excluded_cases = 0
    excluded_blocks = 0
    for block_id, block in blocks.items():
        original_case_count = _block_filter_case_count(block)
        filtered_block = filter_block_for_llm(block, excluded)
        if filtered_block is None:
            excluded_blocks += 1
            excluded_cases += original_case_count
            continue
        filtered[block_id] = filtered_block
        excluded_cases += max(0, original_case_count - _block_filter_case_count(filtered_block))
    return filtered, {"excluded_cases": excluded_cases, "excluded_blocks": excluded_blocks}


def build_block_messages(block: dict[str, Any]) -> list[dict[str, str]]:
    contract = prompt_contract()
    # Extract case_ids from case_spans
    case_spans = block.get("context", {}).get("case_spans", [])
    case_ids = [span.get("case_id") for span in case_spans if span.get("case_id")]
    
    payload = {
        "task": block.get("llm_payload", {}).get("task", "classify_clinical_recommendation_mention_block"),
        "block_id": block["block_id"],
        "case_ids": case_ids,
        "document": {"id": block["document_id"], "title": block["document_title"], "link": block["document_link"]},
        "labels": LABELS,
        "target_kinds": TARGET_KINDS,
        "found_terms": block.get("primary_terms", []),
        "case_spans": case_spans,
        "evidence_level_candidates": block.get("context", {}).get("evidence_level_candidates", []),
        "context_text": block.get("context", {}).get("text", ""),
        "output_schema": contract["output_schema"],
    }
    return [
        {"role": "system", "content": contract["system"]},
        {"role": "user", "content": contract["user_task"] + "\n\nReturn valid JSON only.\n\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def _extract_json(
    text: str,
    *,
    block: dict[str, Any] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    raw_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        debug_path = None
        if block is not None and raw_response is not None:
            debug_path = _debug_dump_invalid_openrouter_json(
                block=block,
                model=model or "unknown",
                max_tokens=max_tokens or -1,
                raw_response=raw_response,
                content=text,
                json_text=None,
                exc=ValueError("json_object_not_found"),
            )
        raise OpenRouterInvalidJsonError(
            "OpenRouter returned no JSON object | "
            f"block_id={block.get('block_id') if block else None} | "
            f"content_len={len(text)} | "
            f"debug_path={debug_path}"
        )

    json_text = text[start : end + 1]

    try:
        value = json.loads(json_text)
    except json.JSONDecodeError as exc:
        debug_path = None
        if block is not None and raw_response is not None:
            debug_path = _debug_dump_invalid_openrouter_json(
                block=block,
                model=model or "unknown",
                max_tokens=max_tokens or -1,
                raw_response=raw_response,
                content=text,
                json_text=json_text,
                exc=exc,
            )

        choice0 = ((raw_response or {}).get("choices") or [{}])[0]
        raise OpenRouterInvalidJsonError(
            "OpenRouter returned invalid JSON | "
            f"block_id={block.get('block_id') if block else None} | "
            f"document_title={block.get('document_title') if block else None} | "
            f"model={model} | "
            f"max_tokens={max_tokens} | "
            f"finish_reason={choice0.get('finish_reason')} | "
            f"usage={(raw_response or {}).get('usage')} | "
            f"content_len={len(text)} | "
            f"json_len={len(json_text)} | "
            f"json_start={start} | json_end={end} | "
            f"json_error_line={exc.lineno} | "
            f"json_error_col={exc.colno} | "
            f"json_error_pos={exc.pos} | "
            f"debug_path={debug_path} | "
            f"error_window={_slice_around(json_text, exc.pos, radius=300)!r}"
        ) from exc

    if not isinstance(value, dict):
        raise OpenRouterInvalidJsonError(
            "OpenRouter JSON root is not object | "
            f"block_id={block.get('block_id') if block else None} | "
            f"root_type={type(value).__name__}"
        )

    return value


def normalize_review_response(raw: dict[str, Any]) -> dict[str, Any]:
    label = str(raw.get("label") or "unclear")
    target_kind = str(raw.get("target_kind") or "other")
    return {
        "label": label if label in LABELS else "unclear",
        "target_kind": target_kind if target_kind in TARGET_KINDS else "other",
        "evidence_quote": raw.get("evidence_quote"),
        "reason": raw.get("reason"),
        "recommendation_strength": raw.get("recommendation_strength"),
        "evidence_level": raw.get("evidence_level"),
        "confidence": raw.get("confidence"),
    }


def _normalize_predictions(raw_predictions: list[dict[str, Any]], block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert array of predictions to dict keyed by case_id."""
    result: dict[str, dict[str, Any]] = {}
    case_ids = block.get("case_ids", [])
    for i, pred in enumerate(raw_predictions):
        case_id = pred.get("case_id") or (case_ids[i] if i < len(case_ids) else f"case:{i}")
        result[case_id] = normalize_review_response(pred)
    return result


def expand_single_prediction_to_cases(prediction: dict[str, Any], block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    case_ids = block.get("case_ids") or [span.get("case_id") for span in block.get("context", {}).get("case_spans", []) if span.get("case_id")]
    normalized = normalize_review_response(prediction)
    if not case_ids:
        return {block["block_id"]: normalized}
    return {str(case_id): dict(normalized) for case_id in case_ids}


def max_tokens_for_block(block: dict[str, Any]) -> int:
    if os.getenv("OPENROUTER_MAX_TOKENS"):
        return int(os.environ["OPENROUTER_MAX_TOKENS"])
    case_count = max(1, len(block.get("case_ids") or block.get("context", {}).get("case_spans", [])))
    return max(1200, 450 + case_count * 350)


def _base_url(base_url: str | None = None) -> str:
    return (base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")


def openrouter_key_status(api_key: str, base_url: str | None = None) -> dict[str, Any]:
    response = requests.get(
        _base_url(base_url) + "/key",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    return body.get("data", body)


def _safe_key_status(api_key: str, base_url: str | None = None) -> dict[str, Any] | None:
    try:
        return openrouter_key_status(api_key, base_url)
    except Exception as exc:  # pragma: no cover - network diagnostics only
        return {"error": str(exc)}


def _money_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not before or not after or before.get("error") or after.get("error"):
        return output
    for key in ("usage", "usage_daily", "usage_weekly", "usage_monthly", "limit_remaining"):
        if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float)):
            output[key + "_delta"] = after[key] - before[key]
    return output

class OpenRouterInvalidJsonError(RuntimeError):
    pass


def _slice_around(text: str, pos: int | None, radius: int = 900) -> str:
    if pos is None:
        return text[: radius * 2]
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end]


def _debug_dump_invalid_openrouter_json(
    *,
    block: dict[str, Any],
    model: str,
    max_tokens: int,
    raw_response: dict[str, Any],
    content: str,
    json_text: str | None,
    exc: BaseException,
) -> Path:
    debug_dir = Path(os.getenv("OPENROUTER_DEBUG_DIR", "reports/llm/openrouter_debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)

    block_id = str(block.get("block_id", "unknown")).replace(":", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = debug_dir / f"invalid_json_{ts}_{block_id}.json"

    choice0 = (raw_response.get("choices") or [{}])[0]
    err_pos = getattr(exc, "pos", None)

    payload = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_pos": err_pos,
        "block": {
            "block_id": block.get("block_id"),
            "document_id": block.get("document_id"),
            "document_title": block.get("document_title"),
            "case_count": block.get("case_count"),
            "case_ids": block.get("case_ids"),
            "primary_terms": block.get("primary_terms"),
        },
        "request": {
            "model": model,
            "max_tokens": max_tokens,
        },
        "response_meta": {
            "id": raw_response.get("id"),
            "model": raw_response.get("model"),
            "usage": raw_response.get("usage"),
            "finish_reason": choice0.get("finish_reason"),
        },
        "content_len": len(content),
        "content_head": content[:2000],
        "content_tail": content[-2000:],
        "content_error_window": _slice_around(content, err_pos),
        "json_text_len": len(json_text) if json_text is not None else None,
        "json_text_head": json_text[:2000] if json_text is not None else None,
        "json_text_tail": json_text[-2000:] if json_text is not None else None,
        "raw_response": raw_response,
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def complete_openrouter(block: dict[str, Any], *, api_key: str, model: str, base_url: str | None = None) -> dict[str, Any]:
    max_tokens = max_tokens_for_block(block)
    messages = build_block_messages(block)
    key_before = _safe_key_status(api_key, base_url)
    started_at = datetime.now(timezone.utc).isoformat()
    request_body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        _base_url(base_url) + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request_body,
        timeout=180,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    try:
        response.raise_for_status()
    except HTTPError as exc:
        body = response.text[:1000]
        raise RuntimeError(f"openrouter_http_error status={response.status_code} body={body}") from exc
    raw_response = response.json()
    if "error" in raw_response:
        import json as json_module
        error_msg = json_module.dumps(raw_response.get("error", {}), ensure_ascii=False)
        raise RuntimeError(f"OpenRouter API error: {error_msg}")
    if "choices" not in raw_response:
        import json as json_module
        raise RuntimeError(f"Invalid OpenRouter response (no 'choices'): {json_module.dumps(raw_response, ensure_ascii=False)[:500]}")
    content = raw_response["choices"][0]["message"]["content"]
    key_after = _safe_key_status(api_key, base_url)
    raw_json = _extract_json(
    content,
    block=block,
    model=model,
    max_tokens=max_tokens,
    raw_response=raw_response,
)
    
    # Handle both old single-object format and new predictions array format
    if "predictions" in raw_json and isinstance(raw_json["predictions"], list):
        predictions_dict = _normalize_predictions(raw_json["predictions"], block)
    else:
        predictions_dict = expand_single_prediction_to_cases(raw_json, block)
    
    result = {
        "predictions": predictions_dict,
        "_openrouter": {
            "started_at": started_at,
            "finished_at": finished_at,
            "request": {
                "model": model,
                "max_tokens": max_tokens,
                "message_count": len(messages),
                "message_chars": sum(len(message.get("content", "")) for message in messages),
            },
            "response": {
                "id": raw_response.get("id"),
                "model": raw_response.get("model"),
                "usage": raw_response.get("usage", {}),
                "finish_reason": (raw_response.get("choices") or [{}])[0].get("finish_reason"),
            },
            "key_before": key_before,
            "key_after": key_after,
            "key_delta": _money_delta(key_before, key_after),
        }
    }
    return result


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _confusion_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    labels = sorted(set(LABELS) | {str(row["gold_label"]) for row in rows} | {str(row["pred_label"]) for row in rows})
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in rows:
        matrix[str(row["gold_label"])][str(row["pred_label"])] += 1
    return matrix


def _per_label_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    metrics = {}
    labels = list(matrix)
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in labels if gold != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall),
        }
    return metrics


def _case_ids_for_gold_item(item: dict[str, Any]) -> list[str]:
    ids = [str(span.get("case_id")) for span in item.get("case_spans", []) if span.get("case_id")]
    if not ids:
        ids = [str(case_id) for case_id in item.get("case_ids", []) if case_id]
    if not ids and isinstance(item.get("case_golds"), dict):
        ids = [str(case_id) for case_id in item["case_golds"]]
    if not ids:
        ids = ["default"]
    return ids


def _case_golds(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ids = _case_ids_for_gold_item(item)
    golds: dict[str, dict[str, Any]] = {}
    if isinstance(item.get("gold"), dict):
        for case_id in ids:
            golds[case_id] = dict(item["gold"])
    if isinstance(item.get("case_golds"), dict):
        for case_id, gold in item["case_golds"].items():
            if isinstance(gold, dict):
                golds[str(case_id)] = dict(gold)
    return golds


def _filter_gold_item_for_block(item: dict[str, Any], block: dict[str, Any]) -> dict[str, Any] | None:
    kept_ids = set(_block_case_ids(block))
    if not kept_ids:
        return dict(item)
    filtered = dict(item)
    if isinstance(item.get("case_ids"), list):
        filtered["case_ids"] = [case_id for case_id in item["case_ids"] if str(case_id) in kept_ids]
    if isinstance(item.get("case_spans"), list):
        filtered["case_spans"] = [span for span in item["case_spans"] if str(span.get("case_id") or "") in kept_ids]
    if isinstance(item.get("case_golds"), dict):
        filtered["case_golds"] = {str(case_id): gold for case_id, gold in item["case_golds"].items() if str(case_id) in kept_ids}
    has_case_payload = any(filtered.get(key) for key in ("case_ids", "case_spans", "case_golds"))
    if not has_case_payload and not filtered.get("gold"):
        return None
    return filtered


def _filter_gold_items_for_blocks(gold_items: list[dict[str, Any]], blocks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_items: list[dict[str, Any]] = []
    for item in gold_items:
        block_id = str(item.get("block_id") or "")
        block = blocks.get(block_id)
        if block is None:
            continue
        filtered_item = _filter_gold_item_for_block(item, block)
        if filtered_item is not None:
            filtered_items.append(filtered_item)
    return filtered_items


def _gold_item_has_label(item: dict[str, Any]) -> bool:
    return any(bool(gold.get("label")) for gold in _case_golds(item).values())


def load_openrouter_prediction_sources(base_dir: str | Path, primary_output_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    base = Path(base_dir)
    primary = Path(primary_output_path).resolve() if primary_output_path else None
    candidates = [path for path in sorted(base.glob("openrouter*_results.json")) if primary is None or path.resolve() != primary]
    if primary is not None and primary.exists():
        candidates.append(primary)
    predictions: dict[str, dict[str, Any]] = {}
    for path in candidates:
        try:
            source_predictions = read_json(path).get("predictions", {})
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(source_predictions, dict):
            continue
        for block_id, prediction in source_predictions.items():
            if isinstance(prediction, dict):
                predictions[str(block_id)] = prediction
    return predictions


def _all_runner_candidates(
    blocks: dict[str, dict[str, Any]],
    gold_items: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[list[str], int, int]:
    queued: list[str] = []
    queued_set: set[str] = set()
    for item in gold_items:
        block_id = str(item.get("block_id") or "")
        if block_id in blocks and block_id not in predictions and block_id not in queued_set and _gold_item_has_label(item):
            queued.append(block_id)
            queued_set.add(block_id)
    gold_missing_count = len(queued)
    for block_id in blocks:
        if block_id not in predictions and block_id not in queued_set:
            queued.append(block_id)
            queued_set.add(block_id)
    pending_visible_count = len(queued)
    return queued, gold_missing_count, pending_visible_count


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


def score_gold(gold_items: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    label_ok = 0
    evidence_ok = 0
    strength_ok = 0
    for item in gold_items:
        block_id = item["block_id"]
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
    matrix = _confusion_rows(rows)
    return {
        "total": total,
        "label_accuracy": label_ok / total if total else 0.0,
        "recommendation_strength_accuracy": strength_ok / total if total else 0.0,
        "evidence_level_accuracy": evidence_ok / total if total else 0.0,
        "confusion_matrix": matrix,
        "per_label": _per_label_metrics(matrix),
        "rows": rows,
    }


def run_gold_openrouter(
    review_cases_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    model: str | None = None,
    env_file: str | Path | None = None,
    limit: int | None = None,
    log_progress: bool = False,
    save_each: bool = False,
    resume: bool = True,
    excluded_preparations_path: str | Path | None = None,
) -> dict[str, Any]:
    loaded_env_vars = load_env_file(env_file)
    prompt_version = prompt_contract().get("prompt_version")
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    model = model or os.getenv("OPENROUTER_MODEL")
    if not api_key or not model:
        raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL are required")
    review_cases_path = Path(review_cases_path)
    if excluded_preparations_path is None:
        excluded_preparations_path = review_cases_path.with_name("excluded_preparations.json")
    excluded_keys = load_excluded_preparation_keys(excluded_preparations_path)
    raw_blocks = iter_blocks(read_json(review_cases_path))
    blocks, excluded_stats = filter_blocks_for_llm(raw_blocks, excluded_keys)
    gold_items = read_json(gold_path)["items"][:limit]
    visible_gold_items = _filter_gold_items_for_blocks(gold_items, blocks)
    output_path = Path(output_path)
    predictions = {}
    if resume and output_path.exists():
        old_report = read_json(output_path)
        predictions.update(old_report.get("predictions", {}))
        if old_report.get("prompt_version") != prompt_version and log_progress:
            print(
                json.dumps(
                    {
                        "event": "resume_kept_existing_predictions_prompt_version_mismatch",
                        "old_prompt_version": old_report.get("prompt_version"),
                        "current_prompt_version": prompt_version,
                        "kept_prediction_count": len(predictions),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    for index, item in enumerate(gold_items, start=1):
        if item["block_id"] in predictions:
            if log_progress:
                print(json.dumps({"event": "gold_item_skip", "index": index, "total": len(gold_items), "block_id": item["block_id"]}, ensure_ascii=False), flush=True)
            continue
        if item["block_id"] not in blocks:
            if log_progress:
                print(json.dumps({"event": "gold_item_excluded", "index": index, "total": len(gold_items), "block_id": item["block_id"]}, ensure_ascii=False), flush=True)
            continue
        block = blocks[item["block_id"]]
        if log_progress:
            print(json.dumps({"event": "gold_item_start", "index": index, "total": len(gold_items), "block_id": item["block_id"]}, ensure_ascii=False), flush=True)
        predictions[item["block_id"]] = complete_openrouter(block, api_key=api_key, model=model)
        if save_each:
            checkpoint_visible_gold_items = _filter_gold_items_for_blocks(gold_items[:index], blocks)
            report = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "review_cases": str(review_cases_path),
                "gold": str(gold_path),
                "excluded_preparations": str(excluded_preparations_path) if excluded_preparations_path else None,
                "env_file": str(env_file) if env_file else None,
                "loaded_env_vars": loaded_env_vars,
                "model": model,
                "prompt_version": prompt_version,
                "completed": len(predictions),
                "completed_all": len(predictions),
                "completed_visible": sum(1 for block_id in predictions if block_id in blocks),
                **excluded_stats,
                "predictions": predictions,
                "score": score_gold(checkpoint_visible_gold_items, predictions),
            }
            write_json(output_path, report, indent=2)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_cases": str(review_cases_path),
        "gold": str(gold_path),
        "excluded_preparations": str(excluded_preparations_path) if excluded_preparations_path else None,
        "env_file": str(env_file) if env_file else None,
        "loaded_env_vars": loaded_env_vars,
        "model": model,
        "prompt_version": prompt_version,
        "completed": len(predictions),
        "completed_all": len(predictions),
        "completed_visible": sum(1 for block_id in predictions if block_id in blocks),
        **excluded_stats,
        "predictions": predictions,
        "score": score_gold(visible_gold_items, predictions),
    }
    write_json(output_path, report, indent=2)
    return report


def run_openrouter_all(
    review_cases_path: str | Path,
    output_path: str | Path,
    *,
    gold_path: str | Path | None = None,
    api_key: str | None = None,
    model: str | None = None,
    env_file: str | Path | None = None,
    excluded_preparations_path: str | Path | None = None,
    limit: int | None = None,
    log_progress: bool = False,
    save_each: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    loaded_env_vars = load_env_file(env_file)
    prompt_version = prompt_contract().get("prompt_version")
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    model = model or os.getenv("OPENROUTER_MODEL")
    if not api_key or not model:
        raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL are required")

    review_cases_path = Path(review_cases_path)
    output_path = Path(output_path)

    if excluded_preparations_path is None:
        excluded_preparations_path = review_cases_path.with_name("excluded_preparations.json")

    if gold_path is None:
        default_gold = review_cases_path.with_name("llm_gold_40.json")
        gold_path = default_gold if default_gold.exists() else None

    excluded_keys = load_excluded_preparation_keys(excluded_preparations_path)
    raw_blocks = iter_blocks(read_json(review_cases_path))
    blocks, excluded_stats = filter_blocks_for_llm(raw_blocks, excluded_keys)

    gold_items = read_json(gold_path).get("items", []) if gold_path and Path(gold_path).exists() else []

    predictions = load_openrouter_prediction_sources(output_path.parent, output_path) if resume else {}

    candidates, gold_missing_count, pending_visible_count = _all_runner_candidates(
        blocks,
        gold_items,
        predictions,
    )
    selected_block_ids = candidates[:limit] if limit is not None else candidates

    completed_before = sum(
        1
        for value in predictions.values()
        if isinstance(value, dict) and not value.get("_error")
    )
    failed_before = sum(
        1
        for value in predictions.values()
        if isinstance(value, dict) and value.get("_error")
    )

    new_completed = 0
    new_failed = 0

    def build_report(current_selected: list[str]) -> dict[str, Any]:
        visible_gold_items = _filter_gold_items_for_blocks(gold_items, blocks)

        failed = sum(
            1
            for value in predictions.values()
            if isinstance(value, dict) and value.get("_error")
        )
        completed_ok = sum(
            1
            for value in predictions.values()
            if isinstance(value, dict) and not value.get("_error")
        )
        completed_visible = sum(
            1
            for block_id, value in predictions.items()
            if block_id in blocks and isinstance(value, dict) and not value.get("_error")
        )
        failed_visible = sum(
            1
            for block_id, value in predictions.items()
            if block_id in blocks and isinstance(value, dict) and value.get("_error")
        )

        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "review_cases": str(review_cases_path),
            "gold": str(gold_path) if gold_path else None,
            "excluded_preparations": str(excluded_preparations_path) if excluded_preparations_path else None,
            "env_file": str(env_file) if env_file else None,
            "loaded_env_vars": loaded_env_vars,
            "model": model,
            "prompt_version": prompt_version,
            "selection_strategy": "gold_labeled_without_prediction_first_then_visible_pending",
            "limit": limit,
            "selected_block_ids": current_selected,
            "gold_missing_count": gold_missing_count,
            "pending_visible_count": pending_visible_count,
            "new_completed": new_completed,
            "new_failed": new_failed,
            "completed": completed_visible,
            "completed_ok": completed_ok,
            "completed_all": completed_ok,
            "attempted_all": len(predictions),
            "completed_visible": completed_visible,
            "failed": failed,
            "failed_visible": failed_visible,
            "completed_before": completed_before,
            "failed_before": failed_before,
            **excluded_stats,
            "predictions": predictions,
            "score": score_gold(visible_gold_items, predictions) if visible_gold_items else {},
        }

    for index, block_id in enumerate(selected_block_ids, start=1):
        block = blocks[block_id]

        if log_progress:
            print(
                json.dumps(
                    {
                        "event": "all_item_start",
                        "index": index,
                        "total": len(selected_block_ids),
                        "block_id": block_id,
                        "priority": "gold_missing" if index <= gold_missing_count else "visible_pending",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        try:
            predictions[block_id] = complete_openrouter(block, api_key=api_key, model=model)
            new_completed += 1

            if log_progress:
                print(
                    json.dumps(
                        {
                            "event": "all_item_done",
                            "index": index,
                            "total": len(selected_block_ids),
                            "block_id": block_id,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        except Exception as exc:
            predictions[block_id] = {
                "predictions": {},
                "_error": {
                    "type": type(exc).__name__,
                    "message": str(exc)[:4000],
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "block_id": block_id,
                    "document_id": block.get("document_id"),
                    "document_title": block.get("document_title"),
                    "case_ids": block.get("case_ids", []),
                    "case_count": block.get("case_count"),
                    "primary_terms": block.get("primary_terms", []),
                },
            }
            new_failed += 1

            if log_progress:
                print(
                    json.dumps(
                        {
                            "event": "all_item_error",
                            "index": index,
                            "total": len(selected_block_ids),
                            "block_id": block_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if save_each:
            write_json(output_path, build_report(selected_block_ids[:index]), indent=2)

    report = build_report(selected_block_ids)
    write_json(output_path, report, indent=2)
    return report
