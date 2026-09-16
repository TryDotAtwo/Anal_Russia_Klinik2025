from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anal_russia_klinik.llm_review_openrouter import (  # noqa: E402
    _base_url,
    _extract_json,
    _money_delta,
    _safe_key_status,
    _normalize_predictions,
    backup_once,
    build_block_messages,
    expand_single_prediction_to_cases,
    filter_blocks_for_llm,
    iter_blocks,
    load_env_file,
    load_excluded_preparation_keys,
    read_json,
    score_gold,
    write_json_atomic,
)


def _filter_gold_items_for_blocks(gold_items: list[dict[str, Any]], blocks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    block_ids = set(blocks)
    return [item for item in gold_items if item.get("block_id") in block_ids]


def _is_error_prediction(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("_error"))


def _complete_openrouter_without_max_tokens(
    block: dict[str, Any],
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    messages = build_block_messages(block)
    key_before = _safe_key_status(api_key, base_url)
    started_at = datetime.now(timezone.utc).isoformat()
    request_body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    response = requests.post(
        _base_url(base_url) + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request_body,
        timeout=240,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    try:
        response.raise_for_status()
    except HTTPError as exc:
        body = response.text[:1000]
        raise RuntimeError(f"openrouter_http_error status={response.status_code} body={body}") from exc

    raw_response = response.json()
    if "error" in raw_response:
        error_msg = json.dumps(raw_response.get("error", {}), ensure_ascii=False)
        raise RuntimeError(f"OpenRouter API error: {error_msg}")
    if "choices" not in raw_response:
        raise RuntimeError(f"Invalid OpenRouter response (no 'choices'): {json.dumps(raw_response, ensure_ascii=False)[:500]}")

    content = raw_response["choices"][0]["message"]["content"]
    key_after = _safe_key_status(api_key, base_url)
    raw_json = _extract_json(
        content,
        block=block,
        model=model,
        max_tokens=-1,
        raw_response=raw_response,
    )

    if "predictions" in raw_json and isinstance(raw_json["predictions"], list):
        predictions_dict = _normalize_predictions(raw_json["predictions"], block)
    else:
        predictions_dict = expand_single_prediction_to_cases(raw_json, block)

    return {
        "predictions": predictions_dict,
        "_openrouter": {
            "started_at": started_at,
            "finished_at": finished_at,
            "request": {
                "model": model,
                "max_tokens": None,
                "max_tokens_policy": "omitted_from_request",
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
        },
    }


def rerun_openrouter_errors_no_max_tokens(
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
    dry_run: bool = False,
) -> dict[str, Any]:
    loaded_env_vars = load_env_file(env_file)
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

    report = read_json(output_path)
    predictions = report.get("predictions", {})
    if not isinstance(predictions, dict):
        raise ValueError("output predictions must be a JSON object")

    excluded_keys = load_excluded_preparation_keys(excluded_preparations_path)
    blocks, excluded_stats = filter_blocks_for_llm(iter_blocks(read_json(review_cases_path)), excluded_keys)
    gold_items = read_json(gold_path).get("items", []) if gold_path and Path(gold_path).exists() else []
    visible_gold_items = _filter_gold_items_for_blocks(gold_items, blocks)

    error_block_ids = [block_id for block_id, value in predictions.items() if _is_error_prediction(value) and block_id in blocks]
    selected_block_ids = error_block_ids[:limit] if limit is not None else error_block_ids

    if dry_run:
        return {
            "output": str(output_path),
            "loaded_env_vars": loaded_env_vars,
            "model": model,
            "error_blocks_total": len(error_block_ids),
            "selected_block_ids": selected_block_ids,
            "dry_run": True,
        }

    backup_path = backup_once(output_path)
    new_completed = 0
    new_failed = 0

    def build_report(current_selected: list[str]) -> dict[str, Any]:
        failed = sum(1 for value in predictions.values() if _is_error_prediction(value))
        completed_ok = sum(1 for value in predictions.values() if isinstance(value, dict) and not value.get("_error"))
        completed_visible = sum(
            1
            for block_id, value in predictions.items()
            if block_id in blocks and isinstance(value, dict) and not value.get("_error")
        )
        failed_visible = sum(1 for block_id, value in predictions.items() if block_id in blocks and _is_error_prediction(value))
        updated = dict(report)
        updated.update(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "review_cases": str(review_cases_path),
                "gold": str(gold_path) if gold_path else None,
                "excluded_preparations": str(excluded_preparations_path) if excluded_preparations_path else None,
                "env_file": str(env_file) if env_file else None,
                "loaded_env_vars": loaded_env_vars,
                "model": model,
                "selection_strategy": "rerun_error_blocks_without_max_tokens",
                "limit": limit,
                "selected_block_ids": current_selected,
                "rerun_error_blocks_total": len(error_block_ids),
                "rerun_error_new_completed": new_completed,
                "rerun_error_new_failed": new_failed,
                "completed": completed_visible,
                "completed_ok": completed_ok,
                "completed_all": completed_ok,
                "attempted_all": len(predictions),
                "completed_visible": completed_visible,
                "failed": failed,
                "failed_visible": failed_visible,
                **excluded_stats,
                "predictions": predictions,
                "score": score_gold(visible_gold_items, predictions) if visible_gold_items else {},
                "backup_path": str(backup_path) if backup_path else None,
            }
        )
        return updated

    for index, block_id in enumerate(selected_block_ids, start=1):
        block = blocks[block_id]
        if log_progress:
            print(json.dumps({"event": "error_rerun_start", "index": index, "total": len(selected_block_ids), "block_id": block_id}, ensure_ascii=False), flush=True)
        try:
            predictions[block_id] = _complete_openrouter_without_max_tokens(block, api_key=api_key, model=model)
            new_completed += 1
            if log_progress:
                print(json.dumps({"event": "error_rerun_done", "index": index, "total": len(selected_block_ids), "block_id": block_id}, ensure_ascii=False), flush=True)
        except Exception as exc:
            predictions[block_id] = {
                "predictions": {},
                "_error": {
                    "type": type(exc).__name__,
                    "message": str(exc)[:4000],
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "max_tokens_policy": "omitted_from_request",
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
                        {"event": "error_rerun_error", "index": index, "total": len(selected_block_ids), "block_id": block_id, "error_type": type(exc).__name__, "error": str(exc)[:500]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        write_json_atomic(output_path, build_report(selected_block_ids[:index]), indent=2)

    final_report = build_report(selected_block_ids)
    write_json_atomic(output_path, final_report, indent=2)
    return final_report


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="rerun-openrouter-errors-no-max-tokens")
    parser.add_argument("--review-cases", default=str(base_dir / "llm_review_cases.json"))
    parser.add_argument("--gold", default=str(base_dir / "llm_gold_40.json"))
    parser.add_argument("--output", default=str(base_dir / "openrouter_all_results.json"))
    parser.add_argument("--excluded-preparations", default=str(base_dir / "excluded_preparations.json"))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / "config" / "openrouter.env"))
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = rerun_openrouter_errors_no_max_tokens(
        args.review_cases,
        args.output,
        gold_path=args.gold,
        env_file=args.env_file,
        model=args.model,
        excluded_preparations_path=args.excluded_preparations,
        limit=args.limit,
        log_progress=not args.quiet,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "dry_run": report.get("dry_run", False),
                "error_blocks_total": report.get("error_blocks_total", report.get("rerun_error_blocks_total")),
                "selected_count": len(report.get("selected_block_ids", [])),
                "rerun_error_new_completed": report.get("rerun_error_new_completed"),
                "rerun_error_new_failed": report.get("rerun_error_new_failed"),
                "completed_all": report.get("completed_all"),
                "attempted_all": report.get("attempted_all"),
                "failed": report.get("failed"),
                "failed_visible": report.get("failed_visible"),
                "backup_path": report.get("backup_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
