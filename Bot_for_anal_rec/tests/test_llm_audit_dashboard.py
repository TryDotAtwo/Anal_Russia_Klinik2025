from __future__ import annotations

from pathlib import Path

from anal_russia_klinik.jsonio import write_json
from anal_russia_klinik.llm_audit_dashboard import AuditState
from anal_russia_klinik.llm_review_cases import prompt_contract
from anal_russia_klinik.llm_review_openrouter import load_env_file


def _review_report() -> dict:
    return {
        "clinical_recommendations": [
            {
                "document_id": "doc:1",
                "document_title": "Doc One",
                "document_link": "https://example.test/doc1",
                "llm_blocks": [
                    {
                        "block_id": "block:1",
                        "document_id": "doc:1",
                        "document_title": "Doc One",
                        "document_link": "https://example.test/doc1",
                        "case_count": 1,
                        "case_ids": ["case:1"],
                        "primary_terms": [{"canonical": "Drug", "host_word": "Drug", "inside_word": False}],
                        "context": {
                            "context_start": 10,
                            "context_end": 40,
                            "block_span_start": 5,
                            "block_span_end": 9,
                            "text": "Text Drug context.",
                            "case_spans": [{"case_id": "case:1", "span_start": 5, "span_end": 9, "text": "Drug"}],
                            "highlight_spans": [{"case_id": "case:1", "span_start": 5, "span_end": 9, "text": "Drug"}],
                            "evidence_level_candidates": [],
                        },
                        "cases": [
                            {
                                "case_id": "case:1",
                                "primary_terms": [{"canonical": "Drug", "host_word": "Drug", "inside_word": False}],
                            }
                        ],
                    },
                    {
                        "block_id": "block:2",
                        "document_id": "doc:1",
                        "document_title": "Doc One",
                        "document_link": "https://example.test/doc1",
                        "case_count": 1,
                        "case_ids": ["case:2"],
                        "primary_terms": [{"canonical": "Other", "host_word": "Other", "inside_word": False}],
                        "context": {
                            "context_start": 50,
                            "context_end": 80,
                            "block_span_start": 6,
                            "block_span_end": 11,
                            "text": "Text Other context.",
                            "case_spans": [{"case_id": "case:2", "span_start": 6, "span_end": 11, "text": "Other"}],
                            "highlight_spans": [{"case_id": "case:2", "span_start": 6, "span_end": 11, "text": "Other"}],
                            "evidence_level_candidates": [],
                        },
                        "cases": [
                            {
                                "case_id": "case:2",
                                "primary_terms": [{"canonical": "Other", "host_word": "Other", "inside_word": False}],
                            }
                        ],
                    },
                ],
            }
        ]
    }


def test_prompt_contract_contains_strict_indomethacin_rule() -> None:
    task = prompt_contract()["user_task"]

    assert "Metacin" in task
    assert "indomethacin" in task
    assert "inside_word=true is not automatic error" in task
    assert "different drugs" in task
    assert "Децит" in task
    assert "рисдиплам" in task


def test_llm_audit_dashboard_summarizes_case_cost(tmp_path: Path) -> None:
    write_json(
        tmp_path / "llm_gold_40.json",
        {
            "items": [
                {
                    "block_id": "block:1",
                    "document_title": "Doc",
                    "primary_terms": [{"canonical": "Drug", "host_word": "Drug"}],
                    "gold": {"label": "error"},
                }
            ]
        },
    )
    write_json(
        tmp_path / "openrouter_gold40_results.json",
        {
            "model": "openai/gpt-5.5",
            "prompt_version": prompt_contract()["prompt_version"],
            "predictions": {
                "block:1": {
                    "label": "error",
                    "_openrouter": {
                        "key_delta": {"usage_delta": 0.015},
                        "response": {"usage": {"total_tokens": 123}},
                    },
                }
            },
        },
    )

    state = AuditState(tmp_path)
    payload = state.state_payload()
    cases = state.cases_page({})

    assert payload["completed"] == 1
    assert payload["score"]["label_accuracy"] == 1.0
    assert cases["items"][0]["usage_delta"] == 0.015
    assert cases["items"][0]["total_tokens"] == 123


def test_load_env_file_can_override_changed_api_key(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "openrouter.env"
    env_file.write_text("OPENROUTER_API_KEY=new-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "old-key")

    loaded = load_env_file(env_file, override=True)

    assert loaded == 1
    assert __import__("os").environ["OPENROUTER_API_KEY"] == "new-key"


def test_llm_audit_dashboard_updates_gold_file(tmp_path: Path) -> None:
    gold_path = tmp_path / "llm_gold_40.json"
    write_json(
        gold_path,
        {
            "items": [
                {
                    "block_id": "block:1",
                    "document_title": "Doc",
                    "primary_terms": [],
                    "gold": {"label": "error", "target_kind": "other"},
                }
            ]
        },
    )
    write_json(tmp_path / "openrouter_gold40_results.json", {"predictions": {}})
    state = AuditState(tmp_path)

    result = state.update_gold(
        {
            "index": 1,
            "label": "recommendation",
            "target_kind": "drug",
            "recommendation_strength": "C",
            "evidence_level": "4",
            "evidence_quote": "Рекомендуется препарат.",
            "reason": "manual check",
        }
    )

    assert result["item"]["gold"]["label"] == "recommendation"
    assert result["item"]["gold"]["evidence_level"] == "4"
    assert state.gold_items()[0]["gold"]["recommendation_strength"] == "C"


def test_llm_audit_dashboard_lists_all_review_blocks_and_nested_predictions(tmp_path: Path) -> None:
    write_json(tmp_path / "llm_review_cases.json", _review_report())
    write_json(
        tmp_path / "llm_gold_40.json",
        {
            "items": [
                {
                    "block_id": "block:1",
                    "case_spans": [{"case_id": "case:1"}],
                    "case_golds": {"case:1": {"label": "error", "target_kind": "drug"}},
                }
            ]
        },
    )
    write_json(
        tmp_path / "openrouter_gold40_results.json",
        {
            "predictions": {
                "block:1": {
                    "predictions": {"case:1": {"label": "error", "target_kind": "drug"}},
                    "_openrouter": {"response": {"usage": {"total_tokens": 17}}},
                }
            }
        },
    )

    state = AuditState(tmp_path)
    payload = state.state_payload()
    page = state.cases_page({"limit": ["10"]})
    case_payload = state.case_payload({"block_id": ["block:1"]})

    assert payload["review_block_count"] == 2
    assert payload["completed"] == 1
    assert page["total"] == 2
    assert page["items"][0]["status"] == "ok"
    assert page["items"][1]["status"] == "pending"
    assert case_payload["cases"][0]["prediction"]["label"] == "error"
    assert case_payload["context"]["block_span_start"] == 5


def test_llm_audit_dashboard_lists_predicted_blocks_first(tmp_path: Path) -> None:
    write_json(tmp_path / "llm_review_cases.json", _review_report())
    write_json(
        tmp_path / "openrouter_gold40_results.json",
        {
            "predictions": {
                "block:2": {
                    "predictions": {"case:2": {"label": "recommendation", "target_kind": "drug"}},
                }
            }
        },
    )

    state = AuditState(tmp_path)
    page = state.cases_page({"limit": ["10"]})

    assert [item["block_id"] for item in page["items"]] == ["block:2", "block:1"]
    assert page["items"][0]["predicted_count"] == 1
    assert page["items"][1]["predicted_count"] == 0


def test_llm_audit_dashboard_request_body_uses_dynamic_token_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    report = _review_report()
    block = report["clinical_recommendations"][0]["llm_blocks"][0]
    block["case_ids"] = [f"case:{index}" for index in range(4)]
    block["case_count"] = 4
    block["context"]["case_spans"] = [
        {"case_id": f"case:{index}", "span_start": index, "span_end": index + 1, "text": "D"}
        for index in range(4)
    ]
    write_json(tmp_path / "llm_review_cases.json", report)

    state = AuditState(tmp_path)
    payload = state.case_payload({"block_id": ["block:1"]})

    assert payload["request_body"]["max_tokens"] == 1850


def test_llm_audit_dashboard_update_gold_creates_item_from_review_block(tmp_path: Path) -> None:
    write_json(tmp_path / "llm_review_cases.json", _review_report())
    state = AuditState(tmp_path)

    result = state.update_gold(
        {
            "block_id": "block:2",
            "case_id": "case:2",
            "label": "recommendation",
            "target_kind": "drug",
            "comment": "manual note",
        }
    )

    saved = state.gold_items()[0]
    assert result["item"]["block_id"] == "block:2"
    assert saved["case_golds"]["case:2"]["label"] == "recommendation"
    assert saved["case_golds"]["case:2"]["comment"] == "manual note"
    assert saved["preview"] == "Text Other context."


def test_llm_audit_dashboard_excludes_case_from_model_stats(tmp_path: Path) -> None:
    write_json(tmp_path / "llm_review_cases.json", _review_report())
    write_json(
        tmp_path / "llm_gold_40.json",
        {
            "items": [
                {
                    "block_id": "block:1",
                    "case_spans": [{"case_id": "case:1"}],
                    "case_golds": {
                        "case:1": {
                            "label": "recommendation",
                            "target_kind": "drug",
                            "exclude_from_model_stats": True,
                        }
                    },
                }
            ]
        },
    )
    write_json(
        tmp_path / "openrouter_gold40_results.json",
        {"predictions": {"block:1": {"label": "error", "target_kind": "drug"}}},
    )

    state = AuditState(tmp_path)

    assert state.state_payload()["score"]["total"] == 0


def test_llm_audit_dashboard_excludes_preparation_from_dashboard_denominator(tmp_path: Path) -> None:
    write_json(tmp_path / "llm_review_cases.json", _review_report())
    write_json(
        tmp_path / "openrouter_gold40_results.json",
        {"predictions": {"block:1": {"predictions": {"case:1": {"label": "error", "target_kind": "drug"}}}}},
    )
    state = AuditState(tmp_path)

    result = state.exclude_preparation(
        {
            "term": {
                "source": "mediq",
                "term_id": "",
                "canonical": "Drug",
                "search_word": "Drug",
                "host_word": "Drug",
            }
        }
    )

    assert result["state"]["excluded_preparation_count"] == 1
    assert result["state"]["review_block_count"] == 1
    assert result["state"]["completed_all"] == 1
    assert result["state"]["completed_visible"] == 0
    assert result["state"]["completed"] == 0
    assert result["state"]["excluded_cases"] == 1
    assert result["state"]["excluded_blocks"] == 1
    assert state.cases_page({"limit": ["10"]})["items"][0]["block_id"] == "block:2"
