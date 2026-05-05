from __future__ import annotations

import json
from pathlib import Path

import pytest

from anal_russia_klinik.aho_report import build_aho_host_word_report
from anal_russia_klinik.aho_compact import apply_host_word_filters, build_compact_report, deduplicate_compact_rows
from anal_russia_klinik.aho_detailed_filter import write_filtered_detailed_report
from anal_russia_klinik.aho_filter_store import merge_filter_words, read_filter_words, update_filter_words
from anal_russia_klinik.aho_location_groups import build_location_groups
from anal_russia_klinik.llm_review_cases import build_review_cases_report
from anal_russia_klinik.llm_review_openrouter import (
    build_block_messages,
    collect_excluded_preparation_keys,
    expand_single_prediction_to_cases,
    filter_block_for_llm,
    max_tokens_for_block,
    run_openrouter_all,
    run_gold_openrouter,
    score_gold,
)
from anal_russia_klinik.config import default_config
from anal_russia_klinik.dictionaries import load_marker_terms, load_preparation_terms, variants_for_term
from anal_russia_klinik.jsonio import read_json, write_json
from anal_russia_klinik.llm import build_provider
from anal_russia_klinik.manual_filters import apply_manual_filters, load_filter_dictionary
from anal_russia_klinik.matcher import build_raw_matches
from anal_russia_klinik.pipeline import run_pipeline
from anal_russia_klinik.text import load_legacy_clinical_json


ONKO = "\u041e\u043d\u043a\u043e"
ONCOLOGY = "\u041e\u043d\u043a\u043e\u043b\u043e\u0433\u0438\u044f"


def _clinical_fixture(path: Path, text: str) -> Path:
    write_json(
        path,
        {"recommendations": {"kr_1": {"text": text, "title": "KR", "link": "memory"}}},
    )
    return path


def test_substring_recall_keeps_onko_inside_oncology(tmp_path: Path) -> None:
    markers = tmp_path / "markers.json"
    write_json(markers, [ONKO])
    clinical = _clinical_fixture(tmp_path / "clinical.json", f"{ONCOLOGY}: term. {ONKO} standalone.")

    raw = [
        match.to_dict()
        for match in build_raw_matches(
            load_legacy_clinical_json(clinical),
            load_marker_terms(markers),
            context_before=10,
            context_after=10,
        )
    ]

    assert len(raw) == 2
    assert raw[0]["matched_text"] == ONKO
    assert raw[0]["word_text"] == ONCOLOGY
    assert raw[0]["inside_word"] is True


def test_manual_filter_rejects_false_positive_after_raw_match(tmp_path: Path) -> None:
    markers = tmp_path / "markers.json"
    write_json(markers, [ONKO])
    clinical = _clinical_fixture(tmp_path / "clinical.json", f"{ONCOLOGY} and {ONKO}.")
    filter_file = tmp_path / "manual_filters.csv"
    filter_file.write_text(
        "term,host_word,action,reason,confidence,created_at\n"
        f"{ONKO},{ONCOLOGY},reject,false_positive,1,2026-04-29\n",
        encoding="utf-8",
    )

    raw = [
        match.to_dict()
        for match in build_raw_matches(load_legacy_clinical_json(clinical), load_marker_terms(markers))
    ]
    kept, rejected = apply_manual_filters(raw, load_filter_dictionary(filter_file))

    assert len(raw) == 2
    assert len(rejected) == 1
    assert len(kept) == 1
    assert rejected[0]["match"]["word_text"] == ONCOLOGY


def test_pipeline_exports_legacy_json_and_dashboard(tmp_path: Path) -> None:
    clinical = _clinical_fixture(tmp_path / "clinical.json", f"{ONCOLOGY} and {ONKO}.")
    markers = tmp_path / "markers.json"
    blacklist = tmp_path / "blacklist.json"
    preparations = tmp_path / "preparations.json"
    filters = tmp_path / "manual_filters.csv"
    output_dir = tmp_path / "run"
    write_json(markers, [ONKO])
    write_json(blacklist, [])
    write_json(preparations, [])
    filters.write_text(
        "term,host_word,action,reason,confidence,created_at\n"
        f"{ONKO},{ONCOLOGY},reject,false_positive,1,2026-04-29\n",
        encoding="utf-8",
    )
    config = default_config(
        clinical_json_path=clinical,
        markers_path=markers,
        blacklist_path=blacklist,
        preparations_path=preparations,
        manual_filters_path=filters,
        output_dir=output_dir,
        provider="fake",
    )

    paths = run_pipeline(config, provider_name="fake")
    legacy = read_json(paths.legacy_json)

    assert paths.raw_matches.exists()
    assert paths.classified_csv.exists()
    assert paths.dashboard_html.exists()
    assert "clinical_recommendations" in legacy
    assert legacy["clinical_recommendations"][0]["drugs_mentioned"][0]["mentions"][0]["char_start"] == len(ONCOLOGY) + 5


def test_aho_compact_report_removes_occurrences_and_filters_host_words(tmp_path: Path) -> None:
    source = tmp_path / "host_words_by_search_word.json"
    write_json(
        source,
        {
            "schema_version": 3,
            "generated_at": "2026-04-29T00:00:00+00:00",
            "workers": 1,
            "document_count": 1,
            "summary": {"total_match_count": 50},
            "by_search_word": [
                {
                    "source": "blacklist",
                    "term_id": "t1",
                    "canonical": "\u0410\u0433\u0440\u0438",
                    "search_word": "\u0410\u0433\u0440\u0438",
                    "normalized_search_word": "\u0430\u0433\u0440\u0438",
                    "match_count": 50,
                    "host_words": [
                        {
                            "host_word": "\u043f\u0430\u0440\u0430\u0433\u0440\u0438\u043f\u043f\u0430",
                            "inside_word": True,
                            "count": 43,
                            "occurrences": [{"document_id": "1"}],
                        },
                        {
                            "host_word": "\u0430\u0433\u0440\u0438",
                            "inside_word": False,
                            "count": 7,
                            "occurrences": [{"document_id": "1"}],
                        },
                    ],
                }
            ],
        },
    )

    compact = build_compact_report(source)
    entry = compact["by_search_word"][0]
    assert entry["match_count"] == 50
    assert "occurrences" not in entry["host_words"][0]
    assert entry["host_words"][0]["host_word"] == "\u043f\u0430\u0440\u0430\u0433\u0440\u0438\u043f\u043f\u0430"

    filtered = apply_host_word_filters(compact, ["\u043f\u0430\u0440\u0430\u0433\u0440\u0438\u043f\u043f\u0430"])
    filtered_entry = filtered["by_search_word"][0]
    assert filtered["removed_match_count"] == 43
    assert filtered_entry["match_count"] == 7
    assert filtered_entry["host_words"][0]["host_word"] == "\u0430\u0433\u0440\u0438"


def test_detailed_filter_keeps_occurrences_and_removes_filtered_hosts(tmp_path: Path) -> None:
    source = tmp_path / "host_words_by_search_word.json"
    filters = tmp_path / "host_word_filters.json"
    output = tmp_path / "host_words_by_search_word_filtred.json"
    write_json(
        source,
        {
            "schema_version": 3,
            "generated_at": "2026-04-30T00:00:00+00:00",
            "search_word_count": 1,
            "summary": {"total_match_count": 3},
            "by_search_word": [
                {
                    "source": "blacklist",
                    "term_id": "t1",
                    "canonical": "\u0410\u0433\u0440\u0438",
                    "search_word": "\u0410\u0433\u0440\u0438",
                    "normalized_search_word": "\u0430\u0433\u0440\u0438",
                    "match_count": 3,
                    "inside_word_match_count": 2,
                    "host_words_count": 2,
                    "inside_host_words_count": 1,
                    "host_words": [
                        {"host_word": "\u043f\u0430\u0440\u0430\u0433\u0440\u0438\u043f\u043f\u0430", "inside_word": True, "count": 2, "occurrences": [{"document_id": "1"}]},
                        {"host_word": "\u0430\u0433\u0440\u0438", "inside_word": False, "count": 1, "occurrences": [{"document_id": "2"}]},
                    ],
                    "inside_host_words": [
                        {"host_word": "\u043f\u0430\u0440\u0430\u0433\u0440\u0438\u043f\u043f\u0430", "inside_word": True, "count": 2, "occurrences": [{"document_id": "1"}]},
                    ],
                }
            ],
        },
    )
    write_json(filters, {"host_words": ["\u043f\u0430\u0440\u0430\u0433\u0440\u0438\u043f\u043f\u0430"]})

    result = write_filtered_detailed_report(source, filters, output)
    data = read_json(output)
    row = data["by_search_word"][0]

    assert result["removed_match_count"] == 2
    assert row["match_count"] == 1
    assert row["inside_word_match_count"] == 0
    assert row["host_words"][0]["occurrences"][0]["document_id"] == "2"


def test_location_groups_preserve_duplicate_matches_by_position() -> None:
    occurrence = {
        "document_id": "1",
        "document_title": "Doc",
        "document_link": "memory",
        "char_start": 10,
        "char_end": 14,
        "word_start": 10,
        "word_end": 14,
        "matched_text": "test",
        "page": None,
        "section": None,
    }
    report = {
        "by_search_word": [
            {
                "source": "blacklist",
                "term_id": "b1",
                "canonical": "A",
                "search_word": "A",
                "normalized_search_word": "a",
                "host_words": [{"host_word": "test", "inside_word": False, "occurrences": [occurrence]}],
            },
            {
                "source": "mediq",
                "term_id": "m1",
                "canonical": "B",
                "search_word": "B",
                "normalized_search_word": "b",
                "host_words": [{"host_word": "test", "inside_word": False, "occurrences": [dict(occurrence)]}],
            },
        ]
    }

    groups = build_location_groups(report)

    assert len(groups) == 1
    assert groups[0]["match_count"] == 2
    assert groups[0]["duplicate_match_count"] == 1
    assert {item["canonical"] for item in groups[0]["matches"]} == {"A", "B"}


def test_llm_review_cases_include_context_spans_and_validation_schema(tmp_path: Path) -> None:
    clinical = _clinical_fixture(
        tmp_path / "clinical.json",
        "AAA BBB CCC. УУР A, УДД 2. Drug should be checked. ZZZ",
    )
    locations = tmp_path / "locations.json"
    text = "AAA BBB CCC. УУР A, УДД 2. Drug should be checked. ZZZ"
    start = text.index("Drug")
    write_json(
        locations,
        {
            "by_location": [
                {
                    "location_id": "loc:1",
                    "document_id": "kr_1",
                    "char_start": start,
                    "char_end": start + 4,
                    "word_start": start,
                    "word_end": start + 4,
                    "matched_text": "Drug",
                    "page": None,
                    "section": "3. Treatment",
                    "matches": [
                        {
                            "source": "mediq",
                            "term_id": "m1",
                            "canonical": "Drug",
                            "search_word": "Drug",
                            "normalized_search_word": "drug",
                            "host_word": "Drug",
                            "inside_word": False,
                        }
                    ],
                }
            ]
        },
    )

    report = build_review_cases_report(locations, clinical, window_chars=20)
    case = report["clinical_recommendations"][0]["cases"][0]
    block = report["clinical_recommendations"][0]["llm_blocks"][0]

    assert case["context"]["text"][case["context"]["span_start"] : case["context"]["span_end"]] == "Drug"
    assert block["context"]["case_spans"][0]["text"] == "Drug"
    assert report["llm_review_unit"] == "clinical_recommendations[*].llm_blocks[*]"
    assert case["context"]["evidence_level_candidates"]
    assert case["llm_result"]["status"] == "pending"
    assert "false_positive_count_by_label" in report["dashboard_validation_schema"]["metrics"]


def test_llm_review_cases_group_nearby_locations_into_blocks(tmp_path: Path) -> None:
    text = "A" * 20 + "DrugA" + "x" * 50 + "DrugB" + "y" * 150 + "DrugC"
    clinical = _clinical_fixture(tmp_path / "clinical.json", text)
    locations = tmp_path / "locations.json"
    rows = []
    for index, word in enumerate(("DrugA", "DrugB", "DrugC"), start=1):
        start = text.index(word)
        rows.append(
            {
                "location_id": f"loc:{index}",
                "document_id": "kr_1",
                "char_start": start,
                "char_end": start + len(word),
                "word_start": start,
                "word_end": start + len(word),
                "matched_text": word,
                "page": None,
                "section": None,
                "matches": [{"source": "mediq", "term_id": str(index), "canonical": word, "search_word": word, "normalized_search_word": word.lower(), "host_word": word, "inside_word": False}],
            }
        )
    write_json(locations, {"by_location": rows})

    report = build_review_cases_report(locations, clinical, window_chars=10, block_gap_chars=100)
    blocks = report["clinical_recommendations"][0]["llm_blocks"]

    assert [block["case_count"] for block in blocks] == [2, 1]
    assert report["summary"]["llm_block_count"] == 2
    assert report["summary"]["multi_case_block_count"] == 1


def test_openrouter_prompt_and_gold_scoring_shape() -> None:
    block = {
        "block_id": "block:1",
        "document_id": "d1",
        "document_title": "Doc",
        "document_link": "memory",
        "primary_terms": [{"canonical": "Drug"}],
        "context": {"text": "Рекомендуется Drug. УУР A, УДД 2.", "case_spans": [], "evidence_level_candidates": []},
        "llm_payload": {"task": "classify_clinical_recommendation_mention_block"},
    }
    messages = build_block_messages(block)
    score = score_gold([{"block_id": "block:1", "gold": {"label": "recommendation", "evidence_level": "2"}}], {"block:1": {"label": "recommendation", "evidence_level": "2"}})

    assert messages[0]["role"] == "system"
    assert "context_text" in messages[1]["content"]
    assert score["label_accuracy"] == 1.0


def test_openrouter_payload_filters_excluded_preparation_cases_without_mutating_raw_block() -> None:
    block = {
        "block_id": "block:1",
        "document_id": "d1",
        "document_title": "Doc",
        "document_link": "memory",
        "case_ids": ["case:drug", "case:other"],
        "case_count": 2,
        "primary_terms": [
            {"source": "mediq", "term_id": "m1", "canonical": "Drug", "search_word": "Drug", "host_word": "Drug", "inside_word": False},
            {"source": "mediq", "term_id": "m2", "canonical": "Other", "search_word": "Other", "host_word": "Other", "inside_word": False},
        ],
        "context": {
            "text": "Drug then Other.",
            "case_spans": [
                {"case_id": "case:drug", "span_start": 0, "span_end": 4, "text": "Drug"},
                {"case_id": "case:other", "span_start": 10, "span_end": 15, "text": "Other"},
            ],
            "highlight_spans": [
                {"case_id": "case:drug", "span_start": 0, "span_end": 4, "text": "Drug"},
                {"case_id": "case:other", "span_start": 10, "span_end": 15, "text": "Other"},
                {"type": "evidence_level", "span_start": 16, "span_end": 17, "text": "A"},
            ],
            "evidence_level_candidates": [],
        },
        "cases": [
            {
                "case_id": "case:drug",
                "primary_terms": [{"source": "mediq", "term_id": "m1", "canonical": "Drug", "search_word": "Drug", "host_word": "Drug", "inside_word": False}],
            },
            {
                "case_id": "case:other",
                "primary_terms": [{"source": "mediq", "term_id": "m2", "canonical": "Other", "search_word": "Other", "host_word": "Other", "inside_word": False}],
            },
        ],
        "llm_payload": {"task": "classify_clinical_recommendation_mention_block"},
    }
    excluded = collect_excluded_preparation_keys({"items": [{"source": "mediq", "term_id": "m1", "canonical": "Drug"}]})

    filtered = filter_block_for_llm(block, excluded)
    assert filtered is not None
    messages = build_block_messages(filtered)
    payload = json.loads(messages[1]["content"][messages[1]["content"].index("{") :])

    assert block["case_ids"] == ["case:drug", "case:other"]
    assert filtered["case_ids"] == ["case:other"]
    assert filtered["case_count"] == 1
    assert [term["canonical"] for term in payload["found_terms"]] == ["Other"]
    assert [span["case_id"] for span in payload["case_spans"]] == ["case:other"]
    assert [case["case_id"] for case in filtered["cases"]] == ["case:other"]


def test_openrouter_legacy_single_prediction_expands_to_all_cases() -> None:
    block = {
        "block_id": "block:1",
        "case_ids": ["case:1", "case:2"],
        "context": {"case_spans": []},
    }
    expanded = expand_single_prediction_to_cases(
        {"label": "contraindication", "target_kind": "drug", "evidence_level": "5"},
        block,
    )
    score = score_gold(
        [
            {
                "block_id": "block:1",
                "case_spans": [{"case_id": "case:1"}, {"case_id": "case:2"}],
                "gold": {"label": "contraindication", "target_kind": "drug", "evidence_level": "5"},
            }
        ],
        {"block:1": {"predictions": expanded}},
    )

    assert set(expanded) == {"case:1", "case:2"}
    assert expanded["case:1"] == expanded["case:2"]
    assert expanded["case:1"] is not expanded["case:2"]
    assert score["total"] == 2
    assert score["label_accuracy"] == 1.0


def test_openrouter_output_token_budget_scales_with_case_count(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    block = {"case_ids": [f"case:{index}" for index in range(8)]}

    assert max_tokens_for_block(block) == 3250

    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "777")

    assert max_tokens_for_block(block) == 777


def test_openrouter_resume_keeps_existing_predictions_across_prompt_version(tmp_path: Path, monkeypatch) -> None:
    review_path = tmp_path / "llm_review_cases.json"
    gold_path = tmp_path / "llm_gold.json"
    output_path = tmp_path / "openrouter_results.json"
    block = {
        "block_id": "block:1",
        "document_id": "doc:1",
        "document_title": "Doc",
        "document_link": "memory",
        "case_ids": ["case:1"],
        "primary_terms": [{"canonical": "Drug"}],
        "context": {"text": "Drug.", "case_spans": [{"case_id": "case:1", "span_start": 0, "span_end": 4, "text": "Drug"}], "evidence_level_candidates": []},
        "llm_payload": {"task": "classify_clinical_recommendation_mention_block"},
    }
    write_json(review_path, {"clinical_recommendations": [{"llm_blocks": [block]}]})
    write_json(gold_path, {"items": [{"block_id": "block:1", "case_spans": [{"case_id": "case:1"}], "gold": {"label": "error", "target_kind": "drug"}}]})
    write_json(
        output_path,
        {
            "prompt_version": "old-prompt",
            "predictions": {"block:1": {"predictions": {"case:1": {"label": "error", "target_kind": "drug"}}}},
        },
    )

    def fail_complete_openrouter(*_args, **_kwargs):
        raise AssertionError("complete_openrouter_must_not_run_for_existing_prediction")

    monkeypatch.setattr("anal_russia_klinik.llm_review_openrouter.complete_openrouter", fail_complete_openrouter)
    report = run_gold_openrouter(review_path, gold_path, output_path, api_key="fake-key", model="fake-model")

    assert report["completed"] == 1
    assert report["score"]["label_accuracy"] == 1.0
    assert read_json(output_path)["predictions"]["block:1"]["predictions"]["case:1"]["label"] == "error"


def test_openrouter_runner_skips_fully_excluded_blocks_before_llm_call(tmp_path: Path, monkeypatch) -> None:
    review_path = tmp_path / "llm_review_cases.json"
    gold_path = tmp_path / "llm_gold.json"
    output_path = tmp_path / "openrouter_results.json"
    excluded_path = tmp_path / "excluded_preparations.json"
    drug_term = {"source": "mediq", "term_id": "m1", "canonical": "Drug", "search_word": "Drug", "host_word": "Drug", "inside_word": False}
    other_term = {"source": "mediq", "term_id": "m2", "canonical": "Other", "search_word": "Other", "host_word": "Other", "inside_word": False}
    write_json(
        review_path,
        {
            "clinical_recommendations": [
                {
                    "llm_blocks": [
                        {
                            "block_id": "block:drug",
                            "document_id": "doc:1",
                            "document_title": "Doc",
                            "document_link": "memory",
                            "case_ids": ["case:drug"],
                            "case_count": 1,
                            "primary_terms": [drug_term],
                            "context": {"text": "Drug.", "case_spans": [{"case_id": "case:drug", "span_start": 0, "span_end": 4, "text": "Drug"}], "evidence_level_candidates": []},
                            "cases": [{"case_id": "case:drug", "primary_terms": [drug_term]}],
                            "llm_payload": {"task": "classify_clinical_recommendation_mention_block"},
                        },
                        {
                            "block_id": "block:other",
                            "document_id": "doc:1",
                            "document_title": "Doc",
                            "document_link": "memory",
                            "case_ids": ["case:other"],
                            "case_count": 1,
                            "primary_terms": [other_term],
                            "context": {"text": "Other.", "case_spans": [{"case_id": "case:other", "span_start": 0, "span_end": 5, "text": "Other"}], "evidence_level_candidates": []},
                            "cases": [{"case_id": "case:other", "primary_terms": [other_term]}],
                            "llm_payload": {"task": "classify_clinical_recommendation_mention_block"},
                        },
                    ]
                }
            ]
        },
    )
    write_json(
        gold_path,
        {
            "items": [
                {"block_id": "block:drug", "case_spans": [{"case_id": "case:drug"}], "gold": {"label": "error"}},
                {"block_id": "block:other", "case_spans": [{"case_id": "case:other"}], "gold": {"label": "recommendation"}},
            ]
        },
    )
    write_json(excluded_path, {"items": [{"source": "mediq", "term_id": "m1", "canonical": "Drug"}]})
    called_blocks: list[str] = []

    def fake_complete_openrouter(block: dict, **_kwargs) -> dict:
        called_blocks.append(block["block_id"])
        return {"predictions": {"case:other": {"label": "recommendation"}}}

    monkeypatch.setattr("anal_russia_klinik.llm_review_openrouter.complete_openrouter", fake_complete_openrouter)

    report = run_gold_openrouter(
        review_path,
        gold_path,
        output_path,
        api_key="fake-key",
        model="fake-model",
        excluded_preparations_path=excluded_path,
    )

    assert called_blocks == ["block:other"]
    assert set(report["predictions"]) == {"block:other"}
    assert report["completed_all"] == 1
    assert report["completed_visible"] == 1
    assert report["excluded_cases"] == 1
    assert report["excluded_blocks"] == 1
    assert report["score"]["total"] == 1


def test_openrouter_all_prioritizes_gold_blocks_without_existing_predictions(tmp_path: Path, monkeypatch) -> None:
    review_path = tmp_path / "llm_review_cases.json"
    gold_path = tmp_path / "llm_gold_40.json"
    output_path = tmp_path / "openrouter_all_results.json"
    existing_path = tmp_path / "openrouter_gold40_results.json"

    def block(block_id: str, case_id: str, canonical: str) -> dict:
        term = {"source": "mediq", "term_id": case_id, "canonical": canonical, "search_word": canonical, "host_word": canonical, "inside_word": False}
        return {
            "block_id": block_id,
            "document_id": "doc:1",
            "document_title": "Doc",
            "document_link": "memory",
            "case_ids": [case_id],
            "case_count": 1,
            "primary_terms": [term],
            "context": {"text": f"{canonical}.", "case_spans": [{"case_id": case_id, "span_start": 0, "span_end": len(canonical), "text": canonical}], "evidence_level_candidates": []},
            "cases": [{"case_id": case_id, "primary_terms": [term]}],
            "llm_payload": {"task": "classify_clinical_recommendation_mention_block"},
        }

    write_json(
        review_path,
        {
            "clinical_recommendations": [
                {
                    "llm_blocks": [
                        block("block:gold-missing", "case:gold-missing", "GoldDrug"),
                        block("block:gold-existing", "case:gold-existing", "ExistingDrug"),
                        block("block:pending", "case:pending", "PendingDrug"),
                    ]
                }
            ]
        },
    )
    write_json(
        gold_path,
        {
            "items": [
                {"block_id": "block:gold-missing", "case_spans": [{"case_id": "case:gold-missing"}], "gold": {"label": "error"}},
                {"block_id": "block:gold-existing", "case_spans": [{"case_id": "case:gold-existing"}], "gold": {"label": "recommendation"}},
            ]
        },
    )
    write_json(
        existing_path,
        {
            "predictions": {
                "block:gold-existing": {
                    "predictions": {"case:gold-existing": {"label": "recommendation"}}
                }
            }
        },
    )
    called_blocks: list[str] = []

    def fake_complete_openrouter(block: dict, **_kwargs) -> dict:
        called_blocks.append(block["block_id"])
        return {"predictions": {block["case_ids"][0]: {"label": "error"}}}

    monkeypatch.setattr("anal_russia_klinik.llm_review_openrouter.complete_openrouter", fake_complete_openrouter)

    report = run_openrouter_all(
        review_path,
        output_path,
        gold_path=gold_path,
        api_key="fake-key",
        model="fake-model",
        limit=2,
    )

    assert called_blocks == ["block:gold-missing", "block:pending"]
    assert report["selected_block_ids"] == ["block:gold-missing", "block:pending"]
    assert report["new_completed"] == 2
    assert report["completed_all"] == 3
    assert report["completed_visible"] == 3
    assert report["gold_missing_count"] == 1
    assert set(read_json(output_path)["predictions"]) == {"block:gold-missing", "block:gold-existing", "block:pending"}


def test_short_generated_forms_are_removed_but_original_two_letter_term_is_kept() -> None:
    assert "\u042d" not in variants_for_term("\u042d")
    assert "\u0410\u0411" in variants_for_term("\u0410\u0411")
    assert variants_for_term("\u0413\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d").count("\u0433\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d") == 0
    variants = variants_for_term("\u0414\u043b\u0438\u043d\u043d\u043e\u0435", ["\u044d"])
    assert "\u044d" not in variants


def test_compact_rows_deduplicate_case_variants_without_count_inflation() -> None:
    rows = deduplicate_compact_rows(
        [
            {
                "source": "mediq",
                "term_id": "m1",
                "canonical": "\u0413\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d",
                "search_word": "\u0413\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d",
                "normalized_search_word": "\u0433\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d",
                "host_words": [{"host_word": "\u0433\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d", "inside_word": False, "count": 7}],
            },
            {
                "source": "mediq",
                "term_id": "m1",
                "canonical": "\u0413\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d",
                "search_word": "\u0433\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d",
                "normalized_search_word": "\u0433\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d",
                "host_words": [{"host_word": "\u0433\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d", "inside_word": False, "count": 7}],
            },
        ]
    )

    assert len(rows) == 1
    assert rows[0]["search_word"] == "\u0413\u043b\u044e\u043a\u043e\u0437\u0430\u043c\u0438\u043d"
    assert rows[0]["match_count"] == 7


def test_filter_store_shards_and_merges_filter_words(tmp_path: Path) -> None:
    filters = tmp_path / "host_word_filters.json"
    words = [f"word_{index:04d}" for index in range(1001)]

    update_filter_words(filters, words, enabled=True, shard_size=500)
    parts = sorted((tmp_path / "host_word_filters_parts").glob("host_words_*.json"))

    assert len(parts) == 3
    assert read_filter_words(filters) == set(words)
    merged = merge_filter_words(filters, shard_size=500)
    assert merged["host_word_count"] == 1001


def test_mediq_terms_do_not_search_mnn_or_drug_codes(tmp_path: Path) -> None:
    source = tmp_path / "drugs.json"
    write_json(
        source,
        [
            {
                "drug": {
                    "name": "Brand",
                    "aliases": ["Brand alias"],
                    "atx": "A11HA03",
                    "atxGroup": "Vitamin group",
                    "mnn": {
                        "name": "tocopherol",
                        "complex": [{"name": "ascorbic acid"}],
                    },
                }
            }
        ],
    )

    term = load_preparation_terms(source, expand_word_forms=False)[0]

    assert "Brand" in term.variants
    assert "Brand alias" in term.variants
    assert "A11HA03" not in term.variants
    assert "Vitamin group" not in term.variants
    assert "tocopherol" not in term.variants
    assert "ascorbic acid" not in term.variants


def test_provider_defaults_and_openrouter_validation() -> None:
    assert build_provider("fake").name == "fake"
    assert build_provider(None).name == "g4f"
    with pytest.raises(ValueError):
        build_provider("openrouter")


def test_aho_host_word_report_groups_inside_matches(tmp_path: Path) -> None:
    markers = tmp_path / "markers.json"
    write_json(markers, ["onco"])
    clinical = _clinical_fixture(tmp_path / "clinical.json", "oncology and onco.")

    report = build_aho_host_word_report(load_legacy_clinical_json(clinical), load_marker_terms(markers))
    row = next(item for item in report["by_search_word"] if item["search_word"] == "onco")

    assert row["match_count"] == 2
    assert row["inside_word_match_count"] == 1
    assert row["inside_host_words"][0]["host_word"] == "oncology"
