from __future__ import annotations

import json
from pathlib import Path

from tools.build_expert_review_pages import build_expert_review_pages


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_build_expert_review_pages_creates_chunked_static_html(tmp_path: Path) -> None:
    review = {
        "clinical_recommendations": [
            {
                "llm_blocks": [
                    {
                        "block_id": "block:1",
                        "document_id": "100_1",
                        "document_title": "КР тест",
                        "document_link": "https://cr.minzdrav.gov.ru/view-cr/100_1",
                        "context": {"text": "Назначить Агри пациенту.", "case_spans": [{"case_id": "case:1", "span_start": 10, "span_end": 14}]},
                        "primary_terms": [{"source": "blacklist", "term_id": "blacklist:01a7bdf40b6d", "canonical": "Агри", "search_word": "Агри", "host_word": "Агри"}],
                        "cases": [
                            {
                                "case_id": "case:1",
                                "primary_terms": [{"source": "blacklist", "term_id": "blacklist:01a7bdf40b6d", "canonical": "Агри", "search_word": "Агри", "host_word": "Агри"}],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    results = {"predictions": {"block:1": {"predictions": {"case:1": {"label": "recommendation", "recommendation_strength": "C", "evidence_level": "5", "reason": "test"}}}}}
    blacklist = [{"Название препарата": "Агри", "Альтернативные названия": [], "Описание": "гомеопатия"}]
    write_json(tmp_path / "review.json", review)
    write_json(tmp_path / "results.json", results)
    write_json(tmp_path / "drugs.json", [])
    write_json(tmp_path / "blacklist.json", blacklist)
    write_json(tmp_path / "metadata.json", {"100_1": {"МКБ-10": "A00", "Возрастная группа": "Взрослые"}})
    write_json(tmp_path / "excluded.json", {"items": []})

    report = build_expert_review_pages(
        review_cases_path=tmp_path / "review.json",
        results_path=tmp_path / "results.json",
        drugs_path=tmp_path / "drugs.json",
        blacklist_path=tmp_path / "blacklist.json",
        metadata_path=tmp_path / "metadata.json",
        excluded_preparations_path=tmp_path / "excluded.json",
        output_dir=tmp_path / "out",
        chunk_size=1,
    )

    html = (tmp_path / "out" / "expert_recommendations_01.html").read_text(encoding="utf-8")
    assert report["recommendation_count"] == 1
    assert report["contraindication_count"] == 0
    assert "mark" in html
    assert "Агри" in html
    assert "Рекомендовал бы" in html
    assert "Не рекомендовал бы" in html
    assert "Ошибочно найдено" in html
    assert "false_positive" in html
    assert "Уровень убедительности рекомендации" in html


def test_build_expert_review_pages_skips_excluded_preparations(tmp_path: Path) -> None:
    term = {"source": "mediq", "term_id": "m1", "canonical": "Drug", "search_word": "Drug", "host_word": "Drug"}
    write_json(
        tmp_path / "review.json",
        {
            "clinical_recommendations": [
                {
                    "llm_blocks": [
                        {
                            "block_id": "block:1",
                            "document_id": "100_1",
                            "document_title": "КР тест",
                            "document_link": "https://cr.minzdrav.gov.ru/view-cr/100_1",
                            "context": {"text": "Drug text", "case_spans": [{"case_id": "case:1", "span_start": 0, "span_end": 4}]},
                            "primary_terms": [term],
                            "cases": [{"case_id": "case:1", "primary_terms": [term]}],
                        }
                    ]
                }
            ]
        },
    )
    write_json(tmp_path / "results.json", {"predictions": {"block:1": {"predictions": {"case:1": {"label": "recommendation"}}}}})
    write_json(tmp_path / "drugs.json", [{"drug": {"name": "Drug"}}])
    write_json(tmp_path / "blacklist.json", [])
    write_json(tmp_path / "metadata.json", {})
    write_json(tmp_path / "excluded.json", {"items": [{"source": "mediq", "term_id": "m1", "canonical": "Drug"}]})

    report = build_expert_review_pages(
        review_cases_path=tmp_path / "review.json",
        results_path=tmp_path / "results.json",
        drugs_path=tmp_path / "drugs.json",
        blacklist_path=tmp_path / "blacklist.json",
        metadata_path=tmp_path / "metadata.json",
        excluded_preparations_path=tmp_path / "excluded.json",
        output_dir=tmp_path / "out",
    )

    assert report["recommendation_count"] == 0
    assert report["excluded_preparation_count"] == 1
