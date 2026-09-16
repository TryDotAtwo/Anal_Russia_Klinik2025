from __future__ import annotations

import json
from pathlib import Path

from tools.build_gold_review_pages import build_gold_review_pages


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_build_gold_review_pages_creates_static_chunk(tmp_path: Path) -> None:
    term = {"source": "mediq", "term_id": "m1", "canonical": "Drug", "host_word": "Drug"}
    review = {
        "clinical_recommendations": [
            {
                "llm_blocks": [
                    {
                        "block_id": "block:1",
                        "document_id": "1",
                        "document_title": "КР",
                        "document_link": "https://example.test",
                        "primary_terms": [term],
                        "case_ids": ["case:1"],
                        "context": {"text": "Drug text", "block_span_start": 0, "block_span_end": 4, "case_spans": [{"case_id": "case:1", "span_start": 0, "span_end": 4, "text": "Drug"}]},
                        "cases": [{"case_id": "case:1", "primary_terms": [term]}],
                    }
                ]
            }
        ]
    }
    write_json(tmp_path / "review.json", review)
    write_json(tmp_path / "results.json", {"predictions": {"block:1": {"predictions": {"case:1": {"label": "recommendation", "reason": "r"}}}}})
    write_json(tmp_path / "gold.json", {"items": []})
    write_json(tmp_path / "excluded.json", {"items": []})

    report = build_gold_review_pages(
        review_cases_path=tmp_path / "review.json",
        results_path=tmp_path / "results.json",
        gold_path=tmp_path / "gold.json",
        excluded_preparations_path=tmp_path / "excluded.json",
        output_dir=tmp_path / "out",
        chunk_size=400,
    )

    html = (tmp_path / "out" / "gold_review_01.html").read_text(encoding="utf-8")
    assert report["block_count"] == 1
    assert "Экспорт gold JSON" in html
    assert "recommendation" in html
    assert "Не учитывать в статистике модели" in html
