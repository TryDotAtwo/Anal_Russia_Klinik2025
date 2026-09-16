from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "reports" / "llm"
SOURCE_RESULTS = LLM_DIR / "openrouter_all_results.json"
REVIEW_CASES = LLM_DIR / "llm_review_cases.json"
METADATA = ROOT / "data" / "input" / "MetaData.json"
CSV_OUTPUT = LLM_DIR / "openrouter_all_results.csv"


LABEL_RU = {
    "recommendation": "рекомендация",
    "contraindication": "противопоказание",
    "literature_mention": "упоминание литературы",
    "error": "ошибка/ложное совпадение",
    "unclear": "неясно",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def pred_items(block_id: str, prediction: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    nested = prediction.get("predictions")
    if isinstance(nested, dict) and nested:
        return [(str(case_id), value) for case_id, value in nested.items() if isinstance(value, dict)]
    return [(block_id, prediction)]


def prediction_rows(report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for block_id, prediction in report.get("predictions", {}).items():
        if isinstance(prediction, dict):
            for case_id, item in pred_items(str(block_id), prediction):
                rows.append((str(block_id), str(case_id), item))
    return rows


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def ratio(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def compact_counter(counter: Counter[str], empty: str = "-") -> str:
    if not counter:
        return empty
    return ", ".join(f"{key}: {value}" for key, value in counter.most_common())


def unique_join(values: set[str], limit: int = 24) -> str:
    cleaned = sorted(value for value in values if value)
    if not cleaned:
        return "-"
    shown = cleaned[:limit]
    suffix = f"; +{len(cleaned) - limit}" if len(cleaned) > limit else ""
    return "; ".join(shown) + suffix


def build_review_index(review_cases: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    block_index: dict[str, dict[str, Any]] = {}
    document_index: dict[str, dict[str, Any]] = {}
    for cr in review_cases.get("clinical_recommendations", []):
        doc_id = str(cr.get("document_id", ""))
        document_index[doc_id] = cr
        for block in cr.get("llm_blocks", []):
            block_index[str(block.get("block_id", ""))] = block
    return block_index, document_index


def source_bucket(source: str) -> str:
    if source == "mediq":
        return "mediq"
    if source == "blacklist":
        return "blacklist"
    if source == "marker":
        return "marker"
    return "other"


LEVEL_TRANSLATION = str.maketrans({"А": "A", "В": "B", "С": "C", "а": "A", "в": "B", "с": "C"})


def normalize_level_part(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().translate(LEVEL_TRANSLATION).upper()
    if text in {"", "NONE", "NULL", "NAN", "НЕ ОПРЕДЕЛЕН", "НЕ ОПРЕДЕЛЕНО"}:
        return ""
    return text


def level_key(prediction: dict[str, Any]) -> str | None:
    strength = normalize_level_part(prediction.get("recommendation_strength"))
    evidence = normalize_level_part(prediction.get("evidence_level"))
    if strength and evidence:
        return f"{strength}{evidence}"
    if strength:
        return strength
    if evidence:
        return evidence
    return None


def aggregate_documents(
    report: dict[str, Any],
    block_index: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    global_labels: Counter[str] = Counter()
    global_levels: Counter[str] = Counter()
    global_sources: Counter[str] = Counter()
    costs = 0.0
    tokens = 0

    for block_id, case_id, prediction in prediction_rows(report):
        block = block_index.get(block_id, {})
        doc_id = str(block.get("document_id") or block_id)
        meta = metadata.get(doc_id, {})
        doc = docs.setdefault(
            doc_id,
            {
                "id": doc_id,
                "title": meta.get("Название клинической рекомендации") or block.get("document_title", "-"),
                "mkb10": meta.get("МКБ-10", "-"),
                "age": meta.get("Возрастная группа", "-"),
                "developer": meta.get("Разработчик", "-"),
                "date": meta.get("Дата размещения", "-"),
                "status": meta.get("Статус применения КР", "-"),
                "link": block.get("document_link", "-"),
                "levels": Counter(),
                "labels": Counter(),
                "mediq": set(),
                "blacklist": set(),
                "marker": set(),
                "mediq_count": 0,
                "blacklist_count": 0,
                "marker_count": 0,
                "case_count": 0,
            },
        )
        label = str(prediction.get("label") or "unclear")
        doc["labels"][label] += 1
        global_labels[label] += 1
        doc["case_count"] += 1
        lvl = level_key(prediction)
        if label == "recommendation" and lvl:
            doc["levels"][lvl] += 1
            global_levels[lvl] += 1
        case_terms = []
        for case in block.get("cases", []):
            if str(case.get("case_id", "")) == case_id:
                case_terms = case.get("primary_terms") or case.get("matches") or []
                break
        if not case_terms:
            case_terms = block.get("primary_terms", [])
        for term in case_terms:
            if not isinstance(term, dict):
                continue
            bucket = source_bucket(str(term.get("source", "")))
            canonical = str(term.get("canonical") or term.get("search_word") or term.get("host_word") or "")
            if bucket in {"mediq", "blacklist", "marker"}:
                doc[bucket].add(canonical)
                doc[f"{bucket}_count"] += 1
                global_sources[bucket] += 1

        meta_usage = prediction.get("_openrouter", {}).get("response", {}).get("usage", {})
        costs += float(meta_usage.get("cost") or 0.0)
        tokens += int(meta_usage.get("total_tokens") or 0)

    rows = []
    for doc in docs.values():
        rows.append(
            {
                "ID": doc["id"],
                "Название": doc["title"],
                "МКБ-10": doc["mkb10"],
                "Возраст": doc["age"],
                "Разработчик": doc["developer"],
                "Дата размещения": doc["date"],
                "Статус": doc["status"],
                "Ссылка": doc["link"],
                "Уровни рекомендаций (A1: 5, B2: 3 и т.д.)": compact_counter(doc["levels"], "—"),
                "MedIQ (кол-во)": len(doc["mediq"]),
                "Blacklist (кол-во)": len(doc["blacklist"]),
                "Маркеры (кол-во)": len(doc["marker"]),
                "Препараты MedIQ": unique_join(doc["mediq"]),
                "Маркеры": unique_join(doc["marker"]),
                "Чёрный список": unique_join(doc["blacklist"]),
                "LLM-кейсы": doc["case_count"],
                "Метки LLM": compact_counter(doc["labels"], "—"),
            }
        )
    rows.sort(key=lambda row: (-int(row["LLM-кейсы"]), str(row["ID"])))
    summary = {
        "documents": len(rows),
        "cases": sum(row["LLM-кейсы"] for row in rows),
        "labels": global_labels,
        "levels": global_levels,
        "sources": global_sources,
        "cost": costs,
        "tokens": tokens,
    }
    return rows, summary


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "ID",
        "Название",
        "МКБ-10",
        "Возраст",
        "Разработчик",
        "Дата размещения",
        "Статус",
        "Ссылка",
        "Уровни рекомендаций (A1: 5, B2: 3 и т.д.)",
        "MedIQ (кол-во)",
        "Blacklist (кол-во)",
        "Маркеры (кол-во)",
        "Препараты MedIQ",
        "Маркеры",
        "Чёрный список",
        "LLM-кейсы",
        "Метки LLM",
    ]
    with CSV_OUTPUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def model_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = prediction_rows(report)
    labels = Counter(str(pred.get("label") or "unclear") for _, _, pred in rows)
    levels = Counter(level_key(pred) for _, _, pred in rows if str(pred.get("label")) == "recommendation" and level_key(pred))
    usage_cost = 0.0
    usage_tokens = 0
    for _, _, pred in rows:
        usage = pred.get("_openrouter", {}).get("response", {}).get("usage", {})
        usage_cost += float(usage.get("cost") or 0.0)
        usage_tokens += int(usage.get("total_tokens") or 0)
    return {
        "model": report.get("model", "-"),
        "prompt_version": report.get("prompt_version", "-"),
        "predictions": len(report.get("predictions", {})),
        "case_predictions": len(rows),
        "score_total": report.get("score", {}).get("total"),
        "label_accuracy": report.get("score", {}).get("label_accuracy"),
        "strength_accuracy": report.get("score", {}).get("recommendation_strength_accuracy"),
        "evidence_accuracy": report.get("score", {}).get("evidence_level_accuracy"),
        "labels": labels,
        "levels": levels,
        "tokens": usage_tokens,
        "cost": usage_cost,
    }


def write_infographics(mini: dict[str, Any], summary: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    s_mini = model_summary(mini)
    palette = ["#66d9ef", "#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#94a3b8"]
    score = mini.get("score", {})
    per_label = score.get("per_label", {})
    confusion = score.get("confusion_matrix", {})
    label_order = ["recommendation", "error", "literature_mention", "contraindication", "unclear"]
    label_names = {
        "recommendation": "Рекомендация",
        "error": "Ложное совпадение",
        "literature_mention": "Литература",
        "contraindication": "Противопоказание",
        "unclear": "Неясно",
    }

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        names = ["arialbd.ttf" if bold else "arial.ttf", "seguisb.ttf" if bold else "segoeui.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        for name in names:
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                continue
        raise RuntimeError("Cyrillic TrueType font required: install DejaVu Sans (Linux: fonts-dejavu-core) or Arial.")

    def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (1600, 900), "#050607")
        draw = ImageDraw.Draw(img)
        return img, draw

    def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, color: str = "#f7f7f8", bold: bool = False) -> None:
        draw.text(xy, value, fill=color, font=font(size, bold))

    def metric(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, label: str, color: str = "#66d9ef") -> None:
        text(draw, (x, y), value, 72, color, True)
        text(draw, (x, y + 82), label, 25, "#a5adbb")

    def hbar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, label: str, value: int, max_value: int, color: str, label_w: int = 190) -> None:
        text(draw, (x, y), label, 25, "#e6e9ef", True)
        bw = max(4, int(w * value / max_value))
        draw.rounded_rectangle((x + label_w, y + 5, x + label_w + bw, y + 33), radius=14, fill=color)
        text(draw, (x + label_w + 15 + bw, y + 1), f"{value}", 24, "#f7f7f8", True)

    img, draw = canvas()
    text(draw, (80, 76), "Сколько найдено рекомендаций и типов совпадений", 48, bold=True)
    metric(draw, 90, 245, str(s_mini["labels"].get("recommendation", 0)), "рекомендаций", "#34d399")
    metric(draw, 480, 245, str(s_mini["labels"].get("contraindication", 0)), "противопоказаний", "#fb7185")
    metric(draw, 875, 245, str(s_mini["labels"].get("literature_mention", 0)), "литературных упоминаний", "#fbbf24")
    metric(draw, 1245, 245, str(s_mini["labels"].get("error", 0)), "ложных совпадений", "#a78bfa")
    text(draw, (90, 475), "Словарные источники внутри оценённых кейсов", 34, bold=True)
    metric(draw, 90, 580, str(summary["sources"].get("mediq", 0)), "MedIQ", "#66d9ef")
    metric(draw, 480, 580, str(summary["sources"].get("blacklist", 0)), "Blacklist", "#fb7185")
    metric(draw, 875, 580, str(summary["sources"].get("marker", 0)), "Маркеры", "#fbbf24")
    img.save(LLM_DIR / "infographic_slide_1.png", quality=95)

    img, draw = canvas()
    levels = s_mini["levels"]
    top_levels = levels.most_common(12)
    tail = Counter(dict(levels.most_common()[12:]))
    max_level = max((value for _, value in top_levels), default=1)
    text(draw, (80, 76), "Уровни найденных рекомендаций", 50, bold=True)
    metric(draw, 90, 235, str(levels.get("C5", 0)), "C5", "#34d399")
    metric(draw, 350, 235, str(levels.get("B2", 0)), "B2", "#66d9ef")
    metric(draw, 610, 235, str(levels.get("C4", 0)), "C4", "#fbbf24")
    metric(draw, 870, 235, str(levels.get("A1", 0)), "A1", "#a78bfa")
    metric(draw, 1130, 235, str(sum(levels.values())), "всего с уровнем", "#fb7185")
    text(draw, (90, 430), "Распределение top-12", 33, bold=True)
    for idx, (key, value) in enumerate(top_levels):
        hbar(draw, 90, 490 + idx * 31, 780, key, value, max_level, palette[idx % len(palette)])
    text(draw, (1110, 455), "Редкие / неполные", 30, "#e6e9ef", True)
    y = 505
    for chunk in [", ".join(f"{k}: {v}" for k, v in tail.most_common()[i:i + 4]) for i in range(0, len(tail), 4)]:
        text(draw, (1110, y), chunk, 20, "#a5adbb")
        y += 34
    img.save(LLM_DIR / "infographic_slide_2.png", quality=95)

    img, draw = canvas()
    text(draw, (80, 76), "Качество накопленных ответов: 143 gold-кейса", 48, bold=True)
    metric(draw, 90, 245, percent(score.get("label_accuracy")), "label accuracy", "#66d9ef")
    metric(draw, 470, 245, percent(score.get("recommendation_strength_accuracy")), "strength accuracy", "#34d399")
    metric(draw, 875, 245, percent(score.get("evidence_level_accuracy")), "evidence accuracy", "#fbbf24")
    text(draw, (90, 440), "F1 по классам", 33, bold=True)
    for idx, label in enumerate(label_order):
        f1 = per_label.get(label, {}).get("f1") or 0
        hbar(draw, 90, 500 + idx * 58, 450, label_names[label], int(f1 * 100), 100, palette[idx % len(palette)], label_w=310)
    text(draw, (900, 440), "TP / FP / FN", 33, bold=True)
    for idx, label in enumerate(["recommendation", "error", "literature_mention", "contraindication"]):
        row = per_label.get(label, {})
        y = 500 + idx * 58
        text(draw, (900, y), label_names[label], 24, palette[idx % len(palette)], True)
        text(draw, (1220, y), f"{row.get('tp', 0)} / {row.get('fp', 0)} / {row.get('fn', 0)}", 24, "#e6e9ef")
    text(draw, (900, 760), "Ошибки матрицы", 27, "#e6e9ef", True)
    text(draw, (900, 805), "contraindication→error: 2", 22, "#a5adbb")
    text(draw, (900, 835), "error→recommendation: 5", 22, "#a5adbb")
    text(draw, (1220, 805), "recommendation→error: 5", 22, "#a5adbb")
    img.save(LLM_DIR / "infographic_slide_3.png", quality=95)


def main() -> None:
    all_results = load_json(SOURCE_RESULTS)
    review_cases = load_json(REVIEW_CASES)
    metadata = load_json(METADATA)
    block_index, _ = build_review_index(review_cases)
    rows, summary = aggregate_documents(all_results, block_index, metadata)
    write_csv(rows)
    write_infographics(all_results, summary)
    report = {
        "csv": str(CSV_OUTPUT.relative_to(ROOT)),
        "rows": len(rows),
        "source_model": all_results.get("model"),
        "source_predictions": len(all_results.get("predictions", {})),
        "source_case_predictions": summary["cases"],
        "labels": dict(summary["labels"]),
        "levels": dict(summary["levels"]),
        "sources": dict(summary["sources"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
