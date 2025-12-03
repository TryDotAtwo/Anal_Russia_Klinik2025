import json
from collections import defaultdict
import csv

INPUT_JSON = "clinical_matches_final_combined.json"
OUTPUT_CSV = "Финальный_отчет_4_таблицы_одинаковая_шапка.csv"

# Единая шапка для всех таблиц
HEADER = [
    "ID", "Название", "МКБ-10", "Возраст", "Разработчик", "Дата размещения", "Статус",
    "Ссылка", "Уровни рекомендаций (A1: 5, B2: 3 и т.д.)",
    "MedIQ (кол-во)", "Blacklist (кол-во)", "Маркеры (кол-во)",
    "Препараты MedIQ", "Маркеры", "Чёрный список"
]

def build_row(kr, level_counter, mediq_terms, blacklist_terms, marker_terms):
    meta = kr.get("metadata", {})
    mkb = ", ".join(meta.get("МКБ-10", [])) if isinstance(meta.get("МКБ-10"), list) else str(meta.get("МКБ-10", ""))
    age = meta.get("Возрастная группа", "")
    developer = meta.get("Разработчик", "")
    date = meta.get("Дата размещения", "")
    status = meta.get("Статус применения КР", "")

    levels_str = ", ".join(f"{k}: {v}" for k, v in sorted(level_counter.items(), key=lambda x: -x[1])) if level_counter else "—"
    mediq_str = ", ".join(sorted(mediq_terms)) if mediq_terms else "—"
    blacklist_str = ", ".join(sorted(blacklist_terms)) if blacklist_terms else "—"
    marker_str = ", ".join(sorted(marker_terms)) if marker_terms else "—"

    return [
        kr.get("kr_id", ""),
        kr.get("kr_name", ""),
        mkb,
        age,
        developer,
        date,
        status,
        kr.get("kr_link", ""),
        levels_str,
        len(mediq_terms),
        len(blacklist_terms),
        len(marker_terms),
        mediq_str,
        marker_str,
        blacklist_str
    ]

def generate_uniform_csv():
    print("Загружаем данные...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    kr_list = data["clinical_recommendations"]
    print(f"Обрабатываем {len(kr_list)} КР")

    # Списки для таблиц
    all_rows = []
    rec_rows = []
    contra_rows = []
    lit_rows = []

    for kr in kr_list:
        mediq_terms = set()
        blacklist_terms = set()
        marker_terms = set()

        # Счётчики уровней по типам
        rec_levels = defaultdict(int)
        contra_levels = defaultdict(int)
        lit_levels = defaultdict(int)

        has_rec = False
        has_contra = False
        has_lit = False

        for source, field in [
            ("mediq", "mediq_matches"),
            ("blacklist", "blacklist_matches"),
            ("marker", "marker_matches")
        ]:
            for term_obj in kr.get(field, []):
                term = term_obj["term"]
                if source == "mediq": mediq_terms.add(term)
                elif source == "blacklist": blacklist_terms.add(term)
                else: marker_terms.add(term)

                for ctx in term_obj.get("contexts", []):
                    analysis = ctx.get("analysis") or {}
                    typ = analysis.get("Тип", "").lower()

                    udd = str(analysis.get("УДД", "")).strip()
                    uur = str(analysis.get("УУР", "")).strip()
                    level = f"{uur}{udd}" if udd and uur else (udd or uur or "—")

                    if "рекомендация" in typ:
                        rec_levels[level] += 1
                        has_rec = True
                    elif "противопоказание" in typ:
                        contra_levels[level] += 1
                        has_contra = True
                    elif "литература" in typ:
                        lit_levels[level] += 1
                        has_lit = True

        # Общие уровни (для сводки)
        all_levels = defaultdict(int)
        for d in (rec_levels, contra_levels, lit_levels):
            for k, v in d.items():
                all_levels[k] += v

        # Добавляем строку в нужные таблицы
        row_all = build_row(kr, all_levels, mediq_terms, blacklist_terms, marker_terms)
        all_rows.append(row_all)

        if has_rec:
            row_rec = build_row(kr, rec_levels, mediq_terms, blacklist_terms, marker_terms)
            rec_rows.append(row_rec)

        if has_contra:
            row_contra = build_row(kr, contra_levels, mediq_terms, blacklist_terms, marker_terms)
            contra_rows.append(row_contra)

        if has_lit:
            row_lit = build_row(kr, lit_levels, mediq_terms, blacklist_terms, marker_terms)
            lit_rows.append(row_lit)

    # === Запись в CSV ===
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        # 1. Все КР
        writer.writerow(["=== ВСЕ КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ ==="])
        writer.writerow([])
        writer.writerow(HEADER)
        writer.writerows(all_rows)
        writer.writerow([])
        writer.writerow([f"Итого КР: {len(all_rows)}"])
        writer.writerow([])

        # 2. Только с рекомендациями
        writer.writerow(["=== КР С РЕКОМЕНДАЦИЯМИ ==="])
        writer.writerow([])
        writer.writerow(HEADER)
        writer.writerows(rec_rows)
        writer.writerow([])
        writer.writerow([f"КР с рекомендациями: {len(rec_rows)}"])
        writer.writerow([])

        # 3. Только с противопоказаниями
        writer.writerow(["=== КР С ПРОТИВОПОКАЗАНИЯМИ ==="])
        writer.writerow([])
        writer.writerow(HEADER)
        writer.writerows(contra_rows)
        writer.writerow([])
        writer.writerow([f"КР с противопоказаниями: {len(contra_rows)}"])
        writer.writerow([])

        # 4. Только с упоминаниями литературы
        writer.writerow(["=== КР С УПОМИНАНИЯМИ ЛИТЕРАТУРЫ ==="])
        writer.writerow([])
        writer.writerow(HEADER)
        writer.writerows(lit_rows)
        writer.writerow([])
        writer.writerow([f"КР с литературой: {len(lit_rows)}"])
        writer.writerow([])

    print(f"\nГОТОВО! Файл сохранён:")
    print(f"→ {OUTPUT_CSV}")
    print("Все 4 таблицы — с одинаковой шапкой. Открывай в Excel — будет идеально!")

if __name__ == "__main__":
    generate_uniform_csv()