import json
import asyncio
import time
import os
import csv
from tqdm import tqdm
import aiohttp
from asyncio import WindowsSelectorEventLoopPolicy
import ahocorasick

# Импортируем вспомогательные функции
from Logic.Clinick_Extract import ClinicalAnalyzer
from Logic.Levels_extract import analyze_mention
from Logic.Work_witch_word import generate_word_forms


# Список русских предлогов и служебных односложных слов, которые не ищем как отдельные термины
RUSSIAN_PREPOSITIONS = {
    "в", "во", "на", "с", "со", "к", "ко", "по", "о", "об", "обо",
    "от", "ото", "до", "из", "изо", "у", "при", "про", "над", "под",
    "за", "через", "между", "для", "без", "перед", "передо",
}


def is_word_char(ch: str) -> bool:
    """Проверка, является ли символ частью слова (буква/цифра, латиница или кириллица)."""
    if not ch:
        return False
    return ch.isalnum() or ch in "ёЁ"


def load_sources(config):
    """
    Загружает три источника:
    - MedIQ (`drugs.json`)
    - расстрельный список (`blacklist_drugs.json`)
    - слова-маркеры мракобесия (`AXTUNG.Json`)
    """
    # MedIQ
    try:
        with open(config["additional_drug_info_path"], "r", encoding="utf-8") as f:
            mediq_data = json.load(f)
        print(f"Загружено записей из MedIQ (drugs.json): {len(mediq_data)}")
    except Exception as e:
        print(f"Ошибка загрузки MedIQ (drugs.json): {e}")
        mediq_data = []

    # Расстрельный список
    try:
        with open(config["blacklist_json_path"], "r", encoding="utf-8") as f:
            blacklist_data = json.load(f)
        print(f"Загружено записей из расстрельного списка (blacklist_drugs.json): {len(blacklist_data)}")
    except Exception as e:
        print(f"Ошибка загрузки расстрельного списка (blacklist_drugs.json): {e}")
        blacklist_data = []

    # AXTUNG-маркеры
    try:
        with open(config["axtung_path"], "r", encoding="utf-8") as f:
            axtung_data = json.load(f)
        print(f"Загружено слов-маркеров мракобесия (AXTUNG.Json): {len(axtung_data)}")
    except Exception as e:
        print(f"Ошибка загрузки AXTUNG.Json: {e}")
        axtung_data = []

    return mediq_data, blacklist_data, axtung_data


def build_unified_index(mediq_data, blacklist_data, axtung_data):
    """
    Строит единый индекс терминов из трёх источников без дублирования форм:
    - каждый термин имеет id, canonical (основное название) и source: mediq/blacklist/marker
    - для каждого термина собираем список форм (слово + словоформы для одиночных слов)
    - возвращаем:
        terms_by_id, form_to_term_ids, per-source метаданные
    """
    terms_by_id = {}
    form_to_term_ids = {}
    next_id = 0

    mediq_meta = {}
    blacklist_meta = {}
    marker_meta = {}

    # --- MedIQ ---
    for item in mediq_data:
        drug = item.get("drug", {})
        name = (drug.get("name") or "").strip()
        if not name:
            continue
        name_clean = name.lower()
        if name_clean in RUSSIAN_PREPOSITIONS:
            # Не индексируем служебные слова как термины
            continue
        atx = (drug.get("atx") or "").strip()
        atx_group = (drug.get("atxGroup") or "").strip()
        mnn = (drug.get("mnn", {}) or {})
        mnn_name = (mnn.get("name") or "").strip()

        canonical = name
        key = canonical.lower()
        if key not in mediq_meta:
            mediq_meta[key] = {
                "term": canonical,
                "atx": atx,
                "atxGroup": atx_group,
            }

        # Формы: основное название + MNN (без генерации для многословных фраз)
        surface_forms = set()
        for s in [name, mnn_name]:
            if not s:
                continue
            s_clean = s.lower().strip()
            if not s_clean:
                continue
            if s_clean in RUSSIAN_PREPOSITIONS:
                # Пропускаем предлоги и подобные служебные слова
                continue
            if " " in s_clean:
                surface_forms.add(s_clean)
            else:
                for form in generate_word_forms(s_clean):
                    form = form.strip()
                    if form and form not in RUSSIAN_PREPOSITIONS:
                        surface_forms.add(form)

        if not surface_forms:
            continue

        term_id = next_id
        next_id += 1
        terms_by_id[term_id] = {
            "id": term_id,
            "canonical": canonical,
            "source": "mediq",
        }

        for form in surface_forms:
            if form not in form_to_term_ids:
                form_to_term_ids[form] = set()
            form_to_term_ids[form].add(term_id)

    # --- blacklist ---
    for entry in blacklist_data:
        main_name = (entry.get("Название препарата") or "").strip()
        description = entry.get("Описание")
        if not main_name:
            continue
        if main_name.lower() in RUSSIAN_PREPOSITIONS:
            continue
        alt_names = entry.get("Альтернативные названия", []) or []

        canonical = main_name
        key = canonical.lower()
        if key not in blacklist_meta:
            blacklist_meta[key] = {
                "term": canonical,
                "description": description,
            }

        names = [main_name] + [alt for alt in alt_names if alt]
        surface_forms = set()
        for s in names:
            s_clean = s.lower().strip()
            if not s_clean:
                continue
            if s_clean in RUSSIAN_PREPOSITIONS:
                continue
            if " " in s_clean:
                surface_forms.add(s_clean)
            else:
                for form in generate_word_forms(s_clean):
                    form = form.strip()
                    if form and form not in RUSSIAN_PREPOSITIONS:
                        surface_forms.add(form)

        if not surface_forms:
            continue

        term_id = next_id
        next_id += 1
        terms_by_id[term_id] = {
            "id": term_id,
            "canonical": canonical,
            "source": "blacklist",
        }

        for form in surface_forms:
            if form not in form_to_term_ids:
                form_to_term_ids[form] = set()
            form_to_term_ids[form].add(term_id)

    # --- AXTUNG-маркеры ---
    for word in axtung_data:
        s = (word or "").strip()
        if not s:
            continue
        if s.lower() in RUSSIAN_PREPOSITIONS:
            continue
        canonical = s
        key = canonical.lower()
        if key not in marker_meta:
            marker_meta[key] = {
                "term": canonical,
            }

        surface_forms = set()
        s_clean = s.lower().strip()
        if " " in s_clean:
            surface_forms.add(s_clean)
        else:
            for form in generate_word_forms(s_clean):
                form = form.strip()
                if form and form not in RUSSIAN_PREPOSITIONS:
                    surface_forms.add(form)

        if not surface_forms:
            continue

        term_id = next_id
        next_id += 1
        terms_by_id[term_id] = {
            "id": term_id,
            "canonical": canonical,
            "source": "marker",
        }

        for form in surface_forms:
            if form not in form_to_term_ids:
                form_to_term_ids[form] = set()
            form_to_term_ids[form].add(term_id)

    print(f"Всего терминов в едином индексе: {len(terms_by_id)}")
    print(f"Всего уникальных словоформ в индексе: {len(form_to_term_ids)}")

    return terms_by_id, form_to_term_ids, mediq_meta, blacklist_meta, marker_meta


def build_aho_automaton_unified(form_to_term_ids):
    """
    Строит Ахо–Корасик автомат по всем словоформам.
    Значением для каждой формы является кортеж (сама_форма, [id_терминов]).
    """
    A = ahocorasick.Automaton()
    for form, ids in form_to_term_ids.items():
        A.add_word(form, (form, list(ids)))
    A.make_automaton()
    return A


async def find_all_mentions(automaton, terms_by_id, text, context_before, context_after, verbose=False):
    """
    Один проход по тексту КР:
    - находит все совпадения по Ахо–Корасику
    - фильтрует по границам слова (АТФ ≠ АТФормирование)
    - группирует по источникам: mediq / blacklist / marker
    """
    text_lower = text.lower()
    # Накапливаем контексты во множествах, чтобы не было дублей для одного термина
    results = {
        "mediq": {},
        "blacklist": {},
        "marker": {},
    }

    for end_index, (matched_form, term_ids) in automaton.iter(text_lower):
        match_end = end_index
        match_start = end_index - len(matched_form) + 1

        # Проверяем границы слова:
        # считаем совпадение валидным, только если слева и справа пробел или дефис,
        # либо это начало/конец строки.
        before_ch = text[match_start - 1] if match_start > 0 else " "
        after_ch = text[match_end + 1] if match_end + 1 < len(text) else " "

        # Считаем совпадение валидным, если слева и справа пробел/любая пробельная
        # или дефис, либо это начало/конец строки.
        before_ok = before_ch.isspace() or before_ch == "-"
        after_ok = after_ch.isspace() or after_ch == "-"
        if not (before_ok and after_ok):
            continue

        context_start = max(0, match_start - context_before)
        context_end = min(len(text), match_end + 1 + context_after)
        context = text[context_start:context_end]

        for term_id in term_ids:
            term = terms_by_id[term_id]
            src = term["source"]
            canonical = term["canonical"]
            bucket = results[src].setdefault(canonical, set())
            bucket.add(context)
            if verbose:
                print(f"[{src.upper()}] {canonical}: {context[:80]}...")
    # Преобразуем множества контекстов обратно в списки
    for src in results:
        for term in list(results[src].keys()):
            results[src][term] = list(results[src][term])

    return results

# --- Построение объекта КР после этапа 1 ---
def build_kr_result_stage1(kr_key, kr_data, mentions, mediq_meta, blacklist_meta, marker_meta):
    """
    Формирует объект результата для одной КР после этапа 1:
    - метаданные КР
    - три списка совпадений: MedIQ / blacklist / AXTUNG-маркеры
    Внутри каждого списка:
      { "term": <слово>, "contexts": [<контекст1>, ...], ...доп.метаданные... }
    """
    result = {
        "kr_id": kr_key,
        "kr_name": kr_data.get("Название"),
        "kr_link": kr_data.get("Ссылка"),
        "metadata": kr_data.get("metadata", {}),
        "mediq_matches": [],
        "blacklist_matches": [],
        "marker_matches": [],
        "summary": {
            "mediq": {"terms": 0, "mentions": 0},
            "blacklist": {"terms": 0, "mentions": 0},
            "marker": {"terms": 0, "mentions": 0},
        },
    }

    # MedIQ
    for term, contexts in mentions["mediq"].items():
        key = term.lower()
        meta = mediq_meta.get(key, {})
        result["mediq_matches"].append(
            {
                "term": term,
                "atx": meta.get("atx"),
                "atxGroup": meta.get("atxGroup"),
                "contexts": [{"context": c, "analysis": None} for c in contexts],
            }
        )
        result["summary"]["mediq"]["terms"] += 1
        result["summary"]["mediq"]["mentions"] += len(contexts)

    # blacklist
    for term, contexts in mentions["blacklist"].items():
        key = term.lower()
        meta = blacklist_meta.get(key, {})
        result["blacklist_matches"].append(
            {
                "term": term,
                "description": meta.get("description"),
                "contexts": [{"context": c, "analysis": None} for c in contexts],
            }
        )
        result["summary"]["blacklist"]["terms"] += 1
        result["summary"]["blacklist"]["mentions"] += len(contexts)

    # markers
    for term, contexts in mentions["marker"].items():
        result["marker_matches"].append(
            {
                "term": term,
                "contexts": [{"context": c, "analysis": None} for c in contexts],
            }
        )
        result["summary"]["marker"]["terms"] += 1
        result["summary"]["marker"]["mentions"] += len(contexts)

    return result

# --- Функция загрузки клинических рекомендаций ---
async def load_data(config):
    """Загрузка клинических рекомендаций."""
    analyzer = ClinicalAnalyzer(config)
    await analyzer.load_metadata()
    await analyzer.load_clinical_recommendations()
    # На этом этапе в каждый элемент clinical_recommendations уже встроены все метаданные.
    print(f"Загружено клинических рекомендаций для анализа: {len(analyzer.clinical_recommendations)}")
    return analyzer.clinical_recommendations


async def perform_llm_analysis_on_matches(all_kr_results, batch_size=100):
    """
    Этап 2 — с надёжным сохранением прогресса после каждого батча
    """
    start_time = time.time()
    tasks = []
    refs = []

    # Собираем все упоминания
    for kr_idx, kr in enumerate(all_kr_results):
        for category in ("mediq_matches", "blacklist_matches", "marker_matches"):
            for term_idx, term_obj in enumerate(kr.get(category, [])):
                term_name = term_obj.get("term")
                for ctx_idx, ctx_obj in enumerate(term_obj.get("contexts", [])):
                    context = ctx_obj.get("context", "")
                    tasks.append(analyze_mention(term_name, context, None))
                    refs.append((kr_idx, category, term_idx, ctx_idx))

    total_mentions = len(tasks)
    print(f"\nНачало LLM-анализа. Всего упоминаний: {total_mentions}")

    batch_dir = "батчи"
    os.makedirs(batch_dir, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        # Пересобираем задачи с session
        wrapped_tasks = []
        for kr_idx, category, term_idx, ctx_idx in refs:
            term_name = all_kr_results[kr_idx][category][term_idx]["term"]
            context = all_kr_results[kr_idx][category][term_idx]["contexts"][ctx_idx]["context"]
            wrapped_tasks.append(analyze_mention(term_name, context, session))

        tasks = wrapped_tasks
        processed = 0
        batch_num = 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in range(0, len(tasks), batch_size):
            batch_start = time.time()
            current_batch = tasks[i:i + batch_size]
            current_refs = refs[i:i + batch_size]

            print(f"Обработка батча {batch_num}/{total_batches} ({len(current_batch)} упоминаний)...")

            analyses = await asyncio.gather(*current_batch, return_exceptions=True)

            # Применяем результаты
            for (kr_idx, category, term_idx, ctx_idx), analysis in zip(current_refs, analyses):
                if isinstance(analysis, Exception):
                    analysis = {
                        "error": str(analysis),
                        "Тип": "Ошибка LLM",
                        "Комментарий": "Запрос к модели не удался"
                    }
                all_kr_results[kr_idx][category][term_idx]["contexts"][ctx_idx]["analysis"] = analysis

            processed += len(current_batch)
            batch_time = time.time() - batch_start

            # ← ВОТ ГЛАВНОЕ: сохраняем ПОЛНЫЙ результат после каждого батча
            batch_filename = os.path.join(batch_dir, f"batch_{batch_num:04d}_of_{total_batches}.json")
            snapshot = {
                "batch_number": batch_num,
                "processed_mentions": processed,
                "total_mentions": total_mentions,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clinical_recommendations": all_kr_results  # ← ВСЁ, что нужно для восстановления
            }
            try:
                with open(batch_filename, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                print(f"Прогресс сохранён → {batch_filename}")
            except Exception as e:
                print(f"ОШИБКА сохранения батча: {e}")

            # Прогноз времени
            elapsed = time.time() - start_time
            try:
                eta = (elapsed / processed) * (total_mentions - processed)
            except:
                eta = 0
            print(f"Батч {batch_num}/{total_batches} готов за {batch_time:.1f}с | "
                  f"Обработано: {processed}/{total_mentions} | Осталось ~{eta:.0f}с")

            batch_num += 1

    # Статистика (остаётся как у тебя)
    for kr in all_kr_results:
        stats = kr.setdefault("detailed_stats", {})
        for src, field in [("mediq", "mediq_matches"), ("blacklist", "blacklist_matches"), ("marker", "marker_matches")]:
            cat_stats = {
                "total_mentions": 0, "recommended": 0, "contraindicated": 0,
                "literature": 0, "not_found": 0,
                "udd_distribution": {}, "uur_distribution": {},
            }
            for term_obj in kr.get(field, []):
                for ctx_obj in term_obj.get("contexts", []):
                    analysis = ctx_obj.get("analysis") or {}
                    cat_stats["total_mentions"] += 1
                    t = (analysis.get("Тип") or "").lower()
                    if "рекомендация" in t: cat_stats["recommended"] += 1
                    elif "противопоказание" in t: cat_stats["contraindicated"] += 1
                    elif "литература" in t: cat_stats["literature"] += 1
                    elif "не" in t and "обнаружено" in t: cat_stats["not_found"] += 1

                    for key in ("УДД", "УУР"):
                        val = analysis.get(key)
                        if val:
                            cat_stats[f"{key.lower()}_distribution"][val] = cat_stats[f"{key.lower()}_distribution"].get(val, 0) + 1
            stats[src] = cat_stats

    total_time = time.time() - start_time
    print(f"\nLLM-анализ полностью завершён за {total_time:.1f} сек. ({total_mentions} упоминаний)")


# --- Основная функция main ---
async def main():
    # Конфигурация
    config = {
        "metadata_path": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/MetaData.json',
        "pdf_folder": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/Клинические_Рекомендации',
        "blacklist_json_path": "blacklist_drugs.json",
        "clinical_recommendations_json": "clinical_recommendations.json",
        "additional_drug_info_path": "drugs.json",
        "axtung_path": "AXTUNG.Json",
        "max_concurrent_pdf": 36,
        "max_pdf_workers": 36,
        "pdf_batch_size": 10,
        "json_indent": 2,
        "context_before": 1000,
        "context_after": 1000,
    }

    # --- Этап 0: загрузка источников и КР ---
    mediq_data, blacklist_data, axtung_data = load_sources(config)
    terms_by_id, form_to_term_ids, mediq_meta, blacklist_meta, marker_meta = build_unified_index(
        mediq_data, blacklist_data, axtung_data
    )
    automaton = build_aho_automaton_unified(form_to_term_ids)

    clinical_recommendations = await load_data(config)

    # --- Этап 1: поиск совпадений и формирование JSON ---
    print("\n=== Этап 1: поиск совпадений в клинических рекомендациях ===")
    all_kr_results = []
    global_counts = {"mediq": {}, "blacklist": {}, "marker": {}}

    for kr_key, kr_data in clinical_recommendations.items():
        text = kr_data.get("Текст", "")
        mentions = await find_all_mentions(
            automaton,
            terms_by_id,
            text,
            context_before=config["context_before"],
            context_after=config["context_after"],
            verbose=False,
        )
        kr_result = build_kr_result_stage1(kr_key, kr_data, mentions, mediq_meta, blacklist_meta, marker_meta)
        all_kr_results.append(kr_result)

        # Обновляем глобальный топ
        for src, bucket in [
            ("mediq", mentions["mediq"]),
            ("blacklist", mentions["blacklist"]),
            ("marker", mentions["marker"]),
        ]:
            for term, ctxs in bucket.items():
                global_counts[src][term] = global_counts[src].get(term, 0) + len(ctxs)

        print(
            f"КР {kr_key}: MedIQ {kr_result['summary']['mediq']['mentions']} упоминаний, "
            f"Blacklist {kr_result['summary']['blacklist']['mentions']} упоминаний, "
            f"AXTUNG {kr_result['summary']['marker']['mentions']} упоминаний"
        )

    # --- Перенос терминов с > 500 упоминаний в отдельный JSON ---
    overflow_threshold = 200
    overflow_terms = {
        src: {term for term, cnt in global_counts[src].items() if cnt > overflow_threshold}
        for src in ("mediq", "blacklist", "marker")
    }

    overflow_store = {"mediq": {}, "blacklist": {}, "marker": {}}

    for kr in all_kr_results:
        for src, field in [
            ("mediq", "mediq_matches"),
            ("blacklist", "blacklist_matches"),
            ("marker", "marker_matches"),
        ]:
            terms_list = kr.get(field, [])
            kept_terms = []
            for term_obj in terms_list:
                term = term_obj.get("term")
                contexts = term_obj.get("contexts", [])
                if term in overflow_terms[src]:
                    # Обновляем summary для КР
                    kr["summary"][src]["terms"] -= 1
                    kr["summary"][src]["mentions"] -= len(contexts)

                    # Переносим в overflow_store
                    ov = overflow_store[src].setdefault(
                        term,
                        {
                            "source": src,
                            "term": term,
                            "atx": term_obj.get("atx"),
                            "atxGroup": term_obj.get("atxGroup"),
                            "description": term_obj.get("description"),
                            "total_mentions": 0,
                            "by_kr": [],
                        },
                    )
                    ov["total_mentions"] += len(contexts)
                    ov["by_kr"].append(
                        {
                            "kr_id": kr.get("kr_id"),
                            "kr_name": kr.get("kr_name"),
                            "kr_link": kr.get("kr_link"),
                            "contexts": [c.get("context") for c in contexts],
                        }
                    )
                else:
                    kept_terms.append(term_obj)
            kr[field] = kept_terms

    overflow_terms_list = []
    for src in ("mediq", "blacklist", "marker"):
        for _term, data in overflow_store[src].items():
            overflow_terms_list.append(data)

    overflow_file = "clinical_matches_overflow.json"
    with open(overflow_file, "w", encoding="utf-8") as f:
        json.dump({"overflow_terms": overflow_terms_list}, f, ensure_ascii=False, indent=config["json_indent"])
    print(
        f"\nТермины с более чем {overflow_threshold} упоминаниями перенесены в {overflow_file}. "
        f"Всего таких терминов: {len(overflow_terms_list)}"
    )

    # Сохраняем результаты этапа 1 (уже без тяжёлых терминов)
    stage1_file = "clinical_matches_stage1.json"
    with open(stage1_file, "w", encoding="utf-8") as f:
        json.dump({"clinical_recommendations": all_kr_results}, f, ensure_ascii=False, indent=config["json_indent"])
    print(f"\nЭтап 1 завершён. Результаты сохранены в {stage1_file}")

    # Проверяем пороги по количеству упоминаний перед этапом 2
    totals_by_src = {"mediq": 0, "blacklist": 0, "marker": 0}
    for kr in all_kr_results:
        for src in ("mediq", "blacklist", "marker"):
            totals_by_src[src] += kr["summary"][src]["mentions"]

    print(
        f"\nСвод по количеству упоминаний (после переноса тяжёлых терминов): "
        f"Mediq={totals_by_src['mediq']}, "
        f"Blacklist={totals_by_src['blacklist']}, "
        f"Markers={totals_by_src['marker']}"
    )

    # Выводим топ по категориям (по полному счётчику, до переноса)
    print("\nТоп упоминаний по категориям (до переноса):")
    for src in ("mediq", "blacklist", "marker"):
        items = sorted(global_counts[src].items(), key=lambda x: x[1], reverse=True)
        print(f"\n[{src.upper()}] уникальных терминов: {len(items)}")
        for term, cnt in items[:10]:
            print(f"  {term}: {cnt}")

    # --- Этап 2: LLM-анализ контекстов (только если нет перегруза по совпадениям) ---
    if any(totals_by_src[src] > 20000 for src in totals_by_src):
        print(
            "\nЭтап 2 (LLM-анализ) пропущен, так как в одной из категорий "
            "количество совпадений превысило 20000. "
            "Проверьте сводные числа выше и сузьте список или критерии."
        )
        return

    print("\n=== Этап 2: LLM-анализ контекстов ===")
    await perform_llm_analysis_on_matches(all_kr_results, batch_size=100)

    stage2_file = "clinical_matches_stage2.json"
    with open(stage2_file, "w", encoding="utf-8") as f:
        json.dump({"clinical_recommendations": all_kr_results}, f, ensure_ascii=False, indent=config["json_indent"])
    print(f"\nЭтап 2 завершён. Обогащённый JSON сохранён в {stage2_file}")

    # Также формируем CSV для дальнейшего анализа
    csv_file = "clinical_matches_stage2.csv"
    rows_written = 0
    with open(csv_file, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv, delimiter=";")
        writer.writerow(
            [
                "kr_id",
                "kr_name",
                "kr_link",
                "source",
                "term",
                "atx",
                "atxGroup",
                "blacklist_description",
                "УДД",
                "УУР",
                "Тип",
                "Комментарий",
                "context",
                "metadata_json",
            ]
        )

        for kr in all_kr_results:
            kr_id = kr.get("kr_id")
            kr_name = kr.get("kr_name")
            kr_link = kr.get("kr_link")
            metadata_str = json.dumps(kr.get("metadata", {}), ensure_ascii=False)

            for src, field in [
                ("mediq", "mediq_matches"),
                ("blacklist", "blacklist_matches"),
                ("marker", "marker_matches"),
            ]:
                for term_obj in kr.get(field, []):
                    term = term_obj.get("term")
                    atx = term_obj.get("atx")
                    atx_group = term_obj.get("atxGroup")
                    bl_desc = term_obj.get("description")
                    for ctx_obj in term_obj.get("contexts", []):
                        analysis = ctx_obj.get("analysis") or {}
                        writer.writerow(
                            [
                                kr_id,
                                kr_name,
                                kr_link,
                                src,
                                term,
                                atx,
                                atx_group,
                                bl_desc,
                                analysis.get("УДД"),
                                analysis.get("УУР"),
                                analysis.get("Тип"),
                                analysis.get("Комментарий"),
                                ctx_obj.get("context"),
                                metadata_str,
                            ]
                        )
                        rows_written += 1

    print(f"CSV с результатами этапа 2 сохранён в {csv_file} (строк: {rows_written})")

if __name__ == "__main__":
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
