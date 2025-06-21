import json
import asyncio
import time
import os
from tqdm import tqdm
import aiohttp
from asyncio import WindowsSelectorEventLoopPolicy
import ahocorasick

# Импортируем вспомогательные функции
from Logic.Clinick_Extract import ClinicalAnalyzer
from Logic.Levels_extract import analyze_mention
from Logic.Black_Extract import create_drug_dict, check_blacklist_drugs, build_aho_automaton_word_forms
from Logic.Work_witch_word import generate_word_forms

# --- Новые функции для батчевого поиска упоминаний через Ахо–Корасик ---
def build_aho_automaton_from_list(drug_list):
    """
    Строит Ахо–Корасик автомат для оригинальных названий препаратов из drug_list.
    Значением для каждого названия является кортеж: (очищенное_название, [список оригиналов]).
    """
    form_to_drugs = {}
    
    for drug in drug_list:
        # Очищаем название: нижний регистр + удаление пробелов по краям
        drug_clean = drug.lower().strip()
        
        # Добавляем только очищенную форму (без генерации вариантов)
        if drug_clean not in form_to_drugs:
            form_to_drugs[drug_clean] = [drug]
        else:
            if drug not in form_to_drugs[drug_clean]:
                form_to_drugs[drug_clean].append(drug)
    
    # Создаем автомат
    A = ahocorasick.Automaton()
    for form, drugs in form_to_drugs.items():
        A.add_word(form, (form, drugs))  # Ключ: очищенная форма, значение: список оригиналов
    
    A.make_automaton()
    return A

async def find_mentions_batch(automaton, text, config, verbose=False):
    """
    Выполняет однократный проход по тексту клинической рекомендации с использованием автомата.
    Возвращает словарь, где ключ — оригинальное поисковое имя препарата,
    а значение — список найденных контекстов.
    """
    text_lower = text.lower()
    results = {}    
    for end_index, (matched_form, drugs) in automaton.iter(text_lower):
        match_start = end_index - len(matched_form) + 1
        context_start = max(0, match_start - config["context_before"])
        context_end = min(len(text), end_index + 1 + config["context_after"])
        context = text[context_start:context_end]
        for drug in drugs:
            if drug not in results:
                results[drug] = []
            results[drug].append(context)
            if verbose:
                print(f"[{drug}] Найдено упоминание: {context[:50]}...")
    return results

# --- Обновлённая функция извлечения упоминаний из клинических рекомендаций ---
async def process_clinical_recommendation_extraction(kr_key, kr_data, preparations, blacklist_results, config, additional_drug_dict, cr_automaton):
    """
    Этап 1. Извлечение упоминаний препаратов без LLM-анализа.
    Для каждой клинической рекомендации выполняется один проход по тексту через автомат,
    после чего для каждого препарата из батча берутся найденные контексты.
    """
    result = {
        "kr_name": kr_data["Название"],
        "kr_link": kr_data["Ссылка"],
        "drugs_mentioned": [],
        "summary": {
            "total_drugs_mentioned": 0,
            "drugs_in_blacklist": 0,
            "total_mentions": 0,
            "recommended_mentions": 0,
            "contraindicated_mentions": 0
        }
    }
    
    # Выполняем батчевый поиск упоминаний по всему тексту рекомендации
    mentions_dict = await find_mentions_batch(cr_automaton, kr_data["Текст"], config, verbose=False)
    
    for prep in preparations:
        prep_lower = prep.lower()
        additional_infos = additional_drug_dict.get(prep_lower, [])
        # Определяем поисковое имя: если препарат задан через ATX-код – берем имя из доп. информации, иначе оставляем prep
        is_atx = additional_infos and any(info["drug"].get("atx", "").lower() == prep_lower for info in additional_infos)
        search_name = additional_infos[0]["drug"]["name"] if is_atx and additional_infos else prep

        # Получаем упоминания для данного препарата из результатов батчевого поиска
        mentions = mentions_dict.get(search_name, [])
        if mentions:
            # Поиск информации из черного списка по search_name
            blacklist_info = next((item for item in blacklist_results if item["preparation"] == search_name), None)
            if blacklist_info is None:
                in_blacklist = False
                blacklist_description = None
            else:
                in_blacklist = blacklist_info["in_blacklist"]
                blacklist_description = blacklist_info["blacklist_description"]
            
            drug_info = {
                "drug": prep,
                "in_blacklist": in_blacklist,
                "blacklist_description": blacklist_description,
                "mentions": []
            }
            
            # Поиск дополнительной информации
            if additional_infos:
                drug_info["additional_info"] = []
                for info in additional_infos:
                    mnn_data = info["drug"].get("mnn", {})
                    mnn_name = mnn_data.get("name", "Не указано")
                    description = info["drug"]["description"]
                    criterions = mnn_data.get("criterions", [])
                    points = {
                        "WHO_ADULT_LIST": next((c["points"] for c in criterions if c["criterion"] == "WHO_ADULT_LIST"), 0),
                        "RXLIST": next((c["points"] for c in criterions if c["criterion"] == "RXLIST"), 0),
                        "COCHRANE": next((c["points"] for c in criterions if c["criterion"] == "COCHRANE"), 0),
                        "WHO_KIDS_LIST": next((c["points"] for c in criterions if c["criterion"] == "WHO_KIDS_LIST"), 0),
                        "PUBMED": next((c["points"] for c in criterions if c["criterion"] == "PUBMED"), 0)
                    }
                    drug_info["additional_info"].append({
                        "mnn_name": mnn_name,
                        "description": description,
                        "points": points
                    })
            
            for context in mentions:
                drug_info["mentions"].append({
                    "context": context,
                    "analysis": None
                })
                result["summary"]["total_mentions"] += 1
            
            result["drugs_mentioned"].append(drug_info)
            result["summary"]["total_drugs_mentioned"] += 1
            if drug_info["in_blacklist"]:
                result["summary"]["drugs_in_blacklist"] += 1
    
    return result

# --- Функция загрузки клинических рекомендаций ---
async def load_data(config):
    """Загрузка клинических рекомендаций."""
    analyzer = ClinicalAnalyzer(config)
    await analyzer.load_metadata()
    await analyzer.load_clinical_recommendations()
    return analyzer.clinical_recommendations

async def perform_llm_analysis(all_kr_results, batch_size):
    """
    Этап 2. Обработка извлечённых упоминаний батчами с отслеживанием прогресса.
    """
    start_time = time.time()
    total_mentions = sum(
        len(drug["mentions"]) 
        for kr in all_kr_results 
        for drug in kr["drugs_mentioned"]
    )
    
    print(f"\nНачало LLM-анализа. Всего упоминаний: {total_mentions}")
    
    async with aiohttp.ClientSession() as session:
        # Собираем все задачи
        tasks = []
        task_refs = []
        
        for kr_idx, kr_result in enumerate(all_kr_results):
            for drug_idx, drug_info in enumerate(kr_result["drugs_mentioned"]):
                for mention_idx, mention in enumerate(drug_info["mentions"]):
                    task = analyze_mention(
                        drug_info["drug"], 
                        mention["context"], 
                        session
                    )
                    tasks.append(task)
                    task_refs.append( (kr_idx, drug_idx, mention_idx) )

        # Обрабатываем батчами
        processed = 0
        batch_num = 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
        for i in range(0, len(tasks), batch_size):
            batch_start = time.time()
            
            current_batch = tasks[i:i+batch_size]
            current_refs = task_refs[i:i+batch_size]
            
            # Выполняем текущий батч
            analyses = await asyncio.gather(*current_batch)
            
            # Обновляем результаты
            for (kr_idx, drug_idx, mention_idx), analysis in zip(current_refs, analyses):
                all_kr_results[kr_idx]["drugs_mentioned"][drug_idx]["mentions"][mention_idx]["analysis"] = analysis
            
            # Рассчитываем статистику
            batch_time = time.time() - batch_start
            processed += len(current_batch)
            remaining = len(tasks) - processed
            avg_time_per_batch = (time.time() - start_time) / batch_num
            est_remaining = avg_time_per_batch * (remaining // batch_size + 1)
            
            print(
                f"LLM батч {batch_num}/{total_batches} обработан за {batch_time:.1f} сек. "
                f"Осталось: ~{est_remaining:.1f} сек. "
                f"({remaining} упоминаний)"
            )
            batch_num += 1

    # Обновляем итоговую статистику
    for kr_result in all_kr_results:
        recommended_count = 0
        contraindicated_count = 0
        for drug_info in kr_result["drugs_mentioned"]:
            for mention in drug_info["mentions"]:
                if mention["analysis"]:
                    if mention["analysis"].get("recommended", False):
                        recommended_count += 1
                    if mention["analysis"].get("contraindicated", False):
                        contraindicated_count += 1
        kr_result["summary"]["recommended_mentions"] = recommended_count
        kr_result["summary"]["contraindicated_mentions"] = contraindicated_count

    total_time = time.time() - start_time
    print(f"\nLLM-анализ завершён за {total_time:.1f} сек. ({total_mentions} упоминаний)")


# --- Основная функция main ---
async def main():
    # Загрузка списка препаратов для анализа
    with open('filtered_names_drugs_NEW.json', 'r', encoding='utf-8') as f:
        preparations = json.load(f)

    batch_size = 33730
    total_mentions = 0

    # Конфигурация
    config = {
        "metadata_path": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/MetaData.json',
        "pdf_folder": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/Клинические_Рекомендации',
        "blacklist_json_path": "blacklist_drugs.json",
        "clinical_recommendations_json": "clinical_recommendations.json",
        "additional_drug_info_path": "drugs.json",
        "max_concurrent_pdf": 36,
        "max_pdf_workers": 36,
        "pdf_batch_size": 10,
        "json_indent": 2,
        "max_drug_workers": 36,
        "tqdm_total": 1 + len(preparations),
        "context_before": 2000,
        "context_after": 2000
    }

    # Загрузка чёрного списка и построение индекса (один раз)
    try:
        with open(config["blacklist_json_path"], 'r', encoding='utf-8') as f:
            blacklist_drugs = json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки чёрного списка: {e}")
        blacklist_drugs = []
    
    drug_dict = create_drug_dict(blacklist_drugs)
    automaton = build_aho_automaton_word_forms(drug_dict)

    # Загрузка дополнительной информации о препаратах
    try:
        with open(config["additional_drug_info_path"], 'r', encoding='utf-8') as f:
            additional_drug_info = json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки дополнительной информации о препаратах: {e}")
        additional_drug_info = []
    
    # Создание словаря для поиска по имени и ATX-коду
    additional_drug_dict = {}
    for item in additional_drug_info:
        name = item["drug"]["name"].lower()
        atx = item["drug"].get("atx", "").lower()
        if name not in additional_drug_dict:
            additional_drug_dict[name] = []
        additional_drug_dict[name].append(item)
        if atx:
            if atx not in additional_drug_dict:
                additional_drug_dict[atx] = []
            additional_drug_dict[atx].append(item)

    batch_number = 1

    for i in range(0, len(preparations), batch_size):
        batch = preparations[i:i + batch_size]
        start_time = time.time()
        
        # Формирование поисковых имён для батча с учётом дополнительной информации
        search_names = []
        drug_mapping = {}  # Для сопоставления search_name с оригинальным названием препарата
        for drug in batch:
            drug_lower = drug.lower()
            additional_infos = additional_drug_dict.get(drug_lower, [])
            is_atx = additional_infos and any(info["drug"].get("atx", "").lower() == drug_lower for info in additional_infos)
            search_name = additional_infos[0]["drug"]["name"] if is_atx and additional_infos else drug
            search_names.append(search_name)
            drug_mapping[search_name] = drug

        # Один вызов для проверки всех препаратов в чёрном списке с новым алгоритмом
        results = await check_blacklist_drugs(search_names, drug_dict, automaton)
        blacklist_results = []
        for res, search_name in zip(results, search_names):
            res["preparation"] = drug_mapping[search_name]
            blacklist_results.append(res)

        elapsed_time = time.time() - start_time
        print(f"Проверка в чёрном списке батча {batch_number} выполнена за {elapsed_time} секунд")
        
        # Построение автомата для поиска упоминаний в клинических рекомендациях (используем те же search_names)
        cr_automaton = build_aho_automaton_from_list(search_names)

        # Загрузка клинических рекомендаций
        clinical_recommendations = await load_data(config)
        start_time = time.time()
        all_kr_results = []

        # Параллельное извлечение упоминаний из клинических рекомендаций с использованием нового автомата
        extraction_tasks = []
        for kr_key, kr_data in clinical_recommendations.items():
            extraction_tasks.append(
                process_clinical_recommendation_extraction(
                    kr_key, kr_data, batch, blacklist_results, config, additional_drug_dict, cr_automaton
                )
            )
        all_kr_results = await asyncio.gather(*extraction_tasks)

        mentions_in_batch = sum(kr["summary"]["total_mentions"] for kr in all_kr_results)
        total_mentions += mentions_in_batch

        elapsed_time = time.time() - start_time
        print(f"Извлечение упоминаний в батче {batch_number} заняло {elapsed_time} секунд")
        print(f"Количество упоминаний в батче {batch_number}: {mentions_in_batch}")

        start_time = time.time()
        await perform_llm_analysis(all_kr_results, batch_size=1000)  # Размер батча можно регулировать
        elapsed_time = time.time() - start_time
        print(f"LLM-анализ батча {batch_number} завершён за {elapsed_time} секунд")

        # Сохранение результатов батча
        batch_file = f"Match_Clinick_batch_{batch_number}.json"
        try:
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump({"clinical_recommendations": all_kr_results}, f, ensure_ascii=False, indent=config["json_indent"])
            print(f"Данные батча {batch_number} сохранены в {batch_file}")
        except Exception as e:
            print(f"Ошибка при сохранении данных для батча {batch_number}: {e}")

        batch_number += 1

    print(f"Общее количество упоминаний: {total_mentions}")

if __name__ == "__main__":
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
