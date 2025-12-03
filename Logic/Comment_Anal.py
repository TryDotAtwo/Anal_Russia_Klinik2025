import asyncio
import aiohttp
import g4f
from Logic.Black_Extract import create_drug_dict, check_blacklist_drugs
from Logic.Levels_extract import find_mentions, analyze_mention
from Logic.Work_witch_word import generate_word_forms

async def process_drug(preparation, clinical_recommendations, shared_blacklist_drugs, config):
    """Асинхронная обработка одного препарата."""
    async with aiohttp.ClientSession() as session:
        drug_dict = create_drug_dict(shared_blacklist_drugs)
        result = {
            "preparation": preparation,
            "clinical_recommendations": [],
            "in_blacklist": False,
            "blacklist_description": None,
            "analysis_summary": {"total_mentions": 0, "recommended": 0, "contraindicated": 0},
            "comment": None
        }
        # print(f"[{preparation}] Начинается проверка в расстрельном списке JSON")
        blacklist_results = await check_blacklist_drugs([preparation], drug_dict)
        blacklist_result = blacklist_results[0]
        result["in_blacklist"] = blacklist_result["in_blacklist"]
        result["blacklist_description"] = blacklist_result["blacklist_description"]

        # print(f"[{preparation}] Начинается поиск упоминаний в клинических рекомендациях")
        tasks = [
            asyncio.create_task(find_mentions(preparation, kr_data["Текст"], config, verbose=False))
            for kr_data in clinical_recommendations.values()
        ]
        mentions_results = await asyncio.gather(*tasks)
        # print(f"[{preparation}] Поиск упоминаний завершен, найдено {sum(len(m) for m in mentions_results)} упоминаний")

        all_mentions = []
        for kr_key, kr_data in clinical_recommendations.items():
            mentions = mentions_results.pop(0)
            for mention in mentions:
                all_mentions.append((kr_key, kr_data, mention))

        # print(f"[{preparation}] Начинается анализ LLM для {len(all_mentions)} упоминаний")
        analysis_tasks = [analyze_mention(preparation, mention, session) for _, _, mention in all_mentions]
        analyses = await asyncio.gather(*analysis_tasks)
        # print(f"[{preparation}] Анализ LLM завершен")

        recommended_context = None
        contraindicated_context = None
        udd_values = []
        for (kr_key, kr_data, mention), analysis in zip(all_mentions, analyses):
            if "error" not in analysis:
                result["clinical_recommendations"].append({
                    "kr_id": kr_key,
                    "kr_name": kr_data["Название"],
                    "link": kr_data["Ссылка"],
                    "context": mention,
                    "analysis": analysis
                })
                result["analysis_summary"]["total_mentions"] += 1
                if analysis.get("recommended", False):
                    result["analysis_summary"]["recommended"] += 1
                    if recommended_context is None:
                        recommended_context = mention
                if analysis.get("contraindicated", False):
                    result["analysis_summary"]["contraindicated"] += 1
                    if contraindicated_context is None:
                        contraindicated_context = mention
                udd = analysis.get("УДД")
                if udd and udd.isdigit():
                    udd_values.append(int(udd))

        average_udd = sum(udd_values) / len(udd_values) if udd_values else None
        print(f"[{preparation}] LLM начинает анализировать применимость лекарства")
        comment = await analyze_drug(
            drug_name=preparation,
            blacklist_description=result["blacklist_description"],
            mention_count=result["analysis_summary"]["total_mentions"],
            average_udd=str(average_udd) if average_udd else None,
            recommended_context=recommended_context,
            contraindicated_context=contraindicated_context
        )
        result["comment"] = comment
        return result

def process_drug_wrapper(preparation, clinical_recommendations, shared_blacklist_drugs, config):
    """Обертка для асинхронной обработки препарата."""
    import asyncio
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_drug(preparation, clinical_recommendations, shared_blacklist_drugs, config))
    loop.close()
    return result

async def analyze_drug(drug_name, blacklist_description, mention_count, average_udd, recommended_context, contraindicated_context):
    """Генерирует комментарий о препарате с анализом его эффективности."""
    prompt = f"""
    Проанализируй препарат "{drug_name}" на основе следующих данных:

    1. Описание из расстрельного списка:  
    {blacklist_description if blacklist_description else "Нет данных"}

    2. Количество упоминаний в клинических рекомендациях: {mention_count}  
    3. Средний уровень достоверности: {average_udd if average_udd else "Неизвестно"}  
    4. Контекст рекомендации:  
    {recommended_context if recommended_context else "Нет данных"}  
    5. Контекст противопоказания:  
    {contraindicated_context if contraindicated_context else "Нет данных"}

    Задача:  
    - Определи, какой у препарата механизм действия и действующее вещество. Оцени, могут ли они работать.  
    - Проанализируй доказательность препарата на основе данных из расстрельного списка и клинических рекомендаций.  
    - Составь комментарий: стоит ли применять препарат или лучше подробнее изучить ситуацию/обратиться к врачу.  

    Правила:  
    - Если в расстрельном списке указано, что препарат не относится к доказательной медицине с чётким объяснением, комментарий должен быть резко негативным.  
    - Учитывай количество упоминаний и уровень достоверности.  
    - Не выдумывай данные, которых нет.  
    - Комментарий должен быть кратким и понятным.
    - Не рекомендуй никакие другие препараты
    - Гомеопатия, рефлексотерапия, биорезонанс, релиз-активные вещества, сверхвысокие разведения и прочие подобные вещи - не имеет доказанной эффективности, следовательно не работает и быть лечением не может.
    - Расстрельный список является более авторитетным источником информации. Если в нём указано, что препарат неработает - он неработает

    Все свои рассуждения оборачивай в тег <think></think>
    Ответ должен быть строго в формате:

    Комментарий: 

    """

    try:
        response = await asyncio.to_thread(
            g4f.ChatCompletion.create,
            model=g4f.models.command_r,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        if isinstance(response, dict):
            return response.get("content", "Ошибка: комментарий не сгенерирован.")
        elif isinstance(response, str):
            return response
        else:
            return "Ошибка: неожиданный формат ответа от LLM."
    except Exception as e:
        print(f"Ошибка при анализе {drug_name}: {str(e)}")
        return "Не удалось сгенерировать комментарий из-за ошибки."
