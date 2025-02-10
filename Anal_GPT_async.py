import json
import g4f  # gpt4free
import re
import time
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp

# Устанавливаем правильный event loop для Windows
if asyncio.get_event_loop_policy().__class__.__name__ != "WindowsSelectorEventLoopPolicy":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BLACKLIST_URL = "https://encyclopatia.ru/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D1%80%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D0%BF%D1%80%D0%B5%D0%BF%D0%B0%D1%80%D0%B0%D1%82%D0%BE%D0%B2"

def extract_levels_from_block(text, preparation_name):
    """Извлекает уровни достоверности и убедительности из ответа GPT."""
    match_evidence = re.search(r'Уровень достоверности:\s*(\S+)', text, re.IGNORECASE)
    match_conviction = re.search(r'Уровень убедительности:\s*(\S+)', text, re.IGNORECASE)
    return (
        match_evidence.group(1) if match_evidence else "Информация не найдена",
        match_conviction.group(1) if match_conviction else "Информация не найдена"
    )

async def analyze_with_gpt_for_specific_drug(preparation_name, text, session):
    """Анализирует текст с GPT и извлекает уровни достоверности и убедительности."""
    prompt = f"""
    \nДан следующий текст:

    {text}

    \nЗадача: Найдите раздел, в котором упоминается препарат "{preparation_name}", и извлеките:
    
    1. **Уровень достоверности**  (если указан для "{preparation_name}", иначе оставьте поле пустым).
    Важно, что Уровень достоверности может быть указан как УДД – 5.
    2. **Уровень убедительности** (если указан для "{preparation_name}", иначе оставьте поле пустым).
    Важно, что Уровень достоверности может быть указан как  УУР – С.
   

    \n**Ответ должен быть строго в формате:**  
    Уровень достоверности: [значение]  
    Уровень убедительности: [значение]
    
    \n⚠ Важно:
    - **Не придумывайте значения** – если уровень не указан, оставьте поле пустым.  
    - Если в тексте указано **несколько уровней**, выберите тот, который **относится к "{preparation_name}"**.  
    - **Не добавляйте пояснений, выводов или комментариев.**  
    """
    for _ in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model=g4f.models.deepseek_r1,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            result = response.strip() if isinstance(response, str) else response.get("content", "").strip()
            if result and "Уровень достоверности" in result:
                return extract_levels_from_block(result, preparation_name)
        except Exception:
            return "Ошибка при запросе к GPT", "Ошибка при запросе к GPT"
        await asyncio.sleep(5)
    return "Ошибка при запросе к GPT", "Ошибка при запросе к GPT"



import re
from bs4 import BeautifulSoup

BLACKLIST_FILE_PATH = 'Расстрельный список препаратов — Encyclopedia Pathologica.html'  # Путь к локальному HTML файлу

async def get_blacklist_info(preparation_name):
    """Получает информацию о препарате из локального HTML файла."""
    try:
        # Чтение локального HTML файла
        with open(BLACKLIST_FILE_PATH, 'r', encoding='utf-8') as file:
            html = file.read()

        # Обработка HTML с помощью BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()

        # Регулярное выражение для поиска информации о препарате
        pattern = re.compile(rf"({re.escape(preparation_name)}.*?):\s*(.*?)(?=\n[A-ЯЁA-Z]|\Z)", re.DOTALL)
        match = pattern.search(text)
        
        if match:
            description = match.group(2).strip()
            ref_match = re.search(r"см\. (.+)", description)
            if ref_match:
                referenced_name = ref_match.group(1).strip()
                # Рекурсивный вызов, если в описании упоминается другой препарат
                return await get_blacklist_info(referenced_name)
            return description
        return None

    except Exception as e:
        print(f"Ошибка при обработке локального файла: {e}")
        return None
async def analyze_drug_evidence(preparation_name, text, session):
    try:
        blacklist_info = await get_blacklist_info(preparation_name)
  

        """Анализирует препарат на предмет доказательной медицины."""
        prompt = f"""
        \nДан следующий текст:
        {text}
    
        \nДанные из "Расстрельного списка":
        {blacklist_info if blacklist_info else "(Не найдено)"}
    
        \nЗадача: Определить рекомендуют ли применять {preparation_name} в тексте клинической рекомендации, который ты получил или не рекомендуют.
        Если рекомендую то тебе нужно определить, относится ли {preparation_name} к доказательной медицине.
        - Если статьи надежные (РКИ, метаанализы в уважаемых журналах) → объясните, почему.
        - Если статей нет или они сомнительные → объясните, почему {preparation_name} нельзя считать доказательным.

        \n Крайне важно чтобы ты анализировал что написано в тексте, который тебе прислало. 
        Если там видно, что {preparation_name} не рекомендуется, а, например, находится в списке литературы, то тебе так и нужно написать, что {preparation_name} находится в списке литературы. 
        Если {preparation_name} не рекомендуется, то так и напиши, что он не рекомендуется в тексте клинической рекомендации. 
        Если же {preparation_name} рекомендуется к использованию в клинической рекомендации, то сначала напиши что клиническая рекомендация рекомендует использовать {preparation_name}, после чего напиши свой вывод является он доказанно эффективным или нет
        Отвечай максимально коротко с упором на то, в каком контексте упомянут {preparation_name} и рекумендуют ли его в клинической рекомендации
        **Ответ должен быть строго в формате:**  
        Комментарий: [Ваш ответ]
        """       
        for _ in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    g4f.ChatCompletion.create,
                    model=g4f.models.deepseek_r1,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                result = response.strip() if isinstance(response, str) else response.get("content", "").strip()
                if result and "Комментарий:" in result:
                    return result
            except GeneratorExit:  # Корректный выход из корутины
                raise
            except Exception:
                pass
            await asyncio.sleep(5)
    except GeneratorExit:  
        raise  # Позволяет корректно завершить корутину
    except Exception as e:
        print(f"Ошибка в analyze_drug_evidence: {e}")
        import traceback
        traceback.print_exc()  # Это выведет полный стек вызовов ошибки.

    return "Комментарий: Не удалось получить ответ." 




async def process_clinical_recommendations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    semaphore = asyncio.Semaphore(BUNCH_SIZE)
    tasks = []

    # Список для маркеров
    markers = []

    # Создаем сессию для всех запросов
    async with aiohttp.ClientSession() as session:
        for key, value in data.items():
            for marker_type in [
                "Какие слова из списка препаратов были обнаружены",
                "Какие слова из списка маркеров были обнаружены",
                "Какие слова из списка возможных маркеров были обнаружены"
            ]:
                for marker in value.get(marker_type, []):
                    if isinstance(marker, list) and len(marker) >= 2:
                        preparation_name = marker[0]
                        text = marker[1]
                        markers.append((semaphore, marker, preparation_name, text, session))

        total_markers = len(markers)
        
        # Разбиение маркеров на банчи
        for i in range(0, total_markers, BUNCH_SIZE):
            bunch = markers[i:i + BUNCH_SIZE]
            start_time = time.time()  # Засекаем время начала обработки банча
            await process_bunch(bunch, data, output_file, i, total_markers)
            elapsed_time = time.time() - start_time
            print(f"Банч из {len(bunch)} маркеров обработан за {elapsed_time:.2f} секунд.")
            remaining = total_markers - (i + len(bunch))
            print(f"Осталось обработать {remaining} маркеров.\n")

        print("✅ Обработка завершена!")

async def process_bunch(bunch, data, output_file, start_index, total_markers):
    """Обработка банча маркеров."""
    tasks = []
    for semaphore, marker, preparation_name, text, session in bunch:
        tasks.append(process_marker(semaphore, marker, preparation_name, text, session))
    
    # Ожидание завершения всех задач в банче
    await asyncio.gather(*tasks)

    # Обновление JSON файла после обработки каждого банча
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Банч с {start_index + 1}-{start_index + len(bunch)} маркерами обновлён в файле.")

async def process_marker(semaphore, marker, preparation_name, text, session):
    """Асинхронная обработка маркера."""
    async with semaphore:
        level_of_evidence, level_of_conviction = await analyze_with_gpt_for_specific_drug(preparation_name, text, session)  # Передаем session
        comment = await analyze_drug_evidence(preparation_name, text, session)  # Также передаем session
        marker.append({
            "Уровень убедительности": level_of_conviction,
            "Уровень достоверности": level_of_evidence,
            "Комментарий": comment
        })

BUNCH_SIZE = 200  # Количество обработанных КР перед выводом времени
MAX_RETRIES = 4  # Количество повторных попыток в случае ошибки


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(process_clinical_recommendations('combined_file.json', 'GPT.json'))
    elapsed_time = time.time() - start_time
    print(f"Обработано за {elapsed_time}")