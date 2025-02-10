import json
import g4f  # gpt4free
import re
import time
import requests
from bs4 import BeautifulSoup
import asyncio

# Устанавливаем правильный event loop для Windows
if asyncio.get_event_loop_policy().__class__.__name__ != "WindowsSelectorEventLoopPolicy":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BUNCH_SIZE = 1  # Количество обработанных КР перед выводом времени
MAX_RETRIES = 2  # Количество повторных попыток в случае ошибки
OPENALEX_API_URL = "https://api.openalex.org/works?filter=title.search:"
BLACKLIST_URL = "https://encyclopatia.ru/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D1%80%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D0%BF%D1%80%D0%B5%D0%BF%D0%B0%D1%80%D0%B0%D1%82%D0%BE%D0%B2"

def extract_levels_from_block(text, preparation_name):
    """Извлекает уровни достоверности и убедительности из ответа GPT."""

    # Ищем уровни достоверности и убедительности в любом месте текста
    match_evidence = re.search(r'Уровень достоверности:\s*(\S+)', text, re.IGNORECASE)
    match_conviction = re.search(r'Уровень убедительности:\s*(\S+)', text, re.IGNORECASE)

    # Возвращаем найденные данные, либо "Информация не найдена"
    return (
        match_evidence.group(1) if match_evidence else "Информация не найдена",
        match_conviction.group(1) if match_conviction else "Информация не найдена"
    )



def analyze_with_gpt_for_specific_drug(preparation_name, text):
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

    for attempt in range(MAX_RETRIES):
        try:
            response = g4f.ChatCompletion.create(
                model=g4f.models.deepseek_v3,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            
            result = response.strip() if isinstance(response, str) else response.get("content", "").strip()

            if result and "Уровень достоверности" in result:
                return extract_levels_from_block(result, preparation_name)

        except Exception:
            return "Ошибка при запросе к GPT", "Ошибка при запросе к GPT"

        time.sleep(5)

    return "Ошибка при запросе к GPT", "Ошибка при запросе к GPT"


def search_openalex(preparation_name):
    """Ищет упоминания препарата в научных статьях через OpenAlex API."""
    try:
        response = requests.get(OPENALEX_API_URL + preparation_name)
        if response.status_code == 200:
            data = response.json()
            return [article.get("title", "") for article in data.get("results", [])[:5]]
    except Exception as e:
        print(f"Ошибка при запросе OpenAlex: {e}")
    return []
    
def get_blacklist_info(preparation_name):
    """Получает информацию о препарате из 'Расстрельного списка'."""
    try:
        response = requests.get(BLACKLIST_URL)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Ошибка при получении страницы: {e}")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    pattern = re.compile(rf"({preparation_name}.*?):\s*(.*?)(?=\n[A-ЯЁA-Z]|\Z)", re.DOTALL)
    match = pattern.search(text)
    
    if match:
        description = match.group(2).strip()
        ref_match = re.search(r"см\. (.+)", description)
        if ref_match:
            referenced_name = ref_match.group(1).strip()
            return get_blacklist_info(referenced_name)
        return description
    return None

def analyze_drug_evidence(preparation_name, text):
    """Анализирует препарат на предмет доказательной медицины."""
    articles = search_openalex(preparation_name)
    blacklist_info = get_blacklist_info(preparation_name)
    articles_str = "\n".join(articles) if articles else "(Научные статьи не найдены)"

    prompt = f"""
    \nДан следующий текст:
    {text}
    
    \nТакже найдены следующие научные статьи о препарате "{preparation_name}":
    {articles_str}
    
    \nДанные из "Расстрельного списка":
    {blacklist_info if blacklist_info else "(Не найдено)"}
    
    \nЗадача: Определить рекомендуют ли применять этот препарат в тексте клинической рекомендации, который ты получил или не рекомендуют.
    Если рекомендую то тебе нужно определить, относится ли препарат к доказательной медицине.
    - Если статьи надежные (РКИ, метаанализы в уважаемых журналах) → объясните, почему.
    - Если статей нет или они сомнительные → объясните, почему препарат нельзя считать доказательным.

    \n Крайне важно чтобы ты анализировал что написано в тексте, который тебе прислало. 
    Если там видно, что препарат не рекомендуется, а, например, находится в списке литературы, то тебе так и нужно написать, что препарат находится в списке литературы. 
    Если препарат не рекомендуется, то так и напиши, что он не рекомендуется в тексте клинической рекомендации. 
    Если же препарат рекомендуется к использованию в клинической рекомендации, то сначала напиши что клиническая рекомендация рекомендует использовать препарат {preparation_name}, после чего напиши свой вывод является он доказанно эффективным или нет
    
    **Ответ должен быть строго в формате:**  
    Комментарий: [Ваш ответ]
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            response = g4f.ChatCompletion.create(
                model=g4f.models.deepseek_v3,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            
            result = response.strip() if isinstance(response, str) else response.get("content", "").strip()
            if result and "Комментарий:" in result:
                return result
        except Exception as e:
            print(f"Ошибка запроса: {e}")
        time.sleep(5)
    return "Комментарий: Не удалось получить ответ."

def process_clinical_recommendations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_count = 0
    start_time = time.time()
    
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
                    
                    level_of_evidence, level_of_conviction = analyze_with_gpt_for_specific_drug(preparation_name, text)
                    comment = analyze_drug_evidence(preparation_name, text)
                    
                    marker.append({
                        "Уровень убедительности": level_of_conviction,
                        "Уровень достоверности": level_of_evidence,
                        "Комментарий": comment
                    })
        
        processed_count += 1
        
        if processed_count % BUNCH_SIZE == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_object = elapsed_time / processed_count
            print(f"Обработано {processed_count} объектов. Среднее время на объект: {avg_time_per_object:.2f} сек.")
            
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

    print("✅ Обработка завершена!")

# Запуск программы

process_clinical_recommendations('combined_file2.json', 'GPT.json')
