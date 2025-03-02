import json
import re
import os
import fitz  # PyMuPDF
import asyncio
import aiohttp
import g4f
from tqdm import tqdm
import aiofiles
import concurrent.futures
from hashlib import md5
from asyncio import WindowsSelectorEventLoopPolicy
import pdfplumber
import re
from fuzzywuzzy import fuzz


# Глобальные переменные для хранения данных
global_clinical_recommendations = {}
global_blacklist_drugs = []

class ClinicalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.clinical_recommendations = {}
        self.metadata = {}

    async def load_metadata(self):
        """Асинхронная загрузка и предобработка метаданных"""
        try:
            async with aiofiles.open(self.config["metadata_path"], 'r', encoding='utf-8') as f:
                content = await f.read()
                self.metadata = json.loads(content)
            self.normalized_metadata = {self.normalize_kr_key(k): v for k, v in self.metadata.items()}
        except Exception as e:
            print(f"Ошибка загрузки метаданных: {str(e)}")

    async def load_clinical_recommendations(self):
        """Загрузка или обработка PDF-файлов"""
        if await self.load_from_json():
            print(f"Клинические рекомендации загружены из {self.config['clinical_recommendations_json']}")
            return

        pdf_folder = self.config["pdf_folder"]
        if not os.path.exists(pdf_folder):
            print(f"Папка {pdf_folder} не найдена!")
            return
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"В папке {pdf_folder} нет PDF-файлов!")
            return
        print(f"Найдено {len(pdf_files)} PDF-файлов в папке {pdf_folder}")

        semaphore = asyncio.Semaphore(self.config["max_concurrent_pdf"])
        async def sem_process_pdf(pdf_file):
            async with semaphore:
                return await self.process_pdf_file(pdf_file)

        tasks = [sem_process_pdf(pdf_file) for pdf_file in pdf_files]
        print(f"Начинается обработка {len(pdf_files)} PDF-файлов с максимум {self.config['max_concurrent_pdf']} процессами")
        await asyncio.gather(*tasks)
        print("Обработка PDF-файлов завершена")
        await self.save_to_json()

    async def process_pdf_file(self, pdf_file):
        """Обработка PDF-файла"""
        try:
            base_name = os.path.splitext(pdf_file)[0]
            number = self.extract_number(base_name)
            
            if not number:
                print(f"Не удалось извлечь номер из имени файла: {base_name}")
                return
            
            related_keys = self.find_related_metadata(number)
            if not related_keys:
                print(f"Нет связанных метаданных для номера {number}")
                return
            
            text = await self.parallel_pdf_processing(pdf_file)
            
            for key in related_keys:
                self.clinical_recommendations[key] = {
                    "Текст": text,
                    "Название": self.metadata.get(key, {}).get("Название клинической рекомендации", "Неизвестно"),
                    "Ссылка": await self.generate_kr_link(key)
                }
                print(f"Успешно обработан файл {pdf_file} для ключа {key}")
        
        except Exception as e:
            print(f"Ошибка обработки файла {pdf_file}: {str(e)}")

    async def save_to_json(self):
        """Сохранение клинических рекомендаций в JSON с хэшем PDF"""
        try:
            pdf_files = [f for f in os.listdir(self.config["pdf_folder"]) if f.endswith('.pdf')]
            pdf_hash = await self.calculate_expected_size(pdf_files)
            data = {
                "pdf_hash": pdf_hash,
                "recommendations": self.clinical_recommendations
            }
            async with aiofiles.open(self.config["clinical_recommendations_json"], 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=self.config["json_indent"]))
            print(f"Клинические рекомендации сохранены в {self.config['clinical_recommendations_json']} с хэшем {pdf_hash}")
        except Exception as e:
            print(f"Ошибка сохранения в JSON: {str(e)}")

    async def load_from_json(self):
        """Загрузка клинических рекомендаций из JSON и проверка актуальности"""
        if not os.path.exists(self.config["clinical_recommendations_json"]):
            print(f"Файл {self.config['clinical_recommendations_json']} не найден, требуется обработка PDF")
            return False

        try:
            async with aiofiles.open(self.config["clinical_recommendations_json"], 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                saved_hash = data.get("pdf_hash", "")
                self.clinical_recommendations = data.get("recommendations", {})

            pdf_folder = self.config["pdf_folder"]
            pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
            current_hash = await self.calculate_expected_size(pdf_files)

            if not saved_hash or saved_hash != current_hash:
                print(f"Хэш PDF изменился (сохраненный: {saved_hash}, текущий: {current_hash}), требуется повторная обработка")
                self.clinical_recommendations = {}
                return False
            if not self.clinical_recommendations:
                print("Данные в JSON пусты, требуется повторная обработка")
                return False
            return True
        except Exception as e:
            print(f"Ошибка загрузки из JSON: {str(e)}")
            return False

    async def calculate_expected_size(self, pdf_files):
        """Вычисление хэша PDF-файлов"""
        loop = asyncio.get_event_loop()
        hash_obj = md5()
        for pdf_file in sorted(pdf_files):
            pdf_path = os.path.join(self.config["pdf_folder"], pdf_file)
            with open(pdf_path, 'rb') as f:
                hash_obj.update(f.read())
        return hash_obj.hexdigest()

    def extract_number(self, base_name):
        """Извлечение номера"""
        match = re.search(r'\d+', base_name)
        return match.group(0) if match else None

    def find_related_metadata(self, number):
        """Поиск связанных метаданных"""
        return [k for k in self.normalized_metadata if k.startswith(f"{number}_")]

    async def parallel_pdf_processing(self, pdf_file):
        """Параллельная обработка страниц PDF"""
        pdf_path = os.path.join(self.config["pdf_folder"], pdf_file)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config["max_pdf_workers"]) as executor:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(executor, self._optimized_pdf_extraction, pdf_path)
        return await self.clean_text(text)

    def _optimized_pdf_extraction(self, pdf_path):
        """Оптимизированное извлечение текста с использованием pdfplumber в случае ошибки MuPDF"""
        text = []
        try:
            doc = fitz.open(pdf_path)
            for i in range(0, len(doc), self.config["pdf_batch_size"]):
                batch = [doc.load_page(j) for j in range(i, min(i + self.config["pdf_batch_size"], len(doc)))]
                text.append(" ".join(page.get_text("text", flags=fitz.TEXTFLAGS_SEARCH) for page in batch))
            doc.close()
        except Exception as e:
            print(f"Ошибка MuPDF для {pdf_path}: {str(e)}, пытаемся pdfplumber")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = [" ".join(page.extract_text() or "" for page in pdf.pages)]
            except Exception as e2:
                print(f"Ошибка pdfplumber для {pdf_path}: {str(e2)}")
                return ""
        return " ".join(text)

    def normalize_kr_key(self, key):
        """Нормализация ключа метаданных"""
        return re.sub(r'[^0-9_]', '', key).strip('_')

    async def generate_kr_link(self, kr_key):
        """Генерация ссылки на клиническую рекомендацию"""
        parts = kr_key.split('_')
        base = parts[0]
        suffix = parts[1] if len(parts) > 1 else "1"
        return f"https://cr.minzdrav.gov.ru/view-cr/{base}_{suffix}"

    async def clean_text(self, text):
        """Очистка текста"""
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def create_drug_dict(blacklist_drugs):
    """Преобразование списка препаратов в словарь для быстрого доступа."""
    drug_dict = {}
    for entry in blacklist_drugs:
        main_name = entry["Название препарата"].lower().strip()
        drug_dict[main_name] = entry["Описание"]
        for alt in entry["Альтернативные названия"]:
            alt_name = alt.lower().strip()
            drug_dict[alt_name] = entry["Описание"]
    return drug_dict
# После класса ClinicalAnalyzer и других функций

def get_word_stem(word):
    """Определяет основу слова, отрезая типичные окончания."""
    word = word.lower().strip()
    # Список окончаний для всех частей речи (начинаем с длинных)
    endings = [
        "ая", "яя", "ия", "ое", "ее", "ье", "ый", "ий", "ой", "ые", "ие",  # Прилагательные
        "ого", "ому", "ым", "ими", "ую", "ей", "им", "их", "ыми", "ою",      # Прилагательные (падежи)
        "а", "я", "о", "е", "ь", "й", "и", "ы", "у", "ю", "ом", "ем", "ём",  # Существительные
        "ами", "ями", "ах", "ях", "ов", "ев", "ёв", "ам", "ям", "ей", "ой", "ою", "ми", "мя", "ин", "ын",
        "ть", "ти", "чь", "у", "ю", "ешь", "ет", "ем", "ете", "ут", "ют",   # Глаголы
        "л", "ла", "ло", "ли", "я", "а", "в", "вши", "ши",                  # Глаголы (прошедшее, деепричастия)
        "ущий", "ющий", "ащий", "ящий", "вший", "ший", "емый", "омый", "имый"  # Причастия
    ]
    for ending in sorted(endings, key=len, reverse=True):  # От длинных к коротким
        if word.endswith(ending):
            return word[:-len(ending)]
    return word  # Если нет окончания, возвращаем слово как есть

def generate_word_forms(word):
    """Генерирует все возможные словоформы слова для всех частей речи и падежей."""
    word = word.lower().strip()
    stem = get_word_stem(word)
    
    # Полный список окончаний
    noun_endings = [
        "", "-а", "-я", "-о", "-е", "-ь", "-й", "-и", "-ы", "-у", "-ю", "-ом", "-ем", "-ём",
        "-ами", "-ями", "-ах", "-ях", "-ов", "-ев", "-ёв", "-ам", "-ям", "-ей", "-ой", "-ою",
        "-ми", "-мя", "-ин", "-ын"
    ]
    adj_endings = [
        "-ый", "-ий", "-ой", "-ая", "-яя", "-ия", "-ое", "-ее", "-ье", "-ые", "-ие",
        "-ого", "-ому", "-ым", "-ими", "-ую", "-ей", "-им", "-их", "-ыми", "-ой", "-ою"
    ]
    verb_endings = [
        "-ть", "-ти", "-чь", "-у", "-ю", "-ешь", "-ет", "-ем", "-ете", "-ут", "-ют",
        "-л", "-ла", "-ло", "-ли", "-я", "-а", "-в", "-вши", "-ши",
        "-ущий", "-ющий", "-ащий", "-ящий", "-вший", "-ший", "-емый", "-омый", "-имый"
    ]
    adverb_endings = ["-о", "-е", "-и", "-у"]
    
    # Генерация словоформ
    forms = []
    for ending in noun_endings + adj_endings + verb_endings + adverb_endings:
        form = stem + ending
        forms.append(form)
    
    # Добавляем исходное слово и убираем дубликаты
    forms.append(word)
    return list(set(forms))

def get_description(drug_name, drug_dict, visited=None):
    """Получение описания препарата с учётом ссылок 'см.' и полных описаний."""
    if visited is None:
        visited = set()
    
    drug_name = drug_name.lower().strip()
    
    # Проверка на зацикливание
    if drug_name in visited:
        print(f"Цикл обнаружен для '{drug_name}'")
        return None
    visited.add(drug_name)
    
    # Предполагается, что функция generate_word_forms определена
    drug_forms = generate_word_forms(drug_name)
    
    for form in drug_forms:
        for key in drug_dict:
            if key.lower().strip() == form:
                desc = drug_dict[key]
                print(f"Найдено совпадение для '{drug_name}' -> '{key}': '{desc}'")
                cleaned_desc = desc.strip()
                
                # Проверяем, начинается ли описание с "см."
                if cleaned_desc.lower().startswith("см."):
                    # Извлекаем текст после "см." до точки или запятой
                    match = re.search(r"см\.\s*([^.,]+)", desc, re.IGNORECASE)
                    if match:
                        ref_text = match.group(1).strip()
                        # Проверяем количество слов
                        word_count = len(ref_text.split())
                        if word_count <= 3:
                            # Это ссылка, ищем описание рекурсивно
                            print(f"Переход по ссылке: '{key}' -> '{ref_text}'")
                            ref_desc = get_description(ref_text, drug_dict, visited)
                            if ref_desc is not None:
                                print(f"Успешно получено описание для '{ref_text}': '{ref_desc}'")
                                return ref_desc
                            else:
                                print(f"Ссылка '{ref_text}' не найдена, возвращаем исходное описание")
                                return desc
                        else:
                            # Более 3 слов — это полное описание
                            print(f"Текст после 'см.' содержит более трёх слов, возвращаем как описание: '{ref_text}'")
                            return ref_text
                # Если не "см.", возвращаем описание как есть
                return desc
    
    print(f"Препарат '{drug_name}' не найден в словаре")
    return None

def is_in_list(drug_name, drug_dict):
    """Проверка наличия препарата в расстрельном списке с учётом окончаний."""
    drug_forms = generate_word_forms(drug_name.lower().strip())
    for form in drug_forms:
        for key in drug_dict:
            if key.lower().strip() == form:
                print(f"Препарат '{drug_name}' найден как '{key}'")
                return True
    print(f"Препарат '{drug_name}' не найден")
    return False

async def check_blacklist_drugs(drugs, drug_dict):
    """Проверка препаратов в расстрельном списке с использованием drug_dict."""
    results = []
    for drug in drugs:
        in_list = is_in_list(drug, drug_dict)
        description = get_description(drug, drug_dict) if in_list else None
        results.append({
            "preparation": drug,
            "in_blacklist": in_list,
            "blacklist_description": description
        })
        print(f"[{drug}] В JSON расстрельном списке: {in_list}, Описание: {description[:50] + '...' if description else 'Не найдено'}")
    return results

async def find_mentions(drug, text, config, verbose=False):
    """Гибкий поиск упоминаний препарата в тексте с использованием словоформ из расстрельного списка"""
    
    # Генерируем все возможные словоформы очищенного названия препарата
    drug_forms = generate_word_forms(drug)
    
    mentions = []
    text_lower = text.lower()
    
    # Создаём паттерн для поиска всех словоформ
    pattern = re.compile(rf'(?:{"|".join(re.escape(form) for form in drug_forms)})(?:[а-яёa-z]*|\b)', re.IGNORECASE)
    
    if verbose:
        print(f"[{drug}] Поиск упоминаний с паттерном: {pattern.pattern}")
    
    for match in pattern.finditer(text_lower):
        start = max(0, match.start() - config["context_before"])
        end = min(len(text), match.end() + config["context_after"])
        context = text[start:end]
        mentions.append(context)
        if verbose:
            print(f"[{drug}] Найдено упоминание: {context[:50]}...")
    
    return mentions

async def analyze_mention(drug, context, session):
    """Анализ упоминания препарата с помощью GPT"""
    prompt = f"""
    \nПроанализируй контекст из клинической рекомендации:

    {context}

    \nЗадача: Найдите раздел, в котором упоминается препарат "{drug}", и извлеките:
    
    1. **Уровень достоверности**  (если указан для "{drug}", иначе оставьте поле пустым).
    Важно, что Уровень достоверности может быть указан как УДД – 5 (цифры).
    2. **Уровень убедительности** (если указан для "{drug}", иначе оставьте поле пустым).
    Важно, что Уровень достоверности может быть указан как  УУР – С(латинские или русские буквы).
   3. **Тип упоминания** (рекомендация/литература/противопоказание)

    \n**Ответ должен быть строго в формате:**  
    УДД: <значение>
    УУР: <значение>
    Тип: <тип>
    
    \n⚠ Важно:
    - **Не придумывайте значения** – если уровень не указан, оставьте поле пустым.  
    - Если в тексте указано **несколько уровней**, выберите тот, который **относится к "{drug}"**.  
    - **Не добавляйте пояснений, выводов или комментариев.**  
    - У клинических рекомендаций, зачастую, сохраняется следующая структура:
    Рекомендация:
    Уровень убедительности рекомендаций C(буква) (уровень достоверности доказательств – 5(цифра))
    Комментарии:

    - Тем не менее возможны отклонения от подобной структуры, например отсутствие комментария

    Примеры разделов:
    Для исключения вирус-индуцированной тромбоцитопении всем пациентам рекомендуется определение антител к вирусу простого герпеса (Herpes simplex virus) в крови, антител к капсидному антигену (VCA) вируса Эпштейна-Барр (Epstein-Barr virus) в крови, определение антител класса G (IgG) к капсидному антигену (VCA) вируса Эпштейна-Барр (Epstein-Barr virus) в крови, определение антител класса G (IgG) к ядерному антигену (NA) вируса Эпштейна-Барр (Epstein-Barr virus) в крови, определение антител к вирусу ветряной оспы и опоясывающего лишая (Varicella-Zoster virus) и цитомегаловирусу (Cytomegalovirus) в крови [4,9].

    Уровень убедительности рекомендаций C (уровень достоверности доказательств – 5)
    
    Комментарии: Проведение полимеразной цепной реакции (ПЦР) на эти вирусы следует проводить при положительных серологических тестах, при подозрении на рецидив, латентную инфекцию или персистенцию вируса.
    
    Еще пример:
    Рекомендуется диагностика хеликобактер пилори (Helicobacter pylori) любым методом (иммунохроматографическое экспресс-исследование кала на хеликобактер пилори, 13С-уреазный дыхательный тест на H.pylori, определение антител к хеликобактер пилори в крови; микробиологическое (культуральное) исследование биоптата стенки желудка на H.pylori, молекулярно-биологическое исследование фекалий на хеликобактер пилори; молекулярно-биологическое исследование биоптатов слизистой желудка на H. Pylori) для исключения одной из причин тромбоцитопении, у пациентов с отягощенным анамнезом или клиническими проявлениями язвенной болезни желудка и двенадцатиперстной кишки [13].

    Уровень убедительности рекомендаций C (уровень достоверности доказательств – 3)

    Рекомендуется исследование костномозгового кроветворения для исключения других заболеваний гематологической и негематологической природы. Стернальная пункция и цитологическое исследование мазка костного мозга (миелограмма) проводится всем пациентам. Трепанбиопсия и патологоанатомическое исследование биопсийного (операционного) материала костного мозга – по показаниям [4-6,9,12].

    Еще пример: 
    Препаратов хондроитина сульфата, глюкозамина, а также их комбинаций взрослым пациентам перорально или в виде растворов парентерально в соответствии с инструкциями по применению указанных препаратов для эффективного облегчения симптоматики гонартроза и получения структурно-модифицирующего эффекта при длительном использовании [81, 174, 175, 182].

    Уровень убедительности рекомендаций А (уровень достоверности доказательств – 2).
    """
    try:
        response = await asyncio.to_thread(
            g4f.ChatCompletion.create,
            model=g4f.models.deepseek_r1,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return await parse_gpt_response(response)
    except Exception as e:
        print(f"Ошибка анализа упоминания {drug}: {str(e)}")
        return {"УДД": "Не определен", "УУР": "Не определен", "Тип": "Не определен", "error": str(e)}

async def parse_gpt_response(response):
    """Парсинг ответа GPT"""
    result = {"УДД": "Не определен", "УУР": "Не определен", "Тип": "Не определен"}
    content = response.get("content", "") if isinstance(response, dict) else str(response)
    udo_match = re.search(r'УДД:\s*(\S+)', content)
    uur_match = re.search(r'УУР:\s*(\S+)', content)
    type_match = re.search(r'Тип:\s*(\S+)', content)
    if udo_match:
        result["УДД"] = udo_match.group(1)
    if uur_match:
        result["УУР"] = uur_match.group(1)
    if type_match:
        result["Тип"] = type_match.group(1)
        result["recommended"] = "рекомендация" in result["Тип"].lower()
        result["contraindicated"] = "противопоказание" in result["Тип"].lower()
    return result

async def process_drug(preparation, clinical_recommendations, shared_blacklist_drugs, config):
    """Асинхронная обработка одного препарата"""
    async with aiohttp.ClientSession() as session:
        # Создаем drug_dict из shared_blacklist_drugs
        drug_dict = create_drug_dict(shared_blacklist_drugs)

        result = {
            "preparation": preparation,
            "clinical_recommendations": [],
            "in_blacklist": False,
            "blacklist_description": None,
            "analysis_summary": {"total_mentions": 0, "recommended": 0, "contraindicated": 0},
            "comment": None  # Новое поле для комментария
        }

        # Проверка в JSON-черном списке
        print(f"[{preparation}] Начинается проверка в расстрельном списке JSON")
        blacklist_results = await check_blacklist_drugs([preparation], drug_dict)
        blacklist_result = blacklist_results[0]
        result["in_blacklist"] = blacklist_result["in_blacklist"]
        result["blacklist_description"] = blacklist_result["blacklist_description"]

        print(f"[{preparation}] Начинается поиск упоминаний в клинических рекомендациях")
        tasks = [
            asyncio.create_task(find_mentions(preparation, kr_data["Текст"], config, verbose=False))
            for kr_data in clinical_recommendations.values()
        ]
        mentions_results = await asyncio.gather(*tasks)
        print(f"[{preparation}] Поиск упоминаний завершен, найдено {sum(len(m) for m in mentions_results)} упоминаний")

        all_mentions = []
        for kr_key, kr_data in clinical_recommendations.items():
            mentions = mentions_results.pop(0)
            for mention in mentions:
                all_mentions.append((kr_key, kr_data, mention))

        print(f"[{preparation}] Начинается анализ LLM для {len(all_mentions)} упоминаний")
        analysis_tasks = [analyze_mention(preparation, mention, session) for _, _, mention in all_mentions]
        analyses = await asyncio.gather(*analysis_tasks)
        print(f"[{preparation}] Анализ LLM завершен")

        # Переменные для сбора данных
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
                        recommended_context = mention  # Первый контекст рекомендации
                if analysis.get("contraindicated", False):
                    result["analysis_summary"]["contraindicated"] += 1
                    if contraindicated_context is None:
                        contraindicated_context = mention  # Первый контекст противопоказания
                # Собираем числовые уровни достоверности (УДД)
                udd = analysis.get("УДД")
                if udd and udd.isdigit():
                    udd_values.append(int(udd))

        # Вычисляем средний уровень достоверности
        average_udd = sum(udd_values) / len(udd_values) if udd_values else None

        print(f"[{preparation}] LLM начинает анализировать применимость лекарства")

        # Генерация комментария с помощью analyze_drug
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



async def load_data(config):
    """Загрузка всех данных в глобальные переменные"""
    global global_clinical_recommendations
    analyzer = ClinicalAnalyzer(config)
    await analyzer.load_metadata()
    await analyzer.load_clinical_recommendations()
    global_clinical_recommendations = analyzer.clinical_recommendations


async def analyze_drug(drug_name, blacklist_description, mention_count, average_udd, recommended_context, contraindicated_context):
    """Генерирует комментарий о препарате с анализом его эффективности и рекомендации.

    Аргументы:
        drug_name (str): Название препарата.
        blacklist_description (str | None): Описание из расстрельного списка.
        mention_count (int): Количество упоминаний в клинических рекомендациях.
        average_udd (str | None): Средний уровень достоверности.
        recommended_context (str | None): Контекст рекомендации.
        contraindicated_context (str | None): Контекст противопоказания.

    Возвращает:
        str: Комментарий с анализом и рекомендацией.
    """
    # Формируем запрос для LLM
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
        # Вызов LLM (предполагается использование библиотеки g4f или аналога)
        response = await asyncio.to_thread(
            g4f.ChatCompletion.create,
            model=g4f.models.deepseek_r1,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        if isinstance(response, dict):
            return response.get("content", "Ошибка: комментарий не сгенерирован.")
        elif isinstance(response, str):
            return response  # Если ответ — строка, считаем её комментарием
        else:
            return "Ошибка: неожиданный формат ответа от LLM."
    except Exception as e:
        print(f"Ошибка при анализе {drug_name}: {str(e)}")
        return "Не удалось сгенерировать комментарий из-за ошибки."








from multiprocessing import Manager

def main():
    preparations = ["Кагоцел", "Эргоферон","Кагоцел", "Эргоферон","Кагоцел", "Эргоферон","Кагоцел", "Эргоферон","Кагоцел", "Эргоферон"]
    tqdm_total = 1 + len(preparations)
    
    config = {
        "metadata_path": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/MetaData.json',
        "pdf_folder": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/Клинические_Рекомендации',
        "blacklist_json_path": "blacklist_drugs.json",
        "clinical_recommendations_json": "clinical_recommendations.json",
        "max_concurrent_pdf": 36,
        "max_pdf_workers": 36,
        "pdf_batch_size": 10,
        "json_indent": 2,
        "max_drug_workers": 36,
        "tqdm_total": tqdm_total,
        "context_before": 2000,
        "context_after": 2000
    }

    # Загрузка JSON расстрельного списка
    global global_blacklist_drugs
    try:
        with open(config["blacklist_json_path"], 'r', encoding='utf-8') as f:
            global_blacklist_drugs = json.load(f)
        print(f"JSON расстрельного списка загружен, найдено {len(global_blacklist_drugs)} препаратов")
    except Exception as e:
        print(f"Ошибка загрузки JSON расстрельного списка: {str(e)}")
        global_blacklist_drugs = []

    # Создание drug_dict
    drug_dict = create_drug_dict(global_blacklist_drugs)

    # Инициализация asyncio и загрузка данных
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop()
    
    with tqdm(total=config["tqdm_total"], desc="Загрузка данных") as pbar:
        loop.run_until_complete(load_data(config))
        pbar.update(1)

        # Создание общего словаря с помощью Manager
        manager = Manager()
        shared_clinical_recommendations = manager.dict(global_clinical_recommendations)

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=config["max_drug_workers"]) as executor:
            futures = [
                executor.submit(
                    process_drug_wrapper,
                    prep,
                    shared_clinical_recommendations,  # Передаем общий словарь
                    drug_dict,
                    config
                ) for prep in preparations
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(f"Обработка препарата {result['preparation']} завершена")
                results.append(result)
                pbar.update(1)

    # Сохранение результатов
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=config["json_indent"])
    print("Результаты сохранены в results.json")








def analyze_preparations(preparations, config):
    """Функция для анализа списка препаратов, возвращает результат в виде JSON-объекта."""
    # Устанавливаем общее количество для прогресс-бара
    config["tqdm_total"] = len(preparations)

    # Используем глобальные переменные
    global global_clinical_recommendations, global_blacklist_drugs

    # Очищаем названия препаратов
    cleaned_preparations = [
        re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s-]', '', prep).lower().strip(' -')
        for prep in preparations
    ]

    # Создаем Manager для управления общими данными
    manager = Manager()
    shared_clinical_recommendations = manager.dict(global_clinical_recommendations)  # Общий словарь
    shared_blacklist_drugs = manager.list(global_blacklist_drugs)                    # Общий список



    results = []
    with tqdm(total=config["tqdm_total"], desc="Обработка препаратов") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=config["max_drug_workers"]) as executor:
            futures = [
                executor.submit(
                    process_drug_wrapper,
                    prep,
                    shared_clinical_recommendations,
                    shared_blacklist_drugs,  # Передаем общий список вместо drug_dict
                    config
                ) for prep in cleaned_preparations
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(f"Обработка препарата {result['preparation']} завершена")
                results.append(result)
                pbar.update(1)


    return results


async def initialize_globals(config):
    """Инициализация глобальных переменных для использования на сервере или в боте"""
    global global_clinical_recommendations, global_blacklist_drugs
    try:
        # Загрузка чёрного списка препаратов
        with open(config["blacklist_json_path"], 'r', encoding='utf-8') as f:
            global_blacklist_drugs = json.load(f)
        print(f"Чёрный список загружен, найдено {len(global_blacklist_drugs)} препаратов")

        # Загрузка клинических рекомендаций
        analyzer = ClinicalAnalyzer(config)
        await analyzer.load_metadata()
        await analyzer.load_clinical_recommendations()
        global_clinical_recommendations = analyzer.clinical_recommendations
        if not global_clinical_recommendations:
            raise ValueError("Клинические рекомендации не загружены")
        print(f"Клинические рекомендации загружены, найдено {len(global_clinical_recommendations)} записей")
    except Exception as e:
        print(f"Ошибка при инициализации глобальных переменных: {str(e)}")
        raise  # Передаём ошибку дальше для обработки на сервере/боте
    







if __name__ == "__main__":
    main()