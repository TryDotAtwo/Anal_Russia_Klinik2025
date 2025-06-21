import json
import re
import os
import fitz  # PyMuPDF
import asyncio
import aiofiles
import pdfplumber
from hashlib import md5
from asyncio import WindowsSelectorEventLoopPolicy
import concurrent.futures

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
