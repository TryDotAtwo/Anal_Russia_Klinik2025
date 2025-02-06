import os
import json
import PyPDF2
import time
import fitz  # PyMuPDF
import re


def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        return json.load(file)

def clean_text(text):

    # 2. Склеиваем слова, разорванные дефисом и переносом строки
    # Пример: "синхрофа-/nзатрон" -> "синхрофазатрон"
    text = re.sub(r'(\w)-/n\s*(\w)', r'\1\2', text)

    # 3. Убираем лишние пробелы между словами
    text = re.sub(r' +', ' ', text)

    # 4. Убираем пробелы в начале и в конце текста
    text = text.strip()

    return text





def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Не удалось обработать файл {pdf_path}: {e}"

def extract_text_from_pdf_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        return f"Не удалось обработать файл {pdf_path}: {e}"

def find_words_in_text(text, word_list):
    Leight_of_bunch_info = 60
    found_words = []
    text_lower = text.lower()
    for word in word_list:
        word_lower = word.lower()
        if word_lower in text_lower:
            start_index = text_lower.find(word_lower)
            end_index = start_index + len(word_lower)
            context = text[max(0, start_index-Leight_of_bunch_info):min(len(text), end_index+Leight_of_bunch_info)]
            found_words.append((word, context))
    return found_words

def create_json_entry(pdf_path, medicines_list, markers_list, full_medicines_list):
    text = extract_text_from_pdf_pymupdf(pdf_path)
    text = clean_text(text)

    if isinstance(text, str) and text.startswith("Не удалось обработать файл"):
        return {os.path.splitext(os.path.basename(pdf_path))[0]: {"Ошибка": text}}
    
    found_medicines = find_words_in_text(text, medicines_list)
    found_markers = find_words_in_text(text, markers_list)
    found_full_medicines = find_words_in_text(text, full_medicines_list)  # Новый список для поиска
    
    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]
    
    json_entry = {
        pdf_id: {
            "Какие слова из списка препаратов были обнаружены": found_medicines,
            "Какие слова из списка маркеров были обнаружены": found_markers,
            "Какие слова из списка возможных маркеров были обнаружены": found_full_medicines  # Новое поле
        }
    }
    
    return json_entry

def save_json(json_dict, filename='output.json'):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    existing_data.update(json_dict)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# Загрузка списков слов из JSON файлов
medicines_list = load_word_list('Names_of_medicines.json')
markers_list = load_word_list('AXTUNG.json')
full_medicines_list = load_word_list('Names_of_medicines _But_Full.json')  # Загрузка нового списка

# Путь к папке с PDF файлами
pdf_folder = 'Клинические_Рекомендации'
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Создаем JSON записи для каждого PDF файла
json_dict = {}
batch_size = 50
start_time = time.time()

for i, pdf in enumerate(pdf_files, start=1):
    entry = create_json_entry(pdf, medicines_list, markers_list, full_medicines_list)
    json_dict.update(entry)
    
    if i % batch_size == 0:
        save_json(json_dict)
        elapsed_time = time.time() - start_time
        print(f"Обработано {i} файлов за {elapsed_time:.2f} секунд")
        start_time = time.time()
        json_dict.clear()

# Сохраняем оставшиеся результаты в файл
if json_dict:
    save_json(json_dict)
