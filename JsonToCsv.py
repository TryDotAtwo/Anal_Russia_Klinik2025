import json
import csv

# Чтение данных из JSON файла
with open('combined_file.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Функция для записи данных в CSV
def write_csv(file_name, is_long_version=False):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Запись заголовков
        headers = [
            "ID", "Название клинической рекомендации", "МКБ-10", "Возрастная группа",
            "Разработчик", "Дата размещения", "Статус применения КР",
            "Ссылка на клиническую рекомендацию", "Слова из списка препаратов", "Слова из списка маркеров",
            "Слова из списка возможных маркеров"
        ]
        writer.writerow(headers)
        
        # Запись данных
        for id, details in data.items():
            # Для "Какие слова из списка препаратов были обнаружены"
            drugs = []
            for item in details.get("Какие слова из списка препаратов были обнаружены", []):
                if is_long_version:
                    drugs.append(f"{item[0]}: {item[1]}")  # long version (item[0] и item[1])
                else:
                    drugs.append(item[0])  # short version (только item[0])
            
            # Для "Какие слова из списка маркеров были обнаружены"
            markers = []
            for item in details.get("Какие слова из списка маркеров были обнаружены", []):
                if is_long_version:
                    markers.append(f"{item[0]}: {item[1]}")  # long version (item[0] и item[1])
                else:
                    markers.append(item[0])  # short version (только item[0])
            
            # Для "Какие слова из списка возможных маркеров были обнаружены"
            possible_markers = []
            for item in details.get("Какие слова из списка возможных маркеров были обнаружены", []):
                if is_long_version:
                    possible_markers.append(f"{item[0]}: {item[1]}")  # long version (item[0] и item[1])
                else:
                    possible_markers.append(item[0])  # short version (только item[0])

            # Запись строки в CSV
            writer.writerow([
                id,
                details.get("Название клинической рекомендации", ""),
                details.get("МКБ-10", ""),
                details.get("Возрастная группа", ""),
                details.get("Разработчик", ""),
                details.get("Дата размещения", ""),
                details.get("Статус применения КР", ""),
                details.get("Ссылка на клиническую рекомендацию", ""),
                '; '.join(drugs),
                '; '.join(markers),
                '; '.join(possible_markers)
            ])

# Запись short версии CSV
write_csv('clinical_recommendations_short.csv', is_long_version=False)

# Запись long версии CSV
write_csv('clinical_recommendations_long.csv', is_long_version=True)

print("Файлы CSV успешно созданы: 'clinical_recommendations_short.csv' и 'clinical_recommendations_long.csv'.")
