import json

# Загрузка данных из первого JSON файла
with open('MetaData.json', 'r', encoding='utf-8') as file:
    data1 = json.load(file)

# Загрузка данных из второго JSON файла
with open('output.json', 'r', encoding='utf-8') as file:
    data2 = json.load(file)

# Объединение данных по ключу
combined_data = {}

# Преобразуем ключи первого файла для сопоставления
for key in data1:
    # Извлекаем числовую часть ключа до и после знака "_"
    parts = key.split('_')
    num_key = parts[0]
    suffix_key = parts[1] if len(parts) > 1 else ''
    kr_key = f"КР{num_key}"
    
    # Создаем новый словарь для объединенных данных
    combined_entry = data1[key].copy()
    combined_entry["Ссылка на клиническую рекомендацию"] = f"https://cr.minzdrav.gov.ru/view-cr/{num_key}_{suffix_key}"
    
    if kr_key in data2:
        combined_entry.update(data2[kr_key])
    
    combined_data[kr_key] = combined_entry

# Добавляем оставшиеся данные из второго файла, которые не были объединены
for key in data2:
    if key not in combined_data:
        num_key = key[2:]  # Убираем "КР" из начала
        combined_data[key] = data2[key].copy()
        combined_data[key]["Ссылка на клиническую рекомендацию"] = f"https://cr.minzdrav.gov.ru/view-cr/{num_key}"

# Сохранение объединенного JSON файла
with open('combined_file.json', 'w', encoding='utf-8') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=4)

print("Объединение завершено. Результат сохранен в 'combined_file.json'.")
