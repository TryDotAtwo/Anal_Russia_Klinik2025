## Данные и пути

Этот файл описывает **источники данных**, форматы файлов и важные пути. Любые изменения в данных нужно отражать здесь.

### 1. Основные файлы

- **Список препаратов для анализа**
  - Файл: `filtered_names_drugs_NEW.json`
  - Тип: JSON‑массив строк.
  - Использование: загружается в `Main.py` как `preparations`.

- **Чёрный список препаратов**
  - Ключ в конфиге: `blacklist_json_path`
  - По умолчанию: `blacklist_drugs.json`
  - Формат: список объектов с информацией о препаратах/группах (детали см. в коде `Logic/Black_Extract.py`).

- **Клинические рекомендации (предобработанный JSON)**
  - Ключ в конфиге: `clinical_recommendations_json`
  - По умолчанию: `clinical_recommendations.json`
  - Формат: объект с `pdf_hash` и словарём `recommendations` (см. `Logic/Clinick_Extract.py`).

- **Дополнительная информация о препаратах**
  - Ключ в конфиге: `additional_drug_info_path`
  - По умолчанию: `drugs.json`
  - Назначение: маппинг имени/ATX на расширенную информацию (MNN, описание, критерии и т.п.).

- **Слова‑маркеры**
  - Ключ в конфиге: `word_markers_path`
  - По умолчанию: `Drugs_Result_Open.json`
  - Формат: JSON‑массив строк (слова/фразы).

### 2. Пути к PDF и метаданным

- **Метаданные клинреков**
  - Ключ: `metadata_path`
  - Путь задан в `core/config.py` (см. текущие абсолютные пути).
  - Назначение: соответствие между номерами/ключами и описанием/названием клинреков.

- **Папка с PDF**
  - Ключ: `pdf_folder`
  - Путь задан в `core/config.py`.
  - Требование: файлы должны быть доступны для чтения; изменения состава PDF → пересчёт `pdf_hash` и переобработка.

### 3. Выходные файлы

- **Результаты матчей по батчам**
  - Шаблон: `Match_Clinick_batch_{N}.json` в корне проекта.
  - Содержимое: объект с ключом `clinical_recommendations` и списком результатов по клинрекам.

### 4. Правила изменения данных

- Любое изменение:
  - формата JSON,
  - расположения файлов,
  - стратегии разбиения на батчи
  
должно быть:

1. отражено в `core/config.py` (если меняются пути/ключи),
2. описано текстом в этом файле (`data-notes.md`),
3. при необходимости — задокументировано как решение в `decision-log.md`.

## 2026-04-29 data paths v0.2

- Canonical runtime config: `bot_for_anal_rec.config.AnalysisConfig`.
- Default input paths: `Drugs_Result_Open.json`, `blacklist_drugs.json`, `AXTUNG.Json`, `clinical_recommendations.json`, `data/manual_filters.csv`.
- Environment overrides: `BOT_ANAL_PREPARATIONS_PATH`, `BOT_ANAL_BLACKLIST_PATH`, `BOT_ANAL_MARKERS_PATH`, `BOT_ANAL_CLINICAL_JSON`, `BOT_ANAL_FILTERS_PATH`, `BOT_ANAL_OUTPUT_DIR`.
- Sample fixture paths: `data/samples/clinical.json`, `data/samples/markers.json`, `data/samples/blacklist.json`, `data/samples/preparations.json`, `data/samples/manual_filters.csv`.
- Generated run outputs: `runs/<run_id>/raw_matches.json`, `filtered_matches.json`, `rejected_matches.json`, `classified_matches.json`, `classified_matches.csv`, `Match_Clinick_results.json`, `dashboard/index.html`.

