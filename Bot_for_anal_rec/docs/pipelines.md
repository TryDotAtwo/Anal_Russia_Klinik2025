## Пайплайны обработки

Этот документ описывает основные пайплайны проекта. При изменении последовательности шагов **обновляйте этот файл**.

### 1. Подготовка данных

1. Загрузка списка препаратов из `filtered_names_drugs_NEW.json`.
2. Чтение конфигурации из `core/config.py`.
3. Загрузка чёрного списка (`blacklist_json_path`) и построение:
   - словаря препаратов `drug_dict`,
   - автомата Ахо–Корасика для словоформ (`build_aho_automaton_word_forms`).
4. Загрузка дополнительной информации о препаратах (`additional_drug_info_path`) и построение `additional_drug_dict` по имени и ATX‑коду.

### 2. Обработка батча препаратов

Для каждой порции препаратов (батч размером `BATCH_SIZE`):

1. Формирование `search_names` с учётом `additional_drug_dict` (имя/ATX → каноническое имя).
2. Один проход по чёрному списку: `check_blacklist_drugs(search_names, drug_dict, automaton)`.
3. Загрузка клинических рекомендаций через `ClinicalAnalyzer`:
   - загрузка/проверка `clinical_recommendations.json`,
   - при необходимости — синхронная обработка PDF.

### 3. Извлечение упоминаний (пайплайн 1)

Для каждой клинреки:

- **MEDIQ‑пайплайн** (`Logic/mediq_match.py`):
  - токенизация текста,
  - построение словоформ для каждого препарата из `search_names`,
  - поиск по границам слов (строгое совпадение, в т.ч. для многословных названий),
  - формирование контекстов с полями `start`, `end`, `drug`, `source="mediq"`, `context`.

### 4. Извлечение упоминаний (пайплайн 2)

- **Расстрельный список + маркеры** (`Logic/blacklist_markers.py`):
  - загрузка слов‑маркеров (`word_markers_path`),
  - построение автомата по всем словоформам чёрного списка и маркеров,
  - подстроковый поиск (вхождение внутри слова допускается),
  - формирование контекстов с `source="blacklist"` или `source="marker"`.

### 5. Дедупликация и агрегация

1. Объединение результатов MEDIQ и маркеров.
2. Дедупликация по позиции (`Logic/dedupe.py`).
3. Фильтрация по батчу (`search_names` и исходные препараты `preparations`).
4. Формирование структуры:
   - `drugs_mentioned` с полями:
     - `drug`,
     - `in_blacklist` / `blacklist_description`,
     - `additional_info`,
     - `mentions` (список контекстов).
   - `summary` по каждой клинреке.

### 6. LLM‑анализ

1. Подсчёт общего количества упоминаний.
2. Формирование задач LLM‑анализа по каждой паре (препарат, контекст):
   - `Logic/Levels_extract.py: analyze_mention`.
3. Обработка батчами с отслеживанием прогресса:
   - количество обработанных упоминаний,
   - оценка оставшегося времени.
4. Запись результатов анализа обратно в `mentions[*].analysis` и обновление сводных метрик.

### 7. Выгрузка результатов

- Для каждого батча:
  - сохранение в `Match_Clinick_batch_{batch_number}.json`,
  - логирование времени и количества упоминаний.

### 8. Правила обновления

- При добавлении нового шага пайплайна:
  - описать его место и вход/выход здесь,
  - при изменении формата данных — дополнительно обновить `data-notes.md`.

## 2026-04-29 pipeline v0.2

1. `ingest`: load `clinical_recommendations.json`, `.txt`, or `.pdf` into `Document`.
2. `match`: load preparations, blacklist, markers; run max-recall Aho-Corasick substring matching; write `raw_matches.json`.
3. `filter`: apply `manual_filters.csv` or JSON after raw matching; write `filtered_matches.json` and `rejected_matches.json`.
4. `classify`: default provider `g4f`; optional provider `openrouter`; test provider `fake`; write `classified_matches.json`.
5. `export`: write `classified_matches.csv`, legacy `Match_Clinick_results.json`, and static dashboard.

Invariant: no early boundary filtering. Example `Онко` inside `Онкология` remains a raw match and can be rejected by manual filter.

