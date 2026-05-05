# Архитектура Bot_for_anal_rec

## Общая идея

Проект анализирует клинические рекомендации, находит упоминания препаратов (два пайплайна: MEDIQ и расстрельный список + слова-маркеры), сопоставляет с чёрным списком и дополнительной базой лекарств, выполняет LLM‑анализ контекстов.

## Корень проекта

В корне папки проекта (`Bot_for_anal_rec/`) находится **только точка входа** `Main.py` и пакеты: `core/`, `io/`, `cli/`, `Logic/`. Конфигурация вынесена в `core/config.py`. Данные по умолчанию лежат в корне проекта (пути в конфиге); при желании можно перенести файлы в `data/` и поправить пути в `core/config.py`.

## Два пайплайна извлечения упоминаний

### 1. Пайплайн MEDIQ (строгий)

- **Назначение:** точное совпадение препарата с учётом склонений, **только по границам слов**.
- **Правило:** «Онко» не должно находиться внутри слова «Онкология».
- **Реализация:** `Logic/mediq_match.py` — токенизация текста по словам, для каждого препарата (одно- или многословное название) генерируются словоформы через `Work_witch_word.generate_word_forms`; поиск только целых слов (и их склонений) в тексте. Многословные названия ищутся как последовательность подряд идущих слов.
- **Вход:** список поисковых названий (`search_names`), текст КР.  
- **Выход:** список `{start, end, drug, source: "mediq", context}`.

### 2. Пайплайн расстрельный список + слова-маркеры (мягкий)

- **Назначение:** поиск подстроки — слово или словоформа может быть частью другого слова.
- **Реализация:** `Logic/blacklist_markers.py` — Ахо–Корасик по словоформам чёрного списка и по списку слов-маркеров (файл задаётся в конфиге `word_markers_path`, опционально). По тексту идёт подстроковый поиск с сохранением позиции.
- **Вход:** чёрный список (JSON), слова-маркеры (JSON-массив строк или пусто), текст КР.  
- **Выход:** список `{start, end, drug, source: "blacklist"|"marker", context}`.

### Объединение и дедупликация

- Результаты обоих пайплайнов объединяются; в `Logic/dedupe.py` выполняется **дедупликация по позиции** `(start, end)`: один препарат в двух списках или в маркерах даёт одно упоминание.
- В отчёт по препаратам попадают только те упоминания, чей `drug` входит в список поисковых имён батча (`search_names`).

## Основные компоненты

- **Точка входа**
  - `Main.py` — загрузка препаратов, конфига, чёрного списка, доп. информации; по батчам: проверка чёрного списка, сборка автомата расстрельный+маркеры, загрузка КР, извлечение упоминаний (MEDIQ + blacklist_markers + dedupe), LLM‑анализ, сохранение `Match_Clinick_batch_*.json`.

- **Конфигурация**
  - `core/config.py` — `ANALYSIS_CONFIG` (пути к JSON, параметры PDF, `context_before`/`context_after`, опционально `word_markers_path`), `BATCH_SIZE`.

- **Доменная логика (`Logic/`)**
  - `Clinick_Extract.py` — загрузка и подготовка клинических рекомендаций.
  - `Levels_extract.py` — LLM‑анализ упоминаний.
  - `Black_Extract.py` — чёрный список: словарь, Ахо–Корасик по словоформам, проверка препарата и описание.
  - `Work_witch_word.py` — генерация словоформ.
  - `mediq_match.py` — пайплайн MEDIQ (по границам слов).
  - `blacklist_markers.py` — пайплайн расстрельный список + слова-маркеры (подстрока).
  - `dedupe.py` — дедупликация упоминаний по позиции.

- **Ядро**
  - `core/config.py`, `core/pipelines.py` — конфиг и вспомогательные функции пайплайнов.

- **Данные**
  - По умолчанию пути из конфига (файлы в корне проекта или в `data/` при переносе).
  - Ключевые файлы: `filtered_names_drugs_NEW.json`, `blacklist_drugs.json`, `drugs.json`, `clinical_recommendations.json`, опционально файл слов-маркеров, `Match_Clinick_batch_*.json`.

## Зависимости слоёв

- `Main.py` → `core.config`, `Logic.*`
- `cli/main_cli.py` → `core.config`, `Main`
- `Logic/blacklist_markers.py` → `Logic/Black_Extract`, `Logic/Work_witch_word`
- `Logic/mediq_match.py` → `Logic/Work_witch_word`

Обновляйте этот файл при появлении новых модулей и изменении пайплайнов.

## 2026-04-29 refactor v0.2

- `src/bot_for_anal_rec/` is the primary package; root `bot_for_anal_rec/` is a compatibility shim for `py -m bot_for_anal_rec` without editable install.
- `Main.py` and `cli/main_cli.py` are wrappers around `bot_for_anal_rec.cli`.
- `core/config.py` is compatibility-only; canonical config is `bot_for_anal_rec.config.AnalysisConfig`.
- Pipeline is stage-based: ingest/load documents -> raw match -> manual filter -> LLM classify -> export legacy JSON/CSV/dashboard.
- Matching policy is max-recall substring matching for all sources; false positives are removed by manual filters, not by early boundary filtering.
- Legacy output contract remains `{"clinical_recommendations": [...]}`; `mentions[*]` now includes positions, host word, source, filter decision, label, UDD, UUR.
- Static dashboard is generated to `runs/<run_id>/dashboard/index.html`.

## 2026-04-29 refactor v0.3

- Project name: `Anal_Russia_Klinik`.
- Canonical package: `src/anal_russia_klinik`.
- Root entrypoint: `Main.py`.
- Docker entrypoint: `python Main.py`.
- Full Aho report service: `docker compose run --rm aho`.
- Full Aho report output: `reports/aho/host_words_by_search_word.json`.
- Aho report shape: `by_search_word[] -> host_words[] -> occurrences[]`; `inside_host_words[]` contains only inside-word matches for manual filter drafting.
- Aho parallelism: 16 Linux container workers; worker initializer builds one automaton per worker; chunks write JSONL partial files before merge.
- Data layout: `data/input/` for local large source files; `data/samples/` for small tests.
- Runtime output layout: `reports/`.
- Legacy layout and artifacts: `old/`.
