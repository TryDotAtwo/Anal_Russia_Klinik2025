## Операционный лог изменений

Фиксируйте здесь существенные изменения в архитектуре, конфиге и поведении пайплайнов.

### Записи

- **Date:** 2026-05-04
  **Change:** Added `reports/llm/run_openrouter_all.py` for batch LLM evaluation outside the fixed gold harness.
  **Details:** New runner uses `llm_review_cases.json` plus `excluded_preparations.json`, loads existing `openrouter*_results.json` predictions to avoid duplicate paid calls, and writes merged results to `reports/llm/openrouter_all_results.json`. Selection order: first blocks that have a manual gold label but no LLM prediction; then remaining visible pending blocks. `--limit` counts new LLM calls, not existing predictions.
  **Goal:** Allow `py run_openrouter_all.py --limit 100` to evaluate 100 additional visible blocks while prioritizing gold-labelled blocks missing LLM output.
  **Verification:** `py -m pytest -q` => `31 passed`; `py reports\llm\run_openrouter_all.py --help` succeeded.

- **Date:** 2026-05-04
  **Change:** Added a unified `excluded_preparations.json` filter at the LLM boundary.
  **Details:** `llm_review_cases.json` remains raw. OpenRouter runner and LLM audit dashboard now load `reports/llm/excluded_preparations.json`, remove cases whose `primary_terms` are all excluded, skip fully empty blocks before any LLM request, and build request payloads from filtered blocks. Filtered payloads clean `case_ids`, `case_count`, `primary_terms`, `cases`, `context.case_spans`, `context.highlight_spans`, and `llm_payload.found_terms`; `context.text` stays broad. Result/state counters now expose `completed_all`, `completed_visible`, `excluded_cases`, and `excluded_blocks`.
  **Goal:** Prevent excluded preparations from reaching `payload.case_spans` and `payload.found_terms` while preserving source review cases and manual audit trail files.
  **Verification:** `py -m pytest -q` => `30 passed`.

- **Date:** 2026-05-01
  **Change:** Converted existing OpenRouter singleton block predictions to case-level nested predictions and added runner safeguards.
  **Details:** `reports/llm/openrouter_gold40_results.json` and `reports/llm/openrouter_gold40_v3_new_results.json` now store `predictions[block_id].predictions[case_id]`; old singleton labels were copied to every `case_id` from each corresponding block. Backups were written with `.before_case_prediction_migration.json`. OpenRouter runner now scales `max_tokens` by case count, expands singleton model responses only as fallback, and keeps already evaluated block IDs during resume even if stored prompt version differs. Dashboard case list now sorts blocks with LLM predictions before pending blocks.
  **Goal:** Preserve 80 paid model answers, make dashboard per-drug compatible, avoid repeated paid calls, and keep output budget sufficient for multi-drug prompts.
  **Verification:** `py -m pytest -q` => `28 passed`; dashboard API `http://127.0.0.1:8788/api/cases?limit=81` returned 80 predicted items first and state `completed=80/6114`.

- **Date:** 2026-05-01
  **Change:** Reworked LLM audit dashboard and separated LLM report artifacts from the Aho-Corasick report folder.
  **Details:** `reports/llm` now contains LLM review cases, gold samples, OpenRouter outputs, LLM dashboard logs, and LLM wrapper scripts; `reports/aho` remains scoped to Aho reports, filters, compact outputs, location grouping, and Aho dashboard files. Dashboard backend now lists every `llm_blocks` entry from `llm_review_cases.json`, supports case-level autosave, and reads nested OpenRouter prediction formats. Dashboard frontend now uses three resizable panes: left clinical recommendation/block list, center original context text with block and exact-word highlighting, right button-based manual review panel.
  **Goal:** Keep folders and scripts semantically separated; make manual gold-dataset creation comfortable and loss-resistant.
  **Verification:** `py -m pytest tests\test_llm_audit_dashboard.py -q`; full verification status is recorded in the current task final response.

- **Дата:** 2026‑03‑10  
  **Изменение:** добавлена структура `docs` (`README.md`, `architecture.md`, `agent-memory.md`, `hypotheses.md`, `operations-log.md`, `roadmap.md`).  
  **Цель:** сделать документацию долговременной памятью агента.  
  **Комментарий:** поведение кода не менялось, добавлена только документация.

- **Дата:** 2026‑03‑10  
  **Изменение:** Разделение на два пайплайна извлечения упоминаний и чистка корня.  
  **Детали:**
  - **MEDIQ-пайплайн** (`Logic/mediq_match.py`): строгое совпадение по границам слов + склонения; одно- и многословные названия; «Онко» не матчится в «Онкология».
  - **Пайплайн расстрельный список + слова-маркеры** (`Logic/blacklist_markers.py`): подстроковый поиск (слово/словоформа может быть частью слова); опциональный файл слов-маркеров в конфиге (`word_markers_path`).
  - **Дедупликация** (`Logic/dedupe.py`): по позиции (start, end); один препарат в двух списках или в маркерах даёт одно упоминание.
  - Конфиг перенесён в `core/config.py`; корневой `config.py` удалён.
  - В корне проекта остаётся только `Main.py` и пакеты (`core/`, `io/`, `cli/`, `Logic/`); добавлена папка `data/` для данных при желании.
  **Цель:** жёсткий пайплайн для MEDIQ, мягкий для расстрельного списка и маркеров; явная дедупликация; чистая структура и документация.

- **Дата:** 2026-04-29
  **Изменение:** Добавлен пакет `src/bot_for_anal_rec` с stage-based pipeline, max-recall raw matching, manual filtering, LLM providers, legacy export, CSV export, static dashboard.
  **Детали:** `Main.py` стал wrapper; `cli/main_cli.py` исправлен через новый package CLI; `core/config.py` больше не содержит абсолютные пути; добавлен sample fixture в `data/samples`; добавлены pytest tests.
  **Цель:** Максимально полный поиск спорных препаратов/маркеров без ранней отсечки; ручные фильтры являются обязательным слоем контроля ложных совпадений.
  **Проверка:** `py -m pytest -q`; sample smoke через `provider=fake`.

