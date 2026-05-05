## Журнал взаимодействия с пользователем

Этот файл служит для **краткой фиксации ключевых запросов пользователя и ответных действий агента**.

### Шаблон записи

- **Дата:** YYYY‑MM‑DD  
  **Пользовательский запрос:** краткое содержание задачи/вопроса.  
  **Действия агента:** что было сделано (основные шаги).  
  **Результат/статус:** готово / частично / требуется уточнение.  
  **Связанные файлы:** какие файлы/модули/документы изменялись.  
  **Комментарии:** опционально, важные детали/ограничения.

### Пример записи

- **Date:** 2026-05-04
  **User request:** Add one unified `excluded_preparations.json` blacklist before every LLM run; keep `llm_review_cases.json` raw; clean filtered payload fields; split completed counters into all/visible; keep manual labels and audit trail data.
  **Agent actions:** Added shared block/case filtering in `llm_review_openrouter.py`; wired the filter into `run_gold_openrouter`; wired dashboard listing and request preview to filtered blocks; added `completed_all`, `completed_visible`, `excluded_cases`, and `excluded_blocks`; added CLI argument `--excluded-preparations`; added regression tests for payload filtering, runner skip behavior, and dashboard counters; preserved metric-only semantics of `exclude_from_model_stats`.
  **Result/status:** Implemented and verified.
  **Related files:** `src/anal_russia_klinik/llm_review_openrouter.py`, `src/anal_russia_klinik/llm_audit_dashboard.py`, `src/anal_russia_klinik/llm_review_cases.py`, `reports/llm/run_openrouter_gold40.py`, `tests/test_refactored_pipeline.py`, `tests/test_llm_audit_dashboard.py`, `docs/operations-log.md`, `docs/decision-log.md`, `docs/agent-memory.md`.
  **Comments:** Verification command `py -m pytest -q` passed with 30 tests.

- **Date:** 2026-05-01
  **User request:** Convert the existing 80 OpenRouter answers so each old singleton answer is duplicated to every drug/case in the block; ensure future multi-drug prompts have enough output tokens; ensure already evaluated blocks are not re-evaluated; show LLM-evaluated clinical recommendations first in the dashboard.
  **Agent actions:** Migrated both OpenRouter result JSON files to nested `predictions[case_id]`; added dynamic `max_tokens` based on case count; kept existing predictions during resume despite prompt-version mismatch; added dashboard ordering by prediction presence; added regression tests for migration, token budget, resume skip, and dashboard ordering.
  **Result/status:** Implemented; migrated files contain 80 nested blocks and 117 case predictions; full test suite passed.
  **Related files:** `src/anal_russia_klinik/llm_review_openrouter.py`, `src/anal_russia_klinik/llm_audit_dashboard.py`, `reports/llm/openrouter_gold40_results.json`, `reports/llm/openrouter_gold40_v3_new_results.json`, `tests/test_refactored_pipeline.py`, `tests/test_llm_audit_dashboard.py`.

- **Date:** 2026-05-01
  **User request:** Improve LLM audit dashboard for human annotators: show all LLM-marked clinical recommendations, highlight exact words and LLM context block, use right-side button controls with autosave, add resizable/hideable panes and open-clinrec button; move LLM artifacts out of `reports/aho` into a logical LLM folder.
  **Agent actions:** Updated dashboard backend to list all `llm_blocks`; added case-level gold autosave and nested prediction support; rebuilt dashboard frontend as left list, center highlighted text, right review panel; moved LLM scripts/data/logs from `reports/aho` to `reports/llm`; renamed dashboard wrapper to `reports/llm/llm_audit_dashboard.py`.
  **Result/status:** Implemented; verification recorded in current task final response.
  **Related files:** `src/anal_russia_klinik/llm_audit_dashboard.py`, `src/anal_russia_klinik/llm_audit_dashboard_html.py`, `reports/llm/*`, `docs/llm-review-cases-schema.md`, `tests/test_llm_audit_dashboard.py`.

- **Дата:** 2026‑03‑11  
  **Пользовательский запрос:** организовать долговременную документацию в папке `docs` и добавить отображение прогресса по клинрекам.  
  **Действия агента:** 
  - добавлены и заполнены новые документы в `docs/` (гайд для агентов, цели проекта, стандарты кода, пайплайны, данные, логи и т.д.);
  - доработан `Main.py` для вывода количества обработанных клинреков.  
  **Результат/статус:** реализовано, файлы готовы для дальнейшего пополнения.  
  **Связанные файлы:** `Main.py`, `docs/agents-guide.md`, `docs/project-purpose.md`, `docs/coding-standards.md`, `docs/pipelines.md`, `docs/data-notes.md`, `docs/decision-log.md`, `docs/interaction-log.md`, `docs/troubleshooting.md`.  
  **Комментарии:** при дальнейшем развитии проекта сюда добавлять новые сессии и задачи.

- **Дата:** 2026-04-29
  **Пользовательский запрос:** Реализовать утвержденный план глубокого рефакторинга: текущий проект как база, reference-проект как источник идей, max-recall matching, manual filters, g4f default, OpenRouter optional, dashboard.
  **Действия агента:** Создан `src/bot_for_anal_rec`; добавлены matcher/filter/LLM/export/dashboard/CLI; сохранен legacy JSON contract; исправлен конфликт `io.files`; добавлены tests и sample fixture; обновлена долговременная память.
  **Результат/статус:** Реализация выполнена; pytest и sample smoke выполняются через `provider=fake`; network smoke для `g4f` требует доступ к провайдерам g4f.
  **Связанные файлы:** `src/bot_for_anal_rec/*`, `Main.py`, `cli/main_cli.py`, `core/config.py`, `tests/test_refactored_pipeline.py`, `data/samples/*`.

- **Дата:** 2026-04-30
  **Пользовательский запрос:** Удалить поиск по действующим веществам и кодам препаратов, потому что обзор MEDIQ по веществам/кодам создает слишком много нормальных совпадений и усложняет ручную фильтрацию.
  **Действия агента:** Удалены MEDIQ MNN/ATX/ATX-group варианты из загрузчика словаря; существующий compact Aho очищен до MEDIQ name-only rows; case duplicates deduplicated by normalized search word; dashboard переведен на отображение только препаратов; substance grouping модуль удален.
  **Результат/статус:** Реализация выполнена; будущий Aho не ищет активные вещества и drug codes; старый full raw report сохранен без перезаписи; working compact rows=1002; filtered JSON rebuilt.
  **Связанные файлы:** `src/anal_russia_klinik/dictionaries.py`, `src/anal_russia_klinik/aho_compact.py`, `src/anal_russia_klinik/aho_filter_server.py`, `src/anal_russia_klinik/aho_filter_dashboard_html.py`, `src/anal_russia_klinik/aho_dashboard_rows.py`, `tests/test_refactored_pipeline.py`.

- **Дата:** 2026-04-30
  **Пользовательский запрос:** Сформировать подробный отфильтрованный аналог `host_words_by_search_word.json` после ручной сборки host_word filters.
  **Действия агента:** Создан streaming filter builder для полного detailed report; применены host_word filters к `host_words` и `inside_host_words`; сохранены occurrences/document_ids; пересобран compact filtered JSON.
  **Результат/статус:** Готово; output=`reports/aho/host_words_by_search_word_filtred.json`; rows=729; matches=9535; bad_filtered_hosts=0; tests=11 passed.
  **Связанные файлы:** `src/anal_russia_klinik/aho_detailed_filter.py`, `reports/aho/filter_detailed_host_words.py`, `reports/aho/host_words_by_search_word_filtred.json`, `reports/aho/host_words_filtered_compact.json`, `tests/test_refactored_pipeline.py`.

- **Дата:** 2026-04-30
  **Пользовательский запрос:** Сгруппировать дубли по `document_id + char_start + char_end`, не удаляя информацию, чтобы LLM оценивала одну позицию один раз.
  **Действия агента:** Создан location-grouping report; каждое уникальное расположение стало отдельной LLM review unit; все дублирующиеся term/host evidence сохранены в `matches`.
  **Результат/статус:** Готово; output=`reports/aho/host_words_by_location_filtred.json`; location_groups=8753; duplicate_matches_grouped=782; tests=12 passed.
  **Связанные файлы:** `src/anal_russia_klinik/aho_location_groups.py`, `reports/aho/group_filtered_locations.py`, `reports/aho/host_words_by_location_filtred.json`, `tests/test_refactored_pipeline.py`.

- **Дата:** 2026-04-30
  **Пользовательский запрос:** Зафиксировать структуру LLM review cases по клинрекам: контекст 2500 символов в обе стороны, span offsets для dashboard, prompt context, placeholders для LLM/manual validation, метрики FP/FN/precision/recall/F1 по label/evidence levels.
  **Действия агента:** Создан review case builder; собран `llm_review_cases.json`; добавлены prompt contract, dashboard validation schema, per-case `llm_result`/`human_validation`, evidence-level candidate spans.
  **Результат/статус:** Готово; clinical_recommendation_count=528; case_count=8753; evidence_level_candidate_count=24851; output_size=204156539 bytes; tests=13 passed.
  **Связанные файлы:** `src/anal_russia_klinik/llm_review_cases.py`, `reports/aho/build_llm_review_cases.py`, `reports/aho/llm_review_cases.json`, `docs/llm-review-cases-schema.md`, `tests/test_refactored_pipeline.py`.


- **Date:** 2026-04-30
  **User request:** Group nearby matches before LLM calls, keep strict location dedup without +/-4 char tolerance, inspect 40 cases, create a gold test set, and prepare OpenRouter execution.
  **Agent actions:** Added `llm_blocks` with gap<=100 chars; refined prompt contract for block-level classification and evidence-level scope; added OpenRouter runner; created `reports/aho/llm_gold_40.json` from 40 manually labelled blocks; rebuilt `reports/aho/llm_review_cases.json`.
  **Result/status:** Ready except live OpenRouter run; current env lacks `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`; tests passed.
  **Related files:** `src/anal_russia_klinik/llm_review_cases.py`, `src/anal_russia_klinik/llm_review_openrouter.py`, `reports/aho/build_llm_review_cases.py`, `reports/aho/run_openrouter_gold40.py`, `reports/aho/llm_review_cases.json`, `reports/aho/llm_gold_40.json`, `docs/llm-review-cases-schema.md`.

- **Date:** 2026-04-30
  **User request:** Create env for OpenRouter token, run 40 manually labelled cases on OpenAI GPT-5.5 model through OpenRouter, then evaluate quality.
  **Agent actions:** Created ignored `config/openrouter.env`; set default model `openai/gpt-5.5`; added env-file loader to OpenRouter runner; added incremental result writes and richer quality metrics.
  **Result/status:** Env ready for user token; live run pending token insertion; tests passed.
  **Related files:** `config/openrouter.env`, `.gitignore`, `src/anal_russia_klinik/llm_review_openrouter.py`, `reports/aho/run_openrouter_gold40.py`.

- **Date:** 2026-04-30
  **User request:** Run the OpenRouter GPT-5.5 test on the 40 manually labelled cases and inspect model quality.
  **Agent actions:** Ran OpenRouter; fixed max-token reservation error; fixed JSON response-format provider error; added checkpoint resume; wrote partial quality analysis.
  **Result/status:** Partial result only: 20/40 completed; run stopped on OpenRouter 403 key total limit exceeded; partial label_accuracy=0.70; evidence_level_accuracy=0.80.
  **Related files:** `src/anal_russia_klinik/llm_review_openrouter.py`, `reports/aho/run_openrouter_gold40.py`, `reports/aho/openrouter_gold40_results.json`, `reports/aho/openrouter_gold40_partial_analysis.json`.

- **Date:** 2026-04-30
  **User request:** Explain why LLM quality was worse than manual labels, improve prompt, and explain why OpenRouter run failed.
  **Agent actions:** Reviewed partial mismatches; changed prompt to `llm-review-v2`; added stricter inside-word/background/glossary/evidence-scope rules; added prompt-version guard to OpenRouter resume.
  **Result/status:** Prompt fixed and tests passed; live validation still blocked by OpenRouter key total limit.
  **Related files:** `src/anal_russia_klinik/llm_review_cases.py`, `src/anal_russia_klinik/llm_review_openrouter.py`, `docs/agent-memory.md`, `docs/operations-2026-04-29-docker-aho.md`.

- **Date:** 2026-04-30
  **User request:** Mark indomethacin substring case as strict `error`; add dashboard visibility for every LLM case, request payload, response, key usage, and per-case cost.
  **Agent actions:** Corrected gold label for `Метацин` inside `индометацин`; added OpenRouter metadata capture; added LLM audit dashboard on port 8788; added key reload fix; verified new key limit; accepted model correction to GPT-5.4 only.
  **Result/status:** Dashboard running at `http://127.0.0.1:8788`; new key status limit=30, remaining=30, usage=0; tests passed.
  **Related files:** `reports/aho/llm_gold_40.json`, `src/anal_russia_klinik/llm_review_openrouter.py`, `src/anal_russia_klinik/llm_audit_dashboard.py`, `src/anal_russia_klinik/llm_audit_dashboard_html.py`, `reports/aho/llm_openrouter_dashboard.py`, `tests/test_llm_audit_dashboard.py`.
