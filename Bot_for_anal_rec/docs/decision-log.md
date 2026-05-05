## Журнал архитектурных и продуктовых решений

Фиксируйте здесь **важные решения**, которые влияют на архитектуру, данные или UX работы с проектом.

### Шаблон записи

- **Дата:** YYYY‑MM‑DD  
  **Контекст:** что побудило к решению (задача, ограничение, проблема).  
  **Варианты:** какие варианты рассматривались.  
  **Решение:** какой вариант выбран.  
  **Аргументы:** почему выбран именно он.  
  **Затронутые части:** какие файлы/модули/данные затронуты.  
  **Последствия:** ожидаемые эффекты и риски.

### Примеры записей

- **Date:** 2026-05-04
  **Context:** Excluded preparations must not be removed from raw `llm_review_cases.json`, but excluded-only cases must not be sent to the LLM as evaluable cases.
  **Options:** Rewrite `llm_review_cases.json` after exclusion; filter only dashboard visibility; add one shared runtime filter before prompt construction and reuse the same filter for dashboard visibility/request preview.
  **Decision:** Keep `llm_review_cases.json` raw and apply a shared `excluded_preparations.json` filter at the LLM boundary in OpenRouter runner and dashboard request preview.
  **Arguments:** Raw review cases remain reproducible audit input; manual `llm_gold_40.json` remains an audit trail; one shared filter prevents drift between UI and runner; filtered payloads cannot contain excluded-only cases in `case_spans` or `found_terms`.
  **Affected parts:** `src/anal_russia_klinik/llm_review_openrouter.py`, `src/anal_russia_klinik/llm_audit_dashboard.py`, `reports/llm/run_openrouter_gold40.py`, `tests/test_refactored_pipeline.py`, `tests/test_llm_audit_dashboard.py`, `docs/agent-memory.md`.
  **Consequences:** Existing predictions for newly excluded blocks remain in result files and count in `completed_all`, while dashboard and current LLM runs use `completed_visible` and skip fully excluded blocks.

- **Date:** 2026-05-01
  **Context:** Existing paid OpenRouter results used the old singleton block-level response shape, but the dashboard now expects per-case/per-drug predictions inside multi-case LLM blocks. Re-running 80 already evaluated blocks would waste API budget.
  **Options:** Re-run all blocks with the new prompt; keep singleton results and add dashboard fallback only; migrate singleton results to per-case nested predictions and keep resume skip behavior.
  **Decision:** Migrate existing singleton results to `predictions[case_id]` by duplicating the old block answer to every case in the block; keep resume from existing block IDs even across prompt-version mismatch; use current multi-case prompt for only new unevaluated calls.
  **Arguments:** Migration preserves paid model output, makes dashboard behavior uniform for old and new results, and avoids repeated token spend. Dynamic output token budgeting reduces truncation risk for blocks containing many cases.
  **Affected parts:** `src/anal_russia_klinik/llm_review_openrouter.py`, `src/anal_russia_klinik/llm_audit_dashboard.py`, `reports/llm/openrouter_gold40_results.json`, `reports/llm/openrouter_gold40_v3_new_results.json`, `tests/test_refactored_pipeline.py`, `tests/test_llm_audit_dashboard.py`.
  **Consequences:** Mixed historical and future result files can contain predictions from different prompt versions, but block IDs already present in output are not re-billed. Future quality comparisons should read per-case rows and prompt metadata.

- **Date:** 2026-05-01
  **Context:** LLM review/gold/dashboard artifacts were stored inside `reports/aho`, which mixed Aho-Corasick matching outputs with LLM annotation outputs and made folder ownership unclear.
  **Options:** Keep all derived outputs in `reports/aho`; create a separate `reports/llm` folder for LLM review and model-evaluation outputs.
  **Decision:** Use `reports/aho` only for Aho-Corasick reports/filtering/location grouping and use `reports/llm` for LLM review cases, manual gold labels, model predictions, dashboard wrappers, and LLM dashboard logs.
  **Arguments:** Folder names now encode pipeline stage and ownership; Aho filtering can evolve without mixing model-review files; manual gold creation works from a dedicated LLM workspace.
  **Affected parts:** `reports/llm/*`, `reports/aho/*`, `src/anal_russia_klinik/llm_audit_dashboard.py`, `reports/llm/build_llm_review_cases.py`, `docs/llm-review-cases-schema.md`.
  **Consequences:** Future commands should use `py reports\llm\build_llm_review_cases.py`, `py reports\llm\run_openrouter_gold40.py`, and `py reports\llm\llm_audit_dashboard.py`.

- **Дата:** 2026‑03‑11  
  **Контекст:** требуется хранить долгосрочную память и правила работы для агентов.  
  **Варианты:** 
  - держать всё в одном большом `README`,
  - разбить на несколько специализированных файлов в `docs/`.  
  **Решение:** создать специализированные файлы (`agents-guide.md`, `project-purpose.md`, `coding-standards.md`, `pipelines.md`, `data-notes.md`, `interaction-log.md`, `decision-log.md`, `troubleshooting.md`) и использовать `README.md` как карту документации.  
  **Аргументы:** проще ориентироваться и обновлять конкретные аспекты; облегчает работу агентам.  
  **Затронутые части:** `docs/*`.  
  **Последствия:** небольшое усложнение структуры, но существенный выигрыш в читаемости и поддержке.

- **Дата:** 2026-04-29
  **Контекст:** Требуется глубокий рефакторинг текущей версии с идеями из reference-проекта, но без потери legacy result shape.
  **Варианты:** ранняя защита word-boundary; max-recall raw matching с ручной фильтрацией.
  **Решение:** Выбран max-recall raw matching без ранней отсечки; совпадения вроде `Онко` внутри `Онкология` сохраняются в raw layer и отсекаются manual filters.
  **Аргументы:** Главная цель проекта — максимальная полнота поиска мракобесия; ложноположительные совпадения дешевле разбирать вручную, чем терять raw evidence.
  **Затронутые части:** `src/bot_for_anal_rec`, `Main.py`, `cli/main_cli.py`, `core/config.py`, `tests`, `data/samples`.
  **Последствия:** Manual filter dictionary становится обязательной частью рабочего процесса; dashboard нужен для быстрой валидации raw/filtered/LLM evidence.

- **Дата:** 2026-04-30
  **Контекст:** MEDIQ active substances and drug codes produce too many normal clinical matches and make host-word filter review inefficient.
  **Варианты:** keep substance dashboard grouping; remove active substances and codes from search surface.
  **Решение:** MEDIQ search surface is name-only for active-substance/code fields: exclude `drug.atx`, `drug.atxGroup`, `drug.mnn.name`, `drug.mnn.complex[*].name`; remove dashboard substance grouping.
  **Аргументы:** User review target is suspect preparation mentions; active substances and codes add high-volume clinically normal evidence with low filter-building value.
  **Затронутые части:** `src/anal_russia_klinik/dictionaries.py`, `src/anal_russia_klinik/aho_compact.py`, `src/anal_russia_klinik/aho_filter_server.py`, dashboard HTML, compact Aho reports.
  **Последствия:** Future Aho reports avoid active-substance/code variants; old full raw report remains archival; dashboard works from cleaned compact/filtered reports.

- **Дата:** 2026-04-30
  **Контекст:** LLM должна оценивать одно уникальное расположение один раз, но dashboard должен показывать все совпавшие термины и давать человеку инструмент для проверки качества.
  **Варианты:** плоский список cases; группировка по клинрекам с cases внутри каждого документа.
  **Решение:** Выбрана структура `clinical_recommendations[*].cases[*]`; case key=`document_id + char_start + char_end`; context window=2500 chars each side; offsets for highlighting are relative to `context.text`.
  **Аргументы:** Dashboard клинреков удобнее строить документ-центрично; LLM получает достаточно локального контекста; дубли term evidence не теряются и не создают повторные LLM calls.
  **Затронутые части:** `src/anal_russia_klinik/llm_review_cases.py`, `reports/aho/build_llm_review_cases.py`, `docs/llm-review-cases-schema.md`, `reports/aho/llm_review_cases.json`.
  **Последствия:** Future LLM/dashboard layer must use `clinical_recommendations[*].cases[*]` as review unit and store model/manual labels in the provided placeholders.


- **Date:** 2026-04-30
  **Context:** LLM calls should not repeat adjacent mentions from the same local paragraph/list/table block, but should not merge near-but-different offsets through fuzzy location dedup.
  **Options:** fuzzy location dedup with offset tolerance; strict location dedup plus nearby block grouping.
  **Decision:** Keep strict dedup by exact `document_id + char_start + char_end`; add `clinical_recommendations[*].llm_blocks[*]` as the LLM review unit; merge only sorted cases with gap `<=100` chars.
  **Arguments:** Exact dedup preserves distinct drug names and small offset differences; nearby block grouping reduces repeated LLM calls while keeping all case IDs and term evidence inside one block.
  **Affected parts:** `src/anal_russia_klinik/llm_review_cases.py`, `src/anal_russia_klinik/llm_review_openrouter.py`, `reports/aho/build_llm_review_cases.py`, `reports/aho/run_openrouter_gold40.py`, `reports/aho/llm_review_cases.json`, `reports/aho/llm_gold_40.json`.
  **Consequences:** Future LLM/dashboard layer must use `llm_blocks` for model calls; `cases` remain available as audit/detail records; OpenRouter gold40 runner is the first tuning harness.
