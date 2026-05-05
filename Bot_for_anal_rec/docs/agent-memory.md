## Долговременная память агента

Этот файл фиксирует факты, которые важно **не забывать** между сессиями работы над проектом.

### Ключевые файлы и пути

- **Корень проекта:** только `Main.py` и пакеты `core/`, `io/`, `cli/`, `Logic/`. Конфиг — `core/config.py`.
- `Main.py` — точка входа, запуск через `asyncio.run(main())`.
- `Logic/Clinick_Extract.py` — загрузка/парсинг клинических рекомендаций.
- `Logic/Levels_extract.py` — LLM‑анализ контекстов (`analyze_mention`).
- `Logic/Black_Extract.py` — чёрный список и Ахо–Корасик по словоформам.
- `Logic/Work_witch_word.py` — генерация словоформ.
- `Logic/mediq_match.py` — **пайплайн MEDIQ:** строго по границам слов + склонения.
- `Logic/blacklist_markers.py` — **пайплайн расстрельный список + слова-маркеры:** подстрока.
- `Logic/dedupe.py` — дедупликация упоминаний по позиции (start, end).
- Важные JSON: `filtered_names_drugs_NEW.json`, `blacklist_drugs.json`, `drugs.json`, `clinical_recommendations.json`, опционально файл слов-маркеров (путь в `word_markers_path`), `Match_Clinick_batch_*.json`.

### Важные инварианты и правила

- **Два пайплайна извлечения:**
  - **MEDIQ:** только целые слова и их склонения; «Онко» не матчится в «Онкология»; поддержка многословных названий.
  - **Расстрельный список + слова-маркеры:** подстрока (слово/словоформа может быть частью слова). После объединения — дедупликация по позиции (start, end).
- **Не ломать формат JSON‑результатов:** структура `Match_Clinick_batch_*.json` должна сохраняться.
- **Конфиг:** все пути и параметры в `core/config.py`; опционально `word_markers_path` для файла слов-маркеров (JSON-массив строк).
- **Ахо–Корасик:** для чёрного списка (проверка препарата) — один автомат; для расстрельный+маркеры — отдельный автомат в `blacklist_markers`.
- **LLM‑анализ:** батчами, с отслеживанием прогресса; возможность продолжить после падения.

### Что важно помнить при доработках

- Не зашивать «жёсткие» пути в коде там, где можно использовать конфиг.
- Поддерживать раздельность уровней:
  - загрузка данных,
  - доменная логика анализа,
  - orchestration/пайплайны,
  - CLI/точки входа.
- При серьёзных изменениях архитектуры обязательно:
  - обновить `docs/architecture.md`;
  - сделать запись в `docs/operations-log.md`.

## 2026-04-29 memory update

- Current canonical package: `src/bot_for_anal_rec`.
- Current launch commands: `py Main.py`, `py -m bot_for_anal_rec run`, `py -m bot_for_anal_rec run --provider fake --texts data/samples/clinical.json --markers data/samples/markers.json --blacklist data/samples/blacklist.json --preparations data/samples/preparations.json --filter-file data/samples/manual_filters.csv --output-dir runs/smoke_sample`.
- Matching invariant: max-recall substring matching; manual filters handle false positives including inside-word cases.
- LLM invariant: provider default is `g4f`; `openrouter` requires `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`; tests use `fake`.
- Output invariant: legacy JSON shape remains `{"clinical_recommendations": [...]}`; mention records are extended with raw positions and audit fields.

## 2026-04-29 docker/aho update

- Current canonical package: `src/anal_russia_klinik`.
- Project name: `Anal_Russia_Klinik`; old `bot_for_anal_rec` package remains only as src compatibility shim.
- Root policy: keep only `Main.py`, `README.md`, `pyproject.toml`, Docker files, `.gitignore`, and top-level folders.
- Local large inputs: `data/input/AXTUNG.Json`, `data/input/blacklist_drugs.json`, `data/input/drugs.json`, `data/input/clinical_recommendations.json`, `data/input/MetaData.json`; `data/input/*.json` ignored by Git.
- Legacy/untracked old material: `old/`; ignored by Git.
- Reports: `reports/`; ignored by Git.
- Docker command for full Aho report: `docker compose run --rm aho`.
- Aho report output: `reports/aho/host_words_by_search_word.json`.
- Aho implementation: `src/anal_russia_klinik/aho_report.py`; 16-worker container run; worker initializer builds one Aho automaton per worker; chunks write JSONL partials then merge to exhaustive JSON.
- Aho report schema version: 3; includes `by_search_word[*].host_words[*].occurrences` and `inside_host_words[*].occurrences`; no max example cap.
- Docker daemon was unavailable on Windows during verification: `docker version` failed on `dockerDesktopLinuxEngine`; run Docker Desktop before full Docker run.

## 2026-04-29 docker full verification update

- Docker Desktop recovery: `com.docker.service` started, stuck Docker Desktop processes stopped, `docker desktop start` succeeded.
- Docker engine verified: `docker version` returned Engine 29.3.1 and Docker Desktop 4.67.0.
- Docker image build verified: `docker compose build app` succeeded.
- Docker CLI launch verified: `docker compose run --rm app --help` succeeded.
- Full Aho command verified: `docker compose run --rm aho`.
- Full Aho run result: success; workers=16; wall_time_sec=917.5; document_count=552; search_word_count=53160; total_match_count=1818308; total_inside_word_match_count=1592286.
- Full Aho output verified: `reports/aho/host_words_by_search_word.json`; size_bytes=2024088916; schema_version=3; partial_count=16; partials_size_bytes=1073669610.
- g4f Docker smoke verified: `docker compose run --rm g4f-smoke`; ok=true; model=`gpt-4o-mini`; elapsed_sec=4.443; report=`reports/g4f/g4f_smoke.json`.
- Docker Compose persistent service status after one-shot runs: `docker compose ps -a` returned no running containers.
- Post-Docker local tests verified: `py -m pytest -q`; result=5 passed in 0.37s.

## 2026-04-29 aho compact/dashboard update

- Aho compact script location requested by user: `reports/aho/aho_filter_dashboard.py`.
- Aho compact module: `src/anal_russia_klinik/aho_compact.py`; streams `reports/aho/host_words_by_search_word.json` instead of loading 2GB source into memory.
- Compact output: `reports/aho/host_words_compact.json`; fields per search word include source, term_id, canonical, search_word, normalized_search_word, counts, and host_words with host_word/inside_word/count only.
- Filter output: `reports/aho/host_words_filtered_compact.json`; source full report remains unchanged.
- Filter state: `reports/aho/host_word_filters.json`; dashboard adds/removes normalized host_word values and rebuilds filtered compact JSON.
- Compact cleanup rule: skip rows with match_count=0.
- Short-variant rule: generated search variants with searchable length 1 or 2 are skipped; original two-character terms are kept; original one-character terms are skipped.
- Dashboard host display: default host_limit=10000; max host_limit=20000; UI exposes host limit input.
- Verification: rebuilt compact from full 2GB report; result search_word_count=8862, zero_rows=0, bad_short_rows=0, total_match_count=1692162.
- Dashboard server verified: `http://127.0.0.1:8787`; API add/remove host_word filter verified; final filter_host_word_count=0.

## 2026-04-29 aho dashboard performance update

- Correct dashboard behavior: clicking `Filter` only updates `reports/aho/host_word_filters.json`; no filtered JSON rebuild on each click.
- Separate filter-apply script: `reports/aho/apply_host_word_filters.py`; builds `reports/aho/host_words_filtered_compact.json` from `host_words_compact.json` and `host_word_filters.json`.
- Dashboard manual apply button: `Apply filters to JSON`; this is the only dashboard action that rewrites filtered JSON.
- Dashboard pagination added: `Prev`/`Next`, row limit, status text.
- Dashboard source filter fixed: option is `mediq`, not `preparation`.
- Dashboard search API optimized: skips per-host scanning when host/inside/filter constraints are not active.
- Filter click latency verified against real dashboard API: about 0.02-0.03 sec per click.
- Dashboard current URL: `http://127.0.0.1:8787`; current server pid=44968.

## 2026-04-30 aho dashboard per-row bulk update

- Filter storage changed from one growing JSON payload to shards: `reports/aho/host_word_filters_parts/host_words_*.json`; shard_size=500.
- Merged filter file remains: `reports/aho/host_word_filters.json`; generated by `reports/aho/apply_host_word_filters.py` and dashboard `Apply filters to JSON`.
- Dashboard filter list is paged through `/api/filters`; `/api/state` no longer returns the full host_word filter list.
- Global `Add all found host_words` button removed.
- Per-row button added: `Add this drug host_words`; endpoint `/api/filter-row`; scope=exact result row (`source`, `term_id`, `search_word`) plus current host/inside filters; default adds only currently unfiltered host_words.
- Current dashboard URL: `http://127.0.0.1:8787`; current server pid=37860.
- Verification: `py -m pytest -q` result=8 passed; browser showed row_bulk_buttons=7 for query `Агри`; `/api/filter-row` no-op latency=0.014s.

## 2026-04-30 aho dashboard drug/substance grouping update

- Dashboard-only grouping added: `src/anal_russia_klinik/aho_dashboard_groups.py`.
- Search result sections: `Препараты` first, `Действующие вещества` second.
- Substance grouping rule: `source=mediq` and normalized `search_word` differs from normalized `canonical`; grouped by `normalized_search_word`.
- Substance host_words merge rule: host_word counts use max per host_word across duplicate MEDIQ drugs, not sum, to avoid duplicate count inflation.
- Filtering semantics unchanged: host_word filters remain global and are applied after Aho through the existing filter store/output flow.
- Per-row bulk button remains scoped to visible row; for substance groups it filters all host_words for that substance group.
- Current dashboard URL: `http://127.0.0.1:8787`; current server pid=65516 during verification.
- Verification: API query `Агри` returned drug_total=2 and substance_total=3; browser showed both sections and 5 row bulk buttons; tests=8 passed.

## 2026-04-30 MEDIQ name-only search update

- User decision supersedes previous dashboard-only substance grouping: MEDIQ active substances and drug codes are removed from search and dashboard analysis.
- Future MEDIQ term loading in `src/anal_russia_klinik/dictionaries.py` excludes `drug.atx`, `drug.atxGroup`, `drug.mnn.name`, and `drug.mnn.complex[*].name`; canonical drug names and explicit aliases remain searchable.
- Existing compact Aho report cleanup rule in `src/anal_russia_klinik/aho_compact.py`: MEDIQ rows with normalized `search_word != canonical` are removed when rebuilding compact output from old raw/compact data.
- Case-duplicate cleanup rule: compact rows are deduplicated by `(source, term_id, normalized_search_word)` using max count per host_word; current compact rows reduced from 1085 to 1002 after removing 83 MEDIQ case duplicates.
- Dashboard grouping module `src/anal_russia_klinik/aho_dashboard_groups.py` removed; dashboard search displays only `Препараты`; API search payload contains `items` and `drug_total`, without substance sections.
- Existing source full report `reports/aho/host_words_by_search_word.json` remains unchanged; cleaned working report is `reports/aho/host_words_compact.json`; filtered output is rebuilt through `reports/aho/apply_host_word_filters.py`.
- Verification target: `py -m pytest -q` and dashboard API on `http://127.0.0.1:8787`.

## 2026-04-30 detailed filtered Aho output

- Detailed host-word filter module: `src/anal_russia_klinik/aho_detailed_filter.py`; wrapper: `reports/aho/filter_detailed_host_words.py`.
- Default command: `py reports\aho\filter_detailed_host_words.py`; output: `reports/aho/host_words_by_search_word_filtred.json`.
- Output keeps detailed `occurrences`, `document_ids`, and per-host counts; removes host_words present in host-word filter shards/merged filter JSON; filters `inside_host_words` with the same host_word dictionary.
- Default behavior drops empty search-word rows and applies existing policy cleanup: no zero matches, no MEDIQ active-substance/code rows, no short generated variants, duplicate `(source, term_id, normalized_search_word)` rows skipped.
- Verified full run from 2.02GB raw source: scanned=53160 rows; written=729 rows; output_size=10229601 bytes; total_match_count=9535; total_inside_word_match_count=5794; bad_filtered_hosts=0; filter_host_word_count=51861.
- Compact filtered output was also rebuilt after filter merge: `reports/aho/host_words_filtered_compact.json`; dashboard state reports `filter_dirty=false`.

## 2026-04-30 location-grouped LLM review output

- Location grouping module: `src/anal_russia_klinik/aho_location_groups.py`; wrapper: `reports/aho/group_filtered_locations.py`.
- Default command: `py reports\aho\group_filtered_locations.py`; output: `reports/aho/host_words_by_location_filtred.json`.
- Dedup key: `(document_id, char_start, char_end)`; each `by_location[*]` is one LLM review unit.
- Duplicate information is preserved in `by_location[*].matches[*]` with source, term_id, canonical, search_word, host_word, inside_word, and original occurrence object.
- Verified output from detailed filtered report: location_group_count=8753; total_grouped_match_count=9535; duplicate_location_group_count=746; duplicate_match_count=782; output_size=16919346 bytes.

## 2026-04-30 LLM review cases scaffold

- Review case module: `src/anal_russia_klinik/llm_review_cases.py`; wrapper: `reports/llm/build_llm_review_cases.py`; schema doc: `docs/llm-review-cases-schema.md`.
- Default command: `py reports\llm\build_llm_review_cases.py --window-chars 2500`; output: `reports/llm/llm_review_cases.json`.
- JSON grouping: `clinical_recommendations[*].cases[*]`; one case equals one deduplicated location from `host_words_by_location_filtred.json`.
- Context structure: `context.text` contains 2500 chars before and 2500 chars after match; `context.span_start/span_end` are match offsets relative to `context.text`; `context.highlight_spans` contains the match plus detected evidence-level candidates.
- Prompt structure: top-level `prompt_contract`; per-case `llm_payload` contains document title, section, found terms, labels, target kinds, and context reference without duplicating prompt text.
- Validation structure: per-case `llm_result` placeholder and `human_validation` placeholder; top-level `dashboard_validation_schema` contains label/evidence metrics names.
- Verified output: clinical_recommendation_count=528; case_count=8753; missing_document_count=0; evidence_level_candidate_count=24851; output_size=204156539 bytes; tests=`13 passed`.

 
## 2026-04-30 LLM nearby-block review and OpenRouter gold40

- Supersedes previous LLM review unit: model input is now `clinical_recommendations[*].llm_blocks[*]`, not individual `cases[*]`.
- Strict location dedup remains exact: `document_id + char_start + char_end`; no fuzzy +/-4 char dedup is applied.
- Nearby block merge rule: cases inside one clinical recommendation are sorted by offsets and merged only when `next.char_start - previous.char_end <= 100`.
- Block context rule: 2500 chars before and 2500 chars after the whole nearby block; `context.case_spans[*]` gives match offsets relative to block `context.text`.
- Rebuilt `reports/llm/llm_review_cases.json`: clinical_recommendation_count=528; case_count=8753; llm_block_count=6114; multi_case_block_count=1654; evidence_level_candidate_count=24851.
- Added OpenRouter runner: `src/anal_russia_klinik/llm_review_openrouter.py`; wrapper: `reports/llm/run_openrouter_gold40.py`.
- Added manual gold sample: `reports/llm/llm_gold_40.json`; unit=`llm_blocks`; item_count=40; labels include recommendation, contraindication, literature_mention, error.
- OpenRouter execution requires env: `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`; current local env check showed both missing.
- Verification: gold block IDs matched current review cases; `py -m pytest -q` result=`15 passed`.

## 2026-04-30 OpenRouter GPT-5.5 env file

- Runtime env file created: `config/openrouter.env`; ignored by project `.gitignore`.
- Default model value: `OPENROUTER_MODEL=openai/gpt-5.5`.
- Gold40 runner loads env file by default: `py reports\llm\run_openrouter_gold40.py --limit 40`.
- Runner writes incremental results to `reports/llm/openrouter_gold40_results.json` after every completed case.
- Score now includes label accuracy, recommendation-strength accuracy, evidence-level accuracy, confusion matrix, and per-label precision/recall/F1.

## 2026-04-30 OpenRouter GPT-5.5 partial gold40 run

- First sandbox run failed with proxy refusal; escalated network run succeeded in reaching OpenRouter.
- OpenRouter 402 was caused by default reserved output budget 65536 tokens; fixed request with `max_tokens=OPENROUTER_MAX_TOKENS or 1200`.
- OpenRouter/Azure 400 required literal word `JSON` in messages; fixed OpenRouter runner user message with `Return valid JSON only.`
- Live run completed 20 of 40 gold items, then stopped on 403: OpenRouter key total limit exceeded.
- Partial output: `reports/llm/openrouter_gold40_results.json`; analysis: `reports/llm/openrouter_gold40_partial_analysis.json`.
- Partial score: label_accuracy=0.70; recommendation_strength_accuracy=0.85; evidence_level_accuracy=0.80; mismatch_count=8 of 20.
- Runner now resumes from existing output by default; after increasing key limit, run `py reports\llm\run_openrouter_gold40.py --limit 40`.

## 2026-04-30 LLM prompt v2 after partial quality review

- Root model-error pattern: prompt v1 allowed the model to classify the surrounding recommendation paragraph instead of the highlighted Aho hit itself.
- Specific failure modes: background/pathophysiology classified as `literature_mention`; generic substring hits like `Потенциал` inside `потенциальных` classified as `recommendation`; abbreviation/glossary rows classified as `unclear`; some regimen tables classified as `literature_mention`.
- Prompt updated to `prompt_version=llm-review-v2`: classify highlighted `host_word` only; `inside_word=true` unrelated/common-word hits are `error`; pathophysiology/background is `error`; bibliography only is `literature_mention`; evidence levels must be tied to the same bullet/row/sentence/paragraph.
- Superseded on 2026-05-01: OpenRouter runner stores `prompt_version`; resume now keeps existing predictions across prompt-version mismatch to avoid repeated paid calls.
- Historical v2-era guidance was a clean run with `py reports\llm\run_openrouter_gold40.py --limit 40`; current runner keeps existing predictions by `block_id` unless resume is disabled or the output file is replaced.

## 2026-05-01 LLM dashboard and folder architecture update

- Current LLM report folder: `reports/llm`.
- Aho report folder policy: `reports/aho` stores only Aho-Corasick reports, host-word filters, compact/filter outputs, location grouping, and Aho dashboard files.
- LLM folder policy: `reports/llm` stores LLM review cases, manual gold labels, OpenRouter outputs, LLM dashboard logs, and LLM wrapper scripts.
- Current LLM review cases path: `reports/llm/llm_review_cases.json`.
- Current manual gold path: `reports/llm/llm_gold_40.json`.
- Current OpenRouter output path: `reports/llm/openrouter_gold40_results.json`.
- Current LLM review builder wrapper: `reports/llm/build_llm_review_cases.py`; default input remains `reports/aho/host_words_by_location_filtred.json`.
- Current OpenRouter runner wrapper: `reports/llm/run_openrouter_gold40.py`.
- Current LLM audit dashboard wrapper: `reports/llm/llm_audit_dashboard.py`.
- LLM audit dashboard default base directory in `src/anal_russia_klinik/llm_audit_dashboard.py`: `reports/llm`.
- Dashboard UX requirements implemented: all `llm_blocks` from `llm_review_cases.json` are listed, block text uses context offsets from original clinical recommendation text, exact match spans are highlighted brighter than the LLM context block, reviewer controls are buttons, review panel is right-side, labels autosave per case, panes are resizable and hideable, current clinical recommendation can be opened through the header button.

## 2026-04-30 OpenRouter dashboard and model policy

- User corrected model policy: use GPT-5.4 only through OpenRouter; do not call GPT-5.5.
- Current env model: `OPENROUTER_MODEL=openai/gpt-5.4`.
- New key verified through `/api/v1/key`: limit=30; limit_remaining=30; usage=0.
- LLM audit dashboard URL: `http://127.0.0.1:8788`; current server process pair started at 18:51.
- Dashboard shows key status, request JSON, response JSON, per-case usage delta when new predictions contain OpenRouter metadata.
- Gold correction: `Метацин` inside `индометацин` is `error` under strict substring filtering.

## 2026-05-01 OpenRouter case-level prediction migration

- Migrated existing 80 OpenRouter block predictions to nested case-level prediction dictionaries.
- Migrated files: `reports/llm/openrouter_gold40_results.json` and `reports/llm/openrouter_gold40_v3_new_results.json`.
- Backup files: `reports/llm/openrouter_gold40_results.before_case_prediction_migration.json` and `reports/llm/openrouter_gold40_v3_new_results.before_case_prediction_migration.json`.
- Migration rule: each old singleton block prediction is duplicated to every `case_id` in the corresponding `llm_block`.
- Result shape: both migrated result files have `flat_blocks=0`; combined nested predictions cover 80 blocks and 117 case predictions.
- Runner fallback: `complete_openrouter` accepts old singleton JSON only as backward-compatible fallback and expands it to per-case predictions.
- Output token rule: default `max_tokens=max(1200, 450 + case_count * 350)`; `OPENROUTER_MAX_TOKENS` still overrides default budget.
- Resume rule: `run_gold_openrouter(..., resume=True)` keeps existing predictions even when stored `prompt_version` differs from the current prompt, preventing repeated paid calls for already evaluated blocks.
- Dashboard ordering rule: `/api/cases` lists blocks with LLM predictions first, then blocks without predictions; original index order is preserved inside each group.
- Verification: `py -m pytest -q` => `28 passed`; dashboard restart on `http://127.0.0.1:8788` verified `completed=80/6114`.

## 2026-05-04 LLM excluded preparations boundary filter

- `reports/llm/llm_review_cases.json` stays raw and must not be rewritten to remove excluded preparations.
- `reports/llm/excluded_preparations.json` is the LLM blacklist source; dashboard button `Исключить препарат из LLM списка` writes entries there.
- Shared filter implementation: `src/anal_russia_klinik/llm_review_openrouter.py` functions `collect_excluded_preparation_keys`, `filter_block_for_llm`, and `filter_blocks_for_llm`.
- Exclusion rule: a case is removed only when all `case.primary_terms` match `excluded_preparations.json`; mixed cases with at least one non-excluded term remain visible and callable.
- Fully empty blocks after case filtering are skipped before `build_block_messages` and before `complete_openrouter`.
- LLM payload filter guarantees excluded-only cases are absent from `payload.case_spans` and `payload.found_terms`; filtered block also updates `case_ids`, `case_count`, `primary_terms`, `cases`, `context.case_spans`, `context.highlight_spans`, and `llm_payload.found_terms`.
- `context_text` remains broad because surrounding text can still be useful, but excluded-only cases are no longer represented as evaluable case spans or found terms.
- Counters: `completed_all` counts all stored predictions; `completed_visible` counts predictions whose block remains after UI blacklist filtering; `excluded_cases` and `excluded_blocks` count runtime removals before LLM calls.
- Manual labels remain in `reports/llm/llm_gold_40.json`; `exclude_from_model_stats=True` only removes a case from metrics and does not delete raw review or gold data.
- Prompt history update: `llm-review-v3-multi` user task now includes explicit strict inside-word examples for `Metacin`/`indomethacin`, `Децит`, and `рисдиплам`.
- Verification: `py -m pytest -q` => `30 passed`.

## 2026-05-05 Git push preparation policy

- Target GitHub repository requested by user: `https://github.com/TryDotAtwo/Anal_Russia_Klinik2025`.
- Current local git root is parent directory: `C:/Users/Иван Литвак/source/repos/Bot_for_anal_rec`; working project directory is nested `Bot_for_anal_rec/`.
- README policy: parent `README.md` is repository entrypoint; nested `Bot_for_anal_rec/README.md` is full project README.
- Git ignore policy changed to preserve accumulated manual work in Git: `config/manual_filters.csv`, `data/samples/manual_filters.csv`, `reports/aho/host_word_filters.json`, `reports/aho/host_word_filters_parts/*.json`, `reports/llm/excluded_preparations.json`, `reports/llm/llm_gold_40*.json`, `reports/llm/openrouter_gold40*.json`, and `reports/llm/openrouter_all_results*.json`.
- Git ignore policy keeps secrets and local runtime files out of Git: `*.env`, `config/openrouter.env`, dashboard logs, `__pycache__`, `.pytest_cache`, `.playwright-mcp`, `old/`, `reports/aho/partials/`, and local smoke/g4f output folders.
- GitHub ordinary-file limit blocks large generated/source files from normal push: `reports/aho/host_words_by_search_word.json` (~2GB), `reports/llm/llm_review_cases.json` (~306MB), and `data/input/clinical_recommendations.json` (>100MB). Use local rebuild, release artifact, or Git LFS for these files.
- Remote check on 2026-05-05 found `origin=https://github.com/TryDotAtwo/Bot_for_anal_rec.git`; update command for requested repository is `git remote set-url origin https://github.com/TryDotAtwo/Anal_Russia_Klinik2025.git`.
