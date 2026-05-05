# 2026-04-29 Docker Aho Update

- Project renamed to `Anal_Russia_Klinik`.
- Current source package: `src/anal_russia_klinik`.
- Large local inputs moved to `data/input/` and ignored by Git.
- Old code and generated artifacts moved to `old/` and ignored by Git.
- Local outputs use `reports/` and are ignored by Git.
- Docker runtime files added: `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- Full Aho run command: `docker compose run --rm aho`.
- Full Aho output: `reports/aho/host_words_by_search_word.json`.
- Aho partials: `reports/aho/partials/chunk_*.jsonl`.
- Aho parallelization: 16 Docker/Linux workers; one Aho automaton built once per worker via `ProcessPoolExecutor(initializer=...)`.
- Aho report is exhaustive: no max example cap; `host_words[*].occurrences` and `inside_host_words[*].occurrences` contain all occurrences.
- Logging: load documents, load terms, worker init, document scan, chunk done, merge partial, write report.
- Verification passed: `py -m pytest -q`; sample Aho smoke; `docker compose config`.
- Docker recovery completed: Windows service `com.docker.service` started, stuck Docker Desktop processes stopped, `docker desktop start` succeeded.
- Docker execution verified: `docker version` returned Docker Desktop 4.67.0 and Engine 29.3.1.
- Docker image build verified: `docker compose build app` succeeded.
- Docker CLI launch verified: `docker compose run --rm app --help` succeeded.
- Full Aho run verified: `docker compose run --rm aho` succeeded.
- Full Aho stats: workers=16; wall_time_sec=917.5; document_count=552; search_word_count=53160; total_match_count=1818308; total_inside_word_match_count=1592286.
- Full Aho output stats: output=`reports/aho/host_words_by_search_word.json`; size_bytes=2024088916; schema_version=3; partial_count=16; partials_size_bytes=1073669610.
- g4f Docker smoke verified: `docker compose run --rm g4f-smoke`; ok=true; model=`gpt-4o-mini`; elapsed_sec=4.443; report=`reports/g4f/g4f_smoke.json`.
- Compose persistent services after one-shot runs: none; `docker compose ps -a` returned an empty service table.
- Post-Docker local tests verified: `py -m pytest -q`; result=5 passed in 0.37s.

## Aho compact and dashboard follow-up

- Added compact/filter dashboard script: `reports/aho/aho_filter_dashboard.py`.
- Added streaming compact builder: `src/anal_russia_klinik/aho_compact.py`.
- Added local dashboard/API server: `src/anal_russia_klinik/aho_filter_server.py`.
- Rebuilt compact JSON: `reports/aho/host_words_compact.json`; size_bytes=20772456; search_word_count=8862; total_match_count=1692162; zero_match_rows=0; bad_short_rows=0.
- Rebuilt filtered compact JSON: `reports/aho/host_words_filtered_compact.json`; size_bytes=20772202; filter_host_word_count=0.
- Added variant filtering: generated one-character and two-character search forms are removed; original two-character terms remain searchable; original one-character terms are removed.
- Dashboard API verified on `http://127.0.0.1:8787`; add/remove host_word filter verified against real compact data.

## Dashboard performance fix

- Changed filter click behavior: dashboard writes only `reports/aho/host_word_filters.json`; filtered JSON is not rebuilt per click.
- Added standalone apply script: `reports/aho/apply_host_word_filters.py`.
- Changed dashboard button text to `Apply filters to JSON`; this button explicitly rebuilds `reports/aho/host_words_filtered_compact.json`.
- Added pagination controls: `Prev`, `Next`, `row limit`, status text.
- Fixed source dropdown value from `preparation` to `mediq`.
- Optimized search endpoint: host-word scanning is skipped unless host/inside/filter constraints require host-word checks.
- Verification: `py -m pytest -q` passed; `/api/search?q=Агри&limit=3&host_limit=5` returned expected rows; `/api/filter` latency about 0.02-0.03 sec.

## Dashboard sharded filters and per-row bulk

- Filter list sharded into `reports/aho/host_word_filters_parts/host_words_*.json`; shard_size=500; current shard_count=3 for 1122 host_words.
- Merged filter JSON remains `reports/aho/host_word_filters.json`; merge happens during `reports/aho/apply_host_word_filters.py` or dashboard `Apply filters to JSON`.
- `/api/state` no longer returns all filter words; `/api/filters` returns paged filter words.
- Removed global `Add all found host_words`.
- Added per-result-row button `Add this drug host_words`; endpoint `/api/filter-row`; scope=exact source+term_id+search_word row.
- Verification: `py -m pytest -q` passed; browser check found `rowBulkCount=7` for query `Агри`; `/api/filter-row` no-op latency=0.014s.

## Dashboard drug/substance grouping

- Added dashboard grouping module: `src/anal_russia_klinik/aho_dashboard_groups.py`.
- Dashboard now displays result sections: `Препараты` first, `Действующие вещества` second.
- MEDIQ substance grouping: rows where normalized `search_word` differs from normalized `canonical` are grouped by `normalized_search_word`.
- Duplicate MEDIQ substance counts are not summed across drugs; host_word count per grouped substance uses the maximum count per host_word.
- Filtering remains global by host_word; dashboard grouping does not change filtered JSON semantics.
- Verification: `/api/search?q=Агри&limit=20&host_limit=5` returned `drug_total=2`, `substance_total=3`; browser showed both sections; tests=8 passed.

## MEDIQ name-only cleanup

- Previous dashboard drug/substance grouping is superseded by the user decision to remove active-substance and drug-code search from MEDIQ review.
- Future MEDIQ dictionary loading excludes `drug.atx`, `drug.atxGroup`, `drug.mnn.name`, and `drug.mnn.complex[*].name`.
- Existing compact dashboard report was reduced from 8862 to 1085 rows by dropping MEDIQ rows whose normalized `search_word` differs from normalized `canonical`.
- Compact case duplicates were removed by `(source, term_id, normalized_search_word)`; current compact rows=1002; duplicate_normalized_keys=0.
- Dashboard search now returns only preparation rows; `Действующие вещества` section and substance grouping module were removed.
- Filter workflow remains unchanged: dashboard edits host-word filter shards, `reports/aho/apply_host_word_filters.py` rebuilds `reports/aho/host_words_filtered_compact.json`.

## Detailed filtered report

- Added streaming detailed-filter builder: `src/anal_russia_klinik/aho_detailed_filter.py`.
- Added report wrapper script: `reports/aho/filter_detailed_host_words.py`.
- Command executed: `py reports\aho\filter_detailed_host_words.py --progress-interval 1000`.
- Input: `reports/aho/host_words_by_search_word.json`; size_bytes=2024088916.
- Filters: `reports/aho/host_word_filters.json` plus shards; filter_host_word_count=51861.
- Output: `reports/aho/host_words_by_search_word_filtred.json`; size_bytes=10229601.
- Output stats: scanned_rows=53160; written_rows=729; total_match_count=9535; total_inside_word_match_count=5794; total_host_word_count=1894; total_inside_host_word_count=1150.
- Validation: JSON parsed successfully; filtered host_words remaining in output=0; tests=`11 passed`.

## Location-grouped LLM review report

- Added location grouping module: `src/anal_russia_klinik/aho_location_groups.py`.
- Added wrapper script: `reports/aho/group_filtered_locations.py`.
- Command executed: `py reports\aho\group_filtered_locations.py`.
- Input: `reports/aho/host_words_by_search_word_filtred.json`.
- Output: `reports/aho/host_words_by_location_filtred.json`; size_bytes=16919346.
- Dedup key: `document_id + char_start + char_end`; output preserves duplicate term evidence inside `matches`.
- Output stats: location_group_count=8753; total_grouped_match_count=9535; duplicate_location_group_count=746; duplicate_match_count=782; inside_location_group_count=5365.
- Validation: JSON parsed successfully; tests=`12 passed`.

## LLM review cases scaffold

- Added review case builder: `src/anal_russia_klinik/llm_review_cases.py`.
- Added wrapper script: `reports/aho/build_llm_review_cases.py`.
- Added schema doc: `docs/llm-review-cases-schema.md`.
- Command executed: `py reports\aho\build_llm_review_cases.py --window-chars 2500`.
- Input locations: `reports/aho/host_words_by_location_filtred.json`.
- Input clinical texts: `data/input/clinical_recommendations.json`.
- Output: `reports/aho/llm_review_cases.json`; size_bytes=204156539.
- Output grouping: `clinical_recommendations[*].cases[*]`.
- Case context: 2500 chars before + match + 2500 chars after; `span_start/span_end` are relative to `context.text`.
- Dashboard scaffold: `highlight_spans`, `llm_result`, `human_validation`, and `dashboard_validation_schema`.
- Prompt scaffold: `prompt_contract` with labels `recommendation | contraindication | literature_mention | error | unclear`.
- Output stats: clinical_recommendation_count=528; case_count=8753; missing_document_count=0; evidence_level_candidate_count=24851.
- Validation: JSON parsed successfully; tests=`13 passed`.

## LLM nearby-block grouping and OpenRouter gold40

- Updated review case builder: `clinical_recommendations[*].llm_blocks[*]` is the LLM call unit.
- Exact location dedup remains unchanged: `document_id + char_start + char_end`; no +/-4 char fuzzy merge.
- Nearby merge rule: join cases only when gap between previous `char_end` and next `char_start` is `<=100` chars.
- Command executed: `py reports\aho\build_llm_review_cases.py --window-chars 2500 --block-gap-chars 100`.
- Output: `reports/aho/llm_review_cases.json`; size_bytes=306123895 at inspection time.
- Output stats: clinical_recommendation_count=528; case_count=8753; llm_block_count=6114; multi_case_block_count=1654; evidence_level_candidate_count=24851.
- Added OpenRouter module: `src/anal_russia_klinik/llm_review_openrouter.py`.
- Added OpenRouter wrapper: `reports/aho/run_openrouter_gold40.py`; default command after env setup: `py reports\aho\run_openrouter_gold40.py --limit 40`.
- Added manual gold dataset: `reports/aho/llm_gold_40.json`; unit=`llm_blocks`; item_count=40.
- Env check: `OPENROUTER_API_KEY=false`; `OPENROUTER_MODEL=false`; live OpenRouter run was not executed.
- Validation: gold block IDs all present in current `llm_review_cases.json`; tests=`15 passed`.

## OpenRouter GPT-5.5 env setup

- Created ignored env file: `config/openrouter.env`.
- Env keys: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`.
- Default model: `openai/gpt-5.5`.
- Runner default env file: `reports/aho/run_openrouter_gold40.py --env-file config/openrouter.env`.
- Runner command after token insertion: `py reports\aho\run_openrouter_gold40.py --limit 40`.
- Runner output: `reports/aho/openrouter_gold40_results.json`; save_each=true; partial report is written after every completed case.
- Quality report fields: `label_accuracy`, `recommendation_strength_accuracy`, `evidence_level_accuracy`, `confusion_matrix`, `per_label`.
- Validation: env loader parsed 3 keys; `OPENROUTER_API_KEY` blank; `OPENROUTER_MODEL=openai/gpt-5.5`; tests=`15 passed`.

## OpenRouter GPT-5.5 partial run on gold40

- Command attempted: `py reports\aho\run_openrouter_gold40.py --limit 40`.
- Sandbox network result: proxy refusal; escalated run was required.
- OpenRouter 402 fix: set request `max_tokens` to `OPENROUTER_MAX_TOKENS` or default `1200`.
- OpenRouter/Azure 400 fix: include literal `JSON` instruction in user message.
- Completed cases before key limit: 20/40.
- Stop reason: HTTP 403; OpenRouter key total limit exceeded during case 21.
- Partial result path: `reports/aho/openrouter_gold40_results.json`.
- Partial analysis path: `reports/aho/openrouter_gold40_partial_analysis.json`.
- Partial score: label_accuracy=0.70; recommendation_strength_accuracy=0.85; evidence_level_accuracy=0.80; mismatches=8/20.
- Resume behavior: existing predictions in output are skipped by default; after increasing key limit, rerun same command to continue at case 21.

## Prompt v2 correction after partial OpenRouter review

- Prompt v1 failure mode: model sometimes classified the whole nearby paragraph rather than the highlighted Aho hit.
- Prompt v2 ID: `llm-review-v2`.
- New hard rules: `inside_word=true` unrelated/common-word host hits are `error`; pathophysiology/background is `error`; glossary/abbreviation lists are `error`; `literature_mention` is limited to bibliography/reference/external-study discussion; regimen table rows can be `recommendation`; evidence levels must be same bullet/row/sentence/paragraph.
- Resume safety: runner writes `prompt_version`; runner resumes only when existing output prompt version matches current prompt version.
- Operational implication: old partial v1 output will not be mixed into v2 run; after OpenRouter key limit increase, rerun `py reports\aho\run_openrouter_gold40.py --limit 40`.

## LLM audit dashboard and strict substring policy

- Strict user policy update: `Метацин` matched inside `индометацин` is `error`, even when `индометацин` appears inside a treatment statement.
- Corrected `reports/aho/llm_gold_40.json` case `block:d18056cb48d1ca33` to `label=error`.
- Added request/cost metadata capture to OpenRouter predictions: key status before/after, usage deltas, response usage, request model, message chars, max_tokens.
- Added LLM audit dashboard module: `src/anal_russia_klinik/llm_audit_dashboard.py`.
- Added dashboard wrapper: `reports/aho/llm_openrouter_dashboard.py`.
- Dashboard URL: `http://127.0.0.1:8788`.
- Dashboard API: `/api/state?key=1`, `/api/key`, `/api/cases`, `/api/case?index=N`.
- New key verification: limit=30; limit_remaining=30; usage=0; usage_daily=0; usage_monthly=0.
- Model policy correction: use `openai/gpt-5.4` only; do not run `openai/gpt-5.5`.
