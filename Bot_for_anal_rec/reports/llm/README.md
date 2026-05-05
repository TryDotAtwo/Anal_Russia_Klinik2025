# LLM Reports

- `llm_review_cases.json`: all LLM review blocks built from filtered Aho locations.
- `llm_gold_40.json`: manual gold sample used for quality checks.
- `openrouter_gold40_results.json`: OpenRouter prediction output for the gold sample.
- `build_llm_review_cases.py`: rebuilds `llm_review_cases.json` from `reports/aho/host_words_by_location_filtred.json`.
- `run_openrouter_gold40.py`: runs the gold sample through OpenRouter.
- `llm_audit_dashboard.py`: starts the manual audit dashboard.

The Aho folder stores Aho-Corasick reports and filters only. LLM review, gold labels, model runs, and LLM dashboard files belong in this folder.
