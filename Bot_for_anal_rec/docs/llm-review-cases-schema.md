# LLM Review Cases Schema

## Purpose

- `reports/llm/llm_review_cases.json` is the canonical input for future LLM classification and dashboard validation.
- One LLM review unit equals one nearby-location block: `clinical_recommendations[*].llm_blocks[*]`.
- Location deduplication key: `document_id + char_start + char_end`.
- Nearby block rule: sorted cases are joined when the distance from previous `char_end` to next `char_start` is `<= 100` characters.

## Case Fields

- `case_id`: stable ID from `document_id + char_start + char_end`.
- `location`: absolute source offsets and original matched text.
- `context.text`: source text window with 2500 characters before and 2500 characters after the match.
- `context.span_start` / `context.span_end`: match offsets relative to `context.text` for dashboard highlighting.
- `context.highlight_spans`: match span plus candidate evidence-level spans.
- `primary_terms`: unique found terms for prompt context.
- `matches`: all duplicate term/host evidence at this exact location.
- `llm_payload`: compact structured payload for future LLM calls.
- `llm_result`: placeholder for model output.
- `human_validation`: placeholder for dashboard/manual review.

## LLM Block Fields

- `llm_blocks[*].block_id`: stable ID from document and block offsets.
- `llm_blocks[*].case_ids`: cases included in this LLM unit.
- `llm_blocks[*].context.text`: text window around the entire block.
- `llm_blocks[*].context.case_spans`: match offsets for every included case relative to block `context.text`.
- `llm_blocks[*].cases`: slim case payloads without duplicated full context.
- `llm_blocks[*].llm_payload`: compact structured payload for one LLM call.

## Labels

- `recommendation`: clinical recommendation or medical indication.
- `contraindication`: contraindication or explicit prohibition/avoidance.
- `literature_mention`: bibliography, citation, study description, or external literature discussion.
- `error`: false match, OCR/table-of-contents/abbreviation noise, or irrelevant substring.
- `unclear`: insufficient context.

## Metrics

- Dashboard should compute label confusion matrix, false positives, false negatives, precision, recall, and F1 by label.
- Evidence-level metrics should be separate for recommendation strength letter (`A|B|C`) and evidence-level number (`1|2|3|4|5`).

## OpenRouter Gold40 Harness

- Gold dataset path: `reports/llm/llm_gold_40.json`.
- Gold unit: `llm_blocks`; every `items[*].block_id` must exist in `reports/llm/llm_review_cases.json`.
- Runner path: `reports/llm/run_openrouter_gold40.py`.
- Dashboard path: `reports/llm/llm_audit_dashboard.py`.
- Default env path: `config/openrouter.env`; ignored by `.gitignore`.
- Required env: `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`; current project policy uses `openai/gpt-5.4`.
- Default smoke command after env setup: `py reports\llm\run_openrouter_gold40.py --limit 40`.
- Output path: `reports/llm/openrouter_gold40_results.json`; partial results are saved after every completed case.
