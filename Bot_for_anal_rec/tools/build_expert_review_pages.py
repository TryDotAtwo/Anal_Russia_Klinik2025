from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "reports" / "llm"
DEFAULT_REVIEW_CASES = LLM_DIR / "llm_review_cases.json"
DEFAULT_RESULTS = LLM_DIR / "openrouter_all_results.json"
DEFAULT_DRUGS = ROOT / "data" / "input" / "drugs.json"
DEFAULT_BLACKLIST = ROOT / "data" / "input" / "blacklist_drugs.json"
DEFAULT_METADATA = ROOT / "data" / "input" / "MetaData.json"
DEFAULT_EXCLUDED = LLM_DIR / "excluded_preparations.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "expert_review"


EVIDENCE_STRENGTHS = ("A", "B", "C")
EVIDENCE_LEVELS = ("1", "2", "3", "4", "5")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def stable_term_id(source: str, canonical: str) -> str:
    import hashlib

    digest = hashlib.sha1(f"{source}|{canonical}".encode("utf-8")).hexdigest()[:12]
    return f"{source}:{digest}"


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def collect_excluded_keys(data: dict[str, Any]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for item in data.get("items", []):
        if isinstance(item, dict):
            keys.add((text_value(item.get("source")), text_value(item.get("term_id")), text_value(item.get("canonical")).casefold()))
    return keys


def term_is_excluded(term: dict[str, Any], excluded: set[tuple[str, str, str]]) -> bool:
    key = (text_value(term.get("source")), text_value(term.get("term_id")), text_value(term.get("canonical")).casefold())
    return key in excluded


def terms_are_excluded(terms: list[dict[str, Any]], excluded: set[tuple[str, str, str]]) -> bool:
    return bool(terms) and all(term_is_excluded(term, excluded) for term in terms if isinstance(term, dict))


def prediction_items(block_id: str, prediction: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    nested = prediction.get("predictions")
    if isinstance(nested, dict) and nested:
        return [(str(case_id), item) for case_id, item in nested.items() if isinstance(item, dict)]
    return [(block_id, prediction)]


def iter_blocks(review_cases: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for cr in review_cases.get("clinical_recommendations", []):
        for block in cr.get("llm_blocks", []):
            block_id = str(block.get("block_id") or "")
            if block_id:
                blocks[block_id] = block
    return blocks


def terms_by_case(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in block.get("cases", []):
        case_id = str(case.get("case_id") or "")
        if not case_id:
            continue
        terms = case.get("primary_terms") or []
        if isinstance(terms, list):
            output[case_id].extend(term for term in terms if isinstance(term, dict))
    return dict(output)


def case_span_by_id(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spans = block.get("context", {}).get("case_spans") or []
    return {str(span.get("case_id")): span for span in spans if isinstance(span, dict) and span.get("case_id")}


def build_mediq_index(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not Path(path).exists():
        return {}
    rows = load_json(path)
    rows = rows.get("preparations", rows) if isinstance(rows, dict) else rows
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        drug = row.get("drug") if isinstance(row.get("drug"), dict) else {}
        canonical = text_value(drug.get("name") or row.get("name") or row.get("canonical") or row.get("term"))
        if not canonical:
            continue
        term_id = stable_term_id("mediq", canonical)
        index[("mediq", term_id)] = {"canonical": canonical, "entry": row}
    return index


def build_blacklist_index(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not Path(path).exists():
        return {}
    rows = load_json(path)
    rows = rows.get("terms", rows) if isinstance(rows, dict) else rows
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        canonical = text_value(first_value(row, "Название препарата", "РќР°Р·РІР°РЅРёРµ РїСЂРµРїР°СЂР°С‚Р°", "canonical", "name", "term"))
        if not canonical:
            continue
        term_id = stable_term_id("blacklist", canonical)
        index[("blacklist", term_id)] = {"canonical": canonical, "entry": row}
    return index


def compact_drug_info(term: dict[str, Any], mediq: dict[tuple[str, str], dict[str, Any]], blacklist: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    source = text_value(term.get("source"))
    term_id = text_value(term.get("term_id"))
    lookup = mediq.get((source, term_id)) or blacklist.get((source, term_id)) or {}
    entry = lookup.get("entry") if isinstance(lookup.get("entry"), dict) else {}
    drug = entry.get("drug") if isinstance(entry.get("drug"), dict) else {}
    mnn = drug.get("mnn") if isinstance(drug.get("mnn"), dict) else {}
    return {
        "source": source,
        "term_id": term_id,
        "canonical": text_value(term.get("canonical") or lookup.get("canonical")),
        "search_word": text_value(term.get("search_word")),
        "host_word": text_value(term.get("host_word")),
        "inside_word": bool(term.get("inside_word")),
        "mediq": {
            "name": text_value(drug.get("name")),
            "kind": entry.get("kind") if isinstance(entry.get("kind"), list) else [],
            "type": text_value(drug.get("type")),
            "quality": text_value(drug.get("quality")),
            "atx": text_value(drug.get("atx")),
            "atx_group": text_value(drug.get("atxGroup")),
            "mnn": text_value(mnn.get("name")),
            "description": text_value(drug.get("description") or entry.get("description")),
            "criterions": mnn.get("criterions") if isinstance(mnn.get("criterions"), list) else [],
        },
        "blacklist": {
            "name": text_value(first_value(entry, "Название препарата", "РќР°Р·РІР°РЅРёРµ РїСЂРµРїР°СЂР°С‚Р°")),
            "aliases": first_value(entry, "Альтернативные названия", "РђР»СЊС‚РµСЂРЅР°С‚РёРІРЅС‹Рµ РЅР°Р·РІР°РЅРёСЏ") or [],
            "description": text_value(first_value(entry, "Описание", "РћРїРёСЃР°РЅРёРµ", "description")),
        },
    }


def highlighted_context(text: str, span: dict[str, Any]) -> str:
    start = max(0, int(span.get("span_start") or 0))
    end = max(start, int(span.get("span_end") or start))
    start = min(start, len(text))
    end = min(end, len(text))
    before = html.escape(text[:start])
    middle = html.escape(text[start:end])
    after = html.escape(text[end:])
    return before + f'<mark class="hit">{middle}</mark>' + after


def make_rows(
    review_cases: dict[str, Any],
    results: dict[str, Any],
    mediq: dict[tuple[str, str], dict[str, Any]],
    blacklist: dict[tuple[str, str], dict[str, Any]],
    metadata: dict[str, Any],
    excluded: set[tuple[str, str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    blocks = iter_blocks(review_cases)
    excluded = excluded or set()
    output = {"recommendation": [], "contraindication": []}
    for block_id, prediction in results.get("predictions", {}).items():
        block = blocks.get(str(block_id))
        if not block or not isinstance(prediction, dict):
            continue
        terms_map = terms_by_case(block)
        spans = case_span_by_id(block)
        context_text = text_value(block.get("context", {}).get("text"))
        for case_id, item in prediction_items(str(block_id), prediction):
            label = normalize_label(item.get("label"))
            if label not in output:
                continue
            terms = terms_map.get(case_id) or block.get("primary_terms") or []
            if not isinstance(terms, list):
                terms = []
            if terms_are_excluded(terms, excluded):
                continue
            document_id = text_value(block.get("document_id"))
            row = {
                "id": f"{block_id}::{case_id}",
                "block_id": str(block_id),
                "case_id": case_id,
                "label": label,
                "model": {
                    "recommendation_strength": item.get("recommendation_strength"),
                    "evidence_level": item.get("evidence_level"),
                    "reason": item.get("reason"),
                    "evidence_quote": item.get("evidence_quote"),
                    "confidence": item.get("confidence"),
                },
                "clinical": {
                    "document_id": document_id,
                    "title": text_value(block.get("document_title")),
                    "link": text_value(block.get("document_link")),
                    "metadata": metadata.get(document_id, {}) if isinstance(metadata, dict) else {},
                },
                "context_html": highlighted_context(context_text, spans.get(case_id, {})),
                "terms": [compact_drug_info(term, mediq, blacklist) for term in terms if isinstance(term, dict)],
            }
            output[label].append(row)
    return output


def js_data(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_page(title: str, rows: list[dict[str, Any]], page_id: str) -> str:
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --bg:#f3f6fb; --panel:#ffffff; --soft:#f8fafd; --line:#d8dde6; --text:#161a22; --muted:#5d6675; --accent:#155eef; --accent-soft:#e8f0ff; --bad:#b42318; --bad-soft:#fff1f0; --good:#087443; --good-soft:#ecfdf3; --warn:#ad6f00; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.45 Arial, sans-serif; background:var(--bg); color:var(--text); }}
header {{ position:sticky; top:0; z-index:5; padding:14px 20px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); display:flex; gap:16px; align-items:center; justify-content:space-between; box-shadow:0 8px 24px rgba(18,32,56,.06); }}
h1 {{ margin:0; font-size:18px; }}
main {{ display:grid; grid-template-columns:320px 1fr; min-height:calc(100vh - 58px); }}
aside {{ border-right:1px solid var(--line); background:#fff; overflow:auto; max-height:calc(100vh - 58px); }}
.search {{ padding:12px; border-bottom:1px solid var(--line); display:grid; gap:8px; background:var(--soft); }}
input, textarea, select {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; background:#fff; }}
textarea {{ min-height:92px; resize:vertical; }}
.list button {{ width:100%; text-align:left; border:0; border-bottom:1px solid var(--line); background:#fff; padding:10px 12px; cursor:pointer; }}
.list button.active {{ background:var(--accent-soft); border-left:4px solid var(--accent); }}
.list .meta {{ color:var(--muted); font-size:12px; }}
.workspace {{ padding:18px; overflow:auto; max-height:calc(100vh - 58px); }}
.case {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; box-shadow:0 14px 40px rgba(18,32,56,.08); }}
.section {{ padding:18px 20px; border-bottom:1px solid var(--line); }}
.section:last-child {{ border-bottom:0; }}
.section h2 {{ margin:0 0 12px; font-size:20px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ border:1px solid var(--line); border-radius:999px; padding:3px 8px; background:#fff; color:var(--muted); }}
.context {{ white-space:pre-wrap; font-size:15px; line-height:1.6; background:#fffdf5; border:1px solid #f2d98d; border-radius:8px; padding:14px; }}
mark.hit {{ background:#ffe08a; color:#111; padding:1px 3px; border-radius:3px; }}
.term {{ border:1px solid var(--line); border-radius:8px; padding:12px; margin-top:10px; background:#fbfcfe; }}
.term h3 {{ margin:0 0 8px; font-size:15px; }}
.kv {{ display:grid; grid-template-columns:180px 1fr; gap:4px 10px; }}
.kv div:nth-child(odd) {{ color:var(--muted); }}
.controls {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.button-row {{ display:flex; flex-wrap:wrap; gap:8px; }}
.pick {{ border:1px solid var(--line); border-radius:8px; padding:10px 14px; background:#fff; cursor:pointer; min-width:44px; transition:background .12s,border-color .12s,box-shadow .12s,transform .12s; }}
.pick:hover {{ border-color:var(--accent); box-shadow:0 4px 14px rgba(21,94,239,.14); }}
.pick.selected {{ background:var(--accent); color:#fff; border-color:var(--accent); box-shadow:0 0 0 3px rgba(21,94,239,.18), 0 8px 20px rgba(21,94,239,.22); transform:translateY(-1px); font-weight:700; }}
.pick.no.selected {{ background:var(--bad); border-color:var(--bad); box-shadow:0 0 0 3px rgba(180,35,24,.16), 0 8px 20px rgba(180,35,24,.18); }}
.pick.yes.selected {{ background:var(--good); border-color:var(--good); box-shadow:0 0 0 3px rgba(8,116,67,.16), 0 8px 20px rgba(8,116,67,.18); }}
.pick.false-positive.selected {{ background:var(--warn); border-color:var(--warn); box-shadow:0 0 0 3px rgba(173,111,0,.16), 0 8px 20px rgba(173,111,0,.18); }}
.actions {{ display:flex; gap:8px; align-items:center; }}
.primary {{ border:0; border-radius:6px; padding:9px 12px; background:#155eef; color:#fff; cursor:pointer; }}
.secondary {{ border:1px solid var(--line); border-radius:6px; padding:8px 12px; background:#fff; cursor:pointer; }}
.link-button {{ display:inline-flex; align-items:center; justify-content:center; width:max-content; margin-top:10px; border:1px solid var(--accent); border-radius:8px; padding:9px 12px; background:var(--accent); color:#fff; text-decoration:none; font-weight:700; }}
.link-button:hover {{ background:#0f49bd; border-color:#0f49bd; }}
.status {{ color:var(--muted); font-size:12px; }}
@media (max-width: 900px) {{ main {{ grid-template-columns:1fr; }} aside {{ max-height:280px; border-right:0; border-bottom:1px solid var(--line); }} .grid,.controls {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="actions"><span id="count" class="status"></span><button class="secondary" id="exportBtn">Экспорт JSON</button></div>
</header>
<main>
  <aside>
    <div class="search"><input id="q" placeholder="Поиск: препарат, клинрека, МКБ"><span class="status">Ответы сохраняются в localStorage браузера для текущего HTML-файла.</span></div>
    <div id="list" class="list"></div>
  </aside>
  <section class="workspace"><div id="detail" class="case"></div></section>
</main>
<script>
const PAGE_ID = {json.dumps(page_id, ensure_ascii=False)};
const ROWS = {js_data(rows)};
const STORE_KEY = 'expert_review:' + PAGE_ID;
let answers = JSON.parse(localStorage.getItem(STORE_KEY) || '{{}}');
let filtered = ROWS.slice();
let active = 0;
let activeRow = null;
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
function save() {{ localStorage.setItem(STORE_KEY, JSON.stringify(answers)); renderList(); }}
function answer(row) {{ return answers[row.id] || {{}}; }}
function setAnswer(row, key, value, rerender=true) {{ answers[row.id] = {{...answer(row), id:row.id, block_id:row.block_id, case_id:row.case_id, label:row.label, [key]:value, updated_at:new Date().toISOString()}}; save(); if (rerender) renderDetail(); }}
function selected(row, key, value, extra='') {{ return answer(row)[key] === value ? 'pick selected '+extra : 'pick '+extra; }}
function renderList() {{
  $('count').textContent = `items=${{filtered.length}}; reviewed=${{Object.keys(answers).length}}`;
  $('list').innerHTML = filtered.map((row,i) => `<button class="${{i===active?'active':''}}" onclick="active=${{i}};renderList();renderDetail();"><b>${{i+1}}. ${{esc(row.terms[0]?.canonical || row.terms[0]?.host_word || 'term')}}</b><div class="meta">${{esc(row.clinical.title)}} | model=${{esc(row.model.recommendation_strength || '-')}}${{esc(row.model.evidence_level || '')}}</div></button>`).join('');
}}
function termHtml(term) {{
  const med = term.mediq || {{}}, bl = term.blacklist || {{}};
  const crit = (med.criterions || []).map(c => `${{esc(c.criterion)}}: ${{esc(c.points)}}`).join('; ');
  return `<div class="term"><h3>${{esc(term.canonical)}} <span class="chip">${{esc(term.source)}}</span></h3>
    <div class="kv">
      <div>Найдено</div><div><mark class="hit">${{esc(term.host_word)}}</mark>; search_word=${{esc(term.search_word)}}; inside_word=${{term.inside_word}}</div>
      <div>MedIQ name/type/quality</div><div>${{esc(med.name)}}; ${{esc(med.type)}}; ${{esc(med.quality)}}</div>
      <div>ATX/MNN</div><div>${{esc(med.atx)}}; ${{esc(med.atx_group)}}; ${{esc(med.mnn)}}</div>
      <div>MedIQ description</div><div>${{esc(med.description)}}</div>
      <div>MedIQ criterions</div><div>${{crit || '-'}}</div>
      <div>Черный список</div><div>${{esc(bl.name)}}; aliases=${{esc((bl.aliases || []).join(', '))}}</div>
      <div>Описание ЧС</div><div>${{esc(bl.description)}}</div>
    </div></div>`;
}}
function renderDetail() {{
  const row = filtered[active];
  if (!row) {{ $('detail').innerHTML = '<div class="section">Нет записей.</div>'; return; }}
  activeRow = row;
  const a = answer(row);
  const md = row.clinical.metadata || {{}};
  $('detail').innerHTML = `<div class="section">
    <div class="grid"><div><h2>${{esc(row.clinical.title)}}</h2><a class="link-button" href="${{esc(row.clinical.link)}}" target="_blank" rel="noreferrer">Открыть клиническую рекомендацию</a></div>
    <div class="kv"><div>ID</div><div>${{esc(row.clinical.document_id)}}</div><div>МКБ-10</div><div>${{esc(md['МКБ-10'])}}</div><div>Возраст</div><div>${{esc(md['Возрастная группа'])}}</div><div>Разработчик</div><div>${{esc(md['Разработчик'])}}</div><div>Дата</div><div>${{esc(md['Дата размещения'])}}</div><div>Статус</div><div>${{esc(md['Статус применения КР'])}}</div></div></div>
  </div>
  <div class="section"><h2>Фрагмент клинической рекомендации</h2><div class="context">${{row.context_html}}</div></div>
  <div class="section"><h2>Информация по найденному препарату / термину</h2>${{row.terms.map(termHtml).join('')}}</div>
  <div class="section"><h2>В каком контексте упоминает в клинреке по мнению ЛЛМ</h2><div class="kv"><div>Класс</div><div>${{esc(row.label)}}</div><div>Уровень УУР/УДД</div><div>${{esc(row.model.recommendation_strength || '-')}}${{esc(row.model.evidence_level || '')}}</div><div>Цитата</div><div>${{esc(row.model.evidence_quote)}}</div><div>Причина</div><div>${{esc(row.model.reason)}}</div><div>Confidence</div><div>${{esc(row.model.confidence)}}</div></div></div>
  <div class="section controls">
    <div><h2>Уровень убедительности рекомендации</h2><div class="button-row">${{['A','B','C'].map(v=>`<button type="button" class="${{selected(row,'expert_strength',v)}}" onclick="setAnswer(activeRow,'expert_strength','${{v}}')">${{v}}</button>`).join('')}}</div></div>
    <div><h2>Уровень достоверности доказательств</h2><div class="button-row">${{['1','2','3','4','5'].map(v=>`<button type="button" class="${{selected(row,'expert_evidence',v)}}" onclick="setAnswer(activeRow,'expert_evidence','${{v}}')">${{v}}</button>`).join('')}}</div></div>
    <div><h2>Экспертное решение</h2><div class="button-row"><button type="button" class="${{selected(row,'expert_decision','recommend','yes')}}" onclick="setAnswer(activeRow,'expert_decision','recommend')">Рекомендовал бы</button><button type="button" class="${{selected(row,'expert_decision','not_recommend','no')}}" onclick="setAnswer(activeRow,'expert_decision','not_recommend')">Не рекомендовал бы</button><button type="button" class="${{selected(row,'expert_decision','false_positive','false-positive')}}" onclick="setAnswer(activeRow,'expert_decision','false_positive')">Ошибочно найдено</button></div></div>
    <div><h2>Объяснение</h2><textarea id="reason" oninput="setAnswer(activeRow,'expert_reason',this.value,false)">${{esc(a.expert_reason || '')}}</textarea></div>
  </div>`;
}}
$('q').addEventListener('input', () => {{
  const q = $('q').value.toLowerCase().trim();
  filtered = ROWS.filter(row => JSON.stringify([row.clinical,row.terms]).toLowerCase().includes(q));
  active = 0; renderList(); renderDetail();
}});
$('exportBtn').addEventListener('click', () => {{
  const blob = new Blob([JSON.stringify({{page_id:PAGE_ID, exported_at:new Date().toISOString(), answers:Object.values(answers)}}, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = PAGE_ID + '_answers.json'; a.click(); URL.revokeObjectURL(a.href);
}});
renderList(); renderDetail();
</script>
</body>
</html>
"""


def write_pages(rows_by_label: dict[str, list[dict[str, Any]]], output_dir: str | Path, chunk_size: int) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    recs = rows_by_label["recommendation"]
    page_count = max(1, math.ceil(len(recs) / chunk_size))
    for page_index in range(page_count):
        start = page_index * chunk_size
        chunk = recs[start : start + chunk_size]
        path = output / f"expert_recommendations_{page_index + 1:02d}.html"
        path.write_text(render_page(f"Экспертная оценка рекомендаций {page_index + 1}/{page_count}", chunk, path.stem), encoding="utf-8")
        written.append(str(path))
    contra_path = output / "expert_contraindications.html"
    contra_path.write_text(render_page("Экспертная оценка контр-рекомендаций", rows_by_label["contraindication"], contra_path.stem), encoding="utf-8")
    written.append(str(contra_path))
    index_path = output / "index.html"
    links = "\n".join(f'<li><a href="{html.escape(Path(path).name)}">{html.escape(Path(path).name)}</a></li>' for path in written)
    index_path.write_text(f"<!doctype html><meta charset='utf-8'><title>Expert review</title><h1>Expert review pages</h1><ul>{links}</ul>", encoding="utf-8")
    written.append(str(index_path))
    return {"output_dir": str(output), "recommendation_count": len(recs), "contraindication_count": len(rows_by_label["contraindication"]), "files": written}


def build_expert_review_pages(
    *,
    review_cases_path: str | Path = DEFAULT_REVIEW_CASES,
    results_path: str | Path = DEFAULT_RESULTS,
    drugs_path: str | Path = DEFAULT_DRUGS,
    blacklist_path: str | Path = DEFAULT_BLACKLIST,
    metadata_path: str | Path = DEFAULT_METADATA,
    excluded_preparations_path: str | Path = DEFAULT_EXCLUDED,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 400,
) -> dict[str, Any]:
    review_cases = load_json(review_cases_path)
    results = load_json(results_path)
    metadata = load_json(metadata_path) if Path(metadata_path).exists() else {}
    excluded = collect_excluded_keys(load_json(excluded_preparations_path)) if Path(excluded_preparations_path).exists() else set()
    rows = make_rows(review_cases, results, build_mediq_index(drugs_path), build_blacklist_index(blacklist_path), metadata, excluded)
    report = write_pages(rows, output_dir, chunk_size)
    report["excluded_preparation_count"] = len(excluded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="build-expert-review-pages")
    parser.add_argument("--review-cases", default=str(DEFAULT_REVIEW_CASES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--drugs", default=str(DEFAULT_DRUGS))
    parser.add_argument("--blacklist", default=str(DEFAULT_BLACKLIST))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--excluded-preparations", default=str(DEFAULT_EXCLUDED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=400)
    args = parser.parse_args()
    report = build_expert_review_pages(
        review_cases_path=args.review_cases,
        results_path=args.results,
        drugs_path=args.drugs,
        blacklist_path=args.blacklist,
        metadata_path=args.metadata,
        excluded_preparations_path=args.excluded_preparations,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
