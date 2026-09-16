from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "reports" / "llm"
DEFAULT_REVIEW_CASES = LLM_DIR / "llm_review_cases.json"
DEFAULT_RESULTS = LLM_DIR / "openrouter_all_results.json"
DEFAULT_GOLD = LLM_DIR / "llm_gold_40.json"
DEFAULT_EXCLUDED = LLM_DIR / "excluded_preparations.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "gold_review"


LABELS = ["recommendation", "contraindication", "literature_mention", "error", "unclear"]
TARGET_KINDS = ["drug", "method", "marker", "other"]
STRENGTHS = ["", "A", "B", "C"]
LEVELS = ["", "1", "2", "3", "4", "5"]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def collect_excluded_keys(data: dict[str, Any]) -> set[tuple[str, str, str]]:
    output: set[tuple[str, str, str]] = set()
    for item in data.get("items", []):
        if isinstance(item, dict):
            output.add((text_value(item.get("source")), text_value(item.get("term_id")), text_value(item.get("canonical")).casefold()))
    return output


def terms_are_excluded(terms: list[dict[str, Any]], excluded: set[tuple[str, str, str]]) -> bool:
    if not terms:
        return False
    for term in terms:
        key = (text_value(term.get("source")), text_value(term.get("term_id")), text_value(term.get("canonical")).casefold())
        if key not in excluded:
            return False
    return True


def case_terms(block: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    for case in block.get("cases", []):
        if str(case.get("case_id") or "") == str(case_id):
            return [term for term in case.get("primary_terms", []) if isinstance(term, dict)]
    return [term for term in block.get("primary_terms", []) if isinstance(term, dict)]


def case_ids(block: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for source in (
        block.get("case_ids", []),
        [case.get("case_id") for case in block.get("cases", [])],
        [span.get("case_id") for span in block.get("context", {}).get("case_spans", [])],
    ):
        for raw in source or []:
            value = str(raw or "")
            if value and value not in ids:
                ids.append(value)
    return ids or ["default"]


def prediction_for_case(prediction_block: Any, case_id: str, block_id: str) -> dict[str, Any]:
    if not isinstance(prediction_block, dict):
        return {}
    nested = prediction_block.get("predictions")
    if isinstance(nested, dict):
        for key in (case_id, block_id, "default"):
            value = nested.get(key)
            if isinstance(value, dict):
                return value
        if len(nested) == 1:
            value = next(iter(nested.values()))
            return value if isinstance(value, dict) else {}
    if isinstance(prediction_block.get("label"), str):
        return prediction_block
    return {}


def gold_by_block(gold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("block_id")): item for item in gold.get("items", []) if isinstance(item, dict)}


def case_gold(item: dict[str, Any] | None, case_id: str) -> dict[str, Any]:
    if not item:
        return {}
    case_golds = item.get("case_golds")
    if isinstance(case_golds, dict) and isinstance(case_golds.get(case_id), dict):
        return case_golds[case_id]
    if isinstance(item.get("gold"), dict):
        return item["gold"]
    return {}


def render_context(text: str, spans: list[dict[str, Any]], block_start: int | None, block_end: int | None) -> str:
    marks: list[tuple[int, int, str]] = []
    if block_start is not None and block_end is not None and block_end > block_start:
        marks.append((block_start, block_end, "block-mark"))
    for span in spans:
        start = int(span.get("span_start") or 0)
        end = int(span.get("span_end") or start)
        if end > start:
            marks.append((start, end, "word-mark"))
    cuts = {0, len(text)}
    for start, end, _ in marks:
        cuts.add(max(0, min(len(text), start)))
        cuts.add(max(0, min(len(text), end)))
    points = sorted(cuts)
    out = []
    for left, right in zip(points, points[1:]):
        part = html.escape(text[left:right])
        classes = " ".join(sorted({cls for start, end, cls in marks if left >= start and right <= end}))
        out.append(f'<span class="{classes}">{part}</span>' if classes else part)
    return "".join(out)


def make_rows(review_cases: dict[str, Any], results: dict[str, Any], gold: dict[str, Any], excluded: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    existing_gold = gold_by_block(gold)
    predictions = results.get("predictions", {}) if isinstance(results, dict) else {}
    rows: list[dict[str, Any]] = []
    for cr in review_cases.get("clinical_recommendations", []):
        for block in cr.get("llm_blocks", []):
            block_id = str(block.get("block_id") or "")
            kept_case_ids = [case_id for case_id in case_ids(block) if not terms_are_excluded(case_terms(block, case_id), excluded)]
            if not block_id or not kept_case_ids:
                continue
            context = block.get("context", {})
            spans = [span for span in context.get("case_spans", []) if str(span.get("case_id") or "") in set(kept_case_ids)]
            block_gold = existing_gold.get(block_id)
            row = {
                "block_id": block_id,
                "document_id": block.get("document_id"),
                "document_title": block.get("document_title"),
                "document_link": block.get("document_link"),
                "context_html": render_context(text_value(context.get("text")), spans, context.get("block_span_start"), context.get("block_span_end")),
                "terms": block.get("primary_terms", []),
                "cases": [],
            }
            prediction_block = predictions.get(block_id, {})
            for case_id in kept_case_ids:
                row["cases"].append(
                    {
                        "case_id": case_id,
                        "span": next((span for span in spans if str(span.get("case_id") or "") == case_id), {}),
                        "terms": case_terms(block, case_id),
                        "prediction": prediction_for_case(prediction_block, case_id, block_id),
                        "gold": case_gold(block_gold, case_id),
                    }
                )
            rows.append(row)
    return rows


def json_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_page(title: str, page_id: str, rows: list[dict[str, Any]], source_review_cases: str) -> str:
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#eef3f7;--panel:#fff;--line:#d8e0ea;--text:#162033;--muted:#667085;--blue:#2563eb;--green:#15803d;--red:#b42318;--amber:#b45309}}
*{{box-sizing:border-box}}body{{margin:0;height:100vh;overflow:hidden;font:13px/1.35 Arial,sans-serif;background:var(--bg);color:var(--text)}}button,input,textarea{{font:inherit}}
header{{height:72px;padding:10px 14px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;box-shadow:0 4px 18px rgba(15,23,42,.08)}}h1{{font-size:18px;margin:0 12px 0 0}}input{{height:32px;border:1px solid #cbd5e1;border-radius:7px;padding:0 10px;min-width:300px}}button{{border:1px solid #cbd5e1;border-radius:7px;background:#fff;padding:7px 10px;cursor:pointer}}button.primary{{background:var(--blue);border-color:var(--blue);color:#fff;font-weight:700}}button.active{{background:#dbeafe;border-color:#60a5fa;color:#1d4ed8;font-weight:700}}
main{{height:calc(100vh - 72px);display:grid;grid-template-columns:330px minmax(420px,1fr) 420px;gap:10px;padding:10px}}.pane{{min-height:0;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 12px 32px rgba(15,23,42,.06)}}.pane-title{{height:38px;padding:10px;border-bottom:1px solid #e5e7eb;background:#f8fafc;font-weight:700}}.pane-body{{overflow:auto;flex:1}}
.row{{padding:10px;border-bottom:1px solid #edf2f7;cursor:pointer}}.row.active{{background:#eff6ff;border-left:4px solid var(--blue);padding-left:6px}}.row-title{{font-weight:700;margin-bottom:5px}}.chips{{display:flex;flex-wrap:wrap;gap:5px}}.chip{{font-size:11px;border-radius:999px;background:#e5e7eb;padding:3px 7px;color:#374151}}.chip.llm{{background:#dbeafe;color:#1e40af}}.chip.gold{{background:#dcfce7;color:#166534}}
.toolbar{{padding:8px 10px;border-bottom:1px solid #e5e7eb;display:flex;gap:8px;align-items:center}}.link-button{{display:inline-flex;background:var(--blue);color:#fff;text-decoration:none;border-radius:7px;padding:7px 10px;font-weight:700}}.terms{{padding:8px 10px;border-bottom:1px solid #e5e7eb;display:flex;gap:5px;flex-wrap:wrap}}.doc{{padding:16px 18px;white-space:pre-wrap;font:14px/1.65 Georgia,serif}}.block-mark{{background:#fff3bf}}.word-mark{{background:#facc15;font-weight:800}}
.card{{border:1px solid #dbe3ee;border-radius:9px;margin:10px;background:#fff;overflow:hidden}}.card-head{{padding:9px 10px;background:#f8fafc;border-bottom:1px solid #e5e7eb}}.card-body{{padding:10px;display:grid;gap:10px}}.kv{{display:grid;grid-template-columns:80px 1fr;gap:4px 8px}}.kv span{{color:var(--muted)}}.choice{{display:grid;gap:5px}}.choice-label{{font-size:11px;font-weight:700;color:#4b5563;text-transform:uppercase}}.choice-buttons{{display:flex;flex-wrap:wrap;gap:5px}}.choice-buttons button{{height:28px}}textarea{{width:100%;min-height:58px;border:1px solid #cbd5e1;border-radius:7px;padding:8px;resize:vertical}}.warn.active{{background:#fffbeb;border-color:#f59e0b;color:#92400e}}.status{{color:var(--muted);margin-left:auto}}
@media(max-width:1000px){{body{{overflow:auto;height:auto}}main{{height:auto;grid-template-columns:1fr}}.pane{{min-height:320px}}}}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><input id="q" placeholder="Поиск: клинрека, препарат, block_id"><button class="primary" id="exportBtn">Экспорт gold JSON</button><span class="status" id="status"></span></header>
<main><section class="pane"><div class="pane-title">Блоки</div><div class="pane-body" id="list"></div></section><section class="pane"><div class="pane-title">Текст клинреки</div><div id="center"></div></section><aside class="pane"><div class="pane-title">Разметка</div><div class="pane-body" id="review"></div></aside></main>
<script>
const PAGE_ID={json.dumps(page_id, ensure_ascii=False)};
const SOURCE_REVIEW_CASES={json.dumps(source_review_cases, ensure_ascii=False)};
const ROWS={json_script(rows)};
const STORE_KEY='gold_static:'+PAGE_ID;
const CHOICES={{label:{json.dumps(LABELS, ensure_ascii=False)},target_kind:{json.dumps(TARGET_KINDS, ensure_ascii=False)},recommendation_strength:{json.dumps(STRENGTHS, ensure_ascii=False)},evidence_level:{json.dumps(LEVELS, ensure_ascii=False)}}};
let answers=JSON.parse(localStorage.getItem(STORE_KEY)||'{{}}'), filtered=ROWS.slice(), active=0, activeRow=null;
const $=id=>document.getElementById(id); const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function caseAnswer(row,caseId){{return (answers[row.block_id]&&answers[row.block_id].case_golds&&answers[row.block_id].case_golds[caseId])||row.cases.find(c=>c.case_id===caseId)?.gold||{{}}}}
function saveCase(row,caseId,patch){{const item=answers[row.block_id]||{{block_id:row.block_id,document_id:row.document_id,document_title:row.document_title,document_link:row.document_link,case_golds:{{}}}}; item.case_golds=item.case_golds||{{}}; item.case_golds[caseId]={{...caseAnswer(row,caseId),...patch,reviewer:'static_gold_manual',reviewed_at:new Date().toISOString()}}; answers[row.block_id]=item; localStorage.setItem(STORE_KEY,JSON.stringify(answers)); renderList(); renderReview();}}
function pick(row,caseId,field,value){{saveCase(row,caseId,{{[field]:value||null}})}}
function renderList(){{$('status').textContent=`items=${{filtered.length}}; labeled=${{Object.keys(answers).length}}`; $('list').innerHTML=filtered.map((r,i)=>`<div class="row ${{i===active?'active':''}}" onclick="active=${{i}};renderAll()"><div class="row-title">#${{i+1}} ${{esc(r.document_title)}}</div><div class="chips"><span class="chip">cases=${{r.cases.length}}</span>${{r.terms.slice(0,3).map(t=>`<span class="chip">${{esc(t.canonical||t.host_word)}}</span>`).join('')}}</div></div>`).join('')||'<div class="row">Нет записей</div>'}}
function renderCenter(){{const r=activeRow; $('center').innerHTML=`<div class="toolbar"><b>${{esc(r.document_title)}}</b>${{r.document_link?`<a class="link-button" href="${{esc(r.document_link)}}" target="_blank" rel="noreferrer">Открыть клинреку</a>`:''}}</div><div class="terms">${{r.terms.map(t=>`<span class="chip">${{esc(t.canonical)}} <b>${{esc(t.host_word)}}</b>${{t.inside_word?' inside':''}}</span>`).join('')}}</div><div class="doc">${{r.context_html}}</div>`}}
function choice(row,caseId,field,label,values,current){{return `<div class="choice"><div class="choice-label">${{label}}</div><div class="choice-buttons">${{values.map(v=>`<button class="${{String(current??'')===String(v)?'active':''}}" onclick="pick(activeRow,'${{caseId}}','${{field}}','${{v}}')">${{esc(v||'none')}}</button>`).join('')}}</div></div>`}}
function renderReview(){{const r=activeRow; $('review').innerHTML=r.cases.map(c=>{{const g=caseAnswer(r,c.case_id),p=c.prediction||{{}},terms=c.terms.map(t=>`${{t.canonical}} [${{t.host_word}}]`).join(', '); return `<div class="card"><div class="card-head"><b>${{esc(terms||c.case_id)}}</b><br><small>${{esc(c.case_id)}} | "${{esc(c.span?.text||'')}}"</small></div><div class="card-body"><div class="kv"><span>LLM</span><b>${{esc(p.label||'-')}} / ${{esc(p.target_kind||'-')}} / ${{esc(p.recommendation_strength||'-')}}${{esc(p.evidence_level||'')}}</b><span>reason</span><b>${{esc(p.reason||'-')}}</b><span>gold</span><b>${{esc(g.label||'-')}} / ${{esc(g.target_kind||'-')}} / ${{esc(g.recommendation_strength||'-')}}${{esc(g.evidence_level||'')}}</b></div>${{choice(r,c.case_id,'label','Label',CHOICES.label,g.label)}}${{choice(r,c.case_id,'target_kind','Target',CHOICES.target_kind,g.target_kind)}}${{choice(r,c.case_id,'recommendation_strength','UUR',CHOICES.recommendation_strength,g.recommendation_strength)}}${{choice(r,c.case_id,'evidence_level','UDD',CHOICES.evidence_level,g.evidence_level)}}<button class="warn ${{g.exclude_from_model_stats?'active':''}}" onclick="saveCase(activeRow,'${{c.case_id}}',{{exclude_from_model_stats:${{!g.exclude_from_model_stats}}}})">Не учитывать в статистике модели</button><label class="choice"><span class="choice-label">Comment</span><textarea oninput="saveCase(activeRow,'${{c.case_id}}',{{comment:this.value}})">${{esc(g.comment||'')}}</textarea></label><label class="choice"><span class="choice-label">Quote</span><textarea oninput="saveCase(activeRow,'${{c.case_id}}',{{evidence_quote:this.value}})">${{esc(g.evidence_quote||'')}}</textarea></label></div></div>`}}).join('')}}
function renderAll(){{activeRow=filtered[active]; renderList(); if(!activeRow){{$('center').innerHTML='';$('review').innerHTML='';return}} renderCenter(); renderReview()}}
$('q').addEventListener('input',()=>{{const q=$('q').value.toLowerCase(); filtered=ROWS.filter(r=>JSON.stringify([r.block_id,r.document_title,r.terms]).toLowerCase().includes(q)); active=0; renderAll()}});
$('exportBtn').onclick=()=>{{const items=Object.values(answers); const data={{schema_version:1,unit:'llm_blocks',source_review_cases:SOURCE_REVIEW_CASES,exported_at:new Date().toISOString(),items}}; const blob=new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=PAGE_ID+'_gold.json'; a.click(); URL.revokeObjectURL(a.href)}};
renderAll();
</script>
</body>
</html>"""


def write_pages(rows: list[dict[str, Any]], output_dir: str | Path, chunk_size: int, source_review_cases: str) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    page_count = max(1, math.ceil(len(rows) / chunk_size))
    for page_index in range(page_count):
        chunk = rows[page_index * chunk_size : (page_index + 1) * chunk_size]
        path = output / f"gold_review_{page_index + 1:02d}.html"
        path.write_text(render_page(f"Gold-разметка {page_index + 1}/{page_count}", path.stem, chunk, source_review_cases), encoding="utf-8")
        files.append(str(path))
    links = "\n".join(f'<li><a href="{html.escape(Path(path).name)}">{html.escape(Path(path).name)}</a></li>' for path in files)
    index = output / "index.html"
    index.write_text(f"<!doctype html><meta charset='utf-8'><title>Gold review</title><h1>Gold review pages</h1><ul>{links}</ul>", encoding="utf-8")
    files.append(str(index))
    return {"output_dir": str(output), "block_count": len(rows), "page_count": page_count, "files": files}


def build_gold_review_pages(
    *,
    review_cases_path: str | Path = DEFAULT_REVIEW_CASES,
    results_path: str | Path = DEFAULT_RESULTS,
    gold_path: str | Path = DEFAULT_GOLD,
    excluded_preparations_path: str | Path = DEFAULT_EXCLUDED,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 400,
) -> dict[str, Any]:
    review_cases = load_json(review_cases_path)
    results = load_json(results_path) if Path(results_path).exists() else {}
    gold = load_json(gold_path) if Path(gold_path).exists() else {"items": []}
    excluded = collect_excluded_keys(load_json(excluded_preparations_path)) if Path(excluded_preparations_path).exists() else set()
    rows = make_rows(review_cases, results, gold, excluded)
    source_path = Path(review_cases_path).resolve()
    source_name = source_path.relative_to(ROOT).as_posix() if source_path.is_relative_to(ROOT) else source_path.name
    report = write_pages(rows, output_dir, chunk_size, source_name)
    report["excluded_preparation_count"] = len(excluded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="build-gold-review-pages")
    parser.add_argument("--review-cases", default=str(DEFAULT_REVIEW_CASES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--excluded-preparations", default=str(DEFAULT_EXCLUDED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=400)
    args = parser.parse_args()
    report = build_gold_review_pages(
        review_cases_path=args.review_cases,
        results_path=args.results,
        gold_path=args.gold,
        excluded_preparations_path=args.excluded_preparations,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
