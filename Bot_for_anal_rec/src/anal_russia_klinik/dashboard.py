from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Document


def write_dashboard(
    output_dir: str | Path,
    documents: list[Document],
    classified: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rejected_for_dashboard = [
        {
            **item,
            "llm": {
                "label": "filtered_reject",
                "confidence": "",
                "reason_short": "rejected_before_llm",
                "udd": "",
                "uur": "",
                "evidence_level": "",
                "recommendation_strength": "",
            },
        }
        for item in rejected
    ]
    data = {
        "documents": [
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "link": doc.link,
                "text": doc.text,
            }
            for doc in documents
        ],
        "matches": [*classified, *rejected_for_dashboard],
        "rejected": rejected,
    }
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    path = directory / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Anal_Russia_Klinik Audit Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8f5;
      --panel: #ffffff;
      --ink: #18201c;
      --muted: #65716b;
      --line: #d9dfd8;
      --accent: #1f7a5c;
      --warn: #b55236;
      --mark: #ffe08a;
      --mark-active: #ffbd59;
      --shadow: 0 12px 28px rgba(24, 32, 28, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    .app {
      display: grid;
      grid-template-columns: 280px minmax(420px, 1fr) 360px;
      min-height: 100vh;
    }
    aside, main, section { min-width: 0; }
    .sidebar, .details {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
      max-height: 100vh;
    }
    .details {
      border-right: 0;
      border-left: 1px solid var(--line);
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.86);
      position: sticky;
      top: 0;
      z-index: 4;
      backdrop-filter: blur(8px);
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 18px; font-weight: 720; }
    h2 { font-size: 15px; font-weight: 700; margin-bottom: 10px; }
    h3 { font-size: 13px; font-weight: 700; color: var(--muted); margin: 16px 0 8px; }
    .doc-list, .match-list { display: grid; gap: 8px; }
    button, select, input {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      min-height: 34px;
      padding: 7px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    .doc-button, .match-button {
      text-align: left;
      width: 100%;
      display: block;
    }
    .doc-button.active, .match-button.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .filters {
      display: grid;
      grid-template-columns: repeat(3, minmax(110px, 1fr));
      gap: 8px;
      align-items: center;
    }
    .content {
      overflow: auto;
      max-height: 100vh;
      background: #fbfcfa;
    }
    .text-pane {
      padding: 22px;
      max-width: 1080px;
      margin: 0 auto;
      white-space: pre-wrap;
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 15px;
      line-height: 1.65;
    }
    mark {
      background: var(--mark);
      border-radius: 3px;
      padding: 1px 2px;
      cursor: pointer;
    }
    mark.active { background: var(--mark-active); outline: 2px solid rgba(181,82,54,.2); }
    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 7px 10px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }
    .key { color: var(--muted); }
    .value { overflow-wrap: anywhere; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e8f2ed;
      color: #14543f;
      font-size: 12px;
      font-weight: 650;
    }
    .pill.reject { background: #f6e7e1; color: #8d371f; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .muted { color: var(--muted); }
    .context {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      white-space: pre-wrap;
      max-height: 220px;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .sidebar, .details, .content { max-height: none; }
      .details { border-left: 0; border-top: 1px solid var(--line); }
      .filters { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>Clinical audit</h1>
      <p class="muted">Raw matches, ручной фильтр, LLM-классификация.</p>
      <h2>Клинреки</h2>
      <div id="docList" class="doc-list"></div>
      <h2 style="margin-top:18px">Совпадения</h2>
      <div id="matchList" class="match-list"></div>
    </aside>
    <main class="content">
      <div class="header">
        <h1 id="docTitle">Документ</h1>
        <div class="filters">
          <select id="sourceFilter"><option value="">source: all</option></select>
          <select id="actionFilter"><option value="">action: all</option></select>
          <select id="labelFilter"><option value="">label: all</option></select>
        </div>
      </div>
      <article id="textPane" class="text-pane"></article>
    </main>
    <section class="details">
      <h2>Match details</h2>
      <div id="detailsPane" class="muted">Выберите совпадение.</div>
    </section>
  </div>
  <script>
    const DATA = __DATA_JSON__;
    const state = { docId: DATA.documents[0]?.document_id || "", matchId: "", source: "", action: "", label: "" };
    const byId = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const matches = () => DATA.matches.filter(item => {
      const m = item.match, d = item.filter_decision || {}, l = item.llm || {};
      return m.document_id === state.docId &&
        (!state.source || m.source === state.source) &&
        (!state.action || d.action === state.action) &&
        (!state.label || l.label === state.label);
    });
    function initFilters() {
      fillSelect("sourceFilter", [...new Set(DATA.matches.map(x => x.match.source).filter(Boolean))], "source: all");
      fillSelect("actionFilter", [...new Set(DATA.matches.map(x => x.filter_decision?.action).filter(Boolean))], "action: all");
      fillSelect("labelFilter", [...new Set(DATA.matches.map(x => x.llm?.label).filter(Boolean))], "label: all");
      byId("sourceFilter").onchange = e => { state.source = e.target.value; render(); };
      byId("actionFilter").onchange = e => { state.action = e.target.value; render(); };
      byId("labelFilter").onchange = e => { state.label = e.target.value; render(); };
    }
    function fillSelect(id, values, label) {
      byId(id).innerHTML = `<option value="">${label}</option>` + values.sort().map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
    }
    function renderDocs() {
      byId("docList").innerHTML = DATA.documents.map(doc => {
        const count = DATA.matches.filter(x => x.match.document_id === doc.document_id).length;
        return `<button class="doc-button ${doc.document_id === state.docId ? "active" : ""}" data-doc="${esc(doc.document_id)}">
          <strong>${esc(doc.title || doc.document_id)}</strong><br><span class="muted">${count} matches</span>
        </button>`;
      }).join("");
      document.querySelectorAll("[data-doc]").forEach(btn => btn.onclick = () => {
        state.docId = btn.dataset.doc; state.matchId = ""; render();
      });
    }
    function renderMatches() {
      const rows = matches();
      if (!state.matchId && rows[0]) state.matchId = rows[0].raw_match_id;
      byId("matchList").innerHTML = rows.map(item => {
        const m = item.match, l = item.llm || {}, d = item.filter_decision || {};
        return `<button class="match-button ${item.raw_match_id === state.matchId ? "active" : ""}" data-match="${esc(item.raw_match_id)}">
          <span class="pill ${d.action === "reject" ? "reject" : ""}">${esc(d.action || "review")}</span>
          <strong>${esc(m.canonical)}</strong><br>
          <span class="muted">${esc(m.matched_text)} · ${esc(m.source)} · ${esc(l.label || "")}</span>
        </button>`;
      }).join("") || `<span class="muted">Нет совпадений для текущих фильтров.</span>`;
      document.querySelectorAll("[data-match]").forEach(btn => btn.onclick = () => {
        state.matchId = btn.dataset.match; render();
      });
    }
    function renderText() {
      const doc = DATA.documents.find(x => x.document_id === state.docId);
      byId("docTitle").textContent = doc?.title || "Документ";
      if (!doc) { byId("textPane").textContent = ""; return; }
      const rows = matches().sort((a,b) => a.match.char_start - b.match.char_start || b.match.char_end - a.match.char_end);
      let cursor = 0, html = "";
      for (let i = 0; i < rows.length; i++) {
        const m = rows[i].match;
        if (m.char_start < cursor) continue;
        const spanIds = rows
          .filter(item => item.match.char_start === m.char_start && item.match.char_end === m.char_end)
          .map(item => item.raw_match_id);
        html += esc(doc.text.slice(cursor, m.char_start));
        html += `<mark class="${spanIds.includes(state.matchId) ? "active" : ""}" data-mark="${esc(spanIds[0])}">${esc(doc.text.slice(m.char_start, m.char_end))}</mark>`;
        cursor = m.char_end;
      }
      html += esc(doc.text.slice(cursor));
      byId("textPane").innerHTML = html;
      document.querySelectorAll("[data-mark]").forEach(mark => mark.onclick = () => {
        state.matchId = mark.dataset.mark; render();
      });
      document.querySelector("mark.active")?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
    function renderDetails() {
      const item = DATA.matches.find(x => x.raw_match_id === state.matchId);
      if (!item) { byId("detailsPane").innerHTML = "Выберите совпадение."; return; }
      const m = item.match, l = item.llm || {}, d = item.filter_decision || {};
      const context = `${m.context_before || ""}${m.matched_text || ""}${m.context_after || ""}`;
      byId("detailsPane").innerHTML = `
        <div class="context">${esc(context)}</div>
        <div class="kv">
          <div class="key">canonical</div><div class="value">${esc(m.canonical)}</div>
          <div class="key">matched</div><div class="value">${esc(m.matched_text)}</div>
          <div class="key">host_word</div><div class="value">${esc(m.word_text)}</div>
          <div class="key">span</div><div class="value">${m.char_start}..${m.char_end}</div>
          <div class="key">word_span</div><div class="value">${m.word_start}..${m.word_end}</div>
          <div class="key">inside_word</div><div class="value">${esc(m.inside_word)}</div>
          <div class="key">source</div><div class="value">${esc(m.source)}</div>
          <div class="key">filter</div><div class="value">${esc(d.action)} · ${esc(d.reason)}</div>
          <div class="key">label</div><div class="value">${esc(l.label)} · ${esc(l.confidence)}</div>
          <div class="key">УДД</div><div class="value">${esc(l.udd || l.evidence_level)}</div>
          <div class="key">УУР</div><div class="value">${esc(l.uur || l.recommendation_strength)}</div>
          <div class="key">reason</div><div class="value">${esc(l.reason_short)}</div>
        </div>
        <div class="toolbar">
          <button onclick="setAction('keep')">keep</button>
          <button onclick="setAction('reject')">reject</button>
          <button onclick="setAction('review')">review</button>
          <button onclick="downloadFilters()">export filters</button>
        </div>`;
    }
    function setAction(action) {
      const item = DATA.matches.find(x => x.raw_match_id === state.matchId);
      if (!item) return;
      item.filter_decision = item.filter_decision || {};
      item.filter_decision.action = action;
      item.filter_decision.term = item.match.canonical;
      item.filter_decision.host_word = item.match.word_text;
      render();
    }
    function downloadFilters() {
      const rows = [["term","host_word","action","reason","confidence","created_at"]];
      for (const item of DATA.matches) {
        const d = item.filter_decision || {}, m = item.match || {};
        rows.push([d.term || m.canonical, d.host_word || m.word_text, d.action || "review", d.reason || "dashboard_export", d.confidence || "", d.created_at || ""]);
      }
      const csv = rows.map(r => r.map(v => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
      const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "manual_filters.csv";
      a.click();
    }
    function render() { renderDocs(); renderMatches(); renderText(); renderDetails(); }
    initFilters(); render();
  </script>
</body>
</html>
"""
