from __future__ import annotations

LLM_AUDIT_DASHBOARD_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM audit dashboard</title>
<style>
:root{--left:330px;--right:380px;--header:86px;--border:#d8e0ea;--ink:#162033;--muted:#667085;--panel:#fff;--bg:#eef3f7;--blue:#2563eb;--green:#15803d;--red:#b42318;--amber:#b45309}
*{box-sizing:border-box}
body{margin:0;height:100vh;overflow:hidden;font:13px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink)}
button,input,select,textarea{font:inherit}
button{height:28px;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:#1f2937;padding:0 9px;cursor:pointer}
button:hover{background:#f1f5f9}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff;font-weight:650}
button.active{background:#dbeafe;border-color:#60a5fa;color:#1d4ed8;font-weight:700}
button.icon{width:30px;padding:0}
input,select,textarea{border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:#111827}
input,select{height:28px;padding:0 8px}
textarea{width:100%;min-height:56px;resize:vertical;padding:8px}
header{height:var(--header);display:grid;grid-template-rows:38px 1fr;gap:6px;padding:8px 10px;background:#fff;border-bottom:1px solid var(--border);box-shadow:0 1px 4px rgba(15,23,42,.06)}
.topbar{display:flex;align-items:center;gap:8px;min-width:0}
.search{width:260px}
.compact{width:96px}
.spacer{flex:1}
.metrics{display:flex;align-items:center;gap:6px;overflow:hidden}
.metric{min-width:84px;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc;padding:4px 7px}
.metric span{display:block;font-size:10px;color:var(--muted);line-height:1.05;text-transform:uppercase}
.metric b{display:block;font-size:13px;line-height:1.15;white-space:nowrap}
.metric.good b{color:var(--green)}.metric.bad b{color:var(--red)}
.meta{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
main{height:calc(100vh - var(--header));display:grid;grid-template-columns:var(--left) 6px minmax(360px,1fr) 6px var(--right);gap:0;padding:10px}
.pane{min-width:0;min-height:0;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.pane-title{height:34px;display:flex;align-items:center;gap:8px;padding:0 10px;border-bottom:1px solid #e5e7eb;background:#f8fafc;font-weight:700}
.pane-title small{font-weight:500;color:var(--muted)}
.pane-body{flex:1;min-height:0;overflow:auto}
.gutter{cursor:col-resize;background:transparent;position:relative}
.gutter::after{content:"";position:absolute;left:2px;top:8px;bottom:8px;width:2px;border-radius:2px;background:#cbd5e1}
.hidden{display:none!important}
.case-row{padding:10px;border-bottom:1px solid #edf2f7;cursor:pointer}
.case-row:hover{background:#f8fafc}
.case-row.active{background:#eff6ff;border-left:3px solid var(--blue);padding-left:7px}
.case-title{font-weight:700;line-height:1.25;margin-bottom:5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{max-width:100%;display:inline-block;padding:2px 6px;border-radius:5px;background:#e5e7eb;color:#374151;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chip.ok{background:#dcfce7;color:#166534}.chip.bad{background:#fee2e2;color:#991b1b}.chip.llm{background:#dbeafe;color:#1e40af}.chip.pending{background:#fef3c7;color:#92400e}.chip.partial{background:#ede9fe;color:#5b21b6}
.text-toolbar{height:34px;display:flex;align-items:center;gap:8px;padding:0 10px;border-bottom:1px solid #e5e7eb;background:#fff;color:var(--muted);font-size:12px;min-width:0}
.doc-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.open-inline{height:24px;font-size:12px;padding:0 8px;flex:0 0 auto}
.terms{display:flex;flex-wrap:wrap;gap:5px;padding:8px 10px;border-bottom:1px solid #e5e7eb;background:#fbfdff}
.term{display:inline-flex;gap:4px;align-items:center;border:1px solid #bfdbfe;background:#eff6ff;color:#1e3a8a;border-radius:999px;padding:3px 7px;font-size:11px}
.doc-text{padding:16px 18px;font:14px/1.68 Georgia,"Times New Roman",serif;white-space:pre-wrap;word-break:break-word}
.block-mark{background:#fff3bf;box-shadow:0 0 0 1px rgba(245,158,11,.22) inset}
.word-mark{background:#facc15;color:#111827;font-weight:800;box-shadow:0 0 0 2px rgba(202,138,4,.32)}
.evidence-mark{background:#dcfce7;color:#14532d;font-weight:700}
.right-scroll{flex:1;min-height:0;overflow-y:scroll;overflow-x:hidden;padding:10px;display:flex;flex-direction:column;gap:10px}
.review-card{border:1px solid #dbe3ee;border-radius:8px;background:#fff;overflow:hidden;flex:0 0 auto}
.review-head{padding:9px 10px;background:#f8fafc;border-bottom:1px solid #e5e7eb;display:grid;gap:4px}
.review-head b{font-size:13px}.review-head small{color:var(--muted)}
.review-body{padding:10px;display:grid;gap:10px}
.kv{display:grid;grid-template-columns:70px 1fr;gap:4px 8px;font-size:12px}
.kv span{color:var(--muted)}.kv b{font-weight:650}
.choice{display:grid;gap:5px}
.choice-label{font-size:11px;font-weight:700;color:#4b5563;text-transform:uppercase}
.choice-buttons{display:flex;flex-wrap:wrap;gap:5px}
.choice-buttons button{height:26px;font-size:12px;padding:0 8px;max-width:100%;white-space:nowrap}
.case-actions{display:flex;flex-wrap:wrap;gap:6px}
.case-actions button{height:26px;font-size:12px}
.case-actions button.danger{border-color:#fecaca;color:#991b1b;background:#fff7f7}
.case-actions button.warn{border-color:#fde68a;color:#92400e;background:#fffbeb}
.save-line{height:22px;color:var(--green);font-size:12px}
details{border-top:1px solid #e5e7eb;padding:8px 10px;background:#fbfdff}
summary{cursor:pointer;color:#475569;font-weight:650}
pre{margin:8px 0 0;max-height:220px;overflow:auto;background:#0f172a;color:#e5e7eb;border-radius:6px;padding:10px;font:11px/1.45 Consolas,Monaco,monospace}
.empty{padding:14px;color:var(--muted)}
.modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center}
.modal.active{display:flex}
.modal-content{background:#fff;border-radius:8px;max-width:600px;width:90%;max-height:80vh;display:flex;flex-direction:column;overflow:hidden}
.modal-header{padding:12px 16px;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center}
.modal-header h2{margin:0;font-size:16px}
.modal-close{width:28px;height:28px;border:none;background:transparent;cursor:pointer;font-size:20px;color:var(--muted)}
.modal-body{flex:1;overflow-y:auto;padding:12px}
.exclude-item{padding:8px;border:1px solid #e5e7eb;border-radius:6px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;background:#fbfdff}
.exclude-item-info{flex:1;font-size:12px}
.exclude-item-name{font-weight:600;margin-bottom:2px}
.exclude-item-meta{color:var(--muted);font-size:11px}
.exclude-item-btn{height:24px;font-size:11px;padding:0 6px}
@media(max-width:1000px){header{height:118px;grid-template-rows:auto auto}.topbar{flex-wrap:wrap}.search{width:100%}main{height:calc(100vh - 118px);padding:8px}.gutter{display:none}}
</style>
</head>
<body>
<header>
  <div class="topbar">
    <input id="q" class="search" placeholder="поиск: клинрека / препарат / block_id">
    <select id="statusFilter" class="compact">
      <option value="">all</option>
      <option value="llm">llm</option>
      <option value="bad">bad</option>
      <option value="ok">ok</option>
      <option value="partial">partial</option>
      <option value="pending">pending</option>
    </select>
    <input id="offset" class="compact" value="0" type="number" min="0">
    <input id="limit" class="compact" value="80" type="number" min="1" max="500">
    <button id="refresh" class="primary">Refresh</button>
    <button id="showExcludedPrep" class="icon" title="Исключённые препараты">X</button>
    <button id="showExcludedStats" class="icon" title="Исключены из статистики">S</button>
    <div class="spacer"></div>
    <button class="icon" data-toggle-pane="left" title="left">L</button>
    <button class="icon" data-toggle-pane="center" title="text">T</button>
    <button class="icon" data-toggle-pane="right" title="review">R</button>
  </div>
  <div class="metrics" id="metrics"></div>
</header>
<main id="layout">
  <section class="pane" id="leftPane">
    <div class="pane-title">Клинреки <small id="caseCount"></small></div>
    <div class="pane-body" id="cases"></div>
  </section>
  <div class="gutter" id="gutterLeft"></div>
  <section class="pane" id="centerPane">
    <div class="pane-title">Текст <small id="textMeta"></small></div>
    <div class="text-toolbar" id="textToolbar"><span class="doc-title">select block</span></div>
    <div class="terms" id="terms"></div>
    <div class="pane-body"><div class="doc-text" id="docText"></div></div>
  </section>
  <div class="gutter" id="gutterRight"></div>
  <aside class="pane" id="rightPane">
    <div class="pane-title">Оценка <small id="saveStatus"></small></div>
    <div class="right-scroll" id="review"></div>
  </aside>
</main>
<div id="excludedPrepModal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2>Исключённые препараты</h2>
      <button class="modal-close" onclick="closeModal('excludedPrepModal')">&times;</button>
    </div>
    <div class="modal-body" id="excludedPrepList"></div>
  </div>
</div>
<div id="excludedStatsModal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2>Исключены из статистики модели</h2>
      <button class="modal-close" onclick="closeModal('excludedStatsModal')">&times;</button>
    </div>
    <div class="modal-body" id="excludedStatsList"></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const CHOICES={
 label:['recommendation','contraindication','literature_mention','error','unclear'],
 target_kind:['drug','method','marker','other'],
 recommendation_strength:['','A','B','C'],
 evidence_level:['','1','2','3','4','5']
};
let state=null,cases=[],current=null,currentIndex=null,currentLink='',leftWidth=330,rightWidth=380,apiMoney='...',lastTotal=0;
let visible={left:true,center:true,right:true};
const debounceTimers=new Map();
async function api(path){const r=await fetch(path);if(!r.ok)throw new Error(await r.text());return r.json();}
async function apiPost(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());return r.json();}
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}
function pct(v){return typeof v==='number'?(v*100).toFixed(1)+'%':'-';}
function qs(){return new URLSearchParams({q:$('q').value,status:$('statusFilter').value,offset:$('offset').value,limit:$('limit').value});}
function metric(k,v,cls=''){return `<div class="metric ${cls}"><span>${esc(k)}</span><b>${esc(v)}</b></div>`}
async function refresh(){state=await api('/api/state');const page=await api('/api/cases?'+qs());cases=page.items;lastTotal=page.total;renderState(lastTotal);renderCases();if(cases.length&&(!current||!cases.some(c=>c.block_id===current.item.block_id)))await loadCase(cases[0].index);refreshKey().catch(()=>{});}
async function refreshKey(){const key=await api('/api/key');const data=key.data||{};apiMoney=typeof data.limit_remaining==='number'?`${data.limit_remaining.toFixed(2)} / ${data.limit??'-'}${data.stale?' stale':''}`:(key.error||'-');renderState(lastTotal||cases.length);}
function renderState(total){
 const s=state.score||{};
 $('metrics').innerHTML=[
  metric('completed',`${state.completed}/${state.review_block_count||state.gold_count}`),
  metric('gold',`${state.reviewed_case_count||0}/${state.review_case_count||state.gold_count}`),
  metric('label',pct(s.label_accuracy),scoreClass(s.label_accuracy)),
  metric('UUR',pct(s.recommendation_strength_accuracy),scoreClass(s.recommendation_strength_accuracy)),
  metric('UDD',pct(s.evidence_level_accuracy),scoreClass(s.evidence_level_accuracy)),
  metric('API $',apiMoney),
  `<div class="meta">${esc(state.model||'?')} | prompt=${esc(state.current_prompt_version||'-')} | shown=${cases.length}/${total??cases.length}</div>`
 ].join('');
}
function scoreClass(v){return typeof v==='number'?(v>=.8?'good':'bad'):'';}
function renderCases(){
 $('caseCount').textContent=`${cases.length}`;
 $('cases').innerHTML=cases.map(c=>{
  const active=current&&current.item&&current.item.block_id===c.block_id?' active':'';
  return `<div class="case-row${active}" data-index="${c.index}">
    <div class="case-title">#${c.index} ${esc(c.document_title||'')}</div>
    <div class="chips">
      <span class="chip ${esc(c.status)}">${esc(c.status)}</span>
      <span class="chip">cases=${esc(c.reviewed_count)}/${esc(c.case_count)}</span>
      <span class="chip">llm=${esc(c.pred_label)}</span>
      <span class="chip">gold=${esc(c.gold_label)}</span>
      ${(c.terms||[]).slice(0,3).map(t=>`<span class="chip">${esc(t)}</span>`).join('')}
    </div>
  </div>`;
 }).join('')||'<div class="empty">no rows</div>';
 document.querySelectorAll('.case-row[data-index]').forEach(el=>el.onclick=()=>loadCase(el.dataset.index));
}
function renderHighlighted(text,ctx){
 text=String(text||'');
 const spans=[];
 const add=(s,e,cls)=>{s=Number(s);e=Number(e);if(Number.isFinite(s)&&Number.isFinite(e)&&e>s){spans.push({s:Math.max(0,Math.min(text.length,s)),e:Math.max(0,Math.min(text.length,e)),cls});}};
 add(ctx.block_span_start,ctx.block_span_end,'block-mark');
 (ctx.case_spans||[]).forEach(x=>add(x.span_start,x.span_end,'word-mark'));
 (ctx.evidence_level_candidates||[]).forEach(x=>add(x.span_start,x.span_end,'evidence-mark'));
 const cuts=new Set([0,text.length]);
 spans.forEach(x=>{cuts.add(x.s);cuts.add(x.e);});
 const points=[...cuts].sort((a,b)=>a-b);
 let html='';
 for(let i=0;i<points.length-1;i++){
  const a=points[i],b=points[i+1],part=text.slice(a,b);
  if(!part)continue;
  const cls=[...new Set(spans.filter(x=>a>=x.s&&b<=x.e).map(x=>x.cls))].join(' ');
  html+=cls?`<span class="${cls}">${esc(part)}</span>`:esc(part);
 }
 return html;
}
async function loadCase(index){
 currentIndex=Number(index);
 current=await api('/api/case?index='+encodeURIComponent(index));
 currentLink=current.item.document_link||'';
 document.querySelectorAll('.case-row').forEach(x=>x.classList.toggle('active',Number(x.dataset.index)===currentIndex));
 renderText();
 renderReview();
}
function renderText(){
 const item=current.item,ctx=current.context||{};
 $('textMeta').textContent=`${item.block_id} | ${ctx.context_start??0}-${ctx.context_end??''}`;
 $('textToolbar').innerHTML=`<span class="doc-title">${esc(item.document_title||'')}</span>${item.document_link?'<button class="open-inline" id="openInline">Open clinrec</button>':''}`;
 const inline=$('openInline');if(inline)inline.onclick=()=>window.open(item.document_link,'_blank','noopener');
 $('terms').innerHTML=(item.primary_terms||[]).map(t=>`<span class="term">${esc(t.canonical)} <b>${esc(t.host_word)}</b>${t.inside_word?' inside':''}</span>`).join('')||'<span class="term">no terms</span>';
 $('docText').innerHTML=renderHighlighted(ctx.text||item.preview||'',ctx);
}
function renderReview(){
 const request=current.request_body||{};
 $('review').innerHTML=(current.cases||[]).map(caseCard).join('')+
  `<details><summary>Request JSON</summary><pre>${esc(JSON.stringify(request,null,2))}</pre></details>`;
 bindReviewEvents();
}
function caseCard(c){
 const g=c.gold||{},p=c.prediction||{},terms=(c.terms||[]).map(t=>`${t.canonical} [${t.host_word}]`).join(', ');
 const excludeStats=g.exclude_from_model_stats?'active':'';
 const excludedPrep=c.preparation_excluded?'active':'';
 return `<div class="review-card" data-case="${esc(c.case_id)}">
   <div class="review-head">
     <b>${esc(terms||c.case_id)}</b>
     <small>${esc(c.case_id)} | text="${esc(c.span&&c.span.text||'')}"</small>
   </div>
   <div class="review-body">
     <div class="kv">
       <span>LLM</span><b>${esc(p.label||'-')} / ${esc(p.target_kind||'-')} / ${esc(p.recommendation_strength||'-')}${esc(p.evidence_level||'-')}</b>
       <span>reason</span><b>${esc(p.reason||'-')}</b>
       <span>gold</span><b>${esc(g.label||'-')} / ${esc(g.target_kind||'-')} / ${esc(g.recommendation_strength||'-')}${esc(g.evidence_level||'-')}</b>
     </div>
     ${choiceGroup(c.case_id,'label','Label',CHOICES.label,g.label)}
     ${choiceGroup(c.case_id,'target_kind','Target',CHOICES.target_kind,g.target_kind)}
     ${choiceGroup(c.case_id,'recommendation_strength','UUR',CHOICES.recommendation_strength,g.recommendation_strength)}
     ${choiceGroup(c.case_id,'evidence_level','UDD',CHOICES.evidence_level,g.evidence_level)}
     <div class="case-actions">
       <button class="warn ${excludeStats}" data-action="exclude-stats">Не учитывать в статистике модели</button>
       <button class="danger ${excludedPrep}" data-action="exclude-prep">Исключить препарат из LLM списка</button>
     </div>
     <label class="choice"><span class="choice-label">Comment</span><textarea data-field="comment">${esc(g.comment||'')}</textarea></label>
     <label class="choice"><span class="choice-label">Quote</span><textarea data-field="evidence_quote">${esc(g.evidence_quote||'')}</textarea></label>
   </div>
 </div>`;
}
function choiceGroup(caseId,field,label,values,current){
 return `<div class="choice" data-field="${esc(field)}"><div class="choice-label">${esc(label)}</div><div class="choice-buttons">
  ${values.map(v=>`<button data-field="${esc(field)}" data-value="${esc(v)}" class="${String(current??'')===String(v)?'active':''}">${esc(v||'none')}</button>`).join('')}
 </div></div>`;
}
function bindReviewEvents(){
 document.querySelectorAll('.review-card button[data-field]').forEach(btn=>btn.onclick=()=>{
  const card=btn.closest('.review-card');
  card.querySelectorAll(`button[data-field="${btn.dataset.field}"]`).forEach(x=>x.classList.remove('active'));
  btn.classList.add('active');
  saveCard(card);
 });
 document.querySelectorAll('.review-card textarea').forEach(el=>el.oninput=()=>{
  const card=el.closest('.review-card'),key=card.dataset.case+':'+el.dataset.field;
  clearTimeout(debounceTimers.get(key));
  debounceTimers.set(key,setTimeout(()=>saveCard(card),450));
 });
 document.querySelectorAll('.review-card button[data-action="exclude-stats"]').forEach(btn=>btn.onclick=()=>{
  btn.classList.toggle('active');
  saveCard(btn.closest('.review-card'));
 });
 document.querySelectorAll('.review-card button[data-action="exclude-prep"]').forEach(btn=>btn.onclick=()=>excludePreparation(btn.closest('.review-card'),btn));
}
function collect(card){
 const value=field=>{const b=card.querySelector(`button[data-field="${field}"].active`);return b?b.dataset.value:'';};
 const text=field=>{const e=card.querySelector(`textarea[data-field="${field}"]`);return e?e.value:'';};
 return {block_id:current.item.block_id,case_id:card.dataset.case,label:value('label'),target_kind:value('target_kind'),
  recommendation_strength:value('recommendation_strength'),evidence_level:value('evidence_level'),
  comment:text('comment'),evidence_quote:text('evidence_quote'),
  exclude_from_model_stats:!!card.querySelector('button[data-action="exclude-stats"].active'),
  reviewer:'dashboard_manual'};
}
function currentCase(caseId){return (current.cases||[]).find(x=>x.case_id===caseId)||{};}
async function excludePreparation(card,btn){
 const c=currentCase(card.dataset.case),term=(c.terms||[])[0];
 if(!term)return;
 btn.classList.add('active');
 $('saveStatus').textContent='excluding';
 const result=await apiPost('/api/exclude-preparation',{term});
 state=result.state;
 $('saveStatus').textContent='excluded';
 await refresh();
}
async function saveCard(card){
 $('saveStatus').textContent='saving';
 try{
  const result=await apiPost('/api/gold',collect(card));
  state=result.state;
  $('saveStatus').textContent='saved';
  renderState(cases.length);
  setTimeout(()=>{if($('saveStatus').textContent==='saved')$('saveStatus').textContent='';},1200);
 }catch(e){
  $('saveStatus').textContent='error';
  console.error(e);
 }
}
function applyLayout(){
 const cols=[];
 $('leftPane').classList.toggle('hidden',!visible.left);
 $('centerPane').classList.toggle('hidden',!visible.center);
 $('rightPane').classList.toggle('hidden',!visible.right);
 $('gutterLeft').classList.toggle('hidden',!(visible.left&&visible.center));
 $('gutterRight').classList.toggle('hidden',!(visible.center&&visible.right));
 if(visible.left)cols.push(leftWidth+'px');
 if(visible.left&&visible.center)cols.push('6px');
 if(visible.center)cols.push('minmax(320px,1fr)');
 if(visible.center&&visible.right)cols.push('6px');
 if(visible.right)cols.push(rightWidth+'px');
 $('layout').style.gridTemplateColumns=cols.join(' ')||'1fr';
 document.querySelectorAll('[data-toggle-pane]').forEach(b=>b.classList.toggle('active',visible[b.dataset.togglePane]));
}
function startDrag(kind,e){
 e.preventDefault();
 const move=ev=>{
  if(kind==='left')leftWidth=Math.max(220,Math.min(720,ev.clientX-10));
  if(kind==='right')rightWidth=Math.max(280,Math.min(760,window.innerWidth-ev.clientX-10));
  document.documentElement.style.setProperty('--left',leftWidth+'px');
  document.documentElement.style.setProperty('--right',rightWidth+'px');
  applyLayout();
 };
 const up=()=>{window.removeEventListener('mousemove',move);window.removeEventListener('mouseup',up);};
 window.addEventListener('mousemove',move);window.addEventListener('mouseup',up);
}
$('refresh').onclick=()=>refresh().catch(showError);
$('showExcludedPrep').onclick=()=>{loadExcludedPreparations().then(()=>openModal('excludedPrepModal')).catch(showError);};
$('showExcludedStats').onclick=()=>{loadExcludedStats().then(()=>openModal('excludedStatsModal')).catch(showError);};
$('gutterLeft').onmousedown=e=>startDrag('left',e);
$('gutterRight').onmousedown=e=>startDrag('right',e);
document.querySelectorAll('[data-toggle-pane]').forEach(b=>b.onclick=()=>{visible[b.dataset.togglePane]=!visible[b.dataset.togglePane];applyLayout();});
document.querySelectorAll('input,select').forEach(el=>el.addEventListener('keydown',e=>{if(e.key==='Enter')refresh().catch(showError)}));
document.querySelectorAll('.modal').forEach(m=>m.onclick=e=>{if(e.target===m)closeModal(m.id);});
function showError(e){$('review').innerHTML=`<div class="empty">${esc(e.message)}</div>`;}
function openModal(id){const m=$(id);if(m)m.classList.add('active');}
function closeModal(id){const m=$(id);if(m)m.classList.remove('active');}
async function loadExcludedPreparations(){
 try{
  const data=await api('/api/excluded-preparations');
  const items=data.items||[];
  $('excludedPrepList').innerHTML=items.length?items.map(item=>`<div class="exclude-item">
   <div class="exclude-item-info">
    <div class="exclude-item-name">${esc(item.canonical)}</div>
    <div class="exclude-item-meta">${esc(item.host_word||'')} | ${esc(item.source||'')} | ${esc((item.created_at||'').substring(0,10))}</div>
   </div>
   <button class="exclude-item-btn danger" onclick="removeExcludedPreparation(${JSON.stringify(item).replace(/"/g,'&quot;')})">Удалить</button>
  </div>`).join(''):`<div class="empty">Исключённых препаратов нет</div>`;
 }catch(e){
  $('excludedPrepList').innerHTML=`<div class="empty">Ошибка: ${esc(e.message)}</div>`;
 }
}
async function loadExcludedStats(){
 try{
  const data=await api('/api/excluded-stats');
  const items=data.items||[];
  $('excludedStatsList').innerHTML=items.length?items.map(item=>`<div class="exclude-item">
   <div class="exclude-item-info">
    <div class="exclude-item-name">${esc(item.canonical)}</div>
    <div class="exclude-item-meta">case: ${esc(item.case_id)} | block: ${esc(item.block_id)} | ${esc(item.label||'')} | ${esc((item.reviewed_at||'').substring(0,10))}</div>
   </div>
   <button class="exclude-item-btn warn" onclick="removeExcludedStats({block_id:'${item.block_id}',case_id:'${item.case_id}'})">Удалить</button>
  </div>`).join(''):`<div class="empty">Нет элементов исключённых из статистики</div>`;
 }catch(e){
  $('excludedStatsList').innerHTML=`<div class="empty">Ошибка: ${esc(e.message)}</div>`;
 }
}
async function removeExcludedPreparation(item){
 if(!confirm(`Удалить "${item.canonical}" из исключённых?`))return;
 try{
  await apiPost('/api/remove-excluded-preparation',{term:item});
  await loadExcludedPreparations();
  await refresh();
 }catch(e){
  alert('Ошибка: '+e.message);
 }
}
async function removeExcludedStats(item){
 if(!confirm('Удалить из исключённых из статистики?'))return;
 try{
  await apiPost('/api/remove-exclude-from-stats',item);
  await loadExcludedStats();
  await refresh();
 }catch(e){
  alert('Ошибка: '+e.message);
 }
}
applyLayout();
refresh().catch(showError);
</script>
</body>
</html>
"""
