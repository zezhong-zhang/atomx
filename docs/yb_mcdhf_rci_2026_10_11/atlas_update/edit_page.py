"""Bounded page edit with a separate current-model payload, no NIST rewrite."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path("/home/zzhang/.local/share/atomx-nist-atlas/site")
page = (HERE / "original_index.html").read_text()

addition = r'''
function currentSummary(update){
if(!update)return '';
const rows=update.models.map(m=>({model:m.title,csfs:m.rci_csf_count,rows:m.rows,passed:m.rows.filter(r=>r.goal_pass).length}));
const by=(r,label)=>r.rows.find(s=>s.label===label);
const moments=update.models.map(m=>({model:m.title,d:m.multiplets.find(x=>x.label==='5d'),p:m.multiplets.find(x=>x.label==='6p')}));
const metric=update.stationarity;
const probe=update.latest_stationarity_probe;
const probeNote=probe?`<p class="warning" data-probe-status="nonstationary">531-CSF 扩展混合CI驻值探针未过：同一原metric残差 ${fmt(probe.baseline_residual)} → 一步 ${fmt(probe.one_step_residual)}，均大于 ${esc(probe.orbital_tolerance.toExponential(1))}。这一步仅DC EOL，平均总能下降 ${fmt(probe.EOL_delta_Ha)} Ha，却未改善四条实验gap；不是完整扩展 MCDHF，也不修改这里的固定005轨道静态Breit谱。<a href="${esc(probe.result_asset)}">查看独立诊断来源</a></p>`:'';
return `<section class="current-summary"><h3>当前主线：原生 MCDHF → RCI</h3><p>五参考共同轨道 MCDHF 已通过原驻值门：${esc(metric.orbital_residual.toExponential(3))} &lt; ${esc(metric.orbital_tolerance.toExponential(1))}，68步；这只认证5-CSF数值基线。</p><p class="warning">294 → 468 → 531 CSF 均在同24个已驻值轨道上做固定轨道 RCI。扩展空间的 MCDHF 驻值尚未认证；不是完整核—价关联，也没有通过整体实验精度门。</p>${probeNote}<p>完整4f族使6s基态降能约815 cm⁻¹；继续加入5p使差能改变304–808 cm⁻¹，5s为15.8–34.2 cm⁻¹。四条最终谱线仍与本地ASD相差约1303–2129 cm⁻¹。上下态必须在同空间和同算符下相减。</p><p class="muted">数据来源：2026-10-11实际运行 · 全部有符号CI根与独立重放通过 · ${esc(update.nuclear_scope)} 每个模型可展开查看边界、态权重和来源。</p><h3>同一6s零点下的四条激发能（cm⁻¹）</h3>${table(rows,[['模型',r=>esc(r.model),'wrap'],['RCI CSF',num('csfs')],['D3/2',r=>fmt(by(r,'5d_3/2').calculated_cm1)],['D5/2',r=>fmt(by(r,'5d_5/2').calculated_cm1)],['P1/2',r=>fmt(by(r,'6p_1/2').calculated_cm1)],['P3/2',r=>fmt(by(r,'6p_3/2').calculated_cm1)],['原1%比较',r=>`${r.passed}/4，通过数不代表整体认证`,'wrap']])}<p class="muted">本地唯一评估 ASD：D3/2 22960.80；D5/2 24332.69；P1/2 27061.82；P3/2 30392.23 cm⁻¹。参考不确定度未给出时保留为空，不补成零。</p><details><summary>比较重心与精细分裂</summary>${table(moments,[['模型',text('model'),'wrap'],['D重心',r=>fmt(r.d.centroid_cm1)],['D分裂',r=>fmt(r.d.splitting_cm1)],['P重心',r=>fmt(r.p.centroid_cm1)],['P分裂',r=>fmt(r.p.splitting_cm1)]])}<p class="muted">按2J+1加权：D为4:6，P为2:4。本地ASD的D分裂1371.89、P分裂3330.41 cm⁻¹；重心和分裂分别判断。</p></details><p><a href="updates/yb-mcdhf-stage002.json" download>下载本轮计算对照 JSON</a> · <a href="evidence/yb-mcdhf-stage002/stage002.json">完整来源绑定汇总</a></p></section>`;
}
function modelMoments(m){
if(!m.multiplets)return '';
const rows=m.multiplets.flatMap(x=>[{label:`${x.label} 重心`,calculated:x.centroid_cm1,reference:x.reference_centroid_cm1,error:x.centroid_gap_cm1},{label:`${x.label} 分裂`,calculated:x.splitting_cm1,reference:x.reference_splitting_cm1,error:x.splitting_gap_cm1}]);
return `<h3>重心 / 分裂（cm⁻¹）</h3>${table(rows,[['量',text('label')],['计算',num('calculated')],['本地ASD',num('reference')],['差',num('error')]])}`;
}
function models(){
const list=payload.models||[];
if(!list.length){$('#content').innerHTML='<p class="empty">当前谱种没有接入本页面的近期相关计算。可查看参考档案和历史计算；历史缓存不代表当前版本认证。</p>';return}
const updateWarning=payload.native_update_error?`<p class="warning">近期 MCDHF 对照读取失败：${esc(payload.native_update_error)}。以下历史模型继续保留，不能冒充当前结果。</p>`:'';
$('#content').innerHTML=updateWarning+currentSummary(payload.native_update)+list.map(m=>{
const rs=m.rows,max=Math.max(1,...rs.filter(r=>Number.isFinite(r.error_percent)).map(r=>Math.abs(r.error_percent)));
const bars=rs.map((r,i)=>{const y=12+i*40;if(!Number.isFinite(r.error_percent))return `<text x="10" y="${y+17}" fill="#9aacbf" font-size="12">${esc(r.configuration)} · 2J=${r.two_j}：未评分</text>`;const x=330+(r.error_percent<0?-Math.abs(r.error_percent)/max*250:0),w=Math.abs(r.error_percent)/max*250;return `<text x="10" y="${y+17}" fill="#9aacbf" font-size="12">${esc(r.configuration)} · 2J=${r.two_j}</text><rect x="${x}" y="${y}" width="${w}" height="22" rx="3" fill="${r.goal_pass?'#60cbb0':'#ff8691'}"/><text x="${r.error_percent>=0?340+w:340}" y="${y+16}" fill="#e4edf8" font-size="12">${fmt(r.error_percent)}%</text>`}).join('');
const status=m.model_kind==='native_mcdhf_baseline'?'五参考MCDHF驻值通过':m.model_kind==='fixed_orbital_rci'?'固定轨道RCI · 扩展驻值未认证':'历史独立模型';
const lineTable=table(rs,[['组态 / 2J',r=>esc(`${r.configuration} / ${r.two_j}`)],['计算 cm⁻¹',num('calculated_cm1')],['本地 ASD cm⁻¹',num('reference_cm1')],['差 cm⁻¹',num('error_cm1')],['差 %',num('error_percent')],['ASD σ cm⁻¹',num('uncertainty_cm1')],['ASD−Handbook',num('asd_minus_handbook_cm1')],['原目标',r=>r.matching!=='unique_nominal_branch'?'未匹配':`<span class="${r.goal_pass?'good':'bad'}">${r.goal_pass?'通过':'未通过'}</span>`],...(m.model_kind?[['bare连接权重',r=>fmt(r.bare_character_weight)]]:[])]).replace('<table>','<table class="model-lines">');
const shifts=m.model_kind==='fixed_orbital_rci'?`<h3>相对前一空间 ${esc(m.previous_stage)} 的上下态增量</h3><p>同算符6s下态能移：${fmt(m.ground.energy_delta_from_previous_Ha)} Ha；上态与下态能移相减后才是谱线增量。</p>${table(rs,[['上态',text('label')],['上态 ΔHa',num('energy_delta_from_previous_Ha')],['谱线 Δcm⁻¹（前一空间）',num('transition_delta_from_previous_cm1')],['谱线 Δcm⁻¹（五参考）',num('transition_delta_from_bare_cm1')]])}`:'';
return `<div class="model" data-model-id="${esc(m.id||m.saved_run)}" data-model-kind="${esc(m.model_kind||'historical')}"><details ${m.is_current?'open':''}><summary><strong>${esc(m.title)}</strong><span class="badge">${esc(status)}</span></summary><p class="muted">${esc(m.scope)}</p>${m.stationarity_note?`<p>${esc(m.stationarity_note)}</p>`:''}<p>原名义比较目标：${Math.round(rs[0].relative_goal*100)}%；${rs.filter(r=>r.goal_pass).length}/${rs.length} 通过。整体实验精度与最高水平尚未认证。</p><svg class="chart" viewBox="0 0 780 180" role="img" aria-label="计算与本地 ASD 激发能的相对误差"><line x1="330" y1="0" x2="330" y2="175" stroke="#4a5b6e"/>${bars}</svg>${lineTable}${modelMoments(m)}${shifts}<p class="source">保存运行：${esc(m.saved_run)}<br>证据 SHA256：${esc(m.sha256)}${m.source_asset?`<br><a href="${esc(m.source_asset)}">查看已保存的完整计算证据</a>`:''}${m.verification_asset?` · <a href="${esc(m.verification_asset)}">独立重放证据</a>`:''}<br>${esc(m.boundary_note||'匹配为名义参考连接分支；尚未认证大空间归一化态的组态权重和 dressed 跃迁率。')}</p></details></div>`;
}).join('');
}
'''

start, end = page.index("function models(){"), page.index("function sources(){")
page = page[:start] + addition + page[end:]
needle = "payload=cache.get(ion.asset)}page=0;draw();"
overlay = """payload=cache.get(ion.asset)}
if(element.z===70&&ion.charge===1){
try{const asset='updates/yb-mcdhf-stage002.json';if(!cache.has(asset)){const current=await fetch(asset);if(!current.ok)throw Error('HTTP '+current.status);cache.set(asset,await current.json());if(cache.size>6)cache.delete(cache.keys().next().value)}if(serial!==request)return;const current=cache.get(asset);payload={...payload,models:[...current.models,...(payload.models||[])],native_update:current}}
catch(error){if(serial!==request)return;payload={...payload,native_update_error:error.message}}
}
page=0;draw();"""
assert page.count(needle) == 1
page = page.replace(needle, overlay)
needle = "function sources(){const ss=payload.sources||[];"
assert page.count(needle) == 1
page = page.replace(needle, """function sources(){const ss=payload.sources||[];const currentSources=payload.native_update?`<h3>近期 MCDHF → RCI 证据链</h3><p>五参考驻值与扩展固定轨道RCI分别认证；JSON包含完整模型边界与同24径向轨道来源。旧NIST载荷和历史模型未改写。</p>${payload.native_update.source_files.map(s=>`<p class="source"><a href="${esc(s.asset)}">${esc(s.path)}</a><br>SHA256 ${esc(s.sha256)}</p>`).join('')}`:'';""")
needle = "<p class=\"source\">档案快照 SHA256：${esc(index.provenance.snapshot_sha256)}"
assert page.count(needle) == 1
page = page.replace(needle, "${currentSources}" + needle)
needle = "[hidden]{display:none!important}"
page = page.replace(needle, ".current-summary{margin:10px 0 22px;padding:16px;border:1px solid #27516a;border-radius:10px;background:#142838}.current-summary h3{color:var(--blue)}.model>details{padding:0}.model summary{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.model summary strong{font-size:17px}.model .badge{white-space:normal}" + needle)
(SITE / "dist/index.html").write_text(page)
print("Updated current-model summary, eight model details and selected-ion overlay loader")
