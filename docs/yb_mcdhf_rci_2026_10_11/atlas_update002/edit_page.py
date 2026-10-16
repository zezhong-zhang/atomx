"""Add a separate endpoint overlay without modifying the prior model ledger."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "contract.json").read_text())
SITE = Path(CONTRACT["site"])
original = (HERE / "original_index.html").read_text()
if hashlib.sha256((SITE / "dist/index.html").read_bytes()).hexdigest() != CONTRACT["original_page_sha256"]:
    raise ValueError("site page changed after prospective contract")


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("expected exact single preimage: " + old)
    return text.replace(old, new, 1)


page = once(original, "function currentSummary(update){", (HERE / "endpoint_renderer.js").read_text() + "\nfunction currentSummary(update){")
page = once(page, "<h3>当前主线：原生 MCDHF → RCI</h3>", "<h3>五参考基线与固定005轨道判别（保留原模型）</h3>")
page = once(page, "const updateWarning=payload.native_update_error?", "const endpointWarning=payload.endpoint_update_error?`<p class=\"warning\">公共531端点对照读取失败：${esc(payload.endpoint_update_error)}。旧模型仍保留，不能冒充这个端点的结果。</p>`:'';\nconst updateWarning=payload.native_update_error?")
page = once(page, "$('#content').innerHTML=updateWarning+currentSummary(payload.native_update)+list.map(m=>{", "$('#content').innerHTML=endpointWarning+updateWarning+endpointSummary(payload.endpoint_update)+currentSummary(payload.native_update)+list.map(m=>{")
page = once(page, "const status=m.model_kind==='native_mcdhf_baseline'?", "const status=m.model_kind==='nonstationary_endpoint_rci'?'公共531端点 · 原DC驻值未过':m.model_kind==='native_mcdhf_baseline'?")
page = once(page, "...(m.model_kind?[['bare连接权重',r=>fmt(r.bare_character_weight)]]:[])", "...(m.model_kind==='nonstationary_endpoint_rci'?[['原physical参考重叠²',num('physical_reference_overlap_squared')]]:m.model_kind?[['bare连接权重',r=>fmt(r.bare_character_weight)]]:[])")
page = once(page, "${lineTable}${modelMoments(m)}${shifts}", "${lineTable}${modelMoments(m)}${shifts}${endpointShifts(m)}")
page = once(page, "function sources(){const ss=payload.sources||[];", "function sources(){const ss=payload.sources||[];const endpointSources=payload.endpoint_update?`<h3>公共531 accepted160 同点证据链</h3><p>原DC驻值失败；完整RCI和独立保存数组复核不等于轨道驻值或实验认证。</p>${payload.endpoint_update.source_files.map(s=>`<p class=\"source\"><a href=\"${esc(s.asset)}\">${esc(s.path)}</a><br>SHA256 ${esc(s.sha256)}</p>`).join('')}`:'';")
page = once(page, "${currentSources}<p class=\"source\">档案快照", "${endpointSources}${currentSources}<p class=\"source\">档案快照")
old = "catch(error){if(serial!==request)return;payload={...payload,native_update_error:error.message}}\n}"
new = "catch(error){if(serial!==request)return;payload={...payload,native_update_error:error.message}}\ntry{const asset='updates/yb-public531-endpoint160.json';if(!cache.has(asset)){const endpoint=await fetch(asset);if(!endpoint.ok)throw Error('HTTP '+endpoint.status);cache.set(asset,await endpoint.json());if(cache.size>6)cache.delete(cache.keys().next().value)}if(serial!==request)return;const endpoint=cache.get(asset);payload={...payload,models:[...endpoint.models,...(payload.models||[])],endpoint_update:endpoint}}\ncatch(error){if(serial!==request)return;payload={...payload,endpoint_update_error:error.message}}\n}"
page = once(page, old, new)
(SITE / "dist/index.html").write_text(page)
(HERE / "render_source.js").write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])
print(json.dumps({"page_sha256": hashlib.sha256((SITE / 'dist/index.html').read_bytes()).hexdigest(), "old_overlay_unchanged": True}))
