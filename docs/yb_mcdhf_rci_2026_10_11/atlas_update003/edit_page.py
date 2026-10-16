"""Apply a single declared local presentation increment to the exact v3 page."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / "contract.json").read_text())
site = Path(contract["site"])
original = (HERE / "original_index.html").read_text()
if hashlib.sha256((site / "dist/index.html").read_bytes()).hexdigest() != contract["original_page_sha256"]:
    raise ValueError("Exact contracted v3 page required")


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected unique page preimage: " + old)
    return text.replace(old, new, 1)


page = once(original, "function endpointSummary(update){", (HERE / "iteration_renderer.js").read_text() + "\nfunction endpointSummary(update){")
page = once(page, "最新公共531端点：DC → 同点静态 Breit RCI", "此前160点：DC → 同点静态 Breit RCI")
page = once(page, "const endpointWarning=payload.endpoint_update_error?", "const iterationWarning=payload.iteration_update_error?`<p class=\"warning\">最新两项对照读取失败：${esc(payload.iteration_update_error)}。此前12模型48行仍保留，不能冒充新的320点或双激发结果。</p>`:'';\nconst endpointWarning=payload.endpoint_update_error?")
page = once(page, "$('#content').innerHTML=endpointWarning+updateWarning+endpointSummary(payload.endpoint_update)+", "$('#content').innerHTML=iterationWarning+endpointWarning+updateWarning+iterationSummary(payload.iteration_update)+endpointSummary(payload.endpoint_update)+")
page = once(page, "const status=m.model_kind==='nonstationary_endpoint_rci'?", "const status=m.model_kind==='nonstationary_native_mcdhf_endpoint'?'320步DC · 原驻值未过':m.model_kind==='fixed_nonstationary_rank_rci'?'固定160点RCI · 扩展驻值未计算':m.model_kind==='nonstationary_endpoint_rci'?")
page = once(page, "m.model_kind==='nonstationary_endpoint_rci'?[['原physical参考重叠²'", "['nonstationary_endpoint_rci','nonstationary_native_mcdhf_endpoint','fixed_nonstationary_rank_rci'].includes(m.model_kind)?[['原physical参考重叠²'")
page = once(page, "${shifts}${endpointShifts(m)}", "${shifts}${endpointShifts(m)}${iterationShifts(m)}")
page = once(page, "function sources(){const ss=payload.sources||[];", "function sources(){const ss=payload.sources||[];const iterationSources=payload.iteration_update?`<h3>320点DC与另一个固定160点双激发：分别绑定证据</h3><p>原驻值失败；两个点不拼接。数值重放与独立保存数组复核不等于实验精度或完整物理认证。</p>${payload.iteration_update.source_files.map(s=>`<p class=\"source\"><a href=\"${esc(s.asset)}\">${esc(s.path)}</a><br>SHA256 ${esc(s.sha256)}</p>`).join('')}`:'';")
page = once(page, "${endpointSources}${currentSources}<p class=\"source\">档案快照", "${iterationSources}${endpointSources}${currentSources}<p class=\"source\">档案快照")
old = "catch(error){if(serial!==request)return;payload={...payload,endpoint_update_error:error.message}}\n}"
new = "catch(error){if(serial!==request)return;payload={...payload,endpoint_update_error:error.message}}\ntry{const asset='updates/yb-public531-endpoint320.json';if(!cache.has(asset)){const iteration=await fetch(asset);if(!iteration.ok)throw Error('HTTP '+iteration.status);cache.set(asset,await iteration.json());if(cache.size>6)cache.delete(cache.keys().next().value)}if(serial!==request)return;const iteration=cache.get(asset);payload={...payload,models:[...iteration.models,...(payload.models||[])],iteration_update:iteration}}\ncatch(error){if(serial!==request)return;payload={...payload,iteration_update_error:error.message}}\n}"
page = once(page, old, new)
(site / "dist/index.html").write_text(page)
(HERE / "render_source.js").write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])
print(json.dumps({"page_sha256": hashlib.sha256((site / 'dist/index.html').read_bytes()).hexdigest(), "old_overlay_unchanged": True}))
