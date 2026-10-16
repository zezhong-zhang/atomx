"""Apply one reversible local presentation increment to exact frozen v4."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / "contract.json").read_text())
site = Path(contract["site"])
original = (HERE / "original_index.html").read_text()
if hashlib.sha256((site / "dist/index.html").read_bytes()).hexdigest() != contract["original_page_sha256"]:
    raise ValueError("Exact contracted v4 page required")


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected unique page preimage: " + old)
    return text.replace(old, new, 1)


page = once(original, "function iterationSummary(update){", (HERE / "virtual_renderer.js").read_text() + "\nfunction iterationSummary(update){")
page = once(page, "最新两项判别：轨道松弛与5p双激发分别评估", "此前两项判别：轨道松弛与5p双激发分别评估")
page = once(page, "当前共14模型56行", "该归档阶段共14模型56行")
page = once(page, "const iterationWarning=payload.iteration_update_error?", "const virtualWarning=payload.virtual_update_error?`<p class=\"warning\">最新虚层对照读取失败：${esc(payload.virtual_update_error)}。此前14模型56行仍保留，不能冒充新增的核场或屏蔽场结果。</p>`:'';\nconst iterationWarning=payload.iteration_update_error?")
page = once(page, "$('#content').innerHTML=iterationWarning+endpointWarning+updateWarning+iterationSummary(payload.iteration_update)+", "$('#content').innerHTML=virtualWarning+iterationWarning+endpointWarning+updateWarning+virtualSummary(payload.virtual_update)+iterationSummary(payload.iteration_update)+")
page = once(page, "const status=m.model_kind==='nonstationary_native_mcdhf_endpoint'?", "const status=m.model_kind==='fixed_nonstationary_virtual_rci'?'固定160点种子RCI · 29轨道未自洽':m.model_kind==='nonstationary_native_mcdhf_endpoint'?")
page = once(page, "'fixed_nonstationary_rank_rci'].includes(m.model_kind)", "'fixed_nonstationary_rank_rci','fixed_nonstationary_virtual_rci'].includes(m.model_kind)")
page = once(page, "${iterationShifts(m)}", "${iterationShifts(m)}${virtualShifts(m)}")
page = once(page, "function sources(){const ss=payload.sources||[];", "function sources(){const ss=payload.sources||[];const virtualSources=payload.virtual_update?`<h3>固定160点种子：核场与闭核68屏蔽场虚层</h3><p>同一原24轨道、原no-pair子空间和DC算符；新增5轨道未自洽优化，扩展驻值未计算。两个完整有限目标扇区对照分别绑定证据，无跨点修正拼接。</p>${payload.virtual_update.source_files.map(s=>`<p class=\"source\"><a href=\"${esc(s.asset)}\">${esc(s.path)}</a><br>SHA256 ${esc(s.sha256)}</p>`).join('')}`:'';")
page = once(page, "${iterationSources}${endpointSources}${currentSources}", "${virtualSources}${iterationSources}${endpointSources}${currentSources}")
old = "catch(error){if(serial!==request)return;payload={...payload,iteration_update_error:error.message}}\n}"
new = "catch(error){if(serial!==request)return;payload={...payload,iteration_update_error:error.message}}\ntry{const asset='updates/yb-fixed160-virtual-fields.json';if(!cache.has(asset)){const update=await fetch(asset);if(!update.ok)throw Error('HTTP '+update.status);cache.set(asset,await update.json());if(cache.size>6)cache.delete(cache.keys().next().value)}if(serial!==request)return;const update=cache.get(asset);payload={...payload,models:[...update.models,...(payload.models||[])],virtual_update:update}}\ncatch(error){if(serial!==request)return;payload={...payload,virtual_update_error:error.message}}\n}"
page = once(page, old, new)
(site / "dist/index.html").write_text(page)
(HERE / "render_source.js").write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])
print(json.dumps({"page_sha256": hashlib.sha256((site / 'dist/index.html').read_bytes()).hexdigest(), "old_data_unchanged": True}))
