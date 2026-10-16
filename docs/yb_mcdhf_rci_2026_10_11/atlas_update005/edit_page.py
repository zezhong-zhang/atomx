"""Add a reversible append-only correlated-endpoint presentation to the exact v5 page."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / "contract.json").read_text())
site = Path(contract["site"])
original = (HERE / "original_index.html").read_text()
if hashlib.sha256((site / "dist/index.html").read_bytes()).hexdigest() != contract["original_page_sha256"]:
    raise ValueError("Exact contracted v5 page required")


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected unique page preimage: " + old)
    return text.replace(old, new, 1)


page = once(original, "function virtualSummary(update){", (HERE / "correlated_renderer.js").read_text() + "\nfunction virtualSummary(update){")
page = once(page, "最新物理判别：核场与闭核68屏蔽场虚轨道", "此前固定轨道判别：核场与闭核68屏蔽场虚轨道")
page = once(page, "此前14模型56行原值保留，当前共16模型64行。", "此前14模型56行原值保留，该归档阶段共16模型64行。")
page = once(page, "下一步需验证相关轨道自洽MCDHF的完整CI响应，再判断实验gap。", "这一历史对照本身未进行相关轨道自洽优化；后继MCDHF按独立末态模型展示，实验gap仍须逐项判断。")
page = once(page, "const virtualWarning=payload.virtual_update_error?", "const correlatedWarning=payload.correlated_update_error?`<p class=\"warning\">原生29轨道MCDHF末态读取失败：${esc(payload.correlated_update_error)}。此前16模型64行继续保留，不冒充新的轨道更新结果。</p>`:'';\nconst virtualWarning=payload.virtual_update_error?")
page = once(page, "$('#content').innerHTML=virtualWarning+", "$('#content').innerHTML=correlatedWarning+correlatedSummary(payload.correlated_update)+virtualWarning+")
page = once(page, "const status=m.model_kind==='fixed_nonstationary_virtual_rci'?", "const status=m.model_kind==='nonstationary_correlated29_mcdhf_endpoint'?`${m.accepted_steps}步29轨道MCDHF · 原驻值未过`:m.model_kind==='fixed_nonstationary_virtual_rci'?")
page = once(page, "'fixed_nonstationary_virtual_rci'].includes(m.model_kind)", "'fixed_nonstationary_virtual_rci','nonstationary_correlated29_mcdhf_endpoint'].includes(m.model_kind)")
page = once(page, "${virtualShifts(m)}", "${virtualShifts(m)}${correlatedShifts(m)}")
page = once(page, "${m.verification_asset?", "${m.completion_asset?` · <a href=\"${esc(m.completion_asset)}\">闭合元数据（不是独立数值重放）</a>`:''}${m.verification_asset?")
page = once(page, "function sources(){const ss=payload.sources||[];", "function sources(){const ss=payload.sources||[];const correlatedSources=payload.correlated_update?`<h3>原生29轨道相关MCDHF：已闭合末态与完整轨迹</h3><p>各轮末态按声明步数选取，原驻值及实验门失败；运行数值控制与元数据闭环不认证完整物理。所有上、下态均在本轮同点相减，不移植旧修正。</p>${payload.correlated_update.source_files.map(s=>`<p class=\"source\"><a href=\"${esc(s.asset)}\">${esc(s.path)}</a><br>SHA256 ${esc(s.sha256)}</p>`).join('')}`:'';")
page = once(page, "${virtualSources}${iterationSources}", "${correlatedSources}${virtualSources}${iterationSources}")
old = "catch(error){if(serial!==request)return;payload={...payload,virtual_update_error:error.message}}\n}"
new = "catch(error){if(serial!==request)return;payload={...payload,virtual_update_error:error.message}}\ntry{const asset='updates/yb-correlated29-run003-endpoint16.json';if(!cache.has(asset)){const update=await fetch(asset);if(!update.ok)throw Error('HTTP '+update.status);cache.set(asset,await update.json());if(cache.size>6)cache.delete(cache.keys().next().value)}if(serial!==request)return;const update=cache.get(asset);payload={...payload,models:[...update.models,...(payload.models||[])],correlated_update:update}}\ncatch(error){if(serial!==request)return;payload={...payload,correlated_update_error:error.message}}\n}"
page = once(page, old, new)
(site / "dist/index.html").write_text(page)
(HERE / "render_source.js").write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])
print(json.dumps({"page_sha256": hashlib.sha256((site / 'dist/index.html').read_bytes()).hexdigest(), "prior_model_data_unchanged": True}))
