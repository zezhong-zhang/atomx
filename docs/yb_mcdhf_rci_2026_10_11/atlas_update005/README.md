本轮只在既有 Sites checkout 的 dist 准备增量，没有注册、推送、部署或改变托管身份。原 v5 的 16 个完整模型对象、64 行对比、全部 6019 gzip 和 hosting 文件逐项保持；新增两个原生相关 MCDHF 末态，总计 18 模型72行。原始前像见 [prechange_manifest.json](prechange_manifest.json)，全部旧模型原值见 [old_models.json](old_models.json)。

新增单一 [末态与轨迹 overlay](/home/zzhang/.local/share/atomx-nist-atlas/site/dist/updates/yb-correlated29-run003-endpoint16.json)，仅需现有页面一层加载。run003 预声明末态 accepted16，全部17点；同一 overlay 再按 [独立追加合同](append_run004/contract.json) 追加 run004 local64/global80，全部65点。两个轮次共享 global16 起点（NPZ SHA 和五态能量相同），共有81个不同global点，80个更新；不把共享起点重复计为新更新，不按最小 gap 选择中间点。run003 对象与17点轨迹在第二次追加中保持完全相同。将来消费另一轮必须提供明确、已闭合 completion 路径及新前瞻合同；当前没有自动发现或消费运行中结果。

两个模型都使用 f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe 源、原核 no-pair 七个子空间、3451 坐标度量、全部29轨道、完整五目标扇区1046有符号CSF，以及原五态物理UID与0.2等权。2163-CSF父空间其余扇区未计算；不认证全谱或全局基态。这里是69电子 Yb II，Z70 Fermi A175 的DC；没有拼接旧静态/动态Breit、QED或反冲，也没有外部后端。旧单点构造的核场/屏蔽虚层模型仍以原身份留存，不改写为本轮自洽末态。

run003 1175运行控制通过，27H/27CI/17g，16个接受步；EOL −14067.40011728723 → −14067.413113908482 Ha，gap RMS 2048.36463776004 → 2203.8622162949996 cm⁻¹。末态残差0.04090991741227255、能量指示量0.003408179199100851 Ha，原1e−7/1e−8 Ha门均失败。116.915610200027秒是未受控墙钟时间。

run004 接续上述原末态，4201运行控制通过，98H/98CI/65g，再64个接受步；EOL再降0.009743582446390064 Ha，gap RMS降至2105.098759492958 cm⁻¹，仍高于最初2048.36463776004。末态残差0.002641370231623412、指示量1.174884123374752e−5 Ha仍失败；411.9384027328342秒是未受控墙钟时间。每轮14 h/S为7初始化与7CI内部调用。全部16/64个已接受步的保存H精确接受守卫通过；每轮11项闭合检查属于元数据与保存Fraction账核验，不是新H/CI/g独立重放，也不认证物理完备性、实验精度或软件/并行加速。run004 closure001的变量遮蔽失败作为元数据历史另附，实际科学控制全过且仍未驻值。

页面展示每点五态绝对能量、同点6s下态激发能、四线gap、原参考重叠、EOL/残差/指示量、D/P重心及精细分裂。UID与NIST唯一level_id相配；不确定度未给出时保留null，显示为空，不补零。原四线1%比较均失败。每轮末态−本轮起点的上态、下态及净谱线变化独立相减；保存manifest中的逐步delta不冒充全轮累计delta。

本地数据/保全29项检查和桌面/手机58项浏览器检查通过。第一次浏览器58项亦通过，但手机曲线文字偏小，随后仅加横向滚动并重新核验；旧浏览器记录与截图保留为 attempt001。全轨迹、14个来源下载哈希、旧16模型值、缓存重访、新overlay读取失败时保留原16模型64行、手机表格滚动、缺失σ显示和零JS错误均核对。现有Playwright/Chromium使用本地loopback；服务已停止。详情见 [data_verification.json](data_verification.json)、[browser_result.json](browser_result.json)、[browser_execution.json](browser_execution.json) 及 [desktop.png](desktop.png)、[mobile.png](mobile.png)。准备/核验没有Atomx或NumPy导入，没有新H/CI/g/Dirac/谱/优化、NPZ数值重放或外部数据拉取。

发布由root另行审查处理，本目录不持有或读取发布凭据。最终站点资产哈希见 [site_manifest.json](site_manifest.json)。两个已执行的保存JSON builder保留原字节与其运行哈希；Ruff只报两处嵌套if的SIM102格式提示，没有为排版改写已闭合builder。JS语法检查和两种核验通过。
