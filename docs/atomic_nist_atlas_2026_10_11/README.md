# 原子级 NIST 周期表与实际迭代

[远程周期表](https://atomx-atomic-spectrum-atlas.black-fairy-6027.chatgpt.site)
已发布为 owner-private 站点，需要登录对应 ChatGPT 账号。
[部署记录](deployment.json)保留实际 site/version/deployment 标识、源提交和成功状态。
网站源码是单独的展示快照仓库；未提交、重置或推送 Atomx 科学工作区的修改。

版本 2 已加入 [Atomx 原生 Yb⁺ MCDHF→RCI 对照](../yb_mcdhf_rci_2026_10_11/atlas_update/PUBLICATION.md)：
8 组当前模型、32 行结果，以及单独的 531-CSF 未驻值诊断；旧 SD 模型和全部
NIST 压缩档案保留。下面的初版导出与部署记录仍描述原始快照，不认证后继模型。

本地 SQLite 备份为 ASD v5.12 mirror 加已保存的官方扩展，原始构建时间
2026-09-26；不是实时最新数据库。完整保留选定档案的 119394 条能级、296998
条谱线、6019 条电离能和全部 7021 个 species 条目；参考含评估、理论、半经验
及缺失值，不统称直接实验。118 个元素全部呈现；6019 个谱种具有记录载荷。

[导出证据](export003/export.json)、[全记录验证](export003/validation.json)与
[无损压缩证明](packaging004.json)保留数量、来源和缺口。原始 source-count 表有
四个声明行数差异（Br IV、Nd II、Au I、Au II），页面显示其存在。
历史 CSV 的 119394 条状态记录也保留，其中295条标为唯一能级比较；
组态平均、模糊匹配、失败和待计算项不会转成零误差或精度通过。

Yb II 的 DC-SD 与冻结 CI + 动态横向交换分别读取已有模型结果，并按唯一
评估组态／J／宇称及共同 6s 零点重匹配本地 ASD。四个 ASD 能量与旧 Handbook
锚一致；原 SD 1% 门仅一态通过，原冻结动态模型 5% 门也仅一态通过。
这些目标不代表最高水平；未认证组态权重、dressed 强度或寿命。

网站提供能级、实测／Ritz 谱线、各自不确定度、介质、Aki、IE、历史状态、
来源 URL/SHA256、分页、搜索和完整解压后的 JSON 下载。按离子加载 gzip；
6019 个载荷共29370645字节，压缩前402 MB 全部数据无损保留。
现代浏览器需支持 DecompressionStream；最多缓存六个所选离子。

初始失败均保留：[零点评分错误](export_failure001.json)、
[全量内存上限失败](export_failure002.json)。采用逐离子流式处理后通过；
部署遵循平台256 MiB展开归档上限，正确配置为 `static.directory: dist`。
实际打包大小和源身份以部署记录为准。

[浏览器验证](browser_result.json)覆盖118元素、Yb八条独立模型比较、分页、
实测筛选、来源哈希、元素搜索和手机无数据状态；没有 JavaScript 错误。
`test_nist_atlas_export.py`验证模糊、理论、错误J／宇称不能评分；
`test_atomic_iteration_queue.py`验证重复、失败、耗尽和中断恢复，共9项通过。
当前操作说明归入 [NIST guide](../NIST_SPECTRUM_ATLAS.md)，
发卡实际计算归入 [小时迭代](../atomx_hourly_progress_2026_10_10/README.md)。

网页是发布快照，后续计算由小时卡片报告。仍未达到原子级最高水准：
完整非 canonical FS-CCSDT 方程修复、原生全阶 SD 的完整横向图项、
五态共同核—价尾部收敛和有效性质算符仍在待办中。
