# Atomx 原生原子物理路线与成功基线

2026-10-11 用户裁定：在 Atomx 内实现原子物理与求解器；pCI、AMBiT、EXP-T、
GRASP、FAC 用于参考方程和独立比较物理、精度、效率。接入其他求解器不作为
原生能力完成标准。高精度、高效率、高复用、模块化和架构自洽同时约束推进。

## 已有成功实践与勘误

上一轮仅列早期 Ne n12 与 Eu 裸 CI，漏掉后期成功结果。早期模型误差不能
代表最佳实践。修订前原文及来源保存在[快照](previous_sources/manifest.json)，
旧科学数值和失败门保持。

| 原生实践 | 已保存成功结果 | 参考与边界 |
|---|---|---|
| Ne⁷⁺ q/r Kα 卫星线，完整3200Ha声明池 | 908.193003810 / 908.065718918 eV；计算−2024实验 −8.896190 / −13.881082 meV | 未拟合；Togawa2024 Table I，非NIST；原算符残差<1e−10Ha；局部1meV相关收敛与完整极点认证未闭合 |
| Eu³⁺ / Eu IV，原生[Xe]4f⁶多重态/晶场 | ⁵D₀/₁/₂=17275.5 / 19015.4 / 21508.9 cm⁻¹；历史目标标定RMS8.2cm⁻¹；全M3003行列式 | Pr先验Slater标定；原脚本声明Carnall aquo-ion目标，日志NIST标题不替代来源；本地NIST半经验[17270]/[19030]/[21510]，显示值比较RMS9.03cm⁻¹；不是三个独立实测验证 |

Ne 的[实际分析](../xfel_tail_breit_pole_2026_10_06/e3200/analysis.json)、
[实验原件身份](../xfel_tail_breit_pole_2026_10_06/experimental_nominals.json)和
[完整模型](../xfel_tail_breit_pole_2026_10_06/README.md)记录291径向轨道，
g/q/r各78,791/114,698/78,790 CSF；²⁰Ne Fermi核、原生DKB/DF轨道族、DC、
完整静态Breit、QEDMOD模型表的同轨道应用、NMS、SMS。原生signed CSF组装
与tracked Davidson求能，没有pCI/EXP-T/FAC/GRASP代求此谱。QED模型表来源与
原生求解分别记录，不将模型表改称完整从头QED。

Eu 的[成功日志](../review_2026_10_02/codex_fix_validation/eu-tier-a.log)、
[全M晶场日志](../review_2026_10_02/codex_fix_validation/eu-cf.log)、
[目标声明](../../scripts/eu3_tierA_anchor.py)和[来源复核](eu_baseline/evidence.json)
提供原生DF→CI→标定→晶场链。未拟合⁷F的旧目标与当前NIST有不同的行，
需重新匹配。标定成功验证多重态、算符、响应和工程复用；第一性精度推进用
独立、未拟合的能级与性质检验新增机制，不抹去已有成功。

## 已有模块与当前真实依赖

原生CI/MCDHF复用 `methods/csf` 轨道能量/密度梯度、公共角代数和径向核。
`physics/csf/sparse_ci.py` 已有Numba CPU有符号稀疏装配与无竞争分块并行。
`linearized_sd.solve` 已有完整有限标量linked SD；`coupled_cluster.compute`
已有精确幂零指数的有限非线性CCSD，均为有限正确性锚，不是规模化CCSDT。

`JCoupledSDInteraction` 已携带DC/static/dynamic J块、Fock、冻结常数及参考能；
`CorePairLadders` 已有source、PP、正规序HH及linked energy，明确缺Fock、
singles、ring、mixed-rank。它们是原生补图的起点。`FiniteCIModel` 已有同H的
P/Q波算符、物理度量与定向有效算符，可复用但不无验证替代CC度量与左右态。

当前Yb大空间 `CoreSD → ReducedMonovalentSD` 仍调用pCI-derived图项库，
`fscc.compute` 仍运行EXP-T。这些既有路径及证据作为参考/对照保留，依赖
如实说明，不重新贴全链原生标签。逐项原生实现、公共入口验收后再迁移生产。

## 当前六方向并行合同

后续用户选择 **Atomx原生MCDHF→RCI** 为原子结构与Yb示例主线，
当前数值结果、物理预算和失败见[MCDHF权威入口](../yb_mcdhf_rci_2026_10_11/README.md)。
此前SD/CC开发与对照证据保留；它们不替代当前Yb的原生轨道优化和RCI验收。

| 方向 | Atomx内实现 | 独立验收/对照 |
|---|---|---|
| 1 共同轨道MCDHF | 完整混合CI的EOL目标、原no-pair投影与度量、真实态响应和稳定优化 | 完整signed梯度、原驻值门、同点H/CI/数组重放；能量稳定不替代驻值 |
| 2 平衡核价与多参考 | 完整允许核空穴家族、双激发与共同虚轨道空间，保留上下态身份 | 固定点差分判别后再优化扩展模型；完整目标块、signed嵌套H及重心/分裂 |
| 3 同态完整RCI与物理算符 | static/dynamic transverse、QED模型响应、反冲、核模型账 | 同点上下态重解；不跨轨道/源版本拼接；有限实频Breit不等于完整束缚态QED |
| 4 性质与响应 | 同波函数范数、密度、E1/E2/M1、寿命及谱 | 相位/干涉、长度/速度约定与相关响应；历史寿命失败独立保留 |
| 5 实验与独立软件比较 | NIST状态匹配、周期表及FAC/GRASP/AMBiT等对照 | Ne实验与Eu标定分列；绝对能/重心/分裂/强度/计时分别验收；软件不作生产后端 |
| 6 CPU/GPU/复用 | 共同物理和数组合同、Numba CPU/CUDA核、分块收缩/批积分/矩阵自由作用 | 同模型原门、冷/暖及端到端配对计时；真实加速比/并行效率；FAC等仅独立基准 |

方向可并行开发独立模块，生产按依赖准入；实际计时用既有排他锁防互扰。
GPU不修补缺失方程，密集有限锚不成为生产表示；不为每个元素或参考程序
另造一套物理后端。FSCC按近闭壳适用性选择，Ne/Eu成功CI路线继续保留。

## 本轮原生判别与选择机制

[linked SD测前合同](sd_linked001/contract.json)固定已有四电子、12spinor、
495完整行列式、2T₁+10T₂坐标以及DC/static/dynamic真实输入。以公共原生
有限方程和独立Fermi作用检查所有source/K/energy，辨别PP/HH覆盖与缺项；
原signed1e−11Ha、root1e−12Ha门不变。不扩空间、不拟合、不调用pCI/EXP-T。
结果、覆盖和失败见[原生判别目录](sd_linked001/)，不替代生产Yb谱验收。

[独立正常序PH复核](sd_ph_review001/README.md)已用同H的完整费米作用重现
DD剩余项：三模式全部300个signed矩阵元误差≤1.244e−14Ha，排除了Fock/HH
双计数解释。下一项优先补原生PH及混合图。旧全块部分图集残差使用Fock的
QQ投影，不能叫完整Fock交换子残差；这个边界不影响完整linked SD与DD结论。

复用Ne的完整Q反馈作物理筛选：1600→3200Ha跃迁增量约+5.9meV，新增singles
仅约+0.0145meV；上下态电子对耦合决定差分。Eu的原生核极化共同能移约
−0.42Ha，而⁵D−⁷F只−52.6cm⁻¹，见[有效算符记录](../CI_MBPT_SIGMA2_SPEC.md)。
先测上下态/项间重心/分裂/混合，不能用共同总能下降解释光谱缺口。

[每小时进度](../atomx_hourly_progress_2026_10_10/progress.json)改为原生六方向，
Ne直接重读后期JSON，Eu重读来源绑定的标定/本地NIST复核；保留Yb旧五态
实验失败。发卡仍触发有界选择器；开发方向、已注册executor、实际完成分开，
未注册新生产executor时如实显示实现前沿。Yb已完成原生531-CSF公共MCDHF
累计320接受步与同点RCI判别，原驻值和实验门仍失败；没有生产或最高水准认证。
