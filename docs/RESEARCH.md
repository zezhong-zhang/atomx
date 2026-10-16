# Research, design records and evidence

Use [current guides](README.md) for supported interfaces and
[testing](TESTING.md) for executable checks. Reports retain their recorded
source, model, inputs and failures; their test counts do not describe the current
checkout. A later report supersedes an earlier conclusion only where it states
that relationship explicitly.

## Campaigns and deliveries

- [Sn12+ EUV 原生 CI 饱和](TIN_EUV_BASELINE.md)：论文空间已实算至{6}，115715个完整CI根；同阶裸CI三质心最大差0.0567eV，{3}→{6}的2→1红移5.4157eV。分块存储与精确角向复用；核心MBPT、相关算符及实验验收仍未完成。
- [EUV 光源能量闭环](EUV_SOURCE_MODEL.md)：全能级 Sn12+ 条件基线保留；新增多电荷态 Saha EOS、显式连续谱、分层辐射、临界层 Drude 波动、光滑流体与热传导。新增显式共振非LTE定容源，191项相关检查通过；真实Sn速率、非LTE空间耦合、固体烧蚀及实验CE尚未验收。条件32eV带内已存线功率约11.8%来自候选共振上态。原生谱峰13.1096nm仍偏蓝，绝对CE参考已纳入2025标定勘误。
- [积分驱动原子全阶 CCSD / FS-CCSDT](ATOMIC_FSCC.md)：当前接口、完整能窗/核分区、外部后端与资源约定；[电子方程锚与 Yb/Ne 实算](atomic_allorder_fscc_2026_10_09/README.md)保存同积分完整 CI、非对角复旋转、扩窗和失败证据。 [按需积分与尾部续篇](atomic_streaming_tail_2026_10_09/README.md)记录完整原生 Yb 能窗、分块导出与扩大虚池；后端全能窗求解仍受限；[精确轴向/宇称公共分块](atomic_pci_sd_native_2026_10_09/README.md#exp-t完整三激发的精确轴向宇称分块控制)有限19项检查通过，仍保留未压缩资源预检。
- [原生标量线性化 SD](ATOMIC_LINEARIZED_SD.md)：有限 DC/DCB/冻结壳锚、直接约化 DC 核 SD及公共固定能量价SD/Σ₁、完整定向Σ₂与隐式响应；[原生积分与原始 pCI 图类续篇](atomic_pci_sd_native_2026_10_09/README.md)保留非对角 Fock、独立振幅核对和 Ne 完整 s/p/d/f 能窗实算，全核414虚态Yb核方程已解，完整价列/屏蔽的维度布局已修复，后续原生栈问题正在同空间验证；已解Σ₂有限全向量门、有限CI真实Bloch度量/性质锚通过，生产SD的CI能量/度量消费与实验精度待完成。[此前交换子与 pCI 相位锚](atomic_reduced_sd_2026_10_09/README.md)保持封存。
- [有限 CCSD、完整声明 SD 与 Breit 相关响应](yb_sd_breit_completion_2026_10_09/README.md)：当前能力增量；真实非线性有限 CCSD、完整 NR-SD 预检/消费及全部 P/Q 的静态 Breit 已实现，119 项相关测试通过。Yb⁺ 二空穴五态和严格受限 C–B 二阶控制完成，最大谱位误差9.904%，实验/空间门未通过；后续原子 CCSD/FS-CCSDT 的实现状态见本页首项；大虚空间 Breit 尾部仍待完成。

- [AMBiT 同积分核对与有限核心空穴高阶传播](ci_mbpt_matched_qspace_2026_10_09/README.md)：独立执行两个二阶图类与非对角项；原生互补 P/Q 消元及 Yb⁺ 五态 4f 空穴探针完成。Q 相互作用与只重求能量参数已分辨，数值门通过；本轮有限空间未缩小实验谱位差距，该封存阶段未实现全阶 SD/FS-CCSDT；后续实现见本页首项。
- [pCI、AMBiT、EXP-T 源码与 Atomx 有效算符升级](ci_fscc_reference_upgrade_2026_10_09/README.md)：固定版本下载与构建、源码方法覆盖、带符号矩阵 Σ₁ 原生 CI 注入、冻结芯层/P–Q 身份守卫、Davidson 小空间修复；十个旧 Yb 矩阵的消费重放通过，未新增高阶或实验精度结果。

- [Yb⁺ 非对角二阶自能与轨道响应](yb_matrix_self_energy_2026_10_08/README.md)：前序矩阵实现与实验专题；完整带符号矩阵 Σ₂、有限外部空间 Dyson 系数/残重及独立非对角锚，99 项相关测试通过。两套完整五态及矩阵重放完成；细基最大谱位偏差 5.868%，实验及空间门未通过。全阶 SD、平衡空穴与 Breit 交叉项仍是后续模型。
- [Yb⁺ 文献诊断与修复路径](yb_literature_repair_strategy_2026_10_08/README.md)：封存文献策略；原始论文对照支持全阶 SD 与平衡空穴模型，识别百 cm⁻¹ 的二阶 Coulomb–Breit 交叉项；区分 Brueckner/CCSD 的束缚能和谱位误差，核对当时 Atomx scaffold 与消费覆盖。该文献记录未实算高阶谱线；最新能力以本页首项及能力指南为准。
- [Yb⁺ 相关尾部与径向表示](yb_correlation_tail_2026_10_08/README.md)：前序标量诊断；两组完整五态（粗基 ℓ≤8/200 Ha、细基 ℓ≤6/200 Ha）及共同空间控制，最后分波/细基正能增量 39.16/17.58 cm⁻¹，共同 ℓ6/200 的径向响应 2.30 cm⁻¹；数值门通过，相关收敛与 5% 实验目标仍未通过；独立 ASD 电离阈值锚分辨基态和上态的绝对偏差。
- [Yb⁺ 能量自洽与相关尾部](yb_energy_dependent_sigma_2026_10_08/README.md)：上一阶段封存证据；离壳自能、DF 连续分支 Dyson 根与 CI 注入，137 项测试通过，能量反馈和空间收敛分别检验。
- [Yb⁺ 谱线偏差定位与修复续报](yb_diagnosis_repair_2026_10_08/README.md)：前一阶段封存证据；独立 DF/偶极/电流/Breit 对照、外部单价电子在壳二阶自能、完整 ℓ≤6/50 Ha 窗口与 CI 注入、QED 解析回退告警。
- [稀土 Yb⁺ 谱线实算与实验对照](rare_earth_yb_spectrum_2026_10_08/README.md)：5 个能级、7 条目标线和 2 个直接寿命对照；冻结核数值控制通过、实验精度目标未通过，保存 Breit、4f 空穴与 QED 后端敏感性及失败证据。
- [原子、分子、晶体谱学与未来产业研究体系](spectroscopy_science_story_2026_10_06/README.md)：2026-10-06 封存的宏观故事线、12 个案例 setup、QED/XAS/材料模型选择及三个已运行接口例子；完整科学案例仍为计划，当前软件能力以能力指南及本页最新增量为准。此前 [Ne 精密谱学参赛项目书](agentic_science_beijing_light_source_2026_10_06/README.md)保留为专题支持材料。
- [Electronic-structure provider contract](adr/ADR-0001-electronic-structure-provider.md).
- [Historical reports](archive/README.md): dated physics investigations and campaign evidence.
- [Test physics audit](test_audit/README.md): reference independence and coverage assessment.
- [FAC benchmark report](../benchmarks/REPORT.md).
- [External verification](../benchmarks/EXTERNAL_VERIFICATION.md).

## Method surveys and dated investigations

- [SD expansion survey](SD_EXPANSION_SURVEY.md): literature and implementation status at its recorded review.
- [CI+MBPT screening specification](CI_MBPT_SIGMA2_SPEC.md): diagram contracts and retained negative results.
- [Performance investigation](PERFORMANCE_PLAN.md): dated measurements and proposed work; rerun before claiming a current speedup.
- [Numerics research](HPC_NUMERICS_RESEARCH_2026-09-27.md): entry point to the archived investigation.
- [Positioning and roadmap](POSITIONING_AND_ROADMAP.md): research direction; capabilities are governed by the current capability guide.
- [External-code benchmark campaign](BENCHMARKS_EXTERNAL.md): recorded matched-model comparisons and limits.
- [Scattering/FAC investigation](scattering_fac_verification_2026-07.md): historical scattering evidence.
- [Validation history](VALIDATION_HISTORY.md): source-specific test records moved out of the current test instructions.
- [CI repair, 10 October 2026](ci_repair_2026_10_10/README.md): failing runs, reproductions and repair validation.

Local evidence paths shown as code are retained navigation hints. Those artifacts
are outside the tracked repository and are unavailable in a clean GitHub checkout.
They are not replaced by different models or regenerated silently.
