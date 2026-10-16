# XFEL 原子动力学：核心模态、SOTA、前沿挑战与 Agent 探索路径

> 综合技术文档 · 2026-10-09
> 主体基于 atomx 仓库 2026-10-02 → 10-08 XFEL 战役实测记录
> （`docs/XFEL_GAP_ANALYSIS.md` 49 节、`docs/xfel_*` 60+ 战役目录、
> `src/atomx/xfel/` 19 模块 5441 行、820 项通过测试），
> 外部 SOTA 经独立文献调查。atomx 侧所有数字为实测、可复放；
> 外部论断附出处；未确证处显式标注。

---

## 摘要

XFEL（x 射线自由电子激光）把原子物理推进到"一个飞秒脉冲内吸收数十个光子、逐壳层剥离到 +25 电荷态"的强场 regime，且 2022–2025 年已从非相干速率物理跨入**内壳层相干动力学**（亚飞秒 Rabi cycling 已在 LCLS 观测）。本领域现有 SOTA 分裂为两半：以 **XATOM/XMDYN**（非相对论 HFS，亿级过程 MC）为代表的"快但物理近似"，与以高精度结构代码 + 手工工作流为代表的"准但慢、不可交互"。核心空白是——**没有一个代码同时具备相对论精度（Breit/QED）、多电子共振（REXMI）、相干强场动力学与交互式速度**。

agentx 的 XFEL 战役用 agentic-science 方法（测量驱动、闸门文化、参考源码锚定）在两周内建成了这条缺失的链路：从第一性 MCDHF/CI 结构 → 速率表/MC → 密度矩阵相干求解 → 成像观测量，并在一个 Ne q/r 共振谱位上把计算-实验差距从 **911 meV 收敛到 -8.9 meV**（未认证 meV 级）。**AI 自动化 XFEL 原子动力学是已确证的空白点**——现有 AI 工作集中于光束线调谐，原子 kinetics 工作流的 agent 化尚无已发表论文。

---

## 1. XFEL 原子物理的核心模态（physics modalities）

XFEL 脉冲（10–100 fs、10¹²–10¹⁸ photons/μm²、keV 光子）与原子相互作用，本质上是以下六个模态的耦合：

### 1.1 序贯多光子电离级联（sequential multiphoton ionization）
脉冲内原子逐次吸收光子；每吸收/衰变一次，电子结构为新组态重解，全部速率（光电离 PI、Auger、荧光）重算。这是 XATOM `track_elec_config` 的核心（Young et al., Nature 466, 56 (2010)；Son–Young–Santra, PRA 83, 033402）。
- **基准实验**：Young 2010 用 800 eV XFEL 打到 Ne，观测到高达 Ne⁸⁺ 且强度继续升高时电离**饱和**（x 射线透明度）——这是"frustrated absorption"（受挫吸收）的第一个定量证据，也是整个领域的金标准 benchmark。

### 1.2 逐电子蒙特卡洛（MC over the pulse）
对脉冲做时间分 bin，每个 bin 内按 `P = fluence·σ/n_bins` 抽样吸收、按速率比抽样 Auger 分支。`xatom_mc` / `xatom_mc_fly` 是 XATOM 的实现；这是处理多光子饱和/幸存者再吸收的唯一正确方式（单遍确定论迭代会在任意非零流量下错误地电离一切）。

### 1.3 共振增强多光子电离（REXMI / resonance-enabled X-ray MPI）
当光子能量扫过内壳层共振，平滑-σ 级联会严重低估电离。REXMI 是 XFEL 区别于普通强场的标志模态：
- Rudek et al., Nat. Photon. 6, 858 (2012) 提出并用于 Xe → Xe²¹⁺；
- LaForge et al., PRL 127, 213202 (2021) 在欧洲 XFEL 用 SASE 倍频实现类-REMPI 通道，把 Ar 推到 Ar¹⁷⁺；
- Son et al., Nat. Commun. (2023, arXiv:2303.07942) 系统预言**多芯空穴共振谱**：单条 MC 轨迹可含 10 次光电离 + 4 次共振激发 + 15 次 Auger + 3 次荧光，达 +25 电荷态（"massively hollow atoms"）。

### 1.4 次级通道（secondary channels）
直接双电离 / shakeoff（轨道突变重叠）、光电子诱导的碰撞电离（EII）、Compton 散射。这些是荷态分布高电荷尾巴的决定项，却几乎不可第一性计算（XATOM 的 shakeoff 用拟合数据）。

### 1.5 瞬态组态成像观测量（transient-config observables）
弹性散射 f0(q)、反常 f1/f2、每瞬态荷态的衍射信号——单粒子成像（SPI）的直接输入。XATOM 有 `xatom_imaging`。

### 1.6 相干强场动力学（coherent strong-field dynamics）
当 Rabi 频率超过 Auger 线宽，非相干速率物理失效，必须解密度矩阵/Maxwell–Bloch：
- Nandi et al., Nature 608, 488 (2022) 在 FERMI（23.7 eV, He）首次观测 XFEL Rabi 振荡（Autler–Townes 分裂）；
- **Linker et al., Nat. Photon. (2025)** 在 LCLS 观测到 Mn Kα（5.9 keV）、Cu Kα（8.05 keV）内壳层放大中的**亚飞秒 Rabi cycling**，Autler–Townes 分裂 >5 eV、Rabi 周期 <0.86 fs，3D Maxwell–Bloch 模拟确认。这是内壳层相干动力学从预测走向观测的转折点。

---

## 2. SOTA 代码全景与固有张力

XFEL-原子建模的 SOTA 分裂为几个家族，各有不可兼得的长板：

### 2.1 XATOM / XMDYN（Santra 组，DESY/CFEL）——事实标准
- **结构**：非相对论 Hartree-Fock-Slater（HFS）+ 数值网格连续态，~ms/组态。
- **动力学**：速率方程、`xatom_mc/_fly`（MC+MD 耦合，`XMDYN` 树算法 O(N log N)，模拟过 14 万原子核糖体，Stransky et al., Molecules 27, 4206 (2022)）。
- **规模**：Xe 类重原子可处理 ~2000 万多重空穴组态、数十亿 x 射线诱导过程。
- **短板**：非相对论、组态平均、无 Breit/QED、共振处理为孤立线（无 Fano 干涉）、相干动力学仅能唯象叠加。Son et al., PRR 2, 023053 (2020) 自己指出 hollow-shell 态导致 frustrated absorption 在重原子/硬 x 射线区失效。

### 2.2 FAC（Gu）——相对论原子数据引擎
相对论 DFA/CI，自带 `SetCascade`（Auger 级联）与 CRM。但**不做脉冲时间分辨电离动力学**——它输出截面/速率，作为 XATOM、XMDYN 或 CRM 的原子数据后端。FAC 只用零频 Breit 近似（无频率相关推迟，arXiv:1907.07403）；**没有任何 XFEL kinetics 代码在速率网络中系统包含 QED 自能**。

### 2.3 NeSSie（Hau-Riege, LANL）
顺序+非顺序（shake-up/off、双光子）电离模型，是 Rudek 2012/2013 Kr REXMI 分析的理论工具；近年活跃度低。

### 2.4 等离子体 kinetics 代码族（CRETIN / FLYCHK / SCFLY / SCRAM）
面向稠密等离子体（Vinko et al., Nature 482, 59 (2012) 固体密度诊断），超组态近似；对单原子 fs 脉冲的非顺序/共振通道处理有限，非 Maxwell 电子分布是公认短板。

### 2.5 固有张力（the SOTA gap）
把 2.1–2.4 的能力拼起来，会得到一个清晰的能力矩阵——**每一列都缺一角**：

| 能力 | XATOM | FAC | 等离子体族 | 高精度结构 |
|---|---|---|---|---|
| 相对论结构（Breit/QED） | —（非相对论） | ✓（零频 Breit，无 QED） | 近似 | ✓（GRASP2K 级） |
| 组态平均速度 | ~ms/组态 | s | 快 | 分钟（B-spline SCF） |
| REXMI 共振网络 | ✓（孤立线） | 数据 | — | — |
| 相干强场（密度矩阵） | 唯象 | — | — | — |
| 完整光电离/Auger/荧光速率链 | ✓ | ✓ | 部分 | 有模块未接 |

**这个矩阵的空白格（相对论 × 快 × 共振 × 相干）就是本领域的核心 SOTA 缺口。** atomx XFEL 战役正是冲着这个空白格建的。

---

## 3. atomx 的 XFEL 链路（填补 SOTA 空白）

两周内（2026-10-02 → 10-08）从 2 个文件 737 行扩到 **19 个模块 5441 行**，分五层，全部有实测数字：

### 3.1 速率层（rate layer）——真速率图
- **134 个独立弛豫 Gu jj 组态**（含零占据空穴轨道），原生子能量差；图上 **378 条占据 PI 边 + 250 条非零 Auger 边闭合**（`build_atomic_rate_table` 为生产入口）。
- 9 个中子 Ne PI 点在 1028/1714/3428 eV 对归档 FAC 参考闭合到 **0.0343%**；原生 Ne KLL 总量 = 1.0521 × 新鲜 XATOM 参考。
- **MC 编译加速 72×**（`CompiledRateTable`：速率表预编译为组态索引 + CSR 转移数组，Numba 执行，保持原随机流），Ne 单点 21.5–39.7×。

### 3.2 动力学层（kinetics）——序贯级联 + MC
- `cascade()`（确定论单遍）与 `cascade_mc()`（逐光子 MC，时间分 bin）。
- 对 vendored XATOM 二进制的 **Young-2010 电荷-流量锚全 7 点闭合**（F=1…1000 a.u.，比值 0.59–1.20，F≥30 在 1.5× 带内）。
- **关键物理诊断**：F=1000 时把 PI 阈改为轨道能量，平均电荷从 4.83 → 5.98（XATOM 5.93）——主导分歧是**近阈值约定**，为后续 REXMI 工作指明方向。这是"测量压倒推理"原则的范例。

### 3.3 相干层（coherent）——密度矩阵求解器（SOTA 中稀有）
- `CoherentSystem`：显式磁态密度矩阵 + Lindblad jump 算符 + 高斯脉冲复场包络。
- 真实 Ne⁹⁺（1s/2s/2p）：1022.13 eV、10 fs、1e17 ph/cm² 下观测相干激发与饱和，峰值 Rabi 能 0.424 eV，密度矩阵迹闭合到 3.2e-15。
- **多电子 CI 相干**（`MCDHFCoherentModel`）：共享正交轨道基上有符号 E1 振幅；真实两电子 Ne⁸⁺ 给 5 个 CI 能级、13 个磁态，主线 922.87 eV（A=9.40e12/s），10 fs 下激发峰布居 0.98。
- **两电子 CI 连续损失**（`TwoElectronCoherentModel`）：N=2→1→0，有符号 Dyson PI + 完整 Dirac–Coulomb Auger 振幅——这是向相干 REXMI 电离迈出的多电子步。
- 对照：XATOM 系无内建密度矩阵求解器；Linker 2025 用的是外置 Maxwell–Bloch。atomx 把相干动力学**原生接进了相对论结构层**。

### 3.4 谱位精度层（meV 战役）
Ne VIII（Ne⁷⁺）q/r 内壳层共振谱位 vs 实验：
- 实验锚：2001 Wargelin EBIT 混线 + **2024 Togawa 分辨 q/r**（908.2019/908.0796 eV，±0.005 eV 级）。
- 10-04：完整 n12 联合弛豫离实验 911/805 meV；
- 10-06（完整 3200 Ha 尾部 + Breit 约定 + 复极点判别后）：**q/r 计算 − 实验 = -8.9 / -13.9 meV**——两个数量级的收敛。
- 剩余 ~5–14 meV 缺口已归因到 rank2 连续极点（未认证）、跨模型 Breit 分项差（~4–5 meV，GRASP 系约定差异）与 q−r 分裂的 p₁/₂–p₃/₂ 径向/装配审计。
- **诚实边界**：局部 1 meV 增量门与物理极点认证**仍未闭合**，不宣称已达 meV 级认证。

### 3.5 成像与观测量层
- 瞬态组态成像：实际每-jj f0、光学定理 f2、连续谱 f1（解析分段线性主值积分）；mean(|f|²) 及其布居方差。
- 直接双电离摄取检查：归属、单位、图闭合、阈值、重复、与 shakeoff 重叠；**未启用任何未验证拟合数据集**。

---

## 4. 前沿挑战（2023–2026，按"已观测 / 预言未观测 / 空白"分级）

### 4.1 已观测（实验已跑通，理论需追赶）
- **内壳层亚飞秒 Rabi cycling**（Linker 2025, LCLS）：需要把 Maxwell–Bloch 与相对论多电子结构耦合——atomx 的相干层是朝这个方向的原生尝试。
- **空心原子超荧光 / 内壳层 x 射线激光**（Yoneda et al., Nature 524, 446 (2015) Cu 1.5 Å）：XFEL 泵浦 + 原子增益介质，需要泵浦-传播-退激的耦合模型。
- **多芯空穴共振谱**（Son 2023 预言）：质量空心原子的逐电荷共振结构，待系统实验验证。

### 4.2 预言未观测（理论成熟，等待光源）
- **X 射线 EIT / 腔 QED**：核 EIT 已在薄膜腔实现（Röhlsberger 2012, ⁵⁷Fe），但**用 XFEL 宽带脉冲实现 EIT 尚未观测**；腔基 XFEL 路线见 arXiv:2608.15244 (2026)。
- **硬 x 射线区 frustrated absorption 失效**：Son 2020 预言 hollow-shell 态在重原子/硬 x 射线区破坏受挫吸收，**实验未确证**。
- **tender x 射线（2–5 keV）系统性 REXMI**：基本是空白/新兴方向。

### 4.3 空白与机会（尚无 SOTA 覆盖）
- **相对论 × 共振 × 相干 × 速度的整合**：§2.5 的能力矩阵空白格。
- **QED/Breit 进速率网络**：无任何代码在 XFEL kinetics 中系统包含自能修正。
- **ML 速率表代理模型**：non-LTE kinetics 有 LLNL–UCSB 合作的 NN 压缩/生成代理（Jeong & Cho, 2024 起），但**针对 XFEL 原子动力学的同行评审 NN 速率模型是空白**。
- **AI agent 自动化原子动力学工作流**：见 §5.3。

---

## 5. Agent 探索路径（atomx 战役的方法论与可复用模式）

XFEL 战役不是一次性写码，而是 agentic-science 工作流的完整示范。可复用的模式：

### 5.1 测量驱动（measurement-driven）
- 每步先插桩实测再推理：v3 建表 62 组态 56 s（29×）是实测；F·σ 剪枝、骨架共享的倍率都是实测，不是估算。
- **失败也数据化**：B-spline 伪态连续谱在 1 keV 低 9.5×——这个证伪测量直接催生了 shooting+Milne 路径（1.006× XATOM）。每个被证伪的假设在 `trace.md` 留测量值，不是一句"不行"。

### 5.2 闸门文化（gate culture）
- 每个物理增量过三级验证：brute-force 对撞 → 跨引擎一致 → NIST/实验锚。不通过就 **NO-SHIP** 并记录。
- 案例：jj 容量 4l+2 → 2|κ| 的修正由单测触发（345 非物理组态 → 135 正确）；Auger 面板散布 9.77× 的 xfail 诚实保留，不粉饰。

### 5.3 参考源码锚定（reference-source anchoring）
- 不读二手转述：faclib `SlaterTotal`/`AutoionizeRate`、XATOM `decay.f90`/`coulomb.f90` 在本地逐行读；FAC 约定（a_radiative = TR总/(2J+1)）由本地重跑 sfac 脚本一次钉死，终结六轮审计。
- **AI agent 自动化 XFEL 原子动力学是已确证的空白**——现有 AI 工作全在光束线/加速器调优（SASE 贝叶斯调谐、SwissFEL BO、ML 脉冲标定 PRR 5, 023114 (2023)），原子 kinetics 与谱学工作流的 agent 化无已发表论文。atomx 战役提供了一个可引用的 agentic-science 案例。

### 5.4 实验一致化五步路线（SOTA 自评的验证纲领）
atomx 的 SOTA 评估（`docs/xfel_sota_assessment_2026_10_04/`）给出从"能算"到"与实验一致"的五步：
1. **审计同一 Dirac–Coulomb 问题**——同轨道/组态/核势活跑 FAC 或独立 CI，防物理修正掩盖实现误差；
2. **扩大相关空间并给误差预算**——n、l、激发阶数系统增加，跟踪 q/r 物理特征；自电离态需稳定化/complex scaling 收敛证据；
3. **统一处理修正与开放通道**——Breit/QED/反冲对初态、共振、末态一致处理；连续自能实部给能移、虚部给宽度；
4. **对照测到的量**——先弱场分辨共振验中心/宽度/强度/分支，再复现 XFEL 的光谱/时间统计/焦体积/仪器响应；
5. **外部验证与性能认证分开**——meV 级基准上建立可信结构误差预算；效率比较固定物理输入与误差目标。

---

## 6. 差距与下一步（诚实边界）

| 方向 | 现状 | 下一步 |
|---|---|---|
| Ne q/r meV 认证 | -8.9/-13.9 meV，残差归因到 rank2 极点 + Breit 约定 | 复极点收敛认证、跨模型 Breit 分项核实、q−r 装配审计 |
| 多电子相干 REXMI | 单/两电子 CI 相干打通，N≥3 连续损失链在途 | 一般-N 有符号 CI 共振/损失链、Fano 干涉 |
| SASE 集合 / 逐发统计 | 高斯脉冲近似 | SASE 涨落集合、空间与逐发分布、探测器前向模型 |
| Fano 干涉 / 记忆效应 | 孤立非相干线 | 连续态耦合的实能移 + 线型干涉 |
| 强场相干成像 | 成分具备 | 与密度矩阵传播耦合的衍射信号 |
| ML 速率代理 | 空白 | 查表 MC 已是毫秒级；NN 代理做跨元素泛化是自然的下一步 |

---

## 附：关键数字速查

| 指标 | 数值 | 出处 |
|---|---|---|
| Young-2010 锚闭合 | F=1…1000 a.u. 7 点，比值 0.59–1.20 | `test_ne_xfel_cascade_live.py` |
| 真速率图 | 134 组态 / 378 PI 边 / 250 Auger 边 | `xfel_real_rates_2026_10_03/` |
| MC 编译加速 | 72×（完整 134 态图） | `xfel_mc_acceleration_2026_10_03/` |
| Ne q/r 谱位差距 | -8.9 / -13.9 meV vs Togawa 2024 | `xfel_tail_breit_pole_2026_10_06/` |
| 相干 Ne⁸⁺ 主线 | 922.87 eV，A=9.40e12/s，峰布居 0.98 | `xfel_ci_coherent_2026_10_03/` |
| 测试 | 820 passed / 1 skip / 1 strict xfail | commit 233fec1 |

> 本文档随 atomx 仓库演进；复放入口见各 `docs/xfel_*/README.md` 与 `docs/XFEL_GAP_ANALYSIS.md`。
