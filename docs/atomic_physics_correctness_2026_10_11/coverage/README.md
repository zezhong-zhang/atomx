# 消费端物理覆盖与固定空间判别

本审计把三个问题分开：**声明近似的方程有没有写对；近似有没有包含目标所需的物理；所包含的物理有没有空间收敛**。扩大单粒子截断只能检验第三项。即使同一近似在无限虚空间收敛，也不自动补入它没有的簇秩、算符或多参考耦合；几条实验能级变好也可能有误差抵消。

本轮只追踪实际消费链并复算已保存证据，没有运行原子求根、扩大空间或改变生产源码。[前瞻合同](contract.json)固定 20 个输入；[证据矩阵数据](evidence.json)保存完整 SHA-256。[source_epoch](source_epoch/)保留审计时的 14 个源码文件，因其他获授权工作可随后修复消费端；下文关于旧 FSCC 准入漏洞限定于这个源码时点。

## 当前 Yb 消费端，而非全库能力

实际链为 `probe_yb_monovalent_cuda → probe_yb_full_monovalent → CoreSD → ReducedMonovalentSD → atomx_monovalent_map`，最后由能量比较驱动读取全部五根。GPU wrapper 改计算引擎，返回的每根仍重跑完整原 CPU 方程残差。五态共享 DC、Fermi 核、V68、ns123、全部 68 核电子和 l≤3、E<5 Ha 的 414 个径向虚态。它不是全阶非线性 CCSD，也不是 FS-CCSDT。

| 物理／方程 | 当前五态实际消费 | 已证明的正确性范围 | 仍缺失或未验证 | 固定空间高信息量判别 |
|---|---|---|---|---|
| 相对论、有限核与原生 metric | Dirac-Coulomb、Fermi 核与共享原生积分 | 输入、核势、轨道与源身份检查；不能把自旋轨道和有限核说成未实现 | 核参数／同位素响应、无对投影依赖、独立大模型完整 H 验证未认证 | 同 H／轨道的独立有符号积分及 Fock 检查；参数改变另立物理合同 |
| 核及价线性化 SD 方程 | 实际核 T1/T2 和价 S1/S2，裸 ev 图项、固定核分母、价能量反馈 | 12-spinor 独立主方程／费米动作；约化四种轨道变体、40 个全部 M 实例；大 Yb 全向量 CPU/CUDA 对照与原作用残差 | 大 Yb CPU/CUDA 复用相同图项，二者相同不等于独立大 H 方程完备性证明 | 原冻结空间的弱耦合二阶系数，逐核与五态对照独立 MBPT2；不只查能量和 |
| 非线性簇积与连接三激发 | 当前生产线性化 SD 不消费 | 其省略是声明近似，不自动算代码 bug | T1²、T1T2、T2² 等非线性及连接 T3/S3 缺失；大小未知 | 同完整小空间 LCCSD→CCSD→CCSDT→FCI，全有符号振幅与按秩残差分开 |
| 4f 核—价相关和空穴混合 | 68 核包含 4f；价 doubles 包含核空穴／双粒子相关 | 不能称作完全没有 4f 相关 | 单参考 SD 没有平衡的独立 4f-hole 多组态根、相应轨道弛豫及已认证组态权重 | 同 J/parity 的平衡粒子—空穴 P 与互补 Q，保留有符号耦合；同时检查根身份与性质 |
| 静态、动态 Breit | 当前五态明确为 DC，未加入 transverse | 全库有限 CI／FSCC signed transverse 和 J-pair 锚另有证据 | native 大核 SD 的 transverse 图项尚不完整；现有 pair source、pp/hh ladder 只是子集 | 相同小空间完整 DCB／动态 H 的原始费米方程，核／价所有图类与能量账闭合；不能向 Coulomb RINT 塞一个比例因子 |
| QED、反冲 | 当前五态没有消费；库内其他路径的能力不迁移 | 省略在比较结果中明确列出 | 大 SD 同波函数算符及响应、屏蔽 QED、正常／特定质量位移未认证 | 固定 H 的独立 signed 一／二体修正与响应，再同模型重解；不是复制文献能移 |
| 性质和物理度量 | 五态能量比较；另有完整 18 M 的私有 SD norm 消费 | 保留 core spectators、Pauli 和 core-S1／valence-S2 干涉的 norm | norm 正确不建立全 H 驻值、生产 dressed E1/E2/M1、寿命、hfs 或 g 因子的精度 | 同有限波函数对照完整有符号双线性；大空间另立 SD／CC 响应近似合同 |
| 实验态匹配与能量零点 | 同计算 6s 零点的五态 nominal J/parity 分支 | 无拼接根、四谱位和两劈裂分别保留原门 | 组态权重、近简并转移身份及谱强未认证 | 同对称块的权重、重叠、g／hfs／强度联合检查，不能仅靠配置标签 |
| 多价 CI+all-order | 当前五态 standalone SD 没有消费 Sigma2 的 CI 波算符 | directed Sigma1/Sigma2 方程与有限 Bloch metric 锚是单独能力 | 大 SD 的实际 CI 波算符、左右态与 metric 消费未认证 | 同 H 有符号非厄米有效矩阵、物理 metric 与完整 wave equation；不强制做厄米平均 |

消费源码：
[完整 Yb 驱动](source_epoch/docs/atomic_pci_sd_native_2026_10_09/probe_yb_full_monovalent.py)、
[CUDA wrapper](source_epoch/docs/atomic_pci_sd_native_2026_10_09/probe_yb_monovalent_cuda.py)、
[CoreSD](source_epoch/src/atomx/methods/core_sd.py)、
[ReducedMonovalentSD](source_epoch/src/atomx/methods/reduced_monovalent_sd.py)、
[native 价 map](source_epoch/src/atomx/methods/_pci_monovalent.f90)、
[有限 SD 方程](source_epoch/src/atomx/methods/monovalent_sd.py)、
[transverse pair bridge](source_epoch/src/atomx/methods/sd_transverse.py)、
[pair 图类子集](source_epoch/src/atomx/methods/sd_pair_diagrams.py)。

直接结果：
[五态实验比较](../../atomic_pci_sd_native_2026_10_09/yb_monovalent_experiment003/comparison.json)明确列出 Breit、QED、反冲、非线性、完整 triples、尾部和完整空穴混合的限制。
[有限主方程锚](../../atomic_pci_sd_native_2026_10_09/monovalent_sd008/comparison.json)的主方程最大差为 2.8022e−15 Ha，**完整 H 残差仍为 0.00255145 Ha**；
[约化锚](../../atomic_pci_sd_native_2026_10_09/reduced_monovalent008/result.json)的随机全 map 差为 3.9552e−16 Ha，但所保留有限变体的完整 H 残差最高为 0.0593200 Ha。后者包含强 occupied–virtual 变体，不是当前 Yb 的误差界。这直接说明“近似方程写对”和“精确波函数”不同。
[现有大五态](../../atomic_pci_sd_native_2026_10_09/yb_full_monovalent006/result.json)的原作用残差最高 2.3202e−11 Ha；四谱位中三条和两劈裂仍未过原实验门。
[当前 norm 消费](../../atomic_pci_sd_native_2026_10_09/parallel_six_norm001/run002/result.json)保留其私有 norm 范围，不能升级为性质认证。

[原始全阶方法论文](https://arxiv.org/html/0905.2578)区分线性化 SD、非线性 CCSD、valence triples 及其性质双线性式；“all-order”是在选定激发／图类内重求扰动阶，并非所有激发秩。
[Yb⁺ 原始多方法论文](https://mariannasafronova.com/wp-content/uploads/2022/12/PorSafKoz12c.pdf)说明 4f-filled／4f-hole 态混合会改变特定性质，多电子与单价方法各有取舍。该文献支持需要平衡模型的方向，不能给当前实现提供未测的耦合或修正值。

## 已证实的实现漏洞：旧非 canonical FSCC 准入

审计时 [public compute](source_epoch/src/atomx/methods/fscc.py)第 378–387 行只验证 ABI、binary SHA 和 axial 能力；第 419–427 行用 backend Jacobi update × 分母界判断收敛。它没有消费 manifest 的 `noncanonical_attachment_equation`，也没有独立 retained／Bloch 残差门。
[builder](source_epoch/tools/build_expt_backend.py)把 `ATOMX_FOV_T2_001` 保持为 opt-in；
[修复](source_epoch/tools/expt_noncanonical_patch.py)删除 0h1p 中对已 antisymmetric T2 再做一次 P(34) 的重复操作，保留 Fov 和完整 H。

同一完整有限 H 已给出区分，复算来自[最终有限 FSCC benchmark](../../atomic_external_benchmark_2026_10_11/fscc/run003/result.json)：

| 固定输入 | backend | retained attachment residual / Ha | full Bloch residual / Ha | 所有 8 根对 FCI 最大差 / Ha |
|---|---|---:|---:|---:|
| 2 核／10 spinor | 原始 | 4.163911e−7 | 4.163911e−7 | 8.728964e−10 |
| 2 核／10 spinor | 修复 | 1.776357e−15 | 1.776357e−15 | 1.243450e−14 |
| 4 核／12 spinor | 原始 | 1.031257e−2 | 1.031257e−2 | 3.515326e−3 |
| 4 核／12 spinor | 修复 | 6.483789e−14 | 2.283871e−3 | 9.093015e−6 |

原始两核的能量误差很小，仍有真实 retained 方程缺陷；不能用能量看起来接近来放行。修复四核 retained 方程通过，而 T4/S4 未保留导致完整 H 残差和 FCI 方法差，**不是修复失败，也不能通过扩虚态自动获得 T4**。两核／三电子最大激发秩小，有限 CCSDT 精确性的结论不迁移到四核或 Yb。

最窄的合理生产修复是：对 `sector='0h1p'` 且实际 occupied–virtual Fock 非零的请求，在积分输出和 backend 启动前要求 binary／source-bound 的 `ATOMX_FOV_T2_001` 能力；原始或缺失 capability 明确拒绝。禁止静默令 Fov=0、放宽残差或扩大截断绕过。零 Fov、0h0p 和 1h0p 不因此强行换方程；修复的通用扇区推广仍需各自证据。本审计未修改 public consumer。

## 在扩截断前应完成的工作

1. 先关闭上述**已有同 H 证据的方程准入漏洞**，然后把 backend 更新残差、独立 retained 残差和 omitted-rank 方法残差分别标明。
2. 在原 Yb 冻结空间核对两类完整二阶系数／价单列，检验符号、分母、normal ordering 和低阶图项。即使通过，也只支持已声明 LCCSD 方程，不能把 nonlinear／triples 改称已包含。
3. 用同完整有限 H 的 LCCSD、CCSD、修复 CCSDT 和 FCI，分别估算非线性与秩截断响应；用平衡 4f-hole 模型检验另一竞争解释。所有上下态、centroid、splitting、identity 保持共同模型。哪一项主导当前 Yb 尚待测量，不能先定唯一原因。
4. 优先补齐完整 transverse SD 消费链的 singles、mixed-rank／Fock、ring、剩余 core／valence／Sigma2 图类。pair ladders 的正确子集不是全方程；高精度能量与性质需独立消费合同。

这些门通过后，空间收敛才是在明确正确近似内测量计算量需求。最高水平应限定原子／离子、性质与不确定度目标，用多模型、秩、算符和空间控制获得可信的精度范围；不存在“一个无限大截断即可证明软件物理全部完备”的证据。

重放：`.venv/bin/python docs/atomic_physics_correctness_2026_10_11/coverage/audit.py`。已有结果的重放只校验保存输入与 source_epoch，并报告当前源码相对该时点的变化，不改写证据。后续 public guard 的修复及其新源码验证由上级 owner 分别记录。
