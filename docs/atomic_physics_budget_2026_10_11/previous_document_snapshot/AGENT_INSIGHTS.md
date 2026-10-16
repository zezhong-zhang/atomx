# Agent 洞察登记簿

本轮有限关联方法的细化证据见[六方向控制](atomx_hourly_progress_2026_10_10/parallel_six002/README.md)：
扩大占据核后必须按实际保留激发秩检查投影方程，同时公布完整遗漏秩残差和 FCI 方法误差；
不能把两电子有限精确性转移给四电子 CCSDT。CUDA loader、实际资源准入、数组正确性
和受控计时分别判定；修复 loader 后仍可在原预算验证，负载门失败时不报告加速比。
这细化既有完整算符与性能规则，不增加根级规则。

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

按相关主题读取。[工作机制](AGENT_KNOWLEDGE_WORKFLOW.md) 定义记录字段、状态、
晋升与撤回流程；[参考索引](AGENT_REFERENCE.md) 提供完整专题入口。
本次只整理已有代表性洞察与新用户裁定，不声称重新验证全部历史实验。
证据中的日期、模型和源身份限定测量范围，`adopted` 不代表普遍科学认证。

## 当前登记

| 稳定 ID | 状态 / 类型 | 行动与边界 | 依据 / 去向 |
|---|---|---|---|
| convention-first | adopted / 实证经验 | 比较前核对量、平均和归一化；具体 FAC 系数仅适用于所读 fixture | 下方记录；AGENTS.md §3 第一项 |
| measure-end-to-end | adopted / 实证经验及用户决策 | 高性能为长期设计要求；扩规模先估算工作量、驻留内存和数据搬运，优先可扩展表示与收缩结构；helper 提速后仍验实际任务 | 下方记录及 2026-10-09 用户裁定；AGENTS.md §4 |
| preserve-metric | adopted / 实证经验 | 重放残差须保持度量身份；近根另验导数/驻值 | 下方记录；AGENTS.md §3、§5 |
| knowledge-layers | adopted / 用户决策 | docs 随代码修改/合并/删除并蒸馏；根规则只放哲学和必须做/禁止做 | 下方用户裁定及修订；AGENTS.md §6 |
| complex-space-regate | validated / 有限模型实测 | 扩rank后重验束缚虚部/增量/身份；有符号参考随径向号规变换，旧rank1平台不可继承，归因限于实测范围 | 号规协变性与rank2束缚锚复核 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)、12/16控制 (local evidence: `xfel_complex_rank2_control_2026_10_05/README.md`)、表示判别 (local evidence: `xfel_complex_representation_control_2026_10_05/README.md`)；不新增根规则 |
| transition-feedback-priority | validated / 有限模型实测 | 以完整Q耦合预测上态减基态的增量；用完整CI及完整新增singles对照定位rank响应，指标不充当尾项误差界；高k/高L尖点筛选不能相加 | 完整3200Ha尾部、Breit与极点判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)、完整800→1600Ha、QED重解与f/g预检 (local evidence: `xfel_uv_tail_continuation_2026_10_06/README.md`)、完整200→400→800Ha与新单激发对照 (local evidence: `xfel_high_energy_pair_cutoff_2026_10_05/README.md`)、束缚rank3/n6与高能电子对筛选 (local evidence: `xfel_bound_rank3_n6_2026_10_05/README.md`)、因素筛选 (local evidence: `xfel_factor_screening_2026_10_05/README.md`)、完整0–5Ha控制 (local evidence: `xfel_positive_rank2_closure_2026_10_05/README.md`)、窗口回放 (local evidence: `xfel_transition_feedback_2026_10_05/README.md`)；专题策略，不新增根规则 |

## convention-first

- 触发 → 行动 → 原因：跨程序比较率/能量时先读实际生成脚本及定义，
  防止将平均和归一化差异诊断成实现 bug。
- 证据：[原始记录 §1.3、§2](archive/AGENTS_2026_10_05.md) 中，所测 FAC
  DR/Burgess fixture 的 `a_radiative` 与 TR 级总率相差精确 2J+1。
- 边界：这是特定输出约定；不能给所有 FAC 表机械乘除 2J+1。
  fixture/脚本/量定义变化时重新核对。核对约定通过也不排除代码缺陷。
- 补充证据：[Yb⁺ 非局域 Fock 与独立 DF 对照](yb_diagnosis_repair_2026_10_08/README.md)。
  非局域交换模型的裸长度/速度值不一致时，先核对共同局域哈密顿量
  假设及相互作用对易子，再改归一化或符号；本次闭合只验证冻结核
  表示，不代替相关波函数和 RPA/辐射顶角验证。并入现有约定规则，不新增根规则。
- 补充证据：[Yb⁺ 独立绝对阈值锚](yb_correlation_tail_2026_10_08/README.md)：
  定位上下态绝对偏差时另核对电离阈值、离子终态和数据版本，不能因都标作 NIST
  就混用 Handbook、论文和当前 ASD 的零点。旧激发能及纯增量对照保持，新的
  绝对锚单独冻结；本次区分基态与上态系统偏差，但不把模型−实验差直接解释
  为单个遗漏物理项。并入现有约定规则，不新增根规则。
- 2026-10-09 消费边界补充：[pCI/AMBiT/EXP-T 原生升级](ci_fscc_reference_upgrade_2026_10_09/README.md)
  把完整带符号 Σ₁ 接入 CI，同时绑定生产者的芯层、外部轨道相位、度量与核模型；
  CI 打开已积分消去的芯层空穴时必须拒绝并重建互补 P/Q，不能沿用全芯层自能。
  原生重放通过仅证明消费闭合，不改变此前实验失败。并入既有约定规则。
- 同日推进：[AMBiT 同积分核对与有限 P/Q 实算](ci_mbpt_matched_qspace_2026_10_09/README.md)
  已用独立上游代码核对二阶有符号矩阵与两个图类；共同径向表限定其证据为角动量/图求和。
  原生互补 Q 消元从一份裸 DC 矩阵生成，不叠加闭芯 Σ₁；有限 Yb 一空穴模型的 Q 内部
  相互作用反馈约 95–733 cm⁻¹，能量参数重求约 0.4–25 cm⁻¹，二者不能混称自洽修复。
  四条谱位残差反而增大，有限空间的数值通过不能替代实验改善。并入已有约定规则；
  数值量级限于该模型，不推广为 SD、径向 SCF 或 QED 响应上界。
- 同日能力补充：[有限 CCSD、NR-SD 与静态 Breit](yb_sd_breit_completion_2026_10_09/README.md)。
  方法称作 SD 时先核对是配置截断、指数 CC 振幅还是线性化原子全阶方程；用独立
  完整二电子 CI 和两个非相互作用电子对分辨指数中的非连接高阶激发。物理空间扩展
  同时重求基态/上态，约550 cm⁻¹总能量降低可在谱位中抵消到32–45 cm⁻¹。
  受限 Q 的严格 C–B 二阶、相关态一阶 Breit 和完整 DCB 重解是不同定义，不能相加
  或把小受限交叉项视为全部 Breit 尾部上界。数值量级只适用于记录的冻结模型。
  并入既有约定规则；没有新增根规则。
- 去向：根 §3 的先核约定规则；具体系数保留在专题参考。
- 2026-10-08 文献核查补充：[Yb⁺ 文献修复策略](yb_literature_repair_strategy_2026_10_08/README.md)
  将束缚能精度与跃迁精度、一阶 Breit 与 Coulomb–Breit 交叉项分开；实算覆盖决定
  能排查哪一项，方法名称和文献百分比不能替代匹配量。空穴进入 CI 时重建互补
  P/Q，避免与已有核–价自能重复计数；修复效果仍待原生计算。并入约定规则，不新增根规则。

## measure-end-to-end

- 2026-10-11 [外部性能控制](atomic_external_benchmark_2026_10_11/README.md)：
  短子进程耗时出现离散平台时，核对超时等待器的轮询延迟，并分开保存实际公开路径
  壁钟与低开销子进程退出计时；不能把等待策略成本推断成数值内核性能。
  请求的 OpenMP team 大小也不等于总进程线程数：绑定实际运行库并观察线程及 CPU
  时间，越过原资源上限则停止该样本。逻辑 team 效率、进程线程占用与活跃 CPU
  利用率分别标注。细化现有性能规则，不新增根规则。
  [FAC 续测](atomic_external_benchmark_2026_10_11/fac/README.md)补充：完整导出和
  诊断捕获也是工作量；两个程序的诊断导出位置不同，须保留各自自缩放，不能直接
  当作相同生产任务的速度排名。缓存序列化导致子进程退出时，保留未完成样本并以
  私有缓存复验，不删除全局缓存、不把失败耗时计为加速。并入本规则。

- 2026-10-09 表示层续篇：[原生按需积分与虚池扩展](atomic_streaming_tail_2026_10_09/README.md)。
  分别估算原生算符、导出文件和后端振幅的容量；一个阶段去掉四阶驻留数组，
  不代表其他阶段可求解同一完整空间。闭壳层旋转对称性和有界缓存须用共同有符号
  积分、完整磁子空间和真实电子方程验证；缓存大小变化不是物理截断。
  只在记录阶段报告实测性能，不外推整体加速或实验精度。并入已有规则。
- 2026-10-09 用户裁定：“始终记住高性能的原则”。并入现有规则，作为长期设计
  要求：扩规模先估算工作量、驻留内存和数据搬运，优先改进表示与收缩结构，
  再考虑增加硬件或并行进程；收益仍须在同物理模型、同空间和同残差门下测量。
  去向：根 AGENTS.md §4 首项及已有端到端测量规则。此项是用户决策，
  不构成任何新增性能倍数或科学精度声明。
- 触发 → 行动 → 原因：尝试局部性能优化时，先验数值等价，再测它所服务的
  实际阶段/任务；主导成本可能不在该 helper。
- 证据：径向键实验 (local evidence: `xfel_radial_key_speed_2026_10_05/README.md`)：
  helper 观测 2.33×，完整装配 83.382→83.738s，未过预声明 3% 收益门，候选撤回。
- 边界：单次环境数字不构成普遍速度结论，负载门失败不认证受控倍数；
  更换任务、缓存条件、硬件或主导成本后需重测，不把此次失败当永久禁止优化。
- 去向：根 §4 的配对计时、范围声明和失败保留规则。

## preserve-metric

- 触发 → 行动 → 原因：比较优化终点与重启残差时保存原始度量，
  防止预条件器改变导致残差数值不可直接比较。
- 证据：局部梯度与度量修正 (local evidence: `xfel_local_gradient_2026_10_04/README.md`)：
  旧端点同 seed 度量复测差 1.324e−10，通过原 atol1e−9/rtol1e−5；
  归档 §16 的失败判断须结合此修正理解。
- 边界：残差重放通过不代表 n4 已驻值，也不证明所有差异均由度量引起。
  定义、输入或求解器相关依赖改变时复核。
  QED响应审计 (local evidence: `xfel_qed_response_audit_2026_10_05/README.md`)区分固定轨道的CI系数驻值与径向轨道SCF：
  加QED后已重求CI；旧n4/5额外跃迁响应约+0.018meV，含QED轨道SCF未测。
  这个数值不界定缺失轨道方向，也不能再加到已有谱位。
  直接单电子QED轨道探针 (local evidence: `xfel_qed_orbital_probe_2026_10_05/README.md`)在全221原轨道与冻结Fock势/无对投影内重求轨道，参考占据组合额外响应约+0.01785meV；这是有范围的轨道混合量级，仍非密度反馈SCF、相关三态谱位或完整轨道SCF上界。
- 2026-10-09 矩阵阶段补充：[Yb⁺ 完整五态非对角响应](yb_matrix_self_energy_2026_10_08/README.md)
  表明“重求轨道后本征残差通过”不能认证谱位改善。固定 Hermitian Σ₂、同一核/投影、
  嵌套正交外部空间、共同无极点能区与最低分支时，Rayleigh–Ritz 加 Σ′≤0 约束根只会下降；
  已过束缚的绝对基态不能由继续扩大该外部空间修正。应同时测独立阈值、上下态响应与
  劈裂，再决定改变相关核；该结论不推广到改变核场、高阶图或复共振区间。
- 去向：根 §3 的原生度量/驻值区分及 §5 的重放身份要求；
  旧失败数据保留，[历史快照](archive/AGENTS_2026_10_05.md) §16、§22 连同阅读。

## knowledge-layers

- 2026-10-10 用户裁定常规测试约三分钟：默认显式 essential 清单，昂贵 campaign 退出测试收集和 nightly；
  保留源与失败证据，修改代码仍跑受影响子系统。此为测试范围决策，不是物理空间缩减或速度认证；
  [当前命令和边界](TESTING.md)、[实测](ci_repair_2026_10_10/README.md)。

- 2026-10-10 用户要求重组并持续维护文档：当前指南、开发路由与带日期证据分离；
  CI 检查索引新鲜度和入口链接是否已跟踪，语义仍由执行任务的 agent 负责。
  忽略的本地证据保留路径并明确其不可在 GitHub 读取；[处置与实测](ci_repair_2026_10_10/README.md)。

- 来源：2026-10-05 用户明确要求设计 agents 在 docs 记录、蒸馏洞察，
  将 philosophy、must do 与 not do 放入 AGENTS.md 的机制。
- 触发 → 行动 → 原因：获得改变行动的经验后按证据→洞察→规则维护，
  让根文件保持精简并可追溯；采用 [工作机制](AGENT_KNOWLEDGE_WORKFLOW.md) 中的模板。
- 2026-10-05 补充裁定：每次代码更新必须维护对应文档，包括修改与删除旧文档。
  每个逻辑增量核对文档影响，原位修改、合并或删除失效当前说明并修复引用；
  仅新增报告不能算文档同步。无影响须有具体核对依据，科学封存保留原始证据。
- 2026-10-05 英文与范围裁定：README 反映当前设计；根 AGENTS.md 用英文
  保留跨任务哲学与原则，领域契约和详细记录下沉 docs。已原位重写根文件，
  领域细节迁入 [参考指南](AGENT_REFERENCE.md#task-specific-scientific-contracts)，
  当前设计入口同步。处置和验证见 [工作机制 §9](AGENT_KNOWLEDGE_WORKFLOW.md#9-readme-and-root-guidance-refresh-2026-10-05)。
- 边界：这是工作流程裁定，不是“已实测提高 agent 效率”的科学结果；
  用户后续裁定可修改流程。150 行/12 KiB 是本次维护预算，不是模型性能定律。
- 去向：根 §6、工作机制 §6–7、[代码→文档入口](README.md)；
  常规 agent 自行审查维护，无后台自动记录/蒸馏功能。
- 2026-10-05 用户补充、2026-10-06 落地：每次 commit 前重建的是项目结构索引，
  用于减少 agent 上下文读取，而不是仅重建文档目录。
  [精简地图](../PROJECT_INDEX.md) 与限量 AST 查询自动生成；hook 按暂存区更新，
  源码/文档语义仍需人工核对。上下文入口字节数减少可测，不等同 agent 效率已认证。

## transition-feedback-priority

- 触发 → 行动 → 原因：精密跃迁的空间自适应先比较Σ上−Σ下，保留候选Q内完整耦合；
  单态能移可能相互抵消，纯对角分母也可能错过meV级组内响应。
- 补充证据：[Yb⁺ 冻结轨道 Dyson 能量反馈](yb_energy_dependent_sigma_2026_10_08/README.md)：
  同一闭核与 ℓ≤6/50 Ha 空间内，单态自洽响应为 36–220 cm⁻¹，跃迁响应最高
  110 cm⁻¹；ℓ≤4→6 的跃迁增量仍达 558 cm⁻¹。判断自洽是否为主因须测上下态
  响应之差，并保留独立空间门；本例不能推广为完整轨道 SCF 或高阶相关的上界。
- 补充证据：[Yb⁺ 完整相关尾部与径向表示](yb_correlation_tail_2026_10_08/README.md)：
  扫能量窗时同时核对实际虚谱及完整上下态响应，避免把有限表示容量当作尾界。
  本例共同 ℓ6/50 Ha 的粗/细虚轨道为 861/1616，但共同 ℓ6/200 Ha 的实际径向
  跃迁响应最高 2.300 cm⁻¹；最后分波增量仍 39.156 cm⁻¹，细基最后正能
  增量仍 17.577 cm⁻¹。容量增加不等于误差增加；有限局部增量也不认证无限尾部。
  并入现有规则，不新增根规则。
- 证据：同一有限Fermi/frozen DF-DKB全修正H，15→10、10→5Ha的四个跃迁
  预测误差0.004–0.008meV；独立新n4三态的两线预测误差约0.006/0.007meV。
  原窗口的对角反馈误差1.6–2.2meV，未过预声明1meV预测门。
  同n4第一性原理诊断 (local evidence: `xfel_first_principles_diagnosis_2026_10_05/README.md`)：
  相对2017印刷分项，DC差−273.5/−274.2meV，Breit增量差−18.5/−18.8meV；
  该对照对应旧n4/5，不能搬作后续空间的分项误差预算。
  同模型核心定位 (local evidence: `xfel_first_principles_localization_2026_10_05/README.md`)：
  类氦n4/5→n4/0下降88.689meV，Q谱隙88.012Ha，完整反馈误差0.042meV；
  三态条件归因随有限类氦零点同移121.664meV，跃迁差严格不变。
  后续完整三电子0–5Ha控制 (local evidence: `xfel_positive_rank2_closure_2026_10_05/README.md`)同步扩展g/q/r至35687/51948/35688CSF：
  三态实测能移-100.972919/-47.321821/-47.321470meV，跃迁增量+53.651097/+53.651449meV，
  本次完整低正能rank2实际消除原偏差17.80%/17.57%，确认本干预中基态相关遗漏更大；余下约0.25eV主因尚未唯一定位。
  因素筛选 (local evidence: `xfel_factor_screening_2026_10_05/README.md`)进一步测得旧 n4/0 固定态的一阶点核−Fermi跃迁约+0.522meV、半径同形±5%约0.05meV；条件自然同位素反冲约+0.19meV，均为一阶控制。
  后续束缚 rank3、完整 n5/n6 与高能电子对筛选 (local evidence: `xfel_bound_rank3_n6_2026_10_05/README.md`)在同一原模型依次测得束缚 n≤4 rank3 的两线增量 -0.007353/-0.007918 meV，完整 n5 为 +5.226855/+5.263761，n6 为 +3.054951/+3.072827；九态完整残差与全部新耦合树重放通过。小束缚 rank3 不约束正能/高 n rank3，n5/n6 未通过局部 1 meV 门，小高 n 增量也非无限尾项界。按 n6 校准的 n⁻³ 核心穿透模型给 n≥7 两线代理约 +7.764/+7.809 meV，只有两步数据，渐近假设未认证。
  原DF 1s²短程电子对高动量代理在200Ha给百meV筛选尺度，高L形式尾部约55meV；两者重叠同一尖点，不是实际跃迁修正/误差界，也不能相加。后续完整200→400→800Ha与新单激发对照 (local evidence: `xfel_high_energy_pair_cutoff_2026_10_05/README.md`)同步完整补入200→400、400→800Ha的所有新rank1/rank2，两线实际变化 +145.986276/+145.295300、+58.311005/+57.955575 meV。分别只补各步全部新rank1的合计跃迁响应为 -10.898532/-10.616794，再补各步全部新rank2的条件响应为 +215.195813/+213.867669。这些独立中间空间通过原完整残差/身份门；完整Q按rank/core/双虚类型的分区来自同一个耦合解，含跨组干涉，不是独立电子对修正。五算符期望差之和重放实际能移，属于同一状态响应的代数账目，不是算符开关误差预算。新33轨道的原生/QED/反冲/新腿Breit控制通过，旧221轨道和原算符块保持。
  后续完整800→1600Ha、QED重解与f/g预检 (local evidence: `xfel_uv_tail_continuation_2026_10_06/README.md`)实测800→1600Ha两线增量+20.432577/+20.298738meV；完整新singles为-0.013647/-0.012990，rank2-after-singles为+20.446223/+20.311728，均经独立完整残差与身份重放。1600Ha固定CI的QED非线性跃迁响应+0.039504/+0.039272meV已含在谱位，不能加作修正或界定未做的轨道/投影反馈。完整f/g原生轨道和流式计数通过，但当前SE表κ循环会遗漏更高κ通用VP，完整高L必须先修复并独立控制其覆盖；计数不等于已算能移。
  同空间实际DC→Breit→QED→反冲路径通过独立完整算符重放；2017印刷DC差-1.017273/-2.760369meV、Breit增量差-5.028297/-5.492104meV仅是跨模型定位，不把期望账目或路径差写成唯一遗漏物理项。
  旧一般rank1的100→200Ha约−42meV和旧单4f约−0.2meV保留各自空间范围，不能约束双虚轨道尖点尾部。径向/极点、Breit/投影和QED轨道SCF仍无数值上界。
  原0–5Ha控制的完整Q跃迁预测−实际+0.006517/+0.008099meV，预声明1meV预测门=True。
  原0–5Ha控制的三态/15算符、完整空间/旧块、原矩阵与精确Q重放通过。
  在同一完整Q耦合解中，保留2s的坐标分区贡献q/r跃迁反馈
  +67.667/+67.785meV，
  其余分区部分抵消；基态对应1s²激发、上态对应1s–2p激发。
  这是本干预的坐标诊断，已含跨组耦合，不是独立可加电子对修正。
  公共Coulomb正性与已验表示条件下的一体下界减完整修正Q范数，闭Q充分门=True；
  此条件谱隙检验不认证未计算尾项或物理极点。
- 边界：只验证这些同模型完整增量；剩余核心分区是同一个耦合解的有符号坐标诊断。
  不把分组变成独立可加修正，不以预测残差估未计算空间，不外推近极点/新投影/高分波。
  印刷对照不能把全部偏差归到基态，也未排除算符/投影偏差。
  低电子数核心控制不能直接加到三电子跃迁；绝对态归因需额外且收敛的能量锚。
  最新1600Ha计算−实验为-14.810840/-19.755772meV，800→1600Ha局部1meV增量门q/r=False/False。旧尾部预测经新实际增量检验，新的有效p与p=1.5尾部仍仅为条件代理，未认证渐近或径向密度；
  不代替完整相关/径向/rank3/连续谱误差预算或物理极点认证。
  反馈仍需装配H_QQ/H_QP；原0–5Ha控制连接续算7797.942s，
  成本不含此前原驱动/局部试验/隔离准备和停顿，预测便宜不等于整体装配提速。
  四旋量无符号值复用的完整冷任务虽同矩阵，但零缓存命中且13.898→16.907s，未采用；
  不把固定样本的局部缓存或负结果外推为总体speedup/无收益定律。
- 去向：已有专题报告与当前Ne入口；根§3已要求三态同步/上态减下态，无新根规则。

最新完整3200Ha尾部与Breit判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)实测两线增量+5.914650/5.874690meV、偏差-8.896190/-13.881082meV；完整singles为0.014556/0.014465、rank2-after-singles为+5.900094/5.860225meV。原五算符残差、完整Q正谱隙/反馈/全部耦合树/旧轨道与身份独立重放通过；局部相关门未闭合，测前p=1.5预测的负结果保留，不通过重拟合认证尾项界。同空间Breit非线性响应+0.241308/0.291141meV已经包含，参考组态频率有序本征值界0.071999/0.070425meV仅限指定嵌入处方。共同位移与劈裂分别定位，基态严格不影响q−r；未验证权重的论文c.g.不作已匹配误差预算。

## complex-space-regate

- 本次复核：本轮判别控制 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)保留原 H/E/C，独立解析径向号规带入有符号 LS 参考后，q/r 身份指标 .917/.927；逐个占据轨道符号翻转的 8 个完整坐标协变性控制通过。角模板必须与原生轨道号规共同变换，不能因原生特征向量任意号而改权或换根。
- 扩相关空间仍须重验物理门：同点核/static DCB/rank1 pool64 加完整 pool4 rank2 后，N2/N3 束缚锚虚部 +1.2951e−4/+1.5213e−4 Ha，未过原 1e−7 Ha 门；原 rank1 全子矩阵与整个径向数组重放，N2 全矩阵独立重装配为 0 误差。选态残差与完整坐标身份通过，也不能继承 rank1 平台宣称物理极点。
- 外盒判别：同一 N2 DC 的 R4/8/12 控制使 mixed4 虚部从 +1.27e−4 经 −7.80e−7 到 +1.34e−9 Ha，恢复原束缚门；关闭 Breit 后 R4 异常仍在。R12 DCB 中两束缚锚也通过，q/r 检出带负虚部且保留坐标身份的候选。扩大 R 同时改变有限谱的范围和密度，因此仅支持外盒/有限谱表示的干预归因，不能宣称纯墙面误差界。
- 边界：R4 rank1 的参数、保留数和有限径向族门已独立重放通过，但完整小池 rank2 仍改变 R12 跃迁约 −31.50/−32.14 meV；参数平台不能替代相关收敛。原实空间两种 2p 相对号一致，不把复参考缺陷搬作生产谱位错误。当前所有物理极点状态仍为未认证。
- 去向：已有基底/态身份及扩空间重检原则覆盖，更新专题与导航；不新增根规则。完整 3200 Ha 实空间与独立重放已完成；R4 面板、R12 八个完整谱及有限角度门通过，相关与物理极点门未认证。

## 维护

新增或修订前检索稳定 ID。相同机制合并；不同适用条件明确区分。
候选与撤回也要显式登记，不能只留下成功项；详细失败证据归所属专题。
已替代条目保留紧凑 ID→替代链接，长记录移入主题页；维护时按工作机制核对规则去向。

## 2026-10-09 Integral backend and tensor CC audit

- Trigger: an upstream backend reports support for a method. Required action:
  trace the actual signed Fock/ERI reader and every tensor block, then run an
  electronic independent anchor. Reason: the pinned EXP-T PySCF entry contained
  commented ERI reads, zero Fock terms and hard-coded blocks despite successful
  compilation. Scope: external integral adapters; [evidence](atomic_allorder_fscc_2026_10_09/README.md).
- Trigger: residuals converge after expanding the core or virtual pool. Required
  action: identify the same physical multiplets and report individual state and
  transition changes. Do not infer improved experimental accuracy from a larger
  space or replace a failed complete-band request with count-limited orbitals.
  Scope: finite tensor CC models; [current contracts](ATOMIC_FSCC.md).
- Scalar-SD continuation: distinguish nonlinear CCSD, linked linearized SD and
  CISD. At noncanonical orbitals, validate the full commutator, including the
  singles-product subtraction, against an independent Fock-space oracle.
  Count complete symmetry coordinates before selecting contraction storage;
  a smaller coordinate count is neither an implemented large-pool solve nor a
  spectroscopy accuracy gain. This extends the existing convention/resource
  rules, with [evidence](atomic_reduced_sd_2026_10_09/README.md); no root rule added.
- Direct-core continuation: adapting a radial representation to reference
  diagrams requires reproducing the actual quadrature bilinear form. Keep
  component-specific node tags when kinetic-balance orders differ; the exact
  weighted Coulomb action must not receive a second Gauss-weight factor.
  Validate signed mixed components and the full noncanonical Fock commutator,
  then compare solved amplitudes, not only diagonal energies. Scope: reduced
  native atomic SD adapters; [evidence](atomic_pci_sd_native_2026_10_09/README.md).
  A converged complete-band core equation does not supply valence self-energy
  or establish partial-wave and experimental convergence. This merges the
  convention, resource and residual rules; no root rule added.
- Effective-operator continuation: when comparing redundant direct amplitudes,
  audit the fermionic gauge before diagnosing a physical defect. Preserve the
  full same-gauge signed comparison and separately prove that any gauge
  difference cancels in the physical operator. When checking an effective
  operator derivative, declare which energies move and re-solve every dependent
  amplitude; a stored single-energy or fixed-amplitude derivative cannot be
  renamed as a joint total derivative. For sensitive roots, measure conditioning
  and compare full forward responses independently; a small residual alone
  cannot certify response accuracy. Retain measured difference truncation
  errors when refining the numerical stencil. Improve the independent reference
  arithmetic when its residual and conditioning explain the comparison error;
  preserve the original physical inputs and acceptance gate. Before labeling a
  two-particle projection irreducible Sigma2, independently evaluate and subtract
  its embedded Sigma1 with identical denominators and signed pair conventions.
  Check simultaneous particle covariance and adjoint symmetry separately before
  consuming directed fixed-energy amplitudes in CI. A passed signed diagram
  gate does not construct a Hermitian effective metric; preserve the measured
  defect rather than silently averaging matrix elements. A non-Hermitian Bloch
  operator is not intrinsically a physics bug: reconstruct its physical wave
  operator and check H Omega=Omega H_B, H_B^dagger G=G H_B, left/right states,
  norms and full transformed properties. Do not infer this metric by fitting
  eigenvectors of an arbitrary approximate effective matrix. The exact finite
  native CI anchor establishes this distinction; the large SD consumer remains
  unverified. Cross-sector properties need both physical metrics and actual
  parent snapshots: the directed operator is G_b^-1 Omega_b^dagger O Omega_k,
  with a metric-weighted reverse adjoint. Bind both full CSF orders and signed
  spinors; a label match is insufficient. Build a separate photon operator at
  each many-electron line energy, retaining upper/lower statistical weights.
  A small angular coefficient can be a real component of a normalized
  degenerate-state rotation; do not discard it by a magnitude threshold.
  Test weak nonzero quantities with relative bounds and zero absolute tolerance;
  an exact floating equality can incorrectly reject a valid unit-CG contraction.
  Standalone monovalent SD uses core normal ordering: an ordinary
  attached-reference `[H,S_v1]` removes a required valence singles diagram.
  Check literal primary equations before promoting an attachment commutator;
  retain the full SD-state residual separately from its projected gate.
  In noncanonical scalar S1, expand all magnetic components on the bare
  closed-core Fock source; do not extend the attached full-H commutator into
  Coulomb contractions excluded by core normal ordering. Validate particle
  order with signed direct-rho expansion and occupied–virtual rotation.
  When assembling a physical SD norm, retain core spectators, valence Pauli
  blocking and signed core-S1/valence-S2 interference. Equal radial core holes
  still require both magnetic-hole permutations; an unconstrained redundant
  direct rho array need not already have particle-exchange symmetry.
  Use independent complete fermion expansions before radial Gram contractions;
  numerical coordinate overlap is a separate metric. Current private norm003 and
  multicore004 retain840 component/M gates after the preserved norm001 failure;
  The subsequent complete68-core/414-radial-virtual consumer verifies actual
  core-seed bitwise identity, all18M and source/payload bindings separately.
  This supports real linearized-wave normalization in the declared space, not full-H
  stationarity, dressed properties or correlation convergence.
  Near-degenerate unrestricted-M roots can acquire floating leakage outside
  target J: keep that failed comparison, decompose the complete angular
  space independently, and retain the full unprojected residual. A small
  energy defect alone does not certify forward state accuracy. Current
  `reduced_monovalent008` measures outside-J leakage 2.353e-11 versus within-J
  difference 2.17e-15; its 40 complete exact-J controls pass separately.
  Evidence: local `atomic_pci_sd_native_2026_10_09/monovalent_sd001/` failure,
  `monovalent_sd002/` singles-only diagnosis and `monovalent_sd006/` corrected
  all-M panel; [adopted contract](ATOMIC_LINEARIZED_SD.md#finite-standalone-monovalent-sd).
  An occupied–virtual orbital rotation changes the reference
  projector: reconstruct all Fock/integral identities before comparison and
  do not relabel that experiment as invariance of a fixed projector.
  Scope: native/pCI finite standalone SD and CI+all-order
  adapters; [source-bound full-vector evidence](atomic_pci_sd_native_2026_10_09/README.md).
- Large-band performance continuation: profile the actual diagram before adding
  workers. Keep cache capacity explicit and inside the memory preflight; a
  default intended for small integral consumers can repeatedly recompute a
  large core/virtual workload. Reuse a virtual-pair potential across all core
  pairs only after proving its physical inputs are independent of those pairs.
  For frequency-dependent exchange, keep transverse caches separate from DC
  and bind them to the complete fixed orbital-energy map. Bound all retained
  array caches before allocation; distinguish their payload from Python keys,
  input snapshots and active temporaries. Verify eviction with full signed
  matrix/stream replay. For complete norm Gram consumption, give each consumer
  a bounded cache and recompute complete evicted matrices; include view/cache
  metadata and active quartets separately from cache payload. A passing
  complete-space resource count still requires actual source-bound finite
  replay and full-workload consumption. [Adoption evidence](atomic_pci_sd_native_2026_10_09/parallel_tail007/norm_integration001/README.md)
  closes only the scratch budget, without pruning or increasing its original cap.
  Normal ordering must contract the same two-body
  operator into frozen constants, reference, one-body, Fock and ERIs; a scalar
  addition afterward cannot certify the CC Hamiltonian. Scope: fixed real
  transverse integral/FSCC consumers, not native reduced DC diagrams;
  [finite gates and failures](atomic_pci_sd_native_2026_10_09/README.md#动态-breit-的完整积分与有限-fs-ccsdt).
  When a new two-body operator makes Fov nonzero on formerly DC-optimized
  orbitals, recheck the full attachment Hamiltonian independently even if the
  external amplitude residual is tiny. Compare its core and each complete
  symmetry branch, then use a declared same-H tolerance intervention before
  changing equations. Changed-H Fov factors need consistent one-body/Fock
  reconstruction and their own full oracle; zeroing Fov is not a repair.
  Keep the original transverse0h1p strict failure separate from the later
  [finite equation repair](atomic_pci_sd_native_2026_10_09/parallel_fscc012/README.md).
  Lift actual bra-first amplitudes with conjugation before comparing complete
  signed Bloch rows. An antisymmetric input may already contain a permutation:
  the open Fov×T2 case needed one copy, and an additional P(34) doubled it.
  Check complex controls and unchanged integrals; a finite repair does not
  certify other sectors or large-space production accuracy.
  A complete core-cluster exponential plus intermediate-normalized attachment
  wave operator supplies a full signed Bloch-matrix oracle for tiny spaces;
  compare its whole effective operator, not only sorted energies. Reject a
  poorly conditioned model projector and keep reconstruction/normalization
  gates separate. This determinant diagnostic does not supply scalable CC.
  A successful progress receipt is also an evidence checkpoint: re-evaluate
  unresolved gates and choose the smallest registered distinguishing action.
  A stable same-model error plateau calls for an operator diagnosis before
  more mesh work. Separate selection, resource/lifecycle ownership and physics;
  an unavailable implementation is a visible frontier, not a successful job.
  This is the 2026-10-11 user policy; the
  [adaptive finite evidence](atomx_hourly_progress_2026_10_10/adaptive001/README.md)
  supports only the current transverse plateau and bounded task capabilities.
  Isolate OpenMP heap-backed workers from legacy large static scratch, validate
  complete signed actions and roots against the retained original path, and
  time alternating same-model actions. This isolation includes valence/screening
  translation units linked to OpenMP workers: compiling them with OpenMP
  moves legacy local arrays onto the stack, even with one active thread. Bind
  their full COMMON dimensions independently of precompiled placeholder
  modules; test the actual complete layout and entry, beyond a small fixture.
  A separately compiled heap-only valence diagram can use OpenMP after proving
  disjoint signed output channels and retaining the original complete map;
  include every worker buffer in planning and bind the actual inner-one BLAS.
  Complete Yb term2 replay supports this exception, while legacy units remain
  serial-static.
  A BLAS preparation gain or a smaller
  kernel allocation is not an end-to-end gain; retain slower warm-kernel
  results and do not adopt them. Scope: reduced SD/legacy Fortran adapters;
  [measurements and negative controls](atomic_pci_sd_native_2026_10_09/README.md).
  GPU continuation: count complete device batches and keep immutable physical
  geometry resident; preserve float64, tagged weights and all signed outputs.
  Measure transfer-inclusive complete maps against the actual CPU worker path
  before adoption. A resident density subblock can accelerate while small
  complete maps are dominated by dispatch/copy overhead. When a subdiagram
  enters a full map, preserve its additive role and invalidate resident fields
  at every owned COMMON initialization; a state label alone cannot identify
  cached physical data. The full-Yb s-density
  control passes all eight ranks and paired timing. Complete fixed-energy
  68-core/414-virtual SD roots and directed screening now pass their declared
  equation gates; physical convergence and end-to-end GPU gain remain
  unverified. This implements the user's
  2026-10-10 performance preference without extrapolating a subblock speedup.
  Numba continuation: share ordered scalar algorithms and physical data across
  CPU/CUDA, with explicit target scheduling and no implicit fallback. Verify
  compiler output for rounding/contraction assumptions. CUDA array-interface
  zero-copy views still need producer/kernel/consumer stream ordering; skip
  synchronization only after proving a shared stream or recording events.
  Complete eight-rank density and five-column frozen term2 controls support
  this narrow private implementation; their timing scope remains separate
  from full SD and large triples.
  Tensor metadata continuation: count pointer and inverse-index extents as
  well as amplitudes. Dense Cartesian indices can exhaust memory even when
  most tensor blocks are structurally absent. Compact complete keys must
  preserve original block identity/order, reject duplicates and overflow, and
  version persistence explicitly. Enumerate exact nonempty axes only after
  proving the original zero predicate factorizes; retain full symmetry and
  permutation checks. Finite CCSDT storage/restart controls support this
  representation change; they do not certify large triples contractions or
  a timing gain.
  Runtime continuation: pinning a serial BLAS to one inner thread does not
  certify concurrent caller safety. Reject incompatible BLAS thread models
  before preparation, bind the actual loaded runtime and repeat full signed
  actions and paired timing after replacement. The full-Yb concurrency failure
  withdraws the earlier serial-BLAS streamed performance conclusion; it does
  not change the retained numerical observations or the physical space.
  Solver reuse continuation: a recycled vector image must remain consistent
  with the actual complete operator, including when root and implicit-response
  right-hand sides differ. A small passing panel does not certify that reuse.
  Preserve raw failures, compare dense and native operators in the complete
  failing space, and test explicit image recomputation before adoption. The
  finite023 passing refresh was insufficient: independent013 still diverged.
  Complete root-dependent026 replay reproduces loss of image consistency and
  orthogonality even with a small intermediate raw residual. Two full MGS
  passes against all C/V pass finite027 and independent014; fresh public
  recertification015 also passes. These source-bound private controls do not
  establish a universal algorithm fix or production benefit. Public GMRES
  remains in use. For standalone SD, bare-ev diagrams supply a separate exact
  augmented pencil whose full images do not depend on coupling or feedback
  energy. Reuse complete basis images only with full MGS2 and actual complete
  public-map raw/energy checks at every continuation stage and at the final
  affine action; a passing small projected pencil is insufficient. Private
  `monovalent_davidson002` passes 84 independent finite checks, while seven
  explicit numerical tests reject stale images, complex branches and exhausted
  budgets. Counts fall from 262–698 to 22–25 at the same finite raw target;
  this is neither a universal image-cache guarantee nor measured end-to-end
  speedup. A wider numerical-capacity test must actually use dimensions beyond
  the old cap, and reserve the extra complete vector families before allocation.
  The numeric64 finite atomic panel remains bitwise identical with unchanged
  22–25 actions; the separate complete42-matrix stress uses 15/29/40 dimensions
  and 262/81/50 actions at capacities16/32/64. These observations do not establish
  full atomic numeric64 performance. See the [current owner](atomic_pci_sd_native_2026_10_09/README.md).
  Resident vector arithmetic must retain complete FP64 vectors,
  an owned allocator/stream, explicit combined device caps and fresh physical
  gates; a finite matrix pass alone does not certify full SD.
  Stage decomposition continuation: verify the actual complete sum and native
  COMMON extents before using component measurements to select optimizations.
  Legacy response-stage time cannot be summed to explain a value-only map.
  Batch only exactly reachable projection fields when all-field batching adds
  unused products; retain every original signed angular row. Full003 and frozen
  projection002 support these narrow controls, not production performance.
  Fuse signed angular accumulation only after checking complete output-channel
  disjointness in every retained subgroup; preserve row/group order and reject
  collisions. Frozen all-five-field replay and independent024 support this
  arithmetic specialization, not whole-map or controlled timing claims.

- Polynomial-density continuation: when fixed-cell spline products have bounded
  degree, retain every polynomial coefficient through tagged interpolation
  coordinates and transform the original weighted kernel as `A.T K A`. Do not
  replace native integration nodes or apply new quadrature weights. Check
  isolated mixed components, the native origin cell and complete signed
  equations/roots before using the smaller representation. Scope: the shared
  cell-Coulomb engine; complete Ne and finite full-core Yb anchors pass, while
  full-band Yb core signed/root gates now pass at their archived source.
  Complete valence, radial/correlation convergence and end-to-end performance
  remain open. This extends the existing
  exact-metric/performance insight, with [evidence](atomic_pci_sd_native_2026_10_09/README.md);
  no new root rule.
