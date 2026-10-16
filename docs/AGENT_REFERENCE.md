# Agent 开发参考索引

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

按任务读取相关入口；根 [AGENTS.md](../AGENTS.md) 是长期工作契约。
源码、当前测试和实际输入决定实现状态，设计提案或历史运行不能替代验证。

## 当前指南

- [EUV源模型及实验校准](EUV_SOURCE_MODEL.md)：多电荷态EOS、显式连续谱、有限真实共振非LTE能量闭合、分层辐射/波动激光和光滑动力学；原生相邻CI部分PI/RR、有符号Auger与精确角动量约化、配平Sn阈值及开放通道审计、绝对CE标定与2025勘误。160项相关检查通过（3项可选外部跳过）；完整能窗E1/1471共振速率和共同完整CI原驻值门已完成；部分直接EII、独立模型QED与有限15ns级联有各自数值控制，光学/EUV/CE实验验收仍未通过。

- [原生 MCDHF→RCI 与 Yb⁺ 实算](yb_mcdhf_rci_2026_10_11/README.md)：当前主线；真实共同五态梯度驻值、平衡核价/空穴、同态 RCI/实验差分与精确复用性能。SD/CC 为独立对照，首个裸参考不作关联空间认证。
- [Atomx原生原子物理路线](atomic_native_roadmap_2026_10_11/README.md)：恢复Ne后期约9/14meV原生谱位与Eu标定RMS8.2cm⁻¹成功实践；原生六方向并行，其他软件用于物理/精度/效率参考和独立对照，当前pCI/EXP-T依赖如实记录。
- [物理正确性与覆盖审计](atomic_physics_correctness_2026_10_11/README.md)：固定空间完整signed方程与两类二阶极限，公共非正则FSCC入口拒绝已知错误后端；保留秩正确与遗漏秩误差分开，优先物理补全再扫截断。旧方法定位的早期Ne/Eu实例不代表最佳实践，勘误见上一项。
- [物理量级与推进优先级](atomic_physics_budget_2026_10_11/README.md)：当前完整五态的绝对束缚偏差、重心与分裂，历史／文献／条件代理量级分开；先判别重要相关通道，再选完整同模型求根，未宣称已修好谱线。
- [外部精度与并行效率实测](atomic_external_benchmark_2026_10_11/README.md)：同完整 signed 输入的 AMBiT、pCI-derived SD、有限 EXP-T 对照；固定工作量、真实线程数、等待器开销与未通过 GPU 负载门分别记录。
- [FAC 完整精度与效率续测](atomic_external_benchmark_2026_10_11/fac/README.md)：192能级／2003积分的全部对称块、严格未通过门、四离子完整CI/E1流程和实际线程观测；各自优化模型与诊断导出负担分别标注。
- [CI 故障与文档维护记录](ci_repair_2026_10_10/README.md)：实际失败、相同物理输入的判别控制与验证；当前命令见 [测试](TESTING.md)。
- [开发维护](DEVELOPMENT.md)、[研究证据](RESEARCH.md)与[验证历史](VALIDATION_HISTORY.md)：按内容用途分流，不把历史测试计数当作当前状态。
- [Project index](../PROJECT_INDEX.md)：先定位模块、符号及静态测试候选；按需查询，不全文加载 JSON。
- [文档维护与蒸馏机制](AGENT_KNOWLEDGE_WORKFLOW.md)：随代码修改/合并/删除文档、记录、晋升与勘误。
- [洞察登记簿](AGENT_INSIGHTS.md)：可复用结论、状态、证据及规则去向。
- [文档总览](README.md)、[架构](ARCHITECTURE_DESIGN.md)、
  [能力与限制](ARCHITECTURE_CAPABILITIES.md)、[测试](TESTING.md)、
  [科学验证](VERIFICATION.md)。
- [公共原子态设计](ATOMIC_STATE_DESIGN.md)：区分已实现接口与拟议扩展。

## 按任务定位证据

以下入口为带日期的证据。只复用相同源/输入/模型下的结论；更晚的勘误优先。
旧 § 编号对应下方完整历史，方便 `rg '^## 27\.'` 等定点检索。

| 当前任务 | 建议先读 | 旧 § |
|---|---|---|
| 原子全阶 CCSD、FS-CCSDT 与完整核—价虚空间 | [当前接口](ATOMIC_FSCC.md)及[电子方程/规模实算](atomic_allorder_fscc_2026_10_09/README.md)：原生积分、真实外部张量后端、三电子完整 CI、复旋转、明确核层与完整能窗；[按需表示续篇](atomic_streaming_tail_2026_10_09/README.md)保留完整原生能窗并独立约束后端；[公共精确轴向分块续篇](atomic_pci_sd_native_2026_10_09/README.md#exp-t完整三激发的精确轴向宇称分块控制)保留有限19项全带符号/粒子/空穴/FCI检查；不是原生线性化 SD 或一般 CI+all-order | 新专题 |
| 原生标量核 SD 与角动量约化 | [当前接口](ATOMIC_LINEARIZED_SD.md)及[直接核 SD 续篇](atomic_pci_sd_native_2026_10_09/README.md)：共享原生 cell-Coulomb 精确特征、固定源码 pCI DC 核图类、完整非对角 Fock、矩阵自由求解与完整能窗。Be/Ne 独立交换子及振幅锚通过；Ne s/p/d/f/g 完整能窗已实算，可选精确BLAS初始化与共享虚对势/核对并行作用均有同模型完整向量验证和配对计时。精确胞内多项式表示通过完整Ne解和68核/14虚态Yb原作用残差门；新批计时未通过负载门或未配对。Yb 全核414虚态完整作用差1.39e-16Ha，36次作用核解原始最大残差4.44e-13Ha；公开精确term31/32全部坐标重放通过，32项组合live及后续26项完整初值核面板无跳过。完整Yb价列/屏蔽已启动，实验精度尚未通过。公共固定能量价SD/Σ₁接口及源码绑定构建已通过3项live测试；旋转原生DC的六径向列/16磁态完整作用、根、响应独立对照通过，索引/缓冲复用前后逐位一致。浅态响应失败004–008保留，009提高独立参考算术后通过原门。隔离65,536项Σ₂联合导数及非canonical Fock不可约费米锚通过；公共完整已解非canonical Σ₂及25案例全向量/联合导数验收通过，6项组合live测试无跳过；同案例定向消费审计发现明显非厄米差；共同κ能量部分降低Σ₁差，强真实Fov改参考投影控制全向量通过。有限CI已给出真实Bloch波算符/物理度量/左右态和完整性质，17CSF Be+原生锚通过；近似SD的CI能量/左右态/度量消费尚待验收，实验精度未完成。[此前有限 DC/DCB/冻结壳及计数锚](atomic_reduced_sd_2026_10_09/README.md)保持封存 | 新专题 |
| pCI / AMBiT / EXP-T 源码、构建与有效哈密顿量消费 | [当前有限 CCSD、完整 NR-SD 与 Breit 升级](yb_sd_breit_completion_2026_10_09/README.md)：CC 振幅求解、完整声明组态池、DCB 全 P/Q 和严格受限交叉项已实现；119 项测试通过，该封存阶段未完成大规模 SD/FS-CCSDT，当前后端见上一行；定向固定能量Σ₂已在当前原生SD接口实现，完整CI消费仍未完成。[前序同积分核对](ci_mbpt_matched_qspace_2026_10_09/README.md)及[源码下载/构建](ci_fscc_reference_upgrade_2026_10_09/README.md)保持封存 | 新专题 |
| 稀土自由离子谱线、寿命与实验对照 | [当前 SD/DCB 五态及 CCSD 算法锚](yb_sd_breit_completion_2026_10_09/README.md)：二空穴使四条谱位移动32–45 cm⁻¹，完整静态 Breit 后最大误差9.904%；数值通过、实验/空间失败。此模型未叠加旧闭芯 Σ₁，不与旧5.868%混用。[前序有限核心空穴五态探针](ci_mbpt_matched_qspace_2026_10_09/README.md)：原生裸 DC 互补 P/Q 与独立完整 CI 锚通过，Q 相互作用与能量参数迭代分开；有限空间未缩小实验差距。[带符号矩阵 Σ₂ 实现与轨道响应](yb_matrix_self_energy_2026_10_08/README.md)：99 项相关测试及两套完整五态计算/保存矩阵重放；最大谱位偏差 5.868%，物理门未通过；有限外部空间与完整高阶理论分开。[文献支持的下一阶段策略](yb_literature_repair_strategy_2026_10_08/README.md)：全阶 SD、互补 P/Q 空穴模型和二阶 Coulomb–Breit 交叉项；该文献记录未实算新谱线，本轮有限空穴探针见前述新专题。[前序 Yb⁺ 完整尾部与径向控制](yb_correlation_tail_2026_10_08/README.md)仍是测量依据：粗 ℓ8/200、细 ℓ6/200 两组完整五态及 ASD 绝对能量锚；数值/身份门通过，空间及实验目标未通过。[能量依赖阶段](yb_energy_dependent_sigma_2026_10_08/README.md)保留离壳极点、DF 分支 Dyson 及 137 项测试；[在壳与约定阶段](yb_diagnosis_repair_2026_10_08/README.md)保留独立 DF/偶极/电流/Breit 与 QED 告警；[原始基线](rare_earth_yb_spectrum_2026_10_08/README.md)保留七谱线、寿命及失败 | 新专题 |
| 原子/分子/晶体谱学、QED/XAS 与北京光源未来产业故事线 | [2026-10-06 封存研究体系与 12 个案例 setup](spectroscopy_science_story_2026_10_06/README.md)；三个已测接口与完整科学计划分开，当前软件能力见[能力指南](ARCHITECTURE_CAPABILITIES.md)。[此前 Ne 专题项目书](agentic_science_beijing_light_source_2026_10_06/README.md)保留作支持材料；当期赛题契约须另核 | — |
| 公共态选择、占率、态平均 | [态选择](atomic_state_selection_2026_10_04/README.md)、联合电荷优化 (local evidence: `xfel_charge_average_2026_10_04/README.md`)、磁态统计目标 (local evidence: `xfel_dc_audit_2026_10_04/README.md`) | 0.1, 6, 10, 13 |
| 轨道梯度、近根、驻值和残差度量 | 局部梯度与度量勘误 (local evidence: `xfel_local_gradient_2026_10_04/README.md`)、身份响应 (local evidence: `xfel_subspace_response_2026_10_04/README.md`)、边界曲率 (local evidence: `xfel_feasible_curvature_2026_10_04/README.md`) | 14–16, 22, 28–30 |
| 跨径向基底身份、简并子空间 | 基底跟踪 (local evidence: `xfel_basis_tracking_2026_10_04/README.md`)、活动空间 (local evidence: `xfel_active_convergence_2026_10_04/README.md`)；CI key 必须包含完整 coupling tree | 15–16, 37 |
| XFEL 一般 N 通道、连续态及模型边界 | 多电子损失 (local evidence: `xfel_multielectron_2026_10_04/README.md`)、径向快照/空间闭合 (local evidence: `xfel_ci_snapshot_2026_10_04/README.md`)、联合电荷 (local evidence: `xfel_charge_average_2026_10_04/README.md`) | 7, 10–11 |
| 近简并 reservoir、有限时间误差、RWA | 辐射界 (local evidence: `xfel_radiative_bound_2026_10_04/README.md`)、轨迹残差认证 (local evidence: `xfel_residual_certificate_2026_10_04/README.md`) | 8–9 |
| 完整 Breit、开放壳相位、QED/反冲约定 | 完整3200Ha、Breit约定与极点判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)、Breit 相位修正 (local evidence: `xfel_breit_open_shell_diagnosis_2026_10_04/README.md`)、完整修正与 QEDMOD 符号 (local evidence: `xfel_corrected_windows_2026_10_05/README.md`)、同步扩窗修正 (local evidence: `xfel_low_closed_corrected_2026_10_05/README.md`)、QED加入后的CI响应与轨道自洽审计 (local evidence: `xfel_qed_response_audit_2026_10_05/README.md`)、直接QED单电子轨道响应估算 (local evidence: `xfel_qed_orbital_probe_2026_10_05/README.md`) | 19, 26–27, 44, 47 |
| 复尺度、DKB、c-product、目标求解器 | 一电子核 (local evidence: `xfel_complex_scaling_2026_10_04/README.md`)、复 CI (local evidence: `xfel_complex_ci_2026_10_04/README.md`)、完整复 Breit (local evidence: `xfel_complex_breit_2026_10_04/README.md`)、目标谱 (local evidence: `xfel_target_poles_2026_10_04/README.md`) | 17–21 |
| 多电荷复尺度、保留池自适应和极点边界 | 完整3200Ha、Breit约定与极点判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)、多电荷控制 (local evidence: `xfel_multicharge_complex_2026_10_05/README.md`)、束缚邻域 (local evidence: `xfel_complex_neighborhood_2026_10_05/README.md`)、双外部12/16控制与实验差距 (local evidence: `xfel_complex_rank2_control_2026_10_05/README.md`)、角度/径向/算符判别 (local evidence: `xfel_complex_representation_control_2026_10_05/README.md`) | 45, 48–51 |
| 连续谱密度、冻结 DF/DKB、相关 rank/窗口 | 冻结 DF/DKB (local evidence: `xfel_frozen_df_dkb_2026_10_04/README.md`)、一般虚轨道 (local evidence: `xfel_general_virtual_2026_10_04/README.md`)、rank2 闭通道 (local evidence: `xfel_rank2_threshold_2026_10_05/README.md`)、能量零点 (local evidence: `xfel_dc_zero_audit_2026_10_05/README.md`)、同步低能窗口 (local evidence: `xfel_low_closed_corrected_2026_10_05/README.md`) | 31–39, 43, 46–47 |
| CI 装配、对称性、稀疏投影和缓存性能 | 物理对称性 (local evidence: `xfel_physical_symmetry_2026_10_04/README.md`)、稀疏装配 (local evidence: `xfel_ci_sparse_speed_2026_10_05/README.md`)、双激发符号 (local evidence: `xfel_ci_pair_speed_2026_10_05/README.md`)、径向键负结果/角缓存及 SHA 勘误 (local evidence: `xfel_radial_key_speed_2026_10_05/README.md`) | 11, 20–21, 40–42 |
| Ne q/r 实验/理论、跃迁自适应和误差分解 | 完整3200Ha、Breit约定与极点判别 (local evidence: `xfel_tail_breit_pole_2026_10_06/README.md`)、完整800→1600Ha、QED重解与f/g预检 (local evidence: `xfel_uv_tail_continuation_2026_10_06/README.md`)、完整200→400→800Ha与新单激发对照 (local evidence: `xfel_high_energy_pair_cutoff_2026_10_05/README.md`)、束缚rank3/n5/n6与先行电子对筛选（前期控制） (local evidence: `xfel_bound_rank3_n6_2026_10_05/README.md`)、13项因素筛选（旧n4/0控制） (local evidence: `xfel_factor_screening_2026_10_05/README.md`)、完整0–5Ha rank2原控制 (local evidence: `xfel_positive_rank2_closure_2026_10_05/README.md`)、第一性定位：核心电子对与条件零点 (local evidence: `xfel_first_principles_localization_2026_10_05/README.md`)、同 n4/5 算符分项 (local evidence: `xfel_first_principles_diagnosis_2026_10_05/README.md`)、跃迁反馈回放与候选规模 (local evidence: `xfel_transition_feedback_2026_10_05/README.md`)、精密文献审计 (local evidence: `xfel_precision_literature_audit_2026_10_04/README.md`) | 12, 23, 26, 44, 47 |
| 外部软件位置、构建状态、同径向跨代码验证 | 参考程序 (local evidence: `reference_programs_2026_10_04/README.md`)、GRASP 活跑 (local evidence: `grasp_diagnosis_2026_10_04/README.md`)、开放壳 Breit 对照 (local evidence: `xfel_breit_open_shell_diagnosis_2026_10_04/README.md`) | 24–27 |

## Task-specific scientific contracts

These operational details were moved from the root working contract when it
was rewritten in English on 2026-10-05. They remain binding for the relevant
task; read the linked implementation and evidence before using a particular model.
The root file keeps the general principles, while this guide owns the details.

### CI states, orbital objectives, and amplitudes

- [RCCSDT–EUV Playground challenge pilot](rccsdt_euv_challenge_2026_10_15/README.md)
  records the native finite Sn integral consumer, independent CCSDT/EA checks,
  offline task package and explicit gates for full experimental Sn validation.
  It does not change the production equations or the sealed Yb results below.

- [Current paper-alignment discriminator](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/paper_alignment004/README.md)
  compares primary RCCSDT/AMBiT/GRASP input contracts and opens the complete5p3/2
  core at fixed18 virtual spinors.1171 off-shell old-model recovery/sign gates
  and12 tensor tests pass. The22-correlated/46-frozen solve passes531 live gates,
  with all1292379 SDT equations and12 R3 partners. Lines rise815.16/1094.93cm^-1;
  experimental errors worsen to13.1460%/9.59434%. This is a controlled core
  closure, not recovered Sahoo input or experimental convergence.

- [Complete6p virtual continuation and four-model nonadditivity](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_virtual_p003/README.md)
  adds both6p multiplets at fixed18-correlated/50-frozen core,36 total spinors.
  All689589 SDT equations and12 target partners pass; a frozen4f companion
  closes the sd/sdp by4/18-core panel. Conditional line increments differ by
  441.12/489.20cm⁻¹; experimental gaps grow. Local DIIS has independent
  finite-FCI/old-Yb anchors and14 full calls in the large solve; no production
  API/source change or comparative solver-speedup claim.12 current tests pass.

- [Complete4f native CCSDT/EA and measured contraction performance](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_4f002/README.md)
  opens4f14:18 correlated core electrons,50 frozen,30 spinors. All retained
  equations/12 magnetic partners pass; lines shift upward, experimental gaps
  remain. Explicit canonical accumulation/exact scalar solve controls give a
  same-input1.74× residual-kernel speedup;59PASS/2 external-only SKIP,
  514 live/96 saved checks. Preserves SR-cap and small-J-basis failures before
  the highest-weight solve. No large full-H/physical-space/experimental claim.

- [Native tensor CCSDT/EA and both Yb D lines](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/README.md)
  adds attributed complete tensor equations, native-integral consumption and
  no full-particle-sector production enumeration. Actual16-spinor/4-core-electron
  calculations, complete target multiplets, independent sector CI/expm and
  196 live/353 saved controls pass;54 affected tests pass,2 external-only skips.
  The first shared-cache failure is retained.64 electrons including4f14 remain
  frozen; full paper/full-core/experimental precision is incomplete.

- [Actual-Yb native connected-triples pilot](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/yb_native_rccsdt001/README.md)
  records the user-authorized RCC comparison: four correlated core electrons,
  64 frozen electrons including4f14, complete6s/5d3/2 virtual multiplets and
  independent210/252-determinant CI. Nonzero iterative T3 and separate R3
  effects are measured; CCSDT/R3 line minus CI is−0.001114cm⁻¹. Full-H
  residuals remain nonzero, the experimental gap grows, and scalable/native
  production FS-RCCSDT and full paper reproduction remain incomplete.
- [Yb⁺ paper-DHF energies and static quadrupole comparison](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/yb_dhf_table001/README.md)
  binds two fresh68-electron cores, three attached states and their complete
  virtual columns. Original-physical-H/S roots, whole physical Gram, separate
  particle-sector CI and independently anchored static operators pass. This
  is an independentDC starting-point comparison; strict paper inputs,
  RCCSDT results and production MCDHF convergence remain incomplete.
- [Literature-guided analytic CI-response curvature](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision012/README.md)
  owns explicit polar-path, native real-DC radial first/second derivatives and
  complete isolated-real-state2n+1 response. Ten original Yb A/B targets predict
  all16 measured finite CI-response terms, max9.144e-13Ha discrepancy. This
  response-only coefficient excludes fixed-vector H3 and is not a total FD
  bound or optimizer admission. Fifteen new tests pass; two legacy regression
  failures reproduce in the complete before source with39 bitwise equal arrays.

- [Exact saved-H derivative decomposition and native centered replay](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision011/README.md)
  separates reported-root/Rayleigh, finite CI-vector response and fixed-vector
  operator terms. Existing current dense centering improves the literal-H
  replay to38/40 ordinary comparisons; two large-step D3/2 gates still fail.
  Original005 remains failed. No new physical H/gradient or optimizer admission;
  independent constrained CI-response curvature is the next discriminator.

- [Streamed signed DC density and complete radial-gradient qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision010/README.md)
  owns the explicit density-storage/batched-bilinear contracts,113 affected
  tests,357 live and102 saved-array Yb controls. Raw vectors require explicit
  caller binding; analytic-gradient agreement does not supersede failed
  ordinary finite differences or admit an optimizer continuation.

- [Explicit numeric CI streaming and complete missing-class measurement](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision009/README.md)
  owns the current opt-in storage contract, independent signed Fock/RKB tests,
  complete A/B source qualification and full4f12 physical-run status. The default
  remains cached; orbital response and full-spectrum ownership need separate
  qualification.

- [Exact angular row streaming and spectator compaction](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision008/README.md)
  links independent signed CSR tests, complete A/B comparisons and the largest
  full4f12 structural graph. Graph qualification is not H/CI or energy admission.

- [Matched native sparse CI and density qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision006/README.md)
  records complete Yb A/B operator/eigenpair and signed radial-key controls,
  internal CSR projection storage, and its then-remaining full4f12 angular budget.
  A selected-root solve is not a complete-spectrum tracking/response certificate.

- Select levels with `CILevel(J2, parity, root)` and unpolarized populations
  with `CILevelPopulation`; use `CIStateAverage` / `CIChargeStateAverage` for
  orbital objectives. Roots are zero-based energy ranks within symmetry blocks.
  Integer columns are valid only within their bound basis snapshot.
- `CIStateAverage()` weights resolved levels equally; `statistical=True`
  multiplies by `2J+1`. Physical level populations are total level weights,
  distributed over M without an additional degeneracy factor. Changing the
  preparation does not trigger SCF or change operators.
- For exact same-J/parity degeneracy, include the complete subspace with equal
  unpolarized populations or optimization weights. Coherent preparation uses
  an explicit density in a specified magnetic basis. Reject ambiguous tracking;
  do not reset its reference, reweight, or change roots to bypass failure.
- Common-orbital relaxation optimizes the CI Hamiltonian average in one
  orthonormal basis. Shared orbitals alone do not imply joint charge optimization.
  Independently relaxed initial/final states require overlap/biorthogonal
  transformations of the full states and operators before combining amplitudes.
- Common-orbital Auger retains the full signed `h_D + V` amplitude. In h/V
  decompositions, retain channels whose total amplitude is exactly zero while
  the separate contributions are nonzero; cancellation is part of the result.
  Neither proximity to a reference nor weak strength justifies deleting terms.
- Bind angular cache identity to complete CSF ordering, coupling trees, and
  operator contracts. `CSFSignature.key` supplies complete identity; `repr`
  omits coupling information. Cross-basis overlaps use signed determinants and
  minors, without inverting the cross-overlap matrix or discarding weak coefficients.

Current scope: [atomic-state design](ATOMIC_STATE_DESIGN.md),
[level selectors](../src/atomx/physics/csf/levels.py),
[averages](../src/atomx/physics/csf/state_average.py), and
[explicit basis tracking](../src/atomx/physics/csf/tracking.py).
Evidence and corrections are routed by the task table above.

### Continuum, derivatives, and reference conventions

- For physics changes, rerun the relevant live gates and update their `MEASURED`
  panels. Unfinished scientific targets use strict xfail with a measured panel;
  label `KNOWN GAP` explicitly and predeclare matching rules rather than retuning them.
- Declare continuum potential, exchange, boundary conditions, energy
  normalization, no-pair projection, and bound-space projection separately.
  The common-orbital CI channel model uses frozen final-ion direct Hartree
  Dirac distorted waves with bound-space projection. Its direct field includes
  coherent same-kappa RDM cross terms; check `Tr gamma = N` and the tail field.
  Bound-space projection does not supply nonlocal exchange. Other continuum
  entry points have their own model contracts.
- Verify Wigner algebra numerically with SymPy: atomx interfaces use doubled-j
  integers, while SymPy uses physical j. Validate full off-diagonal phases and
  interference in mixed operators, not only diagonal or closed-shell anchors.
- Test orbital/CI gradients on the actual metric-orthogonal no-pair manifold,
  including re-diagonalization response and signed one-/two-body RDMs.
  Near crossings, distinguish state identity, derivative accuracy, the native
  residual metric, and stationarity. An overlap or line-search pass is insufficient.
- For changes in a space or operator, solve the ground and upper states together.
  Report `delta E_g`, each `delta E_upper`, common transition motion, and splitting.
  Do not combine old ground energies with new upper energies or assume that
  overlapping space increments form an additive error budget.
- Inspect reference fixture generators, `script` / `run.sf`, nuclear models,
  units, normalizations, and degeneracy averages before numerical comparison.
  FAC and pure-CI/UTA factors discussed below apply only to their stated fixtures.
  Do not import published Breit/QED/recoil offsets merely to close a discrepancy,
  or add a self-energy already represented in retained CI space.
- Radial caches bind orbitals, potential, nucleus, grid, projector, energy, and
  normalization. A new trial uses a new snapshot; do not reuse cached column
  identities across snapshots. For Numba's “underlying object has vanished,”
  inspect the relevant `.nbi` / `.nbc` files first and clean only generated caches.

Implementations: [CI continuum](../src/atomx/scattering/ci_continuum.py),
[many-electron channels](../src/atomx/scattering/multielectron_ci.py), and
[radial snapshots](../src/atomx/physics/csf/radial_snapshot.py).
Verification entry points: [testing](TESTING.md), [scientific verification](VERIFICATION.md),
and the relevant task/evidence links above.

### Complex scaling and resonance claims

- Use the c-product (transpose, without conjugation). `H.T == H` expresses
  complex symmetry and does not imply Hermiticity.
- Rotate Dirac kinetic/potential terms, DKB representation, and projections
  consistently; the `-2c^2` mass offset does not rotate.
- Complex coefficients squared are not populations. An ordinary box root's
  imaginary part is not automatically an Auger width.
- A local theta plateau or one bound-state imaginary-part gate does not certify
  a physical pole. Recheck radial, continuum, and correlation-space convergence
  when the representation changes. Separate model-error bounds and numerical
  refinement; time-step differences are not strict interval enclosures.

See [complex-scaled Dirac](../src/atomx/numerics/bspline/complex_dirac.py),
[complex CI](../src/atomx/methods/csf/complex_ci.py), and the complex-space
evidence/corrections linked in the task table.

## 易误用的约定与证据边界

- FAC DR/Burgess fixture 的 `a_radiative` 是级总 TR/(2J+1)，不是普遍适用的
  所有 FAC 输出约定。先核对生成脚本。FAC jj 平均需分别声明 p−/p+ 组。
- Auger 纯 CI 磁子态锚 `4πR²` 与 UTA 平均锚 `2πR²` 不互校；
  辐射线 FWHM 为 ħ(Γ上+Γ下)，重叠判据只用参与的态。
- 受限组态空间可能不对同 κ 轨道旋转闭合。“完整”需声明保留轨道、
  κ/l、external rank、能量窗口及全部新旧/跨 κ 配对；束缚池不代表连续覆盖。
- 身份匹配使用完整 `CSFSignature.key`，`repr` 缺 coupling tree 会碰撞。
  跨基底重叠用有符号 `det S`/minors，不对交叉 S 求逆或截弱系数。
- 原 n4 majority 目标驻值仍未通过；独立冻结 DF/DKB 对标可以另行推进，
  不能互相替代目标或验证。§16 的残差重放失败须结合 §22 的度量勘误。
- Ne q/r 的分裂接近不证明绝对中心收敛；Breit/QED 共同模可达数百 meV。
  实验与两列理论引用按原表及相关误差，不能由舍入中心重建差值。
  同 n4 分项相对2017印刷表的 DC 差约−273.5/−274.2meV，支持先查相关/表示闭合；
  跨模型分项差不等于唯一物理误差预算，共同线移本身不能独定位到基态。
- §47 完整修正的 10→5Ha 窗口仍移动约 61meV，绝对线位 1meV 门失败。
  后续 n4 束缚控制又移动约 10.3meV；完整 Q 跃迁反馈可作已测模型内优先级，不能作相关尾项误差界或实验精度。
  §49 rank1局部邻域通过；加入rank2后须重新验收，最新角度/径向/算符判别见专题入口。二维驻点/相关闭合/物理极点未认证。
- 早期“未实现”可能已在后续实现，如联合电荷优化、复 Coulomb/静态 Breit；
  隔离驱动中的完整 QEDMOD 不等于公共接口具有全部非局域覆盖。
  读取实际入口及算符 supported/applied/skipped 记录后再描述能力。
- §40–42 的 CI 装配观测均未通过原负载守卫；暖角缓存收益不是新的首次
  装配或整体加速证书。§42 还修正了 §41 的完整源 SHA 验证覆盖。

## 本地参考源码

已知入口：FAC `/data/code/fac`，XATOM `/tmp/atomx-xraypac-build/source`；
GRASP、AMBiT、pCI、QEDMOD、RATIP 等具体路径/版本/构建见参考程序报告。
使用前检查实际存在与版本，不从软件名推断精度或已构建状态。

## 完整历史与维护

[原 AGENTS.md 快照（2026-10-05）](archive/AGENTS_2026_10_05.md)
按字节保存优化前工作树的全部 1,297 行（旧 §0–48，包括既有未提交增量）。
SHA-256：`498e469e963b5e6ff5d8c0c5460d6c2a64a9d00777b480836da055185936d9fe`。
该快照保留当时陈述及后来被证伪的结论，仅供历史检索，不是当前指令。
快照内 `docs/...`/`src/...` 路径均相对仓库根，不相对其归档目录。

新增实验数据按记录与蒸馏机制进入所属专题目录，`trace.md` 仅作会话账本。
本索引每个主题保留当前入口与必要勘误，完整运行列表留在专题目录。
只有通过晋升条件的长期规则进入根 AGENTS.md；后续修正追踪到洞察和证据。

Exact highest-component PI/RR magnetic reconstruction and the conditional
3.165-million-line cascade are recorded in the [EUV source guide](EUV_SOURCE_MODEL.md#exact-highest-component-dyson-reconstruction-for-pi-and-rr).

Native fixed-space Sn13+ core/valence FSCC equation controls, negative experimental
gates and the non-additive triples/core increments are in the [source guide](EUV_SOURCE_MODEL.md#fixed-space-native-sn13-fscc-core-valence-controls).

The same source guide owns [full9636-channel RR refinement](EUV_SOURCE_MODEL.md#exact-highest-component-dyson-reconstruction-for-pi-and-rr),
[20-terminal M1/E2 scope](EUV_SOURCE_MODEL.md#conditional-capture-and-complete-window-radiative-cascades)
and [canonical blocked-gradient/ground variational comparison](EUV_SOURCE_MODEL.md#measured-exact-angular-reuse-in-the-full-ci-objective).
Their numerical gates do not imply experimental agreement; the mixed-CI
frozen-source and signed finite-difference failures remain explicit in testing.

[Coherent native CI direct EII](EUV_SOURCE_MODEL.md#explicit-coherent-ci-direct-electron-impact-ionization)
records exact unpolarized removal-density reduction, explicit radial-channel
DW reuse, matched CI thresholds and the still-failed Sn partial-wave gate.
The full-CI stationarity gate now passes, with the subsequent static-DCB
experimental gap explicitly retained in the same guide.

[Native bound model-QED qualification and composition](EUV_SOURCE_MODEL.md#native-model-qed-operator-qualification)
records same-Fermi/C potential controls, the five-state Q-component unitary,
complete signed Fortran/native matrices, grid refinement and the separately
active full amplitude solve. Its model range, numerical failures and unfinished
experimental gate remain explicit; it is not the public local-QED backend.

The [finite capture time-window contract](EUV_SOURCE_MODEL.md#finite-capture-time-window)
owns positive finite-time triangular integration, temporal grid refinement
and the6291-state conditional15ns result; it supplies no missing kinetics.

The [finite core-hole operator controls](EUV_SOURCE_MODEL.md#finite-second-order-core-hole-operator-controls)
own independent full-P/Q/complex native anchors, complete induced Sigma0..3,
and sparse canonical/active-CSF projection qualification. Induced Sigma3 is
separate from clusterT3; the partial Sn n4 result still fails experiment,
and full selected-sector use does not supply external-virtual convergence.
