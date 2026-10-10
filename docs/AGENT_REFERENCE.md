# Agent 开发参考索引

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

按任务读取相关入口；根 [AGENTS.md](../AGENTS.md) 是长期工作契约。
源码、当前测试和实际输入决定实现状态，设计提案或历史运行不能替代验证。

## 当前指南

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
| 稀土自由离子谱线、寿命与实验对照 | [Yb⁺ 实算](rare_earth_yb_spectrum_2026_10_08/README.md)：V68 冻结核、5 能级、7 谱线、2 寿命；数值与实验门限分开，含完整声明的 4f 空穴空间、静态 Breit 和近似 QED 敏感性 | — |
| 原子/分子/晶体谱学、QED/XAS 与北京光源未来产业故事线 | [当前研究体系与 12 个案例 setup](spectroscopy_science_story_2026_10_06/README.md)；三个已测接口与完整科学计划分开。[此前 Ne 专题项目书](agentic_science_beijing_light_source_2026_10_06/README.md)保留作支持材料；当期赛题契约须另核 | — |
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
