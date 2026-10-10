# 原子态、弛豫与 XFEL 的统一设计

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

日期：2026-10-04。用户要求：尽可能自洽、复用模块且高性能；可自由设置
基态、激发态与制备方式。本文件区分已接入的增量与后续设计。一般 N
有限空间 REXMI 链已接入，完整物理与任意脉冲认证仍未完成（见 §8）。

## Current implementation status

Checked against the working-tree source on 2026-10-05. The numbered increments
below retain their progression and measured evidence; statements about what an
earlier increment lacked do not override this current interface summary.

| Contract | Implemented entry point | Boundary |
|---|---|---|
| Symmetry/root selection and unpolarized preparation | Public `CILevel` / `CILevelPopulation`; supported coherent drivers' `initial_state` | Energy-ordered roots are not configuration-character tracking |
| Fixed single-/multi-charge orbital objectives | Public `CIStateAverage` / `CIChargeStateAverage` via `Correlation(state_average=...)` | B-spline common-orbital MCDHF; joint charge averaging requires `corrections="none"` |
| Frozen physical subspace tracking | `CIBasisTracking`, explicitly supplied as `level_tracking` to lower-level orbital functionals | Ambiguous/lost identity rejects a trial; not enabled implicitly by `CILevel` or `Atom` |
| Shared radial integrals | `CIRadialSnapshot` and prepared common-orbital channel providers | Reuse is tied to the full owned Hamiltonian/radial snapshot |
| Finite-space coherent loss dynamics | `TwoElectronCoherentModel` / `MultiElectronCoherentModel`, complete retained-orbital residual sectors | Projected frozen-ion direct Hartree DW and explicit reservoir gates |
| Unified views and independent-orbital transformations | Proposed `atom.state(...)`, `AtomicStateView`, and structured `PreparedDensity` | Not implemented; no implicit dynamic orbital relaxation or biorthogonal bridge |

The source contracts are in [level selection](../src/atomx/physics/csf/levels.py),
[state averages](../src/atomx/physics/csf/state_average.py),
[basis tracking](../src/atomx/physics/csf/tracking.py), and the
[single-charge](../src/atomx/methods/csf/variational.py) /
[joint-charge](../src/atomx/methods/csf/charge_variational.py) functionals.
Current [architecture](ARCHITECTURE_DESIGN.md) and
[capabilities](ARCHITECTURE_CAPABILITIES.md) provide the module-level summary.

## 1. 五个独立输入

| 输入 | 决定什么 | 当前实现 | 需要扩展 |
|---|---|---|---|
| CI 空间 | 固定电子数、组态、J/宇称块、相关空间 | 正式 jj CSF 枚举、完整残余 Fock 扇区及成本 cap | 自适应大空间/相关空间收敛 |
| 轨道优化目标 | 对哪些 CI 能级的能量求驻值 | 基态、固定单/多电荷 EOL、显式低层基底跟踪 | 统一活跃/冻结轨道及公共跟踪输入 |
| 性质的物理态 | 能量、密度、占据、散射及跃迁的输入 | 结果默认基态；跃迁可分辨 CI 能级 | 不修改结果的显式 state/ensemble view |
| 动力学制备 | 初始占率、磁子态及相干相位 | CI 选择器、非偏振混合；显式磁密度 | 绑定基底的纯态/密度对象、离子和跨扇区混合制备 |
| 能量参考 | 显示及旋转框架的原点 | 当前共享扇区能量与显式 photon ladders | 统一能量零点及制备/观测时间记录 |

组态只是展开基底。能级是其中哈密顿量的本征态；强混合时不存在唯一的
“该组态激发态”。优化平均的 RDM 用于轨道方程，实验占率的 RDM 用于
物理观测，二者不可相互替代。能量原点不选择基态，也不能分别移动每个
电荷扇区后仍用原差值计算 PI/Auger 阈值。

## 2. 第一增量：公共能级选择与物理初态（已实现）

`atomx.physics.csf.levels` 负责选择、谱验证和简并门禁。
`CIStateAverage` 复用同一个 J/宇称分块及权重归一化内核；不复制 SCF、
角代数、RDM 或 XFEL 求解器。

```python
from atomx import CILevel, CILevelPopulation, CIStateAverage
from atomx.xfel import MCDHFCoherentModel

# root 从零开始，在 J=1、奇宇称块内按能量排序；不是全谱第 1 列。
excited = CILevel(J2=2, parity=-1, root=1)
prepared = CILevelPopulation(((CILevel(0, +1, 0), 1), (excited, 3)))

driver = MCDHFCoherentModel(
    solved_atom,
    photon_numbers=all_level_photon_ladders,
    initial_state=prepared,
    bound_only=True,
)
trajectory = driver.solve(pulse)
```

此示例需调用者提供已收敛的共同 B-spline 轨道原子、完整 photon ladders
与相干脉冲；能级须存在并通过适用性门禁。`TwoElectronCoherentModel`
的 `initial_state` 接受相同类型，当前选择父二电子扇区。两个入口仍接受
旧式整数列号与 `solve(initial_density=...)`。

- `CILevel.resolve(csfs, result)` 得到当前谱中的列号，跨 J/宇称能量
  交错不会选错块。它并不保证跨轨道迭代的组态特征不变。
- `CILevelPopulation` 的权重是**总能级占率**，归一化后每个 M 得到
  `p_level/(2J+1)`。不额外乘统计简并度；无相干的非偏振密度有明确含义。
- 优化对象 `CIStateAverage` 不能作为动力学制备传入。现有优化输入仍为
  `(J2, parity, root, weight)` 元组；`statistical=True` 是显式优化策略。
- 精确同对称简并的单根/不等权混合拒绝；整个简并空间等权的密度在
  任意本征向量旋转下不变。整数列号仅是已记录快照的显式规范选择，
  保留兼容性；可重复的相干制备应直接指定磁密度。
- `default_initial_level_populations`（共同扇区模型）或
  `default_initial_parent_level_populations`（联合扇区模型）记录默认
  制备，不冒充覆盖 `initial_density` 后的实际输入记录。
- 选择不同初态不重做轨道优化，也不改变原子的跃迁/连续态算符。

第一增量尚未实现 `atom.state(...)`、特征 root tracking、指定纯 M 的
结构化接口或任意离子扇区制备。这些应按下述契约实现，而非重新塞进
`config` 或把 `.total_energy`、`.occupations` 改成优化平均。

## 3. 物理态视图（后续设计）

拟议 `atom.state(CILevel(...))` 返回不可变 `AtomicStateView`，绑定已求解
的原子快照：核模型、哈密顿量修正、CSF 顺序、轨道/积分空间、完整
CI 能量和混合矩阵的身份。视图同时记录原始选择器与已解析列号。
“ground”须明确是整个已保留空间的最低能子空间；某 J/宇称块的 root 0
不一定是整个原子的基态。实验基态也需要确认所建空间涵盖实际低能态。

性质通过视图的态密度消费公共算符：

\[
\gamma_{ab}=\mathrm{Tr}(\rho\,a_b^\dagger a_a),\qquad
\Gamma_{ab,cd}=\mathrm{Tr}(\rho\,a_a^\dagger a_b^\dagger a_d a_c).
\]

先实现分辨能级与非相干能级混合 RDM，复用 `CIRadialDensity`；相干
密度需要完整交叉 RDM，而不能仅把本征向量系数平方当作占率。检查
`Tr γ = N`、对称性、Hermiticity、正定性及完整哈密顿量能量闭合。
纯 M 态可能具有非球对称多极密度，不能无提示压成球平均密度来做成像。

性质调用显式传 `state=` 或 `initial=/final=`；原子已有默认基态接口
保持兼容，但结果里标记其状态定义。状态转换是读操作，不隐式启动
SCF。不同视图共享昂贵原子数据，仅缓存与状态相关的收缩。

## 4. 优化目标、弛豫及态跟踪

在同一正交轨道空间优化完整 CI 哈密顿量平均：

\[
\bar E[\phi]=\sum_{N,\alpha} w_{N\alpha}
 E_{N\alpha}[\phi],\qquad \sum w_{N\alpha}=1.
\]

**当前已实现单 N 固定 EOL 及 `CIChargeStateAverage` 多 N 平均（§10）。** 每扇区
独立 CI 对角化，但共享一套径向轨道/no-pair 投影及变分约束。共同优化
的轨道梯度来自同一个完整 h+V/RDM 内核。不同 N 扇区的权重是制备
相关的近似选择；必须固定、记录并做敏感性对照，不靠 FAC 数率拟合。

冻结轨道、活跃轨道和轨道更新策略应该独立于物理初态。只优化基态时，
高能相关轨道可能只用于相关展开；不保证它能作为真实束缚离子末态。
应用电离通道前必须检验末态、阈值及径向/空间收敛。

能量排序和特征跟踪是两个显式策略。特征跟踪须依赖态/子空间的重叠，
跨不同轨道基底先做合法的重叠变换；CI 系数点积一般不代表物理重叠。
接近同对称根交叉时记录重叠质量及子空间变化。真正简并按子空间跟踪，
不得以随意本征向量的相位或主组态标签选一个根。优化收敛门继续检查
流形切向梯度；根换号或能量看起来稳定都不等于已收敛。

若选择独立初态/末态轨道弛豫，先实现重叠/双正交及 CSF/算符的完整
变换，再消费跃迁和连续损失振幅。结果不能混入假设共用正交轨道的
动力学入口。多电荷共同轨道是首先推进的可复用路线，独立轨道是显式
后续能力。相关理论与双正交限制见 [GRASP 理论原文](https://www.mdpi.com/2218-2004/11/1/7)
及 [GRASP 用户手册](https://www.mdpi.com/2218-2004/11/4/68)。

## 5. 连续态：DW 回答及演进路线

当前两电子 CI 连续态的实际调用链：

1. `TwoElectronCIChannels._field(final)` 用冻结的残余离子轨道构造
   直接单极 Hartree 势与核势；裸核用核势。它没有完整交换或离子响应。
2. `_prepare_waves` 调用共享 `solve_dirac_continuum_many`，数值解
   Dirac 径向方程、Coulomb 渐近匹配及 δ(E) 能量归一化，不用平面波。
3. `_store_wave` 在原生积分网格上投影掉所有保留的同 κ 束缚轨道，
   使用 Gram 矩阵检验；渐近归一化不因局域束缚尾部投影而改动。
4. PI 用有符号 Dyson 振幅，Auger 用完整 Dirac-Coulomb h+V。
   公共 Dirac 作用、径向与角系数内核均复用。

这是**投影后的冻结离子局域直接势 DW 近似**。投影提供 Pauli 空间
正交性；它不等于已经求解完整非局域交换方程。投影波在内部不再严格
满足未经投影的局域势方程，不能把此步骤隐藏在“自洽”标签后。
数值正交、网格收敛与正确能量归一化也不证明缺失交换/通道耦合足够小。
这里只描述 CI 通道；其他 Gu 等连续势路径不能混称为同一物理模型。

建议分阶段：先复用公共轨道/RDM 与 Dirac 作用实现显式残余离子
静态交换，并检查投影后的方程残差/Pauli 一致性；随后在确有需求且
独立验证后加入极化与多通道耦合。耦合共振应进入振幅层或有效算符，
不是靠平滑 σ 上随意加 Lorentzian 并忽略相位。一般 N 的 PI/Auger
已经沿磁子态、Dyson、Slater–Condon 链推广，保留明确能力门禁。

缓存依赖原子快照、末态/末态密度、势模型、κ、电子能量、边界、投影
空间及归一化约定。当前缓存局限在 provider 内；未来共用缓存必须
包含这些键。脉冲光子能量或时宽改变后，已有定中心能量损失张量不可
继续复用；谱积分与中心能量近似须显式选择。

近阈值记忆、Fano/通道耦合与完整真空环境仍未实现。已接入显式近简并
簇近似及有符号张量/谱系数变分门；不能通过换初态或放松门禁宣称
任意多电子 REXMI 已认证。

## 6. 动力学制备、观测与规模

拟议制备对象统一支持：单能级非偏振、任意非偏振能级占率、指定 M
纯态、同电荷磁基底的相干纯态/混合态，以及跨电荷的经典混合。
通常制备的不同电荷扇区作直和、无跨电荷相干；要求跨电荷相干时必须
给出相位参考/相关环境模型，不把任意矩阵当作孤立原子合法制备。

密度输入记录基底身份、态顺序、参考时刻、旋转框架和实际矩阵摘要；
允许显式密度覆盖默认占率，但输出必须区分默认设置与实际制备。
只在同一快照内复用整数列号。跨快照输入必须合法变换或拒绝。

观测器消费同一快照的 O(t)：`Tr[ρ(t)O(t)]`。瞬态成像复用已有 f0/f1/f2
与观测驱动接口，先支持可验证的分辨态/非相干混合；相干态的完整
散射强度可能需要跃迁响应与二点关联，不可仅对组态 f 做占率平均。

对高性能的结构性选择：

- 不变角表/CSF 连接图只构造一次；轨道变化仅更新不同的径向积分。
- 同一谱分块排序只在准备或新 CI 对角化后发生；时间步内不解析选择器。
- 同一末态、κ、能量的连续态批量去重，共用 Dirac/径向算符。
- 多扇区先构建稀疏通道图；密度和 Lindblad 操作按适用的对称块组织。
  强驱动近简并需要保留相关相干；不能为省内存自动替换为占率方程。
- 在世俗/退相干假设已认证的区域，显式选择率方程/MC，从同一振幅层
  推导率并核对弱场极限。只占率观察与全密度轨迹应有明确输出策略。
- 保留磁态规模上限和成本预估；不能在状态选择时自动删掉可能到达的
  末态或虚拟耦合，否则会悄悄改变动力学模型。

## 7. 推进顺序与验收

| 顺序 | 交付 | 独立门禁 |
|---|---|---|
| 已接入 | 共享 CILevel/CILevelPopulation，两个 CI driver 的父态初始化 | J/宇称能量交错、简并旋转、零通量指数衰减、默认算符等价 |
| 下一步 | 绑定不可变快照的 AtomicStateView/PreparedDensity | 迹/粒子数/相位协变、错误基底拒绝、覆盖初态的实际记录 |
| 下一步 | 明确态输入的性质/RDM/成像消费 | 完整 h+V 能量闭合、磁多极检查、解析/独立参考 |
| 已接入 | 一般 N→N−1 有限空间共同轨道通道 | N=2,3,4 独立费米锚、完整残余空间、真实 N3 电离、相位/时间/径向门 |
| 已接入 | 多 N 共同轨道平均；显式低层基底跟踪 | 多电荷切向梯度与跟踪身份门；末态/活跃空间收敛仍需独立检验 |
| 随后 | 静态交换 DW、合法的独立轨道转换 | 方程残差、渐近相位/归一化、正交、PI/RR 互逆与参考对照 |
| 最后按适用性 | 非世俗、多通道/近阈值及一般 REXMI | 弱场、时间/空间收敛、概率守恒、真实共振电离与实验/代码比较 |

上一轮独立实验见 CI 态平均归档 (local evidence: `xfel_state_average_2026_10_03/README.md`)：
Ne EOL 的离子根均束缚；statistical EOL Auger 相对 FAC 仍高 15.53%，
原因未隔离，不能归因于某单项或调整权重抹平。N3 的 85.97% 峰值是
bound-only 共振占率，电荷恒为 7。历史归档保持原样；新接口的实现
和验证记录应另建目录。低层 `CIBasisTracking` 的当前契约见本页状态表。


## 8. 第二增量及识别边界

一般 N CI REXMI 归档 (local evidence: `xfel_multielectron_2026_10_04/README.md`)保留三电子真实
共振电离、四电子费米/PI 检查及 K 共振门禁失败。N3 的末电荷现在升到
7.18726049；此前 bound-only 的电荷恒为 7 的记录保持原样。

受限四电子 EOL 空间 `1s2 2s2 / 1s1 2s2 2p1` 最强基态线为 25.1036 eV；
增加 `1s2 2s1 2p1` 空穴伙伴才恢复 897.2415 eV 的 K 区强线。这说明
CI 优化与轨道旋转可改变名义壳层标签的身份。能量排序的 `CILevel.root`
依然不是特征跟踪，实际孔态身份需要径向/密度和空间闭合检验。
补齐伙伴后的四电子低能辐射簇仍被 1e−3 谱系数变化门拒绝，未改门限。

prepared channel provider 可直接复用于 driver，共同缓存包含末态势、κ、
ε 和投影约定；回归检查禁止相同表再次调用 DW 求解器。它不提供跨
原子快照的全局振幅缓存，也没有取代尚未实现的 AtomicStateView。

Finite-time reservoir certification and exact sparse representation have since
been added: see the reservoir increment (local evidence: `xfel_radiative_bound_2026_10_04/README.md`).
It preserves the state/optimization distinction, signed operators and old gates.
N3 is verified under explicit conditional error budgets; the complete N4 K chain
still fails its conservative budget. No joint-charge relaxation, root tracking,
nonlocal exchange, vacuum shifts or microscopic bath certification is implied.

## 9. 轨迹比较与光子阶梯身份

轨迹残差增量 (local evidence: `xfel_residual_certificate_2026_10_04/README.md`)现已在保持1e−3
预算和旧默认门的条件下验证有限 N4 parent K pumping；此结果更新上一节
“仍失败”的当前状态，保留历史归档。photon ladder 应绑定实际谱学态/
基底快照：激发根不自动等于载频附近的根，新增孔态伙伴可能引入低能
价层谱。当前 N4 按实际 K-window 指定 parent ladder，daughter 显式0。

密度初态、仿射生成元及参考算符共同决定严格可达的 Liouville 子空间。
这一表示可以复用同一算符层，保留所有相位和微小振幅；不按优化权重
或壳层标签删支路。残差积分给出精确轨迹的解析模型比较，上层记录
独立的数值加密证据，不能把它混成微观环境或严格数值区间证明。
本增量当时未提供联合电荷弛豫或跟踪；当前实现见 §10 及顶部状态表。
`AtomicStateView` 仍为拟议 API。

## 10. 共同轨道多电荷优化的当前实现

`CIChargeStateAverage` 已将多电荷 CI Hamiltonian 平均接入公共
Atom/MCDHF 求解器。显式 parent 空间与完整 retained-orbital daughter
空间共享正交轨道；每个 trial 重建 CI 谱，有符号 RDM 加权后一次求
径向梯度。接受谱缓存绑定 functional 和完整轨道内容，通道 provider
独立重建的离子谱与目标谱逐项一致。此增量更新上节的“未提供联合电荷
弛豫”：现在提供固定权重的共同轨道联合优化，仍无独立初/末态轨道、
双正交变换或动态弛豫。`AtomicStateView` 仍为设计；显式低层
`CIBasisTracking` 后续已接入，见顶部状态表及其源码契约。

共同轨道上的优化目标不应改变默认初态密度。新规格分别记录各电荷
与各能级权重、目标平均电子数及物理父态电子数。改变目标是物理模型
选择；Ne N3 高轨道重叠及小载频移动仍伴随近相消 Auger 率大幅变化，
因此基底身份之外也必须记录完整 signed 算符与明确的空间收敛范围。
具体契约、失败记录与数值证据见
联合电荷优化归档 (local evidence: `xfel_charge_average_2026_10_04/README.md`)。

## 11. 同一 Hamiltonian 快照的径向复用与空间闭合

共同轨道 trial 现在用 `CIRadialSnapshot` 共享原有 CI 径向 evaluator。
owned 数据与精确核势节点算符保持只读；外部 caller 输入需要完整内容
及模型检查。核势节点算符的批量积分归通用 B-spline 数值层，DW 则
继续使用完整残余势。快照内容、优化目标、物理占率与动力学初态仍
分别记录；共享积分不改变这些物理选择，也不代表独立轨道/动态弛豫。

同一 retained-orbital 完整 Fock 空间在同κ正交轨道旋转下应有相同谱。
Ne N3 的 n=1,2空间验证到1.56e−13Ha；受限组态空间的谱改变0.433Ha，
表明其span随轨道旋转改变。固定轨道扩展和扩展目标重优化必须分别
比较，不能将两者的率变化都归于弛豫或按列号匹配物理态。

本轮34CSF/120父磁态及176磁态完整损失链已计算真实率，尚未认证该
扩展空间的XFEL轨迹。原受限空间强Auger相消在扩展后明显减弱；
全有符号算符与literal-zero分解项也需留档。性能只测量声明阶段，
小面板收益不稳定时明确记录。见
跨电荷积分与空间诊断 (local evidence: `xfel_ci_snapshot_2026_10_04/README.md`)。
