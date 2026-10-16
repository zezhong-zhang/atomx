# pCI、AMBiT、EXP-T 源码核查与 Atomx 有效哈密顿量升级

记录日期：2026-10-09。任务是下载三个官方代码，从实际实现提取改进路线，并完成一个可验证的原生改动。此前 Yb⁺ 的原始计算、失败证据与五个阶段归档保持封存。

## 1. 已完成什么，能得出什么结论

三个官方源码已下载、固定提交并在隔离工具链中构建。Atomx 本次完成了**完整带符号、含非对角元的一体自能矩阵到原生 CI 的消费通路**，同时绑定生产者的轨道相位、冻结芯层、B-spline 度量与核模型，拒绝会重复计入芯层相关的空穴配置。另修复了独立测试发现的 Davidson 小空间重启问题。

两组径向参考、五个目标能级，共十个此前保存的 Yb 自能矩阵，均通过原生 CI 重放。最大 CI–Dyson 能量差 **1.394×10⁻¹¹ Ha**。相关回归共 **123 passed，1 skipped，1 xfailed**；后两项分别是环境不可用的 sfac 外部比较和既有严格标记的独立 2p 径向函数协议限制。

这些结果确认有效算符进入 CI 的数值和表示闭合。**本次没有产生更高阶的 Yb 谱线，也没有改善此前最大 5.868% 的实验谱位偏差。**完整两体屏蔽、all-order SD、通用粒子–空穴互补 Q 算符及 FS-CCSDT 尚未在 Atomx 中实现。

## 2. 下载版本、源码与构建状态

官方地址：[pCI](https://github.com/ud-pci/pCI)、[AMBiT](https://github.com/drjuls/AMBiT)、[EXP-T](https://github.com/aoleynichenko/EXP-T)。本次独立目录（另保存三个 pristine_pinned.tar.gz 原始提交归档，未混入编译补丁）：

```text
/home/zzhang/code/atomx-references/2026-10-09-ci-fscc/
  pCI/       pCI-build/
  AMBiT/     AMBiT-build/
  EXP-T/     EXP-T-build/
  toolchain/ toolchain_packages/
```

| 代码 | 固定版本 | 原始许可文件 | 实際构建和验证范围 |
|---|---|---|---|
| pCI | main，4d685d82bfdf27876098a87808dadc5fcf3038d1 | LICENSE.txt，GPLv3 | 串行程序构建成功；当前没有 MPI，因此未构建 pconf 等并行 CI 程序；上游明确提示 GNU 工具链不支持本版本可选 QED/all-order 包 |
| AMBiT | master，fe1115aaed97b340e1f4d625d451b165c5bcd7a5 | LICENSE，GPLv3 | OpenMP 构建成功；小型 Be⁺ CI+MBPT 实算完成；官方 MgI 空穴例子的空间规模检查完成；未跑生产 Yb 基准 |
| EXP-T | master，aa8d809ad5818d4aa618dcdadbd7f12257ecbc28 | LICENSE 文件为 LGPL2.1；README/源头声明 LGPL3+，存在版本表述差异 | 主程序及工具构建成功；帮助入口和官方振转工具数值测试通过；电子 FS-RCC 测试未执行，当前缺 DIRAC pam 及其变换积分 |

版本、原始许可哈希在 [downloads.json](downloads.json)。最终二进制哈希、源树改动和精确构建命令在 [reference_final_state.json](reference_final_state.json)。构建工具为任务目录中的 GNU Fortran 15、CMake 3.31.6、GSL/Eigen/Boost/Abseil/OpenBLAS；采用下载并解包，没有安装系统包或改变 Atomx 的 Python 环境。

为适配 GNU 编译器，仅在本次参考副本作两项局部修改，分别保存补丁：

- pCI 字符串数组构造显式指定元素长度：[补丁](pCI_gnu_string_constructor.patch)。不改变数值公式。
- EXP-T 对 OpenBLAS 使用其已有 cblas.h 声明：[补丁](EXP-T_gnu15_openblas_declaration.patch)。最终构建无需放宽 C 的隐式函数声明错误；旧 Fortran 参数匹配使用兼容选项，命令已记录。

所有第一次失败和临时构建记录保留。AMBiT 上游构建脚本会从执行工作目录读取 git 信息；第一次从 Atomx 目录配置使横幅误显示 Atomx 提交。已改成从 AMBiT 自身目录构建并重跑，最终输出正确显示 **master fe1115a**。数值结果和源码身份不能只凭程序横幅猜测。

## 3. 三条方法路线实际意味着什么

### pCI：CI+MBPT 与 CI+all-order

pCI 把芯层影响写成有效一体、两体等算符，再在价电子空间做 CI。源码中的 `lib/all-order/allcore-ci.f`、`valsd-ci.f`、`sdvw-ci.f` 对应线性化耦合簇单、双激发路径；不是 EXP-T 的完整指数 FS-CCSDT。

“最多 60 个电子进入 CI”的说法有原论文支持，但它描述已使用过的 CI 系统范围，不等于 60 价电子的 full-CI 或 CI+all-order 精度保证。同篇论文另说明其 CI+all-order 测试达到六个价电子。依据：[pCI 原论文](https://www1.udel.edu/atom/download-files/2025pCI.pdf)，本地保留 PDF 和提取文本。

实现时还必须核对能量分母和覆盖约定：pCI 的二阶扩展可补更大空间，all-order 在重叠区覆盖相应二阶条目；不能直接把两套同一图相加。其输出 `SGC.CON`、`SCRC.CON` 是实际消费的一体/两体修正接口。[官方二阶扩展说明](https://pci.readthedocs.io/en/stable/extensions/second-order.html)

### AMBiT：粒子–空穴空间与减除图

AMBiT 的参考价值在于明确区分 core、valence、excited、high 以及空穴空间，并在有效算符中处理相应图和 subtraction。源码 `src/MBPT/OneElectronMBPT.cpp` 会生成不同外部轨道之间的带符号矩阵元；`CoreMBPTCalculator.cpp` 与 `ValenceMBPTCalculator.cpp` 决定哪些贡献保留。两体部分还有额外 box、off-parity 等通道，不能只把普通 Coulomb 的径向 Rᵏ 改个数值就声称覆盖完整 Σ₂。

官方 MgI 设置把 2p 留作可激发空穴，只冻结 1s、2s；这是学习互补 P/Q 的具体例子。该模型是否提高某种稀土谱线精度，必须另做匹配计算。论文依据：[AMBiT 方法论文](https://arxiv.org/abs/1805.11265)、[粒子–空穴 CI+MBPT 原论文](https://arxiv.org/abs/1605.08111)。后一论文的 Hg⁺ 结果不能直接移植成 Yb 误差预算。

### EXP-T：真正的 Fock 空间耦合簇

EXP-T 有实际的簇振幅、张量图收缩、Fock 扇区和三激发模型源码，例如 `src/rcc/models/sector00_ccsdt.c`、`sector01_ccsdt.c`、`sector11_ccsdt.c`。它还需要电子积分输入、模型空间选择、intruder-state 处理及相应性质算符。

对闭壳参考的电荷与光谱记号要避免混淆：光谱学 Yb III 对应 **Yb²⁺、68 电子 [Xe]4f¹⁴**；向这一参考附着一个电子得到 Yb II / Yb⁺。这适合首先审查 0h1p 方法及其内部芯层激发。强 4f 空穴混合是否进入更大的模型空间，需要重新明确扇区、P/Q 与态身份。

Fock 空间有效哈密顿量一般涉及非 Hermitian 结构与左右态。它不能直接冒充本次实对称冻结 Σ₁ 接口。只有产生实际振幅、通过独立数值基准和原生残差，才能把 Atomx 的 `coupled_cluster.py` scaffold 升级为 FS-RCC。依据：[EXP-T 原论文](https://arxiv.org/abs/2004.03682)。

## 4. 为什么首先修有效算符消费

此前 Atomx 已能生成同 κ 的非对角二阶附着自能 Σ(E)，并解有限外部空间 Dyson 根，但 `ci_mbpt_solve` 只消费 `{(n,κ): Δε}` 对角移位。这样无法把径向配置混合真正送入多电子 CI，也不携带产生自能时的芯层分区。

从第一性出发，有效一体算符是

```text
δH₁(E*) = Σ_ac σ_ac(E*) a†_a a_c
H_CI(E*) = H_DC + δH₁(E*)
```

它的非对角元带符号，并通过真实 Slater–Condon 规则和 CSF 投影形成多电子矩阵；不应改成只移动求解后的能级。Σ₁ 的下标在这里表示**一体算符**；此前“二阶 Σ”中的二阶表示**微扰阶数**。二阶一体自能与两体屏蔽 Σ₂ 是不同对象。

本次改动：

| 层 | 实现 | 验证含义 |
|---|---|---|
| physics/csf | `CIEffectiveOneBody`，不可变实对称标量矩阵和芯层守卫 | 保留非对角符号；跨 κ 非零元拒绝；不同轨道/度量/核模型拒绝 |
| methods/mbpt_matrix | 生产者保存 `CIReferenceBinding`；`as_ci_operator(E*, ...)` | 匹配外部轨道系数和相位、全部相关芯轨道及核/径向几何；旧无绑定快照禁止静默导入 |
| physics/csf/matrix | `one_body_operator` 注入原生 h 径向键 | 复用已有精确行列式→CSF 投影；dense 与 operator 路径相同 |
| methods/ci、ci_mbpt | 转交算符、保存 provenance、拒绝重复注入 | diagonal 旧接口保持；Σ₂ 请求仍明确报未实现 |
| CI Davidson | 容许用剩余子空间容量扩展后再重启 | 修复三维矩阵求两个根的实际不收敛例子；没有放宽残差 |

### P/Q 防重复计数的硬条件

每个 `correlated_core` 轨道已被自能生产者积分消去。因此每个 CI 配置都必须继续完整占据它。若 CI 配置打开这一芯层的空穴，接口拒绝计算，并要求重建互补 Q 图。

这是一项防错守卫，**不是已经实现了通用粒子–空穴 MBPT**。删除 ledger 项、复用原全芯层自能、再加入空穴 CI，不能构成合法升级；生产者绑定也会阻止这种不匹配。未来必须显式重建模型分区和相应图。

## 5. 可运行的 Atomx setup

下面的小 Be⁺ 参考展示完整生产者→消费通路；用于接口与物理符号验证，不代表稀土生产精度。

```python
from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.mbpt_matrix import build_mbpt2_matrix_self_energy
from atomx.methods.ci_mbpt import ci_mbpt_solve

core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2.)], ns=30, ks=7).solve()
_, virtuals = build_virtual_basis(core, -1, e_max=50.,
    potential_mode='VN', orthogonal_to_occupied=True)
targets = [(P, Q, -1, e) for P, Q, e in virtuals[:3]]
keys = ((2, -1), (3, -1), (4, -1))
orbitals = {(1, -1): core.orbitals_by_subshell()[(1, -1)][:2]}
orbitals.update({key: t[:2] for key, t in zip(keys, targets)})

sigma = build_mbpt2_matrix_self_energy(core, targets,
    reference_identity='BeIII:1s2:native-test',
    virtual_kappas=(-1, 1, -2), n_virt_max=2)
energy = targets[0][3] - .01  # 明确声明的冻结能量参数，不是实验拟合
effective = sigma.as_ci_operator(energy, keys, orbitals, core._bsp_data,
    4, 30, correlated_core={(1, -1): 2},
    nuclear_potential=core._nuclear_potential)
csfs = enumerate_jj_csfs_many(['1s2 2s1', '1s2 3s1', '1s2 4s1'])
result = ci_mbpt_solve(csfs, orbitals, get_shared_orbital_basis(csfs),
    core._bsp_data, 4, 30, sigma1_operator=effective,
    nuclear_potential=core._nuclear_potential)
```

这里仅截取两个中间态/分波空间，以便快速演示；它没有收敛精度承诺。若要求能量自洽，需要在明确分支与无极点区间中迭代 Σ(E)，不能把冻结参数 CI 求解称作完成 Dyson 自洽。裸辐射算符也不会因能量迭代自动包含 RPA、归一化或辐射顶角。

## 6. 独立测试与十态 Yb 重放

[新增测试](../../tests/test_ci_effective_one_body.py)包括：

1. 闭芯层加一个电子，新增 CI 矩阵精确等于给定带符号 σ_ac；实际测试使用正、负非对角元。
2. 两个 s₁/₂ 价电子的 singlet 配置，检查占率 2、1、0 和相邻配置 √2·σ_ab 耦合；双替换元为零。
3. 径向轨道相位翻转后的全矩阵协变与谱不变；原生产者相位不匹配时拒绝导入。
4. 冻结芯层空穴、核模型变化、重复对角/矩阵注入、非对称和跨 κ 非零矩阵的拒绝。
5. dense/operator/Davidson 消费一致，裸 CI 默认行为不变，二体未实现状态持续显式记录。

测试日志：[主面板](final_tests.log)，61 passed / 1 skipped；[依赖面板](dependent_tests.log)，62 passed / 1 xfailed。保留第一次失败日志，避免把实现修复后的绿色结果倒写成初始通过。

重放固定条件：[契约](native_replay_contract.json)。读取前阶段的粗、细两套、各五态、每个 κ 三个外部函数的完整自能快照；不重建或截断其中任一物理图。核模型为 Fermi、Z=70，纯 DC；芯层是同一 [Xe]4f¹⁴；先从原生 CI 计算共同芯层零点，再比较附着能。

| 检查量 | 十态最大误差 / 数值 | 预先声明门限 |
|---|---:|---:|
| 裸 CI−共同芯零点与对角 Fock 矩阵 | 1.409×10⁻¹¹ Ha | 10⁻⁸ Ha |
| 实际 CI 增量与完整 Σ₁(E*) | 1.747×10⁻¹² Ha | 10⁻¹⁰ Ha |
| 原生 CI 附着能与矩阵 Dyson 根 | 1.394×10⁻¹¹ Ha | 10⁻⁸ Ha |
| 原生本征残差 | 1.835×10⁻¹² Ha | 10⁻⁸ Ha |
| 系数投影算符差 | 5.496×10⁻¹² | 10⁻⁷ |
| 历史 Hermitian 投影引起的 Σ 变化 | 1.735×10⁻¹⁸ Ha | 10⁻¹⁰ Ha |

十态总用时 71.94 s，记录最大 RSS 5728.9 MiB；这不是控制机器负载后的性能对比。原始核零点和全部态结果在 [native_replay_results.json](native_replay_results.json)。

历史快照经正交旋转产生最大 1.084×10⁻¹⁹ Ha² 的转置末位差。严格原生算符对此拒绝。重放脚本先检查原有 64-eps residue 对称/PSD 门，再**显式**取 (R+Rᵀ)/2，记录其影响；没有删小幅度、合并极点或改变物理参考。这个表示处理不放宽原生算符的对称性契约。

## 7. 外部代码实际跑了什么

AMBiT 使用本专题 [BeII.input](BeII.input)：Z=4，1s² 闭芯层，6spdf 价基和 8spdf MBPT 基，4000 个径向积分点，单电子 leading configurations，0 次 CI 激发/空穴。`-s12` 启用一体/两体 MBPT 接口；对于单价电子，价电子两体矩阵不直接贡献谱位。首次 600 点发出 spline 分段积分点不足告警，记录保留；最后增加积分点后重新执行。计算完成并输出能级、组态权重和 g 因子。[最终输出](AMBiT_BeII.log)

官方 MgI 空穴例子只做 `--check-sizes`，没有生产对角化：全对称性共 525918 个 level/CSF 规模，其中一个块达到 171801 CSFs，并有独立 small-side。它说明“打开芯层”必须先做真实空间和资源预检，不能把输入文件称为已完成光谱。[规模日志](AMBiT_MgI_sizes.log)

EXP-T 完成 `--help` 和官方 `vibrot_levels/test.py` 的四个工具数值测试，包括 Numerov、有限差分及映射和 Franck–Condon 场景。[最终工具输出](EXPT_vibrot.log) 这不等价于 FS-CCSDT 电子结构验证；后者需要 DIRAC 积分，执行范围单独记录于 [fscc_execution_scope.json](fscc_execution_scope.json)。

三个代码没有在同一 Yb 轨道、核模型、电子相关空间与算符约定下进行数值交叉比较，不能把当前小算例结果与已有 Yb 数据拼接成“跨代码精度通过”。

## 8. 后续如何真正减少稀土谱学差距

应从互补模型而非经验谱线偏移推进。当前本次完成的消费层，只是让下一阶段有效哈密顿量可以被正确使用。

| 顺序 | 明确物理问题和实现 | 最小可判别验证 | 不得据此宣称 |
|---|---|---|---|
| 1 | 选定 Yb III 68-electron 闭芯层参考，实现价电子 all-order SD 核及其完整带符号 Σ₁ | 二阶极限还原现有所有图；非对角独立锚；簇振幅残差；五态共同参考的绝对附着能/激发能/劈裂 | 闭壳 pp-ladder 已等于 all-order 价自能 |
| 2 | 对强 4f 空穴混合引入显式 P，重建互补 Q 及 subtraction；实现完整两体/必要更高体有效算符 | 小空间 P∪Q 精确解与有效解对照；重复计数排除；对外部 cutoff/能量分母敏感性 | 只改芯层 ledger 或普通 Rᵏ 已等于 AMBiT |
| 3 | 审查并增加 Coulomb–Breit 相关交叉图 | 同一上下态和基底下分别测一阶、交叉和全增量，独立符号锚 | 一阶 Breit 小就界定高阶 Breit 小 |
| 4 | 对 E1/M1 等性质加入与波函数一致的 RPA/响应、归一化和顶角 | 同一状态身份下矩阵元与寿命；度量、长度/电流约定和独立实验 | 谱位通过就寿命通过 |
| 5 | 在已有独立电子积分与小系统测试后引入 FS-RCC | 真实振幅、参考基准、扇区与 intruder 处理、左右态和性质覆盖 | 下载或编译 EXP-T 就实现 Atomx FS-CCSDT |

QED 加入后是否重求 CI/轨道，需要按照它进入算符的位置分别判断。重新迭代可以测出波函数响应，但它不会自动补齐遗漏的高阶电子相关或空穴通道。对重元素必须声明 QED 模型的覆盖、屏蔽近似与核模型；这里未进行新的 QED 自洽计算。

作为 agentic science 工作流，可把三个参考代码放在独立适配层，统一保存电子计数、核模型、轨道与态身份、Hamiltonian ledger、模型空间、能量零点、单位、残差和资源限制。先用 Atomx 的原生算符闭合和小系统独立基准辨别实现误差，再决定昂贵高阶计算。更广的原子→分子→晶体/XAS→未来产业故事线仍以 [研究体系文档](../spectroscopy_science_story_2026_10_06/README.md) 为主；原子 FS-RCC 不直接替代材料的周期结构、晶场、屏蔽和芯激子模型。

## 9. 复现、源码与文档影响

主要命令，均在 Atomx 仓库中执行；从新目录恢复官方固定源码可用 download_references.py --destination <目录> --apply-build-ports。各参考第一次失败、构建适配和最终状态均保留：

```text
.venv/bin/python docs/ci_fscc_reference_upgrade_2026_10_09/build_references.py
.venv/bin/python docs/ci_fscc_reference_upgrade_2026_10_09/run_reference_smoke.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_ci_effective_one_body.py tests/test_mbpt_matrix.py tests/test_ci_mbpt_sigma1.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/ci_fscc_reference_upgrade_2026_10_09/replay_native_ci.py
```

重放最后一条依赖此前封存的十个完整 matrix_poles.npz、两套轨道快照与闭芯层 reference；轻量源码包不复制这些大文件。输入位置和文件哈希在结果中，本地原件保持可用。构建还依赖 downloads.json 中固定副本、隔离工具链和本次显式编译补丁；构建成功状态属于记录的实际源副本。

本次生产源码身份以全部 Python 源文件 SHA256 固定。没有提交或覆盖其他会话修改。旧快照的身份链保持封存，新重放另有 [source manifest](native_replay_source_sha256.json)。

文档影响已审核：能力表/验证/测试/索引/洞察同步更新；`ci_mbpt` docstring 不再声称只能消费对角；Σ₂ 未实现仍明确报错；架构方向保持 physics 不导入 methods。公共接口为新增可选参数，旧默认和旧无绑定快照的 evaluate/Dyson 行为保留；无绑定快照的新 CI 导出明确拒绝。迁移总表保留，无删除或重命名旧入口。

该阶段交付的是可供后续 all-order 和粒子–空穴有效算符使用的、已经验证的计算基础。稀土实验差距的根本改进，仍需表中真实高阶模型和匹配实验门。
