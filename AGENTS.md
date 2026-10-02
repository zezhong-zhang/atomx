# AGENTS.md — atomx 开发洞察与哲学（2026-10-02 战役蒸馏）

本文从 trace.md（36 条 EARS 记录）、20 个 commit 的收敛弧线和 11 个
数据化死路中蒸馏。目的：让未来的 agent 不重付这些学费。

## 0. 一句话哲学

**测量压倒推理，参考源码压倒二手转述，约定先于 bug。**
2026-10-02 全天 18 个 commit 的每一分进展都来自"插桩实测"，
每一次翻车都来自"想当然"（含本人两次错误结论，均被自己的实验纠正）。

## 1. 六条硬规矩（全部有当日翻车案例背书）

### 1.1 读参考实现源码，不读转述
- faclib（/data/code/fac）、XATOM（/tmp/atomx-xraypac-build/source）
  都在本地。trace 里的二手转述出过错两次：
  - "sd×0.5 双重归一化"假设——读 SlaterTotal 源码发现触发条件是
    `k0==k1 && k2==k3`，Auger 永不满足 → 假设证伪；
  - R_K 输出的 K 列读串 → 角度链验证白跑一轮。
- **能本地活跑参考代码就不要只读**：从 fixture 抽出 sfac 脚本直接
  rerun（LD_LIBRARY_PATH 指向 xatom 工具链的 libgfortran），一次
  就把"a_radiative 约定"钉死为 TR总/(2J+1)——六轮审计的最快一步。

### 1.2 角动量代数永远 sympy 数值验证，绝不手推
- 手推 6j 零结构错过一次、K=2 读串一次；sympy 复制品在 Ne KLL
  三个通道上精确复现 XATOM 的 1.0000 后，接线一次成功。
- 注意约定：仓库 `wigner_6j` 用 doubled-j，sympy 用物理 j
  （0/101 mismatch 教训）。

### 1.3 先排除"约定差异"，再怀疑代码 bug
- Fe 辐射率"8-15× 偏差"追了六轮（单通道→相干和→波长→J纯度→
  引擎互洽→**本地重跑 FAC**），最终答案：fixture 的 a_radiative
  是 DR/Burgess 逐子态约定 = TR总/(2J+1)，五通道比值精确 2J+1。
- 期间本人一度宣布"病灶锁定在某代码路径"——被认证引擎判别实验
  证伪。**宣布归因前先问：两边定义的是同一个量吗？**

### 1.4 校准锚点先行
- 新转录公式先找解析锚：LS 链用 XATOM 自己的 R_K 值闭到
  1.0000；UTA-0 用 2s2s=2πR² 锚定常数；H 1s 的 TRK 连续/分立
  分区 0.5650/0.4350（我记错成 0.5048，闭式级数和纠正了我）。
- 锚定后的转录"接线一次成功"是常态，未锚定的接线平均要 3-4 轮。

### 1.5 半成品不进 main，死路必须数据化
- 每个被证伪的假设（当日 11 个）在 trace.md 留下**测量值**，
  不是一句话"不行"。归因修正（如径向假设被三重证伪）必须显式
  更新此前写下的结论文档——错着留在文档里比没有文档更糟。
- WIP 接线若比 committed 状态差 → revert 保 gates 绿（4f1753b→
  6b5c1a9 先例）；若 uniformly better → commit 并把已知缺口写进
  docstring（永远标注 KNOWN GAP）。

### 1.6 性能优化必须可测量才做
- BLAS 线程 1→8：1.13s→1.27s（负收益，实测）；
- 带状特征解：实测 Fock 矩阵带宽 166/N=219、密度 35%（HF 交换
  填穿带结构）→ 死路；
- 共享机高负载下加速不可认证 → speedup gate 带负载守卫，正确性
  gate 常跑。不发布测不出的优化。

## 2. 已验证的物理结论（不要重新怀疑）

1. **Auger 三级阶梯**（Fe XVII LMM vs FAC AITable，同管线）：
   组态平均 jj 链 0.64–3.56 → 单CSF jj_level 0.10–0.98 →
   行列式 CI 本征级 0.62–1.33 → MCDHF CI-平均弛豫 0.79–1.49 →
   全 nl=3 复形 **0.89–1.11**。每加一层"能级信息"收紧一档。
2. **径向不是 Fe 面板的限制项**（三重证伪：冻MCDHF≈DHF≈弛豫DHF
   同模式）；主差 = CI 混合/能级位置（n=4 伙伴是一阶项但双刃）。
3. **轨道弛豫正典方案 = MCDHF CI 哈密顿平均能级自洽**；等权分数
   占据 DHF 只是构建器能力/对照工具（用户裁定）。
4. **E1 全链认证 2%**（3C/3D vs FAC RME，shooting+gu 契约）；
   mscheme 与 ci_lines 两引擎物理互洽。辐射率的量大偏差先查
   约定（见 1.3）。
5. **散射↔谱学一致是构造保证**：f₂=ωσ/(4πα)（光学定理）+ KK
   色散 + TRK gate；H 1s 连续份额窗口实测 0.373 vs 精确 0.369。
6. **流形辐射率参考约定**：fixture a_radiative = TR 级总/(2J+1)。
   Like-for-like 后 ILEV 32/190 闭到 0.96/0.97。

## 3. 工程地形（实测数字）

- **逐组态成本**：Fe 闭壳 DHF 1.13s（shooting 快路径）；mscheme
  流形 2J=1,3,5,7 全套 warm ~5s。早先"分钟级"是共享机争抢的
  wall time 幻觉——报耗时先看 loadavg。
- **正确并行形态**：进程池 over 独立单元（solve_parallel，fork+
  BLAS 钉1）；XFEL 级联/Z 面板即此形态。BLAS 线程与带状化均
  非杠杆（§1.6）。
- **真热点**（未做的 tier-②）：CI 两体装配的 10 万次逐单元
  compute_slater_rk_cell 内核调用（`_v` 已有 LRU，别再看它）。
- **result 类型动物园**：SCFResult / MCDHFResult / ShootingResult
  属性面不同（orbitals_by_subshell vs orbitals_dict vs _bsp_data）。
  getattr 静默回退曾吞掉 +J[ρ_full] 项造成径向 bug——新代码显式
  支持两类或显式报错，不要静默回退。
- **numba weakref flake**：报 "underlying object has vanished" 时
  先 `find src \( -name '*.nbi' -o -name '*.nbc' \) -delete`，
  环境问题不是代码问题（dr_vs_fac/plasma 在 HEAD 上同样失败）。

## 4. 工作流契约

- **trace.md 是账本**：每轮增量后写 EARS 条目（含测量值）；死路、
  归因修正、用户裁定（如 §2.3）都要落账。trace.md 被 gitignore，
  属于会话内知识——重要结论应同步进本文或 gate docstring。
- **gate 纪律**：live 带如实反映当前测量（改物理后必须重测并更新
  docstring 里的 MEASURED 值）；aspirational 门用 strict xfail 带
  实测面板，禁止静默 xpass；配对/匹配规则必须客观且写明（如
  "±2.5 eV 窗内最强级"），不许调带凑数。
- **fixture 先问约定**：用外部参考前先读其生成脚本（fixture 的
  'script' 字段/归档 run.sf），确认量的定义再比数值。
- **共享机协作**：工作树可能有并行会话的在途改动（git status 里
  不是自己的文件不碰不提交）；耗时长跑先查 loadavg。

## 5. 当前前沿与下一刀（截至 2026-10-02 晚）

- 主线队列：`xfel/cascade.py` 第一增量（规格：docs/
  XFEL_GAP_ANALYSIS.md；Ne 荷态-流量锚，Young 2010；ω 已可在
  匹配约定上定义）。
- 10% gate 尾巴：ILEV 71/72/142 的混合/位置残差；n=4 等权弛豫
  假设未测完（探针 /tmp/atomx-fe-n4relax.py 被会话重启打断）。
- 系统性矩阵 TODO 格与权威工具位置见
  benchmarks/bench_systematic_fac_matrix.py 与 trace.md 尾部。
