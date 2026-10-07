# Critical Review 账本 — mcdhf（~/code/atomx）

评审日期：2026-09-27 ｜ 范围：src/atomx/methods/mcdhf/** + 测试 ｜ 重点：物理正确性 + 高性能数值
基线 commit：385d908 ｜ 方法：先跑测试取证 + 代码走读 + cProfile
修复分支：fix/mcdhf-review-f1-f4（含同工作区在途 WIP，见 F5）

## 判定：有条件通过 → 修复后复评

物理核心经外部锚点验证（He 基态 -2.86181332065585 Ha vs DBSR_HF -2.86181332，偏差 ~1e-8 Ha）。

## 修复后复评（2026-09-27 当日）

| # | 级别 | 复评结论 | 状态 |
| --- | --- | --- | --- |
| F1 | 高 | **FIXED**。conftest.py 在 numpy 导入前 setdefault 钉 BLAS 线程（OPENBLAS/OMP/MKL/BLIS/NUMEXPR）+ pytest_configure 里 threadpool_limits 兜底；pyproject 加 pytest-timeout（600 s，thread 方法——signal 方法打不断卡死的 LAPACK 调用）+ dev deps，uv.lock 已同步。机制实测有效：负载 >64 时一次 numba `_numba_unpickle` wedge 被 per-test Timeout  loud-fail，而不是挂死套件。注意：全套件（2705 测试）在负载 >50 的完整通过记录尚未取得（会话被 bridge 回收三次 + 自带并行任务干扰），留给 CI 或安静窗口补证。 | FIXED（证据部分） |
| F2 | 高 | **FIXED，但结论被证据改写**。钉线程后复测：真实矩阵（Ne ns=113 的 219²/218²）eigh 只要 5 ms，549² 约 80 ms——评审时"eigh 占墙钟 51%"是**未钉线程 OpenBLAS 在负载 ~168 机器上的互踩伪影**，不是算法复杂度。subset 驱动维持拒绝（仓库 docstring 自有测量 + variational 投影子需要整条正能谱，subset 结构性地给不出）。实测落地的免费收益：`_eigh_finite` 切 `driver="gvd"`——单线程与 gv 持平、4 线程 549² 快 1.7×（47 ms vs 80 ms）、本征值逐位相同（max\|Δ\|=0.0）。Cholesky 缓存方案实测更慢（多两轮 O(n³) 三角求解），不做。复 profile（env 钉线程）：He 5 轮迭代墙钟 40 s → 3.1 s；eigh 在 Ne 全流程仅 0.26 s/81 次。锚点绿：He −2.86181332（DBSR_HF）、Ne scalar Breit vs DBSR_HF。 | FIXED |
| F3 | 中 | **FIXED（代码）；重原子外部验证仍 OPEN**。`optimize_orbitals(projector_source="nuclear"\|"fock")` + `MCDHFSolver(variational_projector=...)`。fock 路径用占据数平均的 SCF Fock 场重建 no-pair 空间（GRASP 式构造）；默认 nuclear 行为完全不变。常驻诊断 `no_pair_projector_leakage`（种子轨道在**另一种**投影子下的 S-范数权重，量化两个 no-pair 边界对该壳层是否一致）。验证：Be 双路径收敛、ΔE=7e-9 Ha、泄漏 ~1e-13；新增 gate 测试 `test_variational_projector_fock_matches_nuclear_on_light_atom`。Z≥50 开壳 vs GRASP/DBSR_HF <1% 的验收需外部参考数据（本地无），且 Cs Z=55 (6s+6p) 双路径对比因会话回收未完成——保留 OPEN 至有参考数据。 | FIXED（局部）/ 验证 OPEN |
| F4 | 中 | **FIXED（2026-09-27），2026-10-05 起被取代**：当时的 guard（改名 + `allow_legacy_incorrect_physics` opt-in）已由 2026-10-05 评审 E3 的处置取代——legacy kernel 及其 25 个仅其可达的 helper 已**整体删除**（matrix.py 2316 → 945 行），名称不复存在。原始记录：`_build_ci_matrix_approximate` → `_legacy_build_ci_matrix_approximate(..., allow_legacy_incorrect_physics=False)`，默认调用人话可读地 raise RuntimeError（含实测缺陷数据与 opt-in 说明）；matrix.py docstring 同步。guard 实测触发；生产/测试路径无调用点（测试仅散文注释提及旧名，无需改）。 | FIXED → SUPERSEDED |
| F5 | 中（新） | **OPEN，非本次修复引入**。`test_two_electron_replacement.py::test_split_channel_moves_a_real_calculation` snapshot 偏移 1.7e-3 Ha：HEAD（385d908）通过；工作区在途 WIP（variational.py 鲁棒性扩展：preconditioner_factory/refresh/step 复用，另一会话未提交工作）使轨迹改变所致。我的修复只是让该 WIP 路径可运行（修复前它在首次 scale_blocks 就 AttributeError 崩溃）。物理结构断言（J=2 CSF、能量 lowering ∈ (0, 0.05)）全过，仅自录数值漂移。**处置建议：WIP 所有者评审后 re-baseline 该 snapshot**；在 WIP 定性前不动它。 | OPEN |

## 已核验无问题（复核时勿重复指控）

- 异常吞食：mcdhf/ 下 except-pass / except Exception 计数 = 0
- 测试静默跳过：mcdhf 相关测试 skip/xfail 计数 = 0
- CI 门禁范围：default `-m "not slow"`，nightly 跑 slow+book，无白名单过滤链
- exact_ci 维度爆炸：有 max_determinants 上限保护（exact_ci.py:1035）
- 外部锚点：He 与 DBSR_HF 偏差 ~1e-8 Ha；Ne Breit operator 0.017340 vs DBSR_HF 0.016644 Ha（差值 4.2% 为文档化未实现的 retardation）
- 产物卫生：output/ 888M 已 gitignore；无 __pycache__ 入库
- 修复后锚点回归：He/Ne 两个外部锚点测试在 gvd 驱动下通过（22.4 s）

## 高性能改造路线（钉线程后的真实热点，按 profile 排序）

钉线程后 eigh 已退出热点。Ne ns=113 breit 3 轮的真实分布：`_convol_case4_t`（numba）1.78 s、`_convol_case1_t` 1.46 s、`_build_hf_matrix_dbsr` 累计 3.75 s、eigh 0.26 s。He 全流程 3.1 s（thrash 前 40 s+）。

1. **rkb 积分库跨 SCF 迭代复用**：`_assemble_rkb_sym/_pqpq` 每轮重建（He 中 0.75 s/3.1 s = 24%）。仓库已有 `test_rkb_bank_sharing.py` 的共享先例——把 bank 生命周期从"单次 Fock 构建"提到"SCF 循环"，预计省 20–30% 墙钟。
2. **convol 核的并行/向量化**：`_convol_case4_t` 708 调用 × 3 ms。numba `parallel=True`/prange 按径向节点并行（数据无关），保守 4–8×（需与钉线程策略协调：测试钉 1，生产可调）。
3. **GPU（本机 5090）**：convol 是规整的带状卷积，适合 numba.cuda / CuPy 移植；工作量大，适合作为二期。
4. **生产入口默认钉线程**：pytest 已钉，但 `python -m`/benchmarks 脚本未钉——同一台负载高的机器上脚本仍会 thrash（本次 profile 的教训）。建议在 `atomx/__init__.py` 默认 `threadpool_limits(1)` 并提供 `ATOMX_NUM_THREADS` 逃逸口。

## 标杆（好话要真）

1. breit_ci.py — 部分算符交付的范本：完整 Gaunt、retardation 显式报错 + 实测缺失尺寸，外部锚点随代码
2. variational.py — 黎曼 L-BFGS 的鲁棒性扩展全部带实测失效模式注释（refresh 清历史等）
3. exact_ci.py — 惰性 M 扇区 + 后缀 DP 剪枝，显式拒绝全笛卡尔积

## 复评约定

修复以"验证标准"列关闭；F1 全套件证据、F3 重原子外部验证、F5 WIP snapshot 归属三项留给后续；
F1/F2 的本地可重复验证命令：`OPENBLAS_NUM_THREADS=1 pytest tests/test_mcdhf.py`（钉线程前后对比即 F2 证据）。
