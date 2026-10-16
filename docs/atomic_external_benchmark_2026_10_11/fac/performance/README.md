# FAC 完整冻结组态 CI / E1：精度模型差异与实际并行效率

本次实际运行 Ne-like Ar / Fe / Xe / Pb（Z=18/26/54/82）。每个离子保留基态、
`2p5 3s1` 的全部四级、`2p5 3d1` 的全部十二级，共 17 级；分别使用长波和
有限频率 E1 算符，保留全部十条线。四离子固定工作共 68 级、40 条线。
没有选择强线、降低径向网格或改变物理空间。

FAC 和 Atomx 独立优化各自局域基态势。本面板比较模型差异，不能据此认证
相同轨道上的数值等价、与 NIST 的误差、整体软件排名或最高精度。
共同积分上的 FAC 数值核对由父目录 [precision](../precision/) 单独报告。

## 固定物理与计算约定

- 点核、无限核质量、Dirac–Coulomb；SE / VP / Breit / MS 关闭。
- FAC 原始现有 `sfac`，`USE_MPI=2`，显式 `InitializeMPI(p)`，
  8000 径向点，近原点密度倍率 4，三种轨道能量收敛控制 `1e-12`；
  `SetOptimizeTolerance(1e-4)` 是编译期阈值倍率，不能称为绝对能量容差。
  每次基态最多 150 次迭代，保留实际收敛日志。
- `LimitArray(0,0)` 显式关闭压缩 float Yk 缓存；全双精度 Slater 缓存保持。
  组态能量平移关闭，三个组分别 CI。
- Atomx public `Atom(... engine='shooting', exchange='gu', nuclear='point',
  corrections='none')`，2000 基态网格点，SCF `tol=1e-10, max_iter=150`；
  public `ci_line_strengths` 同一冻结势中的全部虚轨道、CI 和两种光子算符。
- 能级在各 `(configuration,J,parity)` 完整谱内按能量秩核对，未做最近能量拟合。
  此规则不证明避免交叉时的波函数身份；FAC/Atomx 标签同时保留。
- FAC E1 常规线强度 `S = printed_signed_rme**2/2`；实际打印的 A 直接使用。
  每个 `TRTable` 单独一条上能级，避免频率平均。每个组态以同一基态为零，
  Atomx 两组态基态零点四个离子均完全相等。

## 完整模型差异

| 离子 | 全部 16 个激发能的最大相对差 | 全部 10 条 A 的最大相对差 | S 最大相对差 |
|---|---:|---:|---:|
| Ar IX | 0.124292% | 3.785096% | 4.046924% |
| Fe XVII | 0.044422% | 1.617738% | 1.486477% |
| Xe XLV | 0.006903% | 1.351728% | 1.363078% |
| Pb LXXIII | 0.002624% | 0.242472% | 0.236090% |

这是独立局域势的模型一致程度，**不是实验误差**。完整逐能级、逐线、
原始文件 SHA 和零点见 [model_spread.json](model_spread.json)。
原 1% 实验门没有放宽，也没有用这些 FAC 差异替代 NIST 门。

## 实际固定工作强扩展

三个交替配对轮次分别执行 `1,2,4,8` / `8,4,2,1` / `1,2,4,8`；
Atomx 8 线程在预热中因实际线程超限拒绝，完整新轮次执行 1/2/4。
每个观测是四个新子进程完整生命周期之和，包含启动 / 导入、基态 SCF、
虚轨道、CI、长波与有限频率光子、原始导出。操作系统和编译器缓存预热单列。
S 是同一轮 `T1/Tp` 的中位数；E=S/p，时间中位数单独计算，未用比中位数代替
配对统计。没有测独立离子并发进程吞吐，不能把它当作已测强扩展。

| 后端 / 请求 team | 四离子壁钟中位数 s | 配对 S | team 效率 E | 实际进程线程峰值 |
|---|---:|---:|---:|---:|
| FAC / 1 | 4.769212 | 1.000000 | 100.000% | 1 |
| FAC / 2 | 4.718081 | 1.010534 | 50.527% | 2 |
| FAC / 4 | 4.730715 | 1.009978 | 25.249% | 4 |
| FAC / 8 | 4.751842 | 1.005121 | 12.564% | 8 |
| Atomx / 1 | 8.928899 | 1.000000 | 100.000% | 1 |
| Atomx / 2 | 8.981222 | 1.001278 | 50.064% | 3 |
| Atomx / 4 | 8.964119 | 0.998994 | 24.975% | 7 |
| Atomx / 8 | 未接受 | null | null | 15，超过原总上限 8 |

Atomx NumPy / SciPy 的两个 BLAS 池使请求 team 2/4 对应进程线程峰值 3/7。
上表 E 是请求 team 效率；单独的 `resource_efficiency=S/actual_threads_peak`
分别约 33.376% / 14.271%，不是活跃 CPU 利用率。
实际线程逐 5 ms 从 `/proc/<science_pid>/task` 采样；日志同时保存运行态线程数
和累计 CPU ticks；控制器 / 计时器 / 观察线程与科学子进程分开记录。

这些小冻结组态工作没有显著线程加速，不能外推到大 CI / MBPT / SD / MPI。
FAC 的完整实例约 4.77 s，Atomx 私人诊断包装约 8.93 s；两者独立轨道模型且
诊断导出不同，不能将约 1.87 的时间比解释为相同精度、相同工作的软件排名。

FAC 子进程导出原生二进制 Hamiltonian、级表和线表；其 NPZ 解码在计时之后。
Atomx 子进程额外截获完整 H、J 纯化 CI 系数、相干有符号磁子态振幅、径向网格
和轨道能量，并在计时内部导出 NPZ / JSON。没有暗中扣除导出成本。

## 阶段成本和缓存

三个单线程轮次各阶段四离子中位数，完整记录见 [stages.json](stages.json)：

| 阶段 | FAC s | Atomx s |
|---|---:|---:|
| 初始化 / 导入 | 0.570458 | 1.196052 |
| 基态设置与 SCF（各自边界） | 3.253251 | 7.057596 |
| 虚轨道、全部 CI 与 H 导出 | 0.174920 | 包含在各 public CI/E1 调用 |
| 长波 public CI/E1 两组态 | 包含在 CI 及光子阶段 | 0.172758 |
| 有限频率 public CI/E1 两组态 | 包含在 CI 及光子阶段 | 0.058439 |
| 光子及原生线表导出 | 0.020884 | 包含在上两行 |
| NPZ 诊断导出 | 计时后规范化 | 0.008581 |

FAC 的时钟打印有 5–6 位有效数字；Atomx 用私人 `perf_counter`。阶段边界不同，
各阶段中位数不保证相加等于总时间中位数。SCF 主导成本是直接观测；提高小矩阵
BLAS 线程数无法明显改善当前完整实例是本面板推论，下一步应在更大的真实
物理空间重测，或另外测固定离子任务的进程并发吞吐。

FAC 使用已存在的源 / 对象 / 静态库 / `sfac`，没有测其历史构建时间。
新独立每离子 compiler cache 预热第一组四离子 40.555759 s（三个新冷编译，
Pb 已单独冷预检），其后 p2/p4 预热 8.905463 / 8.966575 s；全部排除在测量统计外。
Pb 独立新缓存预检冷 12.876376 s、暖 2.212325 s，也单列保存。

## 失败保留和资源守卫

- 原全局 Numba cache 的 Pb 第一次预热成功，后续测量在 `_save_index` 报
  `ReferenceError: underlying object has vanished`。这是缓存编译执行失败，
  没有实际线程 / AS 超限；四离子流程不完整，因此旧 Atomx 时间和 S/E 无效。
  完整旧输出保留于 [initial_epoch.result.json](initial_epoch.result.json)，
  `measured/Atomx/` 下所有 stderr 及成功的其他离子数组均保留。
- 只设置私人的逐离子 `NUMBA_CACHE_DIR`，先做 Pb 两次完整预检，再重做所有
  四离子预热和三配对轮。物理、源码、网格、容差、总线程上限没有改变，
  没有删除全局 cache。正式 Atomx 统计仅来自 [fresh_cache002](fresh_cache002/)。
- p8 Atomx 每个预热子进程实际观察到 15 线程，立即停止自己的进程；
  没有调高原 8 总线程上限，没有接受不完整工作量的时间。
- 每个科学进程 `setrlimit(RLIMIT_AS,8GiB)` 的返回值检查，600 s 壁钟停止，
  持有父级共享 `timing.lock`；科学计时串行。负载门 `load_before+p<=affinity32`
  且前后漂移 <=4。所有正式 FAC / 新 Atomx 轮通过，没有丢掉慢轮次。
- Python `subprocess(timeout)` 轮询不参与纯子进程计时。私人 C++ 时钟采用
  monotonic `steady_clock` 和阻塞 `wait4`，保留 user/system/peak RSS/exit/PID。
- `stopped_attempt.json` 是原控制器结束后到达的停止请求，实际没有剩余子进程
  被停止；不把它描述成真正提前终止。初次启动缺少 `PYTHONPATH` 与一次最终
  规范化相对路径错误只影响私人控制器，分别保留独立失败记录。

## 完整数组与实际源码 / 运行库

每个 FAC 观测重读全部 17 级、10 有符号线振幅 / A、三个原生 binary H 的
全部矩阵块；每个完整 Atomx 观测保存 142 个 H / dets / eigenvalues / eigenvectors /
twoJ / 相干有符号振幅 / grid / orbital-energy 数组。没有用强度平方替换签名振幅。
同后端跨线程全部数组在原绝对 `1e-11` 与相对 `1e-10` 门下通过，保留每个原生
量的单位；矩阵是 Ha、级能是 eV、A 是 s⁻¹，不能把所有坐标称作 Ha。
本轮所有正式输出实际完全相等。数组及原始 FAC 文件不提供实验精度认证。

两个源 epoch 的共享 1084 个实际 dirty 源 / build 文件一致；新 epoch 加入两个
私人缓存 helper，共 1086 文件前后不变。两个实际 source/build tar 包均保存。
逐科学子进程 `/proc/maps` 并集 113 个实际文件初始和终端 SHA 全相等，
没有只拿 Git HEAD 或 `ldd` 当作运行身份。

入口与结果：

- [run.py](run.py)：原完整 FAC / Atomx 预热与计时；失败也完整保留。
- [cache_preflight.py](cache_preflight.py)、[fresh_atomx_epoch.py](fresh_atomx_epoch.py)：
  私人逐离子缓存因果预检和重放；不得复用已有输出目录覆盖证据。
- [finalize.py](finalize.py)：保留历史结果并重哈希运行库、组合只通过门的正式表。
- [result.json](result.json)、[source_epochs_guard.json](source_epochs_guard.json)、
  [runtime_guard.json](runtime_guard.json)、[artifact_manifest.json](artifact_manifest.json)。
- [tests.log](tests.log)：9 个选定 parser、完整线清单、trace sum rule、
  有限频率与配置互换校验通过；不是全仓库回归。实际 live 全数组门单独保存。

重放须从仓库根目录用 `PYTHONPATH=src:. .venv/bin/python <runner>`，普通 bridge
环境变量保留。为新实验指定新的 owner / 输出目录，并沿用父级资源合同；
原报告目录已有文件时 runner 会拒绝覆盖。
