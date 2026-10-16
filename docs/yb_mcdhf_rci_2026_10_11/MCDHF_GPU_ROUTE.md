# 原生 MCDHF 的 CPU/GPU 执行路线

此文件是当前 MCDHF 消费层的技术方案，尚未实现或认证 GPU 主线。
复用已有[Numba CPU/CUDA 数值核实践](../atomic_pci_sd_native_2026_10_09/README.md#numba-cpu与cuda共享数值核)
及其隔离 CUDA runtime；不重复安装环境，不把旧 SD 算术控制当成 MCDHF 资格。

2026-10-12 本机 metadata 查询为 RTX 4090、驱动595.91.07、显存49140 MiB，
当时使用1296 MiB、利用率0%。这只是瞬时查询，不是独占设备证明。
主 Python 环境为 NumPy2.2.6、SciPy1.15.3、Numba0.65.0；主环境没有
numba-cuda/CuPy 发行包，已存在的独立 runtime 与主环境分别记录。

[实际29轨道暖态比较](correlated29_radial_reuse_qualification003_preparation/warm_benchmark_preparation/RESULTS.md)
中，H trial 中位约1.742→1.275秒，g约1.107→1.124秒。
总 primary 观测比1.18872，但负载门失败；这些数字不能认证加速或说明GPU收益。
下一性能候选针对实际 MCDHF 的 H/g，先测阶段与传输成本，再决定采用。

统一物理描述与数组身份，CPU和GPU分别实现数值执行。Numba的
[CPU并行机制](https://numba.readthedocs.io/en/stable/user/parallel.html)
与[CUDA kernel 支持范围](https://nvidia.github.io/numba-cuda/user/cudapysupported.html)
不同；CPU函数不能仅增加一个decorator就成为CUDA求解器。
[CUDA内存API](https://nvidia.github.io/numba-cuda/user/memory.html)
允许显式设备数组、传输和stream。方案据此选择持有设备快照，减少重复传输。

|实际层|共享物理与身份|执行候选及优先级|
|---|---|---|
|角代数、CSF完整扇区、UID跟踪|完整有符号拓扑、相位、简并span|先保留CPU及现有缓存；禁止幅度筛选|
|径向 Coulomb/J/K 收缩|已有cell几何、PP/QQ/PQ/QP标签、全部矩与边|优先候选：设备常驻几何、批量收缩与融合归约；CPU为独立锚|
|归一化有符号RDM与orbital g|完整态权重、密度、metric与Lambda连接|采用同一数学收缩规范，分别Numba CPU/CUDA；先通过同点导数门|
|RCI本征求解|原完整H、全目标块、全谱残差/Gram|保留当前CPU锚；只有完整块及传输的实际收益成立再选择GPU|
|优化、线搜索与持久化|原no-pair P、极分解回缩、exact能差、原门|CPU控制；每个trial有自己的H/CI/径向快照身份|

已有[CellPairContraction](../../src/atomx/numerics/bspline/pair_contraction.py)
与[CellCoulombFeatures](../../src/atomx/numerics/bspline/coulomb_features.py)
是可复用数值表示，是否被具体 frozen 源的 MCDHF 消费必须逐入口核对。
不增加GPU版物理方法，也不让数值引擎决定核势、关联空间或算符覆盖。

首个可执行合同应固定已独立资格的同点完整H/CI/g输入，关闭fastmath，保留FP64、
全部坐标与有符号项。声明GPU owned payload、临时量、驻留数组、传输字节、
CPU线程与host RSS；同几何一次构造及每trial失效分别计数，不把getter当cache miss。
先过完整H、signed degenerate span、RDM、g、metric、全谱残差与独立导数门，
再交错配对完整fresh trial。冷编译、设置、传输、同步与端到端成本分别记录。
独占设备或CPU负载门失败则不认证加速；原SD小kernel计时不代替此项。

目前先完成同208物理导数与屏蔽f种子判别；GPU实现不得通过改变物理问题来加速。
