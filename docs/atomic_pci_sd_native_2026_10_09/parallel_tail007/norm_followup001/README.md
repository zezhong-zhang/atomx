# 完整 l4 物理范数：有界 Gram 暂存候选

[原完整尾部预检](../run004/result.json)绑定 V68、原 Fermi A175、ns123 和全部
五个 Yb 态，其 l4／E<5、10Ha 两档原范数 scratch 规划为 88,381,952／88,618,112B，
超过原 64MiB 门。本候选调整角 Gram 的所有权与储存方式，保持完整径向视图、
全部磁分量、所有有符号 Gram、等核空穴反对称化、价参考泡利阻塞和核 S1／价 S2
干涉。它没有修改范数方程、振幅、轨道带、Hamiltonian、能量零或 64MiB cap。

最小自包含[最终补丁](run004/proposed_helper.patch)供主任务顺序整合。
`grams` 的原 4096-entry 全局缓存移除，完整角求和函数本体不变；每个范数消费者
拥有独立的 256-entry／4MiB 数组 payload LRU 缓存，只淘汰完整 Gram 矩阵，
再次需要时完整重算。任何单项超过数组预算均拒绝，不裁减角通道。数组仍只读。
当前小型候选实现位于 [candidate.py](candidate.py)；规范资源计划另计 zero-copy
径向 view／字典 metadata、单激发 metadata、全部组 metadata、单 quartet 的角构造、
完整径向 stacks／收缩和原 1Mi 元素有限性 tile。预算排除解释器／Numba／共享
Wigner cache 的 runtime RSS，完整任务仍保留独立 runtime 储备。

[最终独立实验](run004/result.json)从实际封存的原 helper `06936ba9…4528e4`
构造最终代码，不读取已由主任务采用的共享新 helper 作为旧参考。
[前瞻契约](run004/contract.json)固定抽象正交的完整 1s² 核、10 个径向虚轨道，
包含 s／p／d／f／g 全部 κ 与磁多重态及两个不同径向 s 轨道；每条完整冗余 rho
均由记录 seed 719230 生成，保留正负号。这里没有生成原子 Hamiltonian、求根或实验谱。

全部 104 个态／M／±振幅情形的八个独立物理分量，对显式 fermion
`(1+T_core+S_v)|Phi_v>` 展开的最大差为 4.441e-16，低于原 1e-11 门。
范数的完整 M 协变、非零干涉翻号、完整 Gram 淘汰重建及预算拒绝都通过。
原默认 64MiB 拒绝保留在[负结果](run004/original_cap_failure.json)，没有为运行
旧方法提高 cap。[十项测试](checks003.xml)另覆盖非法缓存限额、readonly、命中、
淘汰、单完整 Gram 过预算拒绝、解码前拒绝和四档全部五态的冻结物理身份。

完整最坏 g／g／g／g quartet 保留 670 个磁行和 10 个 ranks；F／R 载荷精确
107,200B，三个输出 Gram 共 2,400B，角构造 traced peak 见最终结果，低于
1,540,800B 规划。单范数 traced peak 约 108kB，是上述有限输入的实测，不是
完整 Yb 消费的实测峰值。保留所有径向坐标后的暂存推导如下：F／R 数组上界
`16*maxmag³*maxrank`；包括 Python 系数列表及转换的角暂存按
`maxmag³*(256+128*maxrank)+48*maxrank²` 储备，最后一项含替换时仍存活的上一组
Gram；完整径向 stacks、等核空穴组合和收缩按 `64*maxrank*maxradial²` 储备。
缓存本身占用另计，未将缓存数组与工作数组混为一项。

| 原完整 Yb 窗口 | 五态最大候选 scratch / B | 有界缓存最多数组载荷 / B |
|---|---:|---:|
| l3／E<5Ha | 14,148,096 | 393,216 |
| l3／E<10Ha | 14,363,648 | 393,216 |
| l4／E<5Ha | 18,273,472 | 614,400 |
| l4／E<10Ha | 18,538,304 | 614,400 |

这些是四套原完整坐标和每套全部 6s₁/₂、6p₁/₂、6p₃/₂、5d₃/₂、5d₅/₂ 的计数规划，
每态明细和原物理 identity 在结果中。虚轨道仍为 414／441／532／567，所有
68 核电子和 28,218,500 至 50,351,931 核 rho 坐标不变；没有分配全 Yb rho 或
消费新的大空间 norm。因此它支持原 cap 下的资源候选，不能独自证明完整 Yb
实消费通过，不能解除 tail executor／实时 GPU／正式 angular 计数准入缺口。
原 NIST 对照、已有能级及 1% 门不变；没有新精度或受控加速比声明。

整合依赖顺序：先采用最终 helper 补丁并重放原有限／多核／guard 控制；随后给
[尾部资源工具](../../../../tools/atomic_tail_preflight.py)显式注入 source-bound
`plan_scratch`，对同一 core／valence 全部五态取最大值，保留原 64MiB 门和
`max(64MiB,norm_scratch)` 回放储备。工具不直接依赖私有 docs 模块，也不复制
第二份内存公式。主任务须冻结新源并以新目录重新预检；原尾部 run004 身份及
其已注册卡片批次保持封存。候选不授权长 Yb 求根。

重放使用新输出目录，驱动固定 threads1、RLIMIT_AS4GiB、alarm120s：

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_tail007/norm_followup001/probe.py \
  --output docs/atomic_pci_sd_native_2026_10_09/parallel_tail007/norm_followup001/replay005
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/pytest -q docs/atomic_pci_sd_native_2026_10_09/parallel_tail007/norm_followup001/test_candidate.py
```

[source](run004/source_sha256.json)和[input](run004/input_sha256.json)冻结实际源和
四套完整坐标；dirty 源封存在 `run004/source/`，种子振幅的独立 hash 在
[signed_input_sha256.json](run004/signed_input_sha256.json)，
[后核验](run004/source_and_input_postcheck.json)要求源和输入均未改变。
run001 保留初始外部注入候选；run002 保留自包含但仍留原 global cache 入口的版本；
[run003 失败](run003/failure.json)保留共享 helper 已变化后拒绝旧源码模式的结果。
run004 改为读取封存原 helper 并最终移除 global cache，当前依据仅此版本。
[文档影响与 EARS](documentation_impact.json)列明共享文档由主任务整合。
