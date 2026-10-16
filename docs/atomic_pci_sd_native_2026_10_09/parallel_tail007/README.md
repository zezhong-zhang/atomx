# 完整 Yb 核—价虚轨道尾部：并行预检 007

[前瞻契约](contract.json)冻结原 V68／Z70／Fermi A175／ns123、实 Dirac–Coulomb、
全部 68 核电子、五个 6s／6p／5d 联合态。每个窗口的激发能必须以同任务实际
6s 附着能为零；不拼接不同空间的上下态、不拟合平移。四档为 lmax3／4 × Emax5／10Ha。

[probe.py](probe.py)复用公共 `build_atomic_space`：只对最大 l4／E<10Ha 完整
VN 正交补轨道带求解一次，再按原公共接口的严格 `<` 能窗保留完整子带。
相同 κ 的 Fock 本征问题不重复求四次，轨道顺序、符号、原核、共同 B-spline
度量保持绑定。[可复用预检工具](../../../tools/atomic_tail_preflight.py)调用既有
core／standalone SD 规划器，分别列出完整坐标、数值64、norm 回放、GPU staging、
工作与磁盘储备。它的当前联合态策略明确为这五个 Yb 态，不是新的通用 SD 求解器。

[最终实跑](run004/result.json)的十项完整径向／workspace／五列坐标／原 numeric64
内存公式／NIST 载荷／源码门全通过，2.467s，峰值 RSS 675,037,184B。时间范围从
驱动入口至结果写入前，含输入核验、源码归档、进口和径向构造；这是现场观察，
没有相同任务的受控端到端加速比。原径向系数最大差见[回放](run004/radial_replay.json)，
workspace 逐位一致。预检未编译／装载相关 backend、未作用或求解 SD 根、未使用 GPU。

| 完整窗口 | 径向虚轨道 | 核直接ρ坐标 | 含 norm／GPU staging 的主机载荷 / B | 设备载荷界 / B | norm 额外 scratch / B |
|---|---:|---:|---:|---:|---:|
| l3／E<5Ha | 414 | 28,218,500 | 16,932,678,612 | 520,003,072 | 23,609,344 |
| l3／E<10Ha | 441 | 31,991,337 | 19,045,445,524 | 584,137,408 | 23,798,272 |
| l4／E<5Ha | 532 | 44,367,272 | 26,509,474,292 | 671,069,696 | 88,381,952 |
| l4／E<10Ha | 567 | 50,351,931 | 29,903,029,236 | 754,411,584 | 88,618,112 |

主机载荷仍排除解释器／JIT／编译器 RSS，准入另外保留每任务 1GiB runtime。
GPU 任务仍保留原 2GiB device 上限，主机原 32GiB 上限不变。磁盘储备规定最多
16 个完整核向量、每态 32 个完整向量及 128MiB 小型元数据／构建储备，未经压缩
约 7.48／8.46／11.64／13.19GB；这是未来执行器必须落实的写入协议，不是任意
额外产物的无限上界。记录的 160 次作用向量条目数只衡量向量流量；完整图项 FLOP
或端到端时间界仍未知。

资源审计发现两个具体缺口。首先，历史 numeric64 主机界没有显式追加 CUDA
prepare 阶段的两份 angular rows／coeff、phi／energy staging 和 geometry staging。
此次基线另计 303,476,640B，不改写旧 006／norm 的证书或结果。设备公式独立对照
旧[5d5/2 实际 native GPU 规划](../yb_full_monovalent006/cuda/context_000005/cuda_full_plan_5_-3.json)
完全一致。

其次，现有物理 norm helper 的独立 64MiB 额外 scratch cap 不能容纳 l4 两档。
其完整 Gram 储备实际公式为 `2MiB +128*nκ^4*(KK+1)^2 +64*(KK+1)*maxradial^2`。
9 个 κ 的完整 l4 空间需要约 88MB，即使整轮主机／设备界通过，该消费者仍拒绝。
该拒绝是 run004 冻结源的真实门。后续
[有界 Gram 所有权候选](norm_followup001/README.md)已独立检验完整 l4 的 104 个
带符号／全部 M 范数情形，全部五态资源计数保留原 64MiB 门；具体暂存规划与
最小补丁另封存。它仍须由主任务采用并以新源重放完整 Yb 消费，不能覆盖本次
run004 的身份和失败，也没有提高 cap、减掉 g 波或直接重标长算准入。

四个物理空间可在独立进程中并行，分别拥有 source／core／backend／COMMON 和
完整五态。单个 owner 的原 native COMMON／锁约束不允许把其五态并行写入同一
backend；共享必须是明确来源相同的不可变输入。现有 CUDA dispatcher 明确要求
8 worker，预检同时给出 worker1 CPU 与 worker8 GPU 的独立载荷，未声称 worker1
已经能执行该 CUDA ABI。四个整轮的资源不能默认同时准入：每批先累加全部
主机／设备／磁盘储备，再读实时余量和负载；GPU 默认单完整任务独占。

全部长算仍为 `deferred`：缺少 source-bound tail executor、正式 native angular plan
计数和实时 device 余量；l4 还缺少符合原 cap 的 norm 消费。预检通过是计数与绑定
通过，没有新收敛能级或实验精度证书。实际
[NIST 对照](../yb_monovalent_experiment003/comparison.json)保持激发能最大偏差
4.1413%、劈裂最大偏差 9.3620%，原 1% 总门未过；本增量只引用并核验已有五态
root 与实验来源，没有生成新谱或预言尾部一定消除偏差。

重放（输出必须为新目录）：

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_tail007/probe.py \
  --output docs/atomic_pci_sd_native_2026_10_09/parallel_tail007/replay005
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/pytest -q tests/test_atomic_tail_preflight.py
```

驱动固定 threads1／RLIMIT_AS4GiB／alarm120s；失败保存 `failure.json`，预算不足
不能缩模型。[源码清单](run004/source_sha256.json)冻结实际 dirty Python／Fortran
和消费规划源，[输入清单](run004/input_sha256.json)绑定原参考、完整径向输入、
006 根、norm、NIST 与源码审计；源和输入运行前后都必须一致。
[九项测试](checks003.xml)全通过，涵盖保存的真实独立坐标／numeric64／设备／norm
证据、完整能窗、不得调用 native 构造器、联合态缺失拒绝、预算失败不得裁剪及
独立 norm cap 拒绝。初始 [run001](run001/result.json) 和 [run002](run002/result.json)
保留，为尚未加入显式 l4 norm scratch cap 检查的规划版本；run003保留为顶层card_summary字段位置修正前版本；当前依据为run004。
[文档影响与 EARS](documentation_impact.json)注明共享指南由主任务整合。
