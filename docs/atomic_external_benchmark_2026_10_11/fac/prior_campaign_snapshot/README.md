# Atomx 与参考软件：精度、效率与并行效率

实测完成。完整共同输入上的数值实现一致性已通过；大 Yb term2 阶段在8线程达到 **4.790× / 59.87%**，小 Na⁺ 完整 SD action 的4线程只有 **1.292× / 32.30%**。Atomx 的固定能量 Σ₁ 接口在本面板比 AMBiT 耗时更长。实验精度仍未通过原1%目标，不能据这些数字宣布整体领先或达到最高水准。

依据：[前瞻合同](contract.json)、[原始来源及汇总身份](aggregation_provenance.json)、[完整汇总](metrics.json)、[可重放 CSV](performance.csv)。CPU 是32逻辑线程/16核心 Ryzen 9 9950X3D；GPU运行时报告 RTX4090。三路独立准备、实际计时共用 flock；每进程原8GiB地址上限、每流程600秒、实际线程上限8不提高。软件/接口一般5轮交错配对，大Yb完成3个独立完整流程。S=逐轮T₁/Tₚ的中位数，E=S/p；时间另取各档中位数。图中范围包含所有样本，不是置信区间。

## 共同输入的数值精度

| 对照对象 | 实际比较的完整范围 | 最大差或残差（Ha） | 结论边界 |
|---|---|---:|---|
| Atomx / AMBiT | Ne 19轨道、37,350 signed径向积分、27矩阵项×2p1h/2h1p/总和 | 1.429×10⁻²¹ | 独立 AMBiT MBPT/角动量公式；径向输入共享，非实谱精度 |
| 当前 / 原 pCI-derived 图式 ABI | Na⁺ 10核电子/8径向虚轨道，五价列完整公开 ValenceSD action 与响应 | 8.674×10⁻¹⁹ | 同图式谱系，非独立 pCI 主程序；不是完整价波函数求解计时 |
| 当前 / 原 pCI-derived 图式 ABI | Yb 68核电子/414径向虚轨道，全64×64×1076数值及同形响应 | 3.117×10⁻¹⁹ | **封存 signed Gaussian stress 输入/零核seed**；各档逐位相同，非已收敛Yb SD波函数 |
| 显式修复 EXP-T / 独立完整费米投影 | 两核10spinor，全部120附着行、8模型列与六类振幅 | 1.776×10⁻¹⁵ | 全 Bloch 方程同样通过；8根相对有限FCI差1.243×10⁻¹⁴ |
| 显式修复 EXP-T / 独立保留秩投影 | 四核12spinor，全部792附着行、8模型列与六类振幅 | 6.484×10⁻¹⁴ | 仅保留的CCSDT方程闭合；全Bloch残差2.284×10⁻³，FCI方法误差9.093×10⁻⁶ |

共同算符原 absolute1e−11Ha/relative1e−10门未放宽。FSCC 两套实际本地构建都含 Atomx ABI/积分交换补丁，不能称未修改上游；原本地默认后端在这两个 **noncanonical Fov 压力模型** 的附着方程残差4.164×10⁻⁷/1.031×10⁻²Ha未通过，显式修复后才闭合保留方程。四核仍缺T4/S4，FCI方法误差不并入数值容差。失败不推广成 EXP-T 全部生产任务的精度判断。

来源：[AMBiT独立实跑](ambit/README.md)、[SD完整空间实跑](sd/README.md)、[FSCC全数组与遗漏秩审计](fscc/README.md)。pCI pin为4d685d82、AMBiT为fe1115aa、EXP-T为aa8d809a；实际修改、二进制和运行库身份在各owner。Atomx FSCC直接调用EXP-T，不作为两套独立求解器。

## 接口耗时与冷准备

| 实测范围 | 当前Atomx | 参考/对照路径 | 可作出的比较 |
|---|---:|---:|---|
| 完整Ne固定能量Σ₁ warm消费 | 41.160ms | AMBiT 18.630ms | 配对接口耗时比2.259；Atomx额外构造可复用poles/residues和物理binding，非端到端排名 |
| 完整Na⁺五列SD action warm | 4线程20.375ms | 原图式串行52.125ms | 同输入同响应，配对当前/原图式加速2.536；不是独立pCI程序倍数 |
| 两核完整EXP-T调用 | 私有低开销wait wrapper315.232ms | 裸backend312.197ms | 同一引擎，配对额外时间中位5.759ms（−2.460至11.792ms）；两个时间中位数之差另为3.035ms |

Na⁺每backend冷setup4.14–4.39s，峰RSS333MB；SCF/虚轨道时间未独立测量，记unknown而非零。Yb round000/001/002冷setup分别109.87/115.88/108.01s，峰RSS3.223GB；其计时排除冷准备、压缩保存/数组复验和共享锁等待。AMBiT每次BeII重新建科学工作目录，包含完整HF/基/MBPT/CI与I/O，OS/运行库缓存warm。有限全FCI矩阵构造及全特征解成本独立保留在FSCC报告，四核精度/工作量与CCSDT不同，不作速度排名。

原 Python `subprocess(timeout=...)` 轮询等待会量化短程序的退出计时。AMBiT使用C++ monotonic fork/exec/wait4重测；EXP-T用私有blocking waitpid+独立watchdog重测，生产公开等待器未修改。原公开路径壁钟记录是保留的部署观察，不能推断纯backend耗时。private wrapper数字也不能代替未修改公开wrapper的延迟。

## 固定工作量CPU并行效率

| 工作量 | 1/2/4/8线程中位耗时 | 2线程 S / E | 4线程 S / E | 8线程 S / E |
|---|---|---|---|---|
| Atomx Na⁺完整SD action，5轮 | 26.355/22.120/20.375/22.412ms | 1.195 / 59.77% | 1.292 / 32.30% | 1.159 / 14.49% |
| Atomx Yb完整term2 stress，3轮 | 82.119/43.954/24.563/17.087s | 1.868 / 93.42% | 3.408 / 85.20% | 4.790 / 59.87% |
| AMBiT BeII完整CLI，每档5对 | 约1.028/0.824/0.751/0.762s | 1.239 / 61.95% | 1.379 / 34.47% | 1.390 / 17.37% |
| EXP-T四核有限CCSDT，5轮 | 0.627/1.121/拒绝/拒绝s | 0.559 / 27.97% | null | null |

AMBiT各档有各自配对的串行基线（1.018/1.028/1.056s），表中1.028为15次串行的合并中位数，S由各自配对重算。AMBiT线程数来自程序OpenMP日志，未测OS活跃占用。SD保留编译num_threads、绑定BLAS=1与阶段稳定窗口实际CPU线程观测；首轮Yb单线程未采样，保持null。

EXP-T请求2线程时实际进程线程观测峰5、活跃峰3，采样CPU/壁钟约2.695–2.785；E=27.97%是请求team口径，按采样进程峰线程数另为11.19%，都不是CPU利用率。请求4线程首次超限9、最终观测峰10；8线程旧路径峰25，均超过原8线程cap而终止，效率保持null。负结果和原运行完整保留。

![各固定工作量的配对加速比和逻辑线程效率](parallel_efficiency.png)

导出：[SVG](parallel_efficiency.svg)、[PDF](parallel_efficiency.pdf)；从CSV重画：`.venv/bin/python docs/atomic_external_benchmark_2026_10_11/plot_scaling.py`。完整汇总重放：`.venv/bin/python docs/atomic_external_benchmark_2026_10_11/aggregate.py`。

## GPU和实验精度

[CPU/CUDA实际控制](gpu/README.md)在原4GiB地址/32MiB payload门内全数组逐位一致，449项运行依赖前后哈希一致；实际设备payload1.362MB。五轮小矩阵观察中位CPU7.783µs、GPU含上传/下载133.875µs、resident host34.684µs。GPU当时有其他compute进程，负载门失败，**受控speedup/效率均null**。这是小矩阵与传输/启动范围，不能推断大规模GPU全SD性能或用CUDA核心数定义并行效率。

[原完整YbII/NIST上下文](experimental_precision_context.json)采用原五态同H/空间/6s零点与本地ASD5.12（349levels/327lines）。四激发偏差+0.925/+1.168/+4.141/+3.380%，只1项通过1%门；两精细分裂+3.146/−9.362%，均失败；四目标线仅两项唯一observed、其中一项通过。参考/理论u及协方差未知，不画伪造误差条。**未重新求谱**，没有本轮同完整模型的上游生产Yb谱，不能比较三软件的实验精度排名。

![原五态完整YbII实验差距](experimental_precision.png)

[实验CSV](experimental_precision.csv)、[图来源身份](plot_provenance.json)。历史其他软件的独立模型保留在[外部验证总览](../BENCHMARKS_EXTERNAL.md)，不混入此次公平比较。

## 核验与后续方向

各owner重新读取全部signed输出。根独立重读Na⁺全部500数组；另有独立1,829文件/483MB重哈希审查，0缺失/变更；Yb30完整数值/响应数组、365源码/6完整输入/二进制及实际运行库逐轮通过。AMBiT4项、FSCC16科学回归+2启动器检查，以及卡片相关15项检查通过；实际卡片渲染和当前工作树链接验收记录在交付证明。原失败、超限和撤回的计时记录均保留。

下一阶段优先完整已收敛求解的阶段profiling与共同模型交换，补齐独立pCI主程序/AMBiT同完整Yb谱对照；当前没有已编译独立pCI all-order程序，串行all-order目标不要求MPI，MPI工具缺失仅限制MPI-CI测试。性能方向是复用Σ₁数据构造、合适任务粒度、避免小张量过多线程，以及完整大数组GPU驻留/批处理后的负载合格重测；科学方向仍是生产全阶SD/FSCC保留秩与物理空间收敛、性质和NIST精度。实验差距不靠放宽容差或性能倍数关闭。

卡片每次重新读取本汇总并记录SHA；成功receipt仍走现有gap/资源规划。未登记的新实现缺口不伪装成已经自动执行的生产计算。没有云端重部署，Lark发送的图表与卡片可远程查看。实际发送、服务端回读和最终来源重核验见[交付证明](delivery_proof.json)。
