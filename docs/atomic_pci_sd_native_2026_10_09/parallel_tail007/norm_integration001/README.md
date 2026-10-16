# 有界范数缓存：采用与完整尾部重新预检

本次采用保持原完整范数方程、signed 干涉、径向坐标、全部 M 分量及原
1e−11／64MiB 门。每个消费者独立拥有 256-entry／4MiB 的只读完整 Gram
缓存，淘汰后完整重算；移除原全局 4096-entry 缓存。资源计划包括径向视图、
Python metadata、全部组及角／径向临时数组，runtime RSS 单独储备。
[前瞻合同](contract.json)与[采用身份](adoption_identity.json)记录原源和最终源；
函数／类 AST、缓存限额与[最终候选](../norm_followup001/run004/result.json)一致，
仅 import／格式整理不同。原 helper 封存在 `original_helper.py`。

| 实际新源控制 | 数量 | 最大差 |
|---|---:|---:|
| [原有限体系 fermion 展开](finite004/result.json) | 672 分量／M 门，80 个情形 | 4.441e−16 |
| [原多核 fermion 展开](multicore002/result.json) | 168 分量／M 门，20 个情形 | 2.220e−16 |
| [完整 s/p/d/f/g 独立候选控制](../norm_followup001/run004/result.json) | 104 个 signed／全部 M 情形 | 4.441e−16 |

最初 finite002 缺少原生构建环境，finite003 因串行／OpenMP BLAS 同 SONAME
混用被运行时门拒绝，日志和输入保留。finite004 统一使用 OpenMP BLAS 后实际
通过；multicore002 使用最终整理源。早期 multicore 通过属于前一源版本，不替代
multicore002。tail001 初版结果引用了早期 multicore，本次改为与源码检查一致的
multicore002，并在新目录[tail002](tail002/result.json)重新运行；旧结果不改写。

`tools.atomic_tail_preflight.estimate_task(..., norm_planner=plan_scratch)` 显式
复用已验证规划器，工具不依赖私有 docs 模块，也不复制公式；默认保留旧规划，
历史回放身份不变。全部四档五态最大 scratch 为14,148,096／14,363,648／
18,273,472／18,538,304B；同一 V68／Z70／ns123／Fermi A175、414／441／532／567
径向虚轨道与原 numeric64 计数保持不变。全部七项预检门通过；原 l4 超预算
[失败](../run004/result.json)保留。大档约29.90GB完整host载荷并未因缓存预算
下降而消失，四个重任务仍不能同时准入。

这是完整坐标预算与有限消费证明。新尾部空间的完整 Yb norm、相关根、GPU
生产及实验精度尚未执行；全部四档生产状态仍 deferred，等待实时 GPU 内存、
完整 native angular 规划和 source-bound tail executor。不能用这些计数证明
新的收敛谱或最高水平，也没有受控性能倍数声明。

[新探针合同](tail_probe_contract.json)绑定完整原径向带、当前源、实际 finite004／
multicore002 及采用身份；[自动并行控制器](../../../atomx_hourly_progress_2026_10_10/parallel001/README.md)
以新 job 身份领取，只重新执行此次有界预算探针，已完成的其他方向不重复。
运行前合同、实际源／输入档案和后核验保留在每次输出目录。
