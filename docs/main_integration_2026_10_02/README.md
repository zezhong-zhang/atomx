# 主线进展与工作区修复整合

当前工作区已在 `main`，HEAD 为 `1ca20d5`。执行 `git fetch origin main` 后没有发现远端新增提交，`git merge origin/main` 返回 `Already up to date`。主线最近七个提交已在当前代码中，本轮未产生新的 merge commit，也未覆盖、提交或推送已有工作区改动。

本轮验证了主线的完整 n=3 CI、共振轨道输入接口与工作区 DF/PI 修复的兼容性。相关检查 **56 项通过、1 项预期失败**；[日志](validation.log)、[源码身份](identity.json)。这些检查包含 Hamiltonian 平均的 CSF 干涉项、两套径向求解器、能量方向导数、统计电子对权重、PI/RR 外部面板以及散射接口。

## Fe LMM 的最新实测结果

| FAC ILEV | FAC 共振能量 eV | atomx 共振能量 eV | Auger atomx/FAC |
|---|---:|---:|---:|
| 32 | 287.19 | 285.319 | 0.931276 |
| 71 | 319.28 | 318.242 | 0.892227 |
| 72 | 321.38 | 319.970 | 0.977110 |
| 142 | 371.97 | 371.274 | 1.114327 |
| 190 | 409.13 | 409.192 | 1.091806 |

[原始数值](fe_lmm.json)来自本轮实际运行的 `test_fe_lmm_mscheme_panel.py`。五个锚点均通过原有 15% 门槛，最大偏差 11.43%；10% 门槛仍为预期失败，未放宽测试。配对使用相同 J 块中、参考能量 ±2.5 eV 内宽度最大的候选态，因此这五个比值不构成严格逐能级对应的精度证明。

该面板使用 `fac_frozen`：先优化参考组态，再在公共轨道上对角化完整 n=3 CI。`active_result` 只传入另一径向计算的轨道，不能证明这些轨道对 Auger 流形的 Hamiltonian 已达到驻点。源码和测试中的“CI 平均松弛”错误描述已修正。本轮没有把冻结面板改称 Hamiltonian 松弛面板。

## Hamiltonian 松弛与统一矩阵

现有 `hamiltonian` 单组态路径及默认 `mcdhf` 变分路径用完整的 CI Hamiltonian 期望值更新轨道，保留非对角 CSF 干涉项，并在轨道变化后重新对角化。当前目标是基态及其严格简并态平均。多共振目标态的加权 Hamiltonian 松弛还需要明确目标态及权重，不能由平均占据数或冻结参考近似替代。

[更新后的九格矩阵](matrix.json)并列使用上一轮的 PI/荧光实测面板和本轮 Fe 完整 CI 锚点，其余 Auger、产额、散射仍指向各自归档证据。矩阵统计的是覆盖范围，各模型及配对限制按格标明。

```bash
cd ~/code/atomx
.venv/bin/python benchmarks/bench_systematic_fac_matrix.py \
  --consistency docs/atomic_consistency_2026_10_02 \
  --integration docs/main_integration_2026_10_02 --json
```

本轮没有重新计算六元素 PI/荧光 Z 面板或 K 壳产额。此前的数字和源码归档保留在 [统一场对比报告](../atomic_consistency_2026_10_02/README.md)，本页记录最新主线的兼容性检查与新增 Fe CI 证据。
