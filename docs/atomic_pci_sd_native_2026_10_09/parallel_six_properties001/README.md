# 实际相关波算符的 E1/E2/M1 有限性质锚

新增 [FiniteCorrelatedWave](../../../src/atomx/properties/transitions/fscc_effective.py) 消费完整实际 eᵀΩ、total HEFF 和物理 metric，复用公共 `FiniteCITransitionOperator` 的 full-basis parent 绑定。它以完整同 H residual 为 admission gate；不会从能量或非 Hermitian HEFF 猜 metric，也不会接受此合同之外的截断波算符。

固定新生成的 Z=4、ns=30、ks=7、1s² 核与完整 2s/2p 的十个磁 spinor，采用同一实有限频横向 H。实际 backend 的 120 × 8 signed wave 与独立完整 CI 的相同低八维 invariant subspace 比较，P Ψ=I 固定相位/基，消除简并本征向量任意旋转的混淆。

| 完整检查 | 最大缺陷 |
| --- | ---: |
| E1、E2、M1 全部 11 个 spherical components，各 8 × 8 signed matrix | 2.665e-15 au |
| 实际 eᵀΩ 完整 H intertwining residual | 5.261e-13 Ha |
| metric 与 CI 相等 | 2.220e-16 |
| 完整 wave 与 CI 相等 | 5.085e-14 |
| spherical adjoint 相等 | 2.602e-18 au |

算符使用公共 `angular_eL/angular_mL` 和 `radial_eL/radial_mL`，独立 occupation-list 单电子 remove/insert signs 生成完整 determinant matrices。Bloch 与 orthonormal 两种完整变换均对照 CI；E1/E2/M1 三族都有非零结果。与 bare P O P 对照，E1 最大差异为 0.001728 au、E2 为 9.594e-6 au，验证测试实际消费了相关波与归一化，完整矩阵保存在 [数组](frozen001/full_arrays.npz)。

此锚仅认证声明的有限空间、长波 length E 算符和磁 spherical components。它们不是 reduced line amplitudes，未认证有限光子频率、dressed rates、lifetimes 或实验谱。`highest_level_reached=false`。

[前瞻合同](frozen001/contract.json)、[结果与全部门](frozen001/result.json)、[handoff](handoff.json)、[源码冻结清单](../parallel_six_fscc_scope001/frozen_sources.json)、[源码归档](frozen001/source_inputs.tar.gz)、[科学输入归档](frozen001/scientific_inputs.tar.gz) 提供所有实际身份。原生 P/Q 和完整 B-spline workspace 保存在 [径向输入](frozen001/native_radial_inputs.npz)，没有拼接历史不同 H。

入口运行实际新 H 的 backend solve 和全部 signed 性质检验：

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_six_properties001/probe.py --output <fresh-directory>
```

单线程、4 GiB 地址上限、backend 512 MiB、120 秒；封存运行 0.96 秒、RSS 367.0 MB。新增原理测试覆盖矩形双 endpoint metric、完整 signed CI 干涉、1e-18 弱算符、复数 metric/相位协变、owned 输入、错误 H/metric/parent 以及资源拒绝。连同四核 unique-pair lift 和既有方程/性质测试：[21 passed，0 skips](affected-panel.xml)。
