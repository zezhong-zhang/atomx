# 四核电子 FS-CCSDT 有限投影扩展

公开可选修复 backend 的 12-spinor 四核电子锚已通过保留秩方程，仍存在真实 T4/S4 截断。此目录没有替换 backend，没有改变科学门来匹配 FCI，也没有原子实验精度或生产大空间声明。

| 实际同 H 检查 | 最大缺陷 |
| --- | ---: |
| 核 T1/T2/T3 的 424 个投影 | 5.62e-14 Ha |
| 附着 S1/S2/S3 的 456 × 8 个投影 | 6.48e-14 Ha |
| 全 8 × 8 signed HEFF 与实际 Hbar Ω 投影 | 5.01e-15 Ha |
| 未保留核 rank 4 residual | 0.002117 Ha |
| 未保留附着 rank 3/4 residual | 0.002284 Ha |
| 核 CCSDT 与完整 FCI 方法误差 | 3.17e-6 Ha |
| 附着 CCSDT 与完整 FCI 方法误差 | 9.09e-6 Ha |

固定 Z=4、ns=30、ks=7 的两核 DF/VN 轨道，1s²2s² 参考、完整 2p 与 3s 虚态；495 个核与 792 个附着 determinants。核—虚 Fock 最大耦合实际为 **0.05469766 Ha**，因此该扩展具有非 canonical 耦合信息。所有 792 × 8 residual、完整 signed T/S、Hbar、Ω、eᵀΩ、metric 和 FCI 根都保存在 [完整数组](frozen001/full_arrays.npz)。

[结果与门](frozen001/result.json)、[前瞻合同](frozen001/contract.json)、[冻结源码清单](frozen_sources.json)、[handoff](handoff.json) 提供源、二进制和输入身份。[科学输入归档](frozen001/scientific_inputs.tar.gz) 包含实际积分、原始 DG 幅度、原始 stdout/status、完整数组和合同；[源码归档](frozen001/source_inputs.tar.gz) 包含实际 dirty Python/C 源及已编译的 public backend。

入口实际重新生成声明的小规模 H，并重新运行 backend 和完整独立投影审计：

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_six_fscc_scope001/probe.py --output <fresh-directory>
```

单 BLAS/CPU 线程、地址上限 4 GiB、backend 512 MiB、120 秒；封存运行 6.70 秒、RSS 403.6 MB。既有 `expt.x` 和参考 checkout 均只读。当前 `highest_level_reached=false`，下一缺口是 T4/S4、完整核—价空间及其收敛。

负结果也保留：[1e-14 更新阈值的 200 次未收敛](core4_n12_001/result.json) 与 [原始日志](core4_n12_001/fscc/cc.log)。独立前瞻重试仅将迭代控制设为 1e-13，实际完整投影 1e-11 Ha、HEFF 1e-10 Ha 门保持不变。先前 [n10 启动失败](core4_n10_001/result.json) 是缺少共享库路径；[修正路径后的 n10 控制](core4_n10_002/result.json) 通过，但 Fov=0，仅为截断子条件控制。

独立 unique-pair occupation-list lift 验证完整 T1/T2/T3 与 S1/S2/S3，另以有限 nilpotent 多项式而非 expm 重建核投影。最初 1e-14 测试门有一个幅度坐标出现 1.068e-14 数值反对称噪声，[负测试](unique_pair_numerical_floor001.xml) 及 [当时精确测试源码](unique_pair_test_before.py) 已保存；最终采用前瞻科学合同的 1e-12 signed lift/归一化门，投影能量门未改变。两个新测试及既有性质/方程审计合计 21 passed、0 skips，见 [联合测试](../parallel_six_properties001/affected-panel.xml)。
