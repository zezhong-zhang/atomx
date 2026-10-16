# 按实际 gap 选择方向与执行（2026-10-11）

用户要求每次发卡按高精度、高效率、高复用、模块化和架构自洽自主找到迭代方向。
本增量将固定领取列表改成 [证据选择器](../iteration_planner.py)、
[声明策略](../iteration_policy.json)、[生命周期 worker](../iteration_worker.py)和
[注册科学动作](../run_gap_step.py)；现有小时 timer 和当前 bridge 身份继续使用。
原始 [契约](contract.json)先于诊断和动作编写。

每次预览只作选择；成功发卡产生真实回执，再在串行锁内重读结果／数组哈希，
核对完整科学身份和资源。job 身份绑定动作、输入、策略和执行器源；已领取的
相同 job 不重复，失败和中断保留。实际结果推动下一选择，不是用一串测试代替科学迭代。
控制器没有可执行候选时仍输出实现方向，状态为 `implementation_frontier`。
当前已接入的自主能力是有边界的规则选择与有限实验执行，不能自动生成任意正确物理方程。

## 实际闭环

动态 ns=42/60/80/100 的最大差8.7498475e−10 Ha，相对展宽7.3593e−6；
DC 已保存对照在数值门内。策略要求同模型至少三点、展宽≤2%，因此停止继续网格扫描。
Z／核／后端／完整标签／物理源／频率处方不一致的数据不组成误差平台。

| 真实卡片消息 ID | 自动选择的动作 | 实际结果 |
|---|---|---|
| `om_x100b639914c198a0c120b4781150225` | 完整 signed Bloch 核孔分解 | 闭合8.8818e−15 Ha；56个一核孔及56个二核孔行列式均保留 |
| `om_x100b639913d5c0a0c2feacc1f385780` | 全横向 λ=0.5 | 完整 HEFF／FCI 差4.4020e−10 Ha；相对原差0.50310 |
| `om_x100b63992ed038a0c42ba9dccf009b0` | 全横向 λ=1.5 | 完整 HEFF／FCI 差1.3043e−9 Ha；相对原差1.49071 |

[分解结果](../../atomic_pci_sd_native_2026_10_09/hourly_iterations/adaptive_bloch_partition_c8e4c816342e38460b57/result.json)、
[λ=0.5](../../atomic_pci_sd_native_2026_10_09/hourly_iterations/adaptive_whole_transverse_control_8cdb1d19d577277ade5c/result.json)、
[λ=1.5](../../atomic_pci_sd_native_2026_10_09/hourly_iterations/adaptive_whole_transverse_control_305bc66990493034e3d7/result.json)
都保留实际源快照和完整数组。分解不使用未导出的后端振幅；用 exact Ω 代入
backend HEFF 的残差只是有效算符不一致诊断，不能据其扇区大小确定某个图项缺失。
全强度对照复用 ns80 的共同径向 DC／动态积分，在整个 one-body／ERI／常数／Fock
账本一致地缩放横向差；每个改变的 H 有自己的完整120行列式／8根 FCI。
没有删除 Fov、拟合移位、截小振幅或改写原门限。

响应相对线性的最大偏离约0.62%，因此下一审计优先核—价非 canonical 线性依赖项。
2% 响应偏离是方向选择的启发条件，不是精度证书；超出该条件时选择 Tcore／附着簇
非线性耦合与反馈审计。当前未确定缺失图项，原严格1e−10 Ha 目标仍失败。
下一步要推导完整图式，接入独立候选后端并逐项通过 Bloch 对照。

## 资源、复用和边界

完整有限磁空间适合一线程 CPU；地址空间4 GiB、服务6 GiB、任务600秒，
CCSDT后端512 MiB／120秒。执行前至少1 GiB可用内存、128 MiB空闲磁盘；
不足时不领取、不缩小物理模型，后续卡片继续重新判断。
科学动作复用已保存积分、公共解码器、pinned后端和独立 oracle。
DC 源是横向源集合的共同子集，横向四个额外算符明确保留；不能要求整个源集合相等。
[初次验证失败](test_failure001.json)记录这一身份判定错误及修正，原门未变。

动作内部测量时间约0.006／0.447／0.509秒、进程峰值 RSS约162／168／167 MB；
计时从源码归档后开始，排除 Python 导入和归档。这些是当前负载下的阶段观察，
不是受控计时或端到端 CPU/GPU 加速比，不外推生产规模。
新物理图项、Yb完整空间尾部、dressed性质和受控大规模GPU性能仍未完成。

首次三项执行的物理源／job／输入快照完整，选择器对应版本保存了哈希与完整决定；
此后控制器另外归档 exact 源／策略／进度，避免仅靠 hash 而缺少原文件。
原先分子／晶体计划继续暂缓；NIST远程网站仍是此前发布的私有快照。
不把有限模型原门通过或历史对照提升为原子级最高水平。

[交付回执与控制器快照](delivery_proof.json)记录三项真实动作、最终方向卡片、
18项受影响检查和499个本地文档链接（0个缺失）。
[检查日志](checks.log)与[JUnit](checks.xml)保留测试结果；这些测试不替代三次实际科学执行。
[下一决定](next_decision.json)保存完整证据依据与当前实现前沿。

```bash
.venv/bin/pytest -q tests/test_atomic_iteration_queue.py tests/test_atomic_iteration_planner.py
```
