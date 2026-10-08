# 显式 CI 初态与统一原子态设计

本轮实现把实验初态与轨道优化平均分开；原则与前轮实测洞察已写入
[AGENTS.md](../../AGENTS.md)，完整设计见 [ATOMIC_STATE_DESIGN.md](../ATOMIC_STATE_DESIGN.md)。
新接口已落在 atomx，共享既有物理算符与 CI 内核。

## 已实现

- 公共 `CILevel(J2, parity, root)`：按 J/宇称块内能量顺序选择根，
  无跨对称能量交错的整数列号歧义。不是组态特征 tracking。
- 公共 `CILevelPopulation(((CILevel(...), weight), ...))`：权重为总
  能级占率；归一化后每 M 分配 `p/(2J+1)`，无额外统计权。
- `MCDHFCoherentModel(initial_state=...)` 及
  `TwoElectronCoherentModel(initial_state=...)` 接收这些选择。
  后者当前在父二电子扇区制备；显式磁密度的既有入口仍可使用。
- 优化 `CIStateAverage` 与实验初态复用
  `physics/csf/levels.py` 的谱验证、分块与简并权重检查。
  两者类型独立，拒绝把优化对象当成实验制备。
- 同对称精确简并只允许整个子空间等权的非偏振混合；相干制备需
  显式指定磁基底密度。兼容整数列号仅解释为当前快照的列。
- 默认制备的归一化能级占率记录在模型 provenance；显式密度覆盖时
  该字段仍标为 default，未宣称它是覆盖后的实际输入记录。

## 连续态结论

当前两电子 CI 连续通道在冻结残余离子的核势+直接 Hartree 势中数值
求解 Dirac 连续态，再投影掉保留的同 κ 束缚空间：静态、投影后的 DW
近似。渐近 δ(E) 归一化及径向/角核复用既有模块。没有非局域交换、
离子响应、多通道/Fano 或完整自洽散射；内部投影波不等于未经投影
局域势方程的精确本征解。详见设计 §5 及源码
`scattering/ci_continuum.py:_field/_prepare_waves/_store_wave`。

## 验证

[tests.log](tests.log) 记录 **70 passed，77.10 s**，命令：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  tests/test_ci_level_selection.py tests/test_ci_state_average.py \
  tests/test_xfel_mcdhf_coherent.py tests/test_xfel_ci_continuum.py \
  tests/test_xfel_coherent.py tests/test_mcdhf.py
```

覆盖：跨对称顺序、简并子空间旋转、超大相对权重、无效选择拒绝、
非等权 EOL 流形切向梯度、公开 MCDHF 路由、真实 Ne 激发态/混合初态
零通量解析指数衰减、相干相位协变、N2/1/0 联合密度嵌入、连续通道
与原有物理门禁。新混合初态和基态的驱动算符及基底标识完全一致。

[verify.py](verify.py) / [verify.log](verify.log) 重放已有完整基态密度
轨迹：**最大差异 0.0**。**226** 个历史科学归档文件、**134** 个 SCF
快照和参考程序/源码哈希保持不变。既有 FAC/XATOM 数率未重新生成。
[identity.json](identity.json) 固定本轮源码及证据身份。代码及验证脚本
Ruff 检查通过，`git diff --check` 通过。没有本轮整体加速宣称。

新增单测最初把统计归一化后的浮点 0.6 误要求为十进制字面值的位级
相等，偏差 1.11e-16；已改为数值等价检查。最终记录来自修正后的完整
运行，不能把早期 40/41 或之前 123 项的重叠子集相加。

## 边界与后续

`atom.state(...)`、绑定快照的结构化纯 M/相干制备、性质态视图、特征
root tracking、多电荷共同优化、独立弛豫轨道转换、一般 N→N−1 损失
及非世俗连续环境均属于设计后续，尚未实现。一般多电子 REXMI 不因
新增初态接口而完成。存量相干模型适用性门禁保持。

未 commit/push，工作树保留其他会话的在途改动。
