# 有效态推进：跨 J、宇称的原生电多极谱线

2026-10-10，技术路线 M1 的有限精确消费锚已从同一 sector 的标量性质扩展到
不同 J/宇称之间的带符号跃迁。这一步实现公共底层接口和完整原生对照；
大空间 `ValenceSD`/`Sigma2SD` 的物理波算符与度量尚未由此建立。

## 接口与物理约定

`methods.ci_effective.FiniteCITransitionOperator` 保存只读矩形算符和两端实际
`FiniteCIModel` 快照。行属于 bra 的完整 CSF 顺序，列属于 ket。
字符串标签相同也不能换用另一个 parent；输入数组修改不会改动已保存算符。
泛用复矩阵只是一套显式数学控制；原生电多极 producer 另验实际物理身份。

`FiniteBlochReduction.effective_transition` 必须显式选择表示：

```text
O_B(b,k) = G_b^-1 Omega_b† O(b,k) Omega_k
O_orth(b,k) = W_b† O(b,k) W_k,  W = Omega G^-1/2
M(b,k) = L_b O_B(b,k) C_k = Psi_b† O(b,k) Psi_k
```

这里每端保留自己的 G、P 投影 C 和双正交左态 L。对普通复算符，反向算符满足
`G_k O_B(k,b) = O_B(b,k)† G_b`，不能把裸的 Euclidean 共轭转置当作反向
Bloch 算符。约化球张量反向时还保留对应的 J 相位。
`transition_amplitudes` 返回已选完整物理根之间的带符号矩阵元；它不丢弃 Q。

`properties.transitions.ci_effective.build_finite_ci_electric_operator` 复用
Hamiltonian 的 CSF tensor 相位、已有 Babushkin/Coulomb 径向算符。
它验证两端实际 CSF 顺序、带符号 spinor、workspace、Z、ns、核模型与共同
DC 或 static-DCB 覆盖；共同 union 的轨道度量必须正交。独立优化的非正交
端点需要另一套 overlap-aware 消费，不能放入此接口。

`finite_ci_electric_line` 直接从两端所选完整根取多电子能量差，在每个轨道对中
使用同一光子频率。默认使用有限波长算符；长波近似必须显式指定。速度形式保留
自己的径向核，不强迫其与长度形式相等。结果保存 `<upper||E_L||lower>`、S、
A、吸收 f/gf、物理根编号及完整算符。A 用 upper 简并度，吸收 f 用 lower。

```python
from atomx.properties.transitions.ci_effective import finite_ci_electric_line

# upper/lower 是各自 native parent.bloch_reduction(...) 的结果。
# upper_csfs/lower_csfs 必须保持 parent 的完整顺序。
line = finite_ci_electric_line(
    upper, lower, upper_csfs, lower_csfs, orbitals, workspace, Z, ns,
    upper_state=0, lower_state=0, representation="bloch",
    gauge="length", photon_approximation="finite_wavelength",
    nuclear_potential=nucleus,
)
print(line.omega_Ha, line.reduced_matrix_element_au, line.A_rate_per_s)
```

state 参数是 reduction 所选根列表中的位置，结果记录的是完整 parent 根编号。
一个矩形 photon 算符对应固定频率；其其余根对的矩阵元不能被自动当作各自频率的
谱线。每条实际谱线由 line 接口单独建立。这里没有 Atom 默认 dispatch。

## 完整判别与资源契约

本锚固定 Be⁺、Z=4、ns=30/ks=7、闭壳层 1s² 原生 DHF 和 VN 正交虚态。
完整轨道池是 1s、2s、3s、2p₁/₂、2p₃/₂；每个 sector 枚举所有三电子 CSF，
包括核空穴，没有按振幅筛选或删去高阶组态。

| sector | 完整 CSF | P / Q | `H Omega - Omega H_B` 最大残差 / Ha |
|---|---:|---:|---:|
| J=1/2，偶 | 17 | 2 / 15 | 6.85e-15 |
| J=1/2，奇 | 13 | 1 / 12 | 2.54e-15 |
| J=3/2，奇 | 15 | 1 / 14 | 5.43e-15 |

预先选择偶 sector 最低两根、两个奇 sector 各最低一根；不按结果替换根。
四组跃迁是 `2p₁/₂→2s`、`2p₃/₂→2s`、`3s→2p₁/₂`、`3s→2p₃/₂`。
与 length/velocity、finite/long wavelength 组合为 16 条控制记录。
独立 oracle 在最低 M 展开 CSF，执行显式费米产生/湮灭和单电子张量作用，
不调用 producer 的 `one_body_tensor_terms` 或有效算符变换；径向光子核是共同
定义，因而这项对照检验角收缩、度量和态消费，不是新的独立径向精度认证。

| 保持原门的全数组比较 | 最大差 |
|---|---:|
| 完整原生算符 vs 磁量子数行列式作用 | 2.67e-15 a.u. |
| Bloch/正交表示消费 vs 完整归一化态矩阵元 | 5.78e-15 a.u. |
| 带符号完整态重构 | 1.12e-16 |

原生签名/归一化比较门均为 `1e-11`，不是以本次机器精度结果重设门。
物理归一化态仅消费 P O P 所漏出的矩阵元最大差，在各控制中至少为
`2.3266e-3 a.u.`；实际 Q 贡献不能省略。该值是矩形矩阵的最大差，
不代表每条被选谱线都具有同样大小的修正。

构建前检查完整 CSF 维数、每个 CSF 的磁行列式计数上界、角工作上界、
矩形矩阵/拷贝/收缩/左态工作、角缓存和径向工作空间。默认 256 CSF/2000
磁行列式/1000 万角工作单位/64 MiB 新增工作预算；超过预算报错。
这是有限空间精确锚，不将 dense 小矩阵当作可扩展全原子 SD 实现。
缓存径向元只在同一固定频率算符内复用；没有删掉小系数。

本次同时移除了共享 CSF local/magnetic 展开、正交与非正交 one-body tensor
末端的四处 `1e-14` 振幅剔零，改为只省去实际等于零的项。完整计数上界仍在
展开前控制资源；保留非零项不会触发自动截断。新回归在同一个 5g³、J=9/2、
重数为二的角子空间内做正交旋转，保留全部 M 成员及归一化，验证约
`2.7756e-17` 的非零行列式分量没有被丢掉。另在复有效性质中检验 `1e-16`
量级算符的相对精度，绝对容差设为零。

当前证据保存在 [预设契约](native_bloch_transitions004/contract.json)、
[完整比较](native_bloch_transitions004/comparison.json)、
[完整原生数组](native_bloch_transitions004/complete_native_arrays.npz) 和
[源码哈希](native_bloch_transitions004/source_sha256.json)。运行前保存全部
Python 来源、runner 和 oracle 快照，运行后核验源码未变。
[runner](probe_native_bloch_transitions.py) 和新增测试均保留。
两份有限 CI 测试文件共 **21 项**；因共享角展开改变，扩展检查 14 份受影响的
角耦合、CSF、CI 与跃迁测试文件，最终 **240 passed、0 skipped**，见
[原始日志](validation004-tests.log)。五个修改/新增 Python 文件 Ruff 检查通过。
初次弱分量回归使用了不恰当的浮点严格相等断言；实际附加的单位 CG 因子
带来约 4e-16 相对舍入差。保留[诊断](weak_coefficient_fixture001.json)，改为
绝对容差零、相对容差 `1e-14`，仍能拒绝丢掉该非零分量；原生物理比较门未变。
此前 001–003 的完整证据均保留。
另验三份 scalar effective-CI、CI-MBPT 和有限 CCSD 消费测试，
[13 项全通过](validation004-consumers.log)，两组共 253 项、无跳过。
移除剔零前后的 205 个完整数组保留；完整 H、所选归一化态及物理矩阵元
逐位相同，16 个原生算符中保留的小项最大变动为 3.89e-16 a.u.。
[当前来源与本地文档核验](native_transition_review005.json)通过；545 个本地
路径均存在。全工作树对 HEAD 的 tracked-link 检查含先前未提交来源，
其失败记录保留在 004，不能据此声称干净发布 checkout 已验证。

## 能证明什么，下一关是什么

这一步证明有限精确模型的有效态能够用于真实原生电多极算符，且两种表示给出
同一归一化谱线强度。它未证明基组/径向收敛、实验精度、Yb 谱线精度、
大规模全阶 SD 物理度量、完整 FS-CCSDT 扩展性或任何 CPU/GPU 加速比。
同空间的 static-DCB 端点可由公共接口校验消费；本次原生 16 条记录固定 DC。
此有限模型的实离散能级也不是开放通道的复共振极点认证。

接下来仍须分别推导 standalone 单价 SD 的相关能/价单激发反馈，以及
CI+all-order 的能量参数和有效态/性质处方，再在同一原生空间与独立费米图对照。
固定分波参数的 Σ(E) 列不能直接当作这里的完整总能量 Hamiltonian，
也不能从任意 Σ′ 反推 G。只有这一关通过，才可把完整 Yb 带的方程根推进为
物理谱能、E1 及共同空间收敛，并检验既定实验门。
