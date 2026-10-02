# Gu 冻结场生产实现：共享绑定 SCF 势，避免第二次 SIC 拟合

本轮把[上一轮隔离实验](../error_isolation_2026_10_02/README.md)确认的不一致修到生产路径：B-spline Gu SCF 收敛后保存原生 Gauss 网格上的电子中心场；`fac_frozen_core` 直接复用这个场。连续态不再从另一个径向网格重建密度、Hartree 场和 SIC。点核与有限核的 HDF5 保存/加载也保留冻结场。相关测试 **100 项通过**，新增针对冻结场的测试 15 项。

## SIC 的含义与物理判断

SIC 是 self-interaction correction（自相互作用修正）。Hartree 场包含电子自身的电荷贡献；精确 HF 的对应交换项会抵消它，局域交换近似通常只能部分抵消。[Engel 的交换势研究说明](https://itp.uni-frankfurt.de/~engel/opm.html)给出了这一物理区别。

atomx 的 Gu/FAC 路径是局域近似：屏蔽电荷的尾拟合配合 Latter 修正，使孤立原子/离子的势具有 `−(Z−N+1)/r` 尾部。相关实现位于 [local_potential.py](../../src/atomx/numerics/bspline/local_potential.py)，移植对照为 [FAC radial.c](/home/zzhang/code/fac/faclib/radial.c:1957)。尾电荷正确是必要约束；它不能证明整个空间的自相互作用已精确消除。这不是完整轨道依赖的 Perdew–Zunger SIC，也不等价于非局域 HF。

从 Hamiltonian 一致性判断，冻结近似应冻结同一个场。旧路径先在 Gauss 网格完成绑定 SCF，再在默认 4000 点对数网格重新拟合 SIC；离散转接点及其附近斜率随网格变化，使绑定态和连续态使用不同中心场。增加连续网格或换插值只能改变这个差异，不能保证两者共享 Hamiltonian。因此本轮保存场本身，未通过调参或输入 FAC 能量缩小差异。

## 实现与接口

- [DFSolver](../../src/atomx/methods/dirac_fock.py) 在收敛结果的最终密度上，用同一原生 Gauss 网格、核模型及 Gu 参数计算并保存中心场。它与绑定 direct/exchange 矩阵构造器独立组装后的矩阵闭合已经测试；不宣称最后一次未完全消失的 SCF 残差为零。
- [FrozenLocalField](../../src/atomx/potentials/frozen_field.py) 保存只读的 `r * V_electronic` 表，插值电子屏蔽电荷并保留原解析核势。原点电子势有限，外部延拓使用规定尾电荷；原 Gauss 节点上的场值保持一致。连续网格不触发第二次 SIC 拟合。
- [PotentialModel](../../src/atomx/scattering/models.py) 拒绝不收敛结果、缺失或占据/Z 不匹配的快照，以及冻结场上的额外电子移除、密度缩放和 Gu 参数不匹配。`frozen_scf()` 明确继承保存的实际 Gu 参数。
- [PI](../../src/atomx/scattering/photoionization/subshell.py) 记录实际场来源 `bound_scf_snapshot`；每个子壳层构建的势交给积分流程复用，避免为 provenance 再做一次势重建。
- [HDF5](../../src/atomx/io/results.py) 增加可选 `local_central_field` 组，保存径向表、参数和组态签名；加载时核对核模型身份并恢复原核势。旧文件仍可加载，但缺少快照的 Gu 文件不能自动假装使用同一冻结场，需要重新求解或显式选择重建模型。

默认 Gu 使用示例：

```python
from atomx.methods.dirac_fock import DFSolver
from atomx.scattering.models import ContinuumModel, PotentialModel
from atomx.scattering.photoionization import subshell_photoionization

result = DFSolver.from_configuration(
    10, "1s2 2s2 2p6", exchange="gu", ns=180, ks=9,
    r_max=80, starting_potential="tfda",
).solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
model = ContinuumModel.fac_frozen_core(n_grid=16000, r_max=80)
pi = subshell_photoionization(result, [1028, 1714, 3428], continuum_model=model)
result.save("ne-gu.h5")

# 自定义 Gu 参数时显式继承快照，而非要求默认 FAC 参数。
custom_model = ContinuumModel(
    method="shooting", potential=PotentialModel.frozen_scf(),
    shooting_n_grid=16000, shooting_r_max=80, orbital_removal="none",
)
```

适用范围为孤立、收敛的 B-spline Gu 结果。等离子体屏蔽核势和 shooting Gu 结果尚无这个快照，冻结接口会明确报错。HF 没有可直接冻结的局域交换 Hamiltonian：既有 HF 绑定态＋Gu 连续态路径保留，来源标为 `hf_bound_gu_continuum_hybrid`。需要诊断旧重建近似时，可显式使用 `PotentialModel.static_hartree_exchange(exchange="fac")`，并选择 `orbital_removal="none"`；该模型的来源是 `density_reconstruction`。

## 六元素重算结果

[逐点 CSV](pi_validation.csv) 有 288 个值：六元素 ns=180，以及 C/Ne ns=360；每个结果四个连续态控制，每个控制包含 1s/2s/2p 的三个光子能量。绑定 SCF 使用原生 Gu；所有控制固定先前归档的原生 Gu relaxed ΔSCF 阈值，没有输入 FAC 能量。这检验场一致性和径向表示，不是重新计算整套松弛阈值。

ns=180 的生产 Gu / 归档 FAC 比值范围（每元素九点）：

| 元素 | 最小比值 | 最大比值 |
|---|---:|---:|
| C | 0.999389248 | 1.000068253 |
| Ne | 0.999657515 | 0.999978286 |
| Ar | 0.999908667 | 1.000031458 |
| Fe | 0.999885155 | 1.000033724 |
| Kr | 0.999532968 | 1.000084407 |
| Xe | 0.997362581 | 1.000453844 |

Ne 高能 2p（3428 eV）的比值从旧重建路径的 **0.998878716** 变为 **0.999816893**，截面相对旧值变化 +0.093922%。修复并不保证每点更接近 FAC：例如 C 2s 低能点比值从约 0.999404 降为 0.999389；旧路径的误差有时也会相互抵消。

控制结果：

- 全部八个绑定结果，将势重建参数 `n_radial` 改为 128000 后，生产 PI **逐位一致**；冻结场不再依赖这个重建参数。
- 六元素 ns=180 的 shooting 网格 16000→32000，九点截面的最大相对变化为 **0.000340979%**。
- ns=180 的显式旧重建模型复现旧九点归档，全部 54 点闭合至相对 `1e-11`。
- Ne 高能 2p 的完整绑定 SCF ns=180→360 变化从旧路径的 **+0.152331%** 降为 **+0.032323%**。新路径 Ne 九点最大变化仍为 **0.060357%**；C 最大变化为 **0.031421%**。SCF 本身的 SIC 离散及绑定表示仍需继续收敛，不能宣称径向误差已消失。

[势构造计时](construction_timing.json)为 Ne ns=180、预热后 200 次构造：快照约 1.79 μs，旧密度重建约 626 μs。这只测势工厂调用，不代表 SCF 或整个 PI 计算获得同样倍数的加速。

## 验证、复现与未解决事项

测试覆盖独立绑定/连续矩阵闭合、禁止重拟合 SIC、离子与单电子尾电荷、有限核原点、参数/占据身份检查、HF hybrid，以及点核/有限核 checkpoint 后势和 PI 的逐位一致。单电子测试只验证尾电荷，不宣称 Gu 单电子体系已精确无自相互作用。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FAC_SFAC_PATH="$HOME/code/fac/bin/sfac"
D=docs/frozen_gu_implementation_2026_10_02
.venv/bin/python "$D/validate_panel.py"
.venv/bin/python "$D/verify_artifacts.py" --live-check
.venv/bin/pytest -q tests/test_frozen_gu_field.py tests/test_scattering_models.py \
  tests/test_configuration_spectroscopy.py tests/test_bspline_local_potential.py \
  tests/test_xray_scattering.py tests/test_pi_rr_vs_fac.py \
  tests/test_scfresult_save_load.py tests/test_scf_potential.py \
  tests/test_dfsolver_slater.py tests/test_dfsolver_fac.py --disable-warnings
```

原始数据、最终测试日志和源码身份见 [result_identity.json](result_identity.json)，数据完整性和控制结果见 [summary.json](summary.json)。原报告的 24 个归档文件及 FAC 参考数据保持原 hash；生产源码已改变，旧报告的源码身份不能当作本轮实现身份。重跑数值后数据 hash 如变化，应生成新的身份记录，不能覆盖旧报告来伪造闭合。

本轮未重算 Auger、荧光和 ωK，也未将新辐射宽度与旧 Auger 混成新的产额。Xe/Kr 的 jj 分辨阈值、C 荧光的分辨能量与 ω³ 权重、Fe 输出截断与 CI 空间、HF/Gu 模型差别仍按[逐项归因](../error_attribution_2026_10_02/README.md)保留。用户要求的 jj CI Hamiltonian average 与迭代轨道松弛尚需独立实现、能量泛函与通道验证；本次共享局域场不能替代它。
