# Yb⁺ 矩阵自能源码与复算包

本包包含全部 333 个 Atomx 原生 Python 源文件、11 个受影响测试文件及共享测试 helper、两套冻结闭核输入、完整虚态预检快照、十态/三外部维数结果、混合径向函数与本轮报告。`BUNDLE_MANIFEST.json` 在 ZIP 根目录，逐文件列出 SHA-256 与字节数。生产捕获与最终兼容修复的身份链另见 `final_source_chain.json`。

约 3.1 GB 的十态完整极点矩阵留在原仓库；本 ZIP 提供再生它们所需的完整输入与原生实现。旧专题仅包含复算所需输入及报告，未打包四轮全部归档。因此，`analyze_matrix.py` 的完整旧归档哈希审计须在原仓库执行。`regenerate_from_bundle.py` 在本包独立运行，不依赖那些归档或外部计算软件。

## 环境

使用 Python 3.10，计算依赖版本见 [native_dependencies.json](native_dependencies.json)。原生计算与 PDF 渲染分开；重算谱线不需 Markdown/WeasyPrint。可新建环境后安装本包与记录版本：

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -r docs/yb_matrix_self_energy_2026_10_08/replay_requirements.txt
```

## 重算

在解包的仓库根目录运行；新输出名必须是尚不存在的单一目录名。

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/yb_matrix_self_energy_2026_10_08/regenerate_from_bundle.py \
  --case coarse_l6_e200 --output yb_matrix_reproduction_coarse
```

粗基完整五态与细基完整五态分别选择 `coarse_l6_e200` / `fine_l6_e200`；`--case all` 重算十态。原完整十态耗时约 17.4 分钟，单工作进程峰值 RSS 约 5.64 GiB；这不是对其他机器的资源保证。脚本保持每态 5400 s、16384 MiB、五个工作进程及一线程 BLAS 上限，完整保留 ℓ≤6/200 Ha 的中间虚态，不因预算缩小物理模型。每态实际虚轨道需逐系数匹配预检，然后重放旧标量锚，再求三个外部维数。新 15/30 根逐维数比较原始结果，1e−10 Ha 门。

## 受影响测试

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  tests/test_mbpt_matrix.py tests/test_mbpt_transitions.py tests/test_mbpt2.py \
  tests/test_ci_mbpt_sigma1.py tests/test_mbpt_sigma2.py \
  tests/test_architecture_boundaries.py tests/test_numerical_design_contracts.py \
  tests/test_qed_approximation_coverage.py tests/test_qed_self_energy_table.py \
  tests/test_pair_ladder.py tests/test_ladder_and_sampling.py
```

原仓库 99 项通过；最终物理/架构面板 76 项与 pair/ladder 面板 23 项没有重复计数。包内 helper 提供独立磁态耦合锚，不算入这 99 项选择。25 条预期 QED 覆盖告警表示模型回退，不能作精密 QED 通过。

独立解包验证结果归于 [bundle_validation.json](bundle_validation.json)：99 项测试通过，粗基完整五态的 15 根与原结果最大差为 0 Ha，333 个生产源码和重算所需脚本/输入与最终包一致。最后仅测试文件的空行格式有变，AST 完全相同且兼容测试另重检通过；文档与封存记录在独立验证后完成。细基不重复重算全部积分，其原完整捕获及最终源码保存矩阵重放已通过。五态完整重算耗时约 249 s，原始副本见 `bundle_recomputed/`；其中额外大型极点矩阵也留在原仓库。总体验证和未过物理门归于 [validation.json](validation.json)。矩阵数值残差通过，但四激发能 5% 目标、外部/径向 1 cm⁻¹ 收敛门均未通过。该结果支持转向同模型高阶相关覆盖，未实现全阶 SD、互补空穴 CI、Breit 交叉项或完整相关辐射顶角。
