# 新版比较重跑方法

`snapshot.json` 记录采样时的工作树及逐文件哈希。源码和选定测试后来提交为 227982a；运行结束全部 Python 源码与快照一致。快照归档含 src、benchmark Python 工具、项目配置和此次使用的测试/fixture。比较脚本优先使用 `ATOMX_COMPARE_SOURCE_ROOT`；未设置时会使用当前仓库源码，因此重跑历史结果时应先解包并设置该变量。

在 `~/code/atomx` 使用现有 venv。下列命令覆盖本报告目录内的自有运行结果，保留上一份 `comparison_xraypac_fac_2026_10_02` 基线目录。

```bash
cd ~/code/atomx
comparison_report=docs/comparison_xraypac_fac_rerun_2026_10_02_113532
comparison_source=$(mktemp -d /tmp/atomx-recompare-source-XXXXXX)
tar -xzf "$comparison_report/source_snapshot.tar.gz" -C "$comparison_source"
export ATOMX_COMPARE_SOURCE_ROOT="$comparison_source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FAC_SFAC_PATH="$HOME/code/fac/bin/sfac"
export ATOMX_COMPARE_FAC_POINTS=3200 ATOMX_COMPARE_FAC_DENSITY=4

for comparison_stage in xatom xatom-coarse fac fac-coarse angular \
    atomx atomx-matched pi-refine summary; do
    .venv/bin/python "$comparison_report/compare.py" "$comparison_stage" || break
done
.venv/bin/python "$comparison_report/public_auger.py"
.venv/bin/python "$comparison_report/fe_panel.py"
.venv/bin/python "$comparison_report/summarise_public.py"
```

独立驱动先生成参考及模型控制结果，`public_auger.py` 再计算公开 API。`fe_panel.py` 使用本报告保存的 Fe FAC fixture，不调用新鲜 Fe FAC。`summarise_public.py` 生成公开路径 CSV、径向积分 CSV、Fe CSV、与上次结果的逐数值比较以及 PNG/PDF 图。旧公开 API 的完整原始模块在 `baseline_coarse_api.py`，来自 08b4368，保持旧版本原样。

XATOM 默认可执行文件为 `/tmp/atomx-xraypac-build/build/src/xatom`，可用 `compare.py --xatom /path/to/xatom` 指定。FAC 通过 `FAC_SFAC_PATH` 配置。原本地 Fortran/BLAS/LAPACK 工具链仍在 `/tmp/atomx-xraypac-toolchain`，驱动在该路径存在时自动补充运行库路径。若需要重建，沿用 [上次构建记录](../comparison_xraypac_fac_2026_10_02/REPRODUCE.md)。外部源码和参考二进制的身份记录在 `metadata.json`；每个新鲜外部输入和输出在 `raw`。

针对性测试可使用快照中的文件，保证与此次计算版本一致：

```bash
PYTHONPATH="$comparison_source/src" .venv/bin/pytest \
    "$comparison_source/tests/test_auger_ne_kll_xatom.py" \
    "$comparison_source/tests/test_auger_fac_panel.py" \
    "$comparison_source/tests/test_auger_kll_vs_fac.py" \
    -q -o addopts= --tb=short
PYTHONPATH="$comparison_source/src" .venv/bin/pytest \
    "$comparison_source/tests/test_jj_level_auger.py" \
    -q -o addopts= --tb=short
```

第一组本次有 1 个硬失败、4 个通过、2 个预期失败，退出码为 1；第二组 3 个通过、1 个预期失败。README 解释了失败条件，没有放宽或跳过硬门限。各公开 Ne/Ar 速率都记录了电子动能来源；Gu/阈值控制、公开 LS 组态率和 FAC 逐能级率具有不同条件，应按各文件说明比较。
