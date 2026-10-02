# 重算统计组态与统一场对比

在仓库根目录、安装数值依赖的 venv 中运行。参考文件来自上一轮实际 FAC 和 XATOM 计算，原始输入及输出仍在 `../systematic_xray_2026_10_02/raw/`；本轮复用参考数值，不重新调用外部程序。以下命令输出到新目录。

```bash
cd ~/code/atomx
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
.venv/bin/python benchmarks/bench_atomic_consistency.py \
  --reference docs/systematic_xray_2026_10_02 \
  --output docs/atomic_consistency_replay \
  --ns 180 --shooting-grid 16000
```

驱动保留 HF 和 Gu 两个模型，每个模型输出四套 PI：残余核场加轨道阈值、初态冻结场加轨道阈值、初态冻结场加冻结组态阈值、初态冻结场加松弛 ΔSCF 阈值。荧光保留三套光子能量：轨道差、冻结组态差、松弛 ΔSCF 差。每次 SCF 必须收敛；失败时停止，不删除样本。

固定源码的复现方式：

```bash
consistency_source=$(mktemp -d /tmp/atomx-consistency-source-XXXXXX)
tar -xzf docs/atomic_consistency_2026_10_02/source_snapshot.tar.gz -C "$consistency_source"
.venv/bin/python "$consistency_source/benchmarks/bench_atomic_consistency.py" \
  --reference "$PWD/docs/systematic_xray_2026_10_02" \
  --output "$PWD/docs/atomic_consistency_replay" \
  --ns 180 --shooting-grid 16000
```

`source_identity.json` 记录源码和参考 JSON 的 SHA256、依赖版本及 Git 基线。源码快照保留计算所需 Python 文件和 benchmark 工具；Git 提交号不能单独表示本轮工作区的源码。

完成十二组计算后，在新目录内复制 `make_report.py`，并把其 `REF` 指向已有参考 JSON，即可从 JSON 重建 CSV 和 PNG/PDF。报告脚本校验 432 个 PI 点与 108 个荧光宽度。九格矩阵可选择新面板：

```bash
.venv/bin/python benchmarks/bench_systematic_fac_matrix.py \
  --consistency docs/atomic_consistency_2026_10_02 --json
```

本轮接口及外部面板的检查命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_configuration_spectroscopy.py tests/test_pi_rr_vs_fac.py \
  tests/test_scattering_models.py tests/test_xray_scattering.py \
  tests/test_jj_coupling.py tests/test_higher_multipoles.py tests/test_rr.py \
  tests/test_dhf_radial_identity.py tests/test_dhf_semiseparable_exchange.py \
  tests/test_shooting_guerra.py
```

`rr_before.csv` 与 `rr_after.csv` 保留刷新前后的 RR 面板；FAC 列未变。历史计时列保留原值，本轮没有把它们作为性能测量。`pre_exchange_fix/`、`pilot.py` 与 `run.log` 只用于诊断；正式面板以十二个顶层元素模型 JSON 和 `run_final.log` 为准。
