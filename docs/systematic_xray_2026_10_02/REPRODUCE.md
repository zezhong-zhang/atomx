# 复现

在仓库根目录使用已安装数值依赖的 venv；原始外部输入/输出保存在 `raw/`。以下命令写入独立的新报告目录。XATOM 与 SFAC 路径需设置为本机已构建版本。Linux 的本轮 Fortran/BLAS/LAPACK 运行库在 `/tmp/atomx-xraypac-toolchain`；驱动在该目录存在时自动补充 LD_LIBRARY_PATH。外部二进制需重新构建时参照 [已有构建记录](../comparison_xraypac_fac_2026_10_02/REPRODUCE.md)。

```bash
cd ~/code/atomx
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FAC_SFAC_PATH="$HOME/code/fac/bin/sfac"
export XATOM_BIN=/tmp/atomx-xraypac-build/build/src/xatom
systematic_source=$(mktemp -d /tmp/atomx-systematic-source-XXXXXX)
tar -xzf docs/systematic_xray_2026_10_02/source_snapshot.tar.gz -C "$systematic_source"
export ATOMX_SYSTEMATIC_SOURCE="$systematic_source"
export ATOMX_SYSTEMATIC_BENCHMARKS="$systematic_source/benchmarks"
systematic_output=docs/systematic_xray_replay
for systematic_stage in references atomx scatter radiative-controls summary; do
  .venv/bin/python docs/systematic_xray_2026_10_02/bench_systematic_xray.py \
    "$systematic_stage" --output "$systematic_output" || break
done
```

原报告目录的脚本副本固定本轮驱动；仓库 benchmarks 下的版本用于继续开发。缓存仅在参考命令/脚本相同且存在完成标记时复用；更换外部二进制请使用新输出目录。配置、壳层、源对、能量和单位按 JSON/CSV 保存，失败不会自动换成近似或删去不匹配点。

径向收敛抽查使用同一参考数据：

```bash
mkdir -p "$systematic_output/refinement"
cp "$systematic_output/references.json" "$systematic_output/refinement/references.json"
.venv/bin/python docs/systematic_xray_2026_10_02/bench_systematic_xray.py atomx \
  --elements C Fe Xe --ns 240 --output "$systematic_output/refinement"
```

补充 Ne/Fe 锚点依赖 `858cd29` 的测试及公共接口；源码/fixture 身份另保存在 metadata 中。使用该提交及本轮散射新增文件时可重跑：

```bash
.venv/bin/python docs/systematic_xray_2026_10_02/check_ne_ls.py
.venv/bin/python docs/systematic_xray_2026_10_02/check_fe_jj.py
.venv/bin/pytest tests/test_auger_fac_panel.py tests/test_auger_kll_vs_fac.py \
  tests/test_auger_ne_kll_xatom.py tests/test_fac_dr_radiative.py \
  tests/test_pi_rr_vs_fac.py tests/test_xray_scattering.py -q -o addopts= --tb=short
.venv/bin/python benchmarks/bench_systematic_fac_matrix.py
.venv/bin/python docs/systematic_xray_2026_10_02/make_figures.py
```

`make_figures.py` 从当前报告目录的 CSV 生成 PNG/PDF；查看新输出时把脚本复制到新目录运行。FAC 波函数控制是独立磁子态投影计算，不能换成调用 atomx 散射函数来生成“参考”。
