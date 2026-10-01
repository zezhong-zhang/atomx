# 重跑与来源记录

运行目录为 `~/code/atomx`。比较驱动直接导入当前检出的 atomx 源码和 `benchmarks/fac_runner.py`，需要该项目的 Python 环境；外部程序使用本地 XATOM 和 SFAC 二进制。驱动会覆盖本目录中相应阶段的自有结果，外部源目录只读。

```bash
cd ~/code/atomx
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FAC_SFAC_PATH="$HOME/code/fac/bin/sfac"
export ATOMX_COMPARE_FAC_POINTS=3200 ATOMX_COMPARE_FAC_DENSITY=4
comparison_driver=docs/comparison_xraypac_fac_2026_10_02/compare.py
xatom_binary=/tmp/atomx-xraypac-build/build/src/xatom

.venv/bin/python "$comparison_driver" xatom --xatom "$xatom_binary"
.venv/bin/python "$comparison_driver" xatom-coarse --xatom "$xatom_binary"
.venv/bin/python "$comparison_driver" fac
.venv/bin/python "$comparison_driver" fac-coarse
.venv/bin/python "$comparison_driver" angular
.venv/bin/python "$comparison_driver" atomx
.venv/bin/python "$comparison_driver" atomx-matched
.venv/bin/python "$comparison_driver" pi-refine
.venv/bin/python "$comparison_driver" summary --xatom "$xatom_binary"
```

仅重新生成 CSV、网格检查和 PNG/PDF 图，运行最后的 `summary` 阶段即可。仅重新验证 XATOM 径向积分下的 13 个角动量/统计因子通道，运行 `angular` 即可。每个外部输入文件保存在 `raw`，FAC 粗/细网格分别在 `raw/fac-coarse`、`raw/fac`，XATOM 在 `raw/xatom-coarse`、`raw/xatom`。FAC 表中的二进制文件未纳入资料包，文本级数、速率和输入均已保留。

本机初始缺少 Fortran 和 BLAS/LAPACK 运行库，因此本次使用 Ubuntu 26.04/x86_64 的软件包，仅下载并在 `/tmp` 解包，没有安装系统包。实际软件包版本和构建日志保存在 `raw/build`；版本也可从下载日志核对。FAC 的现有 SFAC 可执行文件同样使用这些本地运行库；驱动检测到下列目录时会自动添加其库路径。

下面记录实际构建方式。在已有系统 Fortran、CMake、BLAS/LAPACK 的机器上，可以直接把源目录复制到临时目录后用系统 CMake 编译，无需这些 Ubuntu 专用路径。XATOM 的 CMake 会把 Fortran 模块写入源目录，所以应先复制源文件。

```bash
comparison_toolchain=/tmp/atomx-xraypac-toolchain
mkdir -p "$comparison_toolchain/packages" "$comparison_toolchain/root"
cd "$comparison_toolchain/packages"
apt-get download gfortran-15-x86-64-linux-gnu libgfortran-15-dev \
    libgfortran5 libblas3 libblas-dev liblapack3 liblapack-dev
for comparison_deb in ./*.deb; do
    dpkg-deb -x "$comparison_deb" "$comparison_toolchain/root"
done

# 本次使用已有的 miniconda Python，它有 pip；atomx venv 本身没有 pip。
"$HOME/miniconda3/bin/python" -m pip install \
    --target "$comparison_toolchain/python" cmake==4.4.3

comparison_build=/tmp/atomx-xraypac-build
mkdir -p "$comparison_build"
cp -a "$HOME/code/xraypac/xatom" "$comparison_build/source"
comparison_lib="$comparison_toolchain/root/usr/lib/x86_64-linux-gnu"
comparison_cmake="$comparison_toolchain/python/cmake/data/bin/cmake"
export LD_LIBRARY_PATH="$comparison_lib:$comparison_lib/blas:$comparison_lib/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$comparison_cmake" -S "$comparison_build/source" -B "$comparison_build/build" \
    -DCMAKE_Fortran_COMPILER="$comparison_toolchain/root/usr/bin/x86_64-linux-gnu-gfortran-15" \
    -DCMAKE_Fortran_FLAGS="-B$comparison_toolchain/root/usr/lib/gcc/x86_64-linux-gnu/15/ -B/usr/lib/gcc/x86_64-linux-gnu/15/" \
    -DBLAS_LIBRARIES="$comparison_lib/blas/libblas.so" \
    -DLAPACK_LIBRARIES="$comparison_lib/lapack/liblapack.so"
"$comparison_cmake" --build "$comparison_build/build" --target xatom -j6
```

复制命令应在 `source` 尚不存在的新构建目录中执行，以免产生 `source/xatom` 的嵌套目录。Ubuntu 软件包名/版本具有平台依赖性；`metadata.json` 的源码、二进制哈希是本次结果的身份记录。XATOM 源码没有 Git 元数据，FAC 使用本地已编译二进制，没有重建或修改用户的 FAC 工作树。

光电离使用中性原子；Auger 使用总电荷 +1 的单 1s 空穴。atomx 核模型为点核，FAC 保持自身默认核设置；没有高 Z 核尺寸敏感性测试。FAC 关闭 Breit、SE、VP、MS，并不包括跨组态 CI；XATOM 是非相对论 HFS。atomx 的 Auger 比较实现是限定闭合价层与单 1s 空穴的独立磁量子数求和；其适用范围和生产 API 的区别见 README。Gu 控制借用 FAC 阈值/平均动能，所以不是独立预测这些量。

单位和命令语义的参考是 [XATOM 官方短手册](https://www.desy.de/~xraypac/nut_xatom.html)、本地 `~/code/fac/doc/manual.tex` 的 `TotalPICross`/`AITable`，以及 XATOM `src/decay.f90`、`src/unbound.f`。
