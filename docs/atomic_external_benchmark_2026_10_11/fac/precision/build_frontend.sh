#!/bin/sh
# Private read-only dump frontend; local FAC source and libraries are not edited.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH= cd -- "$here/../../../.." && pwd)
fac=/home/zzhang/code/fac
runtime=/data/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu
${CC:-gcc} -std=gnu17 -O3 -fopenmp -pthread \
  -I"$fac" -I"$fac/faclib" -I"$fac/sfac" \
  -o "$here/fac-ci-diagnostic" \
  "$repo/benchmarks/runs/2026-09-30-ci-common-integrals/sfac.c" \
  "$fac/sfac/stoken.c" -L"$fac" -lfac -lmlapack -lmblas \
  "$runtime/libgfortran.so.5.0.0" -lm -lquadmath
