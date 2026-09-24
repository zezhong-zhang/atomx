# FAC DW-EII Gate Results

This is a historical EII panel. The measurements below predate the 2026-09-24 continuum matching-derivative correction. Current atomx values are in [results_fac_dw_eii.csv](results_fac_dw_eii.csv); their nine-point RMS difference from the unchanged FAC column is 0.585%. The [continuum FAC and performance report](../docs/CONTINUUM_FAC_PERFORMANCE_2026-09-24.md) records the change and live wave comparison.

Fresh FAC DW-EII gate run for HYP-832, rerun on 2026-07-22 after the
HYP-862 integrity changes were present on PR #41.

## Reproduction

- PR #41 code head present for rerun: `0e2a614ada8bb56fb2284b3603dddd78fb08b185`
- Atomx Python runtime: Python `3.13.10`, NumPy `2.2.6`
- FAC/sfac provider environment: conda `em`
- FAC package: conda `fac 1.1.5`, build `py39h4665d0e_0`, channel `zezhong-zhang`
- Python package view: `PFAC 1.1.5`
- sfac binary: `/home/zzhang/miniconda3/envs/em/bin/sfac`
- sfac binary size: `4676616` bytes
- sfac binary mtime: `2025-02-13 14:27:55.838016271 +0800`
- sfac binary birth time: `2025-09-02 18:16:18.072188186 +0800`

Exact commands, run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib FAC_SFAC_PATH=/home/zzhang/miniconda3/envs/em/bin/sfac \
  python benchmarks/bench_fac_dw_eii.py --sequence he-like

MPLCONFIGDIR=/tmp/matplotlib \
  python benchmarks/scan_eii_shooting_convergence.py
```

The `em` environment supplies the FAC `sfac` binary and PFAC package metadata.
The Atomx benchmark/test process above uses the repository's active Python
runtime (`python`, Python 3.13.10 / NumPy 2.2.6), not the `em` Python runtime.

The HYP-862 commit (`0e2a614`) was already present when the FAC gate and
convergence scan were rerun. The later HYP-832 revision only normalizes CSV line
terminators and updates generator/provenance text; it does not change the
default numerical DW-EII path exercised by the rerun.

The FAC script template is embedded in `benchmarks/bench_fac_dw_eii.py` as `FAC_TEMPLATE`. It uses:

```python
SetCIQkMode('dw')
SetCIPWOptions(1e-3, 36, 4, 36)
SetCIEGrid([...])
SetUsrCIEGrid([...])
CITable(...)
```

The internal and user CI energy grids are both pinned to the same above-threshold energy grid so FAC evaluates the requested points directly instead of interpolating across its default threshold grid.

## FAC Gate

Source CSV: `benchmarks/results_fac_dw_eii.csv`.

| Z | sequence | E/I | atomx sigma (cm^2) | FAC sigma (cm^2) | rel_diff |
|---:|---|---:|---:|---:|---:|
| 10 | he-like | 1.5 | 1.6125084909048453e-20 | 1.5921e-20 | 0.012818598646344733 |
| 10 | he-like | 2.0 | 2.1140716582683506e-20 | 2.0790665145512597e-20 | 0.01683695229185405 |
| 10 | he-like | 5.0 | 2.085617560544958e-20 | 2.02612343158012e-20 | 0.029363526445395288 |
| 26 | he-like | 1.5 | 3.0590971033326612e-22 | 3.0177999999999997e-22 | 0.013684506373073606 |
| 26 | he-like | 2.0 | 4.002018141986261e-22 | 3.9356842411350725e-22 | 0.016854477337861194 |
| 26 | he-like | 5.0 | 4.0652877054354833e-22 | 3.942991388205832e-22 | 0.031016125877287144 |
| 54 | he-like | 1.5 | 1.5899893245514026e-23 | 1.5782999999999998e-23 | 0.007406275455491893 |
| 54 | he-like | 2.0 | 2.1099643743594508e-23 | 2.0868174081046268e-23 | 0.01109199404074718 |
| 54 | he-like | 5.0 | 2.4412411930090708e-23 | 2.3746311762083255e-23 | 0.028050678972009614 |

RMS relative difference over the he-like panel: `0.020298`.

All requested Ne/Fe/Xe `{2I, 5I}` points are below the `0.15` relative-difference gate. The near-threshold proxy point `1.5I` is also below the gate for all three Z values.

## Shooting Mesh Convergence

Fixed point: He-like Fe K-shell, `E/I = 2.0`, threshold `8838.182325349604 eV`.

Source CSV: `benchmarks/results_eii_shooting_convergence.csv`.

### n_grid Scan

`r_max = 50.0 a0`, `kappa_max = 10`, `n_outgoing_energies = 25`.

| n_grid | sigma (cm^2) | relative change from previous |
|---:|---:|---:|
| 8000 | 4.472583646580479e-22 | |
| 12000 | 4.1806726760402595e-22 | 0.06526674370045601 |
| 16000 | 4.0845124381618737e-22 | 0.02300114008673464 |
| 20000 | 4.041537132264323e-22 | 0.010521526509756596 |
| 24000 | 4.018787704148058e-22 | 0.00562890488736376 |

Convergence conclusion: at `n_grid = 20000`, further refinement to `24000` changes sigma by `0.5629%`. This numerical relative-change criterion is the convergence threshold used here.

### r_max Scan

`n_grid = 20000`, `kappa_max = 10`, `n_outgoing_energies = 25`.

| r_max (a0) | sigma (cm^2) | relative change from previous |
|---:|---:|---:|
| 40.0 | 4.0148924976130828e-22 | |
| 45.0 | 4.0272608645251846e-22 | 0.0030806221883786554 |
| 50.0 | 4.041537132264323e-22 | 0.003544907623167229 |

Convergence conclusion: within the valid Fe B-spline box (`50.53782210642609 a0`), increasing `r_max` from `45` to `50 a0` changes sigma by `0.3545%`. The scan does not use Z/E fitted constants. It uses the computed relative change in sigma.

The exploratory `r_max = 70 a0` value is intentionally not included because it exceeds the bound-orbital B-spline box for this atom. The code now raises a clear `ValueError` for such a mesh request instead of extrapolating bound orbitals outside their represented radial domain.

## Threshold Convention Candidates

The exact threshold point has `sigma(threshold) = 0`, so a relative FAC/atomx comparison is ill-defined. Candidate conventions for acceptance discussion:

1. Use `E/I = 1.5` as the formal near-threshold proxy. This matches the current benchmark, avoids arbitrary eV-scale offsets, and samples the steep threshold rise at a stable reduced energy across Z.
2. Add explicit near-threshold rows at `E/I = 1.01` and `1.05`, and report absolute sigma plus relative difference only when both codes exceed a small numerical floor. This preserves true near-threshold information while avoiding division by zero at exactly `I`.

No acceptance-definition change is made in this branch.
