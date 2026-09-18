# Migration Notes

Changes are grouped by the version that introduced them. The package version
is `atomx.__version__` (`src/atomx/__init__.py`); see `CHANGELOG.md` for the
per-version summary.

## Unreleased

### Package version is single-sourced

`pyproject.toml` no longer carries a literal version — it declares
`dynamic = ["version"]` and hatchling reads `src/atomx/__init__.py`. The two
values had drifted (`0.1.0` in the metadata that `pip`/`uv` reported,
`0.2.0` in `atomx.__version__`); the code value wins. Nothing to change in
downstream code, but `importlib.metadata.version("atomx")` now agrees with
`atomx.__version__`.

### Heavy visualization/notebook packages moved to extras

`pip install atomx` no longer pulls `manim`, `plotly`, `nbformat`, `ipython`
or `ipykernel`. Install the back-end you use:

```bash
pip install "atomx[viz]"          # plotly
pip install "atomx[viz-manim]"    # manim (also needs system pangocairo)
pip install "atomx[notebook]"     # nbformat/ipython/ipykernel
pip install "atomx[all]"
```

`matplotlib` stays a base dependency because `import atomx` imports
`atomx.viz.mpl` unconditionally. Code that relied on a transitive plotly/manim
import should import it explicitly.

### Test markers renamed

`pytest -m fac_parity` no longer selects anything. The marker conflated two
different things and was renamed accordingly:

| old | new | meaning |
|-----|-----|---------|
| `fac_parity` (stored external data) | `external_fac` | asserts atomx against stored external FAC reference data |
| `fac_parity` (self-regression) | `snapshot` | asserts atomx against its own recorded output |

`pytest` also runs with `--strict-markers` now, so a stale selector fails
loudly instead of silently collecting nothing.

### Benchmark scripts take their reference code from the environment

`benchmarks/fac_runner.py`, `bench_local_exchange.py`,
`bench_shooting_vs_radial.py`, `scripts/energies_gs.py`,
`scripts/density_gs.py` and the FAC calibration helper no longer contain
machine-specific absolute paths. Set `FAC_SFAC_PATH` (or `FAC_BIN` /
`FAC_BIN_DIR`), `FAC_DYLD_LIB`, `RADIAL_DHFS_DIR`, `ATOMX_MCDHF_REF_CSV`,
`ATOMX_WF_TXT_DIR` as needed; each script stops with an actionable message
when its reference is missing rather than silently skipping.

## 0.2.0 — QED backend default

`Atom(..., qed=True)` now defaults to `qed_backend="auto"`.

`auto` keeps the historic Mohr/MSS self-energy backend for `Z <= 36` and
uses the FAC/Indelicato-Mohr K/L-shell backend for `Z >= 37`.  This closes
the documented high-Z residual against FAC.

Users who need bit-stability against the old MSS-only output should pass:

```python
Atom(Z=82, config="1s1", qed=True, qed_backend="mss")
```
