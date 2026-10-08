# Testing Atomx

Run tests from the repository root with the development dependencies installed.
For a source checkout:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest --collect-only -q
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_architecture_boundaries.py tests/test_numerical_design_contracts.py
```

Limiting BLAS threads avoids oversubscription when calculations or test processes
run concurrently. The development dependencies include `pytest-timeout`; without
it, pytest reports unknown timeout configuration options.

## Organization

Tests remain grouped by subsystem through their filenames. Shared physical
behavior is tested against package APIs. CLI tests belong to commands that still
exist; removed campaigns do not leave skipped or failing test placeholders.

| Contract | Representative tests |
|---|---|
| Ownership and imports | `test_architecture_boundaries.py`, `test_numerical_design_contracts.py` |
| Public requests and physical identity | `test_architecture_scientific_fidelity.py`, `test_method_request_controls.py` |
| Shooting HF, open shells and constraints | `test_shooting_hf.py`, `test_shooting_guerra.py`, `test_shooting_open_shell_constraints.py` |
| Radial branch identity | `test_dhf_radial_identity.py`, `test_shooting_core_hole_stationarity.py` |
| Dirac radial operator accuracy | `test_shooting_radial_operators.py`: analytic moments, boundary stencils, symmetry and refinement |
| Electron projector and correlated gradient | `test_shooting_electron_reference.py`, `test_shooting_correlated_grid.py` |
| Projection between radial representations | `test_shooting_seed.py` |
| Checkpoints and correction replay | `test_shooting_result_persistence.py`, `test_scfresult_save_load.py`, `test_correction_application_contract.py` |
| NIST matching and uncertainty semantics | `test_nist_comparison_workflow.py`, `test_shooting_ionization_comparison.py` |
| Source parsers | `test_nist_bibliography.py`, `test_nist_spectrum_notes.py`, `test_nist_reference_annotations.py` |

Tests of retired archive importers, campaign commands and HTTP routes are
preserved in [the historical test archive](archive/2026-09/retired_script_tests/README.md),
outside the current test collection.

A source parser test does not establish that a removed importer or HTTP route is
still available. CLI schemas and options follow the retained command versions.

## Scientific evidence

- Compare matched Hamiltonians, state identities and numerical controls.
- Keep independent references distinct from Atomx snapshots and cross-engine checks.
- Preserve native metric, correction ledger and convergence evidence through IO.
- Do not loosen a scientific tolerance to accommodate a file move.
- Keep benchmark experiment setup in its owning test when no reusable production
  service needs it. A physics regression must not depend on a retired CLI.

`slow`, `external_fac`, `snapshot` and `book` markers are defined in
`pyproject.toml`. In particular, `external_fac` uses stored reference data;
regenerating it requires the external code. See [verification](VERIFICATION.md)
for the interpretation and provenance of each gate.

A passing collection checks imports, not numerical correctness. Focused tests
validate the affected contracts; only a complete run establishes full-suite
status. Historical test counts and campaign results live in the
[archive](archive/README.md).

The [2026-10-08 CI/XFEL delivery record](code_delivery_2026_10_08/validation.json)
binds the selected regression run to its source hashes and preserves the initial
failure and stale-column fixture diagnosis. Physical inputs, tolerances and
strict known-gap gates are retained; slow campaigns are outside that run.
The multi-electron PI check compares every signed coordinate with dense Fock
annihilation matrices, including zero amplitudes and interference; a floating
nonzero channel count is not a completeness or convergence gate.

## Small correctness gates and expensive transfer campaigns

`test_slater_scaling.py` uses a complete 91-determinant 4f² space with fixed,
normalized radial functions. It gates antisymmetry, non-diagonal interference,
complete J-sector extraction and parameter recovery. It does not claim free-ion
stationarity or agreement with experiment. The original Pr/Eu/Nd models and
numerical acceptance rules remain in `test_slater_scaling_transfer.py` (slow),
with one shared Pr fit. A full `pytest` still runs this campaign.

```bash
.venv/bin/python -m pytest -q -m 'not slow' --durations=30
.venv/bin/python -m pytest -q tests/test_slater_scaling_transfer.py --durations=10
.venv/bin/python -m pytest -q tests/test_slater_kernel_crosscheck.py tests/test_gauss_legendre_rule.py
```

The six-kernel crosscheck retains scalar/profile accuracy gates and independent
reference anchors. Its refinement tests use smaller pairs of grids with the
same refinement factor and convergence-order bounds. The shared Gauss-Legendre
rule is also checked against exact polynomial moments and read-only cache
ownership. [Measurements and limits](code_review_2026_10_05/TEST_PERFORMANCE.md)
distinguish test-workload reduction from same-input kernel acceleration.

## Commit-time project navigation

Install `.venv/bin/pre-commit install` once per clone. The `project-index` hook
runs `python3 scripts/build_project_index.py --staged --stage` before every
commit. It reads staged blobs, including partially staged files, and stages only
`PROJECT_INDEX.md` and `PROJECT_INDEX.json`; unstaged and untracked scientific
work is excluded. Preserve local index edits before rebuilding. Verify staged
navigation with `--staged --check`; a working-tree preview can legitimately differ.

`tests/test_project_index.py` checks deterministic AST/query output, staged
renames/deletes, partial staging, exclusion of unrelated work, refusal to overwrite
local index edits, invalid-source failure and the installed Git hook in a separate
repository. These are navigation checks; they do not prove physical correctness.
