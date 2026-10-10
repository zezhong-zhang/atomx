# Testing Atomx

The default suite targets about **three minutes** and contains a reviewed set of
essential tests listed in `pyproject.toml` under `testpaths`. Plain `pytest -q`
and GitHub Actions use the same selection. An explicit path overrides that
selection, so run affected subsystem tests as well when changing their code.
Do not use `pytest tests` as the routine command: it collects the extended suite.

Use the existing `.venv`. For a fresh clone:

```bash
uv sync --locked --extra dev --python 3.10
.venv/bin/pre-commit install
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
.venv/bin/python -m pytest -q --durations=15
```

The essential suite does not require optional rendering backends or their Cairo
and Pango system libraries. Development dependencies include the units/PySCF
extras and `pytest-timeout`; the lock remains authoritative. Serial BLAS,
OpenMP and Numba avoid oversubscription and reduce scheduling variability.

## GitHub Actions and local equivalents

| Workflow | Trigger | Gates |
|---|---|---|
| [Test](../.github/workflows/test.yml) | Main pushes and pull requests | Essential suite on Python 3.10 and 3.13, including two fresh-process determinism repeats |
| [Lint](../.github/workflows/lint.yml) | Main pushes and pull requests | Ruff 0.15.6, repository documentation links and project-index freshness |

The former nightly slow/book/bounds/full-determinism workflow was removed on
2026-10-10 at the user's request. The three-minute goal is a turnaround target,
not a scientific tolerance or a hardware-independent timing guarantee; each CI
job has a five-minute safety timeout including environment setup.

```bash
# Focused checks explicitly select files or nodes outside the default list.
.venv/bin/python -m pytest -q tests/test_sternheimer_edm.py -m 'not slow'
.venv/bin/python -m pytest -q tests/test_numerical_bounds.py
.venv/bin/python -m ruff check --no-cache src tests scripts benchmarks
.venv/bin/python scripts/check_documentation.py
.venv/bin/python scripts/build_project_index.py --check
```

The explicit bounds diagnostic uses the same C/Ne/Be inputs with
`NUMBA_BOUNDSCHECK=1`. It forces a fresh compilation and is outside routine CI.
The Cf addition and four/twelve-repeat campaigns have been removed. Two fresh
processes still compare energies, every orbital energy and signed wave bytes.

The link check covers root and top-level documentation entry points and requires
tracked repository targets. It excludes external URLs, heading anchors, fenced
examples and archived report bodies. It cannot judge scientific claims; review
the [owning guide](DEVELOPMENT.md#code-to-documentation-entry-points) with each change.
Historical selected-panel counts live in [validation history](VALIDATION_HISTORY.md).

## Essential coverage and removed campaigns

| Essential contract | Selected coverage |
|---|---|
| Public inputs, units, method ownership | Input surface, configuration/method validation, constants, units and architecture boundaries |
| Radial mathematics | Native metric, B-spline identities, analytic Dirac energies, shooting operators, Gauss moments and six-kernel crosschecks |
| Signed CI and eigensolvers | Analytic Slater–Condon, double-excitation signs, complete traces, dense closure, Davidson, radial snapshots and Hermitian clusters |
| Independent physical anchors | Full signed GRASP open-shell Breit blocks and stored FAC CI operator matrices |
| Public live behavior and corrections | Canonical Atom energy panel, correction ledgers/stack and native checkpoint round trips |
| Rates and dynamics | Tracked native Ne table, complete signed multi-electron CI channels and analytic XFEL kinetics |
| Reproducibility and maintenance | Two-process C/Ne/Be comparison, NIST workflow, documentation links and staged project-index behavior |

Selection is explicit and reviewable; no runtime-based skipping, relaxed
thresholds or truncated production models are used. Extended subsystem tests
remain available by explicit path, but their results are not implied by a green
essential run. Add or replace a default test only when its contract earns the
runtime cost; inspect `--durations` and update this guide in the same change.

The multi-minute Pr/Eu/Nd transfer, ns=256/320 DHF bank-comparison and Fe XVII
LMM campaigns were removed from `tests/`. Their unchanged source and SHA-256
manifest are retained in the [historical campaign archive](archive/2026-10/long_running_tests/README.md).
The small complete 91-determinant 4f² Slater gates remain essential; they check
antisymmetry, signed interference, J sectors and parameter recovery, without
claiming free-ion stationarity or experimental agreement. Earlier
[measurements](code_review_2026_10_05/TEST_PERFORMANCE.md) describe the original
workloads and do not measure this new suite. Current measurements and source
identity are in the [CI repair record](ci_repair_2026_10_10/README.md).

## Fixtures and numerical regressions

Tests must read fixtures from tracked paths. `tests/data/ne_native_1000eV.json`
is the unchanged native 134-configuration table, SHA-256
`17d9126593cda0a18a7e47f1d62833b125e57d5110e603eea351402765c008ae`;
it differs from the older `ne_1keV_rate_table.json` model. Its recorded origin
and [repair evidence](ci_repair_2026_10_10/README.md) identify the producer.

`test_ci_dense_orthogonality.py` replays a native clustered CI block at three
energy zeros with the original closure and residual gates. Complete dense CI
uses LAPACK `evd`; partial dense spectra retain `evr`. The nitrogen self snapshot
includes the independently anchored open-shell Breit phase already present in
`233fec1`; it is not an experimental energy reference. The explicitly selected
Sternheimer test retains the Na HF model/tolerance with a 100-iteration budget;
the 40-iteration control failed before convergence at 46. It is outside the
default selection because its extended scaling/heavy-atom work is unnecessary
for every commit.

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

The multi-electron PI check compares every signed coordinate with dense Fock
annihilation matrices, including zero amplitudes and interference; a floating
nonzero channel count is not a completeness or convergence gate.

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
