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
| Electron projector and correlated gradient | `test_shooting_electron_reference.py`, `test_shooting_correlated_grid.py` |
| Projection between radial representations | `test_shooting_seed.py` |
| Checkpoints and correction replay | `test_shooting_result_persistence.py`, `test_scfresult_save_load.py`, `test_correction_application_contract.py` |
| NIST matching and uncertainty semantics | `test_nist_comparison_workflow.py`, `test_shooting_ionization_comparison.py`, `test_configuration_centroids_nist.py` |
| Source parsers | `test_nist_bibliography.py`, `test_nist_spectrum_notes.py`, `test_nist_reference_annotations.py` |
| Retained archive/CLI behavior | `test_nist_spectra_api.py`, `test_nist_spectra_catalog.py`, `test_full_dhf_nist.py` |

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
