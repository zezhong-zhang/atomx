# Result provenance repair and campaign recheck

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Implemented

- Moved SCF HDF5 persistence out of the DHF module into `atomx.io.results`;
  existing `SCFResult.save/load` entry points delegate there.
- Archive schema 2 retains Z, equation when declared, CalculationSpec,
  provenance, solver diagnostics and correction application/results. Preserving
  correction finalization state prevents a loaded corrected result from being
  finalized a second time.
- Schema 1 remains readable with explicit missing-identity metadata. Unknown
  archive/specification schemas are rejected.
- Native provider conversion prefers the declared equation. Legacy inputs use
  a nonzero-Q heuristic, explicitly identified as such. Basis representation and
  full available calculation specification are retained in provider metadata;
  absent model provenance is explicitly labelled.

Verification: 24 tests passed across `test_scfresult_save_load.py`,
`test_native_provider.py`, and `test_correction_application_contract.py`.
These include physical metadata/correction round trips, legacy archives,
unsupported schemas and the zero-Q Schrödinger case. Persistence currently
covers SCFResult; this is not a unified MCDHF/shooting archive implementation.

## Authoritative campaign observation

Read-only query of `output/nist_spectra/full_dhf_corrected_v3.sqlite` on
2026-09-28:

| Scope | done | basis_limited | failed | pending | running status |
|---|---:|---:|---:|---:|---:|
| Z <= 20 | 7,526 | 17 | 0 | 0 | 0 |
| All stages in ledger | 24,110 | 17 | 30 | 864 | 4 |

These are job statuses, not experimental agreement or independently verified
live-process counts. No queue rows were changed by this work.

The historical `benchmark_first20_shooting.py` calls
`solve_from_shooting_seed`; that helper explicitly produces a **B-spline final
solution**, using shooting only for initial orbitals. It does not prove the
requested native-shooting first-20 benchmark is complete. Its old constraint
limitations in the docstring also predate the newer coupled shooting solver.
A fresh native-shooting benchmark must identify the final engine and archive
resolved Hamiltonian, isotope, corrections, state matching and convergence
separately. Existing B-spline records must retain their actual engine identity.

Next campaign work must resolve the 17 first-20 basis-limited cases or report
supported/unsupported states explicitly, and establish the native-shooting
benchmark before treating this ledger as satisfying the shooting objective.
