# Native shooting checkpoints

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Shooting results now support `result.save(path)` and
`ShootingResult.load(path)`. Serialization belongs to `atomx.io.results`;
the result methods delegate to that I/O boundary.

The versioned HDF5 format retains the native radius and grid Jacobian,
large/small spinors in subshell order, optional local potential, orbital and
total energies, exact jj configurations and weights, resolved calculation
specification, solver diagnostics, and any existing correction ledger.
No interpolation or B-spline projection is involved. Writes use an atomic
replacement after shape, finite-value and orbital-identity checks. Invalid
arrays cannot replace an existing valid checkpoint.

`benchmark_native_shooting_nist.solve_stage` accepts `checkpoint_path` and
saves after converged orbital refinement and before correction finalization.
Its returned record contains the checkpoint path, SHA-256 and stage.
The full campaign runner accepts `--checkpoint-dir`; each campaign database
gets a separate directory namespace and each job/grid pair its own file.
Checkpoint storage remains optional because all-table spinor storage is large.

## Verification

An open-shell carbon control verifies exact arrays, native quadrature weights,
jj statistical weights and calculation specification after reload. Reapplying
Breit, Mohr self-energy, vacuum polarization and recoil gives exactly the same
per-correction shifts. A loaded state also converges as a warm SCF seed.
A finalized result retains its correction application ledger after reload.
The persistence, campaign, existing SCF persistence and architecture suite
passes 22 tests.

These checkpoints support re-evaluation of post-SCF corrections on the same
orbitals. Changes to an operator included in the orbital equations still
require a new self-consistent solve. Historical campaign JSON/SQLite records
lack the radial arrays and per-orbital self-energy decomposition: their missing
information cannot be reconstructed by rescaling a total self-energy shift.
They are not silently migrated or relabelled as corrected calculations.

## Mg I production-path control

A separate frozen-source control recalculated `3s2` and `3s4s` at 3,200
and 6,400 grid points, using the updated Mohr screening reference and the
existing Breit, vacuum-polarization and operator recoil corrections. Each
solve was refined with the implicit nonlocal solver at tolerance 1e-13.
All four states converged and passed the undamped orbital-update gate.

| Excitation configuration centroid | cm⁻¹ |
|---|---:|
| NIST evaluated centroid | 41773.885500 |
| Historical corrected Atomx | 34060.604754 |
| Updated corrected shooting, 6400 points | 34060.673851 |
| Updated bare Dirac HF, 6400 points | 34081.189362 |
| Bare Cowan HFR, CORRF=0 | 34638.581902 |

The doubled-grid change is −7.49e-8 cm⁻¹. Updated self-energy normalization
changes the historical prediction by only +0.069097 cm⁻¹. Breit, QED and
recoil together change the new excitation by −20.515510 cm⁻¹; these do not
explain the roughly 7,713 cm⁻¹ discrepancy with NIST or the 557 cm⁻¹ difference
between bare Dirac HF and bare Cowan HFR. Grid convergence alone does not
establish the correctness of the angular functional or model matching.

Replaying all corrections from a loaded checkpoint reproduces every stored
correction shift exactly, taking 0.015–0.030 seconds per state in this warmed
process. The original solves including corrections took 4.47–22.33 seconds;
these timings include differing compilation/cache costs and are not a general
solver speedup estimate. The four checkpoint files occupy 1,985,069 bytes.

Artifacts: `output/nist_spectra/normalization_diagnostic/mg_checkpoint_control/`
contains the driver, frozen-source hashes, full records, HDF5 checkpoints and
centroid comparison. These are separate diagnostics; the historical campaign
and its lower/upper endpoint correction models have not been overwritten.


## Independent B-spline control

The initial 1e-12 B-spline request reached the 300-iteration limit with a
residual of 7.47e-10. Its output recorder also exposed an unrelated persistence
bug: structured energy-component metadata could not be stored as an HDF5
attribute. The I/O layer now preserves complete components in JSON, retaining
legacy scalar attributes and loading historical files without the new dataset.
A structured nuclear-profile roundtrip regression covers the failure.

The repeated, explicitly requested 1e-9 control converges for both endpoints,
with residuals 4.55e-10 and 6.19e-10. Its bare excitation is
34081.189361528 cm⁻¹, within 6.87e-8 cm⁻¹ of the 6400-point shooting result.
Individual total energies differ by less than 1.5e-7 cm⁻¹. This supports
agreement of the two discretizations for this matched Dirac-HF model; it is
not evidence of experimental accuracy or of converged many-electron correlation.
The raw records, B-spline checkpoints, driver and separate frozen-source hashes
are stored alongside the shooting control.
