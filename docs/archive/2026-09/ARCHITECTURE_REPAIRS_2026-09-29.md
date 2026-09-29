# Architecture repairs — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

This records implemented changes following the full codebase review. It is not
a claim that the entire target architecture has been migrated.

The subsequent [numerical boundary migration](NUMERICAL_BOUNDARY_MIGRATION_2026-09-29.md)
completes the shooting split and supersedes the temporary module locations and
compatibility exports described below. Shooting drivers and electron-space
construction now live in numerics; physical field/energy operators live in
physics; shared result data lives in `results.py`.

## Scientific request fidelity

- Transition properties no longer construct a hidden pure-Coulomb local-exchange
  atom. Existing MCDHF states use their own tensor machinery by default.
- Determinant CI requires an explicitly supplied, solved `ci_reference`. It
  validates nuclear charge and the closed-shell configuration, records the
  reference specification, and identifies its CI Hamiltonian as Dirac–Coulomb.
  Reference corrections are not represented as included in that Hamiltonian.
- Unsupported determinant-CI requests raise by default. `fallback='uta'` is an
  explicit opt-in; each returned fallback record carries the requested and
  actual resolution and reason. Invalid input and unexpected failures propagate.
- The old configuration-only reference cache and broad exception suppression
  were removed with the hidden solve.

Public usage:

```python
lines = atom.transitions(line_resolution="ci", ci_reference=solved_reference)
# Explicitly permit statistical output when determinant CI is unsupported:
lines = atom.transitions(line_resolution="ci", fallback="uta")
```

## Method identity and controls

- `methods/breit_pauli.py` is canonical; `relativistic_hf.py` is an import alias.
- `methods/hartree_fock_slater.py` exposes `HartreeFockSlater`. The historical
  `HartreeFock` import aliases it. The theory ladder uses the accurate class name.
- The wrappers forward grid size, box size, nuclear model and screening.
  Direct calls accept an inner radius; unexpected keyword options are rejected.
  The method object's nuclear choice controls the method-based API.
- Unsupported intermediate coupling raises explicitly.
- Breit–Pauli returns the complete radial payload of its base calculation;
  subshells, native quadrature and spinors are preserved. Its result identifies
  the perturbative method and hydrogenic-shift limitation.

These changes do not implement nonlocal NR Hartree–Fock or a complete
many-electron Breit–Pauli Hamiltonian. The existing hydrogenic shifts evaluated
at nuclear Z remain explicitly labeled.

## Shared contracts and numerical ownership

- `config/subshell.py` owns `Subshell`, with an alias in `methods/dirac_fock.py`.
  Configuration, correction and persistence consumers use the shared owner.
- The radial Green preconditioner now lives in
  `numerics/shooting/preconditioner.py`.
- Constrained grid optimization now lives in
  `numerics/shooting/orbital_optimization.py`. Methods supply the functional and
  state-space choices. Existing method paths retain compatibility exports.
- The dependency test resolves absolute, relative and function-local imports
  over all numerical modules. Seven named historical exceptions remain;
  new upward dependencies fail the gate, and obsolete exceptions must be removed.

## Reference-data ownership

`reference_data/nist.py` owns text parsing, source-record types, configuration
normalization and unit conversions. It takes caller-supplied text and has no
archive path or network policy. `benchmarks/nist_asd.py` retains its panel loader
and CLI and re-exports parser names. Ingestion and viewer scripts use the package
parser directly. This is the first extraction; campaign execution, matching,
database queries and HTTP handlers still require migration.

## Follow-up repairs to the main findings

### Numerical and physical ownership (F5/F7)

- `numerics/shooting/scf.py` owns the nonlocal radial iteration: root solves,
  normalization, damping, orthogonalization updates and convergence checks.
  `ShootingHFSolver` supplies whole-array field, energy and stationarity
  callbacks and constructs the physical result. Both coupled and historical
  Schmidt modes retain their previous formulas and stopping criteria.
- `potentials/radial_nucleus.py` constructs the shared native nuclear/environment
  field. Correlated shooting no longer instantiates an unrelated HF solver to
  obtain this field. The original Fermi/SLAG6 convention is preserved.
- `potentials/static_field.py` owns static Hartree/local-exchange field
  construction. Its old numerical path is a compatibility export.
- `state_protocols.py` owns the coefficient-facing result protocols and legacy
  accessors. Native grid access uses the shared radial workspace, including
  shooting quadrature. The scattering path re-exports the same objects.
- Generic workspace evaluation moved from corrections to
  `numerics/radial_workspace.py`; internal consumers use the numerical owner.
- PNC/EDM result conversion now requires an explicitly solved, converged result.
  It never starts a calculation while adapting an argument. Their B-spline
  capability limits remain explicit.
- The whole-numerics gate now has **six**, rather than seven, named upward-import
  exceptions. The static-field path no longer imports scattering.

### Operator inclusion (F8)

Results record `operator_stages`. MCDHF distinguishes a Breit interaction in
its CI Hamiltonian from post-CI scalar and expectation-value corrections.
Post-SCF finalization checks these records before evaluating a backend and
rejects an already included operator, including a zero-valued contribution.
It also rejects adding two self-energy models in one stack. Read-only
`CorrectionStack.apply` remains available for diagnostic reevaluation.
SCF persistence round-trips these stages. Historical missing records are not
inferred from requested flags or energy magnitudes.

### Packaged comparison and imports (F9/F10/F11)

- `reference_data/matching.py` owns configuration expansion, unique-level
  classification, air/vacuum conversion, J weights and complete-manifold
  centroid evaluation.
- `workflows/nist_comparison.py` owns the read-only native campaign comparison.
  Its CLI is now a thin wrapper; provenance hashes include both service modules.
  Existing script imports remain compatible.
- Top-level public APIs and visualization modules load on demand. The public
  `compare` callable remains stable. Matplotlib remains a base installation
  dependency for compatibility, but importing or solving an atom does not
  import a renderer.
- The R-matrix entry-point documentation now reflects its implemented
  multichannel dispatch. The provider ADR distinguishes its historical proposal
  status from implemented interfaces. [Capabilities and limits](../../ARCHITECTURE_CAPABILITIES.md)
  explicitly retain the unsupported methods and finite-space limitations.

### Follow-up verification

- Applied-tree scientific API, state boundaries, native fields, persistence,
  correction inclusion, correlated shooting and matching: **75 passed**.
- Nonlocal SCF extraction: **22 passed**, including open-shell constraints,
  core-hole radial identity, weak exchange and warm starts. Independent runs
  against the preserved original source give exactly identical He/Be legacy
  energies, orbital energies, convergence flags and iteration counts.
- Staged NIST matching/generation checks: **62 passed**.
- Staged provider and scaffold contracts: **170 passed**.
- These suites overlap with earlier checks; their counts must not be added
  together as a unique full-suite total. The full suite was not run.
- An installed wheel was tested outside the checkout with plotting and external
  provider imports blocked. It solved point-nucleus hydrogen and ran the native
  campaign comparison using only explicit database paths.
- All scientific fields in the existing comparison were reproduced exactly:
  35,802 observed-line records, centroid records, rejected states, summary and
  historical campaign identity. Timestamps and CLI artifact hashes are separate.
- Undefined-name checks, changed-file AST parsing and `git diff --check` passed.
  This is structural verification, not a new claim of improved NIST accuracy.

### Preservation of the running recovery

The changes were prepared and tested separately while the historical recovery
continued. Before applying them, its original source was copied to
`/tmp/atomx-recovery-original-source-20260929` and all **260 recorded source
hashes** were verified. The database checkpoint held 70 successful and 148
failed attempts; unfinished attempts remain pending. Changing the working source
activates the original source guard before further database commits. A launcher
waits for the old worker pool to drain, then resumes the pending attempts from
the verified original source, preserving the campaign identity. Those historical
attempts must not be presented as predictions from this newly refactored tree.

## Remaining work

1. Split B-spline SCF/assembly from its physical functional and extract
   electron-space construction from the remaining mixed MCDHF module. Both
   local and nonlocal shooting iteration now live in numerics.
2. Centralize complete request resolution and capability validation across all
   public and direct solver entry points.
3. Finalize shared result lifecycle and support complete native shooting/MCDHF
   persistence; remove downstream private-layout coupling.
4. Complete application services for ingestion, campaign execution and web
   presentation. Native matching/comparison and its installed-wheel check are
   complete; other script workflows still need migration.
5. Extend explicit operator-stage records to remaining historical producers
   and remove remaining compatibility imports after consumers migrate.

No new campaign predictions or performance gains are claimed for these
structural changes. Historical calculation identities remain attached to their
original source revisions.

## Earlier verification batch

- Scientific API, CI, persistence/provider, numerical ownership, correlated-grid
  and theory-ladder checks: **95 passed**.
- NIST archive/parser, levels, API, optional fields and generation checks:
  **71 passed, 2 expected failures**.
- After the final result-label change, method fidelity and Breit–Pauli checks:
  **18 passed** (a rerun of tests included above, not additional coverage).
- Source/script AST parsing and `git diff --check` passed.

The environment reports missing pytest-timeout configuration support. These
checks completed normally. The full suite and installed-wheel smoke test were not run in this earlier
batch; the later installed-wheel verification is recorded above. The two
existing expected failures remain visible.

## Local shooting iteration follow-up

- `numerics/shooting/local_scf.py` now owns the compiled local SCF loop:
  potential mixing, spline updates, Dirac/Schrödinger bound solves,
  normalization and stopping checks. The physical method supplies a compiled
  whole-array field callback; nuclear-field/seed construction and total-energy
  assembly remain physical responsibilities in `methods/shooting_local.py`.
- Preparation, iteration and energy evaluation remain compiled kernels. The
  orchestration wrapper is Python, avoiding an uncacheable Numba wrapper around
  a first-class compiled callback. No Python callback occurs per radial point.
- Unknown equation names now raise instead of silently selecting Schrödinger.
  Invalid iteration counts and nonpositive/nonfinite tolerances fail before
  field evaluation. Removed an obsolete comment claiming spinors were absent.
- Fresh checks: **59 passed** across architecture/scientific fidelity, numerical
  boundaries, field injection, Breit–Pauli and local DFS reference tests.
- A separate recheck of persistence, native provider, correction application
  and native NIST comparison passed **33 tests**: **92 targeted tests** across
  these two disjoint runs. Pytest reported two missing-timeout-plugin warnings
  per run; the full suite was not run. AST parsing and `git diff --check` passed.
- Independent before/after runs compare **24 arrays exactly**, including total
  and orbital energies, convergence, iteration counts, grid, potential and
  spinors for H, He, Schrödinger Ne, and finite-nucleus screened Dirac Ne.
  The historical Schrödinger Ne non-convergence is preserved, not reclassified.
  Evidence: `output/architecture_review_2026-09-29/local_scf_equivalence.json`.
- This extraction establishes numerical equivalence for those controls; it does
  not establish a speedup, repair the historical Ne case, or validate the full
  periodic-table campaign. Compatibility aliases remain explicit.

## Additional request and scattering boundary repairs

- `Atom.solve(DiracFock(...))` now forwards the explicit basis size, grid mode
  and screening potential. Previously these could be ignored even though the
  constructor accepted them. The direct method now preserves an explicit box
  on the non-physics-grid route.
- `MCDHF.solve` accepts and forwards named box/grid/screening controls; both
  method wrappers reject unknown keywords and inapplicable grid-shape controls.
  The Atom MCDHF method path forwards its numerical and environment controls.
  This closes these dispatch defects; complete request centralization remains
  unfinished.
- Replacing the B-spline basis also updates its resolved outer radius and
  kinetic-balance metadata. Per-J-block MCDHF physics grids now receive the
  same requested box and grid-shape parameters as the shared-orbital pass.
- `scattering/photoionization/direct.py` owns E1 channel selection and cross
  section assembly. The numerical continuum module retains eigenproblems and
  radial integrals, with a small historical cross-section compatibility entry.
  Radiative recombination consumes the canonical scattering service. The
  dependency gate permits only that compatibility edge at the old location.
- Integer photon-energy inputs are converted to floating point before cross
  section accumulation, preventing silent integer truncation.

Verification: 21 method/control and scientific-fidelity tests passed; 81
boundary, continuum, scattering, persistence, provider and correction tests
passed. After updating basis metadata and per-block grid forwarding, six
request-control tests and one existing per-block MCDHF test passed. These
runs overlap and are not a full-suite count. The screening control checks an
independent constant-potential energy shift; continuum checks include the
existing hydrogen Stobbe comparison. No new NIST accuracy or speedup is claimed.
