# Shared physical machinery — review disposition

The 2026-10-05 fixes make conventions explicit and reuse established kernels.
They do not change the retained Coulomb, Breit or electromagnetic operators.

- Grant jj and orbital LS coefficients are `reduced_C_jj` and `reduced_C_LS`.
  They retain their signed phases; neither is interchangeable with the
  GRASP/FAC `reduced_cl_j2` one-body convention. Transitions use `angular_eL`
  and `angular_mL`; fluorescence uses `radial_eL`.
- `angular.clebsch.clebsch_gordan` owns CG evaluation; `angular.mscheme` owns
  fermion creation/annihilation signs. Kappa conversions live in
  `angular.recoupling`; labels and the parser share `config.subshell.L_LETTERS`.
  Callers, benchmarks and tests were migrated together.
- `einstein_A_prefactor(omega, L)` is the rate per squared **fixed magnetic
  state amplitude**. `einstein_A_eL` multiplies it by reduced line strength
  divided by `2J_i+1`; magnetic-state channels multiply it by their explicit
  final-state/photon sum. Independent E1/E2/E3 factors and SymPy magnetic sums
  pin this distinction in `test_radiative_normalization.py`.
- Energy, wavelength and temperature conversions use `constants` (including
  `HC_EV_NM` and `BOLTZMANN_EV`). The hygiene gate detects old conversion
  literals; the small numerical change from truncated wavenumber constants
  is intentional. Recorded historical measurements retain their original values.
- Native results share `RadialStateMixin`; its lazy owned-snapshot semantics
  are retained. Driver adoption is completed with the solver-contract increment.

## CI integrity

The known-wrong legacy CI kernel and its private-only stubs were deleted, along
with tests of those deleted internals. Production exact-CI tests remain.
Missing orbitals fail explicitly; empty operator requests return a `(0,0)`
`LinearOperator` with the same auxiliary interface as a nonempty request.

The orbital metric pass is reused only for byte-identical orbitals **and radial
representation** after a successful audit. The fingerprint includes the
quadrature, basis arrays/layout, Z and ns. In-place changes invalidate it;
failed audits never get cached. One-body blocks also require the identical
workspace, the same geometry content and the same nuclear callable **and its
values on the integration grid**. Mutable callable parameters invalidate reuse.
Delayed
radial evaluators bind their nuclear model rather than inherit a later build's
memo context. Build-scoped global integral caches still require sequential use.
No thread-safety or end-to-end speedup is claimed.

`ConvolCache.release_shared_banks()` names the process-wide teardown honestly:
all caches on the shared grid observe it. It is not per-instance cache clearing.

## Documentation impact and evidence

Updated: angular/radial/rate docstrings and callers, [migration notes](../MIGRATION.md),
[crystal-field conventions](../CRYSTAL_FIELD_SPEC.md). Retained: architecture
ownership/dependency rules and capability descriptions, because shared kernels
stay in the same layers and no operator coverage is added. Historical reports
retain their original formulas; the legacy-CI and metric-cadence status records
are corrected with links to this review. Historical probe scripts target their
recorded revisions; their old imports are not current API examples.

Relevant gates include `test_ci_memo_reuse.py`, `test_radiative_normalization.py`,
`test_angular.py` and `test_fac_ci_operator.py`. Internal CI transcription parity
remains a convention check; independent FAC/analytic gates provide separate evidence.
