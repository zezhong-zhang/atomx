# Physics and numerical remediation — completed

All twelve reproduced findings in the [baseline review](../archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-23.md)
have been addressed. Independent identities, broad regression coverage and
measured performance support the changes. The retained model approximations
and validation boundaries are explicit in [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md).
This is not a certification of every atomic model in the repository.

## The twelve findings

| Finding | Implemented correction |
|---|---|
| 1. Nonorthogonal MCDHF orbitals and false convergence | Metric-constrained Dirac solves, occupied rotations, orthonormal retraction, CI overlap validation and honest stationarity diagnostics. Invalid DIIS trials are rejected. |
| 2. Incorrect CI transition amplitudes | Exact CSF one-body tensors in the Hamiltonian's phase convention; actual CI amplitudes; overlap cofactors for independent orbital sets. |
| 3. Inverted R-matrix boundary matching | Correct equation for K=tan(delta), energy-flux normalization and converged Coulomb reference functions. |
| 4. Wrong collision-strength weights | Spin/angular averaging and reciprocal channel normalization. |
| 5. Wrong CRM state directions/weights | Actual state energies, upper-to-lower spontaneous emission and physical degeneracies. |
| 6. Spontaneous/stimulated RR conflation | Stable Maxwellian Milne integral; radiation-field-dependent stimulated recombination is explicit. |
| 7. Ignored radial controls | Box and knot controls reach both scalar and list-configuration MCDHF paths. |
| 8. Shooting PI/RR statistical factor | Initial magnetic-sublevel average is consistent with the B-spline convention. |
| 9. Artificial physical occupations | Solver floors do not enter reported populations; radial density includes off-diagonal one-body coherence. |
| 10. Repeated continuum diagonalization | Workspace-owned bounded eigensystem cache with content-based invalidation, including in-place mutation. |
| 11. Late CI caps and excessive allocation | Dimension preflight, lazy fixed-M sectors, connected determinant pairs, sparse coefficient tables and bounded caches. |
| 12. Ignored iterative-solver/state requests | Requested solvers run in every symmetry block, with projected sparse operators and global selection of requested low states. |

## Additional consistency repairs

B-spline MCDHF now optimizes the full correlated Dirac–Coulomb functional,
including configuration interference, in a fixed nuclear positive-energy
subspace. Its analytic radial derivative uses the same cell integrals as CI.
Screened Fock preconditioning, transported limited-memory updates, line-search
CI reuse and banded pair-potential reuse reduce its cost. The approximate
initializer is capped at three outer passes; final convergence belongs to the
correlated functional. Correlated iterations have a separate budget of eight
steps per requested outer iteration.

Mixed P/Q exchange contractions now use both spline orders; PQ and QP are
mutual transposes. One-electron spaces avoid artificial screening and retain
physical nuclear eigenstate labels. Corrected energies, CI columns, occupations
and QED shifts stay aligned when levels reorder.

Residual-core continuum construction now integrates density on the bound
solver's native mesh, evaluates the nuclear term separately and removes the
captured electron's screening once. The shooting-result path no longer silently
returns a bare nucleus. Rydberg RR uses a common local bound/continuum potential,
Pauli vacancies and physical positive-kappa roots. The shared virtual-basis
builder likewise excludes equal-order P/Q spurious roots, honors the supplied
SCF nuclear potential and does not classify zero-population spectators as occupied.

## Validation

- Broad non-slow run: **2,498 passed, 3 stale self-snapshot failures, 78 skipped,
  20 expected failures, 24 slow tests deselected**, in 813.38 s. That process
  loaded the three old correlation-energy literals before their refresh.
- Final affected-structure rerun, including all three: **77 passed, 5 skipped,
  1 slow test deselected**. No unexpected failure remains from the broad run.
- All **35 review regression gates pass**. They include independent orbital
  derivatives, CI tensor identities, metric constraints, scattering matching,
  reciprocity, Coulomb references, Milne integrals and virtual-root checks.
- Final virtual-basis/MBPT/polarizability checks: **42 passed, 1 existing expected
  failure**; DR/Auger consumers: **19 passed, 3 existing expected failures**.
- All Tl/EDM tests, including the slow enhancement checks, plus the final carbon
  split-channel test: **13 passed**. The Tl fixture now has all 81 electrons;
  its missing six 5d5/2 electrons were restored.
- Surface/potential/transition checks: **45 passed**. RR/Tl checks: **16 passed**.
  These runs overlap; counts must not be added.
- Ruff on all changed/new Python files and `git diff --check` pass.

Exact outputs and scope are in [validation_summary.json](validation_summary.json),
[broad_regression_output.txt](broad_regression_output.txt) and
[final_affected_structure_output.txt](final_affected_structure_output.txt).
Ten independently executed diagnostic modes are in
[remediation_probes.txt](remediation_probes.txt). The stringent helium
`tol=1e-10` probe still reports nonconvergence honestly despite a stable energy
and overlaps near machine precision; it is not counted as a stationarity pass.

The heavy-atom correlated derivative agrees with an independent five-point
occupied-orbital rotation derivative to **6.63e-10 Ha/radian**. Default Pb
converges in **19 optimizer steps**, at -20913.79776848005 Ha, with an orbital
residual of 8.274e-6 against 1e-5 and estimated energy error 6.260e-11 against
1e-8. A repeat reproduces its energy within approximately 3e-11 Ha.

## Performance and external accuracy

The controlled 80-energy, 278-spline continuum benchmark reduces 160
diagonalizations to 2: median **6.131 s → 0.341 s, approximately 18× faster**,
with **bit-identical cross sections**. See
[benchmark_continuum.py](benchmark_continuum.py) and
[continuum_benchmark.json](continuum_benchmark.json).

The default Pb solve took 97.881 s in the earlier isolated run and 123.727 s
while regressions ran concurrently. The earlier nuclear-only preconditioner
exhausted 160 optimizer steps in 817.558 s without converging; the screened
preconditioner converges in 19. This compares optimizer implementations during
remediation, not identical physics against the original nonorthogonal solver.
See [benchmark_structure.py](benchmark_structure.py),
[structure_benchmark.json](structure_benchmark.json) and
[correlation_space_measurements.json](correlation_space_measurements.json).

The repaired Fe K-shell RR panel differs from FAC by about **4%**, passing its
unchanged 15% gate. Separately optimized Li-like final levels differ by less
than 17%, within their existing external bands. External FAC columns are
unchanged from reviewed HEAD; [external_reference_integrity.json](external_reference_integrity.json)
records the comparison. Self snapshots were refreshed separately.

One special 0.2% excited-state cross-engine assertion was reclassified into its
existing 3% open-shell comparison band: the corrected B-spline constraints give
a 0.424% difference from shooting's documented approximate constraint treatment.
Closed-shell and independent analytic-kernel tolerances are unchanged. The split
correlation test now measures the added configurations' actual population rather
than imposing an arbitrary largest-CSF coefficient bound.

For chronological investigation details, see
[IMPLEMENTATION_HISTORY.md](IMPLEMENTATION_HISTORY.md). The validation records
above were collected before the changes were grouped into commits. The
pre-existing `.ace/` directory was not modified.
