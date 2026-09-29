# Numerical boundary migration

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Date: 2026-09-29.

## Reason for this change

The numerical implementations already had canonical owners, but six imports
from numerics into methods or scattering remained as compatibility routes.
Keeping those routes made the dependency direction unenforceable. Separately,
loading shooting orbitals imported the local SCF implementation because it owned
the result class. These are concrete boundary defects with a bounded migration.

## Ownership

Shooting is a numerical engine. The former `methods/shooting_hf.py` and
`methods/shooting_local.py` are removed after separating their responsibilities.

| Responsibility | Owner |
|---|---|
| Local-potential SCF driver and frozen-orbital solves | `numerics/shooting/local.py` |
| Nonlocal HF shooting driver, constraints and numerical rotations | `numerics/shooting/hartree_fock.py` |
| Slater field, exchange coefficient and configuration-average energy | `physics/slater.py` |
| HF occupation pairs, angular coefficients, Fock action and energy | `physics/hartree_fock.py` |
| Nuclear field and initial density on a supplied grid | `potentials/slater_seed.py` |
| Electron-space construction and `ShootingSpace` controls | `numerics/shooting/electron_space.py` |
| Correlated radial optimization and Green preconditioner | `numerics/shooting/orbital_optimization.py`, `preconditioner.py` |
| CI/state orchestration on a radial grid | `methods/mcdhf/grid_solver.py` |
| CI energy/gradient and occupied 1s² reference policy | `methods/mcdhf/grid_functional.py` |
| Native shooting result and wavefunction views | `results.py` |
| Versioned shooting archive | `io/results.py` |
| Numerical hydrogenic reference calculation | `methods/hydrogenic_reference.py` |
| Photoionization process orchestration | `scattering/photoionization/direct.py` |
| Continuum eigenproblem and radial dipole contraction | `numerics/bspline/continuum.py` |

The numerical HF driver composes one `RadialHartreeFock` operator. The operator
specifies allowed rotations and which configurations require tracking a saddle;
the engine performs the rotation search. Physical energy expressions and
numerical minimization are separate. Methods and the Atom facade select and
call the engines. Numerics does not import them back.

## Breaking import changes

Update downstream callers using this table. Repository source, scripts,
benchmarks and tests now use the canonical paths. No forwarding modules remain
at the removed locations. Physical contractions previously accessed through HF
solver private methods now live under `solver.functional`, for example
`solver.functional.two_electron_energy(...)`.

| Previous import (under `atomx`) | Canonical import |
|---|---|
| `methods.shooting_local.ShootingSolver`, `numerics.shooting.solver.ShootingSolver`, or the shooting package export | `atomx.numerics.shooting.local.ShootingSolver` |
| `methods.shooting_hf.ShootingHFSolver` or `numerics.shooting.hf_shooting.ShootingHFSolver` | `atomx.numerics.shooting.hartree_fock.ShootingHFSolver` |
| `ShootingResult` through a shooting solver or package | `atomx.results.ShootingResult` |
| `dfs_frozen_orbital` through the old shooting solver | `atomx.numerics.shooting.local.dfs_frozen_orbital` |
| Slater integrals and `_yk_over_r` through the shooting solver | `atomx.numerics.shooting.coulomb` |
| Inhomogeneous orbital helpers through `hf_shooting` | `atomx.numerics.shooting.dirac_orbital` |
| Homogeneous pair and Green solve through `hf_shooting` | `atomx.numerics.shooting.dirac_green` |
| `methods.mcdhf.grid_shooting` electron-space builders | `atomx.numerics.shooting.electron_space` |
| `methods.mcdhf.grid_shooting.optimize_grid_shooting` | `atomx.numerics.shooting.orbital_optimization.optimize_grid_shooting` |
| `methods.mcdhf.grid_shooting.occupied_1s2_reference_potential` | `atomx.methods.mcdhf.grid_functional.occupied_1s2_reference_potential` |
| `methods.mcdhf.shooting_preconditioner` | `atomx.numerics.shooting.preconditioner` |
| `methods.mcdhf.shooting_solver.ShootingSpace` | `atomx.numerics.shooting.electron_space.ShootingSpace` (public `atomx.ShootingSpace` is unchanged) |
| `methods.mcdhf.shooting_solver.solve_correlated_shooting` | `atomx.methods.mcdhf.grid_solver.solve_correlated_shooting` |
| `numerics.analytic[.hydrogenic_dirac]` hydrogenic helpers | `atomx.methods.hydrogenic_reference` |
| `numerics.bspline.continuum.photoionization_cross_section` | `atomx.scattering.photoionization.photoionization_cross_section` |

The old `numerics/analytic` namespace contained numerical B-spline reference
calculations, not closed-form wavefunctions. Its forwarding files are removed.
Historical design and audit documents may retain the old paths as evidence of
the tree they reviewed.

## Scientific data transition

`ShootingResult` retains its fields, native arrays, metric and physical metadata.
Schema-1 shooting HDF5 files use a format identifier rather than a Python module
path, so they require no rewrite. Python pickle import compatibility is not
provided by this change; the versioned archive is the supported persistence path.

Local Slater results explicitly have no jj configuration ensemble. The archive
writer and validator now accept that `None` value and preserve it as JSON null;
previously they attempted to iterate it. Nonlocal HF ensembles continue to be
validated and preserved for open-shell correction replay.

## Boundary enforcement and scope

The architecture gate examines all numerical modules, including function-local
and relative imports, and rejects imports into methods, corrections, properties,
scattering, the Atom facade, workflows or web. The six-entry exception list has
been removed. Numerical formulas, tolerances and iteration algorithms are
unchanged in this migration. The local iteration module is recompiled because
its compiled physical callback moved modules; a stale Numba dispatcher cache
otherwise retained invalid callback references. This changes no SCF equations.

This completes those import transitions. It does not complete the larger
B-spline functional/iteration split, central request resolution, or packaging
of campaign and web services. Those need separate behavior-driven changes.
Existing running campaigns retain their frozen source snapshots.

## Verification

- Before/after comparison against the source snapshot taken before this change:
  six 800-point controls (local He/Ne, DHF He/Ne, open-shell C, core-hole Be).
  All energies, iteration counts and P/Q arrays are exactly equal across the
  18 compared arrays. Files: `/tmp/atomx-structure-{before,after}.npz`.
- An existing schema-1 campaign checkpoint loaded and round-tripped with method
  imports blocked; arrays, energy, configuration ensemble, calculation inputs
  and solver diagnostics were identical.
- Built the wheel using the locally available Hatchling build tools. Removed
  import paths are absent. An extracted-wheel HF solve, public `ShootingSpace`
  import and archive round-trip pass with all `atomx.methods` imports blocked.
- Repository-wide undefined-name checks pass; `git diff --check` passes.
- Following the user-confirmed retirement of campaign scripts, obsolete CLI
  tests were removed. Projector, radial-identity and projection regressions now
  call package APIs directly. Retained CLI tests follow the retained commands.
- Final affected checks: **129 passed** across the cleanup runs (68 command and
  radial-identity checks, 25 numerical-boundary/persistence/projector/projection
  checks, and 36 retained NIST interface/parser/comparison checks). Two newly
  written projection fixtures initially passed a string to an enum argument;
  correcting the fixtures made both pass without changing tolerances.
- Full collection succeeds: **3,153 tests**, with no missing-script import errors.
  The full numerical suite was not executed. Pytest reports two configuration
  warnings because the local environment lacks the declared timeout plugin.
- Current guide links resolve. No test imports a retired script. The docs root
  contains 11 current guides; 80 historical reports and superseded snapshots are
  indexed under `docs/archive/`.

## Commit preparation and command cleanup

The retained NIST commands now call package-owned configuration expansion,
angular weights, centroid matching and air/vacuum conversion. Duplicate command
implementations were removed; CLI paths remain stable. `scripts/README.md`
indexes archive, comparison, ledger-maintenance and example responsibilities.

Verification before grouped commits:

- Numerical foundations tested against the previous committed source plus only
  that group: **18 passed** (analytic exponential and rank-one nonlocal controls).
- Engine/physics/API/persistence integration: **220 passed**.
- Reference-data and retained NIST commands: **115 passed, 2 expected failures**.
- Full collection: **3,153 tests**, no collection errors. The full suite was
  not run; focused execution and collection are separate evidence.
- Five affected entry points accepted `--help` without starting calculations.
- Current guide links, undefined-name checks and staged whitespace checks pass.

The local session journal remains on disk but is removed from version control;
its existing ignore rule now applies. Assistant cache state is also ignored.
