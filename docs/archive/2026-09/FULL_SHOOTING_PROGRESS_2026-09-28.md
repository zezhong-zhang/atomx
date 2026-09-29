# Toward a full shooting MCDHF backend

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Implemented

There is now an experimental **shooting-only radial optimization path** in
`methods/mcdhf/grid_shooting.py`. It operates on grid-tabulated spinors and
the full correlated Dirac–Coulomb CI functional. It does not call the
B-spline radial solver or diagonalize a B-spline radial Hamiltonian.

The implementation deliberately requires an explicit electron space. It
does **not** silently promote the old occupation-averaged HF+CI path into
full MCDHF, and does **not** replace the public all-element engine yet.

### Full correlated grid functional

`grid_functional.py` contracts the same angular Hamiltonian tables and CI
amplitudes as the B-spline functional. Their shared implementation is now
`radial_density.py`. Off-diagonal one-body terms and configuration
interference are retained, including their radial derivatives.

The grid derivative differentiates the actual discrete integrals:

- The cumulative SLAG6 operator has an O(N) transpose, including all endpoint
  stencils. The Coulomb kernel uses the appropriate forward and transpose
  actions for its two density arguments. Substituting a continuum symmetry
  identity would not give the exact discrete derivative.
- The symmetric Dirac derivative uses both the finite-difference stencil
  and its transpose.
- The CI evaluator's `h_ab / norm_b` normalization is differentiated too;
  those normal terms vanish under exact tangent projection but matter to
  coordinate-derivative consistency and numerical conditioning.
- The actual SLAG6 metric defines norms, same-kappa orthogonality, polar
  retraction, and residuals.

### Inhomogeneous Dirac response

`numerics/shooting/dirac_response.py` reuses a regular/inward homogeneous
pair for multiple sources. Same-kappa overlap constraints are solved with
the coupled border

    x = R source + R C lambda
    (C^T W R C) lambda = target - C^T W R source.

This solves the constrained response equations directly. The optimizer
also retains active-active orbital rotations; imposing zero overlap on
all updates without that rotation component would change the MCDHF
variational problem.

Both decaying and finite-box `P(R)=0` boundary conditions are available.
Decaying responses reject sources outside their practical infinity. Box
responses retain sources throughout the grid. Direct inward quadrature
avoids cancellation from subtracting nearly equal cumulative tail totals.

### Optimizer and no-pair constraint

`optimize_grid_shooting` combines the response with transported Riemannian
L-BFGS, CI energy line searches, and a projected-gradient fallback. A
supplied `FixedElectronSpace` defines the no-pair range; the caller must
validate its physical origin and completeness.

Every trial and transported direction is re-projected. Heavy-ion testing
found that allowing tiny out-of-space components to accumulate could let
secant updates amplify them to about 8e-7 in norm and falsely lower the
energy. The regression checks the full optimization history for this drift.

Convergence uses the norm of the actual projected correlated gradient,
not update size. Near floating-point energy resolution, a trial can be
accepted only if its energy change and predicted decrease are within eight
machine epsilons of the total energy **and** its measured gradient norm
decreases. These acceptances are recorded; the final residual tolerance is
unchanged. This handles deeply bound ions without claiming energy resolution
beyond floating-point arithmetic.

### Warm starts on the existing HF path

`ShootingHFSolver.solve(seed=previous_result)` copies validated radial data
on the same grid and bypasses the local-exchange startup. Occupations and
Guerra weights may change. The MCDHF shooting initializer now carries its
previous result into the next outer iteration. Existing cold-start
statistical regression controls remain unchanged.

## Validation

The targeted regression run passed **62 tests**. The updated exact-normal
derivative and all grid tests were also rerun after their final change.
Coverage includes quadrature/derivative adjoints, finite-difference CI
gradients, analytic manufactured Dirac solutions, nonzero sources at the
box boundary, coupled constraints, projector leakage, energy acceptance,
changed Guerra weights, warm starts, and previous shooting/B-spline controls.

### Shooting-only correlated pilots

All five helium-like pilot ions use a Fermi nucleus, a 1000-point radial
grid, and ground even-J=0 CSFs from `1s2`, `2s2`, `2p2`. The electron space
contains **three nuclear bound shooting roots per kappa**. This intentionally
small space excludes the positive continuum and is not a production atomic
accuracy model. QED, Breit, and recoil are excluded to isolate the DC solver.

| Z | Iterations | Projected gradient norm | Energy difference from rotation control (Ha) |
|---:|---:|---:|---:|
| 2 | 41 | 7.21e-6 | 3.31e-10 |
| 6 | 98 | 9.59e-6 | 3.34e-9 |
| 26 | 274 | 5.47e-6 | 9.92e-10 |
| 54 | 215 | 9.39e-6 | 7.64e-9 |
| 92 | 96 | 8.96e-6 | 4.35e-9 |

All shooting pilots satisfy the same 1e-5 residual tolerance. The independent
control uses finite-difference BFGS over orthogonal rotations in the identical
electron space. Its uranium run reports precision loss and is retained as
such; it is not presented as a fully converged reference. Agreement here is
a solver check, **not agreement with NIST or a complete correlation space**.

The pilot includes all physical gradient terms of its stated DC functional,
but its finite space is too small to reproduce the existing production
B-spline energies. In particular, adding infinitely many nuclear bound
states alone would still omit the positive continuum.

### HF reuse controls across the table

He and Ne-like Ne, Fe, Xe, U were tested with Fermi nuclei at the same
1e-10 relative HF energy-change tolerance. Cold solves took 4, 22, 18, 18,
and 20 HF sweeps respectively; all warm solves took two. Warm/cold energy
differences ranged from 7.8e-11 to 6.4e-7 Ha (the latter for the much larger
uranium total energy). These are repeated-solve controls, not NIST errors or
a claimed speedup for arbitrary multiconfiguration calculations.

## Reproduce

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_shooting_grid_solver.py \
  --pilot-z 2 6 26 54 92 --nuclear fermi \
  --output output/correlation_benchmark/shooting_grid_final.json
```

The JSON contains source hashes, raw timings, CI energy histories, residuals,
constraint errors, projector leakage, fallback/roundoff records, nuclear
root energies, and the independent optimizer's actual success status.

## What remains before switching all calculations

1. Build a validated electron space with positive-continuum coverage and
   demonstrate radial-box/grid/space convergence. The supplied-space
   interface already accepts such a basis; the bound-only helper is explicitly
   diagnostic and cannot supply this completeness.
2. Validate many-electron open shells, diffuse/Rydberg roots, neutral atoms,
   and successive active-space expansions against matched B-spline/GRASP
   problems. The helium-like pilots do not establish that coverage.
3. Connect the converged grid orbitals to the production result/correction
   pipeline with checked orbital-energy and state provenance. Static Breit
   RCI already supports grid workspaces, but this pilot does not claim
   self-consistent Breit, QED, recoil, or frequency-dependent retardation.

No full-table spectra were regenerated, and the production engine default
was not changed.
