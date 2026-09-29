# Correlated orbital updates using two-sided radial Green solves

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Scope

This implements an experimental preconditioner for Atomx's full correlated
Dirac–Coulomb optimizer. It does **not** enable full variational MCDHF in the
older `engine="shooting"` configuration-average solver. The existing B-spline
energy functional, CI coefficients, fixed no-pair projector, polar
orthonormalization, and Armijo acceptance checks are retained. Static Breit
(including retardation), QED and recoil retain their existing treatment;
this change does not move them into the orbital variational functional.

```python
from atomx import Atom, Correlation

atom = Atom(
    2,
    config=["1s2", "2s2", "2p2"],
    engine="bspline",
    correlation=Correlation(orbital_preconditioner="shooting"),
    corrections="none",  # isolate the Dirac–Coulomb optimizer
).solve()
print(atom.result.orbital_optimization)
```

The same option is available on `MCDHFSolver`, `MCDHF`, and
`optimize_orbitals`. The default remains `"spectral"`.

## Numerical construction

For a spinor component with orbital angular momentum l, invert the positive
radial operator

    A_l = -1/2 d²/dr² + l(l+1)/(2r²) + sigma,

with regular origin behavior and a zero value at the outer B-spline box.
The fixed shift is the magnitude of the initial orbital's Fock Rayleigh
quotient (nuclear Dirac quotient when Fock fields are unavailable), floored
at 0.5 Hartree. It sets a numerical update scale; it does not alter energy
levels or the atomic Hamiltonian. Large and small components have their
own correct l values for their shared kappa.

The two homogeneous solutions are sqrt(r) I_(l+1/2)(sqrt(2 sigma) r)
and sqrt(r) K_(l+1/2)(sqrt(2 sigma) r). Their Wronskian is -1. Scaled Bessel
functions and exponential recurrences avoid exponentially growing storage.
The l=0 case uses elementary exponentials and `expm1`. Inward and outward
sweeps apply the Dirichlet Green kernel to all sources in linear work per
radial source. The shared diagonal quadrature term is counted only once.

If U samples the optimizer's fixed positive-energy basis and W contains
quadrature weights, its reduced inverse is

    B = sum_components U^T W G W U / (2 max(occupation, floor)).

The occupation floor is the existing optimizer setting. The Green operator
is symmetric positive; projection preserves this property. Its reduced
matrix is built once per orbital and cached. Subsequent L-BFGS iterations
use small matrix-vector products, with no repeated shooting initialization.
The source is the full correlated CI gradient, including configuration
interference. Orthogonality multipliers are enforced by the existing
Stiefel tangent projection and polar retraction.

This operator approximates inverse curvature for numerical acceleration.
It is not an approximation substituted into the physical Hamiltonian, and
is not a screened, coupled Dirac orbital boundary-value solver.

## Convergence safeguards

- Both methods use the **same spectral residual and energy-error estimate**.
  A small shooting update cannot by itself declare convergence.
- Every accepted update lowers the original correlated CI functional.
- A failed line search retries the spectral descent direction.
- Less than a factor-two residual reduction over 20 iterations switches
  permanently to spectral updates and clears secant history. Diagnostics
  record the switch reason and number of shooting iterations.
- Small and zero occupations use the same finite numerical floor as the
  baseline; no missing physical interaction is inferred from occupation.
- The default algorithm is unchanged. A redundant, unused construction of
  the initial spectral inverse was removed from both paths.

## Reproducibility and validation

`scripts/benchmark_mcdhf_shooting_updates.py` compares identical CSFs,
kinetically balanced B-spline bases, nuclear orbital seeds, and tolerances.
It records source hashes at startup, repeated alternating run order,
optimizer timings, evaluation timings, energy history, and convergence.
The diagnostic Hamiltonian is point-nucleus, infinite-mass Dirac–Coulomb.
These are solver comparisons, not new NIST accuracy claims.

The tests independently check the Dirichlet Green kernel, positivity,
manufactured differential-equation solutions with centrifugal terms,
unchanged stopping diagnostics, orthogonality/no-pair constraints, real
energy descent, spectral fallback, equal stationary energies, and public
API wiring. The kernel quadrature need only precondition accurately; all
accepted physical energies still use the existing cell-integral functional.

## Measured results

Final code, 55 B-splines, balanced component orders, even J=0 helium-like
SD spaces, identical nuclear starts, residual tolerance 1e-5 and estimated
energy-error tolerance 1e-8 Hartree:

| Case | Spectral iterations | Hybrid iterations | Spectral seconds | Hybrid seconds | Absolute energy difference (Ha) |
|---|---:|---:|---:|---:|---:|
| He, n<=2 | 36 | 44 | 0.654 | 0.797 | 1.15e-11 |
| C V, n<=2 | 108 | 125 | 1.758 | 2.203 | 7.26e-10 |
| He, n<=3 | 339 | 456 | 26.313 | 33.250 | 2.19e-13 |

Both paths converge in all rows. Small-space times are medians of two
alternating-order repetitions; the n<=3 extended-budget comparison is one
pair. Timings are from a shared host and should not be treated as precise
machine-independent speed ratios. The n<=3 hybrid used 40 shooting updates,
then spectral updates after the residual-stall safeguard fired. With the
original 400-iteration budget it correctly reported **unconverged**, despite
a small energy difference (2.60e-8 Ha); its residual was 1.11e-4.

Artifacts under `output/correlation_benchmark/`:

- `shooting_hybrid_final_he_n2.json`
- `shooting_hybrid_final_c_n2.json`
- `shooting_hybrid_final_he_n3.json` (400-iteration limit)
- `shooting_hybrid_final_he_n3_long.json` (800-iteration limit)

The final checks passed **86 tests** across the new shooting preconditioner,
virtual stability, public input surface, and structure/numerics review suites.

**This implementation has not demonstrated a speedup.** Approximately 95–99%
of optimizer wall time in these runs is spent evaluating CI energies and
their gradients. The positive kinetic preconditioner omits nuclear attraction
and screened exchange curvature from its inverse, while the baseline uses
nuclear/Fock spectral gaps. Cheaper radial integration cannot compensate for
extra correlated iterations here. The shooting option therefore remains
experimental and the spectral default is retained.

## Remaining work

A complete GRASP-style shooting MCDHF engine still requires correlated
screened Dirac radial sources, coupled same-kappa orthogonality multipliers,
virtual-orbital root control, and independent radial-grid convergence.
Replacing radial diagonalization alone cannot remove CI/gradient costs or
guarantee the lowest stationary solution. This hybrid supplies a guarded
integration point and reproducible performance measurements for that work.
