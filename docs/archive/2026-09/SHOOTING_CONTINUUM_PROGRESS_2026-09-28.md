# Shooting electron-space completion and numerical convergence

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

**Follow-up:** the heavy-ion mismatch discussed below was isolated to the
electron-reference convention and resolved for the matched `1s2` controls.
See [the reference-matching report](SHOOTING_REFERENCE_MATCH_2026-09-28.md).
The measurements below retain their original nuclear-projector scope.

## Implemented

`numerics/shooting/box_continuum.py` builds regular Dirac box eigenfunctions
with `P(R)=0`. Energies exclude rest energy. Positive energies represent
box pseudostates of the electron continuum; negative bound energies here
are electron states, not the negative-energy Dirac sea.

The potential spline and grid are reused across roots. Zero-energy node
counts identify the first positive root. Successive node counts bracket
roots and trigger subdivision if an interval contains several roots.
Normalization uses the same finite-box SLAG6 metric as the CI functional;
no asymptotic scattering-amplitude approximation is used.

Diffuse negative box roots are also included by default. They are found
between a localized-bound-state cutoff and zero, with counts verifying
that the cutoff follows exactly the supplied number of deep roots.
Inward/outward matching near the turning point reconstructs the spinor.
This avoids the unstable growing component that contaminates outward-only
propagation of decaying bound states. The deep localized roots still use
`dbound`; their boundary amplitude and their overlaps with the box roots
are checked before the combined basis is whitened. They approximate the
corresponding deep box states only when the box is sufficiently large.

`continuum_enriched_electron_space` connects this basis to the existing
full correlated grid optimizer. It reports root counts, omitted negative
roots, boundary/matching defects, overlap defects and the finite positive
energy cutoff. The calculation rejects an undersampled output grid even
if the adaptive ODE integrator can reach the boundary: accurate integration
alone does not ensure accurate subsequent fixed-grid CI quadrature.

This closes the absence of continuum and diffuse negative roots in the
diagnostic basis. All-element convergence and public-engine integration
remain separate requirements.

## Performance and convergence fixes

Overlap construction, whitening and repeated projection now use BLAS
matrix products. A single-thread, 160-vector by 2 by 4,000-point projection
microbenchmark measured 1.74 ms before and 0.196 ms after, about 8.9 times
faster, with relative numerical difference 2.7e-15. This is a projection
kernel result, not an overall MCDHF speedup claim. Raw measurements are in
`output/correlation_benchmark/shooting_projector_performance.json`.

The line search now checks unresolved energy changes before the ordinary
Armijo test, using the sufficient-decrease threshold to detect rounding.
Such acceptance still requires a measured reduction of the projected
gradient. After an acceptance checked through the gradient, the next step
starts at full size instead of inheriting a tiny step selected by energy
noise. The final gradient tolerance is unchanged.

This resolves the C V 48-continuum-root regression: previously 500
iterations ended at residual 3.57e-6; now 20 iterations reach 8.21e-7 with
the same 1e-6 tolerance. A dedicated test covers this case.

## Validation

**72 targeted tests passed**, including:

- Free-electron box spectra against analytic spherical-Bessel roots for
  four kappa channels, with relativistic dispersion.
- Coulomb continuum roots against a separate SciPy DOP853 integration.
- Grid refinement, quadrature rejection, box-boundary conditions,
  orthogonality, negative root counts and the localized hydrogenic limit.
- Nested electron spaces and the carbon convergence regression.
- Existing full CI derivative, Green response, leakage, HF/Guerra,
  warm-start and B-spline preconditioner tests.

### Helium radial completeness, single-configuration DC

Point nucleus, infinite nuclear mass, `1s2`; Breit/QED/recoil excluded.
The independent self-consistent shooting HF reference is
−2.861813341703658 Ha.

| Positive box roots | Box radius (bohr) | Grid points | Energy (Ha) | Difference from HF (Ha) |
|---:|---:|---:|---:|---:|
| 0, localized bound roots only | 60 | 4,000 | −2.831228811457427 | 3.06e-2 |
| 160 | 60 | 6,000 | −2.861811760092108 | 1.58e-6 |
| 256 | 60 | 6,000 | −2.861813313462418 | 2.82e-8 |
| 320 | 60 | 6,000 | −2.861813339017759 | 2.69e-9 |
| 384 | 90 | 7,000 | −2.861813311607277 | 3.01e-8 |

The 384-root larger-box calculation has approximately the cutoff of the
256-root smaller box; their total energies differ by 1.86e-9 Ha. Root
counts alone must not be compared across box sizes. At 160 roots, increasing
the smaller-box grid from 4,000 to 6,000 changes the energy by about 2e-14 Ha.
All listed enriched runs satisfy the same projected-gradient tolerance.

### Correlated helium

For even J=0 CSFs from `1s2`, `2s2`, `2p2`, box radius 100 bohr and 5,000
grid points, converged energies at 0/16/48/96 continuum roots are respectively
−2.843450924537, −2.860976750763, −2.882031326577 and −2.894621173671 Ha.
All satisfy a 1e-6 projected-gradient tolerance. These results demonstrate
the missing radial-space contribution to correlation, not a converged
correlation expansion or experimental agreement.

### Heavy-ion limits remain explicit

The Fermi-nucleus `1s2` panel at Z=2, 6, 26, 54, 92 converges for all three
tested spaces (0, 48, 160 continuum roots). At 160 roots the differences
from separate self-consistent HF are +1.58e-6, +3.47e-6, −1.47e-5,
−2.96e-4 and −2.37e-3 Ha.

The uranium difference is unresolved. Increasing this solver's grid from
4,000 to 8,000 changes its energy by only about 3e-11 Ha; increasing the
reference HF grid from 1,000 to 4,000 changes its energy by 1.75e-7 Ha.
Neither accounts for the 0.00237 Ha difference. The present optimizer uses
a fixed nuclear electron projector, whereas the separate HF solver solves
self-consistent unrestricted radial equations. These model definitions
must be matched before attributing the discrepancy to a specific term or
claiming high-Z agreement. Positivity of a space and convergence within it
do not establish equivalence to another no-pair Hamiltonian.

## Artifacts and reproduction

`scripts/benchmark_shooting_continuum_space.py` records source hashes,
parameters, root diagnostics, full convergence status and raw energies.
The final artifacts under `output/correlation_benchmark/` are:

- `shooting_continuum_final.json`: five-ion Fermi panel.
- `shooting_continuum_refined_final.json`: helium grid/cutoff refinement.
- `shooting_continuum_box_final.json`: helium larger box.
- `shooting_continuum_correlated.json`: correlated helium.
- `shooting_continuum_uranium_refined.json`: heavy-grid check.

Example:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_shooting_continuum_space.py \
  --continuum 160 256 320 --grid 6000 \
  --output output/correlation_benchmark/shooting_continuum_refined_final.json
```

Production defaults are unchanged. Open-shell validation, matched heavy-ion
projectors and correction/result integration are still required before
launching a full-table shooting campaign.
