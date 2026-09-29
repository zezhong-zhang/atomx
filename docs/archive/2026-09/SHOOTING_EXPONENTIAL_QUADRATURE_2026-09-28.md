# Exponentially fitted Dirac Green source integration

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Problem and numerical design

The previous scaled SLAG6 rule interpolated `source * exp(log_amplitude)`.
Scaling prevented floating-point overflow but did not resolve steep exponential
variation. For a constant source and log-amplitude increment 2 per grid cell,
the analytic prefix-integral control had approximately 32% relative error.
The step-100 guard was an overflow safeguard, not an accuracy guarantee.

The new rule factors out the local evanescent scale of the radial Dirac ODE.
At each cell midpoint, the coefficient matrix has eigenvalues ±sqrt(d), where

```text
d = (kappa/r)^2 - (E-V)*(E-V+2*c^2)/c^2
a = cell_width * sqrt(max(0, d))
```

This is a reference scale for numerical integration, not a replacement
Hamiltonian or a WKB approximation to the orbitals. The original fundamental
solutions, potential, source and boundary conditions remain in the calculation.
In oscillatory regions the reference scale is zero.

For a prefix ending at index i, with stencil nodes j, let L be the fundamental
solution's stored log amplitude and f the sampled signed source contraction.
Interpolate the six residual samples

```text
b_j = f_j * exp(L_j - L_i - a*(j-i))
```

as a degree-five polynomial b(t). Integrate `exp(a*t)*b(t)` over `[-1,0]`
using 16-point Gauss quadrature, subdividing so each panel spans at most eight
reference log units. Add the preceding prefix scaled by `exp(L[i-1]-L[i])`.
The inward integral uses the corresponding reversed cell rates.

The residual samples retain cancellation between the normalized spinor's
direction and amplitude. Separately interpolating log amplitude and direction
failed a manufactured finite-box boundary control by 6.4%. A curvature-based
hybrid passed that boundary control but did not recover calcium. Both candidate
designs were rejected in favor of the ODE-based scale and residual-source rule.
There is no curvature selector in the final implementation.

Six-node interpolation still requires resolved source variation. The retained
coarse-grid/nonfinite guards and radial refinement checks remain necessary.
This change does not make coarse grids or arbitrary source functions exact.

## Ownership and reuse

- `numerics/shooting/exponential_quadrature.py` owns weight construction and
  the linear prefix application.
- `dirac_green.py` derives the local ODE scale and constructs forward/backward
  integration maps for a fixed fundamental pair.
- `dirac_orbital.py` reuses those maps for the exchange source, every constraint
  response and the normalization derivative at the same trial energy.
- `dirac_response.py` prepares maps once for its fixed operator/energy and
  reuses them across source applications.
- Shooting result diagnostics identify `dirac_evanescent_fitted_gauss16`.

Maps are local to a fundamental pair. No cache is shared across different
energies, potentials, grids or boundary conditions. This preserves the source
linearity required by the constrained inverse and its energy derivative.

A warm 3,200-node prefix microbenchmark measured approximately 0.331 ms to
prepare weights, 0.0129 ms per prepared application and 0.0768 ms per legacy
application. Preparation amortizes after roughly six applications in that
case. These are kernel timings, not a claimed whole-SCF speedup.

## Verification

Forty-three focused checks passed together across exponential quadrature,
correlated-grid response/optimization, open-shell constraints, independent
B-spline comparisons, native corrections, numerical contracts and full shooting
integration. An additional amplitude/direction-cancellation control then passed;
the final exponential test module has 12 passing tests (44 distinct checks
across the two runs). Local pytest reported two warnings because its timeout
plugin is absent.

The analytic controls include constant exponentials through log step 100,
signed degree-five sources, invariance under arbitrary constant log offsets,
source linearity, zero-rate cancellation, Green reciprocity and the identity
`d(H-E)^-1/dE = (H-E)^-2`. Atomic controls retain their existing tolerances.
No tolerance was relaxed to accept the new quadrature.

The preliminary unrestricted prototype recovered Ca I 4s63p at 16,000 and
32,000 points, with corrected energies agreeing within about 1.2e-11 Ha,
but failed the finite-box control and is not the final accepted implementation.
The ODE-based prototype passed that control and recovered Ca I 4s63p at
16,000 points, at −679.1877023838316 Ha. Monkeypatched runs are development
evidence only; their embedded normal source hashes are not production evidence.
Scripts/logs are retained in `exponential_quadrature_development.json`.

Fresh unmodified-source runs are stored separately:

- `native_shooting_first20_ode_exponential.json`: all occupied H–Ca ground
  stages on 1,600 and 3,200 points;
- `native_shooting_calcium63_ode_exponential.json`: Ca I 4s63p on 16,000 and
  32,000 points;
- `native_shooting_calcium_ground_ode_exponential.json`: matched Ca I ground
  controls on those finer grids;
- `native_shooting_excited_ode_exponential.json`: the excited configuration panel.

Inspect each checkpoint's statuses and source hashes before using it. Earlier
`*_exponential.json` and `*_hybrid_exponential.json` filenames refer to rejected
intermediate schemes; do not merge them with the final `*_ode_exponential.json`
records. No production SQLite ledger has been promoted by this numerical repair.

### Final checkpoint verification

All final checkpoint source hashes match the current physical/numerical source.
The 420 ground runs succeeded, providing 210 finer-grid ionization comparisons.
The largest two-grid ionization change is **1.872e-6 cm⁻¹**, down from
4.246e-4 cm⁻¹ before this repair. The 29 evaluated/unique-ground comparisons
still have RMS discrepancy **11,581.0412 cm⁻¹**; the improved numerical integration
does not remove the physical-model discrepancy.

Ca I 4s63p succeeded at both 16,000 and 32,000 points; corrected energies
differ by **1.467e-11 Ha**. Its matched ground controls also succeeded at both
grids. The 48-run excited panel has 46 successes; only the two under-resolved
Ca 4s63p attempts fail. Adding its fine-grid refinements gives at least one
successful solve for every one of the 24 distinct diagnostic configurations.
Mg I 3s20p now also converges on both coarse grids. These successes do not
establish completeness of the full first-20 observed-line endpoint queue.

The refreshed interactive report uses `native_shooting_excited_ode_comparison.json`
and `native_shooting_first20_ode_exponential.json`. Coarse-grid failures remain
in refinement details. Regenerate it with:

```sh
PYTHONPATH=src:. python scripts/compare_native_shooting_excited.py \
  --excited-refinement output/nist_spectra/native_shooting_calcium63_ode_exponential.json \
  --ground-refinement output/nist_spectra/native_shooting_calcium_ground_ode_exponential.json
python scripts/render_native_shooting_benchmark.py
```

## Remaining scope

Ground ionization controls are not the full observed-line calculation queue.
The high-Rydberg series, other excited-state endpoints and cross-periodic-table
controls still require validated generation. Correlation/model residuals against
NIST remain separate from radial numerical error. NIST archive completeness
also remains a separate audit, including the documented plasma-diagnostics
source-access gap; none of these numerical tests establishes archive completeness.
