# Native shooting: refreshed ground states and excited controls

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Completed evidence

The ground campaign `native_shooting_first20_projected_poles.json` completed
420 successful runs: every occupied ground ion stage of H–Ca on 1,600 and
3,200 points. There are 210 finer-grid ionization comparisons. The largest
two-grid change is 0.000424598 cm⁻¹. For the 29 evaluated NIST references with
unique ground levels, the RMS residual is 11,581.0412 cm⁻¹. Numerical grid
stability and agreement with experiment remain separate questions.

The source hashes of all ground/excited/refinement inputs were checked against
the current source files before aggregation. No physical source file changed
during this review. The production SQLite calculation ledger was not modified.

The 24-configuration excited panel covers one low excited configuration per
neutral element plus H 10s, Mg 3s20p, an Ar II core hole and Ca 4s63p. Following
the constrained pole-count repair, 44 of its 48 coarse/fine runs succeed.
He 1s2s now solves on both grids. Refined H/Mg runs at 6,400 and 12,800 points
all succeed; Mg corrected total energies differ by approximately 1.2e-12 Ha.
Ca 4s63p still fails at 16,000 and 32,000 points with the current kernel.

The combined comparison contains 54 attempts, including refinements:

- 40 matched evaluated configuration-centroid comparisons;
- 8 unscored comparisons because the required complete evaluated NIST
  manifolds are unavailable under the matching policy;
- 6 failed attempts, including coarse Mg attempts and all sampled Ca 4s63p grids.

These counts are attempts, not distinct configurations or predicted spectral
lines. The report retains all attempts; its overview shows the finest attempted
grid for each configuration. Details include earlier failures and both endpoint
calculations. He recovery is numerical evidence, not a scored experimental
comparison when its manifold does not meet the reference policy.

## Remaining discrepancies

Representative finer-grid centroid residuals (theory minus NIST):

| Configuration | Residual / cm⁻¹ |
|---|---:|
| Mg I 3s3p | −4,697.047 |
| Ar I 3p5 4s | −5,904.383 |
| Ca I 4s4p | −4,118.472 |
| Ar II 3s 3p6 core hole | +42,180.000 |

They are essentially unchanged by the sampled radial refinement. DHF lacks
the configuration mixing and correlation required for a general experimental
accuracy claim. These data alone do not allocate the entire residual to
correlation: state assignment, stationarity and the scope of each correction
still require separate checks. Configuration centroids must not be advertised
as resolved-level or individual-line predictions.

## Calcium failure investigation

An instrumented, unchanged-source run localized the first 16,000-point failure
to the constrained 2s solve at energy −30.7937006877 Ha. Its border eigenvalues
were approximately 0.0333440, 0.0362570 and 6.66986e12, giving condition number
2.00032e14. The asymmetry norm was only 3.78e-13.

A scratch-only diagnostic that assessed a diagonally equilibrated border
instead of the original condition number did **not** recover a valid solve.
It continued into enormous Green responses and eventually the coarse-grid
guard. Therefore no relaxed guard or equilibration-only change was installed.

The independently known integral

`exp(-a*x) integral_0^x exp(a*t) dt = -expm1(-a*x)/a`

provides a direct test of `_scaled_slag6` without atomic-model uncertainties.
For constant input and a linear log amplitude, maximum interior relative errors
are about 3.46e-9 at step 0.1, 0.00355 at step 1, 0.321 at step 2, and 800.6
at step 5. The current step-100 guard prevents extreme exponent overflow; it
does not guarantee quadrature accuracy. This reproduces a stiffness limitation
of polynomial interpolation of the exponential integrand. It does not prove
the complete causal chain for every calcium iteration.

Next numerical work: an exponentially fitted or adaptively integrated Green
source rule, verified for polynomial/oscillatory sources, reciprocal Green
actions, the native metric, constrained responses and energy derivatives.
Changing normalization tolerances or truncating exchange tails is not a repair
for this issue. Existing converged controls must be preserved before rerunning
the failed Rydberg series and expanding to all observed-line endpoints.

## Reproduction and interactive output

```sh
PYTHONPATH=src:. python scripts/compare_native_shooting_excited.py \
  --excited-refinement output/nist_spectra/native_shooting_diffuse_refinement.json \
  --excited-refinement output/nist_spectra/native_shooting_calcium63_projected_refinement.json \
  --ground-refinement output/nist_spectra/native_shooting_diffuse_ground_controls.json
python scripts/render_native_shooting_benchmark.py
PYTHONPATH=src:. python scripts/probe_scaled_green_quadrature.py
```

Output: `output/nist_spectra/native_shooting/index.html`, also exposed through
the atlas route `/comparison/native-shooting/`. Report input hashes, source maps,
archive hash and comparison-script hash are recorded. Refinement aggregation
rejects duplicate calculations and different physics source maps.

Verification: three provenance regression tests passed; embedded report counts
and failure details were checked, and the generated client script passed
`node --check`. No visual browser inspection or full-suite run is claimed.
The archive-completeness research and all-element production campaign remain
unfinished; this panel is a numerical and matching control, not their substitute.
