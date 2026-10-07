# Test performance: measured hotspot and correctness-sized workloads

## Contract and experiment

The user requested hotspot optimization or smaller calculations where correctness
can be checked without large campaigns. Preserve operator definitions, signed
interference, complete symmetry spaces, reference independence and acceptance
rules. Large measured physical targets keep their original models and gates.
Stop after decisive profiling and affected validation, rather than scan atomic
models or repeat the full suite as a timing experiment.

A warm-cache profile of the original six-kernel/Slater-scaling modules ran
59 tests in **204.72 s**: Pr fit/linearity 57.61 s, Eu transfer 77.24 s, Nd transfer
41.42 s, and the 2000→8000-node Gauss refinement check 20.90 s. These are observed
suite timings on a shared machine, not a controlled end-to-end benchmark.

## Changes

1. The daily Slater test uses fixed normalized 4f-like radial functions on a
   24-function basis and the **entire 4f² space (91 determinants, 14 spinors)**.
   Tests check every pair's exchange signs and Hermitian conjugation, nonzero
   off-diagonal elements, per-rank linearity, full J-manifold counting and
   recovery of known parameters through the public level/fit machinery.
   Parameter recovery is an internal algorithm test, not an independent physical
   reference or a stationary free-ion result.
2. The original Pr/Eu/Nd transfer campaign is retained in
   `tests/test_slater_scaling_transfer.py`, marked slow, with one module-scoped
   Pr fit shared by all consumers. Full pytest still includes it. Its models,
   state matching, reference energies and numerical bounds are unchanged.
3. The kernel refinement tests use 25k→100k trapezoid samples and 500→2000 Gauss
   nodes. The same 4x refinement factor and 6–40 error-ratio bounds distinguish
   second-order convergence from a convention error. Scalar/profile precision
   gates, all six kernels, k=0/1/2, native grids and independent anchors remain.
4. `numerics.quadrature.gauss_legendre_rule` uses SciPy's banded Jacobi solve
   and caches at most 32 canonical orders. Arrays are read-only, keyed solely
   by validated order; physics and interval mapping are not cached. The
   two-electron and sinh-energy quadratures share it. Independent moments
   through degree 2n−1, positive weights, symmetry and mutation rejection gate
   the rule. Existing continuum/integration gates check consuming entry points.

The final focused small suite ran **74 tests in 2.40 s** (69 numerical/Slater
checks plus 5 failure-contract checks). The workloads differ from the 59-test
baseline, so dividing those times would not be a scientific speedup claim.
The retained transfer campaign also passed in the 150-test follow-up.

## Same-order kernel measurement

`benchmarks/bench_gauss_legendre.py --order 2000 --pairs 6` alternates baseline
and candidate order, pins BLAS to one, clears the candidate's cache before each
cold sample and measures warm lookup separately. The predeclared load ceiling
is 0.5 per logical CPU; positive weights and four analytic moments must pass
at absolute error 2e-12. The dense baseline companion is limited to 4000 nodes
(128 MiB for that matrix alone). Default 2000-node work needs 32 MiB for it.

The final run passed its load guard (32 CPUs, ceiling
16.0). Median cold construction was 0.23472 s for NumPy
and 0.05720 s for the shared rule; median paired ratio
**4.10x**. Nodes differ by at most 2.22e-16 and weights
by 1.30e-13; physics crosschecks, not bitwise identity, establish acceptance.
This measures quadrature generation only, not Slater assembly, SCF or the whole
suite. Raw rows, versions, loads and moment errors are in
[quadrature timing](quadrature_timing.json).

A preliminary warm-timing script mistakenly cleared the candidate cache before
its baseline call on alternating pairs; that script's warm samples included one
cold construction and are withdrawn. The registered benchmark fixes this order
and its cold/warm measurements are separate.

## Documentation impact and limitations

Updated: [testing commands](../TESTING.md), [Slater scaling gates](../SLATER_SCALING_SPEC.md),
test docstrings and the review status. Retained: experimental transfer conclusions,
radial-grid defaults and operator-coverage guides; the production physical models
are unchanged. Existing archive timings apply only to their source and load.
Neither the small-space gate nor faster quadrature resolves correlation,
continuum convergence, the known pointwise/bias limits, or the Ne discrepancy.
