# Shooting radial operators: physics, performance and verification

Baseline: `94d7592`. This experiment changes radial operator execution;
the Hamiltonian, physical coefficients, grids and stopping criteria are shared
between implementations. Detailed results are in [summary.json](summary.json),
source/output hashes in [manifest.json](manifest.json), and the initial warm Cf
profile in [hotspots.json](hotspots.json).

## Physical and numerical reasoning

The radial Dirac energy uses the symmetric bilinear form

\[
\frac{c}{2}[Q_a D P_b+Q_b D P_a-P_a D Q_b-P_b D Q_a]
+\left[\frac{c\kappa(P_aQ_b+Q_aP_b)+rV(P_aP_b+Q_aQ_b)}{r}
-2c^2Q_aQ_b\right]\frac{dr}{ds}.
\]

Here `D` differentiates the uniform index coordinate. The kinetic Jacobian
cancels; local terms retain it. The integral keeps the original cumulative
SLAG6 summation and zero inverse-radius convention at the origin. Keeping both
derivative directions preserves the discrete symmetric form, including its
boundary terms; replacing it with a strong-form operator or integration by
parts would need a separate discrete-adjoint derivation.

- Compile the existing eighth-order centered and boundary derivative stencils.
  Evaluate four independent central stencils before storing results to permit
  SIMD packing without fastmath or reassociation. A first scalar implementation
  regressed at 12,000 points; it was replaced before final measurements.
- Compile and fuse the one-body contraction. For diagonal matrix elements,
  reuse derivatives only when data pointers and strides identify the same
  views. Distinct spinors and strided views retain their own derivatives.
- Compile Green source projection, prepared prefix/suffix integration and
  spinor reconstruction as one action. Build the inward source in reverse
  order directly. Preserve the exponential quadrature, pointwise Wronskian
  check, source linearity, full-box boundary and negative-energy response.
  This operation alone is not a no-pair projector.

All scratch is owned by the invocation. Physical occupations, coherent CSF
amplitudes, exchange, Breit, QED and mass corrections remain in their existing
owners. No source-dependent fit, mutable global work buffer or approximate
operator replacement was introduced.

## End-to-end timing

Four fresh processes ran the selected panels in before/after/after/before
order. Each element had four solves per process; its first solve was excluded
from the warm median. The table uses six warm samples per implementation and
element, with one BLAS/OMP thread and no profiling during latency measurements.
Imports and constructors are excluded. The first solves, source fingerprints
and individual timings are retained in the raw reports.

Default shooting HF, Hamiltonian averaging, Breit + QED, tolerance `1e-5`:

| Element | Before | After | Reduction | Iterations |
|---|---:|---:|---:|---:|
| C | 213.29 ms | 190.71 ms | 10.58% | 7 |
| Ne | 154.53 ms | 141.68 ms | 8.31% | 8 |
| Xe | 856.97 ms | 764.51 ms | 10.79% | 10 |
| Cf | 3.69568 s | 3.32592 s | 10.01% | 7 |

Every solve converged. Total and orbital energies are bit-identical across
this panel, with the same iterations and result types. C/Cf use the dedicated
Hamiltonian DF route; these measurements do not select MCDHF.

The warm kernel measurements use 10 batches of 100 actions after preparation
and compilation:

| Operator | 1,000 points | 4,000 points | 12,000 points |
|---|---:|---:|---:|
| Derivative speedup | 7.97x | 3.75x | 2.00x |
| Diagonal Dirac contraction | 16.64x | 7.01x | 4.37x |
| Off-diagonal Dirac contraction | 10.47x | 4.23x | 2.67x |
| Prepared Green action | 1.97x | 1.48x | 1.33x |

Kernel speedups are not whole-solver speedups. These are measurements on one
machine, not a portable latency guarantee. Cold JIT and initial angular-cache
construction remain material costs; no cold-start improvement is claimed.

## Accuracy and safety

- **271 focused numerical/physics/architecture tests pass**, including native
  correlated gradients and adjoints, constrained Green responses, long-range
  exchange sources, nonlocal recovery, heavy-element bound roots and existing
  external-reference controls.
- **2 fresh-process reproducibility/bounds tests pass**: three repetitions of
  the full C/Ne/Be/Cf panel and an independent bounds-enabled compilation.
- New analytic controls cover degree 0–8 polynomials at the minimum stencil
  width and every four-point remainder, Coulomb-potential matrix elements with
  independently integrated radial moments, radial refinement, strided/copy
  inputs and off-diagonal symmetry.
- All **26 observable arrays are bit-identical**, including native Ne bound
  waves, a full-box response, 12 screened continuum waves, and sampled CE,
  EII, photoionization and recombination cross sections. Continuum phases and
  all seven E1/M1/E2 transition records in both gauges are bit-identical.
- The microbenchmarks have bit-identical matrix elements and Green outputs.
  Derivative boundary roundoff differences are below `1e-21` for these probes.

The collision controls explicitly use B-spline bound states and shooting
continua because these coefficient-based public APIs do not yet accept plain
native shooting bound states. Their small channel spaces are regression
controls, not converged cross-section predictions. The local environment lacks
`pytest-timeout` and reports its two configuration warnings. This was a focused
verification, not a new full-repository test audit. Numerical speed and unchanged
output do not establish improved agreement with experiment or complete correlation.

## Remaining costs and next design steps

The initial warm Cf profile has 4.93 million calls. Approximate self times
below include profiling overhead and should not be read as latency benchmarks.

| Remaining work | Evidence | Requirement for a correct optimization |
|---|---|---|
| Local-field initialization | `_dhfs_scf`: 0.676 s | Preserve the seed model and root identity; distinguish per-solve initialization from process/JIT startup. |
| Coherent Coulomb source construction | `_two_body_action`: 405 calls, 0.289 s | Compile the sparse angular contraction and reuse pair potentials only within a declared orbital state. Retain every interference term and multiplicity. |
| Rotation-energy bookkeeping | 64,078 potential lookups; 236 energy callbacks | Preindex the fixed/variable contraction graph per pair rotation. Invalidate changing densities and never replace the coherent energy with occupation products. |
| Root/constraint orchestration | 486 inhomogeneous solves; 1,529 prepared Green maps | Keep Sturm/constraint inertia, normalization and wrong-root guards. Factor a border only while its energy, potential, metric and constraints are fixed. |
| Repeated direct/exchange fields | 567 orbital-field and exchange calls | A reusable field context needs explicit invalidation after each Gauss–Seidel orbital update, rotation and CSF amplitude change. Array identity alone is insufficient. |
| Breit/retardation angular dispatch | 56,310 Breit radial calls, 2,188 retardation channels | Compile/index the immutable angular graph; preserve component ordering, signs and the chosen static or finite-frequency operator. |

The highest-value next step is an explicit sparse radial contraction plan with
orbital versioning. The discrete Coulomb map and its quadrature adjoint must
remain consistent: exchanging its two density arguments is not generally an
exact operation on a finite grid. Parallelizing one Gauss–Seidel sweep or
reusing stale fields would change the iteration. Such changes need matched
physics, derivative and convergence evidence before performance claims.

## Reproduction

The retained drivers select source through `PYTHONPATH`; use an archived source
checkout for the baseline and separate output directories/cache directories.
For example, from the selected source tree:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_df_solver.py --z 6 10 54 98 --repeats 4 --out /tmp/df-panel
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_shooting_radial_kernels.py --out /tmp/radial-kernels
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_shooting_observables.py --out /tmp/shooting-observables
```

For the baseline kernel run, invoke the new kernel driver by absolute path
while `PYTHONPATH` points to the baseline source. All generated outputs are
retained locally under ignored `raw/`. `python summarize.py` rebuilds the compact
results and hashes when those outputs are present. Raw files are not required
for normal package tests and are not committed.
