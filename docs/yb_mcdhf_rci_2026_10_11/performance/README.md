# Native MCDHF radial action reuse

This increment changes only `CSFOrbitalFunctional.gradient_from_weights`.
For each exact Coulomb source pair, the gradient now applies its P/P and Q/Q
potential to each required radial spinor once. The one-body Dirac actions are
also reused within this evaluation. Signed RDM terms retain their original
accumulation order, physical coefficients, selection, and thresholds. Action
arrays are discarded after the source pair; no action crosses an orbital trial.

The [prospective contract](contract.json) fixes a native two-electron He,
12-CSF even J=0 coherent ensemble, `ns=120`, `ks=9`, point nucleus, and the same
cell Dirac–Coulomb Hamiltonian. This is a real MCDHF functional kernel probe;
it does not estimate a Yb spectrum or claim an end-to-end improvement.

The [measured result](gradient_probe001/result.json) shows:

- Full ensemble and signed transition radial gradients agree bitwise with the
  old contraction. Selected gradients differ from the full-gradient subset by
  at most `2.03e-20` Ha, owing to the pre-existing selected-coordinate grouping.
- A metric-tangent, re-diagonalized ensemble directional derivative differs
  from the analytic gradient by `9.29e-10` Ha; the predeclared limit is
  `2.48e-7` Ha. The live tests additionally restrict the direction to the
  same positive-energy Dirac subspace.
- The 31 Coulomb source pairs are unchanged. Component matrix-vector actions
  decrease from 1,728 to 224. This count is an exact redundancy measurement.
- Alternating warm-kernel observations have medians 13.03 ms before and
  11.02 ms for the candidate. The machine load was 9.21, above the contract's
  2.0 guard; no controlled speedup or parallel-efficiency claim is certified.
  These short correctness probes did not acquire the shared production timing
  lock. The result's `speedup_claim` is therefore null.

The [profile](gradient_probe001/profile.txt) places CI matrix construction
ahead of this gradient in the same finite model: about 20 ms and 14 ms,
respectively. CI pair potentials still call the full-cell
`_convol_case1_t`, while this gradient and the scalar Slater integrator select
the exact semiseparable kernel at `ns >= 96`. Unifying that producer is the
next concrete performance experiment, with all four P/Q component identities,
mixed signed integrals, complete CI roots, and constrained gradients required
before adoption. It is deliberately outside this patch.

The current MCDHF functional also calls `ci_solve` without a root-count request.
With `n_states=None`, its `auto` solver therefore computes the complete dense
spectrum even for more than 1,500 CSFs. A large state-average target needs
requested roots **within every named J/parity block**, validated against the
same target identities and stationarity metric. A global lowest-root cutoff
can omit a requested excited symmetry and is not a valid replacement. This
consumer contract is another important scalability gap; public Davidson
support alone does not show that MCDHF uses it.

[Live tests](../../../tests/test_mcdhf_gradient_reuse.py) exercise `ns=35` and
`ns=120`, both cell-kernel branches, full/selected/transition literal
identities, accepted-state diagnostics, action invalidation after a changed
trial, exact once-per-spinor reuse, and independent positive-energy,
metric-tangent finite differences. Together with fixed state-average tests,
30 tests passed in 2.81 s. Charge-sector, CI response, and optimizer-history
consumers add 28 tests passed in 2.17 s. Ruff passed for the changed function,
tests, and probe. These are numerical implementation gates, not experimental
accuracy evidence.

`source_before_action_reuse/` and `source_after_action_reuse/` freeze the source
epochs; [the function-only patch](gradient_action_reuse.patch) separates this
change from concurrent determinant-cap propagation in the same module. The
source manifests include full-file and function-AST hashes. Native Yb's
already frozen calculation source remains distinct.

The local GPU driver detects an RTX 4090, CUDA compute capability 8.9, with
FP32/FP64 ratio 64. NumPy 2.2.6, SciPy 1.15.3, and Numba 0.65.0 are installed;
CuPy and `numba-cuda` are absent. `numba.cuda.detect()` sees the device, but
`cuda.is_available()` is false because `libNVVM` is unavailable. No GPU kernel
or GPU timing was run. CPU/GPU sharing should retain common physical data and
operator definitions, with separate suitable execution kernels; a GPU label
alone provides no precision or throughput evidence.

Replay the original bounded probe with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python docs/yb_mcdhf_rci_2026_10_11/performance/probe_gradient.py \
  --output docs/yb_mcdhf_rci_2026_10_11/performance/gradient_probe002
```

The original observation compared the pre-change production contraction with
the proposed action-reuse function. After adoption, replay compares production
reuse with the same candidate; the frozen old function and live literal
oracle retain the original identity comparison.
