# Complete finite CC/EA state-action integration

The native solvers now accept `action_representation="state_action"`, using
the shared [ClusterAction engine](../native_state_action001/README.md).
They retain complete fixed-N/N+1 determinant spaces and original CCSD/CCSDT,
continuation, analytic Jacobian, attachment ranks and physical normalization.
Only exactly represented zeros are removed. The default remains `"dense"`.
Both paths expose the same projected and full-H residuals. This is an exact
alternative finite action representation, not a production tensor RCCSDT
implementation or the paper's 69-electron Yb calculation.

The option removes the dense excitation-generator banks and attached-sector
T matrix. Dense H, complete determinant enumeration, Jacobian or projected
operators and eigensolvers remain. The conservative NumPy admission estimate
accounts for the selected representation; Python/SciPy/dictionary overhead
is additional. `cluster_term_application_cap=2_000_000` pre-admits complete
operator/state pair counts for each scalar exponential, including zero
amplitudes. It is not a bound on all nonlinear iterations or Jacobian columns.
The original determinant/amplitude caps were not increased.

[Four integration tests](../../../../../tests/test_cc_state_action.py)
exercise independent three-electron FCI with nonzero T3, complete seven-spinor
EA matrices/physical projectors and a retained omitted-rank defect. They forbid
both dense generator constructors on the state path and reject insufficient
work caps before phase or Hamiltonian construction. Empty excitation spaces
remain valid. The shared engine has seven independent complex action tests;
the consuming core Hamiltonian/integral interfaces remain real-only.

Actual affected panel: **50 passed, no skips, 5.17 s** (validation timing),
2026-10-13 UTC, existing `.venv`, CPU 26, BLAS/OMP threads one and isolated
Numba cache:

```sh
NUMBA_CACHE_DIR=/tmp/atomx-paper-reproduction-20261013-architecture-clean \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 26 \
.venv/bin/python -m pytest -q \
  tests/test_finite_ccsd.py tests/test_finite_ccsdt.py \
  tests/test_eom_attachment.py tests/test_cc_state_action.py \
  tests/test_cluster_actions.py tests/test_scaffolds.py \
  tests/test_architecture_boundaries.py tests/test_architecture_scientific_fidelity.py
```

The [bounded measurement contract](contract.json) and [actual result](benchmark_result.json)
use the unchanged RNG194025 seven-spinor/Ncore=3 synthetic Hamiltonian.
All 35 core/35 attached determinants, 34 core amplitudes and all 34 retained
1p/2p1h/3p2h roots remain. After one warm run per path, five alternating paired
end-to-end solves run with tracing disabled; separate solves measure Python
`tracemalloc` peaks. CPU 26, single BLAS/OMP threads, 4 GiB address-space cap
and 120 s wall cap were fixed before execution. The source hashes match before
and after. The complete comparison error is at most **3.10e−13** against
the original **1e−10** gate; retained projected residual max is
**6.22e−15 Ha**. Full residual max over all 34 roots remains **0.123871 Ha**
in both paths, preserving the incomplete attachment-space physics.

| Measured finite case | Dense | State action |
|---|---:|---:|
| Excitation-generator array bytes | 666,400 | 0 |
| Warm end-to-end median seconds | 0.015765 | 0.367816 |
| Separate tracemalloc peak bytes | 880,527 | 239,058 |

The state path uses 72.85% less tracked peak memory but takes 23.33 times as
long here. Tracemalloc is not process RSS; these values are specific to this
small synthetic workload, not atomic performance bounds. The negative timing
result is retained. Production improvement needs normal-ordered linked tensor
contractions or another measured algorithm; merely replacing matrices with
Python dictionaries is insufficient. Complete arrays are retained in
[complete_results.npz](complete_results.npz); [benchmark.py](benchmark.py)
replays this comparison. Neither experimental numbers nor an external backend
enter this calculation.

Documentation disposition: update current [FSCC guide](../../../../ATOMIC_FSCC.md),
[capabilities](../../../../ARCHITECTURE_CAPABILITIES.md),
[testing guide](../../../../TESTING.md), [migration notes](../../../../MIGRATION.md)
and parent paper navigation. Preserve
the old source-only contract, native-triples failure and development receipts
as separate epochs. Existing measurement-before-adoption principles cover this
negative timing result; no new general rule is needed. [Receipt](receipt.json)
binds source and actual artifacts.
