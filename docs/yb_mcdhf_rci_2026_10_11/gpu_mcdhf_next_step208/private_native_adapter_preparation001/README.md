# Private native31 gradient adapter preparation

This is a SOURCE/JSON preparation, not an executed CPU/CUDA adapter or an
MCDHF derivative, stationarity, GPU or performance certificate. The public
source, defaults, dependencies and tests are unchanged.

[native_adapter.py](native_adapter.py) connects the complete native31
`full_normalized` signed RDM table to the one frozen
[batch executor](../private_batch_preparation001/batch_executor.py). It retains
the original one-body updates, two derivative orientations, source insertion
order, per-source action cache and signed additions. Public `_density_sym`,
`get_coulomb_factors` and `_banded_to_full_add` remain the numerical consumers;
no angular coefficients, CI algorithm or RDM normalization formula is copied.
The executor supplies the ordered PP/QQ/PQ/QP band scans with native FP64,
no fastmath/atomic reduction, then the existing CPU actions and additions run.

The adapter returns the original full-normalized **raw radial gradient**. The
CPU raw-h quotient correction, orbital constraints, Lambda/tangent connection,
original inverse metric and preconditioned gradient remain separate. This
preparation does not certify the raw-h-corrected or constrained gradient merely
because the raw kernel is represented.

## Actual source and signed-table binding

The reference is the closed [native31 gradient result](../../screened_f_next_step208/gradient_proposal001/execution_preparation001/actual/result.json)
at f043 source, Z70/69 electrons/Fermi A175/DC, ns123, complete1595 CSFs,
31 radial orbitals and3689 coordinates. The original five tracked UIDs and
equal0.2 weights remain fixed. The reference's438 controls passed, but its
gradient is HF partial; finite-CI-response qualification, expanded MCDHF
optimization and original stationarity remain false. The29-state controls and
older SD GPU results are not31-state qualification.

[actual_signed_union.json](actual_signed_union.json) records the unfiltered
closed JSON row order, coefficient hex identity, derivative incidence and source
first-appearance order. It distinguishes the three actual tables:

| Table | Source jobs |
|---|---:|
| retained |706|
| full minus retained |688|
| full normalized |706|

Their sum is2100 public pair requests across three gradients. It is not evidence
of700 jobs per table. The first source-preparation assumption of700 is preserved
as an unexecuted [metadata failure](historical_metadata_revision001/failure.json),
with its contract and driver. No numerical model or gate changed.

The full table has19589 rows (57 one-body,19532 Coulomb),9824 negative rows,
496 unique source pairs,706 source jobs,2824 four-component jobs and39064
signed derivative incidences. All ranks0–6 and every row remain. The mechanical
31-orbital upper bound3472 source jobs is reported separately, not substituted
for this actual union.

[native_binding_declaration.json](native_binding_declaration.json) binds the
f043 module bytes, complete signed basis, objective, native metric and closed
reference result. The retained public H owner validates the physical input
identity before factors or density construction. H ownership alone does not
prove mutable CI eigenpairs, tracked UIDs, equal weights or arbitrary supplied
RDM tables; complete live residual/Gram/spans and target/normalization gates
remain external prerequisites. Loaded-source bytes and the H matrix SHA are
separate from geometry identity. A new support/order needs a new owner.

## Work and admission bounds

[real_union_mechanical_preflight.json](real_union_mechanical_preflight.json)
uses only JSON descriptors and integer arithmetic. Actual workspace arrays
are unread; orders9/9 are bounds, not measured component orders. At a128-source
chunk the actual706-job union needs six scan and six row launches,992 public
density-band requests and28 factor requests. Requests do not imply cache misses
or new producers. The upper payload is61,409,056 device bytes and196,894,048
host bytes, including copies/scratch and CPU dense postprocessing. Downloaded
bands per gradient are bounded by25,009,344 bytes. Runtime/context/JIT/RSS and
actual device admission are additional; no resident allocation or transfer ran.

Numerical construction explicitly admits complete finite real binary64 DC
inputs, all31 keys, valid structural RKB tails and declared source/target/metric.
It rejects partial keys, changed support/order, complex/float32/casting,
nonfinite values, nonzero padding, unknown operators/negative ranks,
Breit/QED/recoil, undeclared unequal component extents and stale identities.
There is no install, backend fallback or small-term threshold.

## Preparation actually completed

[prepare.py](prepare.py) passed56 SOURCE/JSON/AST/descriptor controls under a
clean stdlib process with1 GiB/30 seconds. An import blocker rejects numerical
libraries. Public h updates, signed action/addition bodies and orientation order
match by AST after removing counters and the local variable alias. Literal
frozen file imports are used; no AST/text execution assembles physical code.
Ruff passed the adapter and preparation driver.

The [proof](preparation_verification.json) records zero NumPy/Numba/Atomx/CUDA
imports, NPZ reads, array allocations, H/CI/gradient/Dirac/optimizer calls,
numeric compilation, GPU execution and benchmarking. Declared NPZ hashes come
from the closed result JSON; those bytes were not inspected or independently
verified here. The original356 public-source map remains unchanged.

The next boundary is a frozen native31 constrained finite-CI-response proof,
then a separately reviewed exact qualification driver comparing same-point
public CPU1/CPUparallel/CUDA full signed raw/tangent/native-metric gradients,
all owned H/CI/UID/normalization gates and explicit producer counts. No numerical
launch is authorized by this preparation. Original scientific failures and the
failed warm-load timing certificate remain unchanged.
