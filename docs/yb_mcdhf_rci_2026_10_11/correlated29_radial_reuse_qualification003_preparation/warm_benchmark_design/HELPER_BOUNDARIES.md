The old qualification003 chain is immutable and has no reusable prepare-point
interface. This proposal makes a new, explicit private benchmark-helper epoch.
It will not execute extracted AST/text. AST may only audit the submitted literal
source and its retained formulas/gates. No numerical work is authorized here.

The four functions below move the existing consumer logic into one owner. There
is no second Hamiltonian, CI, density, gradient or metric implementation. The
driver calls public `CSFOrbitalFunctional.evaluate` for every new trial and its
same-trial gradient. Source/operator maps stay those actually qualified in003.

| Function | Original run.py boundary | State and responsibility |
| --- | --- | --- |
| `prepare_point(contract, gate, close, raw_hs)` |353–358 and361–421 | Public model/functional/tracker construction; ordered1046-CSF/29-orbital identity, original L(raw h), S,3451 coordinates/projectors/inverse blocks and five-target reference. Return `PreparedPoint`. |
| `validate_trial(point, ci, centered_calls, gate, close)` |427–433 and439–453 | Genuine same-point H owner retrieval; five complete sector origins, original-H complete column2-norm residual1e−8, Gram1e−10, read-only storage and stale-point refusal. Return `ValidatedTrial`. No spectral solve. |
| `finish_checked_chain(point, trial, gradient_result, output, gate, close)` |457–458,460–512,548–568 | Original radial→raw/tangent/originalmetric pg projection and all original g/pg/weights/UID/RDM/H/raw h/S/historical-root checks. Preserve signed terms. Write same metadata and compressed archive, fsync, reload and exact array checks. Return paths and validation/archive timing. |
| `compare_saved_chains(contract, ordinary_paths, candidate_paths, check)` |572–632 | Original complete core arrays, all1046 signed cluster projectors/partition/sector zeros, complete signed RDM, UID and physical owner/origin checks. Paths are explicit instead of implicit HERE/actual. No eigensolve/SVD. |

The literal source for the3451 raw/tangent/preconditioned action moves unchanged
from the old chain460–477 into `finish_checked_chain`; it is not rewritten or
maintained in two new adapters. Its original ndarray operations and accumulation
order are retained. Likewise the full1046-span algorithm moves unchanged from
`compare`. The preparation proof will compare these exact statement sequences
and the complete check-name/limit inventory. Every signature/path transformation
is listed explicitly; no runtime source rewriting or dynamic exec is allowed.

`PreparedPoint` holds csfs, waves, workspace, nucleus, public functional, original
metric/layout/inverse blocks, complete saved baseline/gradient arrays, five UID
bindings, captured raw-h/S controls and source identity. Mutable functional
diagnostics are consumed immediately after the corresponding gradient and
copied into that chain's metadata. Original anchors and waves are never replaced.
`ValidatedTrial` holds that point identity, genuine current CI object, original
read-only H and complete sector/origin records. It cannot accept arrays as a
substitute owner. Driver retains the previous CI until the next one is produced
so it can check fresh owner/CI objects without interpreting recycled object IDs.

The driver uses two timers only around the two native public calls: first
`functional.evaluate(point.waves, gradient=False, ci_result=None)`, then
`functional.evaluate(point.waves, ci_result=trial.ci, gradient=True)`. Each timer
stops immediately on return. Original-H ownership/residual checks, metric/UID
checks, all serialization and paired comparisons are outside these primary
timers, with their own disclosed costs. Public same-point ownership/tracking
work inseparable from evaluate remains included symmetrically. Source/input
hashing, imports, constructor/angular setup and cold compilation are separate.

The old ordinary-only514–547 extra five-eigensolve replay is excluded from both
warm chains, expressly rather than hidden in an extraction. That independent
new-centered-epoch replay already passed in closed003 and is a prerequisite.
Every repeated same-worker H and energy spectrum will instead compare bitwise
with its own newly archived cold centered chain, with no additional eigensolve.
The old unshifted energy comparison remains labeled source evolution, at its
original1e−8 tolerance. No earlier failed epoch is reclassified.

Instrumentation and source/load/resource admission remain driver concerns.
Every trial creates a fresh H/CI through ci_result=None, hence a new evaluator
with its own density/pair/integral LRU. The same geometry kernels/factors and
Numba code are intentionally warm. Initial h/S is7 per worker and CI-internal
h/S another7 only on the first H; seven chains per worker retain35 dense block
solves,7 H/CI/gradients,14 tracking resolves and70 SVD. Aggregate caps remain
14 H/CI/gradients,70 eigensolves,28 h/S,28 resolves,140 SVD and30 overlaps.

Preparation will freeze ordinary/candidate source maps, the literal helper and
driver, inherited input/gate map, new-check inventory and exact calls/resources.
Only stdlib syntax/AST/source checks and Ruff may run before root's exact seal.
The numerical launch also waits for both history32 pilot processes to close and
for root to reserve a separate controlled window. No helper code has yet run.
