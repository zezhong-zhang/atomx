# Five-state derivative result

The complete native Yb experiment finished all10 H/CI points,10 full signed
state gradients and50 complete symmetry-block diagonalizations. It retained
all31 orbitals/3689 coordinates, both full A1595/B2574 models, original f043
source, nucleus/projectors/metric and signed five-state tracking.

**The full derivative qualification failed and remains failed.** Of677
controls,674 pass. The ordinary FD panel passes20/20 absolute-state,
4/4 reported-EOL and13/16 line comparisons. All three failures concern
the6s→5d3/2 line:

| Model | Fixed step | Absolute derivative discrepancy / Ha | Original limit / Ha |
|---|---:|---:|---:|
| A | 0.002 | 1.30353591e−8 | 1e−8 |
| B | 0.002 | 1.58719027e−8 | 1e−8 |
| B | 0.001 | 1.08696818e−8 | 1e−8 |

No step scan, retry, tolerance change or Richardson substitution was used.
The two-step Richardson values are separate diagnostics. Truncation,
finite-precision energy evaluation and constrained reconstruction have not
been uniquely separated by this run. A small FD discrepancy is not a
spectroscopic correction or an experimental error bound.

The five actual-norm signed gradients close against the original aggregate
gradient: maximum tangent-vector differences2.19e−11/1.75e−11 and
preconditioned-vector differences1.81e−14/2.89e−14 for A/B. The independent
raw-H/ket-quotient directional correction is at most5.23e−14 Ha. These
measurements and the [independent exact-CSF anchor](independent_anchor001/EARS.md)
give no evidence for a large missing derivative factor, phase or normalization
term in this tested direction. They do not qualify the full orbital space.

The line derivatives survive cancellation of much larger absolute-state
responses. At R0, their analytic values along the fixed direction are:

| Excitation | A dX/dt / Ha | B dX/dt / Ha |
|---|---:|---:|
| 5d3/2 | −0.000886500250 | −0.000759483511 |
| 5d5/2 | −0.002213491753 | −0.002135251568 |
| 6p1/2 | −0.003263785447 | −0.003074643480 |
| 6p3/2 | −0.002946871813 | −0.002634153307 |

The individual state derivatives are about−0.164 to−0.170 Ha. The sum of
absolute31-orbital line contributions divided by the absolute net line
derivative is10.40/11.90 for D3/2 in A/B; other lines give2.53–4.79.
This accounting depends on the fixed orbital basis and chosen direction.
It does not uniquely attribute an experimental discrepancy to a shell or
prove that an omitted correlation class dominates it. Every signed orbital
contribution remains saved, including the terms that cancel.

Only the weighted EOL gradient defines the current shared-orbital stationary
target. Individual state gradients need not separately vanish. Original
stationarity remains failed, and no experimental accuracy is certified.
The next physical choice should compare the measured state-balance response
with explicitly missing correlation classes, starting with the complete4f12
double-hole family; a numerical derivative refinement is a separate task.

The [complete4f12 integer census](missing_4f12_budget.json) now fixes the
size of that missing class at the **same31 radial orbitals**. All82
Pauli-allowed NR occupancies add43,112 target CSFs. A becomes173 NR/44,707
target CSFs; B becomes182 NR/45,686. The census independently reproduces
both old full/selected CSF counts and all five saved native B raw-block
dimensions. It performs no Hamiltonian, CI or energy calculation.

The largest proposed native magnetic-determinant block has91,761/92,715
rows for A/B. A dense FP64 raw H alone would occupy67.36/68.77 GB, before
projection, angular tables or workspaces. **These are dense counterfactual
budgets, not measured sparse-path memory or a lower bound.** Increasing radial
cutoff alone does not generate the absent excitation category.

Root's [source-scope correction](source_scope_review.json) confirms that
`exact_block_matrix(as_operator=True)` already uses sparse determinant H,
sparse exact CSF projection and a LinearOperator action; `ci_solve` already
supports Davidson selected-root solving. Reuse these native implementations,
then budget their full angular-table construction and qualify the larger
model against existing dense controls. The present retained H snapshot accepts
only complete dense DC spectra, so an iterative result cannot silently reuse
that ownership certificate. Computing selected eigenstates is distinct from
truncating the physical CSF space. Actual full4f12 sparse allocation/work,
state tracking and operator ownership remain unqualified by this census.
The orbital-gradient density constructor still requests a dense projection
by default; reuse of the existing sparse projection option there requires
matched signed-density/gradient validation before a larger MCDHF trial.
Neither class size nor selection-rule relevance establishes its contribution
to the experimental gap; that still requires an owned paired calculation.

The [full execution report](execution001/RESULTS.md),
[original result](execution001/actual/result.json) and
[independent small-system result](independent_anchor001/result.json)
retain the evidence. The [original predeclaration](README.md) is an immutable
input to the execution and remains unchanged. Timing133.72 s and peak RSS
2.05 GiB are uncontrolled diagnostic costs, not speedup measurements.
