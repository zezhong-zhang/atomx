# Frozen Yb68 g-band native refinement003

This manually authorized successor preserves [001](../yb_high_l001/RESULTS.md) and
[002](../yb_high_l002/RESULTS.md).002 failed the complete κ4 <50 Ha band ordinary
S-dual residual gate: 1.8192200019014654e−7 Ha against the unchanged1e−8 Ha gate;
κ−5 was not reached. This run tests whether existing native bordered refinement
inside the complete original electron span qualifies both complete bands.

The historical metadata marks19 nuclear orbitals as DHF: Z70/68e/[Xe]4f14,
A175 Fermi, original B-spline ns123/R60.53782210642609. The19 archived rows and
all15 saved geometry fields are restored as owned immutable arrays. Historical
convergence markers remain historical; current occupied Fock residuals are
diagnostics. The missing original driver SHA81fd41dd... remains an explicit
provenance gap. This is neither a fresh converged DF result nor the paper's
171Yb+/40GTO/R500/RCCSDT model, nor a high-l correlation/line response.

[contract.json](contract.json) freezes the same bank, nucleus, raw native V^N H/S,
whole electron branch, complete E<50 Ha selection and all original gates.
Each selected root uses `_refine_eigenpair` in the fixed complete electron span
(expected119, cap123 columns) within the original239 boundary coordinates.
The native function uses the lower-defined projected H/S, at most2 bordered
Newton solves, a ≥.99 original-column overlap and ≤.25 nearest-original-gap
energy movement. Lifted vectors remain in that original electron span. Every
original column overlap and root separation, original/refined E/V and full
ordinary S-dual residual are saved. The original lazy refinement closure is
disabled on the new result; no unbudgeted later root mutation is admitted.

The native refinement minimizes projected FP64 Euclidean residual, which does
not guarantee the full original-H S-dual residual gate. Both declared g channels
are measured before an aggregate failure of that unchanged gate; each failed
check remains false. Root identity, band membership/order, full joint S-Gram,
metric projector, Gauss normalization and resource failures stop immediately.
If the full gate still fails, stop further strategies and preserve the failure.

Caps: CPU27, one thread,8GiB address space,600s inner/615s outer; load1≤16 and
MemAvailable≥10GiB. Nine native Fock/raw H/S builds,≤171 direct and264 exchange
entries, two full Dirac spectra (≤6 LAPACK attempts),≤246 root refinements/
492 bordered solves (dimension≤124),≤11 diagnostic metric solves. Original and
refined residuals share each g-channel batched metric solve. No SCF, CI,
optimizer, altered cutoffs, backend, fit or tolerance. Source-only default mode
and Ruff must pass before sealing; a single explicit `--execute` owns `actual/`.

Results and qualification are recorded separately after the admitted execution.
