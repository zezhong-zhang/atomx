# Full A/B angular construction: matched result

**PASS.** Four cold-process builds in old/new/new/old order preserve every
candidate connection, determinant, radial key, retained pair, table CSR array,
sparse projection array and signed synthetic-integral action bitwise across
all ten complete A1595/B2574 symmetry blocks. This gives 30 comparisons against
the first worker, plus its ten recorded baseline identities. Every worker also
passes ten cache-identity checks, ten independent signed action/bilinear checks
and ten absolute-amplitude negative controls. The largest direct-bilinear error
is 5.69×10⁻¹⁴; the smallest negative-control difference is 22.08.

| Measurement, seconds unless specified | Old 1 | New 1 | New 2 | Old 2 |
|---|---:|---:|---:|---:|
| Full ten-table construction | 62.681 | 56.399 | 57.299 | 63.823 |
| Connection iterator pull, approximate stage | 7.397 | 6.367 | 6.613 | 7.447 |
| Full ten sparse projections | 2.760 | 2.760 | 2.776 | 2.774 |
| Complete angular worker | 67.121 | 60.366 | 61.291 | 68.264 |
| Whole-worker peak RSS, MiB | 969.465 | 795.371 | 806.027 | 982.824 |

With two samples per variant, mean table time is **63.252→56.849 s**, a
**1.113× ratio / 10.12% reduction**. The two adjacent-pair ratios are 1.111×
and 1.114×. Mean whole-worker peak RSS is **976.145→800.699 MiB**, an
**17.97% reduction**. Projection time is unchanged within these samples.
CPU affinity, one-thread environments, source/input guards and all host load
guards pass. No confidence interval or broader workload extrapolation follows
from two samples.

The retained payload is exactly unchanged: **26,418,669 table nonzeros**,
**332,210,280 CSR bytes** and **3,977,122 candidate connections** across the
ten blocks. The reduction concerns construction representation and peak
residency; it does not remove physical coordinates or angular entries.
Connection pull timings use 1024-coordinate batches; the stage split excludes
hashing and downstream work but includes timer/batch overhead. RSS is a process
high-water mark, not a per-table allocation trace.

Old workers inject only the literal archived `_connected_pairs` and
`_coefficient_table` functions; every other dependency uses the same actual
current source. The [contract](contract.json) binds all 358 Python sources,
the full model/basis maps, before archive and resource limits. The
[raw result](actual/result.json), four worker receipts and [independent
standard-library review](review.json) retain hashes and per-block details.
The review passes 19/19 grouped checks; it does not rerun scientific work.

This qualifies cold angular construction on the complete existing A/B models.
It contains **no radial/Hamiltonian assembly, CI solve, parallel scaling,
end-to-end speedup, complete 4f¹² energy or experimental-accuracy result**.
All previous physical and finite-difference failures retain their original
status. Root owns the broader source/documentation impact review and next
physics admission; no new general insight is promoted from this local panel.
