# Singleton-bucket compression: matched result

**PASS.** Four fresh workers in old/new/new/old order reproduce the complete
A1595/B2574 angular construction, all five symmetry blocks per arm. Every
candidate connection, determinant, radial key, retained pair, table CSR array,
sparse projection array and signed synthetic-integral action is bitwise
identical. The 40 recorded block identities comprise the first worker's ten
baseline records and 30 cross-process comparisons. All 40 cache checks,
signed action/bilinear checks and absolute-amplitude negative controls pass.
The largest direct-bilinear error is 5.69×10⁻¹⁴; the smallest negative-control
difference is 22.08.

| Measurement, seconds unless specified | Old 1 | New 1 | New 2 | Old 2 |
|---|---:|---:|---:|---:|
| Complete ten-table construction | 56.823 | 54.636 | 54.732 | 56.862 |
| Connection iterator pull, approximate stage | 6.142 | 3.424 | 3.407 | 6.160 |
| Complete ten sparse projections | 2.760 | 2.764 | 2.837 | 2.800 |
| Complete angular worker | 60.971 | 58.980 | 59.153 | 61.038 |
| Whole-worker peak RSS, MiB | 805.992 | 794.656 | 766.441 | 819.062 |

With two samples per variant, mean full table time is **56.843→54.684 s**,
a **1.039× ratio / 3.80% reduction**. The two adjacent-pair ratios are
1.040× and 1.039×. Mean connection pull time is **6.151→3.415 s** (1.801×),
but downstream coefficient construction dominates total table time. Mean
whole-worker peak RSS is **812.527→780.549 MiB**, a **3.94% reduction**.
Projection time remains about 2.8 s. Source/input guards, one-CPU/thread limits
and all host load guards pass. Two samples per variant do not establish a
confidence interval or performance across other workloads.

The complete retained payload remains unchanged: **26,418,669 nonzeros**,
**332,210,280 table CSR bytes** and **3,977,122 candidate connections** across
the ten blocks. Singleton bucket lists and keys can be omitted from intermediate
storage because they contribute only the explicitly retained diagonal. No
physical coordinate, signed term or determinant connection is removed.
Connection timing uses 1024-coordinate batches; its stage split includes
timer/batch overhead and excludes hashing/downstream work. RSS is a process
high-water mark, not an allocation trace for one block.

The old worker injects only the literal `_connected_pairs` function archived
before this increment. Preparation verifies that every other angular AST node,
including the typed CSR builder, is identical. The [contract](contract.json)
binds all 358 actual Python sources, complete model/basis maps, before source
and resource limits. The [raw result](actual/result.json), four worker receipts
and [standard-library review](review.json) preserve the detailed evidence.
The result-only review passes **19/19 grouped checks** without invoking a
scientific worker. Driver, contract, sealed receipts and the preceding
increment's evidence remain unchanged.

The [preceding increment's result](../../physical_next_decision007/benchmark001/RESULTS.md)
is separate source-bound context. No cumulative or cross-run speed ratio is
claimed. This panel includes no radial Hamiltonian, CI solve, parallel scaling,
end-to-end speedup, complete 4f¹² energy or experimental-accuracy result. Root
owns the larger-class admission and broader documentation review. Existing
exact-representation and matched-performance rules cover this result; no new
general insight is promoted from this local panel.
