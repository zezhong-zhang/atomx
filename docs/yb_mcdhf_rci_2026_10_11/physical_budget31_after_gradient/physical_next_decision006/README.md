# Native sparse CI and orbital-density qualification

The existing native sparse representation passes matched controls in the
complete A1595/B2574 Yb⁺ spaces. The current orbital-density constructor now
uses its existing sparse CSF projection option. This increment retains the
same physical spaces and does not report new experimental accuracy.

## Complete operator and selected eigenpairs

The [once-only execution](execution001/RESULTS.md) built two fresh native
sparse Dirac–Coulomb operators from the frozen f043 source, literal accepted13
orbitals, original A175 nucleus, no-pair P and metric. The owned dense
A_R0/B_R0 spectra from decision004 were read as controls; no dense H was rebuilt.
All five symmetry blocks and all CSF rows were retained. Every saved full
eigenvector was tested against the new operator; Davidson then computed the
lowest two roots separately in each complete symmetry block.

All **273/273** declared controls pass:

| Maximum discrepancy | Measured | Declared limit |
|---|---:|---:|
| Sparse versus dense signed H action / Ha | 1.64e−11 | 1e−8 |
| Full reference eigenpair residual, max component / Ha | 1.82e−11 | 1e−8 |
| Twenty selected energies versus dense / Ha | 2.19e−11 | 1e−8 |
| Fresh selected residual, L2 / Ha | 9.11e−10 | 1e−8; solver tolerance 1e−9 |
| Phase-aligned selected wave coefficients | 4.64e−9 | 1e−8 |
| Selected rank-one projectors | 5.04e−9 | 1e−8 |

This is a fixed-basis operator/solver qualification. `ci_solve(n_states=k)`
performs a global selection after block solves and need not retain every desired
sector. Existing full-spectrum state averages, tracking, orbital response and
retained-H snapshots retain their own contracts. No partial result was passed
off as a complete optimizer or snapshot. The initial shell launcher failed
before Python because `taskset -c27` was invalid; its receipt is preserved.
Only argument spacing was corrected, and the scientific driver ran once.

## Shared density source change

`CIRadialDensity` now requests `csf_block_matrix(..., sparse=True)`. Its signed
amplitudes, column ordering, radial-key algebra and existing coefficient policies
are retained. The [actual Yb before/after comparison](density_sparse001/RESULTS.md)
binds both dirty-current source versions and the same complete saved A/B
eigenstates. It passes **199/199** controls across 24 contractions: equal-weight
five-state densities, individual signed states, same-sector transition pairs
and random signed bilinears. Every union radial key was compared.

The largest coefficient difference is **7.11e−15**, below the predeclared
1e−12 gate; no radial key was lost or added. Seventeen unfiltered coefficients
change sign at cancellation-level magnitudes below 7.8e−21; those values are
retained. Floating-point contraction is not bitwise identical.

| Projection payload | Dense before / bytes | CSR after / bytes |
|---|---:|---:|
| A, all five sectors | 19,988,256 | 219,864 |
| B, all five sectors | 48,996,952 | 475,304 |
| Total | 68,985,208 | 695,168 |

These are projection arrays only, excluding angular tables and construction
temporaries. The actual density comparison calls no Hamiltonian, eigensolver,
gradient or radial solve. The affected regression panels pass **84 tests**,
including an independent signed full-H bilinear/phase negative control and
public single-/joint-charge state-average consumers. Commands and source
identities are in [the increment receipt](increment_receipt.json).

## Complete 4f12 admission remains a separate step

The [integer-only angular budget](angular_budget.json) retains all 82 missing
NR occupancies and 43,112 added target CSFs at the same 31 radial orbitals.
It estimates the actual native sparse construction, rather than interpreting
the previous dense counterfactual as sparse memory.

Each proposed block has about 35.0–43.7 million ordered structural connections
(17.5–21.9 million triangular) before angular cancellations. Its sparse CSF
projection has a 19.6–41.1 MB payload upper bound. The global connection set,
sorted/decoded pair lists and Python coefficient triplets remain the concrete
construction bottleneck; the final sparse matrix alone does not budget their
peak. At the largest block, current connection-return live objects alone have
a **4.37 GB payload lower bound**, excluding set capacity, spectator buckets,
angular tables, caches and allocator overhead. No full4f12 Hamiltonian or energy
has been computed or admitted here. The [root budget review](angular_budget_review.json)
checks these counts and clarifies that the budget's density source-audit sentence
describes the frozen pre-increment constructor; the current CSR change is
qualified above.

The next implementation target is deterministic row-wise connection streaming,
then bounded coefficient-table construction, qualified against the same exact
signed operator. This preserves the complete correlation class. Selected-root
ownership/tracking and orbital response need separate implementation before a
larger MCDHF optimization. No fitted NIST correction, backend replacement or
physical-space reduction is introduced.

The uncontrolled costs were 78.89 s / 1.65 GiB peak RSS for sparse H/Davidson
and 85.82 s / 2.04 GiB for the density comparison. They are diagnostic costs,
not serial, parallel or end-to-end speedup measurements. In the H/Davidson
run, operator construction took 76.58 s and ten Davidson solves together
0.20 s, supporting construction as the next measured optimization target.
Original stationarity,
experimental gaps and the [three failed line-FD gates](../physical_next_decision005/STATUS.md)
remain failed. The detailed [documentation disposition](documentation_impact.json)
records this increment; existing exact-sparsity, metric and evidence rules
already cover its conclusions, so no new general insight is promoted.
