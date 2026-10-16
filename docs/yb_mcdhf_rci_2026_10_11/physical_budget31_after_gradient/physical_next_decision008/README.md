# Compact spectator indexing with exact diagonals

Current source stores a spectator bucket's first row as a shared integer and
allocates its list only on the second occurrence. It then discards singleton
keys and emits each diagonal explicitly. Row-local sorting, complete physical
spaces and all off-diagonal connections are retained. Typed angular CSR from
[the preceding increment](../physical_next_decision007/README.md) is unchanged.

All **232 affected tests pass**: 80 independent graph/storage/projection
controls, plus 152 public density, state-average, gradient, response, static
operator and snapshot checks. Isolated and mixed configurations, duplicate
determinant rows, vacuum, high-bit masks, frozen cores, triangular order,
signed DC/Breit and complex phases are covered. [Test receipt](test_receipt.json).

The [complete A/B ABBA result](benchmark001/RESULTS.md) passes all 40 block
records, including 30 cross-process comparisons, with unchanged candidate,
key, CSR, projection and signed-action identities. Two fresh processes per
implementation pass every load guard:

| Angular-stage measurement | Before008 | After008 |
|---|---:|---:|
| Mean table construction / s | 56.8426 | 54.6842 |
| Mean connection-iterator pull / s | 6.1512 | 3.4153 |
| Mean whole-worker peak RSS / MiB | 812.527 | 780.549 |

Table construction decreases 3.80% and process peak RSS decreases 3.94% for
this complete existing workload. The 26,418,669 retained nonzeros and
332,210,280-byte table payload remain unchanged. This successor compares
007's row-stream/typed-CSR implementation with 008 only; it does not multiply
ratios across runs or claim whole-solver/parallel acceleration.

The [same complete4f12 graph](full_f12_connections001/README.md) also passes.
All **43,720,175** ordered connections and their complete SHA256 match007
exactly, along with raw determinant order, every row/diagonal and replacement
counts. The single source-bound trial used the same CPU28, one thread,
4 GiB address-space cap and300 s limit, after the ABBA workers had exited.

| Largest full B+4f12 graph | Before008 /007 | After008 |
|---|---:|---:|
| Peak RSS / GiB | 3.053 | 1.917 |
| Retained spectator buckets | 16,684,544 | 2,675,248 |
| Retained map/key/list payload / bytes | 2,834,499,496 | 436,174,880 |

The new trial took46.05 s; the prior trial took53.05 s. These are single
diagnostic graph costs, not paired controlled timing. RSS includes the
construction/compaction highwater; the smaller retained-map payload excludes
that transient, masks, caches and allocator arenas. The inferred14,009,296
implicit singleton buckets reconstruct the original U/K/V census exactly.
No angular coefficient table, CSF amplitudes, H/CI or energy was evaluated
for this large class.

The largest original graph has 16,684,544 buckets and 23,456,895 memberships,
so at least 9,912,193 buckets contain one row. Their list payload alone is
at least 872,272,984 bytes on this interpreter. This motivates the exact
storage change, rather than excluding configurations or small amplitudes.
Actual construction peak must include both dictionaries during compaction.

Complete4f12 Hamiltonian/CI construction and its spectral contribution remain
uncomputed. Stored angular coefficient tables, native selected-root ownership
and full orbital response require their own qualification. The original
stationarity, three line-FD failures and experimental gaps are retained.
The next physical target is a paired fixed-orbital full4f12 RCI comparison at
original accepted13. Its coefficient-table/H assembly and state-identity
budget must use these measured construction peaks; graph success does not
admit the complete H/CI workload. [Increment receipt](increment_receipt.json)
and [documentation disposition](documentation_impact.json) retain the source
and validation chain. No new general insight is promoted beyond the existing
exact-representation and source-bound measurement rules.
