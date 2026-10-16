# Complete 4f12 graph with singleton compression

The closed successor passes the exact [decision007 graph](../../physical_next_decision007/full_f12_connections001/README.md):
182 NR configurations, 738 jj patterns, 92,715 fixed-M determinants and
5,717 CSFs by integer multiplicity for 69-electron B+4f12, sector `(1,+1)`.
The [contract](contract.json) binds the original model and the closed prior
result. Its gate requires all **43,720,175 ordered pairs**, the same native
determinant ordering and the same entire ordered-coordinate SHA256 digest.

Only spectator storage changes: first occurrences share the row index;
repeated keys promote to lists; singleton keys are omitted before row streaming
and each diagonal is inserted explicitly. The live generator map therefore
contains only buckets of length at least two. Reported implicit singletons
are inferred as prior original U minus actual retained U. Reconstructed
membership and ordered-visit counts must equal 23,456,895 and 92,428,751.
No production singleton counter or singleton mask array is introduced.

The root-sealed trial retains **4 GiB address space, CPU 28, one thread and
300 seconds**. Peak RSS includes the integer-first construction and dictionary
rebuild transient; the retained-map owned payload has a narrower scope.
Preparation is standard-library integer work only. No graph execution occurs
without the root seal. No angular coefficients, CSF vectors, projection,
Hamiltonian, CI, radial solve or NPZ are evaluated. Separate single-run graph
times do not establish controlled speedup, full-H memory or experimental accuracy.

The [actual result](actual/result.json) retains all **43,720,175 ordered
connections**, the same raw determinant order and exactly the same pair digest
`9a49475f11b27a0889a1d187c9a5a18a445db0e92510688362f0b131d4a290e4`.
Zero-, one- and two-replacement counts remain 92,715, 1,206,876 and 42,420,584;
all 92,715 rows and diagonals are present.

| Structural graph storage | Decision007 | Decision008 |
|---|---:|---:|
| Live bucket count | 16,684,544 | 2,675,248 retained |
| Map + keys + lists / bytes | 2,834,499,496 | 436,174,880 |
| Whole-run peak RSS / KiB | 3,201,428 | 2,010,012 |

The same-graph owned payload falls **2,398,324,616 bytes (84.61%)**. The
observed process peak falls from 3.053 to **1.917 GiB**, including construction
and dictionary-rebuild transient. Both trials used a 4 GiB address-space cap;
the new RSS observation does not certify a 2 GiB address-space fit. Shared row
integer objects, native caches and allocator overhead are excluded from the
owned-payload row and included where resident in process RSS.

The retained map contains 9,447,599 memberships and 78,419,455 ordered bucket
visits, with bucket lengths 2 through 121. The inferred 14,009,296 singleton
buckets reconstruct the original U/K/V exactly. Total elapsed time was
**46.05 s**, including 0.93 s determinant enumeration and scalar validation/
streaming hashing; graph work took 45.12 s. This single diagnostic cost is not
a controlled timing comparison with decision007. The
[result-only closure](receipt.json) verifies sealed identity and exact graph
gates without rerunning science. No new Hamiltonian or energy was evaluated.
