# Complete 4f12 structural-graph trial

The bounded trial passes: the new native row iterator enumerates the
largest complete B+4f12 block without retaining all connection coordinates.
It does not evaluate angular coefficients, CSF amplitudes, a Hamiltonian or
energies. The [contract](contract.json) binds the complete original B model
plus all 82 missing NR occupancies, the current source and prior integer census.

The target is **738 jj patterns, 92,715 fixed-M determinants, 5,717 CSFs by
integer multiplicity, and 43,720,175 ordered structural connections**, for
69 electrons and sector `(2J, parity)=(1,+1)`. The native union layout and
determinant remapping retain the existing ordering. No physical truncation
or radial-cutoff change is permitted.

The graph has 23,456,895 spectator memberships. Cauchy's inequality requires at
least 5,952,974 distinct buckets. A conditional CPython payload floor is
830,576,352 bytes before integer bitwidth, dictionary capacity, list capacity,
native imports and caches. Actual peak memory is unknown. The separately
authorized **4 GiB address-space / CPU 28 / one-thread / 300-second** cap is a
bounded experiment, not a fit assertion. Only the native spectator map, one
row's neighbors, aggregate counters and a 64 KiB hash buffer are retained.

`run.py --prepare` uses integer/standard-library work only. It completes the
target census and writes the source-bound contract; no native enumeration
occurs. The numerical path requires an exact root-issued seal and a new
`actual/` directory. First source, census, ordering, host, cap or timeout
failure stops the trial and preserves partial results. Its hash/count and
resource results cannot certify full-H/CI memory or spectroscopy accuracy.

The [closed actual result](actual/result.json) contains **43,720,175 ordered
pairs**, exactly the prior integer count. The zero-, one- and two-replacement
counts are 92,715, 1,206,876 and 42,420,584. All 92,715 rows and diagonals
are present; every emitted coordinate is strictly ordered and has Slater–Condon
distance 0, 2 or 4. The ordered-coordinate digest is
`9a49475f11b27a0889a1d187c9a5a18a445db0e92510688362f0b131d4a290e4`.

Total elapsed time was **53.05 s**, including scalar validation and streaming
hashing; graph work took 52.13 s. Peak RSS was **3,201,428 KiB (3.053 GiB)**
with the enforced 4 GiB address-space cap. This is one diagnostic run, with
no baseline or speedup inference. Native determinant enumeration took 0.93 s.

The live native index had **16,684,544 distinct spectator buckets**,
23,456,895 memberships, 92,428,751 ordered bucket visits, and a maximum bucket
length of 121. `sys.getsizeof(map)+keys+lists` totaled **2,834,499,496 bytes**;
shared index integers, determinant masks, caches and allocator overhead are
excluded. The remaining concrete representation cost is the spectator map.

The counts imply at least `2*16,684,544-23,456,895 = 9,912,193` singleton
buckets. A separately qualified successor could store the shared first row
index directly and allocate a list only on a second occurrence. In this
interpreter, that would remove at least 872,272,984 bytes of singleton list
payload. This is a source-design bound, not an implemented change or measured
successor result. The [read-only closure](receipt.json) checks the sealed
source/input identity, exact counts, host/resource scope and claim limits.
