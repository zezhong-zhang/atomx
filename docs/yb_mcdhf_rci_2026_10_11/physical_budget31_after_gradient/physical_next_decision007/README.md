# Exact row streaming and typed angular CSR

This source-bound increment replaces the full global connection set/list with
row-wise ordered connections and constructs CSR directly from typed buffers.
It retains every signed coefficient, radial-key encounter order, explicit zero
and connection of the declared physical model. Generic complex callback values
retain their imaginary parts through dynamic buffer promotion.

The [complete A/B ABBA comparison](benchmark001/RESULTS.md) passes: 40 block
records, including 30 cross-process comparisons, have identical candidates,
keys, CSR arrays, projections and signed actions. All load guards pass; two
fresh workers per implementation give mean angular-table construction
**63.252→56.849 s (1.1126×)** and mean peak process RSS
**976.145→800.699 MiB (−17.97%)**. The retained 26,418,669 nonzeros and
332,210,280-byte table payload are unchanged. These are angular-stage results,
not whole-solver or parallel speedups. All 222 affected tests pass.

The [largest complete B+4f12 structural graph](full_f12_connections001/README.md)
also passes its independent integer census: 738 jj patterns, 92,715 raw
determinants, 5,717 CSFs counted without amplitudes and all **43,720,175**
ordered connections. It took 53.05 s at 3.053 GiB peak RSS under a separately
declared 4 GiB address-space cap. All rows, diagonals and 0/1/2 replacement
counts match; no angular coefficient table, Hamiltonian or energy was computed.

Its measured spectator map still owns 2,834,499,496 bytes across 16,684,544
buckets. At least 9,912,193 buckets are singletons that only contribute
diagonals. The [next source increment](../physical_next_decision008/README.md)
tests compact first-occurrence storage and explicit diagonals, preserving the
same graph. This report and its receipts remain bound to the 007 source;
the successor does not rewrite these measurements. See [increment receipt](increment_receipt.json).

The original stationarity, three line-FD failures and experimental gaps remain.
Complete 4f12 H/CI admission, selected-root ownership/response and spectroscopy
are separate unfinished steps. Existing representation and evidence principles
cover this result; no new general insight is promoted.
