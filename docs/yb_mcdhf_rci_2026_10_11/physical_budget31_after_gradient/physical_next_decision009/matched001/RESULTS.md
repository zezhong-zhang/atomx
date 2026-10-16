# Native cached/streamed fixed-basis comparison: PASS

One root-sealed run passed **523/523 recorded gates**.
The independent read-only reviewer passed **222/222 identity/output checks**,
including direct checks of all forty saved selected eigenvectors against owned
004 dense H/E/C. The reviewer neither rebuilt H nor solved CI. No numerical retry
or altered tolerance was used.

Both algorithms used current dirty source map
`4b39195f35c7b2affa5a4841180b2fc2eaa82dbb833694905909e33e13ff77f1` (358 Python files),
including the streamed numeric-CSR implementation and the per-pattern sparse
diagonal. The external historical control remains the separately frozen 004
source epoch. All imported source hashes and complete input-file hashes were
verified before and after the run/review.

## Scientific scope

The fixed problem is accepted13, not terminal16: native 69-electron Yb⁺,
Z70/A175 Fermi nucleus, complete A1595/B2574 selected CSFs in all five symmetry
blocks, original 31 radial waves/3689 coordinates, S metric and frozen nuclear
positive-energy projector. Both modes include the same full Dirac–Coulomb
operator; Breit is off. No amplitudes, matrix entries, configurations, terms,
weights or sectors were discarded to obtain agreement.

Every signed reference eigenvector and every full-CSF unit coordinate was
applied in batches of32. Cached and streamed full Hamiltonian entries agree to
**7.27595761418e-12Ha**; every signed full-reference action
against the owned dense oracle agrees to **1.81898940355e-11Ha**.
The all-reference residual maximum is **1.81898940355e-11Ha**.
Native Davidson retained every CSF row while solving the lowest two roots
separately in every sector, twenty solves total. Complete-spectrum selection or
response machinery was not replaced with the partial output.

Across all forty saved selected roots, independent old-H residuals are at most
**9.2718505043e-10Ha** in L2,
energy differences **3.09228198603e-11Ha**,
signed-wave differences **4.78009243299e-09**,
projector differences **5.2396611494e-09**,
and full saved-vector Gram errors **1.99840144433e-15**.
The predeclared energy/action/wave/projector gates remain1e−8; selected Gram1e−10;
Davidson native L2 tolerance1e−9Ha/direct residual gate1e−8Ha.

## Descriptive resource observations

| Model/mode | H build seconds | Old full-C action max Ha | Direct native L2 residual max Ha | Saved signed-wave difference |
|---|---:|---:|---:|---:|
| A_cached | 28.177996 | 1.63709e-11 | 9.26646e-10 | 4.78009e-09 |
| A_streamed | 23.311843 | 1.63709e-11 | 9.05526e-10 | 4.746e-09 |
| B_cached | 36.211156 | 1.81899e-11 | 8.94438e-10 | 2.09325e-09 |
| B_streamed | 32.480917 | 1.81899e-11 | 9.02533e-10 | 2.02696e-09 |

Total elapsed **123.665345s**, process peak RSS
**1141.984375MiB** under CPU27/thread1/AS8GiB/1200s inner cap.
These fixed-order, instrumented correctness measurements establish no paired
speedup or independent per-mode peak-memory result. Cached operators/tables were
retained during comparisons and streamed modes reused warmed source caches.

Exactly four H builds, twenty sparse projections, ten cached angular tables,
ten streamed numeric matrices, twenty Davidson solves, and43 highest-weight
null-space calls completed. Streamed builds called neither `block_terms` nor
`_coefficient_table`. 1332 instrumented operator actions,
7,954,244 pair-kernel calls, and
52,837,338 signed angular-term emissions stayed within
predeclared resource caps. No numerical-zero threshold was introduced.

## Limits and artifacts

This qualifies the two current fixed-basis algorithms on the complete existing
A/B problems. It does **not** calculate the missing complete4f¹² correlation
class, establish MCDHF stationarity, certify a public partial-spectrum tracker
or retained snapshot, improve NIST/experimental agreement, or remove any earlier
005 derivative-gate failures.

- [Sealed run](actual/result.json), [driver](run.py), [contract](contract.json),
  [root seal](root_execution_seal.json), [current source map](current_source_identity.json).
- [Independent output reviewer](review.py), [review result](result_review.json).
- Four saved complete-row selected-vector archives and UID records are in
  `actual/A_cached*`, `A_streamed*`, `B_cached*`, `B_streamed*`.
