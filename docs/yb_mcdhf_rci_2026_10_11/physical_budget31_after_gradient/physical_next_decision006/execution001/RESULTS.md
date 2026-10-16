# Native sparse Hamiltonian/Davidson result

**PASS:273/273 original controls.** At the complete frozen f043 accepted13
A1595/B2574 point, the existing native sparse Hamiltonian action matches
the complete owned dense reference spectra; native per-sector Davidson
recovers the lowest2 states in all10 complete symmetry blocks. This is
representation/solver qualification at fixed physical inputs, not a new
correlation or experimental-precision result.

All31 orbitals,69 electrons, original A175 Fermi nucleus, full signed CSF
rows, nuclear positive-energy projector and radial metric remain unchanged.
The full1595+2574=4169 reference eigenvectors were checked in batches≤32.
No dense Hamiltonian was rebuilt; the existing dense H/E/C arrays were read
only as controls. All five original target UIDs are present in each model.

| Maximum across both complete models | Observed | Original limit |
|---|---:|---:|
| Signed sparse versus dense action / Ha | 1.637090463191e-11 | 1e−8 |
| All reference eigenpair max-entry residual / Ha | 1.818989403546e-11 | 1e−8 |
| Sparse versus dense diagonal / Ha | 0.000000000000e+00 | 1e−8 |
| Normalized bilinear symmetry discrepancy / Ha | 3.659295089165e-13 | 1e−10 |
| Selected direct residual L2 norm / Ha | 9.107792209579e-10 | 1e−8 |
| Selected direct residual max-entry / Ha | 2.739852789091e-10 | 1e−8 |
| Selected energy difference / Ha | 2.182787284255e-11 | 1e−8 |
| Phase-aligned full-row wave difference | 4.631124261856e-09 | 1e−8 |
| Rank-one projector max-entry difference | 5.033692573875e-09 | 1e−8 |
| Selected Gram max-entry error | 1.554312234475e-15 | 1e−10 |

Cross-sector action entries are exactly zero. Each selected reference root
is isolated under the predeclared1e−10 Ha cluster check. Davidson used its
unchanged native algorithm with internal L2 tolerance1e−9 Ha and200-iteration
cap; actual iterations range22–38. The post-solve L2 and max-entry residuals
above are separate direct applications of the full native operator.

## Measured sparse construction payload

The following are actual angular coefficient-table CSR and exact-CSF
projection CSR payloads, in bytes. They are not determinant-H CSR sizes or
complete resident memory. Table-key and pair-array storage are additional.

| Model | (2J,parity) | Full CSFs | Raw determinants | Coefficient NNZ | Coefficient CSR bytes | Projection CSR bytes |
|---|---|---:|---:|---:|---:|---:|
| A | (1, 1) | 223 | 1705 | 2,425,747 | 30,552,592 | 38,384 |
| A | (1, -1) | 219 | 1640 | 2,238,021 | 28,188,152 | 35,268 |
| A | (3, 1) | 370 | 1660 | 2,377,069 | 29,960,124 | 51,788 |
| A | (3, -1) | 367 | 1619 | 2,249,299 | 28,359,092 | 47,532 |
| A | (5, 1) | 416 | 1324 | 1,829,976 | 23,000,776 | 46,892 |
| B | (1, 1) | 365 | 2659 | 3,369,584 | 42,330,600 | 81,488 |
| B | (1, -1) | 362 | 2623 | 3,241,770 | 40,702,828 | 88,196 |
| B | (3, 1) | 596 | 2478 | 3,180,849 | 39,985,812 | 105,988 |
| B | (3, -1) | 599 | 2466 | 3,111,938 | 39,110,556 | 112,408 |
| B | (5, 1) | 652 | 1918 | 2,394,416 | 30,019,748 | 87,224 |

The10 angular tables contain26,418,669 nonzeros and332,210,280 CSR bytes;
projection CSR payload totals695,168 bytes; pair arrays total60,744,848 bytes.
The once-only diagnostic run took78.89 s and peaked at1689.47 MiB RSS under the8 GiB address-space cap.
The two complete operator builds took76.58 s; all10 Davidson solves
together took0.199 s. These are uncontrolled stage observations, with no
paired dense timing baseline or speedup claim. They locate the cost in
operator construction on this panel, without separating angular and radial
construction costs or extrapolating to the larger4f¹² model.

Counters retain2 full sparse builds,10 tables/projections,43 early angular
nullspace calls,7 raw h/S constructions,2 radial evaluators,138,808 radial
key requests and1984 density constructions. There are455 guarded block
actions plus10 full-operator cross-sector checks; construction and diagonal
contractions are separate work. Five NPZ inputs were decoded and two complete
selected-state outputs saved. All81 actually imported native modules were
bound to the frozen source archive.

## Failure record and scope

The first shell launcher used `taskset -c27`, which failed before the Python
interpreter, created no `actual/`, and made zero science calls. Its
[record](launcher_failure001.json) is preserved. Root corrected only the
launcher argument spacing to `taskset -c 27`, leaving driver, contract and
seal unchanged; the scientific execution then ran once. No numerical retry,
tolerance, physical-space or solver change occurred.

The existing `CIBasisTracking`, `CIStateAverage` and retained Hamiltonian
snapshot interfaces require complete dense spectra. This selected iterative
result is qualified by signed same-basis comparisons to their owned dense
UID controls; it does not qualify those full-only public consumers, a new
iterative ownership certificate, changed-orbital tracking or a larger MCDHF
optimization. No4f¹² Hamiltonian/energy, radial stationarity, missing-physics
contribution or new NIST accuracy was calculated. Prior derivative and
stationarity failures remain failed.

Evidence: [contract](contract.json), [driver](run.py),
[original closed result](actual/result.json), [A](actual/A.json),
[B](actual/B.json), [complete selected A arrays](actual/A_selected.npz),
[complete selected B arrays](actual/B_selected.npz),
[result-only review](result_review.json) and [closure receipt](closure_receipt.json).
The additional review passes158 JSON/source/hash checks with zero NPZ
decoding and zero science calls; it is a consistency review rather than an
independent numerical rerun.

Documentation disposition: no shared source/API/default changed. The new
qualification is owned by this report and the decision006 parent navigation.
Existing full-spectrum consumer restrictions remain applicable. No new
general rule is promoted; the existing same-operator and complete-state
identity rules cover this result.
