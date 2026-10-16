# Native paper-method development: measured finite milestones

**Native finite CCSDT, finite EOM-EA and g/h/i capability controls pass. The
69-electron Yb⁺ numbers in Sahoo2025 have not been reproduced.** The current
Yb MCDHF/RCI spectrum and its failed stationarity/experimental gates are unchanged.
The [initial paper contract](paper_contract.json) keeps published input and
unknowns separate; [development receipt](development_receipt.json) binds this
new source/test epoch and the actual high-l archive.

| Native milestone | Actual validation | Meaning |
|---|---|---|
| Connected core T3 | Complete 3e/6-spinor FCI, independently signed fermions and `log(1+C)` amplitudes; nonzero T3=7.25683422567e−4; CCSDT−FCI=+2.22e−15 Ha | Connected triples now solve jointly with T1/T2 in the bounded native engine |
| Nonlinear/exponential structure | Complete 4e/8-spinor independent-fragment product retains disconnected T3×T1; a separate generic 4e model retains its nonzero omitted-T4 defect | Rank-three CC is neither rank-three CI nor generally full CI |
| Electron attachment | Same owned integral operator/core energy in both complete particle sectors; independent full tensor/Fock signs, non-Hermitian projected matrix, signed phases, physical norms and separate residuals | Core connected rank and attachment 1p/2p1h/3p2h rank are distinct |
| Active T3 in N+1 | 3e/7-spinor core; all 35 attached determinants, 34 R≤3p2h states, omitted 4p3h state retained in full-H diagnostic; independently signed full lift and matrix exponentials compare every projected H̄ entry and physical state | Avoids the Pauli-blocked T3 action of the smaller 6-spinor attachment example; complete-spectrum similarity alone would not prove the lift |
| Live native atomic operator | Be²⁺→Be⁺, complete 1s/2s/3s radial pool, 15/20 magnetic determinants, B-spline DF/native packed integrals versus independent sparse-CI assembly | Checks the actual radial/angular/operator seam, not experimental Be accuracy or core T3 (N=2) |
| High angular momentum | Six 5g/6h/7i partners versus point-Dirac energies, max error1.11e−10 Ha; complete g² pool has153 magnetic states/17 CSFs, full-H residual4.44e−16 and phase covariance0 | Bounded native high-l capability; no Yb high-l correlation shift was measured |

The finite CC and attachment interfaces reuse the same signed cluster generators
and exact nilpotent exponentials. The new attachment module consumes the existing
owned native integral type, carries a separately owned producer/nuclear/radial
identity, and never symmetrizes the EOM matrix. Method labels contain both core
and attachment ranks. No external electronic-structure backend is invoked.
Native MCDHF→RCI remains the production development route; this RCC/EOM work is
an independently labeled paper-method comparison route.

## Validation and retained failures

The final six-file panel passes **39/39 tests, zero skips**, including existing
finite CCSD/scaffolds and architecture/import/request-fidelity controls. The
[JUnit report](development_tests.xml) records that execution. Ruff passes the
CC/EA/scaffold Python files and the FSCC docstring update. See [native triples](native_triples.md)
for equations, numerical anchors and source hashes, and the
[high-l actual results](high_l_native003/RESULTS.md) for all127 controls and
complete retained arrays.

The inherited finite-difference Jacobian first stalled at continuation0.5 in
the unchanged noninteracting-fragment fixture. The native solver now supplies an
exact excitation-commutator Jacobian, independently checked against full-matrix
exponential differences. The original failure remains in the triples note;
the model, continuation and residual gate were not relaxed. Jacobian payloads
are included in the existing memory admission.

The first root EA test run had a test-code `abs(list)` TypeError (1 failed/3
passed); changing that assertion to an array maximum corrected it. An earlier
broader panel gave36 passed/2 failures in existing shooting Numba cache
serialization (`ReferenceError: underlying object has vanished`). The two exact
failed tests passed with an isolated cache; the final expanded panel then passed
39/39 with that cache. No shooting source or physics/test thresholds changed.

The [high-l source-admission failure001](high_l_native001/preexecution_failure.json)
and [work-cap/scalar-export failure002](high_l_native002/actual/failure_receipt.json)
remain unchanged. Explicit successor003 fixes the understated h/S work count and
NumPy boolean serialization; its physical model and all numerical gates match.
Its78 actually loaded module hashes were checked, and no CC/EA module was loaded.

These are correctness measurements. Test elapsed times and the high-l warm-cache
0.621s do not provide controlled efficiency or parallel-speedup comparisons.
Finite CC defaults stay256 determinants/128 amplitudes/128MiB admission; memory
estimates cover declared NumPy payloads, with input integrals and Python/SciPy
overhead outside the EA estimate. EA logical term-action caps cover constructing
the two H matrices, not nonlinear/Jacobian/similarity/eigensolver work.

## Remaining paper reproduction boundary

The primary [Sahoo2025 paper](https://arxiv.org/html/2504.01438v1) uses RCCSDT
throughg, h/i RCCSD increments and self-consistent DHF plus RCCSD Breit response.
It does not provide the actual nucleus/radius, 40-GTO exponents per symmetry,
correlated/frozen partition, virtual/triples cuts or full solver controls.
Another paper's inputs and the old AtomxA175 point do not fill those unknowns.

The finite EOM-EA module is not a scalable valence-universal FS-RCCSDT engine;
large-core angular/tensor contractions, native self-consistent Breit response
and actual matched Yb runs remain required. The high-l pilot uses a point-nucleus
one-electron anchor and frozen g² pool, not a converged Yb closed-core field or
optimized MCDHF wave bank. Next Yb admission needs an actually owned common
closed-core/orbital geometry and complete counted g partners, followed by
separately declared higher-l/core/higher-order runs. Keep the original MCDHF
stationarity and experimental gates, and do not splice published corrections
onto the current spectrum. Cards/timers remain stopped.
