# Complete forward increment: closed arithmetic discriminator

The [root receipt](root_execution.json) closes exit0, **98/98 live controls
passing**, no outer stop and all399 frozen source/input guards unchanged.
The [result](actual/result.json) retains all1595 CI coordinates,31 orbitals,
19589 radial keys and five original signed physical targets. This extends the
[191-control endpoint comparison](../endpoint_radial_contraction_preparation002/RESULTS.md)
at exactly its saved baseline/h0-plus pair, with no new radial integrals,
Hamiltonian construction, CI solve, gradient or optimizer.

## Measured location of the dominant difference

Let `T` be the complete native angular table and `R0,R1` the saved ordinary
radial-key vectors. Hold the full signed trial CI column and its actual norm
fixed. Compare the contraction of `T@(R1−R0)` with the separately rounded
endpoint route `T@R1−T@R0`. The primary contraction uses the same native
FP64 pair products as the transposed density producer; wide table products
and a genuine sparse-projection control are recorded separately.

| Complete weighted route | Increment / Ha |
| --- | ---: |
| Native FP64 forward table applied to ordinary key differences | +3.8324250548312370e−7 |
| Wide forward table applied to ordinary key differences | +3.8324250548305925e−7 |
| Replayed transpose ordinary key differences | +3.8324250548306790e−7 |
| Saved forward absolute-endpoint subtraction | +3.8321581303790710e−7 |
| Saved forward signed-pair endpoint differences | +3.8321581219285590e−7 |
| Saved complete-H radial contraction | +3.8321583218072087e−7 |
| Wide forward table applied to polynomial plus reconstruction | +3.8324478638835690e−7 |
| Replayed transpose polynomial plus reconstruction | +3.8324478638836554e−7 |

Across all five targets, native forward-minus-transpose is at most
**8.2361e−20 Ha** for ordinary differences and3.5059e−20 Ha for the polynomial
route. Wide differences are at most4.6569e−20 Ha; the sparse/dense wide
control differs by at most4.2497e−23 Ha. These are observed comparisons,
not certified arithmetic bounds.

In contrast, the weighted native direct-increment route differs from saved
absolute-endpoint subtraction by **+2.6692445e−11 Ha**. Its difference from
saved signed-pair endpoint subtraction is+2.6693290e−11 Ha. The wide direct
increment differs from stored-H by+2.6673302e−11 Ha, reproducing the dominant
92.12% part of the earlier2.8954208e−11 Ha decomposition.

The two saved endpoint-subtraction variants differ by only+8.4505e−16 Ha
weighted. Thus the dominant discrepancy is already present in the separately
formed forward row actions `T@R1` and `T@R0`, before the final scalar total
subtraction. It is not explained by rounding that final subtraction alone.

This discriminates **finite-precision nonadditivity of the complete forward
endpoint assembly** at the fixed pair: direct forward and transpose
increments agree, while the route through two separately assembled endpoints
does not preserve that increment. Merely widening the final increment-table
product changes it by at most6.77e−20 Ha; it does not recover arithmetic
already performed when the endpoint tables were assembled. No single source
operation, global error bound or production repair is certified by this result.

## Complete signed key evidence

The [key ledger](actual/complete_trial_signed_key_contributions.json) saves
all19589 ordered keys, every trial density coefficient including exact zeros,
and ordinary/polynomial contributions for each complete signed target.
It adds the coefficients missing from the earlier endpoint key ledger.
The [saved-JSON summary](diagnostic_summary001.json) contracts all keys with
the original0.2 weights, then displays a ranking as a diagnostic view.

The smaller polynomial-minus-ordinary part is approximately
2.2809053e−12 Ha:57 one-body keys contribute+1.8458959e−12 Ha and19532
Coulomb keys+4.3500940e−13 Ha. The largest weighted key discrepancy is the
1s one-body diagonal,+1.0835497e−12 Ha. These key differences describe the
smaller radial-production part, **not** the dominant endpoint-assembly part.
Rounded JSON term sums are not an exact dyadic certificate.

The entire earlier2.8954e−11 Ha discrepancy is approximately6.35e−6 cm⁻¹
as an energy scale, far below the hundreds/thousands of cm⁻¹ physical line
responses. It can nevertheless fail a directional derivative check when
divided by a sufficiently small displacement. It is not a spectrum error bound.

## Scope, work and next physical measurement

Actual work is2 complete NPZ loads,38 angular nullspaces,5 complete angular
tables,10 projections,5 unfiltered signed densities,5 target vectors,
20 forward-table products,25 dense and25 sparse target projections and125
signed row contractions. Angular structures retain11,120,112 nonzeros and
1,654,843 pairs. Exact work controls passed; H/CI/hS/radial/functional/
gradient/optimizer/eigh entry blockers remained zero. Eager module imports
are checked against the frozen source origin, separately from call boundaries.

The24 SOURCE controls and independent SOURCE review preceded the execution.
The [independent saved-JSON result review](../endpoint_source_review001/independent_result_review001.json)
passes60 checks over the closed receipt, full5×19589 ledger, norms, bindings,
work and arithmetic-stage conclusion. The [fresh root evidence check](root_evidence_validation001.json)
also rechecks the three preparation guard sets,356 source files,28 physical
decision inputs and10 frozen data artifacts without another scientific call.
Elapsed31.67s and peak RSS875.75MiB are uncontrolled observations, not fair
performance or AI-acceleration measurements. Root elapsed was32.04s under
the original90s inner/105s outer budget, CPU30/thread1/AS4GiB.

Both original StageB ordinary-FD failures, the failed001 target-order
preparation, expanded-core nested-byte failure and saved-spectrum final
module-boundary failure remain failed. The new comparison establishes a
local arithmetic location; it does not qualify a new gradient, objective,
optimizer continuation, source migration or stationary spectrum. AI advice
still abstains; correlated keys/targets supply no independent training groups.

Numerical follow-up can now target stable native increment assembly and its
same-point derivative qualification rather than searching for missing density
terms. The [next physical decision](../../../physical_budget31_after_gradient/physical_next_decision003/README.md)
is the common-displacement1595/2574 four-point correlation/relaxation
measurement. Its source, derivative and whole expanded-gradient resource
qualifications still precede execution. Accepted13 remains a diagnostic anchor;
terminal16 remains the production endpoint. No correction is transferred
between them, and NIST does not choose the displacement.
The [expanded-gradient readiness](../../../physical_budget31_after_gradient/physical_next_decision003/expanded_gradient_readiness001/README.md)
specifies the preceding conditional three-gradient comparison without new
H/CI or an optimizer state. Its prospective resource stops and source-derived
work bounds are not an execution contract or measured full-gradient peak.

Result SHA256:5b631503147217f3728dddb6e17dc8692abe66f2b50079339f968143c56d80af.
Key ledger SHA256:cf3ec93daef4bd6afe5955a7ff64ebb5df3a1472d4b7430457ceb913d08bc9fb.
Receipt SHA256:2ab94d2c48cf0fdb3f42c02b202c4a809cd3865221a551b9ab36c4139ed65667.
Binding SHA256:a2fffe0dac03fe94ad38077a47070064622629a6c9cfca1ea17737e6c62c4b56.
Original frozen SOURCE preparations and all predecessor evidence are unchanged.
