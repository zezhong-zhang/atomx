# Literature-guided analytic CI-response curvature

The variational2n+1 identity is now implemented and applied in native Atomx
to both complete original Yb+ models, all five target states and all31 radial
waves. The independent analytic coefficient predicts all16 previously measured
finite CI-response line terms at the two original steps. Maximum discrepancy
is9.14350e-13 Ha per original direction coordinate; maximum relative discrepancy
is0.02746%, below the predeclared0.5%/2e-12 gates. All155 live controls pass.

The [literature/source map](LITERATURE.md) distinguishes Gonze's variational
theorem, Froese Fischer et al.'s MCDHF review, Li et al.'s GRASP performance
work, and actual pinned GRASP source. They guide native implementation;
no GRASP/backend calculation or paper numerical accuracy is imported.

## Implemented pieces

`numerics/variational_response.py` supplies an explicit real complete-spectrum
isolated-state response. For actual Hamiltonian derivatives H1/H2, it computes
the signed first CI-vector response c1 and the coefficient
`B_CI = c1.T H2 c0 + c1.T(H1-E1)c1`.
All external finite-basis eigenstates enter. Full H residual/Gram are checked;
partial or target-degenerate eigensystems reject. The returned coefficient is
the response-only contribution to E'''/6. The fixed-vector H3 term is absent
and must not silently be inferred.

The same numerical module differentiates the polar orbital path through two
matrix-square-root Sylvester equations. It retains the original finite Gram
and tangency defect, including repeated Gram eigenvalues. This supplies the
actual first/second orbital derivatives, not an unconstrained linear path.

`physics/csf/radial_increment.py` adds an explicit native B-spline real-DC
first/second radial-key derivative factory. One-body derivatives retain raw-h
ket-normalization terms; Coulomb derivatives retain all four-slot, P/Q and
mixed terms in the native cell orientation. Orders share density/potential
structure. Old finite-increment code is unchanged; there is no default/API
change to the CI solver, density or optimizer.

These are numerical/physical operator ingredients. Bare arrays do not establish
state, basis, radial, Hamiltonian or metric ownership. The consuming experiment
binds those identities explicitly. This dense complete-spectrum response is
not a full4f12 matrix-free response solver or a completed MCDHF Hessian.

## Tests and preserved failures

Fifteen new independent tests pass, covering a closed-form two-level energy,
signed eigenprojectors/large energy origins, separate intermediate-normalization
third-order recursion, complete-spectrum/input/gap/cap rejection, exact
single-column polar derivatives, repeated-Gram constraints and native
two-/four-slot radial differentiation. Together with existing polar and
five-state derivative tests,25 relevant tests pass.

The guarded legacy finite-increment panel adds42 passing tests and two failures:
frozen H bitwise parity and a finite central directional comparison. The full
affected run is67 passed/2 failed, not an all-pass suite. A sealed comparison
against the complete actual358-file source *before* this increment reproduces
the same two failures. All13 complete H/eigensystem archives,39 arrays, match
before/after bitwise, and the old radial-increment AST is unchanged. See
[the matched review](before_after_regression.json). No old test or tolerance
is changed or relabelled. The initial unsealed fixture invocation had32 setup
errors before its producers; that preflight mistake is retained in
[the test receipt](test_receipt.json).

## Complete Yb experiment

The [predeclared contract](contract002.json) retains69 electrons, Z70/A175 Fermi
nucleus, A91NR/1595CSF and B100NR/2574CSF, all31 waves/3689 coordinates,
the original complete negative preconditioned direction, original P/S/inverse
metric and five physical UIDs. It builds H0/H1/H2 in every target symmetry
sector and solves each complete H0 once, preserving signed target matching.
No displaced H or step scan enters the analytic coefficient.

The mathematical polar path's zero point differs from literal stored R0 by at
most3.55272e-15 in radial components. That difference is measured explicitly:
the new H0 is assembled at polar0 rather than silently reusing an old owner.
Maximum H0 difference from the literal original matrix is1.09140e-11 Ha.
The original P/S geometry is retained. First directional actions match the
original independent state-gradient actions within1.17168e-12 Ha; maximum
projected CI-response equation residual is1.88058e-13 Ha.

The initial driver passed raw tuples to an API requiring CSFSignature and
stopped at its first matrix-builder entry, with zero completed block matrix,
key evaluation, eigenpair or response output. Radial/one-body setup had run.
The [failed result](actual001/result.json) and original contract are preserved.
The explicit [successor review](source_review002.json) allows only the signature
adapter and separate output/input bindings;4169 exact signature roundtrips
precede it. Physical source, model, steps, gates and resource limits are identical.
The corrected driver then finishes once; no scientific failure triggers a scan.

The [closed result](actual002/result.json) contains30 complete matrix builds,
10 full eigensystems,10 response evaluations and19,828 distinct radial keys.
It retains62,363,256 bytes of H0/H1/H2, full eigenvectors/energies and c1 arrays.
Elapsed96.663s, peakRSS1089.379MiB, one CPU27/thread, under8GiB AS/600s.
These are unpaired feasibility measurements, not a controlled speedup.

## Physical meaning and next boundary

The D3/2-minus-ground B_CI coefficients are0.00269683400034/0.00266915717807
Ha per direction-coordinate cubed for A/B. At h=0.002 their predicted central
response terms are1.07873360014e-8/1.06766287123e-8 Ha per coordinate. These
independent derivatives agree with the previously measured finite endpoint
response. This supports the finite-step CI-response explanation of the two
large-step ordinary-FD discrepancies; it supplies no missing first-derivative
factor or energy correction for experiment.

The [old original005 ordinary failures](../physical_next_decision005/STATUS.md),
the [current centered38/40 replay](../physical_next_decision011/README.md) and
the two matched legacy-test failures remain failed. The predicted response is
not subtracted to retrospectively pass any ordinary gate. H3/fixed-vector
curvature, higher-order remainder and arithmetic/constraint reconstruction
still need separate qualification for a *total* finite-step certificate.
No optimizer is admitted here. This finite complete original-space result
does not certify full4f12 response, stationarity, experiment or the all-element
database, and accepted13 effects are never transferred to terminal16.

[Complete signed tables](RESULTS.md), [literature/source map](LITERATURE.md),
[test record](test_receipt.json), [source identity](source_current002.json).
