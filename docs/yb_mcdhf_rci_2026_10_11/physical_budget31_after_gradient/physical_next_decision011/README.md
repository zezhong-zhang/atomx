# Exact saved-H derivative decomposition and native centered replay

The original005 three ordinary-FD failures now have a measured decomposition.
At h=0.002, the A/B D3/2 line discrepancies contain finite CI-vector response
terms of1.07864e-8/1.06769e-8 Ha per directional coordinate. At h=0.001, B's
reported-root-minus-exact-Rayleigh contribution is6.05058e-9 Ha. These are
different mechanisms; increasing contraction precision alone cannot remove
the two large-step discrepancies.

The existing current native centered dense helper was then applied once to
the same ten saved005 Hamiltonians. All50 complete symmetry-block spectra
pass their original residual1e-8 and Gram1e-10 gates. The ordinary derivative
panel, using the original saved analytic actions, passes38/40: B D3/2 at
h=0.001 now has error4.50322e-9 Ha, while A/B at h=0.002 remain failed at
1.34901e-8/1.31434e-8 Ha. No production source was changed.

## Exact decomposition

The [predeclared contract](contract.json) binds28 actual inputs, including
all ten complete original005 H/C/energies/orbital archives, their full owner
metadata, original source identity and the current358-file source manifest.
The [review](source_review.json) preceded the once-only execution. The driver
imports no Atomx and runs no H/CI/radial/gradient/optimizer producer.

Every binary64 H and signed-vector coefficient is converted to its exact
integer ratio. Signed upper-triangle products and sums are exact dyadics;
normalization uses the exact measured full coefficient norm. Only exact-zero
products are omitted. Removing a fixed baseline scalar inside a quadratic is
an exact arithmetic identity, without rounding or changing the physical H.
The97 independent small-system Fraction controls include signed interference,
tiny nonzero coefficients and phase/scaling invariance.

For each original state and each upper-minus-lower line, at both original steps:

`reported FD - analytic = reported FD - exact Rayleigh FD`
`                      + exact Rayleigh FD - fixed-C0 FD`
`                      + fixed-C0 FD - analytic`.

The identity is checked as an exact Fraction, before conversion for display.
The second term uses the actual signed endpoint CI columns; it is a finite-step
coefficient-response contribution, not a missing first derivative term.
The last term includes finite-step fixed-vector operator response, original
polar reconstruction and stored-H assembly effects; this audit does not
uniquely separate them. It cannot certify the exact continuum Hamiltonian.

The [closed result](actual001/result.json) passes515 identity/readback/closure
controls, plus the97 independent arithmetic anchors. All original ordinary
slopes and decisions are reproduced. It makes80 exact normalized quadratics,
visits7,799,576 upper-triangle entries, and loads10 existing complete NPZ files.
The run took5.170s with peakRSS460.133MiB under CPU27/thread1/4GiB/300s.
This is a bounded diagnostic observation, not a speedup measurement.

## Existing native solver replay

The [separate centered contract](centered_contract.json) and
[source review](centered_review.json) keep the same saved H, all CSFs/UIDs,
steps, direction, analytic actions and tolerances. Current
`atomx.methods.ci._centered_dense_eigh(..., driver='evd')` is the existing
complete dense path used by `ci_solve`; its original-H storage is unchanged.
The existing public complete/subset signed analytic-projector panel passes
[10 tests](test_receipt.json).

The [closed replay](centered001/result.json) passes384 solver/identity controls.
All fifty complete eigenblocks are saved with their rows, energies and signed
vectors. New target matching requires the original local root and maximum
same-point saved-column projector overlap above0.999999, independently of NIST.
Maximum full residual is2.00089e-11 Ha and Gram defect4.44089e-15; minimum target
projector overlap is0.9999999999999967. Fifty additional exact Rayleigh controls
retain all signed components and actual norms. Maximum reported-root-minus-
Rayleigh difference falls from9.84964e-12 to9.06470e-13 Ha on these points.
This compares each route with its own literal vector quotient, not a certified
absolute eigenvalue-error bound.

The replay takes7.005s, peakRSS490.434MiB, under CPU27/thread1/4GiB/180s;
it saves78,162,520 array bytes. It constructs no fresh physical H, radial
integral, gradient or optimization point. It does not recreate a CIResult or
claim public full-spectrum ownership from archived arrays. Original005 used
the frozen uncentered f043 source; this replay explicitly binds the current
source instead of silently replacing its epoch.

## Interpretation and remaining work

Across all eight A/B lines, the measured finite CI-response term at h=0.002
divided by its h=0.001 value lies between3.99901 and4.00073. This supports an
O(h^2) central-difference truncation interpretation. It is not an independent
analytic third-derivative certificate. In particular, the two large-step
ordinary gates still fail even when the saved endpoint Rayleigh quotients are
evaluated exactly.

The next discriminator should independently calculate the signed CI-response
curvature in the original constrained radial space at the same point, then
compare its predicted finite-step term with this measured response. It must
retain the original h/tolerances and all channels, without substituting a
Richardson value or retroactively passing an ordinary gate.

The [original005 record](../physical_next_decision005/STATUS.md) remains a closed
scientific failure. The new replay is38/40, not a qualified continuation.
Its analytic actions are the original005 values, without a new current-source
gradient or fresh-H production run. It does not admit an optimizer, certify
stationarity, change accepted13 into terminal16, improve the spectrum against
experiment, or validate the full4f12 finite-step response. The complete4f12
[density/gradient qualification](../physical_next_decision010/README.md) remains
a separate result. No cards, timers or alternate backend were used.

[Detailed signed tables](RESULTS.md) retain every line and all40 native-replay
outcomes. Contracts, reviews, arrays and source/input checks remain replayable.
