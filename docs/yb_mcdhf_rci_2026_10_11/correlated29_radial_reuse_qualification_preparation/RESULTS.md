# Closed qualification failure

The sole root-authorized foreground run exited1 after11.902s, with225 anchor
checks passed and one failed. `analytic_6_pppp` differed from the unchanged
origin-cell analytic value by3.6586666666948986e−5Ha, exceeding its
3.585493333333332e−14Ha bound. The corresponding independent near/full-bank
comparison was exact, dense-versus-factored potential differed only
6.94e−18, and their signed scalar comparison was exact. Equivalent
representations share this error; equivalence cannot replace the failed
analytic anchor.

All four ns95/96 ordinary/mixed-RKB workspaces completed rank0/1/6 PP/QQ/PQ/QP
controls and actual candidate routes. The origin single-cell rank0/1 cases
completed. Rank6 stopped at PP before QQ/PQ/QP and the candidate route;
end-of-anchor call-count gates, source-final gates, both complete29 chains,
full H/CI/g/metric parity and stage profiles were not executed.

Actual completed calls: basis5, small Gauss eigvalsh5, density20, full-bank
convolution89, factored convolution81, band scalar142, candidate helper14,
bank getters89, factor getters81. H/CI/g/Dirac/optimizer/GPU calls were zero.
The root receipt, one-shot marker, execution seal, logs and failure are
retained. There was no retry, gate change, model reduction or shared-source edit.
All27 immutable input bindings and original prepared run/contract/proof
hashes remain matched. `actual_status_verification.json` verifies this status
and identity only; the scientific qualification failed.

For the source-only causal analysis, each component of the toy quartet has
P=Q=scale*r, with scales1,−2,3,4. Its density product has scale−24 and the
unscaled radial integral is

`I_k = 2 R^5 / [5(3+k)]`.

This matches the runner's analytic formula. The double integral is convergent
at the origin for k6. However, this radial regularity is not a physically
allowed k6 s-s angular channel, and is weaker than the relevant f-f source
regularity. The failure is a mathematical radial-kernel anchor, not a measured
Yb energy defect.

`_triang_sym` fixes the inner rule to `n_sub=ks=4`; the mixed routine uses
`max(ksp,ksq)`. The outer nGauss16 does not change it. For this quartet the
inner polynomial is r^(2+6)=r^8, whereas four-point GL is exact only through
power7. The GL4 unit-interval x^8 error is1/44100 (the squared shifted monic
Legendre4 norm), relative1/4900 to the exact integral1/9. This predicts
3.658666666666…e−5Ha for the signed component, matching the actual failure.
This is an analytical/source explanation; no new quadrature or kernel was run.

For nonnegative Coulomb k and a source B-spline order s, the complete inner
basis product has degree at most2s−2+k. A polynomial-exact inner GL rule
therefore needs `s+floor(k/2)` points. Mixed directions require their source
order separately or the safe `max(ksp,ksq)+floor(k/2)` rule. Negative k follows
a rational-integrand path and this polynomial formula does not cover it.
Nor does repairing the inner rule establish outer-quadrature, radial-basis or
continuum exactness. The helper documents n<=20 and uses at most50 Newton
iterations without an explicit convergence assertion; proposed orders4–12
require their own bounded numerical anchors.

Saved coefficient metadata has ranks0–6. At rank6 there are five unfiltered
signed terms (three negative), four in the historical retained-gradient table,
and two distinct source pairs: 4f7/2–4f7/2 and4f7/2–4f5/2. The absolute
coefficient sum is about1.62854. The f5/2–f5/2 rank6 vertex is excluded by the
j triangle. For a smooth finite nucleus, relevant large f waves behave as
r4; paired large densities behave as r8. Small-component allowed vertices
likewise have the appropriate higher angular regularity. These metadata do
not bound the numerical radial correction. Per-state signed normalized
tables and the actual123-point workspace must be consumed by a separately
sealed discriminator before assigning actual lower/upper transition shifts.

The proposed physical quadrature discriminator is separate from this
performance-route candidate. It must copy ordinary500ac and change only the
two inner orders, retain k<0 unchanged with explicit unsupported accuracy
scope, preserve this original failure, and quantify same-point five-state
fixed-CI expectation shifts without claiming new H/CI stationarity. No
performance, optimizer, experimental-accuracy or GPU certificate was earned.
