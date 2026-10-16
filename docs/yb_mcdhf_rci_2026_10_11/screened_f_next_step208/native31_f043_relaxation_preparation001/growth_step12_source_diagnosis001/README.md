# Why the gradient grew after step12 remains unresolved

Saved scalar evidence shows a real change of search region, rather than a
failed acceptance or a changed stopping metric. All16 finite stored-H
acceptance guards passed. At12→13, EO drops0.00634220Ha, while tangent norm
rises0.1974986→2.1989705 and original residual rises0.00843694→0.1539784.
The step is1 and no fallback occurs. The exact-acceptance uncertainty margin
is approximately3.9e−12Ha, far smaller than this finite descent. This does not
prove the radial derivative over the new region or imply a stationary solution.

The metric and seven nuclear projectors are unchanged. The ratio
`||TMTg||/||Tg||` changes0.042719→0.070023 across this jump; this is an
observed vector-action ratio, not a condition number or error bound. The
tangent gradient itself grows substantially, so the change cannot be
attributed merely to reporting a different metric. The fixed inverse still
acts on a changing tangent space and different vector components; its
conditioning cannot be inferred from scalar norms alone.

All original five UIDs still select their symmetry's local root0. Principal
reference-overlap probabilities change from0.977–0.990 at12 to0.960–0.979
at13, then approximately0.895–0.917 at16. The native majority and ambiguity
guards passed. This is continuous character change under the frozen tracker,
not evidence of a discrete target-label switch. It does not establish that
the finite model retains every desired experimental state character.

The production retained, unfiltered normalized and raw-h/ket-normalization-
corrected projected residuals agree within approximately1e−13 here. This
rules out those measured corrections as the source of the large scalar
increase at these points; their shared physical derivative can still be
wrong in an untested direction. Whole-CI residual/Gram and radial/no-pair
checks passed in the run, but none replaces that directional check.

The frozen public optimizer applies transported L-BFGS secants, a fixed
spectral inverse, and Armijo backtracking starting at1. The Armijo criterion
requires energy decrease, not decreasing gradient norm or Wolfe curvature.
Secants enter only for positive relative curvature; old pairs are tangent-
transported and re-screened at each point. A positive matrix can yield a much
larger direction in a soft or nonconvex region. History capacity10 alone
does not reveal the actual retained pair count, curvature or conditioning.
These quantities and the pre-retraction direction were not checkpointed.
Thus there is no evidence yet for either an optimizer defect or a successful
approach to a new minimum. The inherited label “screened Fock spectral gaps”
does not imply a newly computed Fock operator on this continuation.

The smallest next discriminator is an offline replay of the numerical
two-loop history through point12, using only stored x/g and original inverse
blocks. It must reproduce the actual step13 slope and polar endpoint before
its direction is used. Reconstruct actual retained/dropped curvature pairs,
direction norm and angle to the original-metric steepest direction. All
13 earlier x/g records are needed to identify which pairs survived; reading
only a guessed last10 would silently change history. No H/C arrays need be
decoded, no new geometry/metric may be constructed and no optimizer is run.

If that replay passes, one separate fixed-point derivative experiment at
literal13 can test the new high-gradient region. Use the transported,
normalized actual12→13 direction, fixed h1e−5 and5e−6, fresh complete1595 CI
at the baseline and four trials, and own same-point fullsigned gradient
controls. Preserve ordinary reported EOL and the original FD gate
`max(1e−8, 1e−5*abs(analytic))`; centered mixed-Rayleigh diagnostics remain
separate. This is a proposed5H/5CI/3gradient diagnostic, not an optimization
or inherited whole-space certificate. No h scan, seed choice, new cutoff,
changed weights, source adoption or automatic continuation follows.

The [proposal](contract.json) and [scalar observations](observations.json)
are source/JSON only. Both next numerical stages await independent root
contracts and exact input admission. No new numerical evidence is claimed.
