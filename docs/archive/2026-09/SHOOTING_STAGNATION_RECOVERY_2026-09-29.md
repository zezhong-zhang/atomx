# Adaptive recovery of frozen-exchange orbital plateaus

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Evidence

The corrected first20 campaign recorded adaptive-bootstrap failures for N V
`1s2 5g1`, F VII `1s2 6g1`, and Ne VIII `1s2 6g1`. Each reached its iteration
budget with an orbital-update defect above the unchanged acceptance threshold.
Explicit nonlocal-resolvent bootstrap and refinement converged on both grids.
All three pairs passed the independent recovery audit and are included in
`native_shooting/corrected_recovered_comparison.json`, preserving their original
failed attempts and recording the recovery provenance per endpoint.

## Numerical change

`numerics/shooting/scf.py` now retains a 12-sweep history of each undamped
orbital-update norm. Once the energy-change criterion is satisfied, a singleton
kappa channel is promoted to implicit exchange if every update in its window
remains above the orbital tolerance and the minimum has not dropped below
80 percent of the first update. This targets stalled fixed-point iterations.
Same-kappa constrained channels are excluded; they require their own constrained
operator treatment. No nuclear model, Hamiltonian, root-selection criterion,
energy tolerance or orbital-update threshold is changed.

The existing implicit solver applies exchange to trial orbitals and retains its
node, overlap and Green-equation residual checks. Promotion is permanent within
the SCF solve, avoiding alternation between inconsistent frozen and implicit
updates. The result diagnostics now expose `nonlocal_stagnation_promotions`.
The running first20 campaign remains on its original frozen source snapshot;
this change applies to newly launched source snapshots.

## Verification

At 3200 points with the same 1e-12 bootstrap tolerance, adaptive SCF now converges
in 32 (N), 28 (F), and 28 (Ne) iterations, versus the previous 400-iteration
failures. Final undamped updates are respectively 8.41e-7, 5.98e-7 and 5.83e-7,
all below 1e-6. Maximum implicit Green residuals are below 3.3e-11.
Timing includes machine/JIT effects and is not a general performance claim.

The recorded driver and numerical outputs are in
`normalization_diagnostic/stagnation_recovery/`. Regression tests compare the
recovered adaptive solution with an independent implicit-bootstrap solve for
all three configurations, including total energies and normalized radial spinors. Existing SCF
convergence and nonlocal-eigenproblem tests also pass.

A fresh recovery campaign using the updated adaptive solver completed all six
attempts for F VII 7g, Ne VII 7i and Ne VIII 7g. All three grid pairs passed the
independent audit. The two newly recovered configurations are incorporated in
the current website snapshot, which now includes eight recovered configurations
(3251 accepted pairs at publication). Earlier accepted Ne VIII 7g recovery has
priority, so the repeated control is not double counted.
