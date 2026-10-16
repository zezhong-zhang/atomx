# Native Yb MCDHF → RCI physics audit

The user-selected main route is native Atomx MCDHF followed by RCI. This audit
does not calculate new Yb roots or measure an improvement in its experimental
gap. It verifies the implemented physics, identifies a real no-pair input bug,
and recomputes differential scales from preserved Yb evidence. Ne's independent
high-precision spectrum and Eu's fitted native evidence remain valid within
their original models; changing the Yb priority does not revoke them.

The machine-readable ledger is [results.json](results.json); its twelve inputs
are content-bound. The original code epoch is in
[source_manifest.json](source_manifest.json), and the subsequent optimizer
repair has a separate [saved-input replay](fock_projector_replay.json).
The [pre-disposition report](results_before_post_audit_repairs.json) preserves
the original unresolved items; the current ledger records their subsequent
repair separately.

## What the native implementation actually does

| Stage | Implemented shared machinery | Boundary for Yb |
|---|---|---|
| CI expansion | Signed jj CSFs; same scalar Hamiltonian angular coefficients in CI and RDM | Enumerating references or SD substitutions is not full correlation convergence |
| Orbital objective | `CSFOrbitalFunctional`: fixed CI level weighted Dirac–Coulomb eigenenergies | Breit/QED/recoil do not enter this orbital gradient |
| Correlated gradient | Complete retained CI amplitudes contracted into signed one-/two-body radial weights; analytic differentiation of the same cell Slater integrals | Actual finite-space gradient and stationarity must be tested; RDM implementation also has a numerical coefficient filter at `1e-15`, not an error bound |
| Balanced target | `CIStateAverage`, explicit `(2J, parity, root, weight)` selections in common B-spline orbitals | Root indices are ordered within symmetry; they do not track configuration character across crossings |
| Radial constraints | Same-κ metric orthogonality, polar retraction, positive-energy nuclear or fixed initializer-Fock space | These are distinct no-pair model choices; changing one requires a matched spectrum comparison |
| Convergence | Constrained tangent gradient, preconditioned residual and energy-error estimator | Small energy changes or a successful CI diagonalization do not replace stationarity |
| Static RCI | Full signed static Breit, including retardation, in the final CI Hamiltonian | Scalar uniform Breit comparison cannot change excitation energies or splittings |
| Dynamic RCI | Fixed supplied orbital-difference real transverse operator replaces static Breit | The frequency map is an explicit physical choice; arbitrary correlation labels/optimization multipliers are not automatic spectroscopic energies |
| QED | First-order resolved CI expectations; local VP uses full radial off-diagonal matrix; available SE model matrix or diagonal model | No universal precision/nonlocal SE coverage, no degenerate-manifold rediagonalization, no QED orbital response |
| Recoil | Explicit one-/two-body CI expectations and application ledger; mass/mode validated | Lowest-order/model operator coverage and isotope must be declared; virial NMS is a proxy |

The common-orbital full-RDM gradient is the correct reusable path for the new
five-state EOL run. `per_block_orbital_sets` instead executes population-averaged
DF followed by block CI; it is not a substitute for full correlated EOL
stationarity. The public interface rejects combining that route with state
averaging. Shooting state averaging is also explicitly unsupported.

Code owners: [functional](../../../src/atomx/methods/csf/variational.py),
[RDM](../../../src/atomx/physics/csf/radial_density.py),
[optimizer](../../../src/atomx/numerics/bspline/orbital_optimization.py),
[solver/RCI staging](../../../src/atomx/methods/csf/solver.py), and
[transverse contract](../../TRANSVERSE_INTERACTION.md).

The conceptual reference is the MCDHF/RCI sequence in
[GRASP's primary repository](https://github.com/compas/grasp), which separately
provides `rmcdhf`, `rci`, angular, transition and biorthogonal programs. This is
a method-level comparison inferred from the two implementations, not a claim
that their projectors, quadratures, optimizers, active spaces or correction
models are identical. GRASP supplies independent reference runs; Atomx retains
its own physical producer. The requested GRASP theory page returned HTTP 429
during this audit and was not used to infer additional numerical equivalence.

## A verified correctness defect before expanding the space

The public solver supplied a complete `h_nuclear + HF` preconditioner matrix
for each orbital. The old screened-projector branch added `h_nuclear` again
after averaging those matrices. An actual public Z=10 three-configuration
initializer was captured before repair; its three κ matrices satisfy
`actual − average(full Fock) = h_nuclear` within `7.276e-12 Ha`.
This is an operator identity defect, not a measured Yb line shift.

The parent agent repaired the single addition and documented the seed-fixed
projector. This agent independently replayed the saved matrices through the
repaired real optimizer: input matrix and Dirac spectrum differences are zero,
positive-projector spinor leakage is below `9.655e-14`, and Gram error is below
`6.662e-16`. The new
[regression tests](../../../tests/test_mcdhf_fock_projector.py) use a real CI
functional and two different screened complete Fock operators in the same κ
with 2:1 weights. All three cases fail on the frozen old function and pass
after repair. The old light-atom endpoint comparison was insufficient to
detect this semantic defect.

Evidence: [initial contract](contract.json),
[old capture](fock_projector_result.json),
[exact executed script](executed_probe_fock_projector.py.txt),
[old regression failures](old_optimizer_test_failure.json), and
[corrected replay](fock_projector_replay.json). No production source was
restored to run the old failure tests; the old function was loaded only in an
isolated process's memory. The nuclear model's variational space is unaffected
by this screened-projector fix, but its old alternative-projector leakage
diagnostic was affected.

## Differential physics scales, with their original model identities

Every entry below concerns an upper-minus-lower excitation or splitting. A
common core total-energy shift is not used as a spectral error budget.

| Observation | Measured or evaluated scale, cm⁻¹ | Scope and implication |
|---|---:|---|
| Current large DC SD against cached ASD | 5d centroid +873.805; 6p centroid +320.179 | Different branch-specific errors remain; common energy offset alone cannot remove them |
| Current large DC SD splittings against ASD | 5d −128.436; 6p +104.790 | Relative spin-orbit/correlation response must be distinguished from centroids |
| Frozen-CI static minus DC | −65.542 to +27.587 | Same frozen five-state model; not an additive correction or upper bound for SD/new MCDHF |
| Frozen-CI dynamic minus static | −3.887 to −0.377 | Lower priority for explaining the current hundreds-of-cm⁻¹ defect in this measured control; not a correlated-transverse upper bound |
| Finite correlated CI full static Breit | −82.642 to +35.105 | Correlation changes the Breit differential; retained constant core energy cancels |
| One-hole to two-hole finite CI | +31.661 to +44.752 | This narrow model's added holes shift all four lines but do not certify complete CV or configuration mixing |
| Historical outer-core 5s/5p opening | +403.050 to +1453.179 | Different bounded nonlinear CC model; useful scale for testing CV, not an absent additive shift |
| Historical CCSD to CCSDT | −50.794 to +26.958 | Different bounded model; triple rank is a separate physical control |
| Historical default orbital SE upper minus 6s | +306.920 to +397.690 | Ground and 6p entries use a leading analytic coverage fallback; no precision uncertainty bound |
| Virial NMS estimate | −0.0953 to −0.0720 | Proxy, not a calculation of the full resolved recoil differential |

The self-energy control prevents an unsupported statement that QED must always
be negligible. Its leading-Zα fallback is also not evidence for a hundreds-of-
cm⁻¹ physical QED correction. A declared screened model must be evaluated on
the new orbitals with full state identity and uncertainty/coverage recorded.
The total physical error bound remains **unknown**. These historical models
must not be summed together or spliced into the new spectrum.

## Priority experiments

1. Retain the repaired operator identity and declare the no-pair model. Obtain
   one genuinely stationary common-orbital Dirac–Coulomb EOL result for all
   five states with fixed target weights, common nuclear model and energy zero.
   No experimental fit selects weights or shifts.
2. In one declared CSF space, compare frozen-orbital CI with EOL reoptimization
   and retain the full five-state upper-minus-ground response. This isolates
   radial relaxation from adding configuration space.
3. Precount a balanced outer-core CV/multireference increment, including 4f,
   5s/5p correlation and relevant 4f-hole partners. Track physical state overlap
   and report all five responses. Native Atom does not implicitly enable the
   existing lower-level `CIBasisTracking`.
4. On the same accepted spinors, compare DC → complete static Breit RCI →
   declared model-QED and recoil expectations. Preserve the operator ledger
   and normalized CI amplitudes. Each correction contributes its measured
   upper-minus-lower difference.
5. Allocate the next correlation, dynamic transverse or response increment
   using those measured differences and actual resource cost. Expanding an
   active space approaches the chosen finite Hamiltonian's solution; it does
   not establish that omitted physical operators or experimental matching are
   correct.

The original audited orbital functional/RDM did not forward the public
determinant cap; the parent has now added that propagation and its dedicated
regressions. The original identity remains archived, rather than being
silently rewritten. `n_states=None` still forces a full dense CI spectrum
inside each orbital trial even though the shared public CI solver supports
partial Davidson roots. Active vs
frozen orbital optimization also needs an explicit contract; expansion-frozen
core holes are not the same as fixed core radial orbitals. Angular tables and
signed connections should remain reusable; trial-dependent radial values must
remain bound to the new orbital snapshot. None of these audits certifies a
CPU/GPU or end-to-end speedup.

## Stable line-search energy differences

The initial native Yb run reported a failed stationarity gate near a large
`−14067 Ha` total, where subtracting independently rounded totals can lose a
tiny line-search decrement. The convergence gate should remain unchanged.

For the initial five references there is one CSF in each target symmetry
block, so the weighted CI RDM is fixed. An exact, stable finite-Hamiltonian
difference can be computed from actual retracted orbital increments:

```
Δh_ab = δaᵀ h b + aᵀ h δb + δaᵀ h δb
Δn_b = δbᵀ S b + bᵀ S δb + δbᵀ S δb
Δ(h_ab / n_b) = [Δh_ab − (h_ab / n_b) Δn_b] / (n_b + Δn_b)
δρ_ab = δa b + a δb + δa δb
ΔR_ab,cd = δρ_ab K ρ_cd + ρ_ab K δρ_cd + δρ_ab K δρ_cd
```

These expressions retain every quadratic/quartic term and use the same cell
kernel. They are not a truncated Taylor approximation. Sum signed terms with
compensated accumulation, compare the exact difference with Armijo, and keep
the original tangent-gradient stationarity criteria. For a larger CI space,
the CI density changes as well: either construct the centered trial CI matrix
using stable integral increments and diagonalize it, or include the complete
CI-density response in the difference. Applying a fixed-RDM increment to a
changing CI state would omit physics. A trapezoidal gradient integral can be
an independent discriminator; without a controlled remainder it is not the
sole proof of energy descent.

The parent implemented the restricted unchanged-RDM callback and uses it only
near the total-energy rounding limit. It rejects multiple-CSF blocks,
charge-functional subclasses and physical-root tracking; a differing
contracted objective density also returns `None`. The existing CI solve is
retained at every trial. This is a complete finite-polynomial rearrangement,
with the same floating-point cell banks and angular coefficients; it is not
an arbitrary residual-improvement acceptance rule.

An [independent four-control review](stable_difference_review.json) compares
the callback with full `h/n` and all four component `rkbT` contractions in
long double, without using production increment or gradient routines as the
oracle. It exercises real equal-weight J=0/J=1 one-CSF blocks, a finite
two-orbital quartic change, `1e-8`/`1e-12` positive-space admixtures, and a
ket-norm defect within the unchanged existing metric audit. The largest
stable-minus-oracle discrepancy is `1.995e-17 Ha`; the `1e-12` step has the
same nonzero sign and differs by `4.657e-18 Ha`, while double total subtraction
loses about one third of its change. Forward/reverse changes pass. The
raw angular density in this finite control contains no nonzero omitted
`1e-15` terms. This does not bound that filter in every other model or validate
Yb stationarity. See the [prospective contract](stable_difference_contract.json)
and [independent oracle](review_stable_difference.py).

The native agent's later [real Yb angular audit](../native_run/stable_polar_endpoint/result.json)
does contain `264` nonzero block coefficients removed by the existing
`1e-15` filter, out of `2723` nonzero block entries; their largest magnitude is
`3.053e-17`. These may include floating-point angular cancellations, but this
count alone supplies no energy-difference bound. A later
[paired real-Yb diagnostic](../native_run/radial_constraint_endpoint002/result.json)
recomputed complete unfiltered and retained finite-polynomial differences
on the same eight actual radial increments. Their largest difference is
`3.255e-25 Ha`; this identifies the filter as immaterial to those particular
`1e-12 Ha` discrepancies. It does not certify arbitrary angular truncation,
and the two-electron oracle's zero omitted count remains scoped to that control.
Neither diagnostic establishes a successful stationary Yb solve.

## A numerical successor for the same trial polar map

A further native run still failed stationarity after stable energy differences
were available. The parent therefore introduced a near-unit polar evaluation
that retains the original seed initialization, fixed positive space, objective,
preconditioner and convergence gates. With `D = YᵀY − I`, `||D||∞ < .01`
guarantees a positive Gram matrix and convergence of
`(I + D)^(-1/2) = I + Σ binom(-1/2,n) D^n`. Computing the small correction in
wider precision and returning `Y + Y C` avoids reconstructing identity from
nearly degenerate eigenvectors. Farther from unity the original eigensolve
continues to evaluate the same polar map. The finite series retains a tail far
below float64 output rounding at the declared threshold.

The [independent polar panel](polar_review.json) uses dense 64×8 coordinates,
four tangent-step sizes, both sides of the branch boundary, a right orthogonal
rotation and a column phase flip. Its 75-decimal spectral oracle does not use
the production series. All nine controls and rank-deficiency rejection pass;
near-unit map discrepancy is at most `2.776e-17`, Gram defect at most
`5.552e-16`, and rotation equivariance defect `8.327e-17`. The original
near-unit eigensolve differs from the oracle by up to `7.216e-16` in this
panel. These are representation errors, not cm⁻¹ physics errors. No Yb
stationarity or gap improvement follows from them alone. The
[contract](polar_review_contract.json), [oracle script](review_polar.py) and
separate source epoch retain that distinction.
Exact source for both numerical follow-ups is in the
[post-audit source manifest](post_audit_source_manifest.json). The original
pre-repair snapshots and failed regressions remain separate.

## Same-manifold energy centering and actual increment identity

The real Yb diagnostic resolved another mechanism: its original endpoint had
native radial Gram defects up to `2.384e-15`. Reapplying even a nominal zero
polar step changed the complete off-manifold `h/n` energy by
`2.794e-12` or `4.486e-12 Ha`. The derivative of the actual ket-normalized
one-body extension explains these changes to about `1e-26 Ha`; the raw
polynomial one-body gradient has a different normal derivative and can
predict the wrong sign. This is a representation diagnosis, not a missing
physical correction or proof of stationarity.

The parent added a separate `constrained_energy_difference` API. With
`G=CᵀSC`, actual quotient gradient `g`, and
`Λ = ½ sym(Cᵀg)`, it evaluates
`ΔL = ΔE − Tr(Λ ΔG)`. This Lagrangian equals the physical energy on the exact
metric-orthonormal manifold. Its Gram increment must be complete:
`ΔG = CᵀSδC + δCᵀSC + δCᵀSδC`. Omitting the quadratic term would subtract
legal polar curvature and alter the constrained physical energy. All
same-κ off-diagonal multipliers are retained. The original complete
`fixed_density_energy_difference` API remains unchanged.

Both reference and stored trial must satisfy a floating-scale native Gram
guard. For declared linear-map increments, the reconstructed mathematical
trial must also satisfy that guard and agree with the stored trial within
the declared representation tolerance. Optimizer increments now use the
same fixed projector applied to `candidate−reference`, avoiding subtraction
of two independently reconstructed deep-core arrays. One-CSF block and
identical objective-RDM restrictions remain; there is no frozen response
substitution for mixed CI. Physical residual and energy-error gates remain.

The [independent actual-API review](constraint_center_review.json) compresses
the complete actual reference/increment span with all four component cell
tensors, then evaluates full `h/n` and quartic energy at 80 decimal digits.
The compression itself is 63-bit, with explicitly Hermitian H/S; it is not
an 80-digit reconstruction of the original cell operators. An independent
automatic derivative matches the analytic quotient derivative to
`4.01e-80`, and native projected gradient differs by `8.88e-16 Ha`.
The same-κ off-diagonal multiplier is `0.60194 Ha`, so this is not merely a
diagonal control. Six formula controls and six actual-API controls pass:
at `1e-12` tangent steps API-vs-full-centered errors are at most
`9.33e-29 Ha`, while centered-vs-metric-polar normalized discrepancies are
about `2.9e-28 Ha`. On a genuine `1e-3` polar step the API discrepancy is
`1.96e-19 Ha`; incorrectly dropping quadratic Gram curvature would change
energy by `8.956e-7 Ha`. A `1e-8` normal defect returns `None` as required.
The [initial formula contract](constraint_center_contract.json),
[prospective actual-API extension](constraint_center_api_contract.json) and
[oracle script](review_constraint_center.py) retain their distinct scope.

At finite Gram defects, centering is only first-order equivalent to
canonical metric-polar normalization. The remainder includes multiplier
response times trial defect and quadratic defect terms. A roundoff guard
alone is therefore not a universal sign-error bound; strict acceptance near
zero still needs a source/model-specific margin or controlled remainder.
The small-model discrepancies above cannot supply such a bound for arbitrary
large Yb steps. The native saved endpoint shows a centered decrement around
`−4.49e-13 Ha` at step `.1`, but its gradient linear diagnostic is not the
sole energy oracle, and no production solve is certified here.

The first high-precision review failed because long-double projection
introduced non-Hermitian last bits (`3.966e-19 Ha` in H and `3.578e-20` in S).
Its [failed result](failed_constraint_center_compression.json) and exact
script are retained. Symmetrizing the physically Hermitian compressed
operators restored the strict derivative and Gram identities without
relaxing gates or changing production sources. The exploratory actual-API
panel is also retained before its prospective repeat.

## Validation and limitations

The saved ledger independently recalculates centroids, splittings, correlation
increments and correction differences. Original experimental 1% failures are
retained. Tests verify matrix identity and constrained representation, not
Yb experimental accuracy. The audit's initial replay omitted the workspace
`ks` metadata and failed before physics; the corrected replay restores the
declared unbalanced order 9 and retains the failure record. No measured
Yb improvement, complete correlation convergence or highest-level claim is
made by this evidence.

The wider-precision review first failed only while serializing a NumPy boolean;
its [failure identity](failed_stable_difference_serialization.json) and exact
script are retained. Casting the diagnostic gate to Python `bool` changes no
physical quantity or threshold. The final checks are recorded in
[verification.json](verification.json), and the documentation disposition in
[documentation_impact_review.json](documentation_impact_review.json).
