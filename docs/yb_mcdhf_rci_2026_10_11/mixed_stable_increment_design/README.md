# Mixed-CI stable energy increment: conditional design

This is a **read-only design, not an implementation or a new calculation**.
The current 531-CSF public optimizer already supports immutable original-metric
resume. It remains native Atomx. Its sealed source epoch is
`10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f`.
No source, existing proof, optimizer setting, state weight, physical model or
stationarity gate changed here. See the [contract](contract.json),
[source binding](source_identity.json) and [read-only verification](verification.json).

**The present run does not demonstrate an energy-resolution obstruction.** The
frozen [endpoint annex](endpoint_annex.json) reads all 161 stored EOL values from
the root-run [actual result](../public_mixed_resume_run002/result_unverified.json)
under its separate [saved-JSON contract](endpoint_annex_contract.json). It reached
the declared 160-iteration limit with zero
line-search fallbacks, 202 energy trials and 163 full gradients including two
baseline controls. Its independently replayed residual is 4.7170476466e−5,
above 1e−7; its indicator is 6.8897239138e−9 Ha, below 1e−8 Ha. All 160 accepted
EOL changes decreased. Binary64 diagnostic subtraction gives the last change
−2.331944415345788e−9 Ha, **1,282** endpoint floating spacings; the minimum
accepted decrease, step125→126, is −1.4733814168721437e−10 Ha, **81** spacings.
The [988-check independent endpoint replay](../public_mixed_resume_run002/independent_endpoint_verification.json)
passed its identity controls and failed the original residual gate. No H, CI,
gradient or optimizer call was performed by this design audit. The original
prospective design contract remains unchanged; the separate finalized endpoint
annex binds the actual immutable result, replay and checkpoint160 by SHA.

The next useful investigation is a bounded same-point continuation and
optimizer-conditioning diagnosis with the original metric. The stable increment
below is a contingent numerical option. Deep-core energy magnitude,
nonmonotonic residuals or exhausted iteration budget do not establish its need.

## What is already implemented and what is missing

| Component | Actual source behavior | Design boundary |
|---|---|---|
| `CSFOrbitalFunctional.evaluate` | Full native CI; physical-reference tracking or fixed selectors; EOL is a floating weighted sum of reported eigenvalues | Does not store the full H in `CIResult` |
| `fixed_density_energy_difference` | Complete quadratic/quartic finite orbital increment | Explicitly rejects mixed, tracked, charge and multi-CSF block cases |
| `constrained_energy_difference` | Same fixed-density guards; removes a complete finite S-Gram constraint defect on guarded endpoints | Current multipliers cannot be copied into a mixed target without a derivation |
| `CIRadialDensity.contract` | Full interference, but discards each block coefficient with magnitude ≤1e−15 | That cutoff is not an exact-difference primitive or a quantified gradient error bound |
| `contract_bilinear(left,right)` | Full signed angular table, real symmetrized bilinear amplitudes, removes exact zero terms only | Does not normalize CI vectors; caller must retain their actual norms |
| `energy_for_weights` | Raw one-body polynomial and cell Coulomb contraction | Its ambient one-body convention lacks the public CI evaluator's ket-norm division; it is not an independent off-manifold CI oracle |
| `CIBasisTracking.resolve` | Frozen physical reference; complete equal-weight spans; majority principal overlaps and ambiguity checks | Abstract coefficient overlap alone cannot match changing physical bases |
| `optimize_orbitals` stable callback | Used only near a small direct total-energy difference; scalar or explicit unsupported `None` | Does not consume a typed interval/error budget; the active run has `energy_difference=None` |

This audit reads the current available resume interface, rather than repeating
the historical pending-interface statements in the immutable
[older assessment](../mixed_ci_stationarity_probe/PUBLIC_OPTIMIZATION_SUCCESSOR.md).
Its mathematical observations remain useful within their old source epoch.
See the actual [run contract](../public_mixed_resume_run002/contract.json) and
[resume source](../../../src/atomx/numerics/bspline/orbital_optimization.py).

## Retained finite-norm algebra

In the same complete real signed CSF order, first physically match one isolated
target at two radial endpoints. Let Hᵢ be the actual real symmetric matrix,
cᵢ its selected coefficient vector, nᵢ=cᵢᵀcᵢ>0, λᵢ the stored eigenvalue,
d=c₁−c₀, A₀=H₀−λ₀I and rᵢ=(Hᵢ−λᵢI)cᵢ. Then

\[
\lambda_1-\lambda_0=
\frac{c_1^T\delta Hc_1+d^TA_0d+2d^Tr_0+c_0^Tr_0-c_1^Tr_1}{n_1}.
\]

This is exact algebra for finite norms and residuals. It neither assumes c₀
is an exact eigenvector nor silently changes stored λ to a Rayleigh quotient.
For q₀=c₀ᵀH₀c₀/n₀, set A₀=H₀−q₀I and r₀=A₀c₀. The quotient identity is

\[
q_1-q_0=\frac{c_1^T\delta Hc_1+d^TA_0d+2d^Tr_0+c_0^Tr_0}{n_1}.
\]

In exact arithmetic c₀ᵀr₀=0 for that quotient; the numerical contraction still
has to be carried and measured. The quotient-to-reported difference is exactly
qᵢ−λᵢ=cᵢᵀrᵢ/nᵢ. CI norms, orbital ket norms and ensemble weights are three
different normalizations.

The previous [239-check saved-matrix replay](../mixed_ci_stationarity_probe/mixed_difference_algebra.json)
passed on 45 matched pairs. Its maximum identity errors were 1.47e−21 Ha for
finite reported λ and 4.28e−16 Ha for quotients in an extended type with **63
mantissa bits**. `float128` there is a NumPy type name, not proof of IEEE
quadruple arithmetic. No new algebra replay was executed here.

The previously accepted one-step diagnostic has the following saved EOL terms:

| Term | Ha |
|---|---:|
| Trial signed density applied to δH | −5.83156466322e−4 |
| Quadratic CI response | +3.18469819673e−4 |
| Reference residual cross term | +8.87862082773e−15 |
| Reference residual offset | +2.37715493113e−13 |
| Negative trial residual offset | −8.76283384094e−13 |
| Finite reported-root difference sum | −2.64686647279e−4 |
| Old-density frozen radial increment, comparison only | **+5.04817321150e−5** |

Thus neither frozen old density nor omission of CI response is exact even at
that finite step. These measurements are not a diagnosis of the current
160-step run. The old replay subtracts already-rounded saved H matrices; it
does not prove accuracy of a new tiny-step radial polynomial.

## Smallest reusable prototype

The proposed first implementation has three numerical pieces and one method
consumer. The following names/signatures are proposals, not public capabilities:

```python
# In the existing CSF functional; existing fixed-density public guards remain.
_signed_weights_radial_increment(reference, trial, signed_weights, *, increments)

# Pure-array algebra, no methods/physics imports into the numerical engine.
real_rank_one_ci_response(H0, H1, c0, c1, lambda0, lambda1, *, semantics)

# The method consumer owns physical target, radial/kernel and source policies.
mixed_ci_energy_increment(
    reference, trial, reference_ci, trial_ci,
    *, increments, reference_matrix, trial_matrix, target_binding, semantics,
)
```

1. Factor the existing private finite polynomial into a helper taking explicit
   signed radial weights. Have the existing fixed-density wrapper call it after
   its current guards. Preserve every linear, quadratic, cubic and quartic term,
   source-pair streaming, radial key order and caches. Removing the guards alone
   does not implement mixed CI.
2. In the numerical response helper, compute actual n₀,n₁, phase-aligned d,
   the reference shifted response and both endpoint residual offsets. Lift
   inputs before diagonal centering if using extended precision. Reject a
   platform with no additional mantissa bits when that mode is declared.
   No residual term is discarded because a loose CI gate passed.
3. In the method consumer, resolve the same immutable physical target at each
   endpoint. For each target form
   `contract_bilinear(c1, c1) / (c1.T @ c1)` and sum the unchanged five weights
   of 0.2. Preserve all signed off-diagonal interference. Sum densities before
   streaming the radial polynomial, then add the separately weighted response
   and residual terms.
4. Return a structured **diagnostic** containing each decomposition, norm,
   residual, source/target binding, dtype, component error estimate/bound and
   explicit supported/unsupported reason. First qualify it on saved/frozen
   matched data and independent small oracles. Do not enable the active solver
   while preparing this prototype.

The response algebra needs the actual symmetrized H₀ and H₁. Current `CIResult`
has energies, mixing and provenance, no matrix field. A consumer must supply
source-bound matrices explicitly; it must not read a guessed optional attribute
or infer H from approximate eigenpairs. For a future callback, an explicitly
budgeted lazy reference matrix build is possible once per exact accepted point;
trial matrix rebuilds, if needed, also count as physical work. A kernel cache
binds complete orbital bytes, geometry, Z, sampled nucleus, signed basis,
operator and target identity. Changing the trial invalidates its contents.

No new global state/cache or backend abstraction is needed for the first
prototype. The method composes existing angular/radial machinery; a pure-array
response function can live under `numerics`, which must not import the method.
If a second physical consumer later needs the radial increment, move the shared
kernel through the established `physics/csf` boundary with matched tests rather
than duplicate the current Coulomb formula.

### Exact radial normalization and ordered symmetrization

The public B-spline radial evaluator uses hₐᵦ=aᵀh_D b/(bᵀSb). With explicit
δa,δb, compute the complete quadratic δN and δs, then

\[
\delta h_{ab} = (\delta N-h_{ab}\delta s)/(s+\delta s),\quad s=b^TSb.
\]

Retain ordered hₐᵦ and hᵦₐ. The final public real Hamiltonian is symmetrized;
the bilinear contraction must reproduce that exact mathematical symmetrization,
including off-diagonal one-body terms. It is unsafe to assume ambient hₐᵦ is
symmetric merely because h_D is symmetric and normalized endpoints are close.

For a Coulomb pair density ρ(a,b), use
δρ=ρ(δa,b)+ρ(a,δb)+ρ(δa,δb). If K is the same declared cell kernel,

\[
\delta R=\langle\delta\rho_{ab},K\rho_{cd}\rangle
 +\langle\rho_{ab},K\delta\rho_{cd}\rangle
 +\langle\delta\rho_{ab},K\delta\rho_{cd}\rangle.
\]

Do not drop the quadratic pair increments or their mutual product. The existing
functional uses semiseparable action at ns≥96; the current CI radial evaluator
uses its bank/banded representation. For the actual ns123 model those two
arithmetics require a matched closure test; shared physical formulas alone do
not give a near-roundoff arithmetic identity. No matrix-kernel migration is
proposed in this first increment.

Supply P₊δx from the actual original coordinate increment. Do not recreate it
by subtracting two rounded P₊x endpoints. Bind the stored endpoint reconstruction
and complete S-Gram defects. Explicit increments and reconstructed endpoints
have distinct rounding identities and must be archived together.

### Phases, complete spans and state identity

The initial prototype supports only the five real isolated rank-one targets
in the unchanged full531 space. Physical tracking precedes coefficient sign
alignment. A sign flip may be chosen to reduce d, but the formula must be
invariant when all associated residuals are transformed consistently. Reject
an ambiguous physical target or singular/unsupported gauge; never choose a
different root to make d small.

A future complete equal-weight cluster must consume its **subspace**, not paired
individual columns. For arbitrary full-rank real C, G=CᵀC, its projector trace
energy is `tr(G^-1 C.T H C)`. The exact finite Gram must be retained. This is
invariant under nonsingular changes of span coordinates. It is not identical
to a sum of normalized individual-column quotients when finite off-diagonal
Gram defects are present; any change of finite objective must be declared.

For exactly orthonormal frames Q₀,Q₁ of the same matched rank, choose an
abstract Procrustes gauge only after physical span matching. With D=Q₁−Q₀,
B₀=Q₀ᵀH₀Q₀ and R₀=H₀Q₀−Q₀B₀, the trace identity is

\[
\Delta\operatorname{tr} = \operatorname{tr}(Q_1^T\delta H Q_1)
 +\operatorname{tr}(D^TH_0D-D^TD B_0)+2\operatorname{tr}(D^TR_0).
\]

Finite numerical Gram defects, frame whitening and stored reported-root offsets
need their own complete corrections and bounds before use. Multiply by the
unchanged **per-level** equal weight, not by an invented statistical weight.
Do not rotate unequal weights, split an exact degeneracy, replace physical
overlap with coefficient overlap, or assert column equality inside a cluster.
Rank>1 support is deliberately outside the first prototype.

## The near-roundoff limit must be measured, not inferred

There are two distinct channels:

- The finite-λ identity reproduces differences of the already-rounded stored
  eigenvalues. Including its exact residual offsets cannot also recover a
  decrement that those stored λ values have already lost. Direct floating EOL
  averaging introduces another rounding offset. Adding that offset blindly
  restores the cancellation being investigated.
- A stable normalized-quotient polynomial estimates a continuous energy of the
  retained vectors. Calling it the reported eigenvalue difference without the
  exact quotient-to-λ offsets would silently change finite semantics. Calling
  it the true matched eigenvalue difference needs a bound on eigenvector,
  assembly and arithmetic errors.

If r_q=(H−qI)c, then `||r_q||/||c||` bounds distance of q to **some** eigenvalue
of the same real symmetric matrix. To bind the intended state, retain isolation
and physical tracking. A quadratic residual/gap bound needs independently
verified isolation and the applicable spectral hypotheses; it is not justified
by a nominal gap or by CI residual<1e−8 Ha alone. H assembly error is separate.

For omitted finite residual pieces a simple conservative bound is

\[
B_r \le
(2\|d\|\|r_0\|+|c_0^Tr_0|+|c_1^Tr_1|)/n_1.
\]

The prototype retains these pieces; their **numerical evaluation** still has
error. Lift H before centering its diagonal, bound the actual operation sums
and input rounding, and report extended-precision capabilities. A compensated
sum controls final accumulation, not preceding matrix products or angular
contraction error. Large common offsets, residual floors, complete interference
and both signs of finite differences belong in the independent oracle.

The current gradient uses the filtered ordinary contraction. The energy
polynomial must use the unfiltered density. A prospective exact identity must
therefore record, not hide, its relationship to the accepted gradient:
Σ|omitted coefficient|·|actual integral increment| bounds omitted energy terms
when those complete values are known, and the derivative needs its corresponding
actual metric bound. A per-coefficient 1e−15 cutoff is not an ensemble error
budget. Keep the unchanged gradient/residual yardstick for qualification; any
subsequent normalization or contraction policy change is a separate source epoch
with full matched physical verification.

Constraint centering also needs an independent derivation for the mixed,
ket-normalized quotient objective. A floating Gram guard by itself is not a
decrement-sign bound. The known leading normal contribution has a bound
`||Lambda||_F * ||delta_Gram||_F`; replacing it by zero is not exact. The
complete quadratic δ(CᵀSC), variation of multipliers and any remaining
constraint-centering error must be included. Begin with the uncentered exact
polynomial; add centering only if measured normal contamination requires it
and an independent complete-gradient/finite-step identity qualifies it.

### Conditional solver adoption

A future bounded consumer must retain an interval
`[delta_estimate - B, delta_estimate + B]` for the same declared EOL objective.
The original Armijo coefficient remains 1e−4. Acceptance is safe only if the
**upper bound** is below the unchanged actual Armijo RHS. A valid scalar adapter
could return that conservative upper bound and separately archive the estimate,
bound and all components; a typed engine result would expose them directly.
Both options require an explicit new source/consumer contract, not an invisible
change in this running solver.

`None` means unsupported scope. Numerically unresolved increments must not
silently fall back to a favorable rounded total-energy subtraction. Current
scalar callback behavior lacks that distinction. The smallest consumer may
return a rejected nonfinite comparison with an explicit diagnostic; a public
typed distinction is needed before asserting robust general behavior.
Neither option changes the original gradient or either stationarity gate.

## Independent qualification sequence, not executed here

1. Freeze a complete accepted checkpoint and representative rejected/accepted
   trials. Retain actual EOL differences, coordinate steps, Armijo derivative,
   step size and termination diagnostics. If no rounding-related line-search
   obstruction is demonstrated, stop and prioritize the measured mechanism.
2. Test the pure-array response against independent high-precision full
   normalized quotients and reported-λ identities. Include nonunit norms,
   nonzero residuals, signed off-diagonal interference, sign changes, large
   common offsets and explicit unsupported degeneracies/complex/space changes.
3. Anchor the shared radial polynomial against a small independent
   high-precision complete signed Hamiltonian using the retained native angular
   and cell coefficients as inputs. Include ordered ket norms, final symmetry,
   zero, finite forward/reverse and explicit tiny increments. That is an
   arithmetic oracle for the same model, not another physical backend.
4. Retain all old fixed-density controls. Independently verify the actual ns123
   bank/semiseparable closure, full trial density and all CI-response terms.
   Do not infer a 1e−14 bound from an old 3e−11 finite-step comparison.
5. Bind a new source epoch and a separately authorized bounded diagnostic to
   the same full531 model, five physical targets, original projector, inverse
   metric and gates. Compare diagnostics first; enable a bound-qualified
   callback only after its real decrement error is smaller than the decision
   margin. Safe checkpoints and budget stops remain mandatory.

Success of those numerical controls would qualify only the specified native
mixed-CI energy comparison. It would not establish expanded MCDHF stationarity,
complete core correlation, Breit/QED/recoil accuracy, experimental agreement,
performance speedup or highest atomic accuracy.

## Handoff and documentation disposition

The [verifier](verify_design.py) uses only Python's standard library to check
immutable files, complete source maps, syntax/actual interface boundaries,
registered old evidence and document links. It never imports Atomx, NumPy or
SciPy and never loads numerical arrays. Its passed checks are documentation and
source-binding checks, not a newly passing scientific test suite.

The existing `preserve-metric` insight already covers the measured constraint
and roundoff boundary. This successor adds an actionable candidate plan within
that scope; no global rule or current public capability changed. Root owns
consolidated navigation and any later implementation. Historical source-bound
assessment and failed evidence remain unchanged. No new reusable empirical
insight is promoted from this design alone.
