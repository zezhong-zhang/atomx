This is an independent source and equation review of the frozen [saved-point centering consumer](../correlated29_constraint_centering_diagnosis/contract.json). It performs no numerical replay. The original reported and Rayleigh derivative failures remain failures; the prepared optimizer remains on HOLD.

The proposed control is admissible as a **conditional Q-Lagrange diagnostic at one existing baseline and its original two signed trials**. It is not an extension already implemented in the public optimizer, a replacement for the reported production objective, or a uniform error bound. The existing [public constrained difference](../mixed_ci_finite_increment_implementation/source/atomx/methods/csf/variational.py) explicitly rejects tracked or mixed-CI blocks.

Let Vκ contain every same-κ physical radial column, Sκ be its original metric, and Kκ=VκᵀSκVκ. For a baseline-frozen symmetric Λκ, define Γ=Σκ Tr(Λκ ΔKκ). Then ΔLλ=ΔFλ−Γ has exactly the same reported objective difference on the exact orthogonal manifold: both endpoints have Kκ=I, so Γ=0. For stored floating endpoints, this is a chosen extension away from the manifold, not the literal stored reported energy difference. A diagnostic built from ΔQ has the same statement for Q; this does not identify Q and reported λ.

The complete finite constraint increment is

```text
ΔK = V0ᵀ S δV + δVᵀ S V0 + δVᵀ S δV.
Λ0 = .25 (V0ᵀ gA0 + gA0ᵀ V0).
```

The multiplier uses the unpreconditioned ambient radial gradient, not projected g or inverse-metric pg. Signed offdiagonal entries count twice in the full Frobenius contraction; replacing this with diagonal populations loses normal sensitivity. The quadratic term cancels the linear normal component of an exact polar motion. Omitting it would subtract genuine tangent curvature.

At K0=I, the formula gives the symmetric normal multiplier because the derivative of Tr(ΛK) is 2SVΛ. At a nonideal baseline it is the near-identity construction: an exactly tangential symmetric part would require K0Λ+ΛK0=sym(V0ᵀgA0). No extra linear solve is proposed. The observed baseline defect must remain in the ledger; a small Gram defect by itself does not bound Γ, multiplier error, or energy error.

Physical and coordinate arithmetic have separate identities. If δlin=Pδy and e=V1−V0−δlin, the reconstruction correction is

```text
ΔKactual−ΔKlin = (V0+δlin)ᵀ S e + eᵀ S (V0+δlin) + eᵀ S e.
```

Use the saved actual physical increment for Γ, retain this exact split and all same-κ columns, and do not substitute YᵀY or discard e. This tests the original points without retracting them again.

The known one-body seam is also a conditional ambient correction. For Iab=VaᵀhVb/nb and nb=VbᵀSVb,

```text
∂a Iab = h Vb / nb
∂b Iab = hᵀ Va / nb − 2(VaᵀhVb) S Vb / nb².
```

The transpose matters: saved raw h need not be symmetric. The public gradient uses symmetric L(raw h), retained signed RDM coefficients, and raw CI products. Replacing its retained one-body action by the raw-h/ket-normalized action does not produce the full normalized mixed-CI gradient. A normalized quotient partial gradient requires each target's RDM divided by its own CI norm; the retained/unfiltered two-body difference also remains unbounded in this control. Consequently gA and Λ are explicitly conditional. A coefficient count is not a directional bound.

For completeness, the finite CI response cannot be replaced by frozen baseline density. For one phase-aligned reference a=σc0, d=c1−a, n1=c1ᵀc1, K0=H0−λ0I, r0=K0a and r1=(H1−λ1I)c1,

```text
λ1−λ0 = c1ᵀ(H1−H0)c1/n1
        + [dᵀK0d + 2dᵀr0 + aᵀr0 − c1ᵀr1]/n1.
qj−λj = cjᵀ(Hj−λjI)cj / (cjᵀcj).
```

Whole-column phase must apply to a and r0. Both finite endpoint residual terms, every norm denominator, and the trial signed RDM are required. The analogous quotient response is centered on q0; it retains its finite response terms. These algebraic identities describe complete stored symmetric H. They do not prove that a separately evaluated radial increment equals a subtraction of rounded assembled matrices.

For this proposed test the independent radial anchor is the already saved **explicit mixed polynomial + rounded physical reconstruction + complete CI Rayleigh response**. The exact stored-H quotient is forensic evidence for assembly/reporting roundoff, not a substitute derivative oracle. Their existing central differences differ by about 1.47e−6 Ha per unit step. That closure gap remains unresolved and cannot be hidden by a wider tolerance. Reported−q endpoint offsets and floating EOL averaging offsets remain separately reported. No production λ-to-q change is authorized.

The only allowed result of this control is whether (ΔLQ,+−ΔLQ,−)/(2h) agrees with both the original saved analytic derivative and the conditional centered analytic action at the unchanged original absolute/relative threshold. The original PCG64 seed, h=1e−6, point, five physical UIDs, equal weights, full 1046 CSFs, 29 radials, original nuclear P and new 3451 metric stay fixed. A pass would identify constraint-normal sensitivity for this pair under the stated conditional gradient; a failure or resource refusal stays inconclusive. Neither outcome changes an old gate's status.

Future optimizer admission may cite this narrow diagnostic and the separately qualified exact stored-H acceptance guard only in a new prospective contract reviewed by root. The latter certifies descent in its finite supplied-H arithmetic domain. Neither provides a radial global error bound, missing CI-response gradient, stationarity certificate, experimental certification, or a mixed-CI Armijo callback. The frozen optimization preparation remains unready until an explicit new admission addresses the actual objective, gradients and unresolved domains.

Reject changed source/operator/nucleus/basis/signed order, incomplete same-κ groups, stale endpoint owner or tracker identity, ambiguous rank-one phase, changed weights, invalid CI norms/residuals/Gram, missing response/reconstruction pieces, or any producer/resource breach. No resampling, new points, tolerance changes, discarded small coefficients, or numerical source modifications are part of this review.

Primary source evidence: [reported evaluate and fixed-only constraint policy](../mixed_ci_finite_increment_implementation/source/atomx/methods/csf/variational.py), [finite CI response](../mixed_ci_finite_increment_implementation/source/atomx/numerics/ci_energy_increment.py), [radial reconstruction](../mixed_ci_finite_increment_implementation/source/atomx/physics/csf/radial_increment.py), [retained versus bilinear density](../mixed_ci_finite_increment_implementation/source/atomx/physics/csf/radial_density.py). The [closed one-body diagnosis](../correlated29_rotation_failure_diagnosis/actual/result.json) distinguishes measured normal/norm effects from unmeasured two-body and curvature errors.
