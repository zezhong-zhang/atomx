# Prospective mixed-CI MCDHF continuation and energy-difference design

Status: assessment and saved-matrix algebra only. No shared source was edited, no additional CI problem was solved, no gradient was evaluated and no optimization step was taken for this assessment. The completed [531-CSF diagnostic](README.md) and its result, source and card evidence remain immutable. Its accepted step still fails the original stationarity gates and worsens all four nominal DC NIST gaps.

The [prospective assessment contract](successor_assessment_contract.json) preceded the [saved-matrix algebra replay](mixed_difference_algebra.py). All 239 checks on 45 physically matched target pairs pass in [mixed_difference_algebra.json](mixed_difference_algebra.json). This establishes the finite-matrix identity below. It does **not** implement or qualify a cancellation-free radial polynomial, a full mixed-CI optimizer or a near-stationary decrement error bound.

## Public continuation boundary

The implemented physics is reusable: CSFOrbitalFunctional.evaluate, CIStateAverage, CIBasisTracking, CIRadialDensity.contract_bilinear, the complete public CI matrix/solve and the shared B-spline Dirac/Coulomb kernels support the declared 69e/24-radial/531-CSF DC objective. All five targets are unambiguous exact-J/parity local root 0, with equal weights and complete signed spaces.

The current public optimize_orbitals arguments accept seed orbitals, full preconditioner matrices and numerical settings, but no saved coordinate layout, positive-energy projector or inverse-metric state. It reconstructs the nuclear spectrum/projector, polar-normalizes the seed, reconstructs spectral gaps and rescales the inverse blocks by the **new** accepted density occupations. MCDHFSolver.solve also constructs an initializer and its Fock fields before entering that optimizer. These are valid fresh calculations, but they cannot be declared a byte-identical continuation of original005 metric SHA aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe.

The saved inverse blocks cannot uniquely recover their original full Fock matrices: absolute spectral gaps, the floor and occupation scaling lose information. Passing an invented inverse-derived Fock matrix would therefore be a new metric rather than a public resume of the old one.

Proposed interface work before a full continuation:

1. A validated immutable optimizer-coordinate state carrying ordered orbital keys, shape/slices, physical S-metric and nucleus/workspace identity, the actual no-pair projectors, original inverse diagonal and inverse blocks, coordinate point, numerical-policy identity and their complete SHA.
2. Public optimizer resume that consumes those arrays directly, verifies finite positive and compatible geometry, restores the exact original residual action and forbids silent spectral/occupation rebuilding. Output must identify the actual metric used at every accepted point.
3. The same 531-CSF signed enumeration, same five physically tracked targets and equal weights remain the new objective. Old bare-five secant history is cleared because its objective changed; this does not alter the restored original residual yardstick. An objective transition must be explicit rather than importing a bare-five convergence flag.
4. A bounded successor contract binds the complete source epoch and all identities, keeps 1e−7 / 1e−8 Ha gates and retains failed accepted checkpoints. Source implementation and independent interface tests precede an actual continuation.

This is a proposed shared numerical interface. The implemented physics can remain native Atomx MCDHF.

## Exact mixed-CI difference in a common signed CSF order

Use the same real canonical signed CSF coordinate order at both radial endpoints, with complete exact-J/parity blocks. The physical CI bases change with the orbitals; their common abstract coefficient labels define H₀ and H₁. Public physical-overlap tracking chooses the corresponding states first. A coefficient sign alignment is then a numerical gauge choice for a small δc; it must not replace physical state matching.

For one matched target let nᵢ=cᵢᵀcᵢ, δc=c₁−c₀ and δH=H₁−H₀. If e₀ is the reference Rayleigh quotient, define A₀=H₀−e₀I and r₀=A₀c₀. The exact finite-norm identity is

$$
e_1-e_0 =
\frac{
c_1^T\,\delta H\,c_1
+\delta c^T A_0\delta c
+2\delta c^T r_0
+c_0^T r_0
}{n_1}.
$$

In exact arithmetic c₀ᵀr₀=0 by the reference quotient definition. At an exact reference eigenstate r₀=0, leaving the trial-RDM radial increment and quadratic CI-response term. Both finite CI norms remain explicit; normalizing only the state-average weights is insufficient.

The current functional reports stored CI eigenvalues λᵢ. To reproduce those particular finite values without silently switching to a Rayleigh-quotient objective, instead set A₀=H₀−λ₀I, r₀=A₀c₀ and r₁=(H₁−λ₁I)c₁. Then

$$
\lambda_1-\lambda_0 =
\frac{
c_1^T\,\delta H\,c_1
+\delta c^T A_0\delta c
+2\delta c^T r_0
+c_0^T r_0
-c_1^T r_1
}{n_1}.
$$

This is an algebraic identity for the actual saved vectors, norms, matrices and λ values, including finite eigenvector residuals. The final term cannot be silently dropped merely because CI residuals pass 1e−8 Ha.

Apply the formula to the five independently matched targets with their unchanged weights. Complete equally weighted degenerate clusters would require a basis-invariant paired-subspace treatment; individual sign matching is not a general cluster algorithm. Changed spaces, unsupported complex/charge-sector targets, incompatible signed orders or ambiguous physical tracking must be rejected explicitly. They are outside the present five-root real DC proposal.

## Quantitative saved-array controls

For all eight central perturbations and the one already accepted step, exact-J/parity identity and majority physical overlap are retained. The long-double saved-matrix check obtains maximum finite-reported-eigenvalue identity error 1.47e−21 Ha and quotient-identity error 4.28e−16 Ha across the 45 target pairs.

The accepted step's unchanged equal-weight decomposition is:

| Term | EOL change, Ha |
|---|---:|
| Trial-RDM c₁ᵀδHc₁/n₁ | −0.000583156466322 |
| Quadratic CI response δcᵀA₀δc/n₁ | +0.000318469819673 |
| Reference residual cross term | +8.8786e−15 |
| Reference residual offset | +2.3772e−13 |
| Trial residual offset | −8.7628e−13 |
| Sum of matched reported-root differences | −0.000264686647279 |
| Frozen old-RDM c₀ᵀδHc₀/n₀, for comparison | **+0.0000504817321150** |

The omitted CI-response scale is large for this step. Freezing the old RDM even gives the wrong sign for its radial prediction. The maximum single-root pair of finite residual offsets is 1.21e−11 Ha; this is appreciable at the former bare-model tiny-decrement scale.

The archived functional's float EOL value averages the large reported λ values first. Its accepted-step difference, −0.000264686646915 Ha, differs from the weighted sum of root differences by +3.64e−13 Ha because averaging and subtraction round differently. A future stable callback must state whether it estimates the underlying mathematical EOL or exactly reproduces those stored floating values, retain the norm/residual diagnostics and qualify its numerical error. Adding that averaging-roundoff offset blindly would restore the cancellation being addressed.

The replay computes δH by subtracting already rounded saved matrices. That subtraction and the small measured finite perturbations are deliberately **not** a proof that the new radial polynomial is accurate at 1e−7 stationarity. No tiny-step acceptance was performed here.

## Shared radial polynomial and normalization requirements

The trial signed density can be formed with the implemented public CIRadialDensity.contract_bilinear(c₁,c₁)/n₁, summed over the fixed five target weights. It carries the full interference from the Hamiltonian's angular table. CIRadialDensity.contract filters per-block coefficients at 1e−15; the unfiltered bilinear method removes only exact zero values and is the appropriate primitive for an exact difference proposal. Retained-density equality is not an all-model error bound.

Factor the existing B-spline fixed-density polynomial into a shared radial-increment operation which accepts an explicitly bound signed RDM. Its public current fixed-density guards must remain: both fixed_density_energy_difference and constrained_energy_difference reject this mixed/tracked case. Removing their guard alone would freeze the wrong density and omit the CI response.

The radial proposal has two separate normalizations:

- **Orbital one-body ket norm:** the actual CI radial evaluator uses hₐᵦ=(aᵀh_D b)/(bᵀS b). For explicit orbital increments δa,δb, retain the full quadratic δN and δnorm, and use δh=(δN−hₐᵦδnorm)/(norm+δnorm). The angular contraction must preserve the actual Hamiltonian's final real symmetrization and ordered radial keys.
- **CI amplitude norm:** divide the trial bilinear density and the coefficient-response/residual terms by n₁=c₁ᵀc₁. This is distinct from the orbital ket norm and the normalized ensemble weights.

For Coulomb terms, the shared cell kernels are bilinear in each pair density and quartic in the orbital coefficients. Expand both pair-density changes completely, including the quadratic pieces and their mutual product. Stream shared radial source potentials as the existing fixed-density method does; do not copy a new Coulomb formula or form a giant derivative tensor.

The original005 numerical evidence also shows that a projected coordinate increment P₊δy should be passed explicitly rather than recovered by subtracting two rounded unpacked endpoints. Bind it to the same stored physical endpoint and Gram guards. If a separate constraint-centered Lagrangian difference is used, derive its actual mixed target/quotient normal multipliers and retain the **complete quadratic** δ(CᵀSC). The uncentered mixed-difference API must retain its own semantics. A floating Gram guard alone does not supply a rigorous decrement error bound.

Independent qualification before use should include:

1. A high-precision complete signed mixed-CI two-root oracle with nonzero off-diagonal interference, nonunit CI norms and finite residual perturbations; both quotient and reported-λ identities.
2. Shared unfiltered trial-RDM contraction versus the complete symmetrized public H, including off-diagonal one-body ket-normalization controls.
3. Complete cell polynomial against an independent high-precision radial oracle at zero, forward/reverse finite and tiny steps, with a large common core offset. Verify the CI-response term rather than suppressing it.
4. Saved real Yb endpoint trials in the original projector/metric, before a new full run. Physical root ambiguity, norm/Gram incompatibility or unsupported operator coverage returns an explicit unsupported result.
5. Original orbital residual and energy gates evaluated on the actual accepted full mixed-CI state. Stable energy decrease never substitutes for either stationarity gate or experimental validation.

## Bounded resource expectation

The completed diagnostic used CPU1, 823.97 MiB RSS and 16.69 s under the declared 3.95 GiB / 890 s caps. Its measured uncontrolled stages were: functional construction 3.25 s, physical-reference binding 0.24 s, first full mixed CI+gradient 1.59 s, extra baseline complete matrix validation 0.53 s and the accepted already-solved-CI gradient 0.267 s. The balanced replay's all-root DC/static audit separately observed 1,246 MiB. Neither run is a controlled performance benchmark or a full-convergence time bound.

For an initial successor budget, 32 accepted iterations with an illustrative eight trial evaluations per iteration at 1.6 s each, plus 0.27 s gradient and 0.54 s full validation per accepted point, suggest about 436 s plus setup/archiving. That is a planning estimate, not a promised rate: physical-overlap tracking, polynomial construction, caches and deeper line searches change the work.

The existing optimizer can perform up to 56 trial solves at one failed line-search iteration. At the same illustrative cost, 32 such iterations would exceed 2,800 s. A prospective CPU1 / 3.95 GiB / 890 s full-model continuation therefore needs safe accepted-state checkpoints and a wall-resource stop; it must report nonconvergence rather than shrinking the 531-CSF model or relaxing 1e−7 / 1e−8 Ha. Whether 32 iterations suffice is unknown.

Near-roundoff obstruction for the **mixed** model has not yet been measured: its only two actual residuals are 0.0368 and 0.0250. The exact difference design is preparation, not a diagnosed explanation of its eventual convergence behavior. Even a converged 531 model would qualify this limited one-core-hole/pair pool, not complete atomic correlation or highest experimental accuracy.

## Source scope and documentation disposition

The assessment reads the frozen005 source map 9dff33cc24acecff9e821c4f9b836e02efca0d615291d6f92ff5a7014a726c3f; the current shared optimizer differs only in its callback-scope comment at the time of inspection. The relevant public signatures and operations are unchanged. Exact file hashes and links are registered separately in [successor_assessment_verification.json](successor_assessment_verification.json); earlier scientific/card/README hashes are preserved.

No source, public interface, default or scientific contract was changed. Root owns any subsequent shared implementation, new source epoch, regression suite, full-run contract and current consolidated navigation. This proposal is appended rather than rewriting earlier failed or accepted numerical evidence.
