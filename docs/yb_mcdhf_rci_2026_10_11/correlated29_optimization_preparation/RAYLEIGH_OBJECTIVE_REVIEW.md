# Source-only Rayleigh objective and callback boundary

The first correlated29 probe **failed its declared reported-root derivative gate**. Root closed the run with4H/4CI/1gradient and no optimizer. The saved direction0 central ledger gives analytic−8.728707560749522e−6, reported−7.463868907920965e−6, and normalized retained-Rayleigh−8.728250189363864e−6 Ha per coordinate unit; the reported discrepancy is about1.26484e−6, exceeding1e−8. The Rayleigh discrepancy is about4.57371e−10, below that unchanged threshold by arithmetic, but the runner stopped before that gate was evaluated. This observation does not pass the old reported gate or either unexecuted rotation derivative. The source-only16-step optimization contract remains contingent and unauthorized.

## What is the physical variational functional?

For the same Hermitian finite DC Hamiltonian H(x), the same matched complete-CI target vector c, and its actual norm n=cᵀc, the normalized Rayleigh energy is

`q(x,c) = cᵀ H(x) c / n`.

For the five original physical targets, the objective is `F=Σ_i w_i q_i` with the unchanged weights0.2. This is a proper finite-space variational functional. An excited target remains a physically selected eigenstate, not a globally minimized ground state: the state/orthogonality and matching constraints remain mandatory. At an exact normalized CI stationary state, q equals that eigenvalue and its orbital derivative is the Hellmann–Feynman contraction of the same signed normalized density with H′. A rounded eigenvalue reported by an eigensolver is a numerical approximation to this energy; its rounding need not have a useful derivative at tiny orbital steps.

For finite residual r=(H−qI)c, however, the complete derivative along a numerical CI trajectory is

`q′ = cᵀ H′ c / n + 2 c′ᵀ r / n`.

The norm derivative is already included in the second term. For real symmetric H, `cᵀr=0` algebraically when q is formed consistently. Finite CI residuals and finite c′ still cannot be dismissed merely because column normalization and residual gates pass. The current functional gradient contracts the CI density and differentiates the radial Hamiltonian; it does not explicitly add a differentiated numerical eigensolver response. Exact CI stationarity justifies that simplification in the mathematical method. For actual floating CI vectors, probe agreement or an appropriate bound must qualify the derivative accuracy at the unchanged gates. Two directional agreements do not establish full orbital stationarity.

For a weighted tracked objective, each c must be normalized in its own actual CI norm. Complete signed off-diagonal density terms, both radial P/Q components, all29 active orbitals, the original radial manifold, all physical target identities, and root/degenerate-span rules must remain consistent. A source-only declaration of Rayleigh energy does not certify this implementation or bypass those numerical conditions.

## Why the present diagnostic cannot be adopted as a callback

`CSFOrbitalFunctional.evaluate` in frozen candidate356 returns the weighted **reported roots**, `float(state_weights @ result.state_energies)`. The existing mixed API separately returns `reported_change` and `rayleigh_change` and declares that it is diagnostic-only, has no certified error bound, and is not an optimizer adapter. The public optimizer takes scalar energy from `evaluate` and a callback float/None only inside its128ULP total-energy window. Replacing a reported-root comparison with a Rayleigh difference only in that branch would give two different objective semantics in one run.

A legitimate Rayleigh successor must declare a new objective identity and use normalized Rayleigh energy consistently at baseline, every ordinary energy trial, accepted gradient point, diagnostics, callbacks and checkpoints. It can retain the same physical DC operator, finite CSFs, all29 orbital geometry, P, inverse metric, original tracker and thresholds. A changed computational observable is explicit; a different Hamiltonian or experimental fit is not proposed. Centering the common core origin before extended-precision contraction and normalization avoids unnecessarily forming a large absolute q, but by itself it is a numerical representation, not an error certificate.

The source's reported finite response contains both endpoint eigenpair residual terms so that it reconstructs the *reported roots*, not a magically recovered high-precision physical eigenvalue. The Rayleigh response centers H0−q0I and includes the actual norm and finite change of the CI vector. Dropping response, phase, norm or reconstruction terms changes the observable. Medium-step derivative agreement and closure against the same saved formula are qualifications of their stated scope, not a uniform tiny-step Armijo bound.

## Smallest proper progression

1. **Finish the existing saved decomposition**, without new solves: separately tabulate each tracked λ−q endpoint offset, actual CI norms, finite residual terms, reported/Rayleigh response pieces, radial/reconstruction terms, floating averaging offset and their effect on the central derivative. Bind the complete retainedH owners and same recovered3451 point. This can identify whether the observed reported error comes from the reported-root/quotient difference; it cannot reconstruct lost reported precision or assert missing physical correlation from that error.
2. If that decomposition supports the normalized Rayleigh semantics, root can authorize a new explicitly **Rayleigh-targeted** qualification contract. Reuse the exact saved baseline and direction0 pair; evaluate the still-unexecuted original seeded old/new rotation ±1e−6 pair, without changing seed, h, tangent definition, P, target, metric or original1e−8 absolute/1e−5 relative derivative thresholds. Separate Rayleigh derivative qualification from the preserved old reported failure. A complete samepoint public gradient replay is a separately counted check if required. This is not optimizer authorization.
3. Only if near-roundoff optimization needs a stable callback, independently qualify its full finite delta at the actual new observable and unchanged geometry. The minimal scalar adapter must consume samepoint retained owners, complete unfiltered normalized trial densities, full finite CI response, both actual endpoint norms, explicit projector-mapped orbital increments and reconstruction corrections. It must reject mismatched operators, signed bases, target ranks/weights/phases, owner inputs, Gram/no-pair defects, unsupported complex/degenerate cases or unbounded error. It must return a scalar with a trustworthy error interval or`None`; the existing dictionary is not that interface.

For a qualified estimate Δ̂ with conservative absolute error ε, an Armijo decision is justified only when

`Δ̂ + ε ≤ 1e−4 × step × g·d`.

The error ledger must cover radial contraction rounding, reconstruction, numerical CI-vector response, norm/phase arithmetic, reported-versus-Rayleigh conversion when relevant, and any physical-versus-storedH discrepancy. Direct use of a measured closure error as ε is not conservative: it may compare two correlated implementations. Nor is a fixed1e−8 CI residual gate automatically an energy-difference error bound. If using a residual/gap estimate, actual root isolation and whole degenerate-span coverage must be bound at both endpoints; otherwise reject that estimate. Any constraint-normal centering must have its own same-manifold equivalence and floating-Gram guards, rather than erase a real finite constraint change.

No h scan, additional direction, gate relaxation, source patch, callback implementation, new physical builder, eigensolve or optimizer is authorized by this review. The smallest next action is the already assigned saved decomposition followed, if supported, by the untouched second seeded derivative under an explicitly declared Rayleigh target. Neither a numerical representation repair nor EOL decrease is evidence that the experimental2000cm⁻¹ gap has been remedied.
