# Independent mixed-CI finite-increment source review

The exact frozen candidate passed 1,193 independent read-only source, AST and metadata checks. No source-level material finding remains. This does not qualify numerical decrement accuracy or authorize an optimizer run. Its [contract](contract.json) permits source, algebra, AST and metadata inspection only; no Atomx import, Hamiltonian build, CI solve, gradient, Dirac solve or optimizer call has been performed.

The [final review](review.json), [verification](verification.json) and [exact input binding](final_inputs.json) bind source map `0e5496eb7eab63df2e44d08c4d986574821b33b806a7d3c760e243bda5cc2193`, the prepared tests and independent oracle. [Draft findings](draft_findings.json) preserve issues found and corrected before any numerical execution. The implementation owner has prepared finite Decimal/native oracle qualification, which requires a separate root execution seal.

For one matched target, let `x0 = phase*c0`, `d = c1-x0`, `n1 = c1^T c1`, `A0 = H0-lambda0 I`, `r0bar = A0*x0` and `r1 = (H1-lambda1 I)*c1`. Expansion of `c1 = x0+d` gives the finite reported-root identity

```
lambda1-lambda0 = [c1^T deltaH c1 + d^T A0 d + 2 d^T r0bar
                   + x0^T r0bar - c1^T r1] / n1.
```

The entire reference column and its residual receive the same phase. All four response terms, finite CI norm and signed trial density are necessary. Centering on the retained-vector reference Rayleigh quotient gives a separate quotient identity; it does not restore eigenvalue bits already lost in reported roots.

For ordered orbital one-body elements, the native CI convention is `h_ab = N_ab/s_b`. Its explicit finite change is `(deltaN - h_ab*delta_s)/(s_b+delta_s)`, with the complete quadratic numerator and ket-norm changes. The separately rounded actual trial supplies a further complete numerator/norm correction. Coulomb pair density is quadratic; a linear cell potential acting on two such densities gives all explicit quartic terms. The additional reconstruction density contains all five reference/increment/reconstruction cross terms, and the energy includes all five reconstruction-potential/density products. The source currently retains these terms and the original source/target pair orientation.

These are real-arithmetic identities of finite assets. The production radial evaluator retains the existing native binary64 cell kernel, including its existing fastmath behavior; no new fastmath or alternate semiseparable kernel was added. Roundoff, angular projection and reconstruction accuracy still require the separately contracted finite numerical oracle. A source review is not a certified decrement error bound.

The frozen old354 comparison finds every pre-existing variational, ordinary CI, signed-density and optimizer function unchanged. The matrix path changes only the extraction of its existing pair-band loop. All current and archived source hashes matched before and after the read-only check. Every pre-existing changed-file function except the contracted extraction retains identical source bytes as well as AST. The 352 other pre-existing source files are byte identical.

The original CI column residual gate is 1e−8 and both CI/radial Gram gates are 1e−10. The original Yb optimizer/metric and 1e−7 stationarity, 1e−8 indicator, 1e−9 raw-gradient and 1e−10 projected-gradient gates remain unchanged. The candidate is diagnostic only and has no optimizer adapter. Concurrent eigenpair mutation is unsupported; owner validation protects physical inputs and is separate from the complete eigenpair checks.

Read-only execution used CPU affinity 0, a 1 GiB address-space ceiling and a 30 s timeout; it completed in 0.0583 s with 27 MiB peak RSS. No Atomx import, physics operation or test was executed. Numerical qualification and any future production adoption remain separate decisions.
