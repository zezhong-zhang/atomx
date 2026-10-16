# Screened virtual construction: preserved numerical failure

The closed68 auxiliary native field passed its independent signed scalar controls. The first complete κ=−3 complement eigensystem failed the original full-spectrum residual gate; no screened CI/H calculation or line response exists.

The independent saved-array audit made no Atomx imports, eigensolves, Slater evaluations, CI, gradients, or optimization. It verified 613 controls and replayed the failed quantity without changing the archived invocation.

| Saved quantity | Value |
|---|---:|
| Complete κ=−3 roots | 116 |
| Roots above original relative 1e−8 gate | 49 |
| Lowest physical Rayleigh | −0.1548970463009521 Ha |
| First actual Rayleigh gap | 0.0605615963862013 Ha |
| Lowest original constrained residual | 3.0212938683e−8 Ha |
| Lowest effective lower-triangle solver term | 3.0212923723e−8 Ha |
| Lowest longdouble projection term | 3.3482e−14 Ha |
| Lowest physical mapping term | 3.3739e−14 Ha |
| Maximum failed relative residual (root58) | 3.1130860258e−8 |
| Root58 constrained / solver residual | 1.5540208601e−7 / 1.5540205860e−7 Ha |
| Root58 projection / mapping term | 1.5629e−13 / 3.7090e−13 Ha |
| Projected H infinity norm | 4.4773704480e8 Ha |
| Float projected antisymmetry | 7.6537e−8 Ha |
| Longdouble projected antisymmetry | 1.1599e−10 Ha |
| Float lower triangle versus LD full projection, infinity norm | 2.5255e−7 Ha |
| Projected S conditional infinity condition upper bound | 1.0000000000000837 |

The decomposition uses SciPy's actual default lower triangle: at the physical Rayleigh, the residual is the lower-triangle solve term plus LD projection difference and physical mapping difference. Closure is 3.0614e−10 Ha over the full spectrum (diagnostic gate1e−9 Ha); this is not a physical acceptance change. All native h/J/K matrices are exactly symmetric in the archived representation. Although the float projection has a visible operator-norm defect, its action on the low returned vectors is tiny. For these vectors the returned complete gvd eigensystem itself supplies the failed residual. Stabilizing only projection cannot be claimed to cure it.

The large spectral range and epsilon×H∞≈9.94e−8 Ha are consistent with the observed floating eigensolver scale. This is an inference from residual decomposition, not a rigorous eigenvalue error bound or proof of a particular LAPACK defect. S is close to I and does not show metric ill-conditioning. The failure is numerical qualification of an auxiliary virtual construction; it does not establish a missing physical interaction or a screened spectral improvement.

All80 independent signed PP/QQ/PQ/QP scalar controls remain qualified: maximum error2.2737e−13 Ha; cross terms have measured magnitude0.007212–0.110301 Ha. Actual completed work was10J/K builders,3584scalar Slater calls,7native h/S builds,1complement SVD,1fullgvd. CI/H/gradient/optimizer calls were0.

The next smallest discriminator is complete116-root longdouble Jacobi refinement of this same saved operator, original P, and complement, with bounded full-spectrum iterations and all original gates. It must be separately admitted and cannot select only the lowest or the49 failed roots. A passing diagnostic would qualify this numerical method for the first κ only; a separate five-κ/CI contract would still be required.

Evidence: [original failure](failure.json), [actual execution closure](root_execution.json), [saved audit contract](saved_failure_audit/contract.json), [613-control audit](saved_failure_audit/verification.json), [diagnostic arrays](saved_failure_audit/diagnostic_arrays.npz). The frozen preparation README and original driver/contract remain unchanged.
