# Closed native four-point physical discriminator

**260/260 PASS; one scientific execution, exit0.** All four points retain the complete native f043 model: A91NR/1595CSF and B100NR/2574CSF, common31orbitals,69electrons,Z70/A175Fermi, five fixed physical UID weights0.2, original nuclear no-pair/metric. The sole common displacement is alpha1/16 along all3689 entries of `−checkpoint13.full_normalized_pg`, A’s normalized preconditioned tangent. Each point has a fresh native retained H/CI owner.

Excitations and interaction I=[XB(Rδ)−XA(Rδ)]−[XB(R0)−XA(R0)], cm⁻¹:

| Line | A(R0) | A(Rδ) | B(R0) | B(Rδ) | I |
|---|---:|---:|---:|---:|---:|
| 5d_3/2 | 23964.609237 | 23951.166841 | 24432.674156 | 24420.677664 | 1.445904 |
| 5d_5/2 | 24786.094814 | 24755.053955 | 25459.798894 | 25429.527010 | 0.768975 |
| 6p_1/2 | 25966.939240 | 25926.963888 | 26116.080554 | 26077.537260 | 1.432059 |
| 6p_3/2 | 29440.581404 | 29402.957319 | 29601.404445 | 29566.491663 | 2.711302 |

Absolute radial-displacement energy changes, Ha. The 6s row is the common lower-state change for every line; the other rows give its corresponding upper-state change.

| State | ΔE A | ΔE B |
|---|---:|---:|
| 6s_1/2 | -0.008917816987 | -0.008765800278 |
| 5d_3/2 | -0.008979065051 | -0.008820460316 |
| 5d_5/2 | -0.009059249549 | -0.008903729129 |
| 6p_1/2 | -0.009099958095 | -0.008941416445 |
| 6p_3/2 | -0.009089244930 | -0.008924874617 |

D/P centroids and splittings, cm⁻¹:

| Quantity | A(R0) | A(Rδ) | B(R0) | B(Rδ) | I |
|---|---:|---:|---:|---:|---:|
| D centroid | 24457.500583 | 24433.499109 | 25048.948999 | 25025.987271 | 1.039747 |
| D splitting | 821.485577 | 803.887114 | 1027.124738 | 1008.849346 | -0.676929 |
| P centroid | 28282.700683 | 28244.292842 | 28439.629815 | 28403.506862 | 2.284888 |
| P splitting | 3473.642164 | 3475.993432 | 3485.323891 | 3488.954403 | 1.279244 |

The named5p-double correlation shifts remain +468.06/+673.70cm⁻¹ at R0 and +469.51/+674.47cm⁻¹ at Rδ for the D lines. Its interaction with this one small common displacement is 0.77–2.71cm⁻¹ across the four lines: measurable and locally small relative to the hundreds/thousands cm⁻¹ discrepancies. This does not establish weak coupling for other directions, nonlinear optimization, omitted4f/mixed-hole classes or the terminal16point. Accepted13 remains the existing paired diagnostic anchor; no correction is spliced into terminal16.

Ordinary EOL is separate: A decreases by −0.009029066921Ha and B by −0.008871256157Ha; both satisfy the recorded Armijo diagnostic. Neither energy decrease nor this finite probe qualifies the original ordinary-FD or stationarity failures. No new gradient, optimizer, SCF, virtual channel or external backend was used.

Fresh A_R0 whole H reproduces its saved H with maximum numerical difference0; full saved-column replay residual is at most2.18e−11Ha. All fresh full-block eigenpair residuals are ≤3.28e−11Ha, CI Gram≤4.45e−15 and radial Gram≤3.67e−15. Early exact nuclear quadrature SHA on the original115×9 grid matches the saved A owner. Every point retains complete signed UID tracking and actual norms.

Actual work:4H/4CI/20blockeigh,10angular tables,43nullspaces,7rawh/S,4radial evaluators,277616radial-key requests,2824pair potentials,11296convolutions/factor-getters,3968component densities and50tracking projections. Every instrumented producer attempt completed. Elapsed90.431s; peak RSS1,912,500,224bytes (1.781GiB). This is uncontrolled diagnostic cost, not a performance comparison.

Execution001 remains closed: its shape_choices loader raised KeyError:shape after3input archives, with all H/CI/angular/radial/polar/tracking calls0. Explicit002 corrected only declared shape-schema handling;56consumed fields/104member headers passed stdlib preflight. Root also records an overstrict preseal input-equality rejection and a following no-seal invocation rejected before actual/scientific imports; neither was an additional scientific execution. The corrected root seal is retained.

[Actual result](actual/result.json) · [independent review](result_review.json) · [contract](contract.json) · [root seal](root_execution_seal.json) · [hash closure](closure_receipt.json) · [EARS](trace.md). Complete H/CI/waves and point owners are saved in all four point NPZ/JSON files. The old nested-byte, saved-diagnostic module-boundary, ordinary-FD and production stationarity failures remain failed.
