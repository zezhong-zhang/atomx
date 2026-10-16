# Saved reporting-difference diagnosis

The two ordinary reported-EOL derivative gates remain **FAIL**. The four independent, fixed-Lambda centered comparisons remain **PASS**. The saved scalar ledgers explain the difference quantitatively: the largest contribution is the closure difference between the complete radial-polynomial increment and the independently assembled retained-H quotient increment. This observation does not establish which producer operation causes that closure difference.

This analysis consumed only 14 closed JSON files (764,764 bytes), plus the frozen driver/contract/preparation/handoff hashes. It used Python standard-library scalar/Fraction arithmetic, ran once in 0.00605 s, and performed no NPZ reads or hashes, array reconstruction, numerical imports, H/CI/gradient/Dirac/eigensolver/optimizer calls, or callback/source changes. The original private source is `f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe`, and the point remains literal accepted13, 31 radial orbitals, complete 1595-CSF five-sector DC, original 3689 metric and five physical UID weights .2.

The direction is the StageA transported, normalized direction around accepted13. It is a newly declared local retraction path. It is not the exact derivative of the earlier step13 alpha=1 polar path or a Wolfe-curvature verification.

For each trial let M be the saved complete mixed Rayleigh increment (explicit radial polynomial + rounded-endpoint reconstruction + finite CI response), N be the complete signed fixed-Lambda contraction with actual DeltaGram, Q be the saved retained-H quotient difference, L the weighted reported-root difference, and O the actual floating EOL difference. Then the saved ledgers give

`O - (M - N) = N - (M - Q) + (L - Q) + (O - L)`.

Taking each declared central difference yields the following decomposition in Ha per unit normalized direction. The complete saved-scalar identity closes within 3.14e-18; that is a bookkeeping identity, not a new physics or error-bound gate.

| Central contribution | h=1e-5 | h=5e-6 |
| --- | ---: | ---: |
| Signed Lambda DeltaGram | -1.50472633e-8 | -1.07157580e-7 |
| Negative mixed-Q versus retained-H Q closure | -1.26305938e-6 | -2.05428397e-6 |
| Reported roots minus retained quotient | -1.31598164e-7 | +1.70002412e-7 |
| Floating EOL averaging | -2.65e-18 | +2.18278728e-7 |
| Ordinary minus centered, sum | -1.40970481e-6 | -1.77316041e-6 |

The ordinary analytic action is +0.038323150715044135. Its two measured errors are -1.40955690e-6 and -1.77335478e-6, exceeding the unchanged 3.83231507e-7 gate. The centered derivatives are 0.03832315086294883 and 0.03832315052066765; their errors against the public retained action are +1.47904699e-10 and -1.94376481e-10. The small difference between the summed discrepancy and the ordinary error is exactly the independently measured centered error.

The endpoint EO ULP divided by 2h is 1.81898940e-7 or 3.63797881e-7. These resolutions do not by themselves account for the ordinary errors. The explicit closure ledger is larger: at h=1e-5, mixed-Q minus retained-Q is +3.6930200e-12 for minus and +2.8954208e-11 Ha for plus; at h=5e-6 it is -1.7815327e-11 and +2.7275128e-12 Ha. Their central differences produce the dominant terms above. No new H, density, radial cell, or higher-precision matrix replay was performed, so assembly roundoff, contraction arithmetic, and other producer-route causes remain unresolved.

The signed actual-Gram contribution is also measured, not discarded. N is +5.34743990e-12 / +5.04649464e-12 Ha for h=1e-5 minus/plus and +5.45179345e-12 / +4.38021765e-12 Ha for h=5e-6. Its linear and quadratic pieces cancel strongly: approximately -1.770e-9 +1.776e-9 Ha for the first pair, and -4.39e-10 +4.439e-10 Ha for the second. The saved exact identities retain all seven kappa groups, signed offdiagonals, quadratic curvature, explicit increments, and rounded reconstruction. The JSON report retains their complete contracted decomposition; no new Gram arrays are inferred from scalar traces.

The finite Rayleigh CI-response central slopes are only -1.02484e-11 and -3.26801e-12. The larger reported-response slopes come from endpoint reported-root versus quotient residual offsets, not from omitting the finite CI response. For the baseline, the exported q-minus-lambda values are:

| Physical target | q-minus-reported lambda, Ha |
| --- | ---: |
| 6s_1/2 | -6.63894234e-13 |
| 5d_3/2 | +3.56751729e-12 |
| 5d_5/2 | -8.42653526e-13 |
| 6p_1/2 | +1.78801448e-12 |
| 6p_3/2 | -4.94346927e-12 |

Their weighted baseline offset is -2.18897050e-13 Ha. The actual baseline EO is -14067.43131550464 Ha, and its offset from the exact sum of the exported binary64 weights times reported roots is -6.74292216e-13 Ha. The result JSON contains every endpoint's five UID-bound q-minus-lambda offsets, CI norm defects, residuals, response terms, weighted q/EO offsets, and central contributions. These q offsets are the original exported 63-mantissa-bit longdouble residual diagnostics; they are not newly certified exact dyadic stored-H expectations.

The original 745 controls (743 PASS, two ordinary FAIL), first failure, unmodified reported production objective, existing stationarity FAIL, and all frozen evidence remain unchanged. Centering is a finite diagnostic on this direction and these two h values. It neither authorizes an optimizer callback nor establishes a full-space gradient, uniform finite-step bound, experimental accuracy, or automatic continuation.

Evidence: [scalar result](result.json), [bounded analyzer](analyze_saved.py), [prospective scalar contract](contract.json), [actual derivative domains](../actual/derivative_domains.json), [four-point source result](../actual/result.json), [preserved failure](../actual/failure.json), and [closed root receipt](../root_execution.json). Every consumed JSON and the four frozen preparation files is SHA-bound in the scalar result.
