# Complete native analytic CI-response results

All states use the original complete A1595/B2574 spaces,31 radial waves,3689 original coordinates,69-electron Yb+,Z70/A175 Fermi and original P/S/metric. H0 is freshly built at the actual mathematical polar path zero; the rounded difference from archived R0 is explicitly measured. No displaced calculation supplies H1/H2.

## Analytic state response

Units: first derivative Ha per direction coordinate; second derivative Ha per coordinate squared; B_CI Ha per coordinate cubed. B_CI is the CI-response portion of E3/6, excluding fixed-vector H3.

| Arm | State | Dimension | First | Second | B_CI | Gap / Ha | Response residual / Ha |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 6s_1/2 | 223 | -0.166246919513 | 0.750559733656 | 0.0406455256976 | 0.300669959 | 1.49151e-13 |
| A | 5d_3/2 | 370 | -0.167133419764 | 0.747211269166 | 0.0433423596979 | 0.159582164 | 1.70493e-13 |
| A | 5d_5/2 | 416 | -0.168460411266 | 0.748686428438 | 0.0429596775158 | 0.141185473 | 1.63773e-13 |
| A | 6p_1/2 | 219 | -0.16951070496 | 0.761560209666 | 0.042661689658 | 0.0689219607 | 1.88058e-13 |
| A | 6p_3/2 | 367 | -0.169193791326 | 0.756665065414 | 0.044736194543 | 0.0315409713 | 1.53348e-13 |
| B | 6s_1/2 | 365 | -0.163876272208 | 0.752693103563 | 0.0393290737872 | 0.31745234 | 9.94816e-14 |
| B | 5d_3/2 | 596 | -0.164635755719 | 0.748655118317 | 0.0419982309653 | 0.174236925 | 1.22604e-13 |
| B | 5d_5/2 | 652 | -0.166011523775 | 0.750099924905 | 0.0417186523489 | 0.15490441 | 1.85553e-13 |
| B | 6p_1/2 | 362 | -0.166950915688 | 0.761066678409 | 0.0404957601093 | 0.0848485292 | 1.43661e-13 |
| B | 6p_3/2 | 599 | -0.166510425515 | 0.755325214278 | 0.0414503738476 | 0.047558965 | 1.20124e-13 |

## Independent predicted versus measured finite CI response

Predicted values are h² times the independently calculated upper-minus-lower B_CI. Measured values are exact-dyadic endpoint Rayleigh-minus-fixed-C0 slopes from011; they are not used to build the derivative operators or fit the coefficient. These are derivative actions in Ha per original direction coordinate, not finite relaxed spectral shifts.

| Arm/line | h | Predicted | Measured | Absolute discrepancy | Predeclared limit |
|---|---:|---:|---:|---:|---:|
| A/5d_3/2 | 0.002 | 1.078733600136e-08 | 1.078642165215e-08 | 9.143492e-13 | 5.393211e-11 |
| A/5d_3/2 | 0.001 | 2.696834000340e-09 | 2.696304057235e-09 | 5.299431e-13 | 1.348152e-11 |
| A/5d_5/2 | 0.002 | 9.256607272943e-09 | 9.256868377038e-09 | 2.611041e-13 | 4.628434e-11 |
| A/5d_5/2 | 0.001 | 2.314151818236e-09 | 2.314787366994e-09 | 6.355488e-13 | 1.157394e-11 |
| A/6p_1/2 | 0.002 | 8.064655841605e-09 | 8.064628532780e-09 | 2.730882e-14 | 4.032314e-11 |
| A/6p_1/2 | 0.001 | 2.016163960401e-09 | 2.016501464072e-09 | 3.375037e-13 | 1.008251e-11 |
| A/6p_3/2 | 0.002 | 1.636267538163e-08 | 1.636206082660e-08 | 6.145550e-13 | 8.181030e-11 |
| A/6p_3/2 | 0.001 | 4.090668845409e-09 | 4.090179531674e-09 | 4.893137e-13 | 2.045090e-11 |
| B/5d_3/2 | 0.002 | 1.067662871228e-08 | 1.067689587664e-08 | 2.671644e-13 | 5.338448e-11 |
| B/5d_3/2 | 0.001 | 2.669157178070e-09 | 2.669794099891e-09 | 6.369218e-13 | 1.334897e-11 |
| B/5d_5/2 | 0.002 | 9.558314246828e-09 | 9.558498295503e-09 | 1.840487e-13 | 4.779249e-11 |
| B/5d_5/2 | 0.001 | 2.389578561707e-09 | 2.390116993665e-09 | 5.384320e-13 | 1.195058e-11 |
| B/6p_1/2 | 0.002 | 4.666745288370e-09 | 4.666544546805e-09 | 2.007416e-13 | 2.333272e-11 |
| B/6p_1/2 | 0.001 | 1.166686322093e-09 | 1.166769813489e-09 | 8.349140e-14 | 5.833849e-12 |
| B/6p_3/2 | 0.002 | 8.485200241761e-09 | 8.484656191397e-09 | 5.440504e-13 | 4.242328e-11 |
| B/6p_3/2 | 0.001 | 2.121300060440e-09 | 2.120775490182e-09 | 5.245703e-13 | 1.060388e-11 |

All16 response comparisons and all155 live controls pass. Max discrepancy9.14350e-13Ha; max relative discrepancy0.02746%. This is a measured leading-term consistency test, not a certified O(h4) remainder bound or a passed total ordinary FD.

## Qualification, source and execution

Independent15 new tests plus existing polar/state tests25PASS. Full affected panel67PASS/2FAIL; before-source legacy panel42PASS/the same2FAIL. Thirty-nine saved H/eigen arrays are bitwise unchanged before/after. The initial32 fixture-setup errors and initial tuple/CSFSignature API failure remain preserved; no failed scientific comparison is retried with changed steps, caps, model or gates.

Actual current source fingerprint: `0d73a18232a5ecb1b5dcb5871d71fd6fd7c56df6ab286658d90422d5da59b99a`,359 Python files; only one new numerical module and an additive radial-derivative class/factory differ from the before source. Existing radial-increment AST is unchanged. All source, scientific inputs, contracts and full output arrays are bound.

Corrected once-only execution:30 H0/H1/H2 block builds,10 complete centered eigensolves,10 complete-spectrum projected responses,19,828 shared radial keys.96.6626s/1089.379MiB, CPU27/thread1/8GiB/600s.62,363,256 output-array bytes. Feasibility only; no paired performance comparison.

Max first-action mismatch1.17168e-12Ha; max fresh-polar0-H versus literal-R0-H difference1.09140e-11Ha; max radial zero-point difference3.55272e-15; max projected response residual1.88058e-13Ha.

The original005 failures, current centered replay two failures, two legacy test failures, production stationarity and experimental gaps remain. No optimizer, full4f12 response, total third derivative, fixed-vector H3, higher-order remainder certificate or experimental improvement is claimed.

[Owning report](README.md), [literature](LITERATURE.md), [actual result](actual002/result.json), [preserved API failure](actual001/result.json).
