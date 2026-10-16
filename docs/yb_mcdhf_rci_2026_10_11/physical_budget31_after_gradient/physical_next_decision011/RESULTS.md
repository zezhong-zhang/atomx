# Signed derivative decomposition at the original005 points

All derivatives are Ha per unit of the original complete direction parameter. Values below are diagnostic arithmetic/finite-response quantities, not spectral-energy corrections or experimental error bounds. Original fixed steps, UIDs, spaces, weights and gates are retained.

## Exact saved-H decomposition

`ordinary error = reported-root term + finite CI-vector response + fixed-C0 operator remainder`, checked with exact Fractions. The operator remainder still combines finite-step radial/constraint effects and stored-H assembly; its cause is not uniquely resolved.

| Arm | h | Line | Original signed error | Reported − exact Rayleigh | Finite CI response | Fixed C0 − analytic | Exact Rayleigh − analytic |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 0.002 | 5d_3/2 | +1.3035359098e-08 | -2.2669103791e-10 | +1.0786421652e-08 | +2.4756284835e-09 | +1.3262050136e-08 |
| A | 0.002 | 5d_5/2 | +1.5843525714e-08 | +1.4701075163e-09 | +9.2568683770e-09 | +5.1165498204e-09 | +1.4373418197e-08 |
| A | 0.002 | 6p_1/2 | +9.1400594316e-09 | -2.0008854926e-09 | +8.0646285328e-09 | +3.0763163915e-09 | +1.1140944924e-08 |
| A | 0.002 | 6p_3/2 | +1.7574792177e-08 | +7.6810587937e-10 | +1.6362060827e-08 | +4.4462547121e-10 | +1.6806686298e-08 |
| A | 0.001 | 5d_3/2 | +4.8499067818e-09 | -5.0112169544e-12 | +2.6963040572e-09 | +2.1586139415e-09 | +4.8549179987e-09 |
| A | 0.001 | 5d_5/2 | +7.2033260468e-09 | +3.4132114087e-09 | +2.3147873670e-09 | +1.4753272711e-09 | +3.7901146381e-09 |
| A | 0.001 | 6p_1/2 | +8.2305647298e-09 | +1.8353237956e-09 | +2.0165014641e-09 | +4.3787394702e-09 | +6.3952409343e-09 |
| A | 0.001 | 6p_3/2 | +3.4776242997e-09 | -2.0596603516e-09 | +4.0901795317e-09 | +1.4471051197e-09 | +5.5372846513e-09 |
| B | 0.002 | 5d_3/2 | +1.5871902653e-08 | +2.7094882405e-09 | +1.0676895877e-08 | +2.4855185362e-09 | +1.3162414413e-08 |
| B | 0.002 | 5d_5/2 | +1.6759135912e-08 | +2.1028878362e-09 | +9.5584982955e-09 | +5.0977497807e-09 | +1.4656248076e-08 |
| B | 0.002 | 6p_1/2 | +9.7844058412e-09 | +2.0266316480e-09 | +4.6665445468e-09 | +3.0912296463e-09 | +7.7577741931e-09 |
| B | 0.002 | 6p_3/2 | +1.0632187797e-08 | +1.7136161003e-09 | +8.4846561914e-09 | +4.3391550516e-10 | +8.9185716966e-09 |
| B | 0.001 | 5d_3/2 | +1.0869681794e-08 | +6.0505816016e-09 | +2.6697940999e-09 | +2.1493060920e-09 | +4.8191001919e-09 |
| B | 0.001 | 5d_5/2 | +1.3575904456e-08 | +9.7272970889e-09 | +2.3901169937e-09 | +1.4584903736e-09 | +3.8486073673e-09 |
| B | 0.001 | 6p_1/2 | +8.4201637885e-09 | +2.7745845567e-09 | +1.1667698135e-09 | +4.4788094183e-09 | +5.6455792318e-09 |
| B | 0.001 | 6p_3/2 | +1.2905924551e-08 | +9.2252881172e-09 | +2.1207754902e-09 | +1.5598609439e-09 | +3.6806364341e-09 |

All36 original absolute-state/line decisions replay literally; the four ordinary EOL callbacks are deliberately absent from this no-eigensolve decomposition. Exact endpoint Rayleigh slopes remain a separate diagnostic. In particular, exact Rayleigh evaluation still fails both D3/2 large-step line gates.

## Existing native centered helper: all forty ordinary outcomes

Current complete native `evd` helper, literal original H, all full blocks saved. Original analytic actions are reused without a new gradient. Absolute-state relative tolerances remain distinct from the stricter D3/2 line tolerance. EOL is the literal `float(weights @ sorted_full_energies)` callback.

| Arm | h | Kind | State/line | Analytic | Ordinary FD | Signed error | Original tolerance | Outcome |
|---|---:|---|---|---:|---:|---:|---:|---|
| A | 0.002 | state | 6s_1/2 | -0.166246919514 | -0.16624681848 | +1.0103367507e-07 | 1.6624691951e-06 | PASS |
| A | 0.002 | state | 5d_3/2 | -0.167133419764 | -0.16713330524 | +1.1452378152e-07 | 1.6713341976e-06 | PASS |
| A | 0.002 | state | 5d_5/2 | -0.168460411267 | -0.168460295754 | +1.1551295873e-07 | 1.6846041127e-06 | PASS |
| A | 0.002 | state | 6p_1/2 | -0.169510704961 | -0.169510592514 | +1.1244747125e-07 | 1.6951070496e-06 | PASS |
| A | 0.002 | state | 6p_3/2 | -0.169193791327 | -0.169193673628 | +1.1769897254e-07 | 1.6919379133e-06 | PASS |
| A | 0.002 | line | 5d_3/2 | -0.000886500249661 | -0.000886486759555 | +1.3490106449e-08 | 1.0000000000e-08 | **FAIL** |
| A | 0.002 | line | 5d_5/2 | -0.00221349175276 | -0.00221347727347 | +1.4479283661e-08 | 2.2134917528e-08 | PASS |
| A | 0.002 | line | 6p_1/2 | -0.00326378544705 | -0.00326377403326 | +1.1413796186e-08 | 3.2637854471e-08 | PASS |
| A | 0.002 | line | 6p_3/2 | -0.00294687181275 | -0.00294685514746 | +1.6665297475e-08 | 2.9468718128e-08 | PASS |
| A | 0.002 | EOL | EOL | -0.168109049367 | -0.168108937487 | +1.1187957375e-07 | 1.6810904937e-06 | PASS |
| A | 0.001 | state | 6s_1/2 | -0.166246919514 | -0.166246890331 | +2.9183593628e-08 | 1.6624691951e-06 | PASS |
| A | 0.001 | state | 5d_3/2 | -0.167133419764 | -0.16713338573 | +3.4033500410e-08 | 1.6713341976e-06 | PASS |
| A | 0.001 | state | 5d_5/2 | -0.168460411267 | -0.168460377608 | +3.3658435570e-08 | 1.6846041127e-06 | PASS |
| A | 0.001 | state | 6p_1/2 | -0.169510704961 | -0.169510669366 | +3.5595168955e-08 | 1.6951070496e-06 | PASS |
| A | 0.001 | state | 6p_3/2 | -0.169193791327 | -0.169193756847 | +3.4480207332e-08 | 1.6919379133e-06 | PASS |
| A | 0.001 | line | 5d_3/2 | -0.000886500249661 | -0.000886495399754 | +4.8499067818e-09 | 1.0000000000e-08 | PASS |
| A | 0.001 | line | 5d_5/2 | -0.00221349175276 | -0.00221348727791 | +4.4748419415e-09 | 2.2134917528e-08 | PASS |
| A | 0.001 | line | 6p_1/2 | -0.00326378544705 | -0.00326377903548 | +6.4115753263e-09 | 3.2637854471e-08 | PASS |
| A | 0.001 | line | 6p_3/2 | -0.00294687181275 | -0.00294686651614 | +5.2966137032e-09 | 2.9468718128e-08 | PASS |
| A | 0.001 | EOL | EOL | -0.168109049367 | -0.168109015249 | +3.4117776746e-08 | 1.6810904937e-06 | PASS |
| B | 0.002 | state | 6s_1/2 | -0.163876272208 | -0.16387617643 | +9.5778199791e-08 | 1.6387627221e-06 | PASS |
| B | 0.002 | state | 5d_3/2 | -0.164635755719 | -0.164635646797 | +1.0892161834e-07 | 1.6463575572e-06 | PASS |
| B | 0.002 | state | 5d_5/2 | -0.166011523776 | -0.166011413512 | +1.1026359895e-07 | 1.6601152378e-06 | PASS |
| B | 0.002 | state | 6p_1/2 | -0.166950915688 | -0.1669508124 | +1.0328886888e-07 | 1.6695091569e-06 | PASS |
| B | 0.002 | state | 6p_3/2 | -0.166510425515 | -0.166510320469 | +1.0504614554e-07 | 1.6651042551e-06 | PASS |
| B | 0.002 | line | 5d_3/2 | -0.000759483510903 | -0.000759470367484 | +1.3143418548e-08 | 1.0000000000e-08 | **FAIL** |
| B | 0.002 | line | 5d_5/2 | -0.00213525156766 | -0.00213523708226 | +1.4485399158e-08 | 2.1352515677e-08 | PASS |
| B | 0.002 | line | 6p_1/2 | -0.00307464348049 | -0.00307463596982 | +7.5106690867e-09 | 3.0746434805e-08 | PASS |
| B | 0.002 | line | 6p_3/2 | -0.00263415330684 | -0.00263414403889 | +9.2679457442e-09 | 2.6341533068e-08 | PASS |
| B | 0.002 | EOL | EOL | -0.165596978581 | -0.165596874467 | +1.0411398949e-07 | 1.6559697858e-06 | PASS |
| B | 0.001 | state | 6s_1/2 | -0.163876272208 | -0.163876244187 | +2.8020844509e-08 | 1.6387627221e-06 | PASS |
| B | 0.001 | state | 5d_3/2 | -0.164635755719 | -0.164635723195 | +3.2524063390e-08 | 1.6463575572e-06 | PASS |
| B | 0.001 | state | 5d_5/2 | -0.166011523776 | -0.166011491274 | +3.2501801948e-08 | 1.6601152378e-06 | PASS |
| B | 0.001 | state | 6p_1/2 | -0.166950915688 | -0.166950881976 | +3.3712524192e-08 | 1.6695091569e-06 | PASS |
| B | 0.001 | state | 6p_3/2 | -0.166510425515 | -0.166510393683 | +3.1831822043e-08 | 1.6651042551e-06 | PASS |
| B | 0.001 | line | 5d_3/2 | -0.000759483510903 | -0.000759479007684 | +4.5032188811e-09 | 1.0000000000e-08 | PASS |
| B | 0.001 | line | 5d_5/2 | -0.00213525156766 | -0.0021352470867 | +4.4809574385e-09 | 2.1352515677e-08 | PASS |
| B | 0.001 | line | 6p_1/2 | -0.00307464348049 | -0.00307463778881 | +5.6916796832e-09 | 3.0746434805e-08 | PASS |
| B | 0.001 | line | 6p_3/2 | -0.00263415330684 | -0.00263414949586 | +3.8109775336e-09 | 2.6341533068e-08 | PASS |
| B | 0.001 | EOL | EOL | -0.165596978581 | -0.165596946317 | +3.2263908049e-08 | 1.6559697858e-06 | PASS |

38/40 pass:20/20 state,4/4 ordinary EOL and14/16 line. Both failed lines are A/B D3/2 at h=0.002. The original00537/40 result and all three original failures remain unchanged. The new replay also remains scientifically unqualified.

## Identity, resources and limits

Exact decomposition:515 readback/identity/closure checks plus97 independent small Fraction controls;80 exact quadratics and7,799,576 triangle visits,10 saved NPZ loads.5.170s/460.133MiB. Native centered replay:384 solver/identity checks,50 complete evd blocks and50 exact Rayleigh controls,10 NPZ loads;7.005s/490.434MiB. Timings are unpaired bounded observations.

The current358-file source fingerprint is `f8307f6a107546aac9f528bf8344f127df0584ed352b3acd55678f2d731b49bf`; frozen005 source is `f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe`. No production source changed; all scientific inputs and both source maps are checked. All78,162,520 bytes of new complete block eigenarrays are retained. No new radial integral, H build, gradient, optimizer or NIST fit is performed.

The almost-fourfold CI-response scaling supports finite-step curvature; no independently calculated third derivative or certified truncation bound is available. The fixed-C0 remainder is not uniquely separated into radial curvature, constraint reconstruction and assembly. Current-source fresh-H/gradient admission, full4f12 finite-step response, stationarity and experimental accuracy remain unfinished.

[Exact decomposition](actual001/result.json), [native centered replay](centered001/result.json), [owning report](README.md).
