# Closed run005: original same-space continuation

The sole public reported-EOL optimizer call returned after128 new accepted steps, global208, with all8233 live controls and all128 finite stored-H descent certificates passing. It remains nonstationary: original residual 0.000955326596362 exceeds1e−7; indicator 1.45943653075e-06Ha exceeds1e−8. The indicator is not a rigorous spectral or energy-error bound. No failed live gate or unqualified checkpoint was found. This result does not certify physical completeness or experimental accuracy.

Starting at the literal global80 endpoint, EOL fell -0.000530148017788Ha and four-line nominal RMS changed from 2105.098759493 to 2054.316460634cm−1. Individual same-point lower/upper changes and transition shifts distinguish energy minimization from gap reduction; no minimum-gap checkpoint was selected.

| Physical target | Final excitation/cm−1 | Nominal gap/cm−1 | Energy change since80/Ha | Transition shift since80/cm−1 |
|---|---:|---:|---:|---:|
| 6s_1/2 | 0.000000 | 0.000000 | -0.000166139645444 | 0.000000 |
| 5d_3/2 | 25897.777308 | 2936.977308 | -0.000651774591461 | -106.584551 |
| 5d_5/2 | 26771.597686 | 2438.907686 | -0.000689408909238 | -114.844329 |
| 6p_1/2 | 25965.643737 | -1096.176263 | -0.000547372894289 | -83.671027 |
| 6p_3/2 | 29340.966155 | -1051.263845 | -0.000596044052145 | -94.353111 |

Terminal5d centroid/splitting are 26422.069534924/873.820377592cm−1;6p are 28215.858682513/3375.322418826cm−1. All values use the original physical UIDs and lower6s reference in the same terminal model.

Actual producer calls:187H,187CI,129 full gradients,7 positive-space validation Dirac solves; original P/3451-coordinate inverse metric,1046 target CSFs,29 radial orbitals,69electrons, equal five-state weights, ordinary reported objective and no energy callback were retained. The exact finite-H consumer examined14,851,641 signed pairs and7,054,022 nonzero products. Runtime was858.410s uncontrolled; it is not a performance comparison. L-BFGS history was explicitly rebuilt by the public resume interface.

[completion.json](completion.json) binds actual result, receipt,129 manifests/arrays and source/metric/UID lineage. Once-only [complete_saved.py](complete_saved.py) checked11 metadata conditions and reconstructed the saved Fraction acceptance ledgers, RMS and multiplets; it did not recompute H/C/g or load NumPy. [completion_execution.json](completion_execution.json) records exit0/5.671s,CPU30,1GiB and60s cap. Existing derivative failures and prior HOLD contracts remain historical failures. The next prepared diagnostic reads only global80 and this declared global208 saved g/pg to locate actual coordinate residual and cross-coupled drivers before choosing further optimization.
