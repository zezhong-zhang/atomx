Both same-model native pilots completed 32 accepted updates from the literal global 208 point. All 64 saved finite-H acceptance certificates qualified; neither terminal satisfies the original stationarity gates. No scientific calculation or array replay was added by this closure.

The two trajectories share saved coordinate/state, H owner, physical UID, exact energy-ledger and original metric identity through local update 11. They first differ at local update 12/global 220, the first direction that can use a surviving secant pair beyond capacity 10. This is an observed search-memory effect in this window; the public diagnostics do not export actual retained pair counts, transport drops or history resets. Capacity 40 cannot be fully populated in 32 updates.

| At the common 32-update horizon/global 240 | history10 | history40 |
|---|---:|---:|
| Reported EOL (Ha) | -14067.423393554487 | -14067.423394399128 |
| EOL change from 208 (Ha) | -5.91554089624e-06 | -6.76018134982e-06 |
| Original orbital residual, required <1e−7 | 0.00035579232094 FAIL | 0.000648536557395 FAIL |
| Original indicator (Ha), required <1e−8 | 3.68579044615e-07 FAIL | 3.31255697121e-07 FAIL |
| Nominal four-line RMS (cm−1) | 2051.371139733 | 2051.470109668 |
| H / CI / full gradient calls | 47 / 47 / 33 | 43 / 43 / 33 |
| Saved accepted backtracks / fallback acceptances | 14 / 0 | 10 / 0 |

At this horizon history40 has a lower EOL and four fewer H/CI evaluations, while history10 has a smaller original residual. The indicator is smaller for 40. Both terminate at the declared iteration cap. These outcomes do not select a single winner, establish that memory 40 is generally preferable, or justify a parallel speedup claim. Concurrent timings were uncontrolled.

The physical lower state and all four upper states remain matched by their original UID, with equal 0.2 EOL weights. Individual target shifts demonstrate why reducing EOL does not guarantee lower transition errors: the 6s lower state rises while the four upper energies decrease. Changes below are each branch terminal minus its common 208 baseline.

| Original target | ΔE 10 (Ha) | ΔE 40 (Ha) | Δtransition 10 (cm−1) | Δtransition 40 (cm−1) |
|---|---:|---:|---:|---:|
| 6s_1/2 | 2.37966123677e-05 | 3.32623239956e-05 | 0.000000000 | 0.000000000 |
| 5d_3/2 | -5.91957177676e-06 | -4.46687408839e-06 | -6.521948561 | -8.280601841 |
| 5d_5/2 | -1.93383039004e-05 | -1.5500916561e-05 | -9.467019847 | -10.702294245 |
| 6p_1/2 | -2.21063291974e-05 | -2.62265293713e-05 | -10.074531178 | -13.056294163 |
| 6p_3/2 | -6.0101156123e-06 | -2.0868908905e-05 | -6.541820636 | -11.880432386 |

| Upper target | Excitation 10 (cm−1) | Excitation 40 (cm−1) | Nominal gap 10 (cm−1) | Nominal gap 40 (cm−1) |
|---|---:|---:|---:|---:|
| 5d_3/2 | 25891.255359808 | 25889.496706527 | 2930.455359808 | 2928.696706527 |
| 5d_5/2 | 26762.130666114 | 26760.895391716 | 2429.440666114 | 2428.205391716 |
| 6p_1/2 | 25955.569205451 | 25952.587442467 | -1106.250794549 | -1109.232557533 |
| 6p_3/2 | 29334.424334820 | 29329.085723069 | -1057.805665180 | -1063.144276931 |

| Terminal multiplet | Centroid 10 (cm−1) | Centroid 40 (cm−1) | Splitting 10 (cm−1) | Splitting 40 (cm−1) |
|---|---:|---:|---:|---:|
| 5d | 26413.780543592 | 26412.335917640 | 870.875306306 | 871.398685188 |
| 6p | 28208.139291697 | 28203.586296202 | 3378.855129369 | 3376.498280602 |

The Hamiltonian, 1046 target CSFs, 29 radial orbitals, fixed nuclear positive-energy P, 3451 coordinate metric, five original tracker UIDs and stopping thresholds stayed fixed. The saved “screened Fock spectral gaps” metadata identifies the inherited inverse metric; it does not imply a fresh correlated Fock or Hessian. The indicator and numerical residual are not bounds on missing correlation or the experimental spectrum. No Breit/QED correction is added to these DC points.

The sole stdlib closure completed in 3.019 s on CPU 30 under 1 GiB/60 s caps: 15 metadata checks per branch, 0 NumPy/Atomx/SciPy imports, 0 array numerical replays and 0 H/CI/gradient/Dirac/eigensolver/optimizer calls. Each live branch has 2219 passing numerical controls, 33 qualified checkpoints, 32 qualified finite-H acceptances and no failed live control or forensic unqualified candidate. The closure checks every manifest/NPZ byte hash, complete original UID mapping and exact Fraction acceptance ledger; it does not independently rebuild H/C/g.

All trajectory points, exact certificates, work accounting and terminal five-state rows are preserved in [the comparison completion](completion.json), [history10 completion](history10/completion.json) and [history40 completion](history40/completion.json). [The closure receipt](completion_execution.json) binds the sole execution to [the prospective closure contract](completion_contract.json) and [script](complete_saved.py). The scientific pair is bound by [the root pair receipt](root_pair_execution.json); prior derivative failures and stationarity failures remain unchanged. No subsequent run is authorized by this report.
