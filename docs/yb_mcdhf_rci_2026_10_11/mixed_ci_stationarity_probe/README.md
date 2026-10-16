# Yb⁺ 531-CSF mixed-CI orbital stationarity probe

The complete declared 531-CSF fixed005-orbital RCI model is **not stationary for its own five-state equal-weight MCDHF EOL objective**. Its original-metric orbital residual is 0.0367607184 against the unchanged 1e−7 gate. Two independently re-solved CI finite-difference directions confirm the analytic tangent derivative. One actual Armijo-accepted orbital step lowers the EOL average by 0.000264686647 Ha, but it remains nonstationary and worsens all four nominal DC NIST excitation gaps.

This is a bounded diagnostic, not a completed expanded-space MCDHF solution or a strict missing-physics error bound. The [prospective contract](contract.json) preceded execution and binds the frozen [balanced 531-CSF replay](../balanced_cv_replay/result.json), native005 radial data and source, and the original005 projector/inverse preconditioner metric saved in native004. No shared source, model, target weight or scientific threshold was changed.

## Fixed physical and numerical identity

- Yb⁺, 69 electrons, all 24 native005 radial spinors, 123 B-spline coefficients per component, identical finite Fermi empirical A175 nucleus.
- All 21 NR configurations, 531 complete signed CSFs and 16 J/parity blocks of the 4f/5p/5s single-hole plus complete `{6s,5d,6p}` two-particle frontier. No core-double, higher virtual n/l or other core-hole family.
- Dirac-Coulomb EOL objective: equal 1/5 on replay-bound 6s1/2, 5d3/2, 5d5/2, 6p1/2 and 6p3/2. Static Breit is not included in this orbital objective; no QED, recoil, dynamic Breit or empirical correction.
- Public `CIBasisTracking` freezes the full mixed-CI reference and resolves every perturbation against its physical radial basis. Every trial retains the same exact J/parity local-root identities and majority physical overlap. Global columns D3/2 and P1/2 exchange order after the step; state identity is preserved.
- Original metric SHA `aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe`: archived nuclear no-pair projectors, inverse diagonal and inverse blocks are byte-identical. No new endpoint preconditioner is used as a yardstick. Original orbital/energy gates remain 1e−7 / 1e−8 Ha.

Preflight maximum raw M=J is 232; the selected-state full-overlap pair upper count is 227,582 below 1,000,000. Source map `9dff33cc24acecff9e821c4f9b836e02efca0d615291d6f92ff5a7014a726c3f` and all 353 Python source hashes are verified. Actual saved spinors are used for the baseline gradient. Wide original-projector coordinate recovery reconstructs them to 2.99e−13 maximum coefficient error; the zero-coordinate EOL replay differs by 0 Ha at stored precision. See [preflight.json](preflight.json) and [source_identity.json](source_identity.json).

## Real mixed-CI gradient and independent response controls

Public `CSFOrbitalFunctional.evaluate` re-solves the complete CI problem and differentiates its signed Hamiltonian RDM. The Hellmann–Feynman derivative uses stationary mixed-CI eigenvectors, including configuration interference; it does not freeze a population-only density. The baseline full matrix/eigenvalues reproduce the saved 531 model within their declared gates. Full CI residuals, signed orthogonality, Hermiticity, cross-symmetry zeros, radial Gram constraints and no-pair leakage are checked at every trial.

For two original-projector tangent directions (normalized negative original-metric gradient and a valence-focused slice reprojected into the full tangent), independently re-solve CI at ±1e−4 and ±5e−5 after the unchanged public numerical polar retraction. Central slopes agree with the analytic derivative at all four checks; maximum absolute slope error is 1.119e−7 Ha per coordinate, within the prospective independent-derivative criterion. This validates these two directions, not every one of the 2,856 tangent-coordinate entries. Minimum physical squared trial overlap is 0.9993239636.

Both public stable-difference APIs explicitly return `None` for this mixed/tracked CI case. Every energy comparison therefore uses the full re-solved CI target; no five-one-CSF fixed-RDM polynomial or gradient-trapezoid acceptance is substituted.

## One bounded accepted step

| Quantity | Baseline | Accepted step | Original gate |
|---|---:|---:|---:|
| EOL energy, Ha | −14067.3971221840 | −14067.3973868706 | energy decrease for Armijo |
| Original-metric orbital residual | 0.0367607184 | 0.0249932989 | <1e−7 |
| Original-metric quadratic energy indicator, Ha | 0.000242640292 | 0.000466155583 | <1e−8 |
| Stationarity passed | false | false | both original gates |

The first unit preconditioned step is accepted by the unchanged Armijo coefficient 1e−4 using the actual mixed-CI EOL energy. No additional optimization iterations were run. The metric-dependent quadratic indicator is a diagnostic of this old metric and current gradient; its increase despite lower EOL energy shows why it must not be read as a strict physical/correlation error bound.

| State | Absolute ΔE, Ha | Same lower/upper transition Δ, cm⁻¹ | Baseline DC NIST gap, cm⁻¹ | Step DC NIST gap, cm⁻¹ |
|---|---:|---:|---:|---:|
| 6s_1/2 | -0.000584859 | +0.000 | +0.000 | +0.000 |
| 5d_3/2 | +0.000217506 | +176.099 | +2071.815 | +2247.914 |
| 5d_5/2 | +0.000441198 | +225.193 | +1328.794 | +1553.988 |
| 6p_1/2 | -0.000789763 | -44.971 | -1995.584 | -2040.555 |
| 6p_3/2 | -0.000607515 | -4.972 | -2075.444 | -2080.417 |

All four absolute nominal NIST gaps increase in this step. EOL average energy reduction is not evidence of improved experimental excitation energies, and one negative step does not establish the trend of a fully converged expanded MCDHF model. The full mixed model needs its own numerical stationarity qualification before a physical accuracy conclusion. NIST values are the original saved nominal references; no uncertainty-qualified experimental gate or fit is introduced.

## Public-interface boundary and measured budget

`CSFOrbitalFunctional`, `CIStateAverage`, `CIBasisTracking`, the public CI matrix/solver and shared radial geometry supply the physics. The diagnostic reuses the existing private numerical `_polar_retraction` and composes its one step with the saved original coordinate layout/inverse metric. Public `optimize_orbitals` currently rebuilds its projector and spectral/occupation-scaled preconditioner internally and has no validated original-layout/inverse-metric resume argument. This composed diagnostic is therefore not represented as a full public `MCDHFSolver` run. No initializer or model was silently substituted.

The prospective cap was CPU1, 3.95 GiB address space and 890 s. Actual observation: 16.69 s, 823.97 MiB peak RSS, two analytic gradients, eight central CI energy trials and one accepted line-search trial. Tracking, CI and signed angular caches are warm within the run. These uncontrolled observations do not establish a benchmark or a full-optimization runtime bound.

## Verification and immutable evidence

68 run checks pass. [Independent saved-array verification](independent_verification.json) passes 432 checks, including all frozen inputs/source, original metric actions, complete CI spectra/Gram/residuals, physical-overlap identity, the actual finite-difference slopes, original-geometry Gram and lower/upper energy differences. It performs no extra CI solve or third gradient. Ruff passes for both scripts.

- [result.json](result.json), SHA `1c0663f5897fc13fd7f0734491fad81e354a5543ed82c2941172390eb400e31e` — fixed numerical result, negative stationarity/gap outcome and all trials.
- [verification.json](verification.json) and [independent_verification.json](independent_verification.json) — numeric/source identity gates.
- [card_evidence.json](card_evidence.json), SHA `40ebcb5b28be935109fd1d6c8549cb0d01861dbc9ee97149e1ae55e94436c08e` — stationary=false, completed-MCDHF=false and gap-improvement=false.
- [baseline_arrays.npz](baseline_arrays.npz), [line_trial_0.npz](line_trial_0.npz), [accepted_gradient.npz](accepted_gradient.npz) and [derivative_directions.npz](derivative_directions.npz) — actual signed state/orbital/gradient evidence; every perturbation also has its own NPZ.
- [run.py](run.py), [run.log](run.log), [verify.py](verify.py), [verify.log](verify.log) — execution and independent replay. Run a copy when reproducing; these hashes must remain immutable.

## Documentation impact and scoped lesson

Type: current bounded diagnostic. Dependencies: the frozen005 mixed functional/CI/tracking/polar APIs, the unchanged original metric and frozen balanced replay identities in the contract. No public source, interface, default, model or threshold changed. Prior native005, frontier and balanced-replay results remain unchanged and linked; this record supersedes no earlier numerical certificate. Root owns current consolidated navigation and cards.

Supported topic-local action: after enlarging the CI correlation space on old stationary orbitals, compute that new objective’s actual orbital residual and independently test its physical CI response before reusing any old MCDHF certificate. Here the residual rises from the bare005 5.14e−8 to mixed531 0.0368 in the same metric. Separately, a same-target EOL energy decrease can worsen every experimental transition gap; qualify stationarity and accuracy independently. This follows existing AGENTS principles rather than introducing a new global rule.
