# First-20 completion audit and normalization branch repair

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The stationary campaign has committed all 15,086 attempts: 14,843 successful
and 243 failed. There are 7,421 configurations with two successful grids.
This completes queue execution, not numerical acceptance or the requested
all-element database. The source-locked historical database is preserved.

Of the failed attempts, 220 are orbital failures in neutral K/Ca (54/166),
and 23 are SCF convergence failures across other stages.

## Corrector defect repaired

For a fixed local operator and source,

`N(e) = sum_j |<j|chi>|² / (epsilon_j-e)²`.

Within a pole interval `N''(e)>0`. When `N'(e)<0`, the desired increasing-branch
normalization root, if it exists, is to the right. The old step used `N-1`
to select direction and moved left when `N>1`, toward the preceding pole.
The numerical corrector now retains this point as a lower bound and moves
right. No Hamiltonian, exchange term or tolerance changes.

An independent two-eigenstate source regression starts near the preceding
hydrogenic pole. The resulting energy and wavefunction are checked against
the spectral resolvent formula, with a scalar root found independently.
The four weak-source tests, including the two B-spline comparisons, passed.

## Remaining strong-source problem

The Ca I `1s2 2s2 2p6 3s2 3p6 4s1 5d1` failure was captured at its 5d3/2
orbital update. Correcting the search direction exposes a normalization minimum:

- Energy: −0.0267203859792 Ha.
- Minimum squared norm: 1.03886347220019.

An independent scalar minimization of the same full-box Green solution confirms
this minimum. Thus the frozen source has no unit-norm solution in the selected
local-pole interval. This is not repaired by allowing more Newton iterations
or accepting a small energy step. The state remains failed. The failure does
not prove that the full self-consistent DHF state is absent; source updates and
nonlocal branch tracking need further work.

The captured arrays, hash and failure-group summary are in
`output/nist_spectra/normalization_diagnostic/`. These are diagnostic artifacts,
not new production predictions. The repaired algorithm has not yet been used
to regenerate the full campaign or establish how many failures it recovers.

## Reference comparison snapshot

The completed historical campaign retains all 35,802 standard observed lines:
32,229 configuration-average predictions, 260 inverted configuration orders,
244 lines with unavailable endpoints, 176 intra-configuration unresolved lines,
2,883 missing classifications and 10 unique-level comparisons. The 10 strict
matches have RMS 0.000417658 cm⁻¹; this tiny selected subset does not measure
the accuracy of the full periodic table. There are 2,008 evaluated centroid
comparisons and 5,203 unscored manifold-coverage records.

## Damped SCF recovery

Three alternative local-exchange starters at ordinary damping all failed for
the representative Ca I 4s5d state. Increasing the old-orbital fraction in the
SCF update, with the full exchange operator retained, recovered the state:

| Engine | Grid / controls | DC total energy (Ha) | Iterations |
|---|---|---:|---:|
| Shooting | 3200, damping 0.8 | failed | — |
| Shooting | 3200, damping 0.9 | −679.5476493579953 | 86 |
| Shooting | 3200, damping 0.95 | −679.5476493537336 | 166 |
| Shooting | 6400, damping 0.95 | −679.5476493537337 | 166 |
| B-spline | independent GSZ initialization, refined knots | −679.5476493693473 | — |

All successful runs report convergence. Shooting keeps the original 1e-12
relative-energy tolerance and occupied-rotation gate. The independent B-spline
run uses orbital tolerance 1e-8. Its energy differs by 1.56e-8 Ha from the
0.95-damped shooting solution. These are matched Dirac–Coulomb controls with
isotope 40; corrections are excluded from both sides of this control.
Different damping results differ by 4.26e-9 Ha, so the much smaller two-grid
change must not be interpreted as the full numerical uncertainty.

`recover_native_shooting_failures.py` now reruns all 122 historically failed
configurations on both grids (244 attempts) in a separate source-locked
database, using damping 0.95 and at most 600 iterations. Each production-style
attempt applies the original strict Breit/QED/recoil stack after SCF.
Historical failures remain unchanged. The recovery ledger is a targeted
validation artifact; it does not establish complete regenerated coverage.

Machine-readable matched-engine results:
`output/nist_spectra/normalization_diagnostic/calcium_damping_controls.json`.
Five queue/recovery tests passed, including selection of the successful sibling
grid when only one of a configuration's two historical attempts failed.

## Independent box refinement for the remaining neutral 5g failures

Mg I `1s2 2s2 2p6 3s1 5g1` and Al I `1s2 2s2 2p6 3s2 5g1`
remained unconverged with damping 0.95 in their inherited 75-bohr boxes.
Both converge after enlarging the box, with the same tolerance and corrections:

| Atom | Box (bohr) | 3200-point corrected energy (Ha) | 6400-point corrected energy (Ha) |
|---|---:|---:|---:|
| Mg | 125 | −199.65393015019794 | −199.65393015019853 |
| Mg | 250 | −199.65393015019782 | −199.65393015019865 |
| Al | 125 | −242.07294083470558 | −242.07294083470612 |
| Al | 250 | −242.07294083470572 | −242.07294083470630 |

All eight calculations converged. This establishes successful shooting box/grid
controls for these two configurations; the 75-bohr failed attempts remain in
their ledgers. It does not establish independent B-spline agreement for these
new enlarged-box states or bound every contribution to their numerical error.
The source-identified artifact is `native_shooting_5g_box_controls.json`, generated
by `probe_native_shooting_box_recovery.py`. Box and grid refinement must remain
separate acceptance checks in any subsequent complete campaign.

## Potassium 19d continuation control

K I `1s2 2s2 2p6 3s2 3p5 4s1 19d1` still failed at damping 0.95.
At the inherited 1805-bohr box and original tolerance, stronger damping gives:

| Grid | Damping | Corrected energy (Ha) |
|---:|---:|---:|
| 3200 | 0.98 | −600.3734659372649 |
| 6400 | 0.98 | −600.3734659373141 |
| 3200 | 0.99 | −600.3734659220232 |

All three converge with the original strict correction stack. Grid refinement
changes the energy by 4.92e-11 Ha, whereas changing damping changes it by
1.52e-8 Ha. That difference must remain part of the convergence assessment;
a grid-only error estimate would overstate precision. The full records and
resolved numerical controls are saved in
`normalization_diagnostic/potassium_19d_controls.json`.
This recovers a representative high-Rydberg state, not the whole remaining
series. A final local refinement from the damped state and verification across
the failed series remain necessary before a complete campaign is accepted.

## Potassium warm refinement

The recovered K I core-hole 19d state at 3200 points and R=1805 bohr was
refined from damping 0.98/tolerance 1e-12 using damping 0.5/tolerance 1e-13.
Seven additional iterations lowered the bare energy from
−600.6529690481898 to −600.6529690664855 Ha; the occupied-rotation residual
fell from 3.54e-9 to 2.20e-11 Ha. The strict Breit/QED/recoil corrected energy
of this refined state is −600.3734659537305 Ha.

A further 1e-14 attempt failed with the explicit weak-source-pole regime
check. This is not an arbitrary-precision convergence result, nor a substitute
for a refined two-grid comparison. The complete successful/failed sequence and
resolved correction specification are retained in
`output/nist_spectra/normalization_diagnostic/potassium_19d_warm_refinement.jsonl`.

## Completed recovery audit and stricter SCF convergence

The original-source recovery is terminal: **244/244 attempts**, comprising
74 successful and 170 failed attempts across 122 configurations. Exactly
37 configurations succeeded on both grids. Their largest grid difference is
2.90e-10 Ha. This remains a diagnostic ledger: small grid differences do not
bound the damped SCF error. The original records were not overwritten.
`normalization_diagnostic/damped_recovery_audit.json` preserves the campaign
identity, all successful pairs and failure categories. Of the failed attempts,
166 are K/Ca normalization failures and four are the Mg/Al 5g box cases already
examined separately above.

The 6400-point K I 19d warm refinement now also completed:
−600.3734659537807 Ha, versus −600.3734659537305 Ha on the coarse grid.
Both require seven warm iterations at damping 0.5/tolerance 1e-13. Conversely,
Ca I 4s20d fails during warm refinement on **both** grids even though its initial
0.98-damped run reports convergence. Strong damping alone is not a complete fix.
Full records: `normalization_diagnostic/warm_polish_grid_controls.json` and
`strong_damping_sample.json`.

A newly captured Ca I 4s6d failure at damping 0.98 occurs on sweep 58, in the
kappa=2 orbital. Independent scalar minimization of its frozen-source norm
between the adjacent local 5d and 6d poles gives:

- Local poles: −0.021077634421933722 and −0.014542443090839007 Ha.
- Minimum at −0.01758599085687124 Ha.
- Minimum norm squared: **1.0011125830702312**.

Thus this frozen equation has no normalized solution in that pole interval.
It does not prove absence of a self-consistent DHF state. Loosening a root
solver threshold or renormalizing the minimum would fail to solve the supplied
inhomogeneous equation. Frozen input, capture context and minimization samples
are retained as `calcium_6d_failed_source.npz`, its JSON companion and
`calcium_6d_normalization_minimum.json` in the diagnostic directory.

A separate correctness repair now checks the largest normalized orbital update
**before damping**, in addition to energy change and occupied-rotation residual,
for the coupled nonlocal solver. Its threshold is the square root of the capped
relative energy tolerance, reflecting that energy stationarity alone is a
second-order test. This is an explicit convergence policy, not a rigorous
experimental error bound. Both the measured update norm and threshold are
recorded in solver diagnostics. Legacy Schmidt stopping policy is unchanged.

The rationale also agrees with the supplied DBSR_HF source:
`DBSR_HF/hf_solve_HF.f90` computes separate `orb_diff` and `scf_diff`, and requires
both to pass. Its coefficient maximum-norm definition is not copied onto native
grid spinors; atomx uses its native radial quadrature norm.

This prevents damping-induced energy plateaus from certifying convergence.
It does not yet solve the no-normalized-root SCF steps; an exchange update
strategy that treats the orbital dependence consistently remains necessary.

Verification for the undamped-update gate: **25 tests passed** across synthetic
SCF convergence, recovery selection, numerical boundaries, warm starts,
open-shell constraints, core-hole stationarity and weak exchange. The synthetic
regression holds energy and rotation residual fixed while the undamped orbital
update remains large; strong damping must not make that state converge.
Pytest reports the environment's two missing-timeout-plugin configuration
warnings. The full suite and a regenerated first-20 campaign have not been run.

Reproduce the captured Ca 6d norm scan with:

```sh
PYTHONPATH=src python scripts/diagnose_shooting_normalization.py \
  output/nist_spectra/normalization_diagnostic/calcium_6d_failed_source.npz
```

The command rejects constrained inputs (whose poles require a different
inertia calculation), failed local eigenproblems and failed minimizations.
It does not write to a campaign database or change a failed attempt's status.

Fresh coarse-grid controls with the undamped-update gate both converged at
0.98 damping, tolerance 1e-12 and unchanged strict corrections:

| Configuration | Corrected energy (Ha) | Undamped update norm | Occupied rotation residual (Ha) |
|---|---:|---:|---:|
| Ca I 4s20d | −679.188905449301 | 9.8681e-7 | 2.4150e-13 |
| K I 3p-hole 4s19d | −600.3734659546261 | 9.9019e-7 | 1.9863e-12 |

These change the previous 0.98-damped energies by 3.43e-8 and 1.74e-8 Ha,
respectively, larger than the earlier grid differences. They establish a
measurable convergence correction, not complete numerical acceptance. The
source-hashed records are `orbital_convergence_gate_sample.json`. Both-grid
warm-release controls have been started separately in
`strict_orbital_gate_warm_grid_controls.json`; their outcome remains pending.
