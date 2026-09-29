# Exchange-dependent shooting recovery: diagnostic and implementation plan

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The Ca I 4s6d normalization failure is an SCF iteration failure, not evidence
that the nonlocal DHF operator has no bound state. The captured intermediate
local field and *fixed* exchange source have a minimum norm squared above one
(see `SHOOTING_NORMALIZATION_BRANCH_2026-09-29.md`). Recomputing exchange for
the trial orbital changes that intermediate problem.

## Same nonlocal operator, different numerical solve

For fixed other orbitals and a fixed current local field, write the radial
operator as `F = h_local - K`, with `K` the exchange action. The exchange
source is linear in the target spinor. For an off-pole energy shift sigma,

```
G = (h_local - sigma)^(-1)
(F - sigma)^(-1) = (I - G K)^(-1) G
```

The existing full-box shooting Green solver supplies G. A matrix-free GMRES
solve applies `(I-GK)^(-1)`; a shift-invert eigensolver resolves nearby bound
states. Vectors are scaled by the square root of the native radial quadrature
weights. The zero-weight origin is omitted from these coordinates and restored
as zero in sampled spinors. No local exchange approximation replaces K and no
Breit/QED/recoil term is changed. The current local field remains frozen during
this *inner* solve; the outer SCF still has to update it and converge.

This is an eigenproblem for the full exchange action in a fixed field, unlike
the fixed-source normalization root problem. The supplied DBSR_HF implementation
likewise builds and diagonalizes its complete nonlocal orbital matrix in
`DBSR_HF/hf_solve_HF.f90` (`hf_matrix` followed by `hf_eiv`). This proposal uses
shooting inverses rather than adopting its B-spline representation.

## Frozen Ca 6d diagnostic

At the captured sweep-58 failure, kappa=2, sigma=-0.017553067058200728 Ha:

| Energy (Ha) | Overlap with incoming orbital | Green-equation residual norm |
|---:|---:|---:|
| -0.0173993080145677 | 0.9855327435529354 | 2.986e-12 |
| -0.0123720411834950 | 0.0183753882891143 | 1.637e-12 |
| -0.0262140657151482 | 0.1549875984382966 | 1.024e-11 |
| -0.0092425260069153 | 0.0075809534140114 | 6.719e-10 |

The first root is a promising continuation candidate. These are intermediate
orbital energies, not final atomic energies or NIST transition predictions.
The residual directly compares `psi` against `G(E) K psi`; it does not certify
radial-grid convergence or all-electron SCF stationarity. The 0.15-second inner
solve observed here is a single diagnostic timing, not a general speedup claim.

Artifact: `output/nist_spectra/normalization_diagnostic/calcium_6d_nonlocal_frozen_probe.json`.

## Integration gates

1. Require convergence of the inner linear/eigenvalue solves and verify the
   original exchange-dependent Green equation at the returned energy.
2. Preserve the requested root through energy, overlap and orbital-identity
   checks; refuse ambiguous or complex roots. An overlap alone is insufficient.
3. Handle occupied same-kappa constraints through the common bordered response.
   The current experiment is deliberately limited to unconstrained orbitals;
   constrained cases must continue to use their existing solver or fail clearly.
4. Test analytic/separable exchange operators with known eigenvalues, then the
   captured real-atom failure, followed by two-grid and reduced-damping controls.
5. Record the alternative numerical solve and source identity in diagnostics.
   Do not rewrite the historical campaign's failed records.
6. Keep exchange construction in methods and the generic resolvent/eigenvalue
   algorithms in numerics. Do not put a complete solver in an exception handler.

An isolated SCF experiment used this recovery only for explicit
weak-source-regime failures. It additionally requires overlap above 0.9 and the
requested dominant radial-node count. These are continuation safeguards, not a finished general state-acceptance
policy. The subsequent bounded integration is recorded below.

## Refined controls available during this experiment

With the separate undamped-update gate, K I core-hole 19d passes both-grid
warm refinement. The corrected energies are −600.3734659546568 Ha (3200)
and −600.3734659547061 Ha (6400), differing by 4.93e-11 Ha. Each warm stage
takes three iterations at damping 0.5/tolerance 1e-13 after 876 strongly damped
iterations. Its undamped update norm falls to about 2.05e-7, below the declared
3.1623e-7 threshold. This is a numerical control for that configuration, not
an experimental uncertainty estimate.

Ca I 4s20d still fails the warm-release root solve on the coarse grid even
after the stricter damped bootstrap passes. This confirms that the orbital
convergence gate alone does not repair the missing fixed-source roots.
The fine-grid Ca control and isolated Ca 6d recovery are still running at
this checkpoint; their results must be read before integration proceeds.

## Isolated recovery completed

Ca I 4s6d at 3200 points converged with corrected energy
**−679.205058221052 Ha**, including the original strict Breit/QED/recoil stack.
It required 389 exchange-dependent orbital recoveries. The largest selected
root's Green-equation residual was 2.9862e-12; the final undamped update norm
was 9.9313e-7 at a 1e-6 threshold. The complete sequence is archived in
`calcium_6d_nonlocal_recovery_probe.json`; the diagnostic scripts and source
inventories are alongside it. This single-grid result is not final acceptance.

Both Ca I 4s20d warm-release controls without this recovery have now failed,
including the 6400-point case. The earlier K controls passed on both grids.

## Integrated numerical recovery

- `numerics/shooting/nonlocal_eigen.py` owns the matrix-free resolvent solve.
  It requires successful GMRES/eigenvalue solves and one real bound root passing
  overlap, dominant-node and original Green-equation residual checks.
- `methods/shooting_hf.py` supplies the linear exchange action from the same
  angular coefficients and radial Coulomb kernels as its ordinary orbital field.
  Other orbitals are frozen; the local field is updated by the outer SCF.
- `FrozenSourceConvergenceError` distinguishes this specific failure from
  general orbital errors. The coupled SCF engine invokes the recovery only for
  that exception and only if the target has no occupied same-kappa constraints.
  Other error paths still propagate; the legacy Schmidt solver is unchanged.
- Results record `nonlocal_resolvent_recoveries` and
  `nonlocal_resolvent_max_residual`. Historical campaign records remain intact.
- **26 tests passed**: known rank-one exchange spectral shift, wrong-state
  rejection, SCF convergence, numerical dependency boundaries, weak exchange,
  warm starts, open-shell constraints and core-hole stationarity. The rank-one
  test checks the expected eigenvalue shift to 2e-9 Ha and the original equation
  residual below 1e-8. The environment emits its two timeout-plugin warnings.
  `git diff --check` passed. No full-suite or full-table accuracy claim is made.

New integrated controls for Ca 4s6d and 4s20d at 3200/6400 points are running,
including strong-damping bootstrap followed by reduced-damping refinement.
Their output is `nonlocal_resolvent_warm_grid_controls.json`. Do not accept the
integrated recovery for a complete campaign until those results and relevant
state/grid/cross-engine controls have been inspected.

## Integrated warm-release findings and NIST diagnostic

The integrated 4s20d controls both fail during reduced-damping refinement with
`nonlocal resolvent GMRES failed: status=100`. The coarse 4s6d control likewise
fails during warm release after its bootstrap succeeds. A shift-invert operator
at a nearly converged physical energy is close to singular. A separate experiment
now displaces the numerical shift by `-max(abs(energy)*1e-3,1e-8)` while retaining
the original equation residual and root checks. No production shift policy has
been changed at this checkpoint. The results of
`nonlocal_shift_offset_ca20d_control.json` remain pending.

An independent Ca 4s6d B-spline Dirac–Coulomb control is also running with the
same Fermi nucleus (A=40), statistical configuration and inherited box. Its
bare SCF energy must be compared with the shooting bare SCF energy, not with
a corrected total. Output: `calcium_6d_bspline_control.json`.

The matching Ca ground-state reference has completed under the strict orbital
update gate with the same correction model:

| Grid | Corrected ground energy (Ha) |
|---:|---:|
| 3200 | -679.3762341658286 |
| 6400 | -679.3762341658271 |

The archive contains the complete four-level 4s6d manifold: listed and expected
statistical weights are both 20. Its evaluated centroid is 45035.7541 cm^-1;
the ground centroid is zero. Using the *isolated prototype's coarse-grid*
recovered upper energy gives 37568.7773781 cm^-1, a residual of
**-7466.9767219 cm^-1 (-16.5801%)**. Physical and correction specifications of
these two endpoints were compared for equality before subtraction. Their source
identities differ and the excited state still lacks final numerical acceptance,
so this is explicitly a provisional diagnostic, not an accepted line prediction.
Do not conflate the small grid difference with agreement with experiment or
attribute the whole discrepancy to correlation before completing numerical and
state-identity checks.

Artifacts: `calcium_ground_strict_reference.json`,
`calcium_6d_nist_matching.json`, and
`calcium_6d_provisional_centroid_comparison.json`. The last records hashes of
its input artifacts and the remaining limitations. The centroid values here
come from the local NIST archive; a fresh attempt to retrieve the official ASD
energy endpoint was inaccessible through the web tool and does not constitute
new live verification.

## Offset and retained-formulation experiment

The displaced inverse shift removes the immediate Ca 4s20d GMRES failure, but
alternating the recovered eigenproblem with frozen-source updates remains
unstable: a 300-iteration warm stage ends with an undamped update norm of
0.01790. It is correctly recorded as unconverged.

Retaining the exchange-dependent formulation for each orbital after its first
recovery resolves this case in the isolated coarse-grid experiment: the warm
stage converges in 19 iterations, undamped update norm 1.7530e-7 and corrected
energy -679.1889054493031 Ha. Thirty implicit orbital solves were recorded.
The prototype's native recovery counter remains zero because interception is
outside the engine; the explicit `latched_recovery_events` list is the
correct accounting for this experiment. Artifact:
`nonlocal_latched_ca20d_control.json`.

The offset and retained-formulation changes are prepared in
`/tmp/atomx-resolvent-followup`, alongside the NIST parser prevention fix.
They are not yet applied while the source-guarded independent B-spline control
is live. Their known-answer tests include a shift-invert request whose incoming
energy is exactly the physical eigenvalue of a separable exchange model.

## Staged integrated controls and website update

The isolated, tested follow-up package now passes the full bootstrap and
warm-release sequence for Ca 4s20d on both grids:

- 3200: -679.1889054493031 Ha.
- 6400: -679.1889054493636 Ha.
- Difference: approximately 6.05e-11 Ha.

Ca 4s6d also passes the coarse-grid sequence at -679.2050582212274 Ha; its fine
grid remains pending at this checkpoint. Records and staged-source hashes are
in `nonlocal_retained_warm_grid_controls.json`. The working source has not yet
received the follow-up package because the independent B-spline control is
still running with its original source guard.

The interactive campaign page now includes a Ca I 4s6d investigation panel,
with the live NIST levels/reference, missing-uncertainty explanation, provisional
centroid comparison and a button selecting the related observed-line records.
It explicitly keeps the provisional recovery outside accepted scores. The
published comparison snapshot was refreshed with the repaired archive hash;
all scientific fields were verified unchanged and the previous snapshot was
saved. JavaScript syntax and the native campaign route test passed. No live
browser visual inspection was performed for this panel update.

## Applied follow-up and expanded recovery

All four staged controls completed successfully. Ca 4s6d's fine-grid corrected
energy is -679.2050582212258 Ha, versus -679.2050582212274 Ha coarse. Combining
these with the matched strict ground references gives excitation energies
37568.7773395739 and 37568.7773395989 cm^-1. The fine-grid NIST residual remains
-7466.9767604012 cm^-1; tiny grid differences do not resolve that physical gap.
See `calcium_6d_refined_centroid_comparison.json` for endpoint identity checks
and input hashes.

The independent B-spline run ended at **300 iterations, unconverged** at its
requested orbital tolerance of 1e-9. Its final bare energy,
-679.5387269635809 Ha, agrees with shooting (-679.5387269635793 Ha fine) to
about 1.6e-12 Ha. This is energy agreement, **not a converged independent-engine
acceptance result**. The complete report is `calcium_6d_bspline_control.json`.

With that original-source run finished, the displaced inverse shift, retained
orbital formulation and NIST HTML-cleaner fix were applied to the working tree.
`resolvent_followup_applied.json` records before/after hashes. **21 applied-tree
tests passed**, including exact-eigenvalue inverse conditioning, known nonlocal
spectral shifts, wrong-state rejection, NIST parentage preservation/migration,
phase convergence and NIST comparison contracts.

`scripts/recover_native_shooting_refined.py` now drives both-grid recovery of
all **122 historically failed configurations / 244 attempts**. Each attempt
requires bootstrap plus a reduced-damping refinement before corrections are
applied. Failed refinement records preserve completed stage diagnostics;
a successful bootstrap alone is not published as a corrected result. The two
previously validated Mg/Al 5g boxes are explicitly enlarged to 250 bohr.

The run uses `/tmp/atomx-refined-recovery-20260929`, whose package hashes were
checked against all four validated controls. It writes a separate database,
`native_shooting_first20_refined_recovery.sqlite`, and leaves historical attempts
unchanged. Five queue/recovery tests and two real-helium phase tests passed.
The latter verify that failed refinement never applies corrections and that a
successful refinement records its actual final tolerance.

Initial expanded results reveal additional reduced-damping instabilities in
Be II 7f and several neutral/open-shell high-l configurations: their energy
change can be small while the undamped orbital update remains around 0.002–0.004.
Those attempts are recorded as failures, not accepted on energy alone. They
require further investigation; the first-20 campaign is not yet fully accepted.

## Weak-exchange residual conditioning and explicit refinement mode

The Be II 7f investigation isolated a false rejection in the **1s core orbital**:
a candidate at -5.627319354223336 Ha had unit overlap and the requested zero
nodes, but reinverting the local operator at that energy produced a residual
of 6.8748e-6. Weak exchange places this energy almost at a local bound-state
pole. This residual representation becomes ill-conditioned and is undefined
in the zero-exchange limit.

The residual check now uses the equivalent identity

```
psi = G_sigma [K psi + (E - sigma) psi]
```

with the already prepared off-pole response. The tolerance remains 1e-7; no
exchange term is discarded. Diagnostics identify `shifted_green_identity`.
A known-answer zero-exchange regression ensures the local bound-state limit
is accepted without attempting an inverse at its pole. The archived rejected
roots are in `be_ii_7f_rejected_root_probe.json`.

Using implicit exchange throughout the unconstrained warm-refinement channels
then converged the Be II 7f prototype: corrected energy -13.652935979592748 Ha,
undamped update norm 7.2104e-9 against a 3.1623e-7 threshold. Its experimental
wrapper calls are recorded separately in `be_ii_7f_shifted_residual_warm_probe.json`;
the native recovery counter in that prototype is not the call count.

This is now available explicitly as
`ShootingHFSolver.solve(orbital_solver='nonlocal_resolvent')`. It uses the full
exchange-dependent solve for unconstrained kappa channels. Occupied same-kappa
channels retain their existing coupled bordered solver; unsupported legacy
constraint-mode combinations raise. `adaptive` remains the default. The
resolved numerical request and diagnostics record the selected solver and its
scope, and the campaign phase driver forwards this option.

**25 applied-tree tests passed**, covering spectral shifts, zero exchange,
exact-eigenvalue conditioning, wrong-state rejection, phase provenance,
convergence, warm starts, open-shell constraints and numerical boundaries.
Two-grid checks for Be II 7f, C I 5g and F II 7g have started in
`implicit_refinement_light_panel.json`. The ongoing 122-configuration campaign
retains its earlier frozen source and phase policy; its identity is not updated
in place. New checks must not be presented as results of that historical run.

## Explicit-refinement panel completed

All six native controls passed bootstrap and implicit warm refinement with the
full strict correction stack:

| Configuration | 3200-point corrected energy (Ha) | 6400-point corrected energy (Ha) |
|---|---:|---:|
| Be II 1s²7f | -13.652935979592748 | -13.652935979592904 |
| C I 1s²2s²2p5g | -37.322469515321096 | -37.32246951532167 |
| F II 1s²2s²2p³7g | -97.54880050021578 | -97.54880050021598 |

These are configuration energies, not line wavelengths. The largest two-grid
difference is below 6e-13 Ha; this is a radial-refinement diagnostic, not a
complete uncertainty budget or a claim of experimental agreement.

A separate **12-configuration / 24-attempt** recovery for the historically
failed species through Ar has started in
`native_shooting_light_implicit_recovery.sqlite`, using the explicitly recorded
`nonlocal_resolvent` warm solver. Its source is frozen at
`/tmp/atomx-implicit-recovery-20260929`, verified against all six successful
panel records. It includes the documented Mg/Al box enlargements. The older
122-configuration recovery continues under its own original frozen source;
results from the two policies must retain their separate identities.

## Completed light recovery and bootstrap follow-up

The implicit-refinement recovery completed all 24 attempts: 22 succeeded and
both Ar I 3p5 5g attempts failed the bootstrap orbital-update gate. The 11
complete pairs pass the record, correction-ledger and radial-label checks.
Their largest total-energy grid difference is below 8e-13 Ha. This does not
certify experimental accuracy or all sources of numerical error.

Ar I's bootstrap used adaptive frozen-source iteration and never entered the
implicit recovery: its undamped updates remained about 0.0016 after 1800
iterations. The recovery runner now accepts an explicit bootstrap
`--orbital-solver nonlocal_resolvent` and records that choice independently of
warm-stage controls. The audit verifies both requests. Existing historical
campaigns retain their original settings; no failed records were overwritten.
The new experiment is `native_shooting_argon_implicit_bootstrap.sqlite`, run
from a frozen copy under `/tmp/atomx-argon-implicit-bootstrap-20260929`.
Its outcome must be checked before accepting either state.

Five targeted phase/provenance and refinement-audit tests passed after this
change. The larger K/Ca recovery remains a separate older-source campaign.

The Ar I follow-up completed successfully at both grids (51.7 / 81.2 seconds).
Corrected totals are −527.9300108118331 and −527.9300108118307 Ha; the difference
is 2.3874e-12 Ha. Both staged convergence histories, identities and correction
ledgers passed the audit. This completes the numerical recovery of all 12
selected light-element configurations across the two explicitly separate
campaigns. The new result is not retroactively attributed to the old source.

`scripts/compare_refined_recovery_nist.py` writes a separate preview using
historical ground pairs and checks equal endpoint physical/correction models.
Five of the first 11 recovered states and the new Ar I state have complete
NIST evaluated manifolds; the remaining six are unscored. The Ne I 2p5 5f
centroid residual is still −14313.933 cm^-1 after recovery. Numerical convergence
does not remove the model discrepancy. Raw campaign identities remain in each
preview rather than merging predictions into the production campaign.

A frozen-source diagnostic is now testing K's 3p5 4s 26d and Ca's 3p5 4s2 10d
at both grids with the corrected shifted residual and implicit warm solve.
The older larger recovery failed these states with its previous residual check.
Results are written to `normalization_diagnostic/heavy_shifted_residual_probe.json`;
no result is accepted based on launch or completion alone.
