# Accuracy gates for atomx against NIST ASD

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The all-element DHF worker is paused. A completed calculation is not an
accuracy validation: the energy must describe the same isotope, state, and
observable as the reference, and every material approximation needs an
estimated numerical or physical error. Use the source-typed NIST archive and
retain its uncertainties and bibliographic identifiers.

## Current evidence

| Control | Result | What it establishes |
| --- | --- | --- |
| H–Ca shooting/B-spline branch | 20/20 neutral seeds reproduce the direct grid energy within 5.44×10⁻⁹ Ha | The large first-20 ground gaps are unlikely to be an orbital-root or projection error. |
| Bare Dirac–Coulomb DHF vs DBSR_HF | He and Be agree to the digits of the recorded external reference; Ne agrees within 4×10⁻⁸ Ha | The closed-shell mean-field Hamiltonian and radial solver agree with an independent code. |
| DBSR_HF static Breit port | He/Ne/Ar energies agree within 0.0032% relative on the same orbitals | Breit construction is improved; the H–Ca ionization RMS changes only 0.00074 eV. |
| First-20 matched IE | 19 evaluated, like-for-like pairs; 0.9798 eV RMS absolute residual | A large model gap remains after source and centroid matching. |
| He I correlation | MBPT(2) adds 1.00596 eV, removing 87.9% of its DHF gap | Differential electron correlation is a major missing term. |
| He I SD active spaces n≤2,3 | 17/98 CSFs give diagnostic ionizations 24.42245/24.53582 eV, leaving −0.16493/−0.05157 eV against NIST | The active-space change is 0.11337 eV, so the small space is not a high-accuracy result; neither point is a converged combined calculation. |
| H–Ca one-body recoil | GRASP radial operator changes neutral ionizations by at most 2.57e-7 eV from the virial estimate; denser-grid IE changes ≤1.51e-7 eV | This correction cannot close the ~1 eV DHF gap. See the [recoil audit](RECOIL_OPERATOR_AUDIT_2026-09-27.md). |
| Be I correlation | Default SD-MCDHF moves 8.04578 to 9.07018 eV; NIST is 9.32270 eV | Correlation improves another unique-level IE, but the active space leaves 0.25252 eV. |
| Mg I correlation and identity | Frozen-core SD-MCDHF moves 6.61769 to 7.53295 eV; NIST is 7.64624 eV; optimized `4s` overlaps the physical DHF `3s` by 0.941/0.991 in the two stages | Correlation closes 89% of the gap, but radial-overlap mapping is required and 0.11329 eV remains. |
| Stratified cross-table sample | 19/19 selected evaluated matched pairs have negative historical DHF residuals | The mean-field deficit appears across several electronic families and periods; heavy branches are not independently validated. |

NIST defines an ionization energy as removal from a specified ground level
to a specified level of the next ion. A DHF configuration average can only
be scored against a complete evaluated configuration centroid when the
ground levels are not unique. Observed wavelengths and Ritz wavelengths are
different reference classes. NIST also distinguishes evaluated,
semiempirical `[ ]`, and theoretical `( )` ionization entries. See
[NIST IE help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html)
and [NIST line help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html).

DHF includes antisymmetric exchange but no electron-pair correlation. Its
one-electron density, or one-body density matrix, does not by itself supply
the pair density needed for a correlated Coulomb energy. Exact density
functional theory defines a generally nonzero correlation functional for
multielectron atoms; it vanishes for a one-electron system. An approximate
density functional is a separate theory rung whose excitation and ionization
errors must be checked, rather than an automatic correction to DHF. For an
ionization energy the decisive term is the *difference* of neutral and ionic
correlation energies.

## Required path to a high-accuracy result

1. **Reference identity:** preserve the original NIST line/level/IE row,
   source type, uncertainty, isotope, medium, configuration, term, J and
   bibliographic reference. Do not score theoretical or incomplete NIST
   records as experimental measurements. Compare one predicted state with
   one observed state; mark ambiguous mixing or incomplete manifolds.
   Optimized CI orbital names are not spectroscopic identities: match their
   radial overlap and node character to independently labeled reference
   orbitals before interpreting a dominant CSF. A magnesium SD probe
   yielded a lowest state labeled 4s while that optimized orbital overlaps
   the DHF 3s by 0.991; a label-only gate would misdiagnose the state.
   Dominant-CSF percentages can change under rotations among active
   orbitals, so a high-accuracy assignment should also inspect the
   one-body density and its projection onto independently identified
   spectroscopic orbitals.
2. **Dirac–Coulomb numerical limit:** verify occupied radial nodes and
   state ordering from different starts; tighten SCF, radial box, spline
   order/spacing, positive-energy projection, partial-wave cutoff and
   virtual continuum until the observable is stable. Compare selected cases
   with DBSR_HF and GRASP/FAC using the same Hamiltonian, nuclear model and
   state definition. Record each independent spread, not just `converged`.
3. **Electron correlation:** for each first-20 neutral and ion ground term,
   converge a J-resolved multiconfiguration or many-body expansion. Increase
   core/valence excitations and virtual shells in a documented sequence;
   test size consistency and ground-state identity. Validate against
   independent calculations in one-electron, two-electron, alkali and
   closed-shell controls. The current default SD active space is a probe,
   not a converged model. In particular, `Atom(correlation="sd")` freezes
   1s/2s/2p excitations for Z>10, so its magnesium and heavier results
   cannot claim a converged core–valence correlation energy.
4. **Physical corrections on the correlated states:** add the two-electron
   Breit interaction with retardation and assess frequency dependence;
   leading one-body and two-body recoil for the selected isotope; finite
   nuclear size and its isotope uncertainty; screened self-energy and
   vacuum polarization with an explicit approximation label. A DHF-only
   scalar shift cannot replace state-specific corrections or correlation.
5. **Uncertainty accounting:** keep numerical, active-space, recoil,
   Breit, QED, nuclear and reference uncertainties separately. Vary each
   approximation enough to bound its effect on the *transition or
   ionization difference*, not merely on each absolute total energy.
   Treat systematic model error explicitly; do not shrink it by assuming
   independent Gaussian errors without evidence.
6. **Promotion gate:** publish a theory accuracy score only for rows with
   matched state identity and validated convergence. Require numerical
   uncertainty below the intended comparison scale and a residual
   consistent with the combined theory and NIST uncertainty across an
   independent set of H–Ca and across-table controls. The full-table
   worker resumes after this gate, not after a code path merely runs.

Parts-per-billion agreement is achieved for selected simple systems and
observables with specialized few-body/QED calculations. It is a useful
long-term control, but the present DHF, static Breit, model QED, and
leading recoil stack does not justify that claim for a full many-electron
periodic table. NIST itself provides different uncertainty and provenance
for each record, so the achievable comparison target is record-specific.

## Subsequent QED audit

See [MCDHF QED coherence audit](MCDHF_QED_COHERENCE_AUDIT_2026-09-27.md).
The coherent-VP helium rerun supersedes the previous combined-correction
diagnostic: n<=3 gives 24.5365140473 eV, residual -0.0508749637 eV.
A positive-kappa self-energy label bug was also fixed; historical MCDHF
cache rows require revalidation. These fixes do not establish active-space
convergence or remove the full-table accuracy gate.

## Corrected generation prepared, not started

`output/nist_spectra/full_dhf_corrected.sqlite` has been prepared directly
from the NIST archive with 15,656 configuration jobs and 236,495 observed
line records. All configuration jobs are pending. The ledger records
`dbsr_static_breit_model_qed_grasp_nms_sms_v1` and an explicit
`paused_for_first20_physics_audit` hold. Preparing this queue does not
establish calculability or accuracy of every row.

The separate historical ledger remains at 14,535 done / 2,052 pending,
with its original scalar Breit and virial-recoil metadata. No historical
energy has been replaced. The interactive page distinguishes the prepared
corrected queue from the historical calculated results.

The repaired helium initialization passes an independent saved-state
check: electron count 2.0000000000000004, maximum same-kappa overlap error
2.22e-15, maximum nuclear negative-energy weight 1.13e-29, and exact
reproduction of the saved DC energy. All sixteen natural occupations now
exceed 3e-6. This validates the initialization, not the still-running
orbital refinement or a converged spectrum.

## Audited first-20 ground rows stored

`generate_audited_first20_ground.py` has populated 39 neutral/first-ion
SCF configurations (the bare H nucleus is a zero-energy endpoint) in the
corrected ledger. Every fresh shooting-seeded solve enables static DBSR
Breit, model QED, and one- plus two-body Breit recoil, and must reproduce
the independent first-20 audit before storage. Source hashes and per-job
energy differences are retained in `first20_corrected_ground_generation.jsonl`.

The corrected ledger now contains 39 done / 15,617 pending. Its general
full-table worker hold remains set. The historical ledger is unchanged.
These ground energies do not establish the accuracy of excited-state lines
or eliminate DHF's correlation error. The audit promotion rejects wrong
mass, missing components, energy mismatches, non-ground jobs, completed
jobs, and requests outside Z=1..20 neutral/first-ion scope.
