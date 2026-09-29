# Corrected one-electron production-path audit

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

This isolated diagnostic uses the corrected production solver on one-electron
ions for Z=1–20. It does not release the full-table worker or promote energies
to its ledger.

## Selection and execution

`first20_hydrogenic_corrected_audit.sqlite` was created using SQLite backup
from `full_dhf_corrected.sqlite`. Configuration jobs were restricted to
Z≤20, Z−charge=1 and configurations 1s1, 2s1, 2p1, 3s1, 3p1, 3d1. Line jobs
with known endpoints were restricted to that set; records with null
classifications were retained for coverage accounting. This preserves the original NIST
row indices, isotope choices and model metadata. Only configurations already
present in the prepared queue were selected; it is not a complete n≤3 spectrum.

The diagnostic ledger's run status is `isolated_first20_one_electron_audit`;
the parent production ledger retains `paused_for_first20_physics_audit`.
Two completed ground endpoints were inherited; 44 pending configurations
were calculated with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/generate_full_dhf_nist.py --worker-only \
  --output output/nist_spectra/first20_hydrogenic_corrected_audit.sqlite \
  --max-z 20 --max-calculations 120 --refresh-site-every 0
```

## Results

- 46 completed configurations, with at least one for every Z=1–20.
- Electron–electron energy, Breit two-electron energy and specific mass shift
  are exactly zero in the stored components for all 46 configurations.
- 181 retained NIST line records: 1 unique-level comparison, 50 configuration
  averages, 117 missing-classification records, 3 unresolved configuration
  averages and 10 unresolved hydrogenic n-manifold records.
- The unique-level H 1s–2s record gives 121.56734901695 nm versus the archived
  121.56731231302 nm, a photon-energy residual of −3.01923e−5 percent.
  This is a transition-energy comparison, not a claim of an allowed one-photon
  electric-dipole line or a complete QED prediction.

The p/d shell predictions average different j levels and must not be scored
against an individual fine-structure line. Missing classifications remain
visible in the audit counts. The worker's numerical sanity checks are not
an uncertainty bound; radial-grid and same-Hamiltonian checks remain needed.
In particular, the finite-Fermi-nucleus one-body energy differs from the
point-nucleus Dirac formula by nuclear-size contributions, so their raw
difference cannot be labeled a solver error.

The machine-readable summary is
`output/nist_spectra/first20_hydrogenic_corrected_audit.json`. The website's
corrected-line panel has a separate calculation-set selector for this audit.
Historical, corrected-production and diagnostic rows remain separate.

## Provenance fix

The line collector previously hardcoded the historical `DHF+B+QED+NMS`
label. It now identifies the recorded physics version, labels the corrected
stack `DHF+DBSR Breit+QED+NMS+SMS`, and explicitly labels absent/unrecognized
metadata as unverified. Regression coverage checks all three cases without
changing line predictions or promoting configuration averages into scores.

## Same-observable radial audit and recoil diagnosis

`scripts/audit_first20_one_electron_grid.py` reevaluated all 46 configurations
with a larger 100 a0 box and denser inner/outer mesh. The maximum change in
the corrected total energy was 2.5531e−8 eV (Al XIII 3p configuration average).
The raw results are `first20_one_electron_grid.jsonl`.

Separate point-nucleus calculations use explicit `DiracFock` with
`NuclearModel.POINT`, ensuring the same statistical jj occupations as the
production calculation. Their reference is the 2j+1 weighted analytic Dirac
energy. An initial audit accidentally used the convenience solver's filled
lower-j state; those rows were rejected and preserved in
`first20_one_electron_grid_rejected_state_mismatch.jsonl`. They are not used
in the website or summaries. Explicit method objects also own their nuclear
model; setting the Atom's nuclear option alone does not override that method.

Maximum point-nucleus Dirac disagreement is 8.2726e−7 eV for H 3d. A further
point-grid/tolerance check gave 8.0551e−7 eV for that state, so it is not yet
resolved by radial refinement. This remains an open numerical/state audit
item. No claim of a universal precision bound follows from this grid check.
For Ca XX 1s, finite nuclear size shifts the DC energy by 0.01602534 eV,
while the matched point-nucleus calculation differs from analytic Dirac by
9.66e−10 eV. Nuclear-size shifts must not be mislabeled as solver error.

A subsequent isolated inner-grid probe resolves the H 3d energy discrepancy:
with the same outer grid, inner steps 1e−4 and 1e−6 bohr give analytic errors
−2.4480e−11 and −2.4475e−11 eV; the excessively small 1e−8-bohr step repeats
the 8.2726e−7 eV error. Evidence is in
`hydrogen_point_inner_grid_probe.json`. This indicates numerical sensitivity
from the extreme inner mesh, rather than missing physical correlation.
The precise conditioning mechanism and an automatic mesh-selection policy
still need investigation. This point-nucleus audit does not change the
production Fermi-nucleus energies or justify using the coarser inner mesh
for derivative-sensitive recoil observables at high Z.

`scripts/audit_hydrogen_recoil_remainder.py` isolates the exact
nonrelativistic mass dependence without fitting experiment. With nuclear
mass M in electron masses and E0=−Z²/(2n²), the reduced-mass energy is
E0 M/(M+1). Subtracting E0 and its already included first-order recoil
leaves E0/[M(M+1)]. This is only the nonrelativistic point-Coulomb remainder;
it is not full higher-order recoil. See the finite-mass discussion in
[CODATA](https://physics.nist.gov/cuu/pdf/CODATA_RMP2016.pdf) and the distinction
between relativistic orders and first order in mass ratio in
[Shabaev's recoil review](https://arxiv.org/abs/physics/0009038).

For archived H 1s–2s row 77, this adds 3.0250175e−6 eV to the transition
energy. The residual changes from −3.0792514e−6 to −5.4233839e−8 eV, removing
98.24% of the discrepancy. The diagnostic wavelength is 121.56731295947 nm
versus 121.56731231302 nm archived. Model QED, relativistic/radiative recoil
and other limitations remain. No production energy has been changed.

## Implemented remainder and fresh calculations

The remainder is now available through
`CorrectionStack(include_hydrogenic_mass_remainder=True, nuclear_mass_me=M)`
or the corresponding `DiracFock` option. It requires operator NMS, retains
the original NMS component, and adds a separately named energy component
and provenance. It returns zero for multielectron DHF and rejects MCDHF
correlation-orbital labels rather than silently using them as principal n.
It is opt-in; it does not claim to complete relativistic or radiative recoil.

`scripts/benchmark_hydrogenic_mass_remainder.py` freshly recomputed all
46 configurations, validating each against its independently stored baseline
plus the derived remainder within 1e−10 Ha. The results and per-state identity
checks are in `first20_hydrogenic_mass_remainder.sqlite` and the same-named
JSON summary. The model has a distinct v2 physics version; the v1 audit and
held production ledger retain their original energies. The ordinary v1
worker rejects this v2 ledger, preventing accidental model mixing.

The website calculation selector now offers both first-order recoil and
mass-remainder-included audit results. Tests cover exact two-body
nonrelativistic mass scaling, invalid masses, the first-order prerequisite,
multielectron exclusion, MCDHF rejection, and actual DHF transition shifts.
