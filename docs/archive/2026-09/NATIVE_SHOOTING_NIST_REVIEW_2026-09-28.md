# Native shooting H–Ca benchmark and NIST review

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Scope and model

`benchmark_native_shooting_nist.py --max-charge -1` solves all 210 ground ion
stages from H through Ca at 1,600 and 3,200 radial points. Both endpoints of
each ionization energy come from native shooting. The empty bare nucleus has
zero electronic energy. This ground-stage benchmark does not cover every
excited level or observed line.

The model is statistical configuration-ensemble Dirac–Coulomb HF, isotope-aware
Fermi nuclei, DBSR static Breit including static retardation, automatic model
self energy, Uehling VP, first-order operator NMS/SMS and the applicable
one-electron nonrelativistic mass remainder. Strict correction support is
required. Each result records the actual operators, omissions, final engine,
calculation specification, diagnostics, runtime and source hashes.

The new artifact is `output/nist_spectra/native_shooting_first20_ensemble.json`.
Historical B-spline results remain unchanged. Full experimental accuracy is
not implied by SCF convergence or by agreement between radial engines.

## Open-shell pair-weight defect found and fixed

The initial driver supplied average occupations to shooting but used the full
statistical jj configuration ensemble for corrections. The public statistical
shooting branch had the same omission. Average occupations alone do not retain
the ensemble's joint pair occupancies.

For Ne IV (1s2 2s2 2p3), the initial uncorrected shooting energy was
-124.060543820689 Ha. With the ensemble passed to the SCF functional it becomes
-124.045410291614 Ha, agreeing with the ledger's uncorrected energy within
approximately 6e-11 Ha. This diagnosis isolates a model mismatch rather than
an integration error. The earlier diagnostic artifacts are retained with
`audit_status=superseded_model_mismatch` and must not enter accuracy scores.

An independent regression constructs the same statistical ensemble for
B-spline DHF and verifies the public shooting path against it at tight tolerance.

## Remaining correction differences

For S XIII (1s2 2s2), the matched base energies agree within 4e-10 Ha, but the
corrected total differs from the ledger by about -1.70e-5 Ha. Most of this is
self energy (-2.17e-5 Ha), partly offset by static Breit (+4.69e-6 Ha).
These components require separate radial/correction convergence checks. They
must not be attributed to electron correlation, which is absent in both runs.

## Fresh official NIST documentation review

NIST ionization energies are ground-to-ground thresholds. Its native stored
units are cm^-1; converted output has extra conversion uncertainty. Brackets
identify semi-empirical values, while parentheses identify theoretical values.
Hydrogenic/heliumlike entries can therefore be theory references rather than
independent experiments. We retain source kind and do not pool all categories
into an experimental score. [Official GSIE help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html)

The levels interface includes Landé factors, leading percentages, references,
uncertainties and isotope-specific queries. These fields carry state-matching
information beyond configuration strings. [Official levels help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)

Ritz wavelengths derive from level-energy differences, and the lines interface
can generate additional allowed transitions from available levels. Such rows
are distinct from observed-line records. [Official lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)

## Current archive coverage and limits

A fresh integrity audit passed against the retained source files:
296,998 standard lines, 119,394 standard levels, 3,835 isotope lines and 912
isotope levels. Eight standard level IDs occur only as line endpoints.
Evidence: `output/nist_spectra/nist_integrity_native_shooting_review.json`.
This is snapshot-integrity evidence, not a fresh exhaustive download of NIST.

Read-only database inspection found:

| Field | Populated | Total rows |
|---|---:|---:|
| Level ID | 119,394 | 119,394 |
| Level uncertainty | 62,147 | 119,394 |
| Landé factor | 18,479 | 119,394 |
| Leading percentages | 49,655 | 119,394 |
| Observed wavelength | 236,495 | 296,998 |
| Ritz wavelength | 242,792 | 296,998 |
| Lower endpoint Level ID | 272,500 | 296,998 |
| Upper endpoint Level ID | 272,487 | 296,998 |
| Ionization energy in cm^-1 | 5,847 | 6,019 |
| Total binding energy | 5,370 | 6,019 |

A blank field alone does not establish a failed import: the source may omit it.
The existing source audit records missing-source versus missing-import status,
including the unresolved plasma-diagnostics access gap. This review does not
claim that gap has been filled. Further direct-source checks must preserve
raw output, request parameters and hashes before changing archive coverage.

## Completed ground-stage run

All 420 requested runs completed (210 stages × two grids), yielding 210
ionization comparisons at the finer grid. Source kinds are 68 evaluated,
94 semi-empirical and 48 theoretical. Of these comparisons, 96 have unique
configuration ground levels and 114 retain a configuration-average versus
resolved-ground-term mismatch. The maximum grid change in an ionization
energy is 0.195317 cm^-1. This is a two-grid check, not a proven error bound.

For the 29 evaluated/unique-ground-level comparisons, RMS residual is
11,580.36 cm^-1 (about 1.44 eV). These DHF/model-correction results therefore
remain substantially short of experimental accuracy. Correlation-space
convergence, remaining correction discrepancies, excited states and line
comparisons remain required work.

Summary: `output/nist_spectra/native_shooting_first20_summary.json`.
The completed artifact retains the source hashes at execution time; subsequent
public-API fixes preserve equal/fractional averaging and its provenance. The
benchmark directly calls the numerical HF driver with the explicit statistical
ensemble, so it does not use that public dispatch branch.
