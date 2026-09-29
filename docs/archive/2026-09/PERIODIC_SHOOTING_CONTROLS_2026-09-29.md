# Native shooting controls beyond the first 20 elements

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Scope and model

Started neutral and singly ionized ground-configuration calculations for Fe,
Xe, Hg and U, on 3200- and 6400-point native shooting grids. These are 16
attempts, not a full-element spectrum calculation. Both ionization endpoints
use the same selected isotope and nuclear mass: Fe-56, Xe-132, Hg-202, U-238.

The diagnostic uses adaptive shooting to tolerance 1e-12 (maximum 600 SCF
iterations), followed by nonlocal-resolvent refinement to 1e-13 (maximum 300).
Both stages must converge. Bare spinors are saved before the DBSR_HF Breit,
model self-energy, vacuum-polarization, operator normal-mass and specific-mass
corrections. The one-electron mass remainder is enabled; the separate leading
radiative-mass addition is not enabled in this diagnostic (the heavy controls
are all many-electron systems).

Source frozen at `/tmp/atomx-periodic-shooting-controls-v3-20260929`.
Output: `output/nist_spectra/periodic_shooting_controls_v3/comparison.json`.
Log: `/tmp/atomx-periodic-shooting-controls-v3.log`.
Bare HDF5 orbitals are under that output directory's `orbitals/` subdirectory.
The report records source hashes, iteration stages, correction provenance and
checkpoint hashes. Running this diagnostic does not modify the first20 queue.

The CLI now exposes refinement controls, source/archive paths and checkpoint
directory creation. Initial driver trials used an invalid solver-option name,
then omitted checkpoint-directory creation. Both were stopped; their failed
records remain under the original and v2 output directories. These were driver
errors, not physics convergence failures. The first20 production driver already
had the correct solver name and directory handling.

## End-to-end verification

Before the corrected heavy run, solved He I and He II at 1600 and 3200 points
using the same frozen driver. All four attempts converged and saved successfully.
Reloaded every checkpoint, checked SHA-256, convergence and exact bare energy,
and verified that the final iteration stage used `nonlocal_resolvent`.

He I ionization: 189080.3568539194 cm^-1, versus the cached NIST evaluated
198310.66637 cm^-1, residual -9230.3095160806 cm^-1. The grid change is only
-1.16e-8 cm^-1: grid refinement does not remove this mean-field discrepancy.
He II's reference is marked theoretical, and must not be counted as an
independent experimental validation. This helium diagnostic does not include
the separate leading radiative-mass addition used by the corrected first20 run.

Six existing generation and persistence tests passed. This smoke calculation
additionally checks the actual command-line solve/refine/save path.

## Interpretation limits

The diagnostic labels configuration-average versus ground-term differences;
Fe, Xe and U require care with this distinction. The archive's evaluated,
semiempirical and theoretical reference kinds are retained. Selected isotopes
are model inputs; matching an elemental NIST limit to that isotope is not
established by this run. The comparisons are preliminary diagnostics, not
experimental accuracy certificates. Heavy results must be inspected for
convergence, grid stability and state identity before any full-table launch.

## First completed heavy control: Fe

Fe I and Fe II converged at both 3200 and 6400 points. The 6400-point
configuration-average ionization is 55358.1550062869 cm^-1, versus NIST's
evaluated ground-term limit of 63737.704 cm^-1. Residual: -8379.5489937131
cm^-1; grid change: -1.4971e-6 cm^-1. This demonstrates numerical stability
for these states but does not separate term splitting from missing correlation.
The comparison remains labelled `configuration_average_vs_ground_term`.

## Matched configuration-centroid comparison

Added `scripts/compare_periodic_shooting_controls.py`. It computes the NIST
reference as `ground-term ionization + upper centroid - lower centroid`, using
complete evaluated manifolds from the shared NIST matching service. It rejects
mismatched ground configurations, isotope choices, nuclear physics or correction
requests, and leaves incomplete or nonevaluated references unscored. Both source
ionization records, centroid weights, endpoint correction differences and input
snapshot hash are retained. No combined centroid uncertainty is manufactured.

| Element | 6400-point ionization (cm^-1) | Matched NIST (cm^-1) | Residual (cm^-1) | Grid change (cm^-1) |
|---|---:|---:|---:|---:|
| Fe | 55358.155006 | unavailable | unscored | -1.50e-6 |
| Xe | 94073.102221 | 101346.123667 | -7273.021445 | -2.00e-7 |
| Hg | 69805.866589 | 84184.150000 | -14378.283411 | +1.60e-6 |

Fe I's assigned manifold has statistical weight 157 of expected 210; Fe II has
408 of 420. This is insufficient for a complete-centroid comparison, but is not
by itself evidence of omitted database rows: incomplete experimental assignments
and configuration mixing also need examination. Xe II's centroid lies
3512.336667 cm^-1 above its ground level; this offset is included explicitly.
Hg's two ground configurations each have a unique level.

The website `/comparison/native-shooting/periodic/` provides a residual plot,
grid filter, unscored-reference status and raw comparison inspection. It is
linked from the first20 campaign page. Six reference-identity and HTTP-route
tests pass, and viewer JavaScript syntax checks pass. No visual browser
inspection was performed. Refresh the report after further controls finish:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python scripts/compare_periodic_shooting_controls.py \
  --input output/nist_spectra/periodic_shooting_controls_v3/comparison.json \
  --output output/nist_spectra/periodic_shooting_controls_v3/matched_comparison.json
```

## Completed control panel and live Fe source check

All 16 heavy-control attempts finished normally. Reloaded all 16 native HDF5
checkpoints and verified their hashes, convergence flags, bare total energies,
orbital energies and final nonlocal-resolvent refinement controls. Evidence:
`periodic_shooting_controls_v3/checkpoint_integrity.json`.

U's 6400-point configuration-average ionization is 35965.6001335525 cm^-1;
its grid change is +3.99e-6 cm^-1. U I and U II have only one assigned level
in each ground-configuration group (weights 13/3640 and 10/364), so no matched
centroid residual is reported. The evaluated ground-term limit is 49958.4 cm^-1;
it is not the same observable as the configuration-average calculation.

Fresh read-only requests to the original NIST energy-level service returned
847 Fe I and 1028 Fe II rows. Every level ID and original source field matched
the archive exactly; the full TSV byte hashes also match the original archived
source hashes. There are zero missing, extra or changed energy-order rows in
these two spectra. The same ground-configuration weights, 157/210 and 408/420,
are therefore present in the live sources. This verifies these two imports,
not all NIST spectra or every display variant.

Raw fresh responses, requested URLs, UTC timestamp, SHA-256 hashes, compared
fields and the audit driver are preserved under
`normalization_diagnostic/fe_live_levels/` (`audit.json`, `audit_driver.py`).

NIST's [energy-level help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
explains that strong configuration interaction can distribute a level's
composition across configurations; the displayed leading configuration need
not constitute a pure configuration assignment. Thus a weight deficit under
label grouping is not a license to invent or renormalize missing levels.
Resolved correlated state matching remains necessary for these complex spectra.
