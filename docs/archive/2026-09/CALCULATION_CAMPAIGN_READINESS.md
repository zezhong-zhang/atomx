# All-stage calculation preparation

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Shooting update

Continuum and diffuse negative box roots are now implemented and tested in
the experimental shooting electron space. See
[the continuum progress report](SHOOTING_CONTINUUM_PROGRESS_2026-09-28.md)
for convergence results and the remaining heavy-ion model discrepancy.
That discrepancy has since been resolved for matched `1s2` controls; see
[the reference-matching report](SHOOTING_REFERENCE_MATCH_2026-09-28.md).
The release requirements below remain blocked pending broader validation
and integration. The original bound-only benchmark is historical evidence;
use `--benchmark output/correlation_benchmark/shooting_grid_post_continuum.json`
to regenerate the inventory against the current bound-space regression.

The campaign manifest is `output/calculation_campaign/readiness.json`.
Regenerate it from any working directory with:

```sh
python /data/code/atomx/scripts/prepare_calculation_campaign.py
```

Exit code **2** means the report was written but the production shooting
campaign is blocked. This command only reads the archive and existing DHF
ledger. Each database is read in a consistent transaction; the two snapshots
are independent. It creates no calculation jobs and does not reset workers.
The report is a local inventory, not a new online NIST completeness audit.

## Prepared scope

The current archive contains 7,021 element/charge entries, 296,998 standard
line records, 119,394 standard level records and 6,019 ionization records.
Isotope exports contain another 3,835 line records and 912 level records;
these are counted separately and are not assumed to be disjoint physical
transitions. There are 888 species with observed wavelengths and 1,079 with
line records of any kind. Empty holdings remain represented in the manifest.

The four historical level-page count discrepancies now match the combined
archive counts after term-order recovery. There are zero recorded parse
errors. Neither condition alone proves complete experimental coverage.

Existing DHF job status counts and running claims are recorded separately.
Configuration-average energies cannot be interpreted as individual NIST
fine-structure levels. A completed job is not automatically an accurate
state-resolved comparison.

## Release requirements

1. Construct and converge a shooting electron space including positive
   continuum contributions. Vary radial box, grid and electron-space size.
2. Validate open shells, neutrals, ions and diffuse states for the first 20
   elements, then Fe, Cu, Ag, Xe, W, Au, Hg and U. Compare matched nuclear
   models, spaces and Hamiltonians against B-spline/GRASP/FAC controls.
3. Integrate converged correlated orbitals and consistent energy provenance
   into the production Breit, QED and recoil pipeline. Record which terms
   enter orbital optimization and which enter subsequent corrections.
4. Match isotope, configuration, J/parity, reference energy and wavelength
   medium. Keep experimental observed wavelengths separate from NIST Ritz
   values and theoretical/estimated levels; retain uncertainties and flags.

These gates are deliberately blocked in this initial preparation tool.
Clearing them requires implementation and measured validation evidence;
there is no command-line override that turns small pilot success into
all-element readiness.

## Verified at preparation

The five finite-bound-space shooting pilots match all 233 current Python
source hashes and pass their stated solver residual checks. The independent
uranium rotation control did not converge, and is reported as such. These
pilots omit continuum space and do not establish NIST accuracy.

The readiness, correlated-grid and HF warm-start regression run passed
23 tests. Production solver defaults and the existing four active DHF
workers were left unchanged. No new all-table shooting calculation was
launched.
