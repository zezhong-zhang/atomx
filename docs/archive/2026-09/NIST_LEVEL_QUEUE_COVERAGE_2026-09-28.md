# Standard NIST level coverage

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The observed-line queue did not cover every configuration present only in
the standard NIST level tables. A census found 40,402 level rows mapping to
9,237 configurations absent from the queue (3,988 in elements 1–20).

`scripts/prepare_nist_level_jobs.py` now inventories all 119,394 archived
standard levels in `level_jobs`, preserving source row identities, level IDs,
reference classes, uncertainties, and explicit ground-reference dependencies.
It inserted the 9,237 missing configurations into the existing v3 queue with
the same selected masses. Existing completed/running jobs were not modified.
The addition is idempotent and recorded in `level_job_additions`.

Classification counts:

- 109,118 levels with configurations, numeric energies and evaluated zero references.
- 8,023 rows with unresolved configurations.
- 844 numeric classified rows without an evaluated zero reference.
- 1,409 other classified rows with missing numeric energy.

Theoretical and semiempirical rows remain in the inventory but do not become
experimental scores. The 12 additional missing numeric rows have unresolved
configurations and belong to that category rather than being counted twice.

`scripts/compare_all_nist_levels.py` exports all 119,394 rows to
`output/nist_spectra/theory_comparison/all_levels_dhf_nist.csv`. It separates
unique-level scores, unscored configuration-average residuals, unresolved states
within the same configuration, offset references and incomplete calculations.
All comparisons require a common isotope and mass. At the initial snapshot,
67 rows met the unique-level scoring rules; this is not an accuracy claim.

Validation checks queue idempotence, preservation of existing energies, complete
source-row retention, treatment of missing energies/configurations, and exclusion
of theoretical values from experimental scores. The extended test passed.

The historical 3,481 first-20 configuration completion count applied to the
observed-line/ground queue. The larger level-inclusive queue is unfinished;
the old count must not be presented as completion of all NIST level records.
The interactive comparison now includes a full-level section linked to the
periodic-table selection. `/api/theory-levels` filters by element, optional charge
and status, with validated pagination. Its plot contains only unique-level
evaluated comparisons; error bars represent NIST uncertainty alone. Every row
is accessible through pagination, and `/comparison/all-levels.csv` downloads the
complete export. The site builder refreshes that export atomically.

The viewer on port 8765 was restarted with the new endpoint. Live endpoint checks
covered H, Ca II and an element with no listed levels; the CSV returned HTTP 200.
Twelve level-queue, comparison and API tests passed, and JavaScript syntax checked.
