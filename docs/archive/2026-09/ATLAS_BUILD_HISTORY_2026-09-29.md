# NIST atomic and ionic spectrum atlas

> Historical snapshot, retained as evidence. Current usage is documented in
> [the main documentation](../../README.md). Retired commands are not supported.

For calculated versus NIST line positions and the accuracy map across the periodic table, see [theory comparison](../../THEORY_NIST_COMPARISON.md).

The atlas imports the indexed [Cloudy/Nublado copy of NIST ASD v5.12](https://data.nublado.org/nist/asd_v5.12/) into a local SQLite database. The builder inventories every available line and level file, validates the ion name and isoelectronic sequence in each response, records a SHA-256 hash, parses all table rows, and checks the parsed count against the count printed by NIST. It also compares every ion with [NIST's official line holdings grid](https://physics.nist.gov/cgi-bin/ASD/lines_pt.pl). It records all 7,021 bound atomic charge states for Z = 1–118, including explicit `not_in_archive` statuses. The mirror has no files for many charge states; a blank in the coverage plot is not a zero-line prediction.

NIST's [ASD contents page](https://www.nist.gov/pml/atomic-spectra-database-contents) describes the scope of its critically evaluated line and level data. The [version history](https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml) identifies v5.12 as the November 2024 release. The [lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html) explains observed and Ritz wavelengths and radiative transition probabilities.

## Build and inspect

```bash
.venv/bin/python scripts/build_nist_spectra.py
.venv/bin/python scripts/import_nist_line_details.py
.venv/bin/python scripts/import_nist_weighted_a.py
.venv/bin/python scripts/import_nist_level_details.py
.venv/bin/python scripts/import_nist_term_order_levels.py
.venv/bin/python scripts/import_nist_line_only_levels.py
.venv/bin/python scripts/audit_nist_isotope_holdings.py --kind lines
.venv/bin/python scripts/audit_nist_isotope_holdings.py --kind levels
.venv/bin/python scripts/audit_nist_level_weights.py
.venv/bin/python scripts/audit_nist_line_wavenumbers.py
.venv/bin/python scripts/audit_nist_diagnostics.py
.venv/bin/python scripts/audit_nist_partition.py
.venv/bin/python scripts/import_nist_isotope_records.py
.venv/bin/python scripts/import_nist_isotopes.py
.venv/bin/python scripts/generate_full_dhf_nist.py
.venv/bin/python scripts/verify_nist_live_samples.py
.venv/bin/python scripts/view_nist_spectra.py --render-overview
.venv/bin/python scripts/view_nist_spectra.py --render-gallery
.venv/bin/python scripts/compare_theory_nist.py
.venv/bin/python scripts/view_nist_spectra.py --port 8765
```

Open <http://127.0.0.1:8765/> for the interactive explorer. The first command downloads the archive once; use `--offline` to verify and rebuild from the cache. The output is in `output/nist_spectra/`:

Run `.venv/bin/python scripts/compare_theory_nist.py` before opening the site to enable its **Theory comparison** page at <http://127.0.0.1:8765/comparison>. The site has a clickable periodic table, charge-state coverage map, line-position spectra, ground-state and ionization summaries, and paged access to every imported line and level field. The comparison page links each calculated ion back to its complete NIST records, including the DHF and MCDHF results where validated state pairs exist.

| File | Contents |
| --- | --- |
| `spectra.sqlite` | Searchable source files, species coverage, line and level rows, and parser issues |
| `official_ionization_energies.html` | Unmodified direct NIST ground-state and ionization-energy output for H–Ds |
| `official_line_details/`, `official_weighted_a/`, `official_level_details/` | Cached direct NIST exports of optional line and level fields, verified against archive row order |
| `official_isotope_*_holdings.json`, `official_isotope_records/` | Direct NIST per-element isotope inventories and original isotope-specific line and level exports |
| `theory_comparison/` | Calculated versus NIST line positions, interactive accuracy map, CSV, JSON and static figures |
| `nist_isotopes_source.html`, `isotopes.json` | Saved NIST masses for every listed isotope and documented reference-isotope choices |
| `full_dhf.sqlite`, `full_dhf_run.log` | Resumable mass-aware DHF+Breit+QED configuration and observed-line jobs |
| `coverage.json`, `coverage.csv` | Counts, statuses and the archive-completeness gate |
| `source_manifest.json` | Source URL, byte size and SHA-256 for each indexed file |
| `official_lines_holdings.html`, `official_holdings_comparison.csv` | Saved NIST holdings page and per-ion count comparison |
| `live_checks/`, `live_sample_verification.json` | Direct NIST query responses and field-by-field comparison for H I, He I, Li I, C V, Ne VIII, Na I, Ar VIII, Fe XVII and U II |
| `coverage_overview.png` | Static element-by-charge coverage map |
| `spectra_gallery.png` | Six representative NIST line fields from light and heavy ions |
| `raw/` | Unmodified downloaded HTML responses; the small committed benchmark panel is reused |

For example, select a species directly with SQLite:

```sql
SELECT wavelength_nm, wavelength_source, medium, aki_s1
FROM lines WHERE z = 26 AND charge = 16
ORDER BY wavelength_nm;
```

`wavelength_nm` uses the observed value when one exists, otherwise the Ritz value. The `medium` column preserves NIST's air/vacuum declaration; converting or mixing air and vacuum wavelengths requires care. Rows with no usable wavelength remain in `lines` with `wavelength_nm = NULL`. A missing `aki_s1` stays NULL. The viewer's summed-A curve locates concentrations of strong transition probabilities; it is **not an emission spectrum or a physical intensity**. A plasma spectrum requires populations, temperature/density, line profiles, and other channels.

The extended `lines` table also stores observed and Ritz wavelength uncertainties, A-value accuracy codes, lower and upper level energies, complete classifications, references, and the original HTML table row number. A direct NIST supplementary export now supplies relative intensity, oscillator strength, line strength, log(gf), wavenumber, observed-minus-Ritz, statistical weights, and level IDs; all **1,079** line spectra and **296,998** rows passed source identity checks. Relative intensity is a qualitative NIST field, not an emitted spectral power. The `levels` table stores configuration, term, J, energy, uncertainty, flags, references, original printed energy, energy provenance, Landé factor text, leading percentages, and source row. Direct supplementary level IDs, statistical weights, and splitting passed the same source check for all **6,019** standard level spectra. `ion_summary` materializes each ion's lowest listed level and first `Limit` threshold. The ionization energy in eV is converted from the first limit's wavenumber above the listed ground level; all limits remain in `levels`, including ions with multiple thresholds. A missing threshold is left NULL. The original archived HTML and supplementary direct exports are linked from the site for source inspection.

The line table also shows a derived source-field check when NIST's stored
wavenumber differs by more than 0.1% from the inverse observed vacuum
wavelength or by over ten times the listed wavelength uncertainty. Both
original values remain untouched. The [source audit](NIST_SOURCE_AUDIT_2026-09-26.md)
lists the 27 flagged rows and documents a direct NIST recheck of Sr XI.

For observed lines with an empty configuration in the original line table, the
DHF ledger can use the exact NIST lower/upper level IDs to fetch the endpoint
configuration from the level table of the same ion. This recovers **25,878**
line rows while preserving the raw NIST line fields. The ledger records
`classification_source='nist_level_id'`; run
`.venv/bin/python scripts/generate_full_dhf_nist.py --backfill-level-ids` to
apply the bridge to an existing calculation ledger. The comparison website
labels derived endpoints, and the source audit gives the remaining coverage
limits.

The builder also imports NIST's separate [ground-state and ionization-energy table](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html) directly. The query requests H through Og; NIST currently returns **6,019 rows** through Ds (Z=110), with **5,847 numeric ionization energies**; missing values remain NULL. The `nist_ionization` table preserves ground shells, ground and ionized configurations/levels, uncertainty, reference, the original parenthesized or bracketed energy notation, and the source row. Supplementary direct NIST queries retain the stored cm⁻¹ ionization value for every available row and **5,370** derived total binding-energy rows, including their reference classifications and uncertainties. Square brackets mean semiempirical and parentheses mean theoretical, per [NIST's GSIE help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html); these are not labeled as experimental in the theory accuracy map. The atlas displays the official energy next to the value inferred from the level `Limit` row so rounding or source differences remain visible. The official table does not publish Z=111–118 rows in this query; those cells remain explicitly unavailable. The three unmodified responses are saved as `official_ionization_energies.html`, `official_ionization_cm1.html`, and `official_binding_cm1.html` and can be reimported with `.venv/bin/python scripts/import_nist_ionization.py --offline`.

The site exposes paged `/api/records?kind=lines|levels&z=26&charge=0&page=1`, full per-ion `/api/export?...` CSV for lines, levels, or ionization, and `/api/ion?z=26&charge=0`. A page of 100 rows is displayed at a time, but every archived row is accessible and the full CSV export is untruncated. The isotope selector exposes the separate NIST mass-specific spectra, with paged raw fields, original source files, and complete CSV downloads. NIST line, level, isotope, and ionization reference codes link to the source bibliography when the code identifies a bibliographic record. All standard-ion weighted-A fields are imported from row-verified NIST exports. The theory page preserves both NIST ground-term and complete configuration-centroid ionization differences, with a selectable periodic accuracy map for each reference. A separate excited-configuration centroid map scores only complete evaluated NIST J manifolds against converged DHF configuration energies; the complete CSV retains all skipped rows and reasons. Centroid weights are derived from `2J+1` while the original NIST `g` remains visible; the [source audit](NIST_SOURCE_AUDIT_2026-09-26.md) documents the disagreements. The ion page also computes a temperature-dependent partition sum from imported NIST levels through `/api/partition?z=12&charge=0&temperature_ev=1`, labels missing-weight results as lower bounds, and links the live NIST output. Plasma diagnostics remain a documented source-access gap.

For a direct comparison of your own calculation, select an ion on the atlas page and upload a CSV with `lower_configuration,upper_configuration,lower_j,upper_j,wavelength_nm`. Optional `lower_term,upper_term,z,charge` columns further constrain matching. For lines with empty printed configurations, a CSV may instead contain `lower_level_id,upper_level_id,wavelength_nm`; these are exact NIST IDs from the per-ion line table. If the same ID pair appears on multiple NIST rows, add `nist_row_index` to identify the intended row explicitly. The local site accepts up to 10,000 calculated rows, matches NIST **vacuum** lines by state identity, and shows paired wavelengths, NIST uncertainty, energy error, RMS score, and reasons for unmatched rows. It never picks a row by nearest wavelength. The default NIST basis is observed wavelengths; Ritz is selectable and remains labeled as level-derived. The upload is handled in memory and is not added to the NIST reference database.

The builder exits nonzero if a listed file is missing, fails validation, or the line catalog disagrees with NIST's per-ion holdings grid. Level-page count discrepancies remain flagged but do not fail the line-catalog build. See `parse_errors` in SQLite and `download_errors` in `coverage.json` for exact gaps. `line_holdings_verified: true` means that every indexed line file parses to its printed row count and every cell in NIST's live holdings grid matches the catalog. Optional generated Ritz transitions are outside this grid; the atlas does not invent lines for charge states with no NIST line file.

## Verification on 2026-09-26

All **1,079 line files** and **6,019 level files** listed by the mirror were downloaded or reused from the committed NIST panel and passed file identity checks. All line tables parsed with their printed row counts: **296,998 rows**, including **294,355 usable wavelengths** and **128,558 A values**. The mirror's printed line totals sum to 296,998 as well. NIST's official holdings grid has **6,105 ion cells**, whose counts also sum to **296,998**; all 6,105 individual cells match the catalog. Direct NIST queries for **13 ions**, including Mg I/II and K I/II, match every compared wavelength, uncertainty, A value, accuracy code, energy, classification, and reference field in the archive; the exact queries and response hashes are in `live_sample_verification.json`.

The energy-ordered base catalog contained **119,273 level rows**. Four pages print a count larger than that ordering displays. A direct term-ordered query supplies their omitted Level IDs; the raw energy-order discrepancy remains recorded for provenance:

| Ion | Printed level count | Table rows |
| --- | ---: | ---: |
| [Br IV](https://data.nublado.org/nist/asd_v5.12/levels/asd.035003) | 43 | 41 |
| [Nd II](https://data.nublado.org/nist/asd_v5.12/levels/asd.060001) | 840 | 752 |
| [Au I](https://data.nublado.org/nist/asd_v5.12/levels/asd.079000) | 75 | 62 |
| [Au II](https://data.nublado.org/nist/asd_v5.12/levels/asd.079001) | 48 | 38 |

The [live NIST holdings page](https://physics.nist.gov/cgi-bin/ASD/lines_pt.pl) displays a **300,833-line headline**. The 6,105 standard ion cells contain **296,998** rows, and NIST's per-element isotope holdings contain the remaining **3,835**. All 14 isotope line spectra and **17 isotope level spectra** were imported directly with their source hashes and holdings counts: **3,835 line rows and 912 level rows**. Of the isotope lines, **312** have A values, exactly reconciling NIST's **128,870** transition-probability headline with the standard archive's **128,558**. The term-ordered levels add **113** unique IDs; the NIST line table supplies the final **eight** endpoint-only IDs, whose energy and J remain null. Thus **119,394 standard + 912 isotope = 120,306** level records, exactly NIST's per-element holdings count. The [source audit](NIST_SOURCE_AUDIT_2026-09-26.md) explains the provenance and limits.

After a base rebuild, rerun the two level reconciliation commands above; `import_nist_term_order_levels.py --offline` reuses its cached direct NIST exports. The current coverage report has zero missing level records against NIST's holdings. Eight line-endpoint-only placeholders have no published energy or J, and plasma diagnostics remain a separate source-access gap.
When supplementing an already running DHF ledger, run
`.venv/bin/python scripts/refresh_dhf_state_counts.py` to refresh its
configuration-level counts and line status gates against the augmented level
table; a newly prepared ledger calculates those counts from the current source.
