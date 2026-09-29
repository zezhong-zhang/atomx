# NIST spectrum atlas

The atlas is a local archive and interactive viewer for NIST line, level,
isotope and ionization records. Coverage is determined by stored source manifests
and audit results; an element's presence in the periodic table does not imply
that every ionization stage has experimental data.

## Build and serve

From the repository root, using an environment with Atomx installed:

```bash
python scripts/build_nist_spectra.py --help
python scripts/build_nist_spectra.py --out output/nist_spectra
python scripts/view_nist_spectra.py --out output/nist_spectra --port 8765
```

Open `http://127.0.0.1:8765`. The build downloads source records when necessary.
Use `--offline` to reuse available sources. The viewer reads the local database
and serves it on localhost by default.

The output contains `spectra.sqlite`, source manifests, coverage reports and raw
source files. Keep raw responses and hashes with the database for reproducibility.
A base build and supplementary imports are distinct operations; do not infer
supplementary coverage from the base build alone.

## Supplementary imports and checks

The supported scripts expose their controls through `--help`:

- `import_nist_ionization.py`: ground-state and ionization-energy records.
- `import_nist_level_details.py`, `import_nist_line_details.py`: extended source fields.
- `import_nist_term_order_levels.py`, `import_nist_line_only_levels.py`: source records missing from the base energy-ordered level table.
- `import_nist_isotope_records.py`, `import_nist_isotopes.py`: isotope-specific records and masses.
- `import_nist_weighted_a.py`: weighted transition probabilities.
- `audit_nist_archive_integrity.py`: archive integrity.
- `audit_nist_level_weights.py`, `audit_nist_line_wavenumbers.py`: source consistency checks.
- `verify_nist_live_samples.py`: selected live-source comparisons.

Inspect `parse_errors`, source-file errors and coverage reports after importing.
A missing energy or angular assignment remains missing; a line endpoint used as
a level placeholder does not become a measured level.

## Interface

The viewer provides a periodic-table selector, charge-state coverage, line
spectra, per-ion details and paged records. Supported data endpoints include:

- `/api/ion?z=26&charge=0`
- `/api/records?kind=lines&z=26&charge=0&page=1`
- `/api/export?kind=lines&z=26&charge=0`

Observed and Ritz wavelengths remain distinct. Theory CSV uploads are matched
by state information, not nearest wavelength. See [comparison conventions](THEORY_NIST_COMPARISON.md).

## Historical coverage

Dated counts, live verification samples and previous campaign output are retained
in the [archive](archive/README.md). They describe those source snapshots and do
not certify the current local database. Retired campaign tools and routes are
not current usage instructions.
