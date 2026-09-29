# NIST parent-angular-momentum metadata repair

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

A fresh official Ca I energy-table download revealed a classification loss in
the archive. All 787 rows were present, but 43 `configuration_nist` labels had
lost numeric parent-J annotations such as `<3/2>`. The shared cell cleaner used
`<[^>]*>` as an HTML-tag pattern, which also matches these physical labels.
An energy/count-only completeness check cannot detect this loss.

## Source and audit

The original response and retrieval metadata are retained in:

- `output/nist_spectra/normalization_diagnostic/ca_i_live_20260929.csv`
- `ca_i_6d_live_verification.json` in the same directory, with the exact official
  ASD request URL, UTC retrieval timestamp and raw SHA-256.
- `ca_i_live_all_level_fields_audit.json`, preserving the before-repair mismatch.

The four Ca 4s6d levels give the exact weighted centroid 45035.7541 cm^-1.
Their source uncertainty cells are blank; no zero uncertainty is inferred.
The earlier failed web-tool request was followed by a successful direct HTTPS
retrieval, so this is fresh verification rather than only an archive check.

## Applied repair

`scripts/repair_nist_parentage.py` scanned cached official level-detail TSV
responses and restored **21,607 labels across 5,501 standard spectra**. A label
qualifies only when removing its numeric angle-bracket annotation reproduces
the old label exactly, the energy-order sequence has the same row count and
index, and both energy and J match exactly. Supplemental/term-order records and
unmatched source sequences are not silently aligned or modified.

Changes and provenance are recorded transactionally in
`nist_parentage_repairs`: original label, restored label, source path and hash,
and repair timestamp. Canonical shell configurations and energies are unchanged.
This table makes each update reversible and distinguishes restored source
metadata from calculations and derived classifications.

Verification:

- **21,607 ledger entries**; a repeated dry run finds **zero** remaining eligible
  repairs under this rule.
- All **787 Ca I rows** now match the live source as multisets on configuration,
  term, J, energy, uncertainty, reference, prefix and suffix.
- A before/after hash over every standard level's canonical configuration,
  term, J, energy, uncertainty, reference, ID and energy prefix/suffix is exactly
  unchanged. Counts remain 119,394 standard levels and 296,998 standard lines.
- Three migration tests pass: restoration/idempotence, rejection of an energy
  mismatch and rejection of a stale label.

The dry-run, applied manifest, post-audit and invariant reports are in the same
normalization-diagnostic directory under `nist_parentage_*` names. Original raw
source responses and historical campaign identities were not rewritten. The
current derived archive has a new metadata revision; its previous whole-file
hash must not be presented as its current hash.

## Parser prevention and remaining scope

The parser prevention fix is now applied in `reference_data/nist.py`: only
actual HTML tag names are stripped, while numeric annotations survive repeated
cleaning. A regression test covers HTML superscripts, escaped parent-J text and
idempotent cleaning. It was applied after the source-guarded B-spline control
finished; its recorded calculation identity remains unchanged.

This audit does not prove every isotope/supplemental classification or every
other NIST field is complete. It repairs the specifically verified loss in
cached standard energy-order records. Existing all-record counts do not replace
field-level checks.

## Extended field checks

- All **296,998 standard line rows** match the cached official lower/upper raw
  configuration labels, with no row-sequence mismatches. These contain 51,610
  endpoints with numeric parent-J annotations; their raw-label fields already
  preserved the information, so no line-label repair was needed.
- All **4,747 isotope rows** (lines and levels) match the complete parsed raw
  JSON of **31 hash-verified cached official responses**, with no differences.
- Re-running the historical first-20 comparison preserves the campaign identity,
  summary, all **35,802 observed-line comparison records**, all **7,211 centroid
  records** and rejected-state records exactly. The comparison output carries
  the new archive hash rather than reusing the old whole-file hash.

These are cached-source field checks except for the separately documented live
Ca I retrieval. They do not imply that all sources were downloaded again today.
Artifacts: `nist_line_parentage_audit.json`, `nist_isotope_raw_field_audit.json`,
`campaign_comparison_after_parentage.json`, and
`parentage_comparison_equivalence.json` in the diagnostic directory.
