# NIST ASD source audit (26 September 2026)

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## What is in the current archive

The archived standard-ion NIST ASD v5.12 export contains 296,998 line rows from 1,079
spectra and 119,394 level records from 6,019 spectra after reconciling both NIST
level orderings and line-only Level IDs. The energy-ordered base export supplied
119,273 rows; the term-ordered direct export supplied 113 distinct missing Level
IDs, and the line table supplied eight endpoint-only IDs without level energies.
Every archived
line file parses, and all 6,105 per-ion counts in NIST's live
[lines holdings grid](https://physics.nist.gov/cgi-bin/ASD/lines_pt.pl) match
the archive. The 7,021 species inventory includes ion stages with no NIST
line or level file; an absent file is not counted as zero measured lines.
A fresh 27 September 2026 request to the official holdings grid again found
all **6,105** ion cells matched, totaling **296,998** standard line rows,
with no missing or discrepant catalog cells.
The offline integrity command
`scripts/audit_nist_archive_integrity.py` rehashed all **15,310** retained
standard, supplemental, term-order, weighted-A, and isotope source files;
every stored SHA-256 matched. It also checked each standard ion's archived
line and level count against the source declaration, accounting separately
for the eight line-only level IDs, and checked all 31 isotope spectra against
their holdings counts. The audit reconciles the listed ion spectra with
base and supplemental source-file inventories and checks the isotope holdings
manifests. Its report is
`output/nist_spectra/nist_archive_integrity.json` and currently has zero
issues. This verifies the retained source snapshot; it does not imply that
NIST's inaccessible plasma-diagnostics responses were imported.
Direct line queries for 13 representative spectra, including Mg I/II and
K I/II, were also compared field by field with the retained source exports.
All 13 matched in row count and in the 17 checked fields; the reproducible
result is `output/nist_spectra/live_sample_verification.json`.
During the Ni-like `3d⁹4d` branch review, fresh direct NIST level requests
for Kr IX, Rb X, and Sr XI also returned HTTP 200 and matched the archived
supplementary level-source SHA-256 byte for byte. Their `3d⁹4d` levels and
ionization limits were checked before the three DHF ledger repairs; the
source data were not altered to fit the calculation.
Fresh direct Sr III and Y V level requests for the 4p-core/5p branch review
also returned HTTP 200 and matched the archived level-source SHA-256 values.

NIST's per-element holdings pages also list 14 isotope-specific line spectra
with 3,835 rows and 17 isotope-specific level spectra with 912 rows. All
4,747 rows were imported directly from NIST, checked against the holdings
count for their spectrum, and stored with mass number, original fields,
source URL, and SHA-256. The website exposes these separately from the
standard ion tables. The line total is **296,998 + 3,835 = 300,833**,
exactly [NIST's headline count](https://physics.nist.gov/cgi-bin/ASD/lines_pt.pl).
The isotope line spectra cover D, T, ³He, ¹⁰B, ¹¹B, ¹²C, ¹³C, ¹⁴C,
and ¹⁹⁸Hg in the stages listed by NIST; ³⁹K, ⁴⁰K, and ⁴¹K have
level-only spectra.
The theory comparison page now provides a separate row-level audit for these
isotope records. Its line and level CSVs retain source URL and SHA-256,
mass number, original classification and uncertainty, a mass-aware atomx
estimate where available, and an explicit status for every row. Ritz-only
lines and bracketed or parenthesized levels are kept out of the evaluated
experimental score.

The direct [NIST ground-state and ionization table](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html)
returns 6,019 rows through Ds (Z=110) for an H–Og request; 5,847 rows have a
numeric ionization energy. Another 172 values are blank, and 1,002 of the
7,021 possible stages have no row in this query. Direct cm⁻¹ and binding-energy
queries are now stored separately. The database preserves 5,847 cm⁻¹
ionization values and 5,370 total binding energies with their uncertainty and
raw NIST notation.

## Displayed level-weight inconsistency

The direct extended-level exports include both `J` and a displayed `g`.
For a resolved level the statistical weight must be `2J+1`. The
[`scripts/audit_nist_level_weights.py`](../../../scripts/audit_nist_level_weights.py)
audit checked 110,933 rows with a simple numeric `J` and `g`; **343 rows in
28 spectra disagree**. Of these, **311 are S I**. For example, the original
[NIST S I level export](https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=S+I&units=0&format=3&output=0&multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&ids_out=1&lande_out=on&perc_out=on&biblio=on&splitting=1&submit=Retrieve+Data&de=0)
lists `J=2, g=4` for its ground level; `2J+1` is 5. Raw NIST values remain
unchanged in the database. The audit lists every discrepancy with the
source URL in `output/nist_spectra/nist_level_weight_discrepancies.csv`.
The same check on both endpoints of standard line rows found **2,626
disagreements among 541,587 checkable endpoints**, across four spectra;
**2,618** are S I. They are listed separately in
`output/nist_spectra/nist_line_weight_discrepancies.csv`.
Configuration-centroid comparisons derive the weight from an unambiguous
`J`, count every discrepancy, and require the full expected configuration
degeneracy and evaluated level energies. This avoids treating the displayed
`g` anomaly as missing S I levels.

## Stored wavenumber versus vacuum wavelength

A separate check of all **236,495 observed standard vacuum lines** compares
NIST's stored `wn(cm-1)` with `10⁷ / observed_nm`, without changing either
source field. **23 rows differ by more than 0.1%**; **13** exceed ten
times the listed wavelength uncertainty after conversion to cm⁻¹, including
nine of the 23. The union contains **27 flagged rows**. The three
first rows of Sr XI are conspicuous: the 4.4711 nm line implies about
2,236,586 cm⁻¹, while its stored wavenumber is 1,824,800 cm⁻¹. A fresh
direct download from NIST's original extended Sr XI Lines query reproduced
the 4.4711 nm and 1,824,800 cm⁻¹ fields together. Its full response SHA-256
matches the archived direct export exactly; the verification record is
`output/nist_spectra/live_sr_xi_wavenumber_verification.json`. This is not a local
row-join error. The exact rows, references, official query URLs, and source
hashes are in `output/nist_spectra/nist_line_wavenumber_discrepancies.csv`;
the reproducible check is `scripts/audit_nist_line_wavenumbers.py`.

The atlas's per-ion line table now flags these source-field differences. It
continues to use observed wavelength for wavelength comparisons and retains
the original wavenumber for inspection. [NIST's Lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)
notes that stored observed wavenumbers can have independent precision and
that wavelength uncertainties have limits, so the flag is a consistency
warning rather than a claim that every stored wavenumber is wrong.

## Provenance needed for a valid theory score

NIST's [ionization help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html)
and [level help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
state that square brackets mark semiempirical values and parentheses mark
values obtained from theory or otherwise without evaluated experimental data.
The archive now retains this classification. Among H–Ca ionization entries,
68 are unbracketed evaluated values, 94 are semiempirical, and 48 are
theoretical. Across the full NIST ionization table, the counts are 311, 919,
4,617, and 172 missing, respectively. The periodic ionization accuracy map
uses unbracketed evaluated values; all categories remain in the downloadable
row table. Of the 68 unbracketed evaluated H–Ca comparisons, **29** have a
physically unique ground level on both sides of ionization. Their strict
DHF RMS error is **5.62%**. The remaining 39 compare atomx configuration
averages with NIST ground terms and remain visible as indicative values;
pooling all 68 gives 5.72% RMS. Pooling all 210 reference types gives 3.52%
RMS, but is not an experimental accuracy score.
The separate complete-configuration-centroid comparison admits **67 of 68**
evaluated H–Ca rows and has **5.09% RMS** against those centroids. It uses
only manifolds whose J-derived weights sum to the full configuration
degeneracy and whose contributing level energies are evaluated.

The level table likewise has 96,850 unbracketed energies, 4,475 bracketed
semiempirical energies, 16,535 parenthesized theoretical energies, and 1,413
rows without a numeric energy. These classifications must be retained when
level residuals are scored. NIST's [lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)
distinguishes observed wavelengths from Ritz wavelengths derived from levels;
the existing line accuracy score uses observed wavelengths only.

### Excited-core series limits

The archive includes **6,128** NIST level `Limit` rows. **97 spectra** have
more than one listed limit (up to 24). The lowest level limit agrees exactly
with the independently downloaded direct cm⁻¹ ionization value for all
**5,847** spectra with a numeric value. NIST's
[levels help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
explains that levels can extend above the principal ionization limit and that
higher limits are listed for excited ionic cores. The atlas now displays all
limits for the selected ion next to its principal ionization energy.

For example, Na II has listed limits at **381,390.2** and **382,756.5 cm⁻¹**
for the `2p⁵` core. Its `13d` level at **380,140 cm⁻¹** lies **1,250.2**
below the principal limit but **2,616.5 cm⁻¹** below the higher limit. The
atomx `13d` binding to its Na III `2p⁵` configuration average is **2,611.6
cm⁻¹**. This suggests the higher limit is the relevant parent for that
Rydberg member; the parent assignment is an inference from the series, not a
strict fine-structure-resolved comparison. A subtraction from the principal
limit alone would falsely suggest a **1,361 cm⁻¹** theoretical error. Several
configuration labels also denote different parent terms or levels, so a
configuration match does not establish a unique series parent.

## Fields found on the original NIST forms

The [lines form](https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
offers relative intensity, oscillator strength, line strength, log(gf),
weighted A, observed minus Ritz, wavenumber, statistical weights, level IDs,
and plasma diagnostics in addition to the fields selected by the archived
export. A separate direct NIST tabular query now supplies relative intensity,
oscillator strength, line strength, log(gf), observed minus Ritz, wavenumber,
statistical weights, and level IDs. All **1,079** extended exports matched
the archived observed/Ritz wavelengths, A values, and level energies row by
row before their optional fields were attached to the **296,998** line rows.
The raw exports and hashes are retained. A separate direct query for NIST's
weighted A has passed the same row-identity gate for all 1,079 standard
line spectra. Plasma diagnostics remain unimported. The original NIST
[Lines form](https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
offers `diag_out=1` (include diagnostics) and `line_out=4` (only diagnostics),
and its [version history](https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml)
lists spectra with these data. A fresh source probe found that a normal C III
HTML line query succeeds, while either diagnostic option returns NIST's
`Query error.` page with HTTP 200. A request copied from the checked options
and selected values of NIST's live Lines form gives the same result. Fe XII
behaves the same. The C III
tabular diagnostic request returns an `Input Error` page containing the
same query error. Raw responses, exact URLs, and hashes are preserved in
`output/nist_spectra/nist_diagnostics_audit.json` and its sibling source
directory. This is an explicit source-access gap, not evidence that those
diagnostic records do not exist. A fresh 27 September 2026 check again
returned NIST's `Query error.` for the C III and Fe XII diagnostics options
(HTML and tabular), while the ordinary C III lines query returned data.
A minimal C III request containing only `spectra`, `line_out=4`,
`diag_out=1`, `format=0`, and `unit=1` returned HTTP 500. Its exact URL,
response body, and SHA-256 are retained in the same audit.
The official [version history](https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml)
still lists the affected diagnostic spectra; the raw response audit was refreshed.
No diagnostics values are inferred from the failed requests. NIST warns that relative intensities are often
qualitative and have no common scale across spectra; A alone is not an emitted
intensity without populations.
The live form HTML was also checked directly: its diagnostic checkbox has
`value="1"` and its diagnostic-only radio button has `value="4"`, matching the
probed URLs. [NIST's lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)
describes the diagnostic payload as linked line-pair intensity-ratio plots with
temperature or density validity ranges. The missing payload is therefore
additional diagnostic metadata, not missing ordinary line rows.

### Exact level-ID bridge for incomplete line labels

Of the **50,764 observed standard line rows** with an incomplete configuration
pair in the line export, **25,878** link through both NIST level IDs to levels
with usable configuration text in the same ion. No nonblank line configuration
disagrees with its linked level configuration. The DHF ledger now records
`classification_source='nist_level_id'` for these rows while leaving the raw
line columns unchanged in the NIST archive. **24,640** recovered rows have two
distinct supported configurations, **1,164** have both endpoints in one
configuration, and **74** have configurations outside atomx's supported
notation. The bridge adds only **23 new configuration jobs** because most
endpoints were already needed by other lines. The remaining **24,886** observed
rows still lack a recoverable complete configuration pair. The migration is
idempotent and can be run against an active ledger with
`scripts/generate_full_dhf_nist.py --backfill-level-ids`.

The [levels form](https://physics.nist.gov/PhysRefData/ASD/levels_form.html)
offers Landé factors, leading percentages, level IDs, level splitting, and
partition-function output. Landé and leading-percentage text are present in
many archived pages but were previously discarded; they are now imported
(18,392 and 49,611 nonblank rows, respectively). A direct, row-verified
supplementary import of level IDs, statistical weights, and splitting now
covers all 6,019 standard level spectra. NIST's [levels help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
defines its partition output as a sum over statistical weights and level
energies at a chosen electron temperature; it is not a stored level field.
The atlas now computes that listed-level sum on demand and reports missing
weights or energies instead of silently assigning them. For Mg I at 1 eV,
the imported weighted levels sum to **2.37716**, while [NIST's live query](https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=Mg+I&units=0&format=0&output=0&page_size=15&multiplet_ordered=1&conf_out=on&term_out=on&level_out=on&j_out=on&temp=1)
prints **2.38**; seven listed Mg I levels have no usable weight, so the atlas
labels its result a lower bound. The same Mg I sum also matches NIST's
printed values at 5 eV (**256.82**) and 20 eV (**793.28**); this is
consistent with NIST omitting levels that lack a statistical weight from its
partition output. The live NIST query is linked for every ion and temperature
in the site. Additional direct NIST checks agreed to the printed precision:
Na I at 1 eV (**9.92**), Ca II at 1 eV (**4.12**), Fe II at 1 eV (**76.51**),
and He I at 20 eV (**1833.59**). Those four local sums have no missing
statistical weights or level energies. All seven direct
responses, source URLs, hashes, and printed-precision comparisons are retained
in `output/nist_spectra/official_partition_samples/audit.json`; rerun
`.venv/bin/python scripts/audit_nist_partition.py` to repeat the check.
Total binding energy
was also missing from the ground-state archive and has now been imported for
the 5,370 stages NIST returned.

## Source discrepancies, not silently omitted rows

Four live NIST energy-ordered level pages print counts larger than their
displayed tables: Br IV 43/41, Nd II 840/752, Au I 75/62, and Au II 48/38
(advertised/displayed). A direct query using NIST's documented **term order**
returned 2, 88, 13, and 10 additional unique Level IDs, respectively. For
Nd II, seven rows occur only in the energy-ordered output; the union of the
two orderings has exactly 840 IDs. Common IDs agree on their nonblank level
fields. The 113 recovered rows are appended to `levels` with stable original
row indices, `source_variant='term_order'`, raw source URLs and hashes in
`nist_term_order_level_sources`, and source files in
`official_level_term_order/`. The four historical raw-page discrepancies
remain in `source_count_discrepancies` as provenance; their combined archive
counts now equal NIST's advertised 43, 840, 75, and 48 rows. The official
[levels form](https://physics.nist.gov/PhysRefData/ASD/levels_form.html)
offers both orderings.

NIST's per-element holdings additionally advertise Ba XXV 7 and W XLVII 45
levels, while direct energy-ordered downloads contain 6 and 38. Term order
does not recover the remaining eight IDs, but NIST's own line table identifies
exactly one Ba XXV and seven W XLVII endpoint Level IDs absent from **every**
level export. All eight IDs were checked in the separate direct NIST tabular
line exports. Their line classifications agree across repeated occurrences;
none has a reported energy or J. They are stored as
`source_variant='line_endpoint'` placeholders with null energy, uncertainty,
term, and J, plus the source line and reference in
`nist_line_only_level_sources`. No line wavelength is converted into a
purported level energy. The complete standard archive now has **119,394**
level records; with **912** isotope-specific records it matches all
**120,306** levels in NIST's per-element holdings exactly. The eight
endpoint-only records remain distinguishable from measured energy levels.
NIST's line headline is fully reconciled by the
isotope-specific spectra. Its transition-probability headline exceeds the
standard archive's 128,558 rows with A by 312, exactly the number of
isotope-specific line rows with A.

The atlas now links stored `L…` and `T…` reference codes to the original
[NIST bibliography records](https://physics.nist.gov/PhysRefData/ASD/Html/help.html)
from its level, line, and ionization views. It
preserves suffixes as NIST bibliography comment codes and keeps multiple-reference lists as printed by NIST. The
bibliographic databases themselves are separate NIST resources; their full
publication records are reached through these source links rather than
presented as newly measured spectral data.

## Remaining work for the all-element run

The original 15,659-job DHF ledger covers observed-line endpoint configurations
and one ground configuration for each of the 7,021 ion stages. A direct-NIST
ground-selection repair added 187 configurations. A later ionic-sequence
repair added 711 candidate configurations, and 30 nearby ionic alternatives
raised the ledger to **16,587** jobs. Old solved configurations remain for
comparison. The selected
ground configurations trace directly to 5,722 NIST level rows and 297 NIST
GSIE ground-shell fields. Of the remaining 1,002 stages, **937** use a
separately labelled extrapolation from the nearest lower-Z NIST ionic member
of the same electron sequence; **65** still use neutral isoelectronic filling.
The extrapolation is a theory candidate, not an experimentally assigned NIST
ground. When both it and the old neutral-filling candidate converge, the
lower DHF+B+QED+mass energy is selected. The website and CSV retain this
provenance, and `scripts/supervise_dhf_ground_selection.py` rechecks candidate
pairs as the batch finishes.
It uses an
adaptive radial box for diffuse states and explicit basis-limit statuses.
`scripts/audit_first20_dhf.py --max-z 118` writes
`output/nist_spectra/full_dhf_audit.json` and checks every observed NIST line
key, wavelength and medium against the DHF ledger, all 7,021 ground-stage
jobs, finite converged energies and mass shifts, and the ordering of completed
Rydberg series. It also flags duplicate grounds and nonpositive ionization
energies between completed adjacent ground stages. The command returns nonzero while any ground job is unfinished
or an audit check fails; `--require-complete` additionally requires every
configuration job to finish. The H–Ca default retains the same checks for
Z=1–20.
The [theory comparison audit](../../THEORY_NIST_COMPARISON.md) documents the 188
ions whose abbreviated level ground configuration was replaced by the
complete NIST GSIE ground-shell field. For Mt VII and Ds VIII, the level
configuration and GSIE `ground configuration` agree on `6d²7s`, while the
distinct GSIE `ground shells` field gives `[Rn]5f¹⁴6d³`; the repair report
preserves both fields and leaves the level configuration selected.
Many NIST line classifications lack enough state identity for single-
configuration DHF to resolve term and J components; such wavelengths remain
inspectable configuration averages outside the strict experimental score.
Plasma diagnostics remain a documented source-access gap. On 27 September,
all **7,021 selected ion-stage ground jobs** converged. The full-table
ledger audit confirms their presence, finite energies and mass shifts, and
positive adjacent-stage ionization energies wherever both stages are
available. Alternate ground candidates and the excited-state queue remain
active calculation work.

## 29 September field-level follow-up

A live Ca I check exposed numeric parent-J annotations stripped from standard
level labels by HTML cleaning. **21,607 labels across 5,501 spectra were
restored** from matched cached official records with a transactional provenance
ledger. Scientific level fields and record counts are unchanged; all 787 Ca I
rows now match the live source on eight checked fields. All 296,998 standard
line raw endpoint labels and 4,747 isotope raw records passed separate cached
source comparisons. See [the parentage repair report](NIST_PARENTAGE_REPAIR_2026-09-29.md).
Earlier row-count completeness claims must not be read as proof that every
classification field was preserved before this repair.

## 29 September editorial-note import

The cached source headers contain evaluation notes that had not been exposed
in the database or website. `scripts/import_nist_spectrum_notes.py` now indexes
**1,430 notes for 641 spectra** after verifying the SHA256 digests of all
**7,098** standard line/level source files used by the import. These include
energy-level evaluation notes, line-intensity context and transition-probability
evaluation notes. For example, Fe II's header explains its 1.9 eV effective
intensity scale. The ion-detail view presents escaped note text and source
links; raw HTML and source digests are retained for provenance.

This is a cached-source completeness improvement, not a fresh live-source
audit. Before/after full-row digests are identical for all 296,998 standard
lines, 119,394 standard levels and 6,019 ionization records. Reports are under
`output/nist_spectra/normalization_diagnostic/`: `nist_spectrum_notes_applied.json`
and `spectrum_notes_{before,after}_invariants.json`. The archive file hash changes
because a derived table was added; historical report hashes are not rewritten.

Atlas rebuilds import the same derived note table. Tests cover source-digest
failure before mutation, repeat-import idempotence, catalog-source integration,
and optional-table API compatibility. Notes do not close the separately
documented plasma-diagnostics or full-bibliography gaps, and this parser's scope
is the known editorial-note cells in the cached headers.

## 29 September preservation of complete line source rows

Normalized line records did not expose every original lexical field. For
example, parentheses and brackets around endpoint energies were absent from
the numeric energy fields. The line-detail importer now retains each complete
parsed TSV row in `source_fields_json`, alongside all existing normalized
columns. Original significant figures, omitted-column distinctions, intensity
annotations, references, IDs and energy annotations remain inspectable.

Offline migration imported all **296,998 rows across 1,079 spectra**, with
zero mismatches or missing cache files. Each cached file was checked against
the digest of its previously imported source before enrichment. Whole-row
digests of every pre-existing column remain identical for the line table,
119,394 levels and 6,019 ionization records. Verification artifacts are
`normalization_diagnostic/line_lexical_before.json` and
`normalization_diagnostic/line_lexical_import_verified.json`.

The preserved line fields contain 12,338 parenthesized upper energies and
10,286 parenthesized lower energies, plus 6,709 bracketed upper and 7,165
bracketed lower energies. These are counts of line endpoints, not distinct
levels. The API supplies structured `source_fields`; the website displays
escaped, expandable “Original NIST fields”. CSV exports retain the source-row
JSON. No scientific values were replaced and no fresh live-source completeness
claim is made. The separate bibliography and plasma-diagnostics gaps remain.

The importer also repairs already-imported spectra whose source-row JSON is
missing, without requiring a network refresh. A cached hash mismatch leaves
the previous records and source ledger intact.


## Original level-field preservation (29 September 2026)

The level API now exposes `source_fields`, an exact parsed TSV row, alongside
normalized columns. This preserves printed precision, uncertainty notation,
blank or absent optional columns, leading-percentage spacing, Landé values,
references and level IDs. The existing “Original NIST fields” viewer control
also handles levels. Energy-order and supplemental term-order sources retain
their distinct identities; line-only placeholders have no invented level row.
Offline migrations verify each previously imported source SHA-256 before
attaching fields. Source changes are rejected before updating its records.

The official [NIST level help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
was checked again. It explains that strong configuration mixing can make
configuration labels bookkeeping designations, and that absent predicted
levels are not exhaustively listed. Therefore complete import of available
records does not establish complete physical state coverage or justify
nearest-energy matching to a pure-configuration theory.

The migration also exposed an In II normalization bug: `5s<1/2,F=4>.5g`
was reduced to `5g1`, dropping its occupied 5s shell. Normalization now removes
the angular annotation while retaining `5s1 5g1`; the raw source annotation
remains intact. The same fix covers the F=5 row. No experimental energy is
changed and no hyperfine-resolved theory prediction is inferred from this fix.

Verified coverage: all 119,273 energy-order rows and all 113 supplemental
term-order rows now retain the original source fields. The eight line-only
placeholders remain distinct and correctly have no level-export source row.
All 6,019 level-detail source entries are imported; the final offline pass
reports zero failures and zero uncached sources. This is cached-source
verification, not a fresh download of every spectrum.

Full ordered-row SHA-256 comparisons confirm that every pre-existing column
is unchanged across 296,998 lines, 119,394 levels and 6,019 ionization rows.
The before/after evidence is in
`output/nist_spectra/normalization_diagnostic/level_lexical_before.json` and
`level_lexical_import_verified.json`. Eleven targeted tests pass, including
source-hash rejection, offline migration of existing imports, term-order
reimport, API exposure and hyperfine-shell normalization. Viewer JavaScript
syntax is valid; rendered-browser inspection is still outstanding.

## Bibliography archive expansion (29 September 2026)

Direct retrieval of the public ASBib reference endpoint succeeded after the
web tool could not access it. Inspection of reference L7185 confirmed that
its response contains a title, authors, journal citation, publication links
and spectrum-context navigation, beyond the reference codes stored in the
scientific tables. The importer preserves original response bytes by URL hash,
their SHA-256, capture time and safe extracted citation text/links. Scripts and
styles are excluded from the displayed text; source HTML is never injected
into the website. The requested bibliography ID must appear in the response,
so an HTTP-200 challenge page cannot be imported as a citation.

There are 23,580 distinct standard-data request URLs when reference code,
spectrum and transition-type context are retained. This exceeds the 15,027
contexts counted before considering transition type. The first 12 requests
completed successfully. A full import is running with two workers and a
one-second delay before each uncached request, using frozen source under
`/tmp/atomx-nist-bibliography-20260929`. The log is
`/tmp/atomx-nist-bibliography-full.log`. Failed requests remain retryable;
completed cached sources are checked against stored hashes on restart.

The SQLite table is `nist_bibliography_sources`, and response files live in
`official_bibliography/`. Timestamped `bibliography_manifest_*.json` files
retain the complete request inventory, source-code hashes, request controls
and unparsed reference/annotation strings. Original scientific tables are
not modified. The atlas ion panel now exposes imported citations and errors
with source URLs and hashes. Four parser/API tests pass; JavaScript syntax
passes, with no rendered-browser inspection yet.

This is an in-progress expansion, not a bibliography-completeness claim.
Standalone comment codes, unresolved reference strings, and isotope-only
reference contexts still require separate treatment. The importer does not
retrieve full journal articles or claim to archive all NIST bibliography
entries beyond those cited by the standard records.


### ASBib query-category repair

The first failed annotated citation led to a URL-construction audit. NIST's
cached HTML links use `type=T` for transition-probability bibliography,
`type=L` for wavelength bibliography and `type=E` for level bibliography.
The initial importer and existing website incorrectly supplied transition
multipolarity (or an empty string) for line references. Both now use the
bibliography category. `type=I` is retained for ionization references.

The old importer was explicitly stopped after confirming this defect. Its
173 wrong-category responses are retained with status
`superseded_query_category`; they are excluded from the current citation panel.
Fifty already-correct imports remain reusable. The corrected import runs from
`/tmp/atomx-nist-bibliography-categories-20260929`, logging to
`/tmp/atomx-nist-bibliography-categories.log`. Physics workers were not stopped.

The corrected inventory deduplicates physical transition multipolarities and
also requests each annotated code's base citation independently, resulting in
27,333 source URLs. This is the revised inventory, not additive to the earlier
23,580 requests. There are 125 unresolved standalone reference/annotation
strings, preserved in the manifest.

A live check confirms that the `T1115CA` Li I annotated request still returns
NIST's own Data Request Error page even with the correct category. Its base
`T1115` citation returns successfully. The base citation is a separate record;
it never substitutes silently for the missing CA annotation. Error pages fail
citation-ID validation. Ten parser, inventory and API tests pass, including
category selection, annotated/base separation, multipolarity deduplication,
unsupported strings and safe rendering data.

## Standalone wavelength annotations (29 September)

A fresh Ti I HTML line query established that standalone `c112` is an ASBib
wavelength comment with an empty bibliography ID, not an unparseable citation.
Its exact original link has empty `db_id`, `ref` and `type` parameters. The
response identifies itself as `NIST Atomic Spectra Database Wavelength Comment
c112` and gives source attribution to material quoted in Russell 1927.
The HTML query, comment response and URL/hash manifest are preserved in
`bibliography_diagnostics/standalone_annotations/`.

Added a separate `nist_reference_annotations` table and importer. It discovers
standalone c-number codes in the standard line archive, follows only matching
HTTPS links on the original NIST host and ASBib endpoint, validates the comment
identity, and stores original bytes, safe text, source links and hashes. It
never treats the comment as a bibliographic article and does not change line
measurements or level energies. The existing bibliography importer continues
independently; its historical unparsed-code inventory is not rewritten.

The atlas API returns `reference_annotations`; the ion view displays a separate
archived wavelength-comments panel. Four parser/API tests and viewer JavaScript
syntax checks pass. Other code families, isotope-only annotation contexts and
citation links nested within comment text remain separate completeness work.

An initial H I comment (`c39`) identifies a blended Balmer feature whose quoted
center is calculated under population assumptions and can vary with source
conditions. Its archived row has a Ritz wavelength but no observed wavelength;
therefore the observed-line campaign correctly excludes it. This check avoids
mistaking the newly imported note for a newly discovered measured-line error.

Reproduce with `PYTHONPATH=src:. python scripts/import_nist_reference_annotations.py`;
`--offline` reuses cached responses. Per-context results and implementation hashes
are retained in `official_reference_annotations/import_report.json`.

The initial annotation pass encountered multiple distinct source links for Cu II
`c153` (empty and E2 type parameters). The importer now retains every exact
original link and response in `source_variants`, instead of choosing one query
or overwriting a context. Resumed from cached responses; the bulk import remains
in progress. Five parser/API tests now pass, including this multiple-context
case. These standalone comment parameters are taken from the source HTML and
must not be inferred from the separate bibliography-category rules.

The standalone c-number wavelength-comment pass completed: all **53 discovered
contexts imported**, zero failed or missing source links. This covers c-number
items, including comma-separated items, in standard `lines.line_reference`;
it does not claim coverage of every annotation family or isotope-only source.
The exact scope and source variants remain in the per-context report.

### Exact original hyperlink check for T1115CA

Fetched the original Li I HTML line table and followed its literal T1115CA
hyperlink, whose `type` parameter is empty (unlike the spectrum-header T
category). It is a combined citation/comment link, not two independently
rendered links. This exact original request also returns the NIST `ASBib Data
Request Error` page. The annotation remains unresolved; the independently
archived base T1115 citation must not be presented as its replacement.
Evidence, URLs and hashes are in `bibliography_diagnostics/t1115ca/`:
`Li_I_original_links.html`, `Li_I_original_links_manifest.json`,
`original_row_response.html`, and `original_row_result.json`.
This check does not establish that other failed suffix contexts have the same
cause, and per-row source query parameters must not be inferred from header
links without verification.
