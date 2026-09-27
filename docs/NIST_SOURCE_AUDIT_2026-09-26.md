# NIST ASD source audit (26 September 2026)

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
[`scripts/audit_nist_level_weights.py`](../scripts/audit_nist_level_weights.py)
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
covers all 6,019 standard level spectra. Partition-function output depends
on a chosen temperature and is not a stored level field. Total binding energy
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
The [theory comparison audit](THEORY_NIST_COMPARISON.md) documents the 188
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
