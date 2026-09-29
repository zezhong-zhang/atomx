# High-angular-momentum NIST configuration recovery

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The H–Ca completion review found 137 observed line records marked
`unsupported_configuration`. The shared nonrelativistic parser accepted only
s through i, while the NIST importer and radial solver already supported k.
This unnecessarily rejected valid 8k and 9k configurations.

The parser now uses the same spectroscopic letters as the existing solver
labels (`spdfghiklmno`, omitting j). Tests verify NIST core expansion, the
14:16 jj statistical weights for a k electron, and a point-nucleus Z=10 8k
DHF configuration average against analytic Dirac energy within 1e-9 Ha.
Both tests passed.

`scripts/recover_high_l_nist_jobs.py --apply` recovered 28 H–Ca line endpoint
pairs directly from the archived NIST classifications and added eight unique
configuration jobs. Previous line-ledger rows are preserved in
`high_l_line_recovery_history`; original NIST archive fields are unchanged.
The new jobs use `candidate_pending` so workers launched before the parser fix
cannot claim them. A new worker is calculating them with the full corrected
model and local stationary rotations. Predictions are withheld until the
endpoints finish. The number of prepared first-20 configurations is therefore
now 3,474 rather than 3,466.

109 unsupported records remain after this targeted recovery. This patch does
not infer missing classifications or turn configuration averages into resolved
multiplet spectra. `first20_completion_review.json` includes these limitations
and uses one SQLite backup snapshot for queue counts, integrity checks and
matched NIST comparisons.

## Subsequent raw-notation and QED coverage repair

The offline line-details importer now retains `lower_configuration_raw` and
`upper_configuration_raw` directly in the SQLite archive. All 1,079 cached
sources passed row matching and were imported without failures. Original source
files and their hashes remain unchanged. These fields are available through
the record API and full-record website table.

Parentage removal now preserves subshell separators, and repeated subshell
occupations are summed. Using the raw classifications recovered another 16
observed lines and seven configurations: 44 recovered lines and 15 new
configurations in total, bringing H–Ca to 3,481 configuration jobs.

Two recovered Ne VIII 9l configurations exposed incomplete QED data: the stored
Bethe logarithms covered only l <= 7. The table now contains all 20,100 entries
for 1 <= n <= 200 and 0 <= l < n from the retained reference PDF. The importer
verifies its SHA256, regenerated PDF text, complete triangular coverage, and
exact preservation of all 1,572 previously stored values. Both Ne VIII jobs
now complete with QED enabled; original failures are preserved in
`bethe_log_coverage_retry_history`. The focused parser, analytic Dirac, and QED
table tests pass (19 tests).

The remaining 93 unsupported H–Ca line records have grouped angular momenta
or unresolved principal quantum numbers, such as `9ikl`, `12[k-o]`, and `nd`.
Their raw classifications and reasons are retained in
`first20_unresolved_classifications.json`. They require additional state or
blend assignments before a unique theoretical transition can be compared.
