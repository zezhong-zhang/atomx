# First20 shooting recovery comparison — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

`scripts/compare_recovered_shooting_campaign.py` now compares the original
campaign plus audited recovery results without modifying calculation databases.
Each SQLite input is backed up to a consistent temporary snapshot. Recovery
ancestry must lead to the selected base campaign, and the refinement audit must
accept both grids before a state becomes eligible. An accepted base state is
retained; otherwise, the first eligible recovery in command-line order supplies
the whole pair. Nuclear identity, grid requests and ground-state designation
must match the original job. Radial boxes may differ between independently
validated recoveries. No pair combines attempts from different campaigns.

The JSON preserves campaign identities, snapshot digests, audit counts,
comparison-code hashes and per-endpoint calculation-source identifiers. A
pending recovery can contribute already audited pairs; its pending and failed
attempt counts remain visible. This is a comparison snapshot, not a merged
calculation campaign or a claim that recovery is complete.

## Current snapshot

`output/nist_spectra/native_shooting/recovered_campaign_comparison.json`:

- 7,494 / 7,543 configurations have accepted two-grid records, including
  73 supplied by recoveries.
- All 35,802 standard observed-line records remain represented.
- 27 lines still lack calculated endpoints, all in Ca I.
- 32,446 lines have configuration-average predictions, 176 are unresolved
  within one configuration, and 2,883 lack classifications.
- 260 lines have negative calculated configuration-average intervals.
- 2,009 complete evaluated NIST configuration centroids can be compared.
- Only 10 lines meet the unique-level criterion. Their RMS of about
  0.000418 cm⁻¹ is not a periodic-table accuracy score.

The inversion flags are concentrated in Ar II (113), S I (35) and Ne II
(29). The 12 H I flags include specific-J Lamb-shift transitions such as
2p(1/2) → 2s(1/2), whereas the present prediction averages over the p
configuration. These flags must not be interpreted as proof that the resolved
levels are incorrectly ordered. A J-resolved calculation is required.

## Website and verification

The native shooting campaign page has a snapshot selector and a source-details
panel. The recovered JSON is served separately from the historical JSON.
Checks cover whole-pair selection, recovery priority, preservation of accepted
base results, mismatched mass, missing grids, unrelated ancestry, and both HTTP
routes. The combined targeted suite passed 17 tests including architecture
boundaries; the later route/comparison/audit suite passed 10 overlapping tests.
JavaScript syntax passed; no rendered-browser inspection was performed.

## Reproduction

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/compare_recovered_shooting_campaign.py \
  --campaign output/nist_spectra/native_shooting_first20_stationary.sqlite \
  --recovery output/nist_spectra/native_shooting_heavy_continuation_recovery.sqlite \
  --recovery output/nist_spectra/native_shooting_argon_implicit_bootstrap.sqlite \
  --recovery output/nist_spectra/native_shooting_light_implicit_recovery.sqlite \
  --recovery output/nist_spectra/native_shooting_first20_refined_recovery.sqlite \
  --output output/nist_spectra/native_shooting/recovered_campaign_comparison.json
```

Running recoveries can change the counts on a later snapshot. Complete the
remaining states, review configuration ordering and resolved-state coverage,
and converge correlation before interpreting the residuals as full experimental
accuracy. Full-table shooting generation remains outstanding.

## Targeted observed-endpoint follow-up

The remaining 27 unavailable observed-line intervals were all Ca I 4s4p
to 4snd with n=4–8. A separate five-configuration recovery now prioritizes
these endpoints without stopping the broad high-n queue. Its frozen numerical
source is copied from the successful K/Ca exchange-continuation run; the driver
adds explicit configuration selection and records that filter in its identity.
It retains the historical correction model so its pairs can be combined with
the historical lower endpoints. It does not contain the later Mohr normalization
or leading radiative mass revisions, which have separate one-electron reports.

The first three completed pairs (4s4d, 4s5d, 4s6d) pass their audit and restore
19 observed-line comparisons. The refreshed snapshot has 7,522 / 7,543 accepted
configuration pairs and only eight observed lines with endpoints still pending.
The source is `native_shooting_calcium_observed_recovery.sqlite`; include it
first in the recovery list when regenerating this snapshot. These counts are
an in-progress snapshot, not final campaign completion.

## Observed endpoints recovered

The targeted Ca I recovery completed all ten attempts. All five two-grid
pairs pass the stage-history, orbital-update, correction-ledger and orbital
identity checks; none is flagged. The refreshed combined snapshot accepts
7,526 / 7,543 configurations, with 105 supplied by recoveries. There are now
**zero observed-line records waiting solely for calculated endpoints**.
All 35,802 observed records remain represented: 32,473 configuration-average
predictions, 10 unique-level comparisons, 176 unresolved intra-configuration
intervals, 260 negative configuration intervals and 2,883 missing classifications.
There are 2,014 matched evaluated configuration-centroid comparisons.

This closes the targeted endpoint-convergence gap, not the state-resolution or
correlation gap. Seventeen configurations still lack accepted paired records;
the older broad high-n recovery remains active. The combined historical
correction model is unchanged, including its older self-energy normalization.
Updated one-electron QED calculations are separate snapshots and must not be
silently mixed with these many-electron results.

The subsequent snapshot accepts 7,531 / 7,543 pairs, including 110 recovery
pairs, with 2,030 matched evaluated centroid comparisons. Twelve configuration
pairs remain unavailable; observed-line status counts are unchanged. The broad
recovery coordinator was confirmed live when this snapshot was generated.
