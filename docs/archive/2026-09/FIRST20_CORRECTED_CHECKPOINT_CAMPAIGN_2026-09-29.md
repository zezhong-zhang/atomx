# First20 current-correction shooting campaign

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

A new, separate campaign recalculates all 7,543 configurations in the first20
queue (15,086 grid attempts). It does not replace or merge the historical
stationary campaign and its audited recoveries. The last historical missing
pairs are Ca I 4s53d–4s62d, 4s73p and 4s9d; all were confirmed in the existing
live recovery queue. No duplicate recovery process was started.

## Model and execution

- Native Dirac HF shooting, statistical configuration averages, the parent
  nuclear masses/isotopes and radial boxes, and two grids per configuration.
- Bootstrap: tolerance 1e-12, damping 0.5, up to 400 iterations.
- Warm refinement: implicit nonlocal exchange solver, tolerance 1e-13,
  damping 0.5, up to 300 iterations. Nonconvergence is retained as a failure.
- Post-SCF DBSR_HF Breit, Mohr self-energy with the exact point-Dirac
  screening denominator, Uehling vacuum polarization, operator normal and
  specific mass shifts, and the existing one-electron mass remainder.
- The leading radiative-mass addition is explicitly enabled for one-electron
  stages only. It is not a many-electron radiative recoil calculation.
- Converged bare orbitals are saved before corrections, with path, stage and
  SHA-256 recorded per result. Each campaign gets a separate file namespace.
- Four workers; source frozen at
  `/tmp/atomx-first20-corrected-checkpoints-20260929`.

Database: `output/nist_spectra/native_shooting_first20_corrected_checkpoints.sqlite`.
Orbital files: `output/nist_spectra/shooting_orbitals/`.
Log: `/tmp/atomx-first20-corrected-checkpoints.log`.

The launch used the frozen `scripts/generate_native_shooting_nist.py` with
`--workers 4 --max-iter 400 --warm-tolerance 1e-13 --warm-max-iter 300
--radiative-mass-one-electron --checkpoint-dir` and explicit source/output
paths. Queue and source identities are immutable; resume with the same frozen
source and arguments. Do not resume this database from a changed worktree.

## Initial measured checks

A stage-history audit of the first 170 attempts found 170 successful attempts,
84 complete pairs and no flagged orbital identities. A later independent
checkpoint check verified all 260 completed files in its consistent database
snapshot: hashes, converged bare total energy, orbital energies, exact jj
configurations and refinement tolerance. None already had corrections applied.
These are different capture times, not final totals. Evidence is stored in
`normalization_diagnostic/first20_corrected_checkpoint_audit.json` and
`first20_checkpoint_integrity.json` under the NIST output directory.

Sixteen generation/radiative-mass/persistence tests passed before launch;
eleven campaign/refinement/route tests passed after extending the audit to the
campaign control layout. The website snapshot selector exposes
`corrected_checkpoint_comparison.json` separately and marks it in progress.
It preserves all observed records, including those whose new endpoints are
pending. JavaScript syntax passed; visual browser inspection is outstanding.

This run improves correction consistency and makes future post-SCF updates
reproducible without re-solving orbitals. It does not add correlation or turn
configuration averages into resolved fine-structure lines. Full-table shooting
and systematic correlation validation remain unfinished.

## First observed failure and isolated Ne VIII recovery

The adaptive bootstrap for job 3142 (`Z=10, q=7, 1s2 6g1`, 3200 points)
reached 400 iterations with undamped orbital-update norm 4.66624e-6, above the
unchanged 1e-6 gate. It was correctly recorded as failed.

A separate control using `nonlocal_resolvent` from the initial solve, then the
same 1e-13 refinement, converged at both 3200 and 6400 points. Final update
norms are 1.87405e-7 and 1.87404e-7; corrected total-energy grid difference is
1.7466e-7 cm^-1. Controls, full correction provenance, source hashes and bare
HDF5 checkpoints are retained in `normalization_diagnostic/neon_6g_current/`.
The driver lives in the frozen periodic-control v3 source snapshot. This is an
isolated successful recovery; it has not yet replaced or been merged into the
campaign's audited comparison, and the original failed attempt is preserved.

### Audited Ne VIII recovery incorporated

Re-ran the successful Ne VIII control through the source-locked recovery driver,
writing `native_shooting_current_neon_recovery.sqlite`. Both grids passed the
independent audit with no flagged pair. The recovery driver now records its
actual nested solver controls, writes namespaced bare-orbital checkpoints and
preserves the parent's one-electron radiative-mass policy for applicable jobs.

`native_shooting/corrected_recovered_comparison.json` now combines the current
campaign with this audited pair and retains source identity per endpoint. The
original failed row is retained. At initial publication, 2776 configuration
pairs were accepted, one from recovery. The campaign website offers this as
“Current corrections + audited recoveries (in progress)”. Nine recovery/audit/
HTTP-route tests and JavaScript syntax checks pass.

### Six recovered configurations and current residual review

Audited and incorporated the C IV 5g, O VI 6g and Ne VIII 7g pairs, alongside
N V 5g, F VII 6g and Ne VIII 6g. Every recovery has both successful grids and
passes the existing phase, identity and correction-ledger checks. At this
snapshot the combined report has 3092 accepted configuration pairs, six from
recovery. Original failures remain in the production ledger.

Added `scripts/review_native_shooting_centroids.py` to rank matched centroid
residuals and grid changes and summarize each ion stage without mixing these
with resolved-line scores. The saved partial review includes 918 matched
centroids across 154 stages, with its input snapshot hash. No residual-size
exclusion is applied. Its largest grid change is 0.00057534 cm^-1 (Ar VIII 6h),
compared with a 4397.893 cm^-1 residual in that case. This is numerical evidence,
not an experimental uncertainty estimate. The largest absolute residual is
still the previously investigated Ca XVI 4f assignment; see
`CA_XVI_4F_SOURCE_RECHECK_2026-09-29.md`. NIST values remain unchanged.

Artifact: `native_shooting/corrected_centroid_review.json`. Re-run the script
against the final corrected recovery comparison once the campaign finishes;
these partial statistics do not establish full first20 accuracy.

### Historical closure and current-source progress

The historical stationary campaign plus its six recovery sources now has all
7543/7543 configurations accepted on two grids, including 122 recovered
configurations. All 35802 standard observed-line rows are accounted for:
2883 lack classification, 32473 are configuration-average comparisons,
176 are unresolved intra-configuration lines, 260 have inverted configuration
intervals, and 10 have unique-level matches. There are no unfinished endpoints
in that historical snapshot. Of the negative intervals, 99 also have negative
NIST centroid intervals, 159 lack complete NIST manifolds, and two disagree
with complete-centroid ordering. Historical correction conventions remain
explicit; this does not complete the updated-corrections campaign.

The latest corrected snapshot has 4957 accepted configuration pairs, including
11 from audited recoveries. The newly completed high-l batch contains Ne VIII
8g, Cl VIII 8i and Ar VIII 8i; all six grid attempts passed the phase/identity/
ledger audit. These were calculated using the frozen adaptive-stagnation fix
and incorporated with source identity preserved.

The subsequent live audit recorded 9961 successful and 48 failed attempts,
with 5077 still pending. Reconciling the failures against accepted recoveries
left 23 uncovered configurations (35 attempts), primarily high-l states with
stalled adaptive orbital updates. Started a separate batch over the then-failed
queue using the tested stagnation-aware frozen source, 400 bootstrap iterations,
1e-13 refinement, four workers and saved bare checkpoints. Output:
`native_shooting_current_stagnation_batch2.sqlite`; log:
`/tmp/atomx-current-stagnation-batch2.log`. Repeated previously recovered states
are independent controls and will not be double counted. This batch is pending
its completion and audit before incorporation.
