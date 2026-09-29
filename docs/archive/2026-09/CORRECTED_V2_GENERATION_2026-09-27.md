# Corrected v2 generation: first-20 calculation scope

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Current scope

The separately versioned ledger `output/nist_spectra/full_dhf_corrected_v2.sqlite`
contains the all-element queue, but its worker is restricted to Z≤20 by
`full_table_run_status=first20_excited_state_audit`. A request for Z>20 is
rejected before any job is claimed. This continues the numerical and
state-matching audit; it does not assert agreement with experimental
uncertainties or release the full periodic table for calculation.

The model is DHF with static DBSR Breit, model QED, first-order one- and
two-electron Breit recoil, and the exact one-electron nonrelativistic
mass remainder. The latter is zero for multielectron configurations.
Higher-order relativistic, radiative and finite-size recoil remain absent.
Electron correlation beyond exchange remains absent in DHF.

## Migration and validation

`scripts/prepare_corrected_mass_generation.py` creates a new ledger without
altering the historical or corrected-v1 databases. It records the old energy,
new energy and additive remainder for each of 39 previously audited ground
configurations in `model_migration`. It imports 46 fresh one-electron audit
configurations after checking version, isotope mass, convergence, and
agreement of overlapping endpoints; two overlap the ground set, yielding
83 completed configurations. The one-electron audit retains its own baseline
identity checks and correction components.

A new ordinary worker calculation for Ca XIX 1s² completed at
−389.439529497 Ha (rounded), with the hydrogenic remainder explicitly zero.
At batch start, 84 configurations were completed and 3,382 pending inside
H–Ca. Counts will change as the worker proceeds.

The initial publication attempt exposed a WAL-sidecar issue: a SQLite
connection context manager commits but does not close the connection, and
renaming the main file alone left committed changes under the temporary WAL
filename. The ledger was recovered under its original temporary name,
checkpointed, closed, and republished after an integrity check. Preparation
now uses explicit connection closing and DELETE journal mode before edits,
so the published file is self-contained. A regression test starts from a
live WAL source and verifies all committed rows after publication.

## Execution

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/generate_full_dhf_nist.py --worker-only \
  --output output/nist_spectra/full_dhf_corrected_v2.sqlite \
  --max-z 20 --refresh-site-every 50
```

Log: `/tmp/atomx-corrected-v2-first20.log`. Inspect the process and job ownership
before resuming; do not launch a duplicate merely because output is delayed.
The worker reads the persisted physics version and forwards the required
remainder option through ordinary, damped and shooting-seeded solve paths.
The v1 default remains available for reproducing old audits. Unsupported
versions and attempts to append a different version to completed rows fail.

After verifying spare CPU/memory capacity and the queue's `BEGIN IMMEDIATE`
atomic claim, three additional workers were started with the same Z≤20
restriction and `--refresh-site-every 0`. Their logs are
`/tmp/atomx-corrected-v2-first20-worker{2,3,4}.log`. BLAS/OpenMP thread counts
are one per worker. Only the original worker refreshes the website. Inspect
live PIDs and `worker_pid` ownership in the ledger before adding or resuming
workers; these are shared-queue processes, not independent overlapping runs.

The pool subsequently grew to eight single-threaded workers after the
first-20 core-p branch policy was validated. Current logs use
`/tmp/atomx-corrected-v2-first20-multistart-worker{1,2,3,4,5,6,7,8}.log`.
See [the branch-repair audit](FIRST20_CORE_P_BRANCH_REPAIR_2026-09-27.md)
for the required two-start comparisons and retained intermediate results.

The corrected website also compares excited configuration energies against
complete evaluated NIST centroids. An intermediate snapshot contained 689
qualifying comparisons; incomplete and nonexperimental manifolds remain
unscored. These are separate from individual-line and ionization scores,
and from all historical-model panels.

The website's corrected-generation panel now reads the v2 ledger when
available. Historical accuracy panels keep their original source. Pending,
ambiguous and configuration-average rows are not promoted into strict
individual-line scores.

## Next gates

1. Inspect first-20 failures and convergence/state-identity flags as the batch
   advances; resolve concrete numerical failures before retries.
2. Compare completed unique-level lines and complete evaluated configuration
   centroids with their corresponding NIST references. Retain unclassified
   records and disclose missing theoretical resolution.
3. Review across-table numerical controls before expanding the calculation
   scope. DHF model deficits must remain explicitly separated from solver
   errors; producing a mean-field database does not establish correlated or
   precision-QED accuracy.
