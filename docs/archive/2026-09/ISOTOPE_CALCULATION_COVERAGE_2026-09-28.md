# Explicit isotope calculation coverage

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The existing isotope comparison CSVs use the legacy ledger and a leading
normal-mass-shift transfer. They are estimates, not calculations with the
current v3 Hamiltonian. Redirecting that transfer formula to v3 would not
correctly update specific recoil, charge-size effects, or the other corrections.

`scripts/audit_isotope_calculation_coverage.py` now inventories every imported
isotope line and level, retaining its source URL, hash, and row identity. Its
machine-readable report is `output/nist_spectra/isotope_calculation_coverage.json`.
The inventory uses named isotope masses and configuration dependencies, including
level-only configurations and explicit zero-energy level references.

Snapshot at 2026-09-27 16:42 UTC:

- 3,835 line records and 912 level records: all 4,747 retained.
- 434 distinct isotope/configuration dependencies.
- 42 completed states reusable with matching v3 version, mass, isotope, and
  nuclear-profile metadata.
- 24 matching states pending in the standard v3 ledger.
- 156 states require a different isotope from the standard calculation.
- 212 states have no corresponding configuration in the standard ledger.
- 403 records have both configuration energies available at this snapshot.
- 70 records contain an unresolved configuration or reference dependency.

These counts describe model availability, not spectroscopic accuracy or resolved
fine/hyperfine levels. The current empirical/reference radius prescription is
not a complete set of measured isotope radii. QED remains a model correction;
hyperfine structure and configuration correlation require further treatment.

Next implementation requirement: a separate calculation queue keyed by
`(Z, A, charge, configuration)` for the 368 missing states, with actual named-mass
recoil and the current nuclear model. Standard-queue results can be reused only
after the matching checks above. Keep legacy transfer values explicitly labeled
until explicit results replace them; preserve every unscorable source record.

Validation: three isotope comparison/inventory tests passed. The new test rejects
both a completed state with the wrong mass and one with a missing charge profile,
and verifies that absent/unclassified dependencies remain visible.

## Explicit queues started

`scripts/generate_isotope_dhf_nist.py` prepares separate standard-schema ledgers
for each `(Z,A)`, reusing the production worker without changing its convergence
or correction paths. The 12 ledgers in `output/nist_spectra/isotope_dhf_v3/`
contain all 434 configuration dependencies, with 42 validated completed states
copied from the main run. Pending matching states are calculated independently
in these isolated queues; no running-job ownership is copied or reclaimed.

Preparation audits passed for every ledger. The deuterium pilot completed 1s
and 2p with the named deuterium mass and explicit nuclear model. Its stored-record
audit passed with maximum component-sum discrepancy 5.6e-17 Ha; this checks
accounting and the existing one-electron basis gate, not precision agreement
with measured fine/hyperfine structure. The remaining queues have been launched
through one worker, logged at `/tmp/isotope-v3-explicit-worker.log`.

Two queue/inventory tests passed, including separate D/T masses, refusal to
overwrite queues, and worker dispatch with running-job recovery disabled.
The comparison builder now selects these explicit ledgers when their manifest
exists. Both line and level CSVs include the model identifier and generic
`dhf_nm`/`dhf_cm1` columns. Legacy mass-transfer columns are empty in explicit
mode, and unfinished states have no fallback transfer prediction. Every source
record remains included. Level comparisons require an explicit zero-energy
reference; an arbitrary lowest numeric entry is no longer treated as ground.

The UI reads the new energy columns and model explanation. Existing unique-level
versus configuration-average distinctions remain in place. Tests verify actual
explicit-energy differences, missing-result behavior, and rejection of a ledger
row whose isotope disagrees with its manifest. The comparison/inventory/queue
suite passed 13 tests; the expanded explicit-result test then passed separately.
