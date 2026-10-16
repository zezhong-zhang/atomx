# Native Eu calibration baseline

This audit recovers the successful historical Atomx Eu³⁺ `[Xe]4f6` workflow
without repeating a physical calculation. It reads the saved Tier-A and full-M
crystal-field logs, parses the actual driver's constants, and queries the local
NIST SQLite database in read-only mode. No external solver executes.

The saved three-target calibration reports RMS **8.2 cm⁻¹**, with `s2` fixed at
the Pr prior and `s4/s6` fitted. Recomputing from the one-decimal displayed
energies gives **8.199 cm⁻¹** against the historical targets and **9.030 cm⁻¹**
against the cached NIST bracketed semiempirical levels. Neither comparison is
an independent experimental validation. The script describes its historical
targets as Carnall aquo-ion anchors; their equality to NIST is not assumed.

The unfitted `7F` ladder preserves both its historical target constants and
the current exact NIST rows. Unknown reference uncertainties remain `null`.
The full-M saved control has 3003 determinants and an assembled CF trace
shift of `3.52e-4 cm⁻¹`; this validates the recorded assembly control, not a
measured crystal emission spectrum or complete all-core correlation.

Run from the repository root:

```sh
python3 docs/atomic_native_roadmap_2026_10_11/eu_baseline/audit.py
python3 docs/atomic_native_roadmap_2026_10_11/eu_baseline/audit.py --check
```

The first command refreshes only this directory's `query_snapshot.json` and
`evidence.json`. The second writes nothing and verifies the saved evidence,
source bindings, exact level records and official source identity against
the live inputs. A source change requires a fresh audit; the database as a
whole is not hashed. The query snapshot includes complete selected rows,
stable level IDs, import URL and raw-source SHA, plus a canonical query
payload digest. The evidence's four `inputs` bind three small historical
source files and this snapshot. `audit_identity` separately binds the audit.

Documentation impact: original Eu evidence and scripts are retained intact.
This successor audit supplies machine-readable calibration/holdout/reference
classification for the native roadmap and card renderer. It neither rewrites
the historical successful measurement nor promotes it to an ab-initio claim.

EARS: recover the best recorded native Eu baseline with exact source labels;
read logs/AST and ten precise SQLite level rows; both baseline comparisons
and the saved full-M control were recovered; the next native physics work
must distinguish calibrated success from the remaining correlation and
unfitted-observable gaps.
