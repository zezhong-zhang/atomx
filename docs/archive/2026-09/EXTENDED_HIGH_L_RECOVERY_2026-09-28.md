# Extended angular-momentum configuration recovery

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The standard NIST level inventory contained valid orbital letters beyond `o`
that the configuration parsers rejected. The common configuration parser,
dependency-free archive parser, calculation queue, and DHF orbital labels now
accept `spdfghiklmnoqrtuvwxyz`, while retaining the physical constraint `l<n`.
The Atom entry point also used a separate MCDHF configuration parser; it now
uses the common parser instead of maintaining a shorter alphabet.

## Verification and import

- Eight configuration tests pass, including actual point-nucleus DHF solves
  for `13q` and `20y` at Z=10 against weighted analytic Dirac energies, with
  absolute tolerance 1e-9 Ha.
- Together with both queue-status tests: 10 passed. Existing completed jobs
  are preserved; new jobs can be isolated from workers using older code.
- Production-path pilots for He II `20y`, Ne X `13q`, and Ca XX `20y`
  converged with static DBSR Breit, model QED, isotope nuclear profiles,
  relativistic recoil, and the one-electron mass remainder. Components and
  numerical results are in `output/nist_spectra/high_l_corrected_pilot.json`.
- Inventory regeneration recovered **93 configurations**, **74** within the
  first 20 elements. All 93 completed through a dedicated freshly loaded
  `candidate_pending` worker. These are hydrogenic configurations through
  Z=39. No historical ledger was edited.
- Standard inventory now has 109123 classified rows, 1890 unclassified,
  6128 ionization-limit rows, 844 without evaluated zero-energy references,
  and 1409 without numeric energies (119394 total). Thresholds remain
  excluded from bound-level comparisons.

Report: `output/nist_spectra/extended_high_l_level_recovery.json`.
The rebuilt interactive comparison includes the recovered states. A read-only
ledger snapshot checked 8018 completed configurations, including 1350
one-electron configurations, with zero stored-consistency issues. Maximum
component-sum discrepancy was 7.28e-12 Ha. See
`output/nist_spectra/extended_high_l_ledger_audit.json`.
These configuration-average energies do not resolve individual fine-structure
levels. Recovery is a coverage improvement, not evidence of experimental
accuracy or of an exact QED treatment.

## Separate numerical audit remains open

Update: the specific Sc X rejection below is resolved by the independent
DBSR_HF comparison in `SC_X_NONLOCAL_NODE_AUDIT_2026-09-28.md`. The broader
periodic-table accuracy audit remains unfinished.

The broader suite exposed four historical displaced-orbital branch tests
whose expectations changed after local stationary core-hole rotations became
the default. A controlled old/new rotation comparison is recorded in
`output/nist_spectra/core_hole_guard_mode_audit.json`.
In particular, Sc X `1s2 2s2 2p6 3p1 4p1` from TFDA passes the node guard
with frozen rotations but fails it with stationary rotations, despite SCF
convergence. This is not grounds to relax the state-identity guard or accept
that branch. Its production job was still pending when inspected.

Co II `3d7 4d` and Rb X `3d9 4d` instead change from node-invalid frozen
branches to node-valid stationary branches. Historical regression tests now
explicitly select frozen rotations, consistently with the neighboring Mn and
other original branch tests. This preserves their original guard checks; it
does not establish that all current stationary branches are correct.
All four historical regression tests passed with explicit frozen rotations
(89.54 seconds). The controlled audit also finds the current Sr III branch
node-valid, unlike its frozen-rotation GSZ branch.
