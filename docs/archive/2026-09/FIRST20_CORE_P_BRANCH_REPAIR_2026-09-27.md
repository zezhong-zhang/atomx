# First-20 displaced core-p SCF roots

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The corrected-v2 run reproduced a numerical state-assignment failure in
Na II `1s2 2s2 2p5 3p1`. Converged orbital energies and orthogonality did not
establish the intended radial branch:

| Start | Corrected energy (Ha) | 2p nodes | 3p nodes |
|---|---:|---:|---:|
| GSZ | −160.1173006578904 | 1 | 0 |
| Hydrogenic | −156.7140419544885 | 1 | 0 |
| TFDA | −160.54641227119515 | 0 | 1 |

Both j components follow these node patterns at the 1% amplitude cutoff.
The TFDA solution has the expected radial labels and lower energy. The GSZ
result displaced the `2p` and `3p` roots despite being stationary and
orthogonal. Replacing it makes the `3p→3d` configuration gap positive.
Individual multiplet lines remain unscored configuration averages.

Additional controls at Z=10,14,20 for `2p5 3p`, and Z=9,10,14 for `2p4 3p`,
also show lower TFDA branches. **Ne I is an important exception to a simple
core-node rule:** its lower TFDA branch still has an extra 2p outer lobe.
Its 3p node pattern distinguishes the reproduced bad branch, so the new
check excludes Ne I's core-node count. Nonlocal HF does not obey a universal
local-potential node theorem; no blanket node rule was introduced.

## Changes and revalidation

- First-20 core-hole jobs now try TFDA before nuclear/GSZ starts.
- A targeted node check rejects reproduced `1s2 2s2 2p4/5 3p1` wrong roots
  for Z=9–20, with the Ne I exception above.
- The four workers were stopped, verified absent, and their interrupted jobs
  returned to pending. They restarted after the revised checks passed.
- 127 previously completed configurations with a 2p vacancy and a higher p
  shell, solved from a non-TFDA start, were removed from active predictions
  and requeued for independent-start validation. **This does not assert all
  127 were wrong.** Previous energies, components and starts are retained in
  `core_p_revalidation_history` in the v2 ledger.
- Na II `2p5 3p` has already been repaired through the ordinary worker path.
- The 53 generation tests pass, including real Ne I/Na II wrong-root
  rejection and acceptance of the lower branches.

Evidence: `na_ii_3p_branch_probe.json`, `first20_core_p_swap_sample.json` in
`output/nist_spectra/`, and `/tmp/atomx-na-branch-repair.log`.

The restarted worker logs are
`/tmp/atomx-corrected-v2-first20-rootfix-worker{1,2,3,4}.log`.
Only worker 1 refreshes the website. All remain restricted to H–Ca.

## Stored-record audit

`scripts/audit_corrected_ledger.py` reads a consistent SQLite snapshot and
checks component sums, recoil columns, isotope-chain masses, one-electron
two-body zeros, mass-remainder scope, and predicted vacuum wavelengths
against their stored endpoints. At its first snapshot, 1,102 configurations
and 9,444 predictions passed, with a maximum component-sum discrepancy of
2.27e−13 Ha. This audit correctly did **not** detect displaced roots: internal
record consistency is distinct from physical state identity. A corruption
regression test checks that stale wavelengths and inconsistent energy
components are detected. Results are saved in `corrected_ledger_audit.json`.

## Follow-up: compare starts instead of preferring TFDA

The 127-case broader recheck found that TFDA was higher in energy in 98
cases. Therefore a blanket TFDA preference was insufficient. The production
solver now compares TFDA and GSZ for first-20 configurations with a 2p
vacancy and a higher p shell. Each candidate must pass convergence, orbital
energy ordering and the available targeted node checks. Selection uses the
lower **Dirac–Coulomb** energy, the functional actually optimized; additive
Breit/QED shifts do not choose the SCF branch. Rejected starts and accepted
DC energies are retained in `_scf_branch_trials` in the component record.
Additional starts remain fallbacks when neither primary candidate passes.

Two real controls require opposite selections: Na II `2p5 3p` selects TFDA
after rejecting GSZ's displaced roots, while Z=11 `1s2 2s2 2p1 3p1` selects
the lower admissible GSZ solution. Both regression tests pass, as do the
Ne I/Na II node tests. A lower energy among tested starts is not a proof of
a global minimum or complete spectroscopic identification.

152 completed configurations then requiring this comparison were requeued;
their intermediate values are preserved in `core_p_second_start_history`.
This supersedes the initial 127-case revalidation count. Other configurations
retain the previous start policy. The four workers restarted with logs
`/tmp/atomx-corrected-v2-first20-multistart-worker{1,2,3,4}.log`.

### Completed branch comparison

All 152 requeued configurations completed: 113 selected GSZ and 39 TFDA.
Seven candidate starts were rejected. Every stored selected DC energy equals
the minimum among its accepted candidates. No selected energy exceeded the
intermediate TFDA-only result. The read-only audit now enforces the presence
of both primary-start records and the minimum-selection identity for this
scope. Evidence: `first20_core_p_multistart_review.json`.

Four additional single-threaded workers were subsequently added (eight
workers total), with the same Z≤20 restriction. Worker logs 5–8 use the same
`multistart-worker` prefix; only worker 1 refreshes the site. This supersedes
the earlier four-worker count.
