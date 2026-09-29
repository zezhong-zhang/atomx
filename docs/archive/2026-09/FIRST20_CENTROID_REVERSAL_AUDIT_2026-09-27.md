# Corrected first-20 centroid reversal audit

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The v2 ledger snapshot initially contained 644 observed line rows whose
configuration averages had reversed energy order. Complete evaluated NIST
centroids support the DHF ordering for 88 rows (14 pairs); 482 rows (30 pairs)
lack complete evaluated endpoint manifolds. Two pairs (74 rows) disagreed:
Ar II 3p4 4p–3p4 5s and Al I 3s2 4f–3s2 4d.

## Reproduced Ar II branch error and repair

Independent production-grid DHF calculations with identical Breit, QED and
recoil corrections gave, for 1s2 2s2 2p6 3s2 3p4 4p1:

| Start | Energy (Ha) | Ordered orbitals | Core 3p nodes |
|---|---:|---|---:|
| GSZ | -527.0611093049918 | yes | 2 |
| TFDA | -527.1630360887706 | yes | 1 |
| Hydrogenic | -523.7878760646295 | no | 2 |

The accepted lower Dirac–Coulomb branch is TFDA. The 5s endpoint independently
agrees across all three starts within 5.1e-10 Ha. The corrected 4p–5s
configuration gap is 17333.3822 cm^-1 versus NIST 24214.8144 cm^-1: the
wrong ordering is repaired, but a significant physical discrepancy remains.
The 4p excitation is 163934.2627 versus NIST 164419.1922 cm^-1.

The old production row is preserved in `ar_ii_4p_branch_history`. The corrected
row includes all trial starts; dependent line statuses and wavelengths were
recomputed atomically. The production solver now compares both primary starts
for this specific reproduced case. The read-only ledger audit verifies selection
of the minimum accepted Dirac–Coulomb energy. Other 3p vacancies are not yet
covered by this added condition, and already running workers retain their loaded
code; completed related states require a separate audit.

## Remaining work

Al I 4f and 4d reproduced their stored branches from three independent starts;
the discrepancy is not explained by the starting potential. This is not a
proof of basis convergence or complete correlation treatment.

Newly completed K II 3p5 4p–3p5 5s also reverses the NIST centroid ordering.
Independent endpoint probes are running. The K II 4p endpoint already reproduces the same branch defect: GSZ -600.067067855823 Ha versus TFDA -600.2519182531456 Ha, with core 3p nodes changing from two to one. The hydrogenic result fails orbital ordering. The 5s endpoint agrees across all starts within 6.8e-10 Ha. The 4p record is now repaired, with the old row in `k_ii_4p_branch_history`; dependent lines were recomputed atomically. Production branch selection and the audit now cover this independently reproduced K II case too. This motivates reviewing neighboring
3p-vacancy configurations before releasing the full-table queue.

Artifacts: `first20_inversion_branch_probe.jsonl`,
`first20_corrected_discrepancy_review.json`, and
`first20_corrected_discrepancy_after_ar_repair.json` under `output/nist_spectra`.
These are evolving calculation snapshots, not final first-20 completeness reports.

## Wider audit and interactive presentation

Two calculation processes are checking GSZ and TFDA starts for the 48 completed
first-20 configurations having a filled 2p core, a 3p vacancy and an outer p shell.
Outputs are `first20_3p_branch_audit_0.jsonl` and `_1.jsonl`. These probes do not
modify the production ledger. The corrected centroid section now displays
configuration ordering checks for the selected element, separately identifying
complete-reference disagreements and unresolved comparisons.

The explicit coverage report (`scripts/review_first20_3p_branches.py`) counts
**53** configurations in this family across the full H–Ca queue; the initial
48-state probe snapshot therefore does not establish full coverage. Resume logic
now reads prior trials from both output shards and uses a stable hash of the
configuration identity for assignment, so newly completed production states do
not shift job assignment. Do not launch new shard processes until the original
two processes finish: they retain their original assignment in memory.

Initial results confirm that the lower starting field varies by configuration.
The report flags distinct converged, ordered solutions and node-count changes;
it does not promote the lowest energy automatically. Independent tighter
`tolerance=1e-8` calculations of S I 3p3 5p and Si I 3p 5p are checking whether
the start dependence survives a stricter SCF threshold. Their results are in
`first20_3p_tight_tolerance.jsonl`.

## Tight-tolerance result

All four strict calculations converged with residual below 4e-9.

Z=16: TFDA minus GSZ total energy = -0.000964398047927 Ha.

Z=14: TFDA minus GSZ total energy = 0.000108064116318 Ha.

The distinct energies survive the 1e-8 SCF threshold; simply tightening
the stopping criterion does not resolve this starting-field dependence.
State identity and orbital stationarity still require review before promotion.
