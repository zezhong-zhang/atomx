# Nonlocal exchange continuation — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Reproduced rejection

The corrected shifted-Green residual alone did not recover K's
3p5 4s 26d or Ca's 3p5 4s2 10d configurations. Both grids failed with zero
accepted continuation roots. Candidate diagnostics now expose energies,
overlap, dominant node counts and residuals instead of only the root count.

The requested kappa=2 root was present with the expected node count in both
cases, but its overlap with the unconverged SCF orbital was below 0.9:

| Case | n | Expected dominant nodes | Overlap, 3200 / 6400 |
|---|---:|---:|---:|
| K 26d | 26 | 23 | 0.890930 / 0.890931 |
| Ca 10d | 10 | 7 | 0.856002 / approximately the same |

An unconverged orbital is a mixture of eigenvectors of the new Fock operator.
This fixed overlap gate prevented correction of that mixture. Lowering the
gate would weaken state tracking and was not used.

## Numerical change

For an unconstrained kappa channel, direct continuation remains the first
choice. If root acceptance fails, the incoming dominant node count must still
agree with the requested n,kappa. A local bound-state solve supplies an anchor,
then exchange is introduced through F(lambda)=h_local-lambda K at lambda
0.25, 0.5, 0.75 and 1. The Hellmann–Feynman derivative -<K> predicts each
step's shift. Every step requires the original overlap >=0.9, requested
node count, real bound energy, and shifted-Green residual <=1e-7.

The final operator contains full exchange. Outer SCF field updates, occupied
constraints, undamped-update gates and strict corrections remain in force.
This is a numerical continuation in an unconstrained channel, not a new
physical correlation model or a general no-pair constrained eigensolver.
The node guard remains a dominant-lobe safeguard, not a rigorous proof of
nonlocal orbital identity. Failed continuation remains a failed calculation.

The public numerical implementation returns the intermediate step diagnostics.
SCF results count `nonlocal_exchange_continuations`. The frozen diagnostic
source logs these steps separately; its source hashes are preserved and are
not relabeled as the later main-tree implementation.

## Evidence

- Known Coulomb eigenvalue and rank-one nonlocal exchange shifts pass, including
  an incoming mixed orbital whose overlap with the target is about 0.8575.
- The wrong-principal-state test still rejects the request.
- 17 solver, SCF convergence, warm-start and open-shell checks passed. A later
  15-test run includes the additional rank-one mixed-seed case, phase provenance
  and numerical dependency checks; these test sets overlap.
- Both K 26d grids converged. Corrected energies:
  - 3200: -600.3727598706843 Ha
  - 6400: -600.3727598707416 Ha
  - difference: -5.7298e-11 Ha
- K's weakest exchange-continuation overlap was about 0.965, above the unchanged
  threshold. Its bootstrap and reduced-damping warm refinement both passed.
- The K pair passed identity, phase, correction-accounting and radial-label
  checks. Grid agreement is not a complete uncertainty bound.
- Both Ca 10d grids converged and passed the same pair audit. Corrected
  energies are -678.1420829021761 Ha (3200 points) and
  -678.1420829021888 Ha (6400 points), differing by -1.2733e-11 Ha.

Artifacts: `output/nist_spectra/normalization_diagnostic/` contains
`heavy_root_diagnostic.json`, `heavy_exchange_continuation_probe.json` and
`heavy_exchange_continuation_audit.json`. Frozen source is under
`/tmp/atomx-heavy-exchange-continuation-20260929`.

Main-tree SCF diagnostics also retain the minimum intermediate overlap and
maximum intermediate shifted-Green residual, so new campaign records expose
continuation quality rather than only its call count. Four SCF/phase tests
passed after adding those aggregate fields. The broad older recovery remains
running from its original frozen source; this change does not relabel its
historical successes or failures.

The subsequent frozen main-tree recovery selected 15 previously failed K/Ca
configurations (30 grid attempts). Its first three complete pairs—K 26d,
K 28d and Ca 10d—pass the audit, including the intermediate continuation
overlap and residual gates. This is an in-progress snapshot, not completion
of the 15-configuration recovery or the first20 campaign. Its report is
`normalization_diagnostic/heavy_continuation_recovery_audit.json`.

Final follow-up: all 30 attempts in that frozen recovery have now succeeded,
and all 15 pairs pass the record/phase/identity audit with no flags. The broad
older recovery remains a separate campaign. These calculations predate the
Mohr Dirac-reference normalization fix; their historical correction model and
source identity are preserved.
