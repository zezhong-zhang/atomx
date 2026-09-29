# Atomx, GRASP and FAC correlation controls — 2026-09-28

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Delivered

48 calculations: four elements (He, Li, Be, Mg), neutral and singly ionized
endpoints, reference and restricted SD spaces, three codes. The comparison is
available at `/comparison/correlation/` in the NIST viewer, linked from the
theory comparison navigation. Its offline counterpart and full numerical
metadata are in `output/correlation_benchmark/matched_dc_v3/`.

These are diagnostic Dirac–Coulomb calculations with a **point nucleus and
infinite nuclear mass**. Breit, QED and recoil are disabled deliberately to
isolate orbital optimization and correlation. NIST values are experimental
context, not an assertion that this Hamiltonian includes every correction.
No results were inserted into the production DHF ledger.

## Results

Ionization energies in eV, formed from independently computed total energies
of both endpoints. “SD” here means the explicitly recorded restricted space,
not a converged active-space limit.

| Element | Atomx reference → SD | GRASP reference → SD | FAC reference → CI | NIST |
|---|---:|---:|---:|---:|
| He | 23.448237 → 24.427714 | 23.448237 → 24.427714 | 23.438649 → 23.448152 | 24.587389 |
| Li | 5.342351 → **5.342301 provisional** | 5.342350 → 5.329173 | 5.228017 → 5.228089 | 5.391715 |
| Be | 8.045780 → **8.595400 provisional** | 8.045780 → 8.619225 | 8.067027 → 8.849450 | 9.322699 |
| Mg | 6.617691 → 7.507126 | 6.617692 → 7.507113 | 6.782590 → 7.190910 | 7.646236 |

- He neutral correlated energies differ by 2.84e-8 Ha between Atomx and
  DHF-seeded GRASP. The differential correlation shifts agree within about
  7.3e-7 eV. The remaining experimental gap is about 0.160 eV.
- Mg SD ionization energies differ by about 1.27e-5 eV between Atomx and
  GRASP. Their absolute energies differ more (roughly 1.4–1.7e-5 Ha), so
  cancellation in the ionization energy must not be mistaken for absolute
  numerical convergence. Point-nucleus radial convergence remains necessary.
- Atomx Li neutral and Be+ each hit 1280 orbital iterations. Their residuals
  are 5.29e-4 and 9.88e-4, respectively, above the requested 1e-5. Both remain
  failed convergence checks even though their energies decrease.
- GRASP Be+ hit the initial 200-cycle cap. Restarting from its output orbitals
  stopped after 85 additional cycles. This passes GRASP's native stopping
  criterion, not an independent orbital-gradient audit. Its result still
  differs from the unconverged Atomx result.
- FAC is a useful additional implementation, but its local central-field
  orbitals are reference optimized and frozen during CI. It is not the same
  variational MCDHF optimization. FAC optimizer stdout is retained; its
  convergence is labeled **unchecked** because no residual criterion has
  been independently parsed. The apparent better Be ionization is not proof
  of a more accurate complete method.

## Matched and unmatched settings

- Every code receives the same explicit nonrelativistic configuration list
  and even parity; target ground J is 0 for even electron counts and 1/2 for
  odd counts. GRASP generates only that J block; Atomx/FAC also retain other
  allowed J blocks, and their ground-state identity is checked.
- He virtual orbitals: 2s, 2p. Li/Be: 2p, 3s. Mg: 3p, 4s, with 1s/2s/2p
  excluded from excitations. Mg core **radial orbitals are still optimized**;
  “frozen core” in the input metadata refers only to excitation restrictions.
- One-electron He+ has no electron correlation and uses the reference space
  for both stages. Redundant empty virtual orbitals are not optimized.
- FAC configuration-energy adjustments (`ConfigEnergy`) are omitted. They
  shift Hamiltonian diagonal elements and would contaminate the plain CI
  control. FAC absolute E0 uses FAC's own Ha/eV conversion constant; level
  table excitation zero is never treated as an absolute energy.
- Native alpha constants are recorded and differ slightly. Atomx fixes the
  positive-energy nuclear Dirac projector; GRASP/FAC projector conventions
  and radial discretizations are not asserted to be identical.
- GRASP SD runs start from converged reference orbitals. An unseeded He run
  reached a different stationary result (~7.38e-5 Ha higher); it is retained
  in the earlier diagnostic directory rather than silently discarded.
- External executables and source revisions are hashed. Every Atomx Python
  source file is hashed, including existing uncommitted optimizer changes.
  The committed HEAD alone does not identify the tested implementation.
- Reported raw timing scopes differ (Atomx solve, GRASP SCF, FAC invocation).
  They must not be used for a speed ranking.

## Reproduction

`benchmark_correlation_codes.py` retains input decks, stdout, configurations,
energies, convergence metadata and source/binary provenance. It automatically
restarts GRASP once after an iteration-limit failure and reports persistent
failure. Use a fresh output directory for a changed model or implementation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_correlation_codes.py --elements He Li Be Mg \
  --output output/correlation_benchmark/new_run
```

The default external paths are `~/code/fac/bin/sfac` and
`/tmp/atomx-grasp-benchmark-build/bin`. Both use the Fortran runtime in
`/tmp/atomx-fortran-tools/lib`. GRASP was built out of source with CMake,
`x86_64-conda-linux-gnu-gfortran`, Release mode, explicit BLAS/LAPACK shared
libraries from that runtime directory, and `-fallow-argument-mismatch` for
legacy Fortran interfaces. No upstream physics source was changed.

The experimental references and uncertainties come from the archived NIST
values in `first20_dhf_error_budget.jsonl`; its hash is in `summary.json`.
Earlier `matched_dc_v1` and `v2` directories are setup diagnostics, **not** the
accepted comparison: v1 used FAC diagonal adjustments and unseeded GRASP;
v2 exposed an invalid redundant one-electron virtual optimization.

## Next accuracy gates

1. Resolve Li/Be+ stationary orbital optimization and test sensitivity to
   starting orbitals in both variational codes. Do not relax residual gates.
2. Compare CI Hamiltonians on imported common radial orbitals to separate
   angular/matrix-element errors from orbital/projector choices.
3. Expand both ionization endpoints systematically in n and l; distinguish
   valence, core-valence and core-core correlation and test higher excitations.
4. Verify radial convergence and common nuclear/alpha settings before adding
   matched recoil, full Breit and QED corrections to correlated states.
   Atomx's resolved-CI Breit currently provides Gaunt, so full Breit matching
   is a remaining implementation requirement.
5. Only then extend this validation to excited levels and full-table spectra.

## Verification

Five benchmark regression tests pass (absolute FAC energy/units, state identity,
missing absolute reference, one-electron limit, and helium configuration space).
A fresh GRASP Be/Be+ rerun reproduced all four reported energies, including the
automatic restart. The report JavaScript passes `node --check`; the live
comparison, report, and JSON endpoints return HTTP 200 with all 48 records.
