# PHYSICS — H-like 2p and the spectator mean-field row (2026-09-23)

## Symptom

Two suite gates could not be satisfied by the same code:

* `tests/test_ci_J_block.py::test_one_electron_2p_levels_match_the_dirac_closed_form`
  — H-like Z=1, 2p₁/₂ at −0.3616 Ha instead of the exact −0.1250 Ha,
  fine-structure ordering inverted. Strict xfail at HEAD.
* `tests/test_sd_space_variational.py::test_orbitals_are_screened`
  — Be `correlation="s"` 2p₁/₂ at the hydrogenic Z=4 value −2.0005 Ha
  (Guerra population collapse, pre-R7), repaired by R7 to −0.228585,
  but the repair (a `q_ref − 1` diagonal) is what broke H-like.

## Defect 1 — the self weight was a pure −J_i anti-screening

`DFSolver._get_exchange_coef` returns 0.0 structurally for `i == j`
(same-subshell exchange is folded into the k>0 "direct" coefficients),
and the direct self density is doubled at `j == target_i`
(DBSR_HF `c = c + c` in `_accumulate_direct_density_batch`). A self
weight `w` therefore contributes `w × (+J_i)` with **no** compensating
`+K_i`. The historical R7 row used `coeffs[i,i] = q_ref − 1` (−1 for a
virtual absent from the reference), i.e. pure −J_i anti-screening.

Linear-response proof on H-like Z=1 2p, hand-set seam rows on a
4-subshell solve (`/tmp/seam_convention2.py`):

| self weight | ε(2p) error |
|---|---|
| 0 | exact to 5.4e-10 Ha |
| −0.5 | −0.105 Ha |
| −1 | −0.237 Ha (2p₁/₂ → −0.3616, ordering inverted) |

The correct self contribution of a spectator absent from the reference
is identically zero.

## Defect 2 — the capped reference over-screens a valence spectator

With the diagonal fixed to 0, the capped-reference row
`min(q_ref_j, max_{c ∈ parents} q_j^c)` = `[2, 2]` for Be
(`tests/test_sd_space_variational.py` spectators; parents
`1s²2s¹2p¹` / `1s¹2s²2p¹`) still failed: `UnboundSubshellError` —
the frozen neutral-core field does not bind 2p in the box.

Frozen-field eigenvalue experiment (`/tmp/be_selfterm.py`,
`/tmp/be_rowsweep.py`): converge the no-seam tiny-occupation Be solve
(2p at −0.228, physical), freeze its core orbitals, diagonalize the
κ = +1 Fock built by `_build_hf_matrix_dbsr` for hand-set rows. Harness
validated by reproducing the no-seam Guerra eigenvalue −0.228366
exactly.

| row (1s, 2s), self | κ = +1 bound 2p ε |
|---|---|
| (2, 2), 0 — capped reference | **+0.006 (unbound)** |
| (1.5, 1.5), 0 — equal-prior conditional | −0.300 |
| (1.5, 2), 0 | −0.143 |
| (2, 1), 0 — parent core (2s hole) | **−0.212** |
| (2, 2), −1 — old R7 | −0.229 |

The neutral 1s²2s² reference over-screens the 2p by ~0.2 Ha: the
configurations the spectator actually lives in have promoted a 2s
electron away. Independent confirmation from FAC's own frozen-core
protocol (`OptimizeRadial` on the ground group only): 1s²2s²2p₁ sits
at 1.708 eV — i.e. ε(2p) ≈ −0.063 Ha, the same barely-bound state
(discretization shifts it to +0.006 in the B-spline box). The suite pin
−0.2286 ± 0.02 is the *relaxed* valence value, which the −J_i accident
mimicked for Be while being pure error for H-like (whose zero row has
nothing to compensate).

## The fix — parent-core conditional rows

For a spectator target `i` (Guerra population ϖᵢ = 0, absent from the
reference):

* `Sᵢ = {c : qᵢ^c > 0}` over **raw** occupations (the jj-partner floor
  would invent phantom parents whose core mirrors the reference and
  drag the row back toward over-screening).
* Among parents with the fewest holes vs the reference, keep those
  whose deepest hole is shallowest (`(n, l)` of the removed shell) —
  the valence-promotion channel dominates. Ties average.
* `coeffs[i, j] = ⟨q_j^c⟩_c` for j ≠ i; `coeffs[i, i] = ⟨qᵢ^c(qᵢ^c − 1)⟩_c`
  (conditional factorial moment = same-shell partner count);
  `normalization[i] = 1`.
* Fallback when no raw parent exists (jj-floor phantom presence only):
  the R7 capped reference.

Anchor cases, all reproduced exactly:

| case | parent(s) | row | result |
|---|---|---|---|
| H-like 2p | `2p¹` (hole at 1s) | zero | exact Dirac to 5.4e-10 Ha |
| He 1s²+1s2p | `1s¹2p¹` | `[1]` | ε(2p) = −0.125; 1s2p J=1 splitting 0.0469 Ha (test_ci_analytic_slater_condon) |
| Be corr="s" 2p | `1s²2s¹2p¹` type | `[2, 1]` | ε(2p₁/₂) = −0.212143 (bracket + pin), all spectators bound |

The self term reduces to the FAC-calibrated occupied-shell value
`q(q−1)` for parents that doubly occupy the spectator's shell, and to 0
for single-occupancy parents — one formula for both regimes.

## Side effect — Fe XVII KLL panel ILEV 32

The spectator-row change moved the Fe XVII KLL Auger panel
(`tests/test_jj_level_auger.py`): ILEV 32 ratio 0.097 → 0.102, i.e.
into the [0.1, 1.1] band it had missed by 3.4 % under the capped row
(the 2026-09-20 strict xfail documented that miss as an Auger-module
open item; the row was the residual defect). New panel: 0.102 / 1.007 /
0.236 / 0.412 / 0.936 × FAC (spread 9.9×, was 10.4×). The envelope
xfail is lifted to a hard gate; the < 2× spread gate stays open.

## Side effect — three more strict xfails were the same row

The final full suite flipped two further strict xfails, both re-measured
and lifted to hard gates (2026-09-23):

* **H I MCDHF, "45× worse than DHF"** (`tests/test_nist_levels.py::
  test_h1_mcdhf_orbitals_are_screened`). Old: 2p −0.361624, 3d −0.163935
  Ha against hydrogenic −0.125 / −0.0556 (Z_eff ≈ 1.7 in neutral
  hydrogen); the note's "_collapsed_targets is empty, so this is not the
  R7 seam" snapshot caught an iteration where no orbital had collapsed.
  Once CI weights collapse mid-solve, a one-electron orbital lands in
  the seam branch, and the old capped-reference row handed it the full
  +J of every other orbital in the model — five phantom electrons. New:
  all six orbitals within 2e-6 of hydrogenic (2p₁/₂ −0.12500208),
  because a one-electron parent's parent-core row is the zero row by
  construction — the mean field of a one-electron atom is identically
  zero, as it must be.
* **He-like Z=2 ground E1 gauge ratio**
  (`tests/test_gauge_invariance.py::TestHeLikeGroundTransitionGaugeRatio`).
  Old: |A_v/A_l − 1| = 0.234 against the Gu 2008 §3.2 threshold 0.10,
  blamed on "the He 2p radial function diving to a bare-nucleus
  hydrogenic solution". New: 0.0868. The 2p spectator sees its own
  parent core (1s¹) now; the gauge error was the same row defect, not a
  separate orbital problem. Threshold kept at the literature value
  (0.0868 is 87 % of it).

(The Be 1s²2s2p fine-structure ordering lift is documented in
`tests/test_ci_levels_vs_nist.py` — 13 872.9 / 13 874.7 / 13 878.3 cm⁻¹,
now physical.)

## Evidence paths

* `/tmp/seam_convention2.py` — H-like self-weight linear response.
* `/tmp/be_selfterm.py`, `/tmp/be_rowsweep.py` — frozen-field row sweep
  (harness validated against the no-seam SCF eigenvalue).
* `/tmp/be_frozen_field.py` — tiny-occ vs seam standalone solves.
* `/tmp/be_fac/` — FAC frozen-core deck (1.708 eV) and level file.
* `/tmp/fe17_panel.py` — post-fix panel measurement.
