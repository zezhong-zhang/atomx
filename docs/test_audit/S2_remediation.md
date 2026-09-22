# S2 remediation — CI / MCDHF / correlation slice

Slice document: `docs/test_audit/S2_ci_mcdhf_correlation.md` (24 files, 258 tests,
16 findings).  This file records what was changed, what was measured, what was
left open, and — for every gate whose *claim* changed — what it now does and does
not establish.

Ownership constraints observed during the work: `src/atomx/**` was frozen for
this slice by the orchestrator (the Breit migration owns it), so two changes that
require source edits are reported here instead of being made (§4).  No file under
`docs/`, `CHANGELOG.md` or `docs/test_audit/` other than this one was touched.

## 1. Findings → changes

| finding | file:line | change | measured before → after | new gate |
|---|---|---|---|---|
| **F1** | `tests/test_ci_J_block.py:89` | deleted `distinct >= 30`; replaced by the variational floor + oracle-spectrum gates | legacy kernel also has 37 distinct levels → both kernels pass (useless) → exact kernel passes the floor by 9.1e-13 Ha, legacy fails it by **0.7911 Ha** | `test_fe17_ground_state_is_variational` |
| **F1** | `tests/test_ci_J_block.py:111` | deleted `distinct == len(csfs)` (37 == 37, accidental-degeneracy count); replaced by a full 37-level oracle comparison | legacy kernel passes the count; exact spectrum = oracle to **9.1e-13 Ha** | `test_fe17_spectrum_matches_the_mscheme_reference` |
| **F1** | `tests/test_ci_J_block.py:213` | FAC band 200 eV → **10 eV** | exact kernel vs FAC max **5.21 eV**; legacy kernel vs FAC max **33.84 eV** (legacy−exact 29.07 eV) | `test_fe17_first_10_levels_vs_fac` |
| **F1** | `tests/test_ci_J_block.py:114-145` | `_sfac_available()` now *runs* sfac, repairs the documented `FAC_DYLD_LIB` case itself via `gfortran -print-file-name`, then re-probes | `4 passed, 1 skipped` (the FAC test never ran) → `5 passed, 1 xfailed`, FAC comparison executing | — (restores the file's only external comparison) |
| **F2** | `tests/test_sd_space_variational.py:110` (docstring `:104`) | gate made two-sided (`E_ref > E > BE_EXACT_NR`), stale docstring number replaced by the produced value | docstring −14.4369 Ha (0.139 Ha *above* the 1-CSF reference) vs produced **−14.619159** Ha (399 CSFs); gate green either way → now pinned between −14.575892 and −14.66736 | `test_sd_space_is_variational` |
| **F2** | `tests/test_sd_space_variational.py:95` | singles gate gains an upper bound on the recovered correlation energy (`gain < E_ref − BE_DC_LOWER_BOUND`) | measured gain **0.0031 Ha** (Brillouin), bound 0.0965 Ha | `test_singles_are_variational` |
| **F2** | `tests/test_mcdhf_ci_silent_zeros.py:163` | added the variational floor; one-sided `< single − 1e-3` (satisfied by the −15.05 Ha regression) → two-sided | two-CSF Be = −14.6115 Ha ≫ floor −14.67236 Ha | `test_two_csf_be_lies_below_single_csf` |
| **F2** | `tests/test_sd_space_variational.py` (new) | superset monotonicity E(SD) ≤ E(singles) added as a plain gate (SD −14.619159 < singles −14.578968) | audit expected this to be red; measured green, so it gates instead of xfailing | `test_sd_space_is_a_superset_of_the_singles_space` |
| **F3** | `tests/test_ci_matrix_vs_mscheme_reference.py:1-26` | header rewritten: the oracle shares the angular **formula**, the sign convention, the pattern enumeration and the radial integrals; what it validates is the **assembly** | claim "shares no angular code … genuine cross-check" → precise scope statement + the named independent gates | — (documentation) |
| **F3** | `benchmarks/mscheme_ci_reference.py:1-66` | header rewritten with the transcription evidence and the git provenance (`04cd274` has no oracle file; its only commit `2007d36` is a descendant) | "deliberately sharing no angular code" → "independent assembly + algebra tables, NOT independent of the formula" | — (documentation) |
| **F3** | `src/atomx/methods/mcdhf/exact_ci.py:32-37` | **not made** (source frozen, §4) — the same transcription claim lives in its docstring | — | — |
| **F4** | `tests/test_averaging_decoupling.py:51-56` | docstring corrected: the cross-engine gate validates the **radial integrals** and the SCF fixed point (7.3e-8), not the CI matrix, which both engines share | "cannot agree on a wrong matrix" → explicit scope + pointer to where the matrix *is* gated | — (documentation) |
| **F5** | `tests/test_two_electron_replacement.py:210-213, 240-244` | `except AssertionError: pytest.skip` removed; impossible `J` decided up front from the coupling rule (`_equivalent_J2_allowed` / `_split_J2_allowed`), so `_j2_check` failures now propagate | skip set unchanged (5: four `j²`-to-`J=2` pairs + one split case) but a broken oracle can no longer turn into a green skip | — |
| **F5** | `tests/test_two_electron_replacement.py:204-289, 415` | the 25 dead-code element/spectator tests carry `@pytest.mark.snapshot` and a module-docstring section stating that `_two_electron_replacement_element` is reachable only via the unreachable `_build_ci_matrix_approximate` | "two independently assembled implementations, so agreement pins the angular algebra" → labelled diagnostic-path coverage | — |
| **F6** | `tests/test_off_diagonal_j_ci_helike_1s2p.py` | **deleted** (all 6 tests): the five `_off_diagonal_J_correction` probes (`orbitals_dict={}`, `ns=1`, `abs(delta) > 0.0`) and the `_branch9_double_excitation_scalar` anchor all live on the dead legacy chain | 6 tests → 0 | — |
| **F6** | `tests/test_ci_matrix_vs_mscheme_reference.py:379,395` | `2 G^1` claim corrected to `2 G^1/3` with the Gaunt derivation, and the vacuous `splitting > 0.0` replaced by the quantitative identity | `mcdhf` ratio **0.9984** (splitting 0.046870 vs 2G¹/3 = 0.046947); `fac_frozen` ratio **15.2582** → strict xfail, see §3 | `test_he_like_1s2p_j1_splitting_matches_two_thirds_g1` |
| **F7** | `tests/_be_reference.py` (new) | one module for `BE_EXACT_NR = −14.66736` (literature), `BE_DC_LOWER_BOUND = −14.67236` (finite-nucleus bound, ΔE ≈ −2.5 mHa), `BE_HF_LIMIT = −14.573023` (textbook), `BE_HF_1CSF = −14.575892` (atomx's own, measured) and `be_1csf_energy()` (measured live) | four literals (−14.66736 / −14.66736−0.005 / −14.6674 / −14.5734) → one named module; the −14.5734 "1-CSF HF" was 2.5 mHa off the produced value | `test_ci_analytic_slater_condon.py` now compares against the live 1-CSF solve |
| **F8** | `tests/test_mcdhf.py:73` | `−2.86 ± 0.1 Ha` → DBSR_HF Dirac-Fock `−2.86181332` at `abs=1e-6` | band 3.5 % (accepts the hydrogenic −2.8477) → measured deviation 1e-12 | — |
| **F8** | `tests/test_sd_space_variational.py:127` | `eps_2p > −1.50047` → hydrogenic bracket (−2.00047, −0.125) **and** `approx(−0.2286, abs=0.02)` | measured **−0.228585**; a half-collapsed −1.4 Ha virtual used to pass | `test_orbitals_are_screened` |
| **F8** | `tests/test_two_electron_states.py:42` | `rel=0.10` (±90 eV) → `abs=20 eV`, and the *splittings* split into their own passing gate | absolute residuals measured **45–54 eV** → strict xfail (§3); splittings 11.94/9.29 vs FAC 10.95/7.77 eV within 2 eV | `test_helike_ne_singlet_triplet_splittings_match_fac` |
| **F8** | `tests/test_auger_mscheme_ci.py:175` | `rel=0.35` → `rel=0.10` | measured ratio 0.953 (4.7 % out) vs a 35 % band | `test_strongest_level_matches_fac` |
| **F8** | `tests/test_ci_levels_vs_nist.py:135` | `abs=3000 cm⁻¹` → `abs=1200 cm⁻¹` on the one protocol that passes | measured residual **1029 cm⁻¹** | `test_singlet_p_excitation_matches_nist` |
| **F9** | `tests/test_ci_levels_vs_nist.py:115-128` | ordering gate split per protocol: `fac_frozen` plain assertion, default `mcdhf` strict xfail carrying the measured inversion | fac_frozen 13822.5 < 13858.9 < 13930.0 (physical); mcdhf 18044.9 < 18283.9 < 18404.6 (inverted) | `test_three_p_ordering_is_physical_mcdhf` (xfail) |
| **F10** | `tests/test_ci_J_block.py:64`, `tests/test_csf_J_enum.py:34`, `tests/test_auger_mscheme_ci.py:150`, `tests/test_exact_ci_kernel.py:251` | count/generator tests marked `@pytest.mark.snapshot` | 0 → 4 marked; kept out of the physics count | — |
| **F11** | 13 files | `@pytest.mark.snapshot` applied to the D-class tests (0 marks in the slice before) | see §2 counts | — |
| **F12** | `tests/test_ci_grid_kernel.py:204-219` | `_rk == _gk` tautology deleted (both are the same SLAG6 expression); replaced by the exact hydrogenic forms of **both** pairings | `_rk(1s,2p,1s,2p,k) = 112/2187` (G¹) vs the wrong pairing's `8/81` (F¹), 60 %/48 % apart | `test_rk_pairs_its_arguments_as_ab_cd` |
| **F12** | `tests/test_two_electron_replacement.py:258-289` | the delegation assertion `_rk(...) == compute_slater_rk_cell(...)` (identical call, cannot fail) replaced by the pair-role identity `_rk(a,b,a,b) == _gk(a,b)` plus a liveness guard against `_fk` | `_rk == _gk` exactly; `_fk` 18–95× apart for the same orbitals | `test_rk_pairing_convention_bspline` |
| **F13** | `tests/test_auger_mscheme_ci.py:181` | `max(energy)` misnamed "spread" → `max − min`, compared against the FAC spread derived from the fixture | FAC span 18.04 eV, atomx 22.64 eV (within 30 %) | `test_level_energy_spread` |
| **F13** | `tests/test_mbpt_transitions.py:48` | the 0 == 0 "self-consistent" comparison renamed and documented as the exact-zero single-orbital limit; the non-zero case is explicitly declared open | measured cost of the non-zero case: > 4 min (Be 2s1/2), > 8 min (Ne 2p) → not viable in this 4 s file, §4 | — |
| **F13** | `tests/test_two_electron_replacement.py:397` | `np.all(np.isfinite(...))` → every parity-forbidden matrix element exactly `0.0`, with a non-vacuity guard | passes with the parity filter *removed* before → exact-zero after | `test_pair_channel_does_not_couple_incompatible_parity` |
| **F13** | `tests/test_ci.py:52-54` | `ground == min(eigenvalues)` (true of any `eigh`) → the CI total must equal the DHF total on the same orbitals | measured agreement to `rel=1e-6` | `test_ci_solve_returns_ciresult_and_ground_energy_is_minimum` |
| **F13** | `tests/test_ci_J_block.py:219-241` | the `n_states`/`energy < 0` pair replaced by the closed-form point-nucleus Dirac energies of the 1s and 2s levels | measured 1.2e-12 / 5.5e-10 Ha from the closed form; the 2p levels are 0.017 Ha off → strict xfail (§3) | `test_J_block_lift_h2plus_n2_p_states`, `test_one_electron_2p_levels_match_the_dirac_closed_form` |
| **F13** | `tests/test_mcdhf.py:75-101` | 2-CSF Be: every level above the variational floor, and strictly below its own 1-CSF reference; excitation test gates the SI↔cm⁻¹ unity identity | 2-CSF −14.591898 vs 1-CSF −14.575892 (16.0 mHa gain) | — |
| **F14** | `tests/test_csf_recoupling.py` | module docstring rewritten (no production caller; the live `allowed_coupled_J2` is untested here) + module-level `pytestmark = pytest.mark.snapshot` | 42 tests, 0 marks → 42 snapshot-marked, 3 of them named as in-test brute-force anchors | — |
| **F15** | `tests/test_ci_line_strengths.py:235-236` | brackets `0.7–1.0` / `1.0–1.7` (both contain the no-mixing value 1.0) → `approx(0.820, abs=0.05)` / `approx(1.397, abs=0.05)` + snapshot mark | no-mixing 1.0 now fails by 3.6σ / 7.9× the tolerance | `test_one_electron_consistency` |
| **F16** | see F13 (MBPT) | open; the helper's cost is the blocker | — | — |

## 2. Verification (exact counts, this machine, `-p no:randomly`)

| file | before | after |
|---|---|---|
| `test_ci_J_block.py` | 4 passed, 1 skipped | **5 passed, 1 xfailed** (62 s) |
| `test_sd_space_variational.py` | 4 passed | **5 passed** (382 s) |
| `test_mcdhf_ci_silent_zeros.py` | 8 passed | 8 passed |
| `test_two_electron_replacement.py` + `test_ci_grid_kernel.py` | 25 p / 5 s + 16 p | **42 passed, 5 skipped** (143 s) |
| `test_ci_matrix_vs_mscheme_reference.py` | 14 passed | 15 collected: **14 passed, 1 xfailed** (see §5 log) |
| `test_averaging_decoupling.py` + `test_mcdhf.py` | 14 + 18 passed | **32 passed** (127 s) |
| `test_ci_levels_vs_nist.py` + `test_ci.py` | 5 + 2 passed | **5 passed, 3 xfailed** (8 s) |
| `test_two_electron_states.py` | 9 passed | **5 passed, 4 xfailed** (47 s) |
| `test_auger_mscheme_ci.py` | 5 passed | 5 passed (4 s) |
| `test_mbpt_transitions.py` | 4 passed | 4 passed (4 s) |
| `test_ci_analytic_slater_condon.py`, `test_mscheme.py`, `test_csf_recoupling.py` | 4 + 12 + 42 passed | **66 passed** (36 s, with `test_mcdhf_ci_silent_zeros.py`) |
| `test_ci_line_strengths.py` | 14 passed | 14 passed (9 s) |
| `test_exact_ci_kernel.py` + `test_csf_J_enum.py` | 15 + 4 passed | 19 passed (4 s) |
| `test_off_diagonal_j_ci_helike_1s2p.py` | 6 passed | **deleted** |

`~/miniconda3/bin/ruff check` on all 20 touched/created files: **All checks passed.**
(`tests/_be_reference.py` is a new file; ruff placed it in the third-party import
block alongside `pytest`, which is its configured behaviour for `tests/`.)

## 3. Tightening exposed three genuine red gates (all converted to strict xfail)

1. **`fac_frozen` breaks the LS identity by 15.3×**
   (`test_he_like_1s2p_j1_splitting_matches_two_thirds_g1[fac_frozen]`):
   J=1 splitting 0.00187569 Ha vs 2 G¹(1s,2p)/3 = 0.00012293 Ha on the same
   orbitals (G¹ = 1.844e-4).  The identity needs one radial 2p for both J=1
   CSFs; `fac_frozen` gives 2p1/2 and 2p3/2 independent 0.001-occupation radial
   functions in the frozen reference potential.  The `mcdhf` protocol gives
   ratio 0.9984 (splitting 0.046870 vs 0.046947 Ha).  Same protocol family as
   the inverted fine structure in finding F9.
2. **H-like 2p is 0.017 Ha off the closed-form Dirac energy**
   (`test_one_electron_2p_levels_match_the_dirac_closed_form`): 2p3/2
   −0.10805594, 2p1/2 −0.10675280 Ha vs closed forms −0.12500042 / −0.12500208
   Ha, with the fine-structure ordering inverted.  Identical for
   `max_iter = 3, 10, 40`, so not an SCF convergence cap; the same solve
   reproduces 1s (−0.500006656596) and 2s (−0.125002080739) to 1.2e-12 /
   5.5e-10 Ha, and the SCF's own 2p orbital energy (−0.361613 Ha) disagrees
   with its CI eigenvalue by 0.25 Ha.  Points at the one-electron n=2 p radial
   path, not the angular kernel.
3. **The He-like Ne term energies are 45–54 eV below FAC**
   (`test_helike_ne_term_energies_vs_fac`): the model is hydrogenic plus
   first-order Coulomb and carries no Z=10 screening.  The gate was tightened
   from `rel=0.10` (±90 eV) to `abs=20 eV`; the orbital-insensitive
   singlet–triplet splittings are green and now have their own gate.

## 4. Not fixed, and why

* **`src/atomx/methods/mcdhf/exact_ci.py:32-37` (F3a)** — the docstring still
  describes `two_body_angular` as independently derived.  `src/**` is frozen for
  this slice (Breit migration), so the correction is recorded here: the text
  should say the formula is a transcription of
  `benchmarks/mscheme_ci_reference._angular_terms` (same `q2 % 2` guard, same
  `kmax`, same phase exponent, same term order) and that only the sympy algebra
  tables and the assembly differ.
* **`src/atomx/methods/mcdhf/matrix.py:952, 973-981`** — `_check_orbitals_present`
  and `_get_orb` still document "the angular-only unit-radial probes" as the
  reason an empty `orbitals_dict` is tolerated by `_get_orb`.  Those probes were
  the tests deleted under F6; the comments are now stale but `_get_orb`'s
  tolerance must stay (the surviving element-level tests call
  `_two_electron_replacement_element` with an empty map).  Source edit → report
  only.
* **F14 (test_csf_recoupling.py)** — the library is still dead code.  The file is
  now honestly labelled (`pytestmark = pytest.mark.snapshot` + docstring), but
  the remediation that would make it *evidence* — re-deriving each literal
  through an in-test brute force, or wiring the recoupling helpers into a live
  path — was not attempted.  Two one-body tests already use that pattern.
* **F16 (MBPT(2) transitions)** — the vacuous 0 == 0 comparison is gone, but the
  replace-with-real-physics remediation is blocked by cost:
  `mbpt2_orbital_energy_shift` builds a full virtual basis per kappa and takes
  **> 4 min for Be 2s1/2** and **> 8 min for Ne 2p** (measured, killed after
  those times), against a 4 s file.  Fixing the helper's cost is a source change
  outside this slice.  The test docstring records the gap and the numbers.
* **`tests/test_mbpt_transitions.py` unit-conversion check** — reduced to the
  exact-zero case, which makes the `HARTREE_EV` identity trivial.  It becomes
  meaningful together with F16.
* **F10/F11 residue** — the audit's D-class lists for
  `test_ci_levels_vs_nist.py:112`, `test_mscheme.py:54,63,94,97,101`,
  `test_averaging_decoupling.py:113,135,156,157,169`, `test_ci_J_block.py:73`
  and `test_sd_space_variational.py:86` are marked (13 files touched); I did not
  re-derive every literal.  `test_ci_levels_vs_nist.py:153,185` keep their
  `abs=3000 cm⁻¹` inside strict xfails — tightening an xfail tolerance cannot
  change its outcome, so it was left as documentation of intent.
* **`docs/plans/scattering_gap_closeout_plan.md:128`** now references the
  deleted `tests/test_off_diagonal_j_ci_helike_1s2p.py`; that plan doc is
  historical (HYP-184/189) and was left untouched.

## 5. What the m-scheme oracle does and does not establish

`benchmarks/mscheme_ci_reference.py` is the arbiter for the numbers in
`tests/test_ci_matrix_vs_mscheme_reference.py` and `tests/test_ci_J_block.py`.
After the relabelling in this pass, its scope is:

**It does establish**, on identical orbitals and with the identical nuclear
potential:

* the **CSF-to-determinant map** — that each CSF is expanded into the
  determinant space it claims, with the right fermion signs (`_apply_ops` vs the
  production `_ops_sign`; the *convention* is shared by design, the *application*
  is not);
* the **projection** `H = C^T H_det C`, including symmetry
  (`test_ci_matrix_is_symmetric`), block-diagonality in `(2J, parity)`
  (`test_no_coupling_between_different_total_j`), determinism, and the full
  sorted **spectrum** rather than the ground state alone;
* the **radial-integral call convention** — which orbitals are the bra/ket pairs
  of each `R^k` — and, through
  `test_oracle_is_rotationally_invariant`, that the arbiter itself is a scalar
  operator (levels must not depend on the `2M` sector);
* for Fe XVII, that the production CI is **variational**: its ground state is
  **9.1e-13 Ha** above the oracle's exact minimum over the same space, where the
  removed kernel sat **0.7911 Ha below** it.

**It does not establish**:

* the **angular coefficient formula**.  `exact_ci.two_body_angular` is a
  line-by-line transcription of the oracle's `_angular_terms` — a sign or factor
  error there is invisible to every comparison in these two files;
* the **radial integrals**: `_h_one_body`, `_norm_sq`, `_rk` are the same
  functions on both sides, so only their *usage* is checked;
* the **occupation patterns**, which come from the repo's own
  `enumerate_jj_csfs` (the oracle checks the dimension, not the physics, of the
  enumeration);
* any **independence from history**: the oracle file does not exist at the fix
  commit `04cd274` although the tests that commit added import it, and its only
  commit is a descendant — so "the oracle agrees" cannot be read as pre-existing
  external evidence.

The primary evidence for the CI Hamiltonian is therefore, in order:
`tests/test_ci_grid_kernel.py::test_hydrogenic_slater_integrals` (exact
Condon–Shortley closed forms), `tests/test_ci_analytic_slater_condon.py`
(`E(1P1) − E(3P1) = 2G¹/3`, orbital-free), the
variational/Hylleraas–Undheim–MacDonald ladder in
`tests/test_first_principles_invariants.py` (owned by the orchestrator), and the
external NIST/DBSR_HF level comparisons in `tests/test_ci_levels_vs_nist.py` and
`tests/test_nist_levels.py`.  The oracle gates are kept — they are the strongest
*assembly* gates in the slice — but they are no longer described as validating
the angular algebra.
