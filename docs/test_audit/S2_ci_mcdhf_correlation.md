# S2 — CI / MCDHF / correlation / MBPT (the correlation ladder)

Slice: the 24 files listed in the task (all exist; none skipped, none added).
258 collected tests, 5 066 lines. Method: every test body read, the exercised
source read (`src/atomx/methods/ci.py`, `src/atomx/methods/mcdhf/{matrix,exact_ci,solver,config,excitations}.py`,
`src/atomx/angular/{mscheme,csf_recoupling}.py`, `benchmarks/mscheme_ci_reference.py`),
`git log -S` provenance on every hardcoded literal, and targeted runs/measurements
(no test file modified). Measured commands are quoted with their numbers.

## 1. Verdict

After the N1 rewrite the **kernel-level** half of this slice is genuinely good:
`test_exact_ci_kernel.py` and the "stronger gates" section of
`test_ci_matrix_vs_mscheme_reference.py` compare the production angular kernel
against sympy 3j algebra, compare the Slater–Condon term table against a
brute-force second-quantised determinant Hamiltonian on synthetic radial data,
and compare the **whole eigenvalue spectrum** (not just the ground state) of the
CSF matrix against an independent m-scheme construction on identical orbitals;
`test_ci_analytic_slater_condon.py` pins the one orbital-independent identity
available for a two-electron `ns n'p` block (`E(¹P₁) − E(³P₁) = 2G¹/3`, which I
re-derived independently and confirmed numerically: measured ratio 0.9984 at
Z=2); `test_mbpt2.py` sandwiches He between the HF and exact bounds. Everything
else is weaker than it looks. The three "headline acceptance" gates in
`test_ci_J_block.py` are counting assertions that **pass unchanged on the
physically wrong kernel the repo removed** (measured: Fe XVII ground
−1149.2973 Ha for the legacy kernel vs −1148.5051 Ha exact, 0.79 Ha
non-variational, yet distinct-level counts 37 ≥ 30 and 37 == 37, and the
level-by-level difference is ≤ 29.1 eV inside a 200 eV FAC band). The
variational gates are one-sided: `test_sd_space_variational.py` asserts only
`E > −14.6674`, and its own recorded value (−14.4369 Ha, 0.14 Ha **above** the
1-CSF reference) is not what the code produces today (measured −14.619159 Ha),
so a catastrophic variational failure in the opposite direction is green. 24 of
the 30 element-level tests in `test_two_electron_replacement.py` and all 6 tests
in `test_off_diagonal_j_ci_helike_1s2p.py` exercise helper functions that no
production path reaches, and several assert nothing more than "non-zero".
The m-scheme oracle is real but less independent than its docstring claims and
was authored after the fix it certifies. Slice mix: **A71 B19 C55(≈36
degenerate/dead) D35 E25 F53**; 0 of the D-class literals carry the
`@pytest.mark.snapshot` marker that the repo defines for exactly this.

## 2. File table

| file | tests | class mix | worst | note |
|---|---:|---|---|---|
| test_ci.py | 2 | E1 C1(deg) | S3 | smoke test of `ci_solve`; the "matches first macro-iteration" test runs the same kernel on both sides (docstring: "bit-for-bit"). |
| test_ci_J_block.py | 5 | B1 D1 E3 | **S1** | all three "physics" gates pass on the removed legacy kernel (measured); the FAC gate is skipped here and 200 eV wide anyway. |
| test_ci_grid_kernel.py | 16 | A7 C5(4 deg) D1 E1 F2 | S2 | exact-rational hydrogenic Slater integrals are the model A gate; the CI-vs-SCF gates share `_slater_integral_slag6` by their own admission; `_rk == _gk` is a tautology. |
| test_ci_levels_vs_nist.py | 5 | B3 A1 D1 | S2 | genuine NIST anchor (±3000 cm⁻¹ = 7 %); the fine-structure-ordering gate runs only on the protocol where it passes. |
| test_ci_line_strengths.py | 14 | B4 A4 F6 | S2 | best external anchors in the slice (FAC fixture + A↔gf identity + trace sum rule); the per-line mixing brackets are re-pinned ranges. |
| test_ci_analytic_slater_condon.py | 4 | A4 | — | the only fully analytic Hamiltonian gate in the slice; identity independently re-derived and confirmed. |
| test_ci_matrix_vs_mscheme_reference.py | 14 | A6 C7 D1 | S1 | strongest gates in the slice (full spectrum, rotational invariance, Brillouin), but the oracle shares the angular formula and postdates the fix. |
| test_exact_ci_kernel.py | 15 | C8 A2 D1 F4 | S3 | best-designed kernel gates: sympy 3j, brute-force second quantisation on monkeypatched synthetic integrals. |
| test_mcdhf.py | 18 | F11 E4 A2 D1 | S2 | mostly parsing/bookkeeping; He HF energy pinned as −2.86 ± 0.1 Ha (3.5 %). |
| test_mcdhf_ci_silent_zeros.py | 8 | A3 F4 E1 | S1 | M7/M8 guards are good; the end-to-end "correlation lowers the energy" is one-sided (the −15.05 Ha regression satisfies it). |
| test_sd_space_variational.py | 4 | A3 D1 | **S1** | the variational gates; one-sided, and the recorded SD number is stale by 0.18 Ha. |
| test_averaging_decoupling.py | 14 | F8 D4 C2 | S1 | the cross-engine gate validates the radial integrals, not the CI matrix, contrary to its docstring. |
| test_two_electron_replacement.py | 30 | C24 D3 E3 | S1 | 24 element-level tests hit the dead legacy helper; `except AssertionError → pytest.skip` swallows the oracle's own J² check. |
| test_two_electron_states.py | 9 | B5 A1 D1 F2 | S2 | `rel=0.10` on ~900 eV excitation energies = ±90 eV. (Scattering-module file; belongs to the scattering slice.) |
| test_two_electron_angular.py | 4 | A3 C1(deg) | S3 | three analytic reduced matrix elements; the fourth compares two functions of the same module. |
| test_off_diagonal_j_ci_helike_1s2p.py | 6 | E5 D1 | S4 | 6 tests of dead legacy code; the headline analytic anchor is stated wrong by 3× and never computed. |
| test_csf_recoupling.py | 42 | A13 D11 F8 C6 E4 | S4 | 733 lines over a helper library with no production caller; 3 tests compare against a real brute-force evaluation. |
| test_csf_angular.py | 9 | A5 D3 E1 | S3 | selection rules are A; the FAC UTA population-factor literal has no derivation. |
| test_csf_J_enum.py | 4 | B2 A1 D1 | S3 | Grant App. A CSF counts — combinatorial content, no Hamiltonian content. |
| test_mscheme.py | 12 | A7 D3 C1 E1 | S3 | derived J content and particle–hole symmetry are model A; the ±√2 literals are unprovenanced. |
| test_davidson.py | 6 | A2 F4 | — | LAPACK reference + eigen-residual; clean, no findings. |
| test_mbpt2.py | 8 | A6 B1 F1 | — | the variational sandwich is the model for the whole suite. |
| test_mbpt_transitions.py | 4 | A1 F2 E1 | S4 | the "self-consistent" test compares 0.0 with 0.0. |
| test_auger_mscheme_ci.py | 5 | B3 D1 F1 | S2 | FAC AITable trace/top at 20 %/35 %; the "spread" test computes `max`, not `max − min`. |

Aggregate: A 71 (27.5 %), B 19 (7.4 %), C 55 (21.3 %, ≈36 degenerate or on dead
code), D 35 (13.6 %), E 25 (9.7 %), F 53 (20.5 %). Independent-gate content
(no shared kernel, number not produced by atomx): **90 tests**, of which the
CI-Hamiltonian-specific ones are the ~20 in `test_exact_ci_kernel.py` +
`test_ci_analytic_slater_condon.py` + the oracle gates.

## 3. Findings

### F1 (S1) — Fe XVII "acceptance gates" pass on the kernel the repo removed for being physically wrong
`tests/test_ci_J_block.py:89` (`distinct >= 30`), `:111` (`distinct == len(csfs)`),
`:213` (`|diff| <= 200 eV` vs FAC).

Measured by monkeypatching `matrix.build_ci_matrix = matrix._build_ci_matrix_approximate`
and solving the file's own `FE17_CONFIGS` with `Correlation(j_resolved_csfs=True)`:

| kernel | ground (Ha) | distinct @1e-3 eV | distinct @1e-6 eV | first 10 levels (eV) |
|---|---:|---:|---:|---|
| exact (N1) | −1148.5051464878989 | 37 | 37 | 0, 728.9, 730.9, 741.9, 743.2, 759.1, 762.7, 764.4, 765.5, 767.3 |
| legacy (`_build_ci_matrix_approximate`) | −1149.2972523479575 | 37 | 37 | 0, 752.5, 752.9, 766.1, 766.5, 780.3, 785.4, 788.1, 791.2, 796.4 |

The legacy kernel is 0.79 Ha below the exact one (non-variational, the same
defect class as the Be SD −15.05 Ha), yet both gates pass; the largest
level-by-level difference is 29.1 eV, 6.9× inside the 200 eV FAC band. The
remaining two tests in the file assert `n_states == len(csfs)`,
`state_energies[0] < 0` and nothing else (`:239`-`:241`), and the FAC test is
skipped in this environment (`_sfac_available()` returns False even though
`/usr/local/bin/sfac` exists and is executable — measured: `4 passed, 1 skipped`).

**Remediation.** Replace the count assertions with the two things the file is
supposed to prove: (a) an oracle comparison — the same Fe XVII CSF list run
through `benchmarks/mscheme_ci_reference.mscheme_levels` must give the same
spectrum to 1e-9 (the machinery already exists in
`test_ci_matrix_vs_mscheme_reference.py`); (b) a variational bound relative to
the closed-shell reference, `E(Fe XVII) < E(1s²2s²2p⁶) ≤ E(any CI)`, plus a *level
ordering* claim (2p⁵3s ³P₁ below 2p⁵3d ¹P₁ etc.). Delete `distinct >= 30` and
`distinct == len(csfs)`; the second is an assertion about accidental degeneracies
in a 37×37 matrix and cannot fail for a physics reason.

### F2 (S1) — `test_sd_space_is_variational` is one-sided and its recorded value is stale by 0.18 Ha
`tests/test_sd_space_variational.py:110` asserts only `atom.total_energy > BE_EXACT_NR`
(`BE_EXACT_NR = -14.6674`, line 64). The docstring (line 104) records the value
as **−14.4369 Ha**; measured today with the exact call the test makes
(`Atom(Z=4, config="1s2 2s2", correlation="sd", engine="bspline", corrections="none")`):
**−14.619159 Ha** (399 CSFs). −14.4369 Ha is 0.139 Ha *above* the 1-CSF
reference measured on the same machine (−14.575892 Ha, `correlation=None`), i.e.
the docstring's own number describes an SD space that is *worse than the
reference determinant it contains* — a first-order variational failure — and the
assertion is green either way. Nothing in the suite can detect a 0.18 Ha drift
in the headline Be SD correlation energy.

**Remediation.** Make the gate two-sided and anchored to the reference solve
that is already in the same file:

```python
e_ref = _be(None).total_energy          # -14.575892, itself gated at :86
assert e_ref - 1e-6 > atom.total_energy > BE_EXACT_NR
```

and re-record the docstring number from the produced value. The same
one-sidedness is in `test_singles_are_variational:95`
(`< _be(None).total_energy`, no bound) and in
`test_mcdhf_ci_silent_zeros.py:163`
(`two.total_energy < single.total_energy - 1e-3`) — the latter is satisfied by
−15.05 Ha, i.e. by the exact regression it was written for; add the
`> BE_EXACT_NR − 5 mHa` side there too.

### F3 (S1) — the "independent oracle" shares the formula it certifies and postdates the fix
`benchmarks/mscheme_ci_reference.py` vs `src/atomx/methods/mcdhf/exact_ci.py`.
* `exact_ci.two_body_angular` (`exact_ci.py:609-656`) is a line-by-line
  transcription of `_angular_terms` (`mscheme_ci_reference.py:128-153`): the same
  `q2 % 2` guard, the same `kmax`, the same `exponent = (tja-m2a)//2 + (tjb-m2b)//2 + q`
  phase, the same term order. Its own docstring says so ("the formula is the one
  `benchmarks/mscheme_ci_reference.py::_angular_terms` implements with sympy").
  A sign/factor error in that formula is invisible to the comparison.
* The determinant sign convention is shared by design:
  `exact_ci._ops_sign` ("The same convention as the oracle's `_apply_ops`").
* `patterns_from_configs` calls the repo's own `enumerate_jj_csfs`, and both
  sides share `matrix._h_one_body`, `_norm_sq`, `_rk`, `CSF`/`Subshell`.
* History: `tests/test_ci_matrix_vs_mscheme_reference.py`,
  `test_exact_ci_kernel.py`, `test_ci_analytic_slater_condon.py`,
  `test_sd_space_variational.py`, `test_mcdhf_ci_silent_zeros.py` were all
  **added by the fix commit 04cd274**, and at that commit the oracle file does
  not exist in the tree (`git ls-tree 04cd274 benchmarks/` has no
  `mscheme_ci_reference.py`) although both test files import it; the oracle's
  only commit is `2007d36`, a **descendant** of the fix. The "oracle with its
  own bug, found by rotational invariance" story cannot be corroborated from
  history — it was written in the same working session as the fix.

What remains genuinely independent: sympy `wigner_3j`/`Rational` vs
`atomx.angular.wigner.wigner_3j`; `ck_red` (sympy) vs `reduced_C`; brute-force
second quantisation vs the cached term table; and the analytic identity in
`test_ci_analytic_slater_condon.py` (which I re-derived from the Gaunt
coefficient `A₁ = 4π/3 · |G(0,1,1;0)|² = 1/3`, giving
`E(³P) = F⁰ − G¹/3`, `E(¹P) = F⁰ + G¹/3`, splitting `2G¹/3`; measured
`ratio = 0.9984` at Z=2, `G¹(1s,2p) = 0.07042`).

**Remediation.** (a) Correct the claim in `exact_ci.py:32-37` and in the test
docstring: state that the *formula* is shared and only the algebra tables and
the assembly differ. (b) Add one gate that does not go through the shared
formula: the analytic `2G¹/3` identity already does this for the J=1 exchange
block — extend it to the `¹S/³S`, `¹D/³D` blocks of an `ns n'd` configuration
(`E(¹D) − E(³D) = …`), and to a three-open-subshell case where the coefficient is
a 6j ratio, so at least one coupled coefficient is pinned without the oracle.
(c) Name the oracle's provenance in its docstring (untracked local file, added
after the fix): a reader currently reads "independent oracle" as "pre-existing".

### F4 (S1) — the cross-engine gate cannot detect a wrong matrix, contrary to its docstring
`tests/test_averaging_decoupling.py:51-56` claims "the B-spline Galerkin/cell-integral
and shooting/SLAG6 pipelines share no radial numerics, so they cannot agree on a
wrong matrix". Both engines call the same `build_ci_matrix` → `exact_block_matrix`
(`matrix.py:563-568`); only the radial evaluator differs (`_rk` → cell integral vs
SLAG6). The gate therefore validates the *radial integrals* and the SCF fixed
point, not the CI matrix. Measured demonstration of the principle on a sibling
file: swapping in the legacy assembly changes the Fe XVII total by **0.79 Ha**
(F1) while the angular/CSF/assembly layer is byte-identical, and no cross-engine
gate in the slice notices.

**Remediation.** Rewrite the docstring to claim exactly what is tested
("the two engines' radial integral implementations agree inside the same CI
assembly to 7.3e-8") and add the matrix-level gate the sentence promises, e.g.
the FAC/`sfac` level comparison for Co (`benchmarks/fac_runner.py` exists) or the
oracle spectrum at the same orbitals.

### F5 (S1) — 24 element-level tests exercise dead code; `except AssertionError → skip` swallows the oracle's own symmetry check
`tests/test_two_electron_replacement.py:206-255` (16 pair + 8 split cases) call
`matrix._two_electron_replacement_element`, reachable only through
`_two_diff_element` (`matrix.py:1220`) ← `_ci_matrix_element` (`:750`) ←
`_build_ci_matrix_approximate` (`:705`), **which has no caller anywhere in
`src/`, `tests/`, `benchmarks/` or `scripts/`**. Same for
`test_spectator_guard_leaves_open_shell_parentage_alone` (`:400-417`).
Worse, the parametrised tests wrap the oracle construction in
`try: … except AssertionError: pytest.skip(...)` (`:210-213`, `:240-244`). That
handler also catches the oracle's own `_j2_check` (`:126-128`), which asserts
`⟨J²⟩ == J(J+1)` — the symmetry that is supposed to certify the test's state
construction. A broken orbit/vector turns 5 tests into green skips instead of a
failure. Measured: `25 passed, 5 skipped in 92.04 s`; all 5 skips are the
legitimately impossible `J=2` pairs (`s²_{1/2}` cannot couple to 2).

**Remediation.** (a) Move the `pytest.skip` decision out of the exception
handler: decide the allowed `J` from the triangle rule up front
(`allowed_coupled_J2`) and skip explicitly, so `_j2_check` failures propagate.
(b) Either delete `_build_ci_matrix_approximate` + its 7 helpers and the tests
that only reach them, or mark the element-level tests with the `snapshot`
marker and a comment that they cover a diagnostic path — as written they read
as coverage of the production Hamiltonian and are not.

### F6 (S1) — `test_off_diagonal_j_ci_helike_1s2p.py`: 6 tests of dead code, headline anchor wrong by 3×, assertions are "non-zero"
The whole file targets `_off_diagonal_J_correction` (`matrix.py:1699`), reachable
only from `_ci_matrix_element` → the unreachable legacy kernel. Its assertions:

* `:97`, `:115`, `:220`, `:248` — `assert abs(delta) > 0.0`; at `:220` the call is
  `_off_diagonal_J_correction(alpha, beta, orbitals_dict, bsp_data, ns=ns)` with
  `orbitals_dict = {(1,-1): (ones(4), zeros(4)), …}` and `bsp_data = {}` — a
  dimensionless number computed with no radial data can never be a matrix element,
  and "> 0" is satisfied by any junk.
* `:127` — `cross_block == 0.0` (a guard, not physics).
* `:162` — `k_range == [1]`, `:179` — `abs(sixj) > 1e-9` (reachability checks).
* The module docstring `:18` states `E([¹]P₁) − E([³]P₁) = 2 G¹(1s,2p)`. The
  correct LS-limit identity is `2G¹/3` (`test_ci_analytic_slater_condon.py:68`
  asserts it correctly and passes at ratio 0.9984). The file's own comment
  (`:32-34`) even says the 6j weight *differs* from the configuration-average
  prefactor — the two statements cannot both hold with the same G¹.
* The docstring `:36-40` still describes the function as a scaffold awaiting
  HYP-189 with an "analytic xfail assertion below"; there is no xfail in the file
  and the function it describes was superseded by N1.

**Remediation.** Delete the file with the legacy kernel, or, if the Fe XVII
spectator scalar is worth keeping, keep only
`test_fe17_one_diff_single_spectator_exact_scalar_anchor` (`:251-290`) — it is
the one test with a hand-checkable number (−0.5 and 1/√3, provenance
`95a1ca5`/`ef92606`) — retarget it at the exact path
(`matrix._branch9_double_excitation_scalar` is itself only reached from the same
legacy chain; the production equivalent is `exact_ci.exact_block_matrix`), and
fix the `2 G¹` → `2G¹/3` statement in both files (this one and
`test_ci_matrix_vs_mscheme_reference.py:379`).

### F7 (S2) — the same Be constants are pinned to four different values, with and without a bound margin
| literal | where | meaning |
|---|---|---|
| `-14.66736` | `test_ci_matrix_vs_mscheme_reference.py:50`, `test_ci_analytic_slater_condon.py:43` | "exact NR Be (Drake)" |
| `-14.66736 - 0.005` | `test_ci_matrix_vs_mscheme_reference.py:51` | the DC lower bound actually asserted (`:235`, `:276`) |
| `-14.6674` | `test_sd_space_variational.py:64` | the same quantity, 4e-5 different, with **zero** margin |
| `-14.5734` | `test_ci_analytic_slater_condon.py:30` | "1-CSF HF"; measured on this machine: **−14.575892**; textbook HF limit −14.573023 |
| `-14.5759 ± 2e-3` | `test_sd_space_variational.py:86`, `test_ci_levels_vs_nist.py:112` | the actual single-config value |
| `-14.6166 ± 6e-3` | `test_ci_analytic_slater_condon.py:42` | "literature 2-config MCHF" (introduced by the fix commit; a 6 mHa band on a 0.1 mHa comparison) |

Also: the SD docstring's `-14.4369` (F2) and `-14.652/-14.619` in adjacent
docstrings. A reader cannot tell which number belongs to which protocol.

**Remediation.** One module, `tests/_be_reference.py`, holding
`BE_EXACT_NR`, `BE_DC_LOWER_BOUND = BE_EXACT_NR - 0.005` (with the finite-nucleus
shift ΔE ≈ −2.5 mHa documented) and `BE_HF_1CSF` measured in-test rather than
typed; every file imports it. Quote the literature HF *limit* (−14.573023) and
say explicitly that atomx's basis sits 2.9 mHa above it.

### F8 (S2) — tolerances that cannot distinguish the physics they gate
* `tests/test_mcdhf.py:73` — `result.ground_energy == approx(-2.86, abs=0.1)`.
  He HF is −2.861680 (and this same suite pins it to 1e-5 elsewhere); ±0.1 Ha is
  3.5 % and would accept the *un*correlated hydrogenic value −2.8477 or a
  0.05 Ha-broken SCF. → assert the oracle/numerical HF value to 1e-5, or the
  Löwdin bound `> −2.86168`.
* `tests/test_sd_space_variational.py:127` — `eps_2p > -2.00047 + 0.5`
  (i.e. > −1.50047). Measured correct value −0.228585; a half-collapsed virtual
  at −1.4 Ha passes. → assert `eps_2p == approx(-0.2286, abs=0.02)` against the
  parent-capped mean field, with the value's provenance named, or assert the
  screening scale `|eps_2p| < |E_Dirac(2p,Z=4)|/4`.
* `tests/test_two_electron_states.py:42` — `rel=0.10` on 904–922 eV excitation
  energies = ±90 eV; the Li-like–He-like splitting scale is 10 eV. →
  `abs=20 eV`, or compare the *differences* (1s2s ¹S − 1s2s ³S etc.) which the
  hydrogenic model gets right to a few eV.
* `tests/test_auger_mscheme_ci.py:175` — `rel=0.35` on the strongest Auger
  width (measured ratio 0.958). → 0.10, or gate the *branching* between the two
  strongest levels, which is orbital-insensitive.
* `tests/test_ci_levels_vs_nist.py:135,153,185` — ±3000 cm⁻¹ (0.37 eV) on lines
  whose residual is documented as 1 100–8 150 cm⁻¹. The xfail discipline is
  right (see model tests); the tolerance should be the measured residual of the
  protocol that is supposed to pass, not a round number 3× larger.

### F9 (S2) — the fine-structure ordering gate runs only on the protocol where it passes
`tests/test_ci_levels_vs_nist.py:115-128` asserts `3P0 < 3P1 < 3P2 < 1P1` on the
`fac_frozen` fixture only; the module docstring (`:49-53`) states that in the
**default** `mcdhf` protocol "the fine-structure ordering is also still
inverted", and that fact is recorded in an xfail reason rather than gated. A
reader of the summary line ("three_p_ordering_is_physical") will believe the
solver produces physical fine structure; it does so in one of three protocols.

**Remediation.** Parametrise the ordering gate over `fac_frozen` and `mcdhf`,
xfail-strict the `mcdhf` case with its measured numbers (the file already uses
that pattern at `:138` and `:156`), and state in the docstring of the passing
case which protocol it covers.

### F10 (S2) — CSF/determinant counts asserted as if they validated the Hamiltonian
`test_ci_J_block.py:73` (`len(csfs) == 37`), `test_csf_J_enum.py:43`
(`len(csfs) == 27`), `test_csf_J_enum.py:13,17` (Grant App. A multisets),
`test_ci_matrix_vs_mscheme_reference.py:176` (`n_csf == 399`),
`test_exact_ci_kernel.py:267` (`== 29`), `test_auger_mscheme_ci.py:150` (`== 9`).
These are all legitimate *generator* tests — `test_csf_J_enum.py` names its
external source (Grant App. A) and `test_ci_matrix_vs_mscheme_reference.py`
cross-checks the enumeration against an independent convolution of subshell
2J-multiplicities — but none of them can fail for a physics reason, and F1 shows
what happens when a file's only "physics" gates are counts. Mark them
`snapshot`, and keep them out of the physics count.

### F11 (S3) — the D-class literals in this slice are unmarked
`@pytest.mark.snapshot` appears **once in the entire suite**
(`tests/test_fac_parity_gates.py:49`) and **zero times in these 24 files**, while
35 tests here are D-class (a literal, or a band around one, that is only known
because atomx printed it): `test_ci_grid_kernel.py:252`
(`0.169929196167459`), `test_two_electron_replacement.py:306,307,321,344,378`,
`test_averaging_decoupling.py:57,58,113,135,156,157,169`,
`test_ci_levels_vs_nist.py:112`, `test_sd_space_variational.py:86`,
`test_ci_J_block.py:73`, `test_mscheme.py:54,63,94,97,101`, the
`test_csf_recoupling.py` literals, plus the `test_mcdhf.py` protocol constants
(`0.001` occupations).

**Remediation.** Apply the marker mechanically to every test whose only
assertion is `approx(<literal>)` against an atomx-produced quantity; the marker
already exists and is documented as "asserts nothing external". This is the
single cheapest change that makes the suite honest about its coverage.

### F12 (S3) — `test_rk_pairing_matches_gk` is a tautology
`tests/test_ci_grid_kernel.py:204-219` asserts
`_rk(1s, Q, 2s, Q, 1s, Q, 2s, Q, ws, 0) == _gk(1s, Q, 2s, Q, ws, 0)`. On a
`GridCIWorkspace` the two functions are the same expression
(`matrix.py:186-189`: `_slater_integral_slag6(_grid_pair_density(Pa,Qa,Pb,Qb),
_grid_pair_density(Pc,Qc,Pd,Qd), …)` with the last pair equal to the first, vs
`matrix.py:161-163`: the same with `rho_ab` twice). The assertion cannot fail.
The sibling `test_two_electron_replacement.py:258-289` has the same problem in
its first assertion (`_rk` literally calls `compute_slater_rk_cell`); only its
second assertion (`abs(got - transposed) > 1e-6`) carries information, and that
one is a fixture-discrimination check, not a physics check.

**Remediation.** Keep the informative half (`_rk`'s *pairing* is a convention, so
the discriminating test is against a hand-computed `R^0(1s2s,1s2s)` and
`R^0(1s1s,2s2s)` pair from the analytic hydrogenic table already in this file);
delete the `== _gk` assertion.

### F13 (S4) — misleading names and vacuous tests
* `tests/test_two_electron_replacement.py:390-397` —
  `test_pair_channel_does_not_couple_incompatible_parity` asserts only
  `np.all(np.isfinite(energies))`. The parity block structure it names is gated
  elsewhere (`test_ci_matrix_vs_mscheme_reference.py:240`); this test would pass
  with the parity filter removed entirely, as long as nothing returns NaN.
* `tests/test_auger_mscheme_ci.py:179-182` — `test_level_energy_spread` computes
  `spread = max(L.energy_au) * HARTREE_EV` (no subtraction) and asserts
  `10 <= spread <= 32`; a fully degenerate 20 eV manifold passes. Fix: use
  `(max - min)`, and assert the FAC-implied 18 eV to 30 %.
* `tests/test_mbpt_transitions.py:42-49` — `test_transition_energy_self_consistent`
  compares `omega_corrected_au ≈ omega_DHF_au` where the previous test
  (`:38-39`) has already established both are exactly `0.0`. 0 ≈ 0.
* `tests/test_ci.py:52-54` — `ground_energy == min(state_energies)` (true for any
  eigh call), `provenance["n_csf"] == len(csfs)`.
* `tests/test_ci_J_block.py:219-241` — named "…lift…" but asserts
  `n_states == len(csfs)` and `energies[0] < 0`; the one-electron J-block lift is
  identically zero by construction, so the test cannot detect a regression of the
  lift it names.
* `tests/test_mcdhf.py:105-113` (`de[0] == 0.0`), `:82-84`
  (`state_energies[0] < state_energies[1]`, `ground < 0`),
  `:149-167` (`converged` and `orbitals_dict` non-empty).
* `tests/test_exact_ci_kernel.py:316-327` (`test_cached_structures_are_reused_and_deterministic`)
  and `:330-344` — cache identity and SVD-basis determinism; legitimate but
  nothing physical, and `:341` (`first == second` on a re-derived object) is a
  determinism check with no value gate beyond `‖v‖ = 1`.

### F14 (S4) — `test_csf_recoupling.py` (733 lines, 42 tests) tests a library with no production caller
`chain_wxw_scalar`, `chain_one_body_excitation_reduced`,
`discover_*_coefficients_*`, `classify_recoupled_occupation_change`,
`two_body_scalar_coefficients_simple`, `chain_two_body_scalar_coefficients`,
`chain_state_from_simple_csf_auto`, `scalar_w_subshell_coeff`,
`two_subshell_*_w_reduced` appear **only** in `csf_recoupling.py` itself (internal
chaining) and in `atomx/angular/__init__.py` re-exports; the only consumers in
`src/` are the unreachable legacy helpers
(`matrix.py:1408,1441` inside `_aligned_unique_chain_from_occ` /
`_chain_global_projection` → `_branch9_*`). The only recoupling helper on a live
path is `allowed_coupled_J2` (`exact_ci.py:76,329` and `config.py:16,386`), which
this file does not test. 11 of the 42 tests are D-class literals
(`1/√8`, `−(3/2)^{1/2}`, `−√3/2`, `√3`, −1.0, 0.5) whose introducing commit
`ef92606` cites no source. The 3 tests that do have an independent anchor
(`test_chain_one_body_excitation_tracks_global_fock_phase_convention:452`,
`…_with_spectator:484`) compare the production value to
`−_exact_global_one_body_reduced`, a from-scratch determinant-space evaluation
written in the test file — that is the pattern the other 39 should follow.

**Remediation.** Either wire the recoupling library into a production path or
move the file under a `legacy/` test node with the `snapshot` marker; convert
the numeric literals by re-deriving each through
`_exact_global_one_body_reduced`-style brute force (the machinery is already
there for one-body; a two-body analogue is ~40 lines).

### F15 (S2) — per-line line-strength brackets are re-pinned ranges, not identities
`tests/test_ci_line_strengths.py:235-236` —
`0.7 <= S(2p3/2-hole line)/channel <= 1.0` and `1.0 <= S(2p1/2)/channel <= 1.7`
with the comment "measured 0.820 / 1.397". The intervals are the measured value
±0.2–0.3 and contain the no-mixing value 1.0 in both cases, so they cannot
distinguish mixing from no mixing. The *sum* identity on the next two lines
(`S_a + S_b == channels[-2] + channels[1]`, `rel=1e-10`) is the real gate and is
first-principles (mixing-invariance). → keep the sum rule; delete or replace the
brackets with the mixing angle itself (`atan2` of the CI vector ratio), which is
what the physics is.

### F16 (S2) — MBPT(2) transition tests gate nothing about MBPT(2)
`tests/test_mbpt_transitions.py`: 4 tests, of which 2 are error paths, 1 is a
dict-key check, and the only physics-looking one is the 0 ≈ 0 comparison of F13.
The module's actual claim (a Brueckner orbital-energy shift) is exercised only
through a case where the shift is zero by construction. → add the He 1s→2s
(or Ne 2p) case with the shift compared to the second-order perturbation
expression evaluated independently in the test from `mbpt2_energy`'s own
ingredients, and assert the *sign* and magnitude against the known
Brueckner/MP2 orbital-energy shift.

## 4. Model tests (imitating these would have prevented F1–F6)

1. `tests/test_mbpt2.py:82` — He MBPT(2): the variational sandwich
   `HF (−2.861680) < E_tot < exact (−2.903724)` **and** `E2 > −0.042044`
   (exact correlation energy) **and** `≈ −0.0372 (JCP 102, 6168 (1995))` at
   `rel=0.01`. Bound + external + magnitude: the template for every energy gate
   in this slice.
2. `tests/test_ci_analytic_slater_condon.py:68` — `E(¹P₁) − E(³P₁) = 2G¹(1s,2p)/3`
   as a *ratio* against `MAT._gk` on the same orbitals, restricted to Z ≤ 10 with
   the relativistic drift documented (1.0000 / 1.0019 / 1.0119 measured, 3.533
   for the old kernel). Independent, orbital-free, and it fails loudly for a
   wrong angular coefficient. (I re-derived the identity and reproduced
   ratio = 0.9984 at Z = 2.)
3. `tests/test_exact_ci_kernel.py:146` — `test_slater_condon_terms_match_brute_force`:
   monkeypatches `_h_one_body`/`_rk`/`_norm_sq` to deterministic synthetic values
   and compares the production term table with the oracle's second-quantised
   determinant Hamiltonian to 1e-12, so the test isolates *assembly* from radial
   numerics. The single best-engineered test in the slice.
4. `tests/test_exact_ci_kernel.py:68` — `test_angular_kernel_matches_the_sympy_oracle`:
   ＞900 sampled spinor quadruples, union-of-k comparison at 1e-12, with an
   explicit `nonempty > 100` guard against a vacuous sample.
5. `tests/test_ci_matrix_vs_mscheme_reference.py:286` —
   `test_full_spectrum_matches_mscheme_reference`: **every** eigenvalue, four
   configuration sets, `n_det == len(csfs)` asserted first, 1e-10. Comparing the
   spectrum instead of the ground state is what makes a single wrong element
   visible.
6. `tests/test_ci_matrix_vs_mscheme_reference.py:309` —
   `test_oracle_is_rotationally_invariant`: gates the arbiter itself (levels at
   2M = 0/2/4 must coincide). This is the check that found the oracle's dropped
   exchange term, and it should be copied for any future oracle.
7. `tests/test_ci_matrix_vs_mscheme_reference.py:334` — Brillouin's theorem as an
   end-to-end gate: a singles expansion on HF orbitals buys `< 5e-4` Ha. A
   theorem, not a number.
8. `tests/test_ci_grid_kernel.py:193` — exact Condon–Shortley hydrogenic
   `F^k`/`G^k` (`5/8`, `17/81`, `16/729`, `59/243`, `112/2187`, `93/512`,
   `45/512`) at `rel=1e-6`, with the symbolic verification stated. The only
   absolute radial-integral gate in the slice.
9. `tests/test_ci_line_strengths.py:184` — `S_sum == trace` to 1e-10: a
   mixing-invariant sum rule that is independent of H and of the CI vectors, plus
   the parity-forbidden `trace == 0` case at `:196`.
10. `tests/test_davidson.py:22,31` — eigensolver gated against LAPACK and by the
    eigen-residual `‖HV − Vλ‖ < 1e-8`; the two assertions are independent and
    both first-principles.
11. `tests/test_mscheme.py:38` — particle–hole symmetry
    (`j^q` vs `j^{g−q}` have the same J content), and `:59` where the scalar
    number-operator phase is derived in the comment
    (`(−1)^{2j} N/√(2j+1)`) before being asserted.
12. `tests/test_ci_levels_vs_nist.py:138,156` — the honest-failure pattern: a
    strict xfail that carries the measured number and the reason it is red,
    instead of widening the tolerance until it passes. Equivalent in
    `test_sd_space_variational.py`'s history and the `test_ci_J_block.py`
    docstring's admission of the 200 eV band — the *pattern* is model; what needs
    fixing is that in the latter two the xfail was converted into a pass (F1, F2)
    without tightening the assertion.
