# S3 — Radiative corrections: QED and Breit

Slice: `tests/test_qed.py`, `test_qedmod.py`, `test_qed_uehling.py`,
`test_qed_self_energy_table.py`, `test_qed_indelicato_vs_fac.py`,
`test_qed_orbital_shifts.py`, `test_qed_backend_auto.py`, `test_breit_pauli.py`,
`test_corrections_stack.py`, `test_correction_stack_injection.py`,
`test_higher_multipoles.py`, `test_shooting_corrections.py`,
`test_property_regressions.py` — all 13 exist; 164 collected tests, 0 failures
(`pytest … -q` → 67 + 97 passed). **Added: `tests/test_nist_levels.py`**, which
is not in the slice list but is the only file in the suite that gates the *Breit
interaction* against an external code; two of its six tests are audited here.

---

## 1. Verdict

The **QED (Uehling / self-energy) half of this slice is genuinely fixed and
genuinely gated**: `test_qed_uehling.py` and `test_qed_self_energy_table.py`
carry real first-principles and external anchors whose numbers I re-derived
independently (the defining Uehling integral by my own quadrature for all five
tabulated kernel values; U 1s = −97.936 eV vs the pinned −97.9; and the whole
112-point 1s row plus all 19 `_sd` rows of the self-energy table match FAC's
`faclib/coulomb.c` bit-for-bit, 0 mismatches at 1e-9; my crude H 1s quadrature
gives −216.6 MHz against the published −215.13 MHz, i.e. the test's target is
right). The **Breit half is the opposite**: the default public
API still applies a scalar Breit that is **3.03× DBSR_HF** (measured
`Atom(Z=10)` → `E_Breit = 0.05043710257669187` Ha vs DBSR_HF
`0.016643629692418` Ha), and this slice's tests do not merely miss it — they
**pin it**. `tests/test_qed.py:135` asserts `E_Breit == approx(0.0504, rel=0.05)`
with a false external attribution ("Grant 2007 tables for Ne I"), so the file
would have to be edited to accept the correct physics; the "independent
confirmation" of that number by the shooting engine
(`test_shooting_corrections.py:134`) is C-degenerate because both engines call
the same `corrections/breit.py::compute_breit_energy`; and the one honest
external gate exists only as a strict `xfail`
(`test_nist_levels.py:179`). Worse, the operator that *is* correct
(`methods/mcdhf/breit_ci.py`, now the MCDHF default) had **no test file at all**:
its docstring claims validation by `tests/test_breit_operator.py`, which does not
exist in `HEAD`, and no tracked test imports `breit_ci` (an untracked file of that
name appeared mid-audit — see F1's post-audit note). Net: 164 green tests whose
class mix is **A55 B20 C4 C\*17 D10 E33 F25** — 58 items (35 %) assert nothing
that can fail for a physics reason, 17 more are C-degenerate, and 10 pin atomx's
own output unmarked; only 79 items carry a physics claim, 20 of them against
something external. Every physics assertion about the Breit interaction is
either wrong, degenerate, or an expected failure.

---

## 2. Per-file table

Class mix counts **collected test items** (parametrised functions contribute one
per case), so each mix sums to the "tests" column. `C*` = C-degenerate.

| file | tests | class mix | worst sev | note |
|---|---|---|---|---|
| `test_qed.py` | 17 | A6 D9 E2 | **S1** | pins the wrong scalar Breit (0.0504, 3.03×) at :135 and a total band 81 % of the correction budget at :164; the Ne VP pin at :139 *does* gate the K₁ bug. |
| `test_qedmod.py` | 29 | A2 B2 C4 C\*2 E7 F12 | S1 | backend plumbing is sound; the two `shabaev_local` vs `mohr` comparisons (:190, :211) are C-degenerate by construction (A_κ is calibrated on `_interpolate_se_f`). |
| `test_qed_uehling.py` | 16 | A12 B1 C\*1 E2 | — | model slice: defining integral, ln(10) small-x law, published Zα expansion (−215.1 MHz), U 1s −97.9 eV. |
| `test_qed_self_energy_table.py` | 13 | A8 B4 E1(C\*) | S3 | FAC + Yerokhin–Shabaev anchored; `test_backends_agree_exactly` (:121) compares a delegation to itself. |
| `test_qed_indelicato_vs_fac.py` | 20 | B13 E7 (4 C\*) | S3 | FAC run outputs are real and correct (rel 1e-3), but :51 is a self-comparison and :101 re-checks the same FAC_F numbers with a weaker band. |
| `test_qed_orbital_shifts.py` | 2 | D1 E1 | S2 | 3.5–4.4 eV band on a value re-pinned by the same commit; :52 re-implements `shifted_orbital_energies`. |
| `test_qed_backend_auto.py` | 4 | F4 | S3 | pure dispatch-name plumbing; the Z=37 policy it pins is physically inert now that both backends read one table; duplicates `test_qedmod.py:386-396`. |
| `test_breit_pauli.py` | 11 | A9 E1 F1 | — | genuine A-class (α²/32 Dirac splitting, α²/48 ξ, full-shell SO = 0); **not** the relativistic Breit. |
| `test_corrections_stack.py` | 8 | E3 F5 | S1 | the file that applies the Breit adapter has zero value assertions — `isfinite` only. |
| `test_correction_stack_injection.py` | 4 | E2 F2 | S1 | `E_QED_total == E_SE + E_VP` is true by construction (`stack.py:192-206`); :47 compares two stacks sharing the kernel. |
| `test_higher_multipoles.py` | 14 | A9 E5 | S4 | about *transition* multipoles, not the Breit L-expansion; no v = L / v ≠ L coverage; :101's claim to catch a prefactor error is false. |
| `test_shooting_corrections.py` | 19 | A3 C\*9 E6 F1 | **S1** | the cross-engine gate (:134, rtol 2e-3) "certifies" the 3× wrong Breit because both engines share the kernel. |
| `test_property_regressions.py` | 7 | A6 B1 | — | outside the QED/Breit subject (A-coefficients + NMS), but excellent A-class work. |
| `tests/test_nist_levels.py` *(added)* | 6 (2 audited) | A1 B1 | **S1** | the only external Breit gate (`:179`) is `xfail(strict)`; the module docstring at :19 still calls both xfails current. |

---

## 3. Findings

### S1

**F1 — `breit_ci.py` names a validator that does not exist; the operator is untested.**
`src/atomx/methods/mcdhf/breit_ci.py:52`: *"``tests/test_breit_operator.py`` checks the angular algebra against the repo's independent Coulomb kernel, checks a hand-computable Gaunt matrix element, and compares closed-shell totals with DBSR_HF"*. `ls tests/test_breit_operator.py` → no such file; `tests/test_nist_levels.py:159` records the same absence ("claimed by the breit_ci docstring but ABSENT from the tree"), and `tests/test_nist_levels.py:75` refers to
`test_breit_operator_matches_closed_shell_scalar`, which also does not exist.
`grep -rn "breit_ci\|gaunt_terms\|BreitOperatorError\|RETARDATION_NOTE" tests/*.py` → only comment text in `test_nist_levels.py`; **no test imported the module** as of the audit (see the post-audit note below). It is nevertheless the MCDHF default (`methods/mcdhf/solver.py:287 breit_mode="operator"`, consumed by `exact_ci.py:877`) and it is what produces fine structure.
*Remediation:* write `tests/test_breit_operator.py` with the three gates the docstring already promises — (a) a hand-computable Gaunt element: two electrons in a single determinant, angular part from the closed-form `∫∫ (α₁·α₂)/r₁₂`, compare to `gaunt_terms` at rtol 1e-10; (b) a Wigner–Eckart/3j ratio test against the repo's independent Coulomb kernel (`angular/` `C^k` machinery shared with `_rk`); (c) closed-shell totals vs DBSR_HF: **Ne = 0.017340 Ha** (rel 5 %) for the operator and **He = 6.37774e-05 Ha** (rel 5 %) for the scalar, with `RETARDATION_NOTE` asserted to be reported, not silent. Then correct `test_nist_levels.py:75,159`.

*Post-audit note (same session, 22:24):* an **untracked** `tests/test_breit_operator.py` appeared in the working tree while this audit was running (not in `HEAD`; `git status` → `?? tests/test_breit_operator.py`), presumably from a concurrent agent. It is 38 passing tests of genuine value — `_clm` against the closed-form Racah–Gaunt coefficient, `σ_μ† = (−1)^μ σ_{−μ}`, `Tr(σ_μ σ_ν†) = 2δ_{μν}`, the `Σ_q |C^L_q|² = 1` completeness sum rule, M-conservation, the triangle/parity selection rules, and the `BreitOperatorError`/`RETARDATION_NOTE` contract. It covers gates (a) and the angular half of (b). **Gate (c) is still missing**: its docstring delegates "closed-shell totals against DBSR_HF, and Hermiticity of the assembled CI matrix" to `tests/test_exact_ci_kernel.py`, and that tracked file contains **no** occurrence of `breit`, `Breit`, `DBSR` or `hermitic` (`grep -n` → 0 matches). So the external-anchor half of F1 remains open, and the file must also be `git add`-ed or the docstring claim stays false for every other checkout.
**F2 — `test_qed.py:112-137` pins the wrong scalar Breit with a fabricated external citation.**
`assert E_Breit == pytest.approx(0.0504, rel=0.05)` accepts [0.04788, 0.05292].
Measured this session: `Atom(Z=10, engine='bspline', exchange='hf')` → `0.05044859084011072`; default `Atom(Z=10)` → `0.05043710257669187`; DBSR_HF (`dbsr_hf an=Z term=AV mbreit=1`) → `0.016643629692418`; the repo's own Gaunt operator → `0.017340`. The assertion's band **excludes the correct value by 3×**, so this test is a barrier to the fix, not a gate. The comment at :117-119 attributes `+0.05 Ha` to *"Grant 2007 tables for Ne I"* — no such number exists; the true Ne Breit is 0.0166 Ha. The comment also claims *"rel=5% catches the α²-suppression class of bug"*, which is true, while missing the 3× class.
*Remediation:* replace with `assert E_Breit == pytest.approx(0.016643629692418, rel=0.05)` (DBSR_HF, quoted in-test), tag `@pytest.mark.external_fac`, and delete the Grant attribution. Expected outcome: red until `corrections/breit.py` is fixed.

**F3 — `test_shooting_corrections.py:134-142` is the C-degenerate "proof" that the wrong Breit is right.**
`test_cross_engine_agreement` asserts `|shooting/bspline − 1| ≤ 2e-3` for `E_Breit`, `E_SE`, `E_VP`. Both engines route through `corrections/adapters.py:229-238 → corrections/breit.py::compute_breit_energy`; only the radial representation differs (`_flatten_scf_orbitals` vs `_flatten_grid_orbitals`). Measured: shooting and bspline both report Ne `E_Breit = 0.0504371` (agreement 2.3e-4) while both are 3.03× DBSR_HF. The module docstring (:12-20) and commit `d064cdc` ("the shooting engine independently gives 0.050437") present this as external validation; it validates the *quadrature*, nothing else.
*Remediation:* rename to `test_cross_engine_radial_numerics` and state in the docstring that the shared kernel is `recompute`d, not re-implemented; add one absolute anchor per component (DBSR_HF Ne Breit 0.016643629692418, FAC Ne SE, and the Uehling Z=1/Z=92 anchors already in `test_qed_uehling.py`).

**F4 — `test_qed.py:156-164` passes before and after both QED fixes.**
`assert ne_full_result.total_energy == pytest.approx(-128.6136, abs=0.05)`. Measured current total: **−128.62995892582862** → deviation 0.0164 Ha, inside the 0.05 band. The band is 0.05 Ha = 81 % of the whole correction budget (Breit 0.0504 + SE 0.0121 + VP −0.0006 = 0.0620 Ha), and the docstring still quotes the pre-fix decomposition (VP −0.002355, SE +0.030218), which is now 3.9× and 2.5× wrong respectively. `git show 200d657 -- tests/test_qed.py` shows the assertion was **not** touched by the kernel/table fix.
*Remediation:* delete the total-band assertion and keep only per-component external anchors; if a total is wanted, use `abs=2e-4` (measured per-engine spread is 2e-5 Ha, `test_corrections_shift_total_energy` already proves the additivity).

**F5 — `test_qed.py:83-91` (`test_se_scales_as_Z4`) could not see the 2.5×–14.7× table defect.**
`assert 10 < ratio < 25` for Z=10 → Z=20. Measured with the corrected table: 11.160. With the pre-fix table (`git show 200d657^:src/atomx/data/mohr_se.py`, F(10)=11.6804, F(20)=12.6423): 16 × 12.6423/11.6804 = **17.32** — also inside the band. A 2.5× error at Z=10 and a 14.7× error at Z=92 both pass.
*Remediation:* assert the *ratio of the F values* against FAC, e.g. `F(10)/F(20) = 4.65413/3.24626 = 1.4337` with rel 1e-3 (those two FAC numbers are already quoted at `test_qed_self_energy_table.py:85`), plus the same test at Z=54.

**F6 — `test_qed.py:97-106` and `:103-106`: order-of-magnitude caps with 52×–350× margin.**
`abs(E_Breit) < 0.01` with measured He `E_Breit = 1.915555e-04` (52× margin; the docstring at :98 claims "~10^-5", 19× below the actual value) and `abs(E_QED) < 0.01` with measured He `E_QED = E_SE + E_VP = 2.848106e-05 − 7.3292e-07` (350× margin). The pre-fix (2.5×–14.7× wrong) table also passes both.
*Remediation:* pin He against DBSR_HF `E_Breit = 6.37774e-05` (rel 5 %) — the number is already in the repo at `tests/test_nist_levels.py:173` — and He `E_SE` against a FAC/sfac run; otherwise delete, since `test_qed_self_energy_table.py` already gates the SE table far better.

**F7 — `test_corrections_stack.py` applies the Breit adapter and asserts nothing about it.**
`:27-37` checks only `np.isfinite(applied[name].energy_shift)` for `{"self_energy[mohr]", "uehling_vp", "breit"}`; `:39-47` asserts `stack.total(...) == sum(...)` (the definition of `CorrectionStack.total`, `stack.py:137-142`); `:103-106` asserts a sum over an empty dict is 0.0. This is the file through which the default `Atom` correction stack is exercised, so the 3× Breit error and any future one is invisible here.
*Remediation:* add one value assertion per rung: `applied["breit"].energy_shift == pytest.approx(0.016643629692418, rel=0.05)` for `Atom(Z=10)`, `applied["uehling_vp"].energy_shift == approx(-0.0006009, rel=0.05)`, `applied["self_energy[mohr]"].energy_shift == approx(0.012119, rel=0.05)`; and drop `test_total_matches_sum_of_applied` (or move it to the E-graveyard).

**F8 — `test_nist_levels.py:179` is the only external Breit gate, and it is an expected failure.**
`test_breit_scalar_matches_dbsr_hf` asserts `E_Breit == approx(0.016643629692418, rel=0.05)` and is `@pytest.mark.xfail(strict=True)`: measured `pytest tests/test_nist_levels.py -k breit` → `1 passed, 1 xfailed`. `docs/VERIFICATION.md:170-171` says "Both are gated by `tests/test_nist_levels.py::test_breit_scalar_matches_dbsr_hf`" — honest, but it means the suite's *green* state carries a known 3× error in the default `E_Breit` rung. Compounding it, the xfail reason (written in `9aa3b0f`) says "the default was flipped back to that wrong scalar", while the newer `04cd274` set `breit_mode="operator"` as the MCDHF default (`solver.py:287`), so the reason text is stale.
*Remediation:* keep the xfail (it is the correct pattern) but (a) add a non-xfail *bound* test that fails if the scalar ever gets worse, (b) reference it from `test_qed.py::test_ne_has_breit`, and (c) refresh the reason against `solver.py:287`.

### S2

**F9 — `test_qed_uehling.py:159-168`: a 3-decade band that cannot see the 3.9× kernel error.**
`assert 1e6 < vals[-1]/vals[0] < 1e9`; measured ratio **1.100e8** (Z=1 → Z=92), while the old `K₁(2cr)` kernel gives 3.9× that = 4.3e8, still inside. The discriminating gates are :56/:126/:148.
*Remediation:* replace by the analytic hydrogenic Z⁴ shape (`|E_VP(92)|/|E_VP(1)| = 92⁴ · f(Zα)` with f from the same expansion) at rel 10 %, or delete.

**F10 — `test_qed_uehling.py:59-66`: the quadrature check compares a spline to the quadrature that built it.**
`_uehling_kernel` is a `CubicSpline` over 900 nodes of `_uehling_I_exact` (`corrections/qed.py:77-98`); the test measures max relative deviation from `_uehling_I_exact` itself, which is ≤ 1e-8 by construction and exactly 0 at the nodes. Class C-degenerate; it validates interpolation, not physics.
*Remediation:* keep it, renamed `test_spline_interpolation_is_machine_accurate`, and note in the docstring that the physics gate is `test_tabulated_values` (I verified those five values against my own `scipy.integrate.quad` of the textbook integrand: max rel. deviation 1.2e-7).

**F11 — `test_qed_orbital_shifts.py:34-49`: a self-pin labelled "a real external-scale check".**
`assert 3.5 <= ev <= 4.4` (measured Fe XVII 1s shift **3.9274 eV**, so −11 %/+12 %) on values re-pinned by the same commit (`200d657`), with a FAC number quoted only in prose ("SE 4.28 − Uehling 0.35"), not in the assertion.
*Remediation:* put the FAC/`sfac` Fe XVII 1s self-energy in the test as a literal with `rel=0.02`, and mark the remaining band `@pytest.mark.snapshot`.

**F12 — `test_qedmod.py:190-209` and `:211-232`: `shabaev_local` vs `mohr` is a calibration check, not a cross-check.**
`shabaev.py` calibrates the amplitude `A_κ(Z)` from `_interpolate_se_f` (`shabaev.py:43, 254`; the test docstring says "By construction: ⟨ψ|V_loc|ψ⟩ = Mohr(1s)"), so `rel=2e-3` agreement is guaranteed and the "closer to Mohr than FG" assertion (:232) is true by definition.
*Remediation:* compare the *potential* to QEDMOD v7's own table (`~/code/ASAP/qedmod_v7` is present in the sibling tree; the backend already ports `f/potgen/se_pot.f`), i.e. assert `A_κ(Z)` against `se_loc(il)/de` from the Fortran data, or delete the backend and its tests.

**F13 — `test_qedmod.py:418-443`: a 5× band on a 25 %-accuracy heuristic.**
`assert 0.5 < E_welton/E_mohr < 2.5`, commented "Allow a factor of 2 either way as a sanity band". Welton is closed-form (`welton.py:105`: `ΔE = (4α³Z/3)·ln(1/(Zα)²)·|ψ(0)|²`), so for a hydrogenic 1s the value is analytic and the ratio to the FAC table can be asserted to a few per cent — the wide band hides exactly the error the backend should be judged on.
*Remediation:* assert `E_welton` against its own closed form with the analytic hydrogenic |ψ(0)|² for Z=2 (A-class), and compare to Mohr at rel 0.25 with the band justified by the backend's documented accuracy.

**F14 — `test_qed.py:139-149`: the Ne VP pin is D-class with an undocumented provenance.**
`approx(-0.000601, rel=0.05)` with measured **−0.0006009391596706532** — introduced in `200d657` (the same commit as the kernel fix) and stated in the docstring as "deterministic −0.000601". It has real discriminating power (the `K₁` value −0.002355 is 3.9× outside), but nothing external anchors the number, and `test_qed_uehling.py` gates only Z=1 and Z=92.
*Remediation:* keep the pin, mark `@pytest.mark.snapshot`, and add a mid-Z external anchor — e.g. evaluate the Yerokhin–Shabaev expansion (already coded at `test_qed_self_energy_table.py:38-54`) against the Ne SCF density with an effective charge, or a DBSR/FAC VP run for Ne.

### S3

**F15 — `test_qed_self_energy_table.py:121-126` and `test_qed_indelicato_vs_fac.py:51-71` compare a function with itself.**
`indelicato.py:42-53` is a one-line delegation: `_interpolate_indelicato_f` returns `_interpolate_se_f`; the two `compute` bodies (`mohr.py:119-140`, `indelicato.py:63-80`) are duplicated line-for-line. So `assert _interpolate_indelicato_f(...) == _interpolate_se_f(...)` (rel 1e-12) and `_interpolate_indelicato_f(Z,1,-1) == approx(_interpolate_se_f(Z,1,-1), rel=1e-12)` cannot fail for any physics reason. They are useful *documentation* of the C2 fix, not gates.
*Remediation:* keep one regression that the two backends' `SEResult.E_SE` agree on a real Ne orbital set (that at least covers the duplicated compute bodies), delete the F-level equality tests, and rely on the FAC literals already present (`test_qed_self_energy_table.py:85`, verified against `coulomb.c` — 0 mismatches over the full 112-point row).

**F16 — D-class pins are not marked `snapshot` anywhere in the slice.**
`pyproject.toml:114` defines `snapshot: frozen atomx SELF-regression against its own recorded output; asserts nothing external`, and `--strict-markers` is on, but the marker appears **once in the whole suite** (`tests/test_fac_parity_gates.py:49`). Unmarked D-class in this slice: `test_qed.py:97,108,139,156,205,218` (8 items); `test_qed_orbital_shifts.py:34` (1 item).
*Remediation:* add `@pytest.mark.snapshot` to each and add a suite-level check that any file whose assertions are 100 % self-referential carries it.

**F17 — duplicated coverage.**
`test_qed_backend_auto.py` (4 tests, all plumbing) duplicates `test_qedmod.py:386` (`test_qed_se_method_defaults_to_auto`) and `:392` (`test_mss_alias_keeps_mohr_output_path`). `test_qed_indelicato_vs_fac.py:45` asserts FAC values 1.78189 (Z=54) and 1.47251 (Z=92) that `test_qed_self_energy_table.py:85` already asserts, and `:101` re-checks the *same* `FAC_F` numbers as `:75` with a weaker band (≤ 0.5 % residual vs rel 1e-3 on the same quantity). Note the Z=37 policy that `test_qed_backend_auto.py` pins is now physically inert — both backends read one table — so `test_qed_self_energy_table.py:106-119` is the only meaningful statement about it.
*Remediation:* collapse `test_qed_backend_auto.py` into one parametrised test of `resolve_se_method`; drop `test_leading_1s_residual_percent_vs_fac_table_is_pinned` or fold its numbers into the `GROUND_LEVELS` gate.

### S4

**F18 — `test_higher_multipoles.py:101-116`: the docstring's claim is false.**
`test_he_like_m1_2s_to_1s_z_scaling` feeds ω·Z² and S·Z⁴ and asserts `A1/A0 == Z¹⁰`. That ratio is `(Z²)³·Z⁴` regardless of the prefactor, which cancels exactly; the docstring's "Catches errors in the (2L+1)(L+1)/[L(2L+1)!!²] prefactor for L = 1" is therefore untrue — any multiplicative error in `einstein_A_mL` leaves the test green. `test_mL_matches_eL` (:51) asserts a definitional identity (the two functions share a body), and `test_oscillator_strength_e1` (:119) re-implements the one-line formula in the test.
*Remediation:* delete `test_he_like_m1_2s_to_1s_z_scaling`; gate the magnetic prefactor against Grant Eq. 8.4.10 with a *value* (the `_exact_A_eL` helper at :31 already transcribes it — reuse it for `einstein_A_mL` with a non-trivial S).

**F19 — `test_higher_multipoles.py` does not cover the Breit multipole expansion.**
The file name suggests the L-expansion whose two families (`v = L` and `v ≠ L`) are the locus of the measured scalar-Breit defect; in fact it is entirely about Einstein A-coefficients of E2/M2 transitions. Nothing in the entire slice asserts anything about a Breit photon multipole channel. Measured split (my own monkeypatch of `_breit_smu_coefficients`, same technique as `docs/VERIFICATION.md:143-155`): He total 1.9156e-04 = v=L 1.9155e-04 + v≠L 7.65e-09; Ne total 5.0437e-02 = v=L 5.0395e-02 + v≠L 4.247e-05. Against DBSR_HF (He 6.37774e-05, Ne 1.6643629692418e-02) the `v = L` family is 3.00×/3.03× too large and the `v ≠ L` family is ~16× too small in absolute terms (4.247e-05 vs the ~6.96e-04 Ha by which the Gaunt-only operator exceeds DBSR_HF), which is the ~50× of `VERIFICATION.md` once expressed as a fraction of the total (0.08 % of 0.0504 vs the required 4.2 % of 0.0166).
*Remediation:* in `test_breit_operator.py` (F1), add the split as a gate: with `v = L` only, Ne must equal the `breit_ci` Gaunt value 0.017340 Ha (rel 5 %); with `v ≠ L` only, Ne must equal DBSR_HF minus the Gaunt value ≈ 6.96e-04 Ha (rel 20 %). Those two numbers are already measured and written down in `docs/VERIFICATION.md`.

**F20 — stale/incorrect descriptions that no test reads.**
`corrections/adapters.py:138` still describes `UehlingVPCorrection` as *"V_VP = −(2Zα/3π)/r · K₁(2cr)"* — the exact defect fixed in `200d657`; `adapters.py:249` still describes NMS as *"ΔE = −(m_e/M)·⟨T_e⟩"*, the sign defect fixed in `9ee0df5` and gated by `test_property_regressions.py:204`; `test_qed.py:98` says He Breit is "~10^-5 Hartree" (actual 1.9e-4). `MohrBackend.description` (`mohr.py:117`) still advertises the removed "hydrogenic n^-3 fallback".
*Remediation:* fix the four strings and add a one-line test asserting `UehlingVPCorrection.description` does not contain "K₁"/"K_1" — cheap, and it is exactly the class of silent doc-drift this audit exists to catch.

---

## 4. Model tests (imitate these)

1. **`tests/test_qed_uehling.py:126` `test_hydrogen_1s_vp_matches_published_expansion`** — the standard to copy. It names the source (Yerokhin & Shabaev 2015 Eqs. 20-21), *derives* the leading coefficient in-test (`E_leading = α³/π · (−4/15) · HARTREE_MHZ` → asserts `≈ −217.0 MHz`), then asserts the full value at `rel=2e-3`. The 3.9× kernel error is 392× outside the band, and the derivation is independent of atomx.
2. **`tests/test_qed_uehling.py:56` `test_tabulated_values`** — five literals of the defining integral. I reproduced all five with my own `scipy.integrate.quad` (max rel. dev. 1.2e-7), so this is A-class in the strongest sense.
3. **`tests/test_qed_uehling.py:148` `test_uranium_1s_vp`** — one number, `rel=5e-3`, at the opposite end of the Z range from (1). My independent quadrature gives −97.9364 eV vs the pinned −97.9 (0.04 %); the docstring should say where the number comes from ("independent evaluation" is enough only because it is verifiable).
4. **`tests/test_qed_self_energy_table.py:62,71` `TestTableAgainstPublishedExpansion`** — the published Zα expansion with a *single* free constant `G = −30` reproducing Z=1 and Z=2, plus the honest statement of where the asymptotic series stops (Z ≤ 10, because freezing G costs 11 % at Z=26).
5. **`tests/test_qed_self_energy_table.py:170,181` `TestPhysicalAnchors`** — bounds from measurement (U 1s SE < the measured 460.2 eV Lamb shift; H 1s SE = 8396.5 MHz and `SE − VP − 6.6 = 8172.87 MHz`), i.e. the assertion is a physical inequality, not a printed number.
6. **`tests/test_qedmod.py:54` `test_hydrogenic_1s_matches_reference`** — the FAC Z=1 value `10.31679` quoted in-test with the formula spelled out (verified against `_qed_se0[0][0]`).
7. **`tests/test_breit_pauli.py:24,35,53`** — the α²Z⁴/32 Dirac 2p splitting, the α²/48 ξ, and the 2s₁/₂–2p₁/₂ degeneracy: all closed-form, asserted at `rel=1e-9`, and the same numbers reappear end-to-end at `:69` and `:129`. (`:35` is a *derived* degeneracy, the cleanest kind of A-class gate.)
8. **`tests/test_property_regressions.py:176-198`** — the `A = 6.6703e15·gf/(g_u·λ²)` identity asserted on every public-API row, i.e. a rubric-named exact theorem used as a cross-cutting invariant; the pre-fix defect (factor `(2j_u+1)/(2j_l+1) = 2`) is 1000× outside `rel=1e-3`.
9. **`tests/test_nist_levels.py:179` (as a *pattern*)** — an S1-class external disagreement written as `xfail(strict=True)` with the full measurement in the reason, so fixing the code converts it to XPASS-failure. The pattern is right; the placement (in the NIST-comparison file rather than in the slice that owns the number) is what let `test_qed.py` keep pinning the wrong value.

---

## 5. Appendix — evidence per test function

One row per test **function** (a parametrised function shares one row; the mix in
§2 counts items). `A` = first-principles, `B` = external oracle, `C` = internal
cross-check, `C*` = C-degenerate, `D` = frozen self-regression, `E` = tautology,
`F` = plumbing.

### `tests/test_qed.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| 47 | sign of V_VP at r = 1e-3…1e-2 | `V < 0` | A | the leading-order Uehling potential is attractive (kernel I > 0, `qed.py:159`). |
| 54 | range of V_VP | `abs(V(10)) < 0.01·abs(V(0.1))` | A | exponential decay on the Compton scale 1/(2c) ≈ 0.0036 Bohr; actual ratio ≈1e-60, so the bound is 100× loose but not wrong. |
| 60 | Z-linearity of V_VP | `V(10) == 10·V(1)`, rel 1e-10 | A | exact: `prefactor = −2Zα/3π` (`qed.py:165`), so the Z-dependence is a pure factor. |
| 70 | sign of E_SE, Z=2 | `E_SE > 0` | A | Lamb shift raises s levels; comment names the "shift added to Dirac–Coulomb" convention. |
| 83 | E_SE(20)/E_SE(10) | `10 < ratio < 25` | A | Z⁴ law; **measured 11.160 now and 17.32 with the pre-fix table** → cannot see C2 (F5). |
| 97 | He E_Breit | `abs < 0.01` | D | measured 1.9156e-4, 52× margin (F6). |
| 103 | He E_QED | `abs < 0.01` | E | measured 2.78e-5, 360× margin — no physics can fail it. |
| 108 | He total energy | `approx(−2.862, abs=0.05)` | D | band 250× the whole correction; a self-regression on a DHF total. |
| 112 | Ne E_Breit | `approx(0.0504, rel=0.05)` | D (claimed B) | measured 0.05044859084011072; DBSR_HF 0.016643629692418 is 3.03× outside → pins the defect (F2). |
| 139 | Ne E_VP | `approx(−0.000601, rel=0.05)` | D | measured −0.0006009391596706532; the K₁ value −0.002355 is 3.9× outside, so it does gate C1 — but nothing external anchors it (F14). |
| 151 | Ne E_SE | `!= 0` | E | presence check. |
| 156 | Ne total energy | `approx(−128.6136, abs=0.05)` | D | measured −128.62995892582862, deviation 0.0164 → passes before *and* after C1/C2 (F4). |
| 166 | signs of E_VP, E_SE | `< 0`, `> 0` | A | attractive VP, repulsive Lamb shift. |
| 205 | default QED shift, Z=10,18 | `PINS` rel 1e-3 | D | self-pin (docstring admits it); has real power — a 3.9× VP or 2.5× SE error moves the shift by >10 %. |
| 218 | SE, VP and their sum, Z=10,18 | rel 1e-3 + `se+vp == shift` | D+E | the sum identity is true by construction (`stack.py:192-206`); also pins `backend == "mohr"`, inert now that both backends share a table (F17). |

### `tests/test_qedmod.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| 31 | registry contents | membership | F | — |
| 38 | backend `.name`/`.description` | non-empty | F | — |
| 44 | unknown backend | `KeyError` | F | — |
| 54 | H 1s E_SE | `α³·10.31679/π`, rel 1e-10 | B | FAC `_qed_se0[0][0]`; verified in `coulomb.c`, formula spelled out. |
| 68 | default dispatch == `method="mohr"` | rel 1e-14 | E | the same code path twice. |
| 81 | Welton without orbitals | `== 0.0` | F | graceful-zero contract. |
| 87 | He Welton E_SE | `1e-6 < E < 1e-3` | E | 1000× band, no anchor. |
| 102 | H 1s FG vs Mohr | `0.9 < ratio < 1.1` | C | independent kernels (FG local potential from PRA 72 052115 vs the FAC table); 10 % is the paper's own claim. |
| 119 | FG without bsp_data | `ValueError` | F | — |
| 130 | WK vs Uehling at Ne | sign opposite, `abs(E_WK) < 0.05·abs(E_U)` | A | Flambaum–Ginges eq. 12: WK is a ~1 % higher-order term, opposite in sign. |
| 149 | E_QED_total = VP+SE+WK | abs 1e-15 | E | true by construction (`qed.py:383-389`). |
| 162 | E_WK through the SCF flag | key present, `> 0` | F | — |
| 178 | `shabaev` backend | `NotImplementedError` | F | — |
| 190 | `shabaev_local` vs `mohr`, H 1s | rel 2e-3 | C* | `A_κ` is calibrated on `_interpolate_se_f` (`shabaev.py:43,254`) — the docstring itself says "By construction" (F12). |
| 211 | Ne: `|E_s − E_m| < 10 %` and `< |E_fg − E_m|` | 10 % | C* | same calibration; the second clause is true by definition. |
| 234 | κ=−4 orbital | `== 0.0` | E | asserts a silent zero for uncovered κ. |
| 251 | SCF with `qed_se_method="shabaev_local"` | `E_SE > 0`, key recorded | F | — |
| 264 | 3d5/2, Z=22 | `!= 0` both, same sign | E | no magnitude. |
| 290 | hydrogenic Dirac 1s | `E ≈ −0.5` abs 1e-4, norm ≈ 1 abs 1e-4 | A | exact Dirac/normalisation; `E_1s = −0.5 − O(α²)` ✓. |
| 317 | Dirac vs NR ⟨r^-…⟩ at Z=1 | rel 1e-2 | C | two independent analytic integrals (`hydrogenic_r_exp_integral_dirac` vs `shabaev._hydrogenic_r_exp_integral`). |
| 327 | same at Z=50 | `I_d > I_nr`, 2 %–30 % | C | (Zα)² ≈ 0.13 shift, order of magnitude stated. |
| 342 | F(3s, Z=10) | `approx(4.9524, rel=1e-3)` | B | FAC `_sd` row `3s+`; **verified in `coulomb.c`**; plus the n⁻³F ordering argument. |
| 363 | F(3p1/2), F(3p3/2) at Z=10 | `abs < 1` | E | band only. |
| 371 | F(3d) at Z=50 | `< 0.5`, `< 0.1` | E | band only. |
| 386 | `solver.qed_se_method == "auto"` | string | F | duplicate of `test_qed_backend_auto.py:18`. |
| 392 | `resolve_se_method("mss", 92) == "mohr"` | string | F | duplicate of `test_qed_backend_auto.py:14`. |
| 398 | Welton end-to-end | `E_SE > 0`, key | F | — |
| 410 | `shabaev` at solve time | `NotImplementedError` | F | — |
| 418 | Welton/Mohr, He | `0.5 < ratio < 2.5` | C | independent kernels, but a 5× band on a closed-form heuristic (F13). |

### `tests/test_qed_uehling.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| 56 | I(x) at x = 0.01/0.5/1/5/20 | 3.899512, 0.4843297, 0.1779336, 6.902014e-4, 3.693416e-11; rel 1e-6 | A | the defining integral `∫₁^∞ e^{−xt}(1+1/2t²)√(t²−1)/t² dt`; reproduced to ≤1.2e-7 with independent quadrature. |
| 59 | kernel vs `scipy.quad` | max rel < 1e-8 | C* | the spline is *built* from `_uehling_I_exact` (`qed.py:93-98`) → interpolation error only (F10). |
| 68 | I(1), I(0.01) vs K₁ | `< 0.5·k1(1)`, `< 0.05·k1(0.01)` | A | discriminates the fixed kernel from the old one by 3.4×/26×. |
| 77 | small-x behaviour | each decade adds `ln 10` ± 1e-5 | A | exact: `I(x) → ln(1/x) + O(1)` for the defining integral (K₁ would diverge as 1/x). |
| 84 | monotonicity | `all(diff < 0)` | A | `dI/dx = −∫ t·(integrand) < 0` since the integrand is positive. |
| 91 | V_VP sign and range | `V < 0`, `abs(V(10)) < 1e-30` | A | same physics as :47/:54; the 1e-30 bound matches `I(80) ≈ 4e-37/r`. |
| 97 | scalar vs array API | rel 1e-14 | E | same function, two call signatures. |
| 103 | r = 0 | `== 0.0` | E | guard. |
| 107 | Z-linearity | ratio = 2, rtol 1e-14 | A | exact factor in `qed.py:165`. |
| 126 | H 1s E_VP | `approx(−215.1 MHz, rel=2e-3)` **and** leading term `α³/π·(−4/15)·HARTREE_MHZ ≈ −217.0 MHz` | A+B | Yerokhin & Shabaev 2015 Eqs. 20-21; the −4/15 derivation is done in-test and is 392× outside the old kernel's value. |
| 148 | U⁹¹⁺ 1s E_VP | `approx(−97.9 eV, rel=5e-3)` | A | my independent quadrature of the same defining integral gives −97.9364 eV (0.04 %); should cite the decomposition it comes from. |
| 159 | |E_VP| growth with Z | `1e6 < ratio < 1e9` | A | Z⁴ trend is right but the band is 3 decades wide — measured 1.100e8, the 3.9× wrong kernel also passes (F9). |

### `tests/test_qed_self_energy_table.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| 62 | F(1,1s), F(2,1s) vs the Zα expansion with one constant | `G = −30`, rel 1e-3 | A+B | Yerokhin & Shabaev Eqs. 7-19 coefficients transcribed in-test; the pre-fix table needed `G = +9705` at Z=2. |
| 71 | F(Z,1s) at Z=5,10 | rel 2e-2 | A | same expansion, with the honest statement of where freezing `G` costs 11 % (Z=26). |
| 83 | F(1s) at Z=1,2,10,54,92 | 10.31679, 8.52833, 4.65413, 1.78189, 1.47251; rel 5e-5 | B | FAC `_qed_se0+_qed_se1`; **I parsed `~/code/ASAP/fac/faclib/coulomb.c` and all 112 grid values of the 1s row match exactly** (0 mismatches at 1e-9). |
| 89 | monotone fall to Z=54, ratio > 5, F(92) < 2 | — | A | the published F(Zα) has a shallow minimum near Z≈87 — a property of the analytic function. |
| 106 | continuity across the old Z=37 switch | `|f37−f36|/f36 < 0.02`, `1.02 < e37/e36 < 1.20` | A | physically F must be smooth in Z; catches the historical 6.3× jump. |
| 121 | `_interpolate_indelicato_f == _interpolate_se_f` | exact, Z ∈ {1,10,36,37,54,92,120} | E (C*) | `indelicato.py:51-53` is a one-line delegation (F15). |
| 128 | adjacent-Z smoothness | `< 25 %` per unit Z | A | same smoothness principle; measured real step < 2 %, so the bound is ~10× looser than needed. |
| 139 | 3d/4d F values | pairwise unequal; `F(3d5/2) > 0 > F(3d3/2)` at Z=92 | B | FAC `_sd` `3d±` rows (verified); the sign flip at Z≈117 makes "across the table" loose but the Z=92 case is right. |
| 153 | n=5 rows present and finite | — | A | coverage: FAC `_sd` is the only source for n=5; the old `F(1s)/n³` fallback is what this guards. |
| 161 | `klamaq_f` l- and n-dependence | unequal, `abs < 20` | A | FAC's `Klamaq` Bethe-log form is O(1) by construction. |
| 170 | U⁹¹⁺ 1s E_SE | `approx(355.1 eV, rel=5e-3)` and `0.5·460.2 < e < 460.2` | B | Gumberidze et al. PRL 94, 223001 (2005) measured Lamb shift, quoted in-test; the pre-fix table gave 5218 eV = 11.3×. |
| 181 | H 1s E_SE and the Lamb-shift identity | `8396.5 MHz` rel 2e-3; `SE − 215.1 − 6.6 ≈ 8172.87 MHz` rel 3e-3 | B | measured 1S Lamb shift + the now-correct VP term; the identity couples this file to `test_qed_uehling.py`. |

### `tests/test_qed_indelicato_vs_fac.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| 45 | F(1s) at Z=54/74/82/92 | `1.78314−0.00125`, `1.53568−0.00484`, `1.49569−0.00840`, `1.49092−0.01841`; rel 5e-3 | B | these are exactly FAC `_qed_se0[0][Z] + _qed_se1[0][Z]` — **verified in `coulomb.c`**; overlaps `test_qed_self_energy_table.py:85` at two points (F17). |
| 51 | `mohr_f == fac_f` | rel 1e-12 | E (C*) | `indelicato.py:51-53` returns the mohr function (F15). |
| 75 | E_SE for H/He/Li-like Xe, W, U | `approx(FAC run, rel=1e-3)` ×9 | B | FAC run outputs including the MOHRFIN rms-radius correction; the 1e-3 vs 5e-7 discrepancy is documented in-test. |
| 101 | residual vs the same `FAC_F` | `≤ 0.5 %` | E | recomputes the same comparison as :75 with a weaker band — duplicated coverage. |

### `tests/test_qed_orbital_shifts.py`, `test_qed_backend_auto.py`, `test_breit_pauli.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| qos:34 | Fe XVII 1s QED shift | `3.5 ≤ eV ≤ 4.4` | D | measured 3.9274 eV; docstring calls it an external-scale check but no FAC number is asserted (F11); would catch the 3.9× VP error (2.91 eV, outside). |
| qos:52 | `shifted_orbital_energies == orbital_energies + shifts` | abs 1e-14 | E | re-implements the one-line function `orbital_shifts.py:81`. |
| qba:5,9,14,18 | `resolve_se_method` strings; solver default | — | F | dispatch names only; duplicates `test_qedmod.py:386,392` (F17). |
| bp:16 | ξ₂ₚ(H) | `α²/48`, abs 1e-12 | A | Cowan's ξ = α²Z⁴/[2n³l(l+1)(l+½)] = α²/48 at n=2, l=1. |
| bp:24 | 2p₁/₂–2p₃/₂ splitting | `α²/32`, abs 1e-12 | A | exact Dirac fine structure: `E(2p₃/₂)−E(2p₁/₂) = α²Z⁴/32` Ha. |
| bp:35 | 2s₁/₂ vs 2p₁/₂ (mass-velocity + Darwin + SO) | `≈ 0` abs 1e-15 | A | the Dirac degeneracy of the n=2 j=½ pair (Sommerfeld), derived in the test. |
| bp:53 | ξ scaling | `Z⁴`, rel 1e-14 | A | exact Z⁴ in the hydrogenic formula. |
| bp:69 | '2p1' H end-to-end | key set `{2p1/2, 2p3/2}`, splitting `α²/32` rel 1e-9 | A | asserts the analytic value through the real solver, not a printed number. |
| bp:85 | He 1s² end-to-end | key set, `E == E_NR + Δmv + ΔD` rel 1e-12 | E | re-applies the very functions the code applies (one key-set assertion is F). |
| bp:107 | jj-split subshells, Z=10 | splitting `α²·10⁴/32` rel 1e-9 | A | same analytic law at a different Z; catches label/routing bugs without a printed value. |
| bp:142 | single 2p₁/₂ electron | `E_spin_orbit == e_so_minus` rel 1e-12; `E_total == ε(2p1/2)` rel 1e-9 | A | occupation-weighted SO sum is the definition of the physical contribution; the defect it guards produced exactly 0. |
| bp:163 | full 2p shell | `E_spin_orbit == 0` abs 1e-15 | A | closed-shell statistical average must cancel — a symmetry, not a printed number. |
| bp:185 | 2p₃/₂¹ alone | `E_spin_orbit == e_so_plus` rel 1e-12 | A | same identity at the other j. |
| bp:209 | custom label | no stale key | F | key hygiene. |

### `tests/test_corrections_stack.py`, `test_correction_stack_injection.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| cs:27 | the three default rungs run | keys present, `isfinite` | E | no value check on the Breit/SE/VP energies (F7). |
| cs:39 | `stack.total == Σ shifts` | rel 1e-12 | E | that *is* `CorrectionStack.total` (`stack.py:137-142`). |
| cs:49 | disabling SE | key set | F | — |
| cs:56 | provenance dict | `backend` string | F | — |
| cs:85 | unsupported flavour | `{"_skipped"}` | F | — |
| cs:92 | skip reasons | `"B-spline" in reason or "Z" in reason` | E | substring matching where the reason object would do. |
| cs:103 | total of a skipped-only stack | `0.0` | E | sum over an empty dict. |
| cs:130 | user correction honoured | `== 0.01` abs 1e-12 | F | protocol plumbing. |
| csi:22 | `E_QED_total == E_SE + E_VP` | rel 1e-12 | E | true by construction (`stack.py:192-206`). |
| csi:32 | stack-built vs flag-built Breit | rel 1e-12 | E (C*) | both call `BreitCorrection → compute_breit_energy`; asserts only that two spellings agree. |
| csi:51 | default Atom carries corrections | keys present | F | — |
| csi:61 | `corrections="none"` | keys absent | F | — |

### `tests/test_shooting_corrections.py`, `test_higher_multipoles.py`, `test_property_regressions.py`

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| sc:71 | all components present on shooting | keys + `E_QED_total == E_SE+E_VP` | E | presence only; the sum is by construction. |
| sc:84 | stack on a shooting result | not NaN | E | — |
| sc:96 | `shabaev_local` on a grid | `> 0`, per-orbital non-empty | E | — |
| sc:108 | total includes the shifts | `abs=5e-9` | E | the invariant is the definition of `finalize_with_corrections`; it does gate the seam, so keep as F if renamed. |
| sc:134 | shooting vs bspline per component | rel 2e-3 ×9 | C* | the two engines share `corrections/breit.py` and `corrections/qed.py`; only the radial representation differs (F3) — it certifies the 3.03× wrong Breit. |
| sc:146 | signs at He/Ne/Fe XVII | `E_Breit > 0`, `E_SE > 0`, `E_VP < 0` | A | correct signs of the three corrections (weak but genuine). |
| sc:161 | Schrödinger shooting | skipped, "Dirac" in reason | F | flavour contract. |
| hm:40 | `einstein_A_eL` vs Grant Eq. 8.4.10 | rel 1e-12, L = 1…4 | A | closed-form published prefactor, transcribed (`_exact_A_eL`) and checked at four L. |
| hm:51 | `einstein_A_mL == einstein_A_eL` | rel 1e-12 | E | `rates.py:107` *is* `return einstein_A_eL(...)` — a delegation asserted against itself. |
| hm:62 | A(ω ≤ 0) | `== 0` | E | guard. |
| hm:68 | A(S = 0) | `== 0` | E | guard. |
| hm:72 | ω-scaling | `2³, 2⁵, 2⁷` rel 1e-12 | A | the ω^(2L+1) exponent is the multipole order — catches an exponent typo. |
| hm:87 | A_E2/A_E1 | `(1/20)·α²` rel 1e-12 | A | prefactor algebra derived in the comment from Grant's formula. |
| hm:101 | Z¹⁰ ratio | rel 1e-12 | E | the inputs ω·Z², S·Z⁴ make the ratio `(Z²)³Z⁴` for *any* prefactor — the docstring's claim to catch a prefactor error is false (F18). |
| hm:119 | `gf`, `f` | `2ωS/3`, `gf/(2j+1)` rel 1e-12 | E | re-implements the one-line definition. |
| hm:129 | `oscillator_strength(ω=0)` | `(0.0, 0.0)` | E | guard. |
| pr:133 | Ly-α A(2p3/2) = A(2p1/2) and ≈ NIST | `rel 1e-3`; `6.2649e8 s⁻¹` rel 0.5 % | B | NIST/FAC A-coefficient quoted; the degeneracy of the two fine-structure components is A. |
| pr:149 | Ne 1s→2p ratio and the ω³S/(2j_u+1) identity | rel 0.01 / rel 1e-6 | A | the identity is derived in-test from the A formula; catches the factor-2 degeneracy bug. |
| pr:176 | A vs its own reported gf (single config) | `6.6703e15·gf/(g_u λ²)`, rel 1e-3 | A | exact theorem `A = 6.6703e15 gf/(g_u λ²[Å])`, one of the rubric's named identities. |
| pr:187 | same on every public E1 row | rel 1e-3 | A | same theorem, cross-cutting. |
| pr:204 | H 1s E_NMS | `+1.7917 THz` abs 1e-3 | A | `E_NMS = (m_e/M)⟨T_e⟩` with the virial ⟨T⟩ = −E; exact reduced-mass value. |
| pr:211 | H→D 1s–2s NMS from the analytic estimator | `+671.5554 GHz` rel 0.5 % | A | exact reduced-mass value; the measured 670.994 GHz is quoted and the 0.08 % residual explained. |
| pr:232 | same from solved levels | `+671.5554 GHz` rel 0.5 % | A | same, through the solver path — the sign defect it guards returned −671.5554. |

### `tests/test_nist_levels.py` *(added to the slice)*

| line | asserted quantity | literal / tolerance | class | justification |
|---|---|---|---|---|
| 66 | C V excitation energies move when Breit is on | `max|Δ| > 100 cm⁻¹` | A | a real Breit operator must split levels differentially (the uniform-scalar defect produced exactly 0); C V rms 11199 → 10652 cm⁻¹. |
| 179 | Ne scalar E_Breit vs DBSR_HF | `approx(0.016643629692418, rel=0.05)`, `xfail(strict=True)` | B | DBSR_HF `an=Z term=AV mbreit=1`; currently XFAILS at 0.050449 (3.03×) — the only external Breit gate in the suite (F8). |

