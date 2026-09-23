# S3 — QED / Breit remediation

Scope: the findings of `docs/test_audit/S3_qed_breit.md` (F1–F20), severity
order.  Every number below was re-measured in this session unless it is
explicitly attributed.

## 0. Headline

The S1 item was **the reporting**, and it is **fixed** — not xfailed:

| route | Breit that is APPLIED | `E_Breit` reported BEFORE | `E_Breit` reported AFTER |
|---|---|---|---|
| `Atom(10, ['1s2 2s2 2p6'])` (list ⇒ MCDHF CI) | **+0.017340451404 Ha** (Gaunt operator) | 0.050448592993 (the scalar — *never applied*) | **0.017340451404** |
| `Atom(10, '1s2 2s2 2p6')` (string ⇒ single-config) | +0.050437102577 Ha (the scalar) | 0.050437102577 | 0.050437102577 |
| `Atom(10, '1s2 2s2 2p6', engine='bspline')` | +0.050448590840 Ha (the scalar) | 0.050448590840 | 0.050448590840 |

`E_Breit` is now, on every route, the shift that is in `total_energy`; the
scalar is still inspectable on the operator route, under
`E_Breit_scalar_not_applied` (a key whose name says so) and in
`breit_provenance()`.  The applied physics is byte-identical before and
after the patch (§4).

The remaining defect is the **scalar itself**: 3.03x DBSR_HF for Ne and
3.00x for He, applied unmodified by every single-configuration route.
That cannot be fixed by the reporting change, and the two cheap local
repairs are refuted with numbers (§3); those routes therefore get
`xfail(strict=True)` gates carrying the measurement and the cause.

## 1. Findings

| # | finding | file:line | change | measured before → after | new gate |
|---|---|---|---|---|---|
| F1 | `breit_ci.py` names a validator that does not exist | `breit_ci.py:52` | docstring now names the validators that DO exist; the external anchor was added in `test_qed.py` | claim → accurate | `test_qed.py::TestBreitRouteReporting::{test_ne,test_he}_breit_operator_matches_dbsr_hf` (PASS) |
| F2 | fabricated "Grant 2007 tables for Ne I" pins the 3x-wrong scalar | `test_qed.py:112-137` | citation **deleted**; pin replaced by a DBSR_HF gate; applied operator value gated separately | band `[0.04788, 0.05292]` (excluded the correct value) → `approx(0.016643630840490, rel=5%)` | `test_ne_breit_scalar_matches_dbsr_hf` (**XFAIL strict**, carries 0.050448043234125516 vs 0.016643630840490) + 2 operator gates (PASS) |
| F3 | C-degenerate "cross-engine proof" of the wrong Breit | `test_shooting_corrections.py:134` | renamed `test_cross_engine_radial_numerics`; docstrings + block comment state the kernels are SHARED and the value is not validated | name + prose → same assertion, honest label | prose gate; DBSR_HF ratio quoted (0.050437 vs 0.016643630840490) |
| F4 | 81 %-of-budget band on the Ne total | `test_qed.py:156-164` | total band **deleted**; replaced by `DC(total of an independent solve) + E_Breit + E_SE + E_VP` | `abs=0.05` → `abs=1e-8` (2026 meas. residual 5.8e-10); docstring re-quoted post-fix SE/VP | `test_ne_total_with_corrections` (PASS) |
| F5 | Z⁴ band 10..25 cannot see the table defect | `test_qed.py:83-91` | ratio asserted **against FAC's F values**: `16·F(20)/F(10)` and `16·F(54)/F(27)` | band `(10, 25)` → `11.1597 ± 1e-3`; pre-fix table 17.32 FAILS | `test_se_scales_as_Z4` (`external_fac`, PASS) |
| F6 | He caps with 52x/350x margin | `test_qed.py:97,103` | He Breit → DBSR_HF gate; He QED → frozen `rel=0.02` (360x → 50x margin, and 0.02 discriminates both defect classes) | `abs<0.01` both → Breit XFAIL(strict), QED snapshot | `test_he_breit_scalar_matches_dbsr_hf` (XFAIL), `test_he_qed_small` (snapshot, PASS) |
| F7 | the file that applies the Breit adapter asserts nothing | `test_corrections_stack.py:27-47` | one anchored value per rung; `test_total_matches_sum_of_applied` **deleted** (it was the definition of `CorrectionStack.total`) | `isfinite` only → SE/VP pinned `rel=1e-3`, Breit gated vs DBSR_HF | `test_default_stack_rung_values` (PASS), `test_default_stack_breit_rung_matches_dbsr_hf` (XFAIL strict) |
| F8 | only external Breit gate is an xfail | `test_nist_levels.py:179` | **not touched** — file owned by another agent (see §5) | 1 xfail → 1 xfail (verified not XPASSing after the reporting fix) | — (duplicated here by design) |
| F9 | 3-decade VP Z-scaling band | `test_qed_uehling.py:159-168` | monotonicity kept (A); band replaced by a frozen `rel=0.02` pin; the analytic route is documented as unavailable at Zα=0.67 | `(1e6, 1e9)` → `1.0998587e8 ± 2%` (K₁ kernel 4.3e8 FAILS) | `test_vp_monotone_in_Z` (snapshot, PASS) |
| F10 | C-degenerate quadrature check mislabelled | `test_qed_uehling.py:59` | renamed `test_spline_interpolation_is_machine_accurate`; docstring names the physics gate | prose → accurate | docstring; `test_tabulated_values` unchanged |
| F11 | self-pin labelled "a real external-scale check" | `test_qed_orbital_shifts.py:34` | `@pytest.mark.snapshot` + docstring says what it is; FAC literal **not** added (§5) | unmarked D → marked D | marker + `-m "not snapshot"` can now exclude it |
| F12–F13 | `shabaev_local` vs `mohr` C-degenerate; Welton 5x band | `test_qedmod.py:190,211,418` | **not fixed** (§5) | — | — |
| F14 | Ne VP pin unmarked, no mid-Z anchor | `test_qed.py:139` | `@pytest.mark.snapshot`, band 5 % → **1e-3** | `rel=0.05` → `rel=1e-3` on -0.0006009529 (K₁ value -0.002355 FAILS) | `test_ne_has_vp` (snapshot, PASS) |
| F15 | self-comparison tests | `test_qed_self_energy_table.py:121`, `test_qed_indelicato_vs_fac.py:51` | **not fixed** (§5) | — | — |
| F16 | 10 unmarked D-class pins | `test_qed.py:97,103,108,139,151,205(×2),218(×2)`, `test_qed_orbital_shifts.py:34` | `@pytest.mark.snapshot` added to **9 items** (the ones still D-class); `:112` is now externally anchored (B) and `:156` is now a seam identity, so marking them `snapshot` would be a lie | 0 marked → 9 marked | `@pytest.mark.snapshot`, mirroring `test_fac_parity_gates.py:49` |
| F17 | duplicated coverage | `test_qed_backend_auto.py`, `test_qed_indelicato_vs_fac.py:101` | **not fixed** (§5) | — | — |
| F18 | `test_higher_multipoles.py:101` false docstring claim | same | **not fixed** (§5) | — | — |
| F19 | no v=L / v≠L Breit coverage | `test_higher_multipoles.py` | split numbers now gated in `test_qed.py`'s Breit xfail reason and documented in `corrections/breit.py`; the operator covers the real multipole sum | — | reason text + module docstring |
| F20 | stale descriptions | `adapters.py:138,249`, `qedmod/mohr.py:117`, `qed.py:189` | all four corrected; the two kernel/description drifts now have tests | `K₁(2cr)` → `I(2cr)`; `−(m_e/M)⟨T_e⟩` → `+⟨T_e⟩/M`; "hydrogenic n^-3 fallback" dropped | `TestCorrectionDescriptions` (PASS) |

Plus one finding the audit did not have, found while fixing F2: **the
`E_Breit` key meant two different things** (§2).  Fixed in
`methods/mcdhf/solver.py` (authorised by the orchestrator).

## 2. What the scalar-vs-operator Breit resolution actually was

### 2.1 The trap, measured

`src/atomx/methods/mcdhf/solver.py` computes the scalar
`compute_breit_energy` once on the converged orbitals and (in operator
mode) *reports* it while guarding the applied shift
(`if not breit_operator_in_ci and E_Breit: state_energies += E_Breit`,
line ~1090).  So for `Atom(10, ['1s2 2s2 2p6'])`:

```
E_Breit reported 0.05044859299279357   <- scalar, never applied
applied shift    0.017340451403924817  <- what moved total_energy
breit_provenance {'mode': 'operator', 'scalar_shift_applied': False}
```

A user reading `energy_components["E_Breit"]` got a number 2.91x the
applied one and 3.03x DBSR_HF, with nothing in the dict to warn them.

### 2.2 The fix (reporting only)

`MCDHFResult.energy_components` now reports `E_Breit` = the applied
shift, and the scalar under `E_Breit_scalar_not_applied`.  The applied
operator shift is *measured*, not assumed: after the SCF loop (outside
it, once) the same CSFs/orbitals/workspace are re-diagonalised without
the Breit operator and

```
E_Breit = E(CI with Gaunt) − E(CI without Gaunt)
```

on the ground state.  `breit_provenance()` gains `applied_shift_source`
so a reader can see where the number came from.  For the per-block
orbital scaffold (whose states come from per-block orbital sets a single
companion solve cannot reproduce) no operator number is invented:
`E_Breit` is then simply *absent* and the scalar sits under its
not-applied key.

### 2.3 DBSR_HF evidence

`dbsr_hf an=Z term=AV mbreit=1` (binary at `~/code/ASAP/dbsr_hf`), `Breit:`
line of each run:

| system | DBSR_HF | atomx scalar | atomx Gaunt operator |
|---|---|---|---|
| Ne 1s2 2s2 2p6 | **0.016643630840490** | 0.050448 (3.03x) | 0.017340451404 (**+4.19 %** = omitted retardation) |
| He 1s2 | **0.000063777432919** | 0.00019156 (3.00x) | 0.00006377915809 (**+0.003 %**, retardation negligible) |
| Co 3d7 4s2 (AOC) | 0.532518297380245 | 1.695447 (3.18x) | 0.569893903352 (+7.0 %) |

The Ne number is the orchestrator's run (`/tmp/atomx_n1/dbsr_ne/Ne.log`
line 338; the audit's 0.016643629692418 is a slightly different
convergence of the same run — both round to 0.016644).  The He and Co
runs are mine (`/tmp/atomx_n1/dbsr_he/He.log:77`,
`/tmp/atomx_n1/dbsr_co/Co.log:536`), as is the He operator value.

The He row is the decisive one: at Z=2 the retardation is negligible, so
the operator and DBSR_HF *must* agree and they do to 0.003 %, while the
scalar is 3.00x.  That rules out "the reference is wrong", "the operator
omits something big at low Z", and "the factor 3 is a high-Z effect".

### 2.4 Why the scalar was not repaired (option (b))

`corrections/breit.py` transcribes DBSR_HF's `SMU`
(`hf_energy_BR.f90:79-115`) but not the four things that go with it:

* DBSR's configuration angular coefficient `coef(i1,i2,j1,j2,k)`, built
  from the determinant expansion at `dbsr_lib_zconfjj.f90:4098`; atomx
  substitutes `oa·ob/((2j_a+1)(2j_b+1))`, which is a different average;
* the transposed angular configuration used for `i < j`
  (`hf_energy_BR.f90:22-26`, `SMU(ki,kj,kj,ki,…)`); atomx only ever uses
  the `i ≥ j` one;
* `L` from 0 (`hf_energy_BR.f90:28`), not 1;
* SMU's parity rule (`hf_energy_BR.f90:73-77`).

Two cheap local repairs were implemented and measured, and **both are
refuted**:

| variant | He 1s2 | Ne 1s2 2s2 2p6 | verdict |
|---|---|---|---|
| baseline | 1.916087e-04 | 5.044804e-02 | 3.00x / 3.03x DBSR_HF |
| + SMU parity rule | 1.916011e-04 | 4.881782e-02 | nowhere near |
| `v=L` coefficients ÷ (2L+1) | 6.387468e-05 (+0.15 % vs DBSR) | 1.658844e-02 | **refuted**: the validated operator gives Ne 1.734045e-02, so this is 4.3 % *below* the correct Gaunt value.  It lands near DBSR_HF only by cancelling the missing retardation against an over-corrected Gaunt term — a compensating error, exactly the failure mode this audit exists to remove |

The per-channel split (my own monkeypatch of `_breit_smu_coefficients`,
matching the audit's numbers): Ne `v=L` 0.050405570438, `v≠L`
4.247280e-05; He `v=L` 1.916011e-04, `v≠L` 7.651e-09.  So the two
families are wrong in opposite directions and no single coefficient
scaling fixes both.  `corrections/breit.py` now records this in its
module docstring so the next person does not repeat the two dead ends.

I did **not** land a coefficient tweak that makes one element match by
cancelling a different error.  The honest gate for those routes is a
loud `xfail(strict=True)`.

### 2.5 The route that has no operator at all (found during remediation)

The applied Breit depends on how the configuration is **spelled**, not
on the engine:

| config spelling | engine | applied Breit |
|---|---|---|
| `"1s2 2s2 2p6"` | shooting (default) | +0.050437102577 (scalar) |
| `["1s2 2s2 2p6"]` | bspline (default) | +0.017340451404 (operator) |
| `"1s2 2s2 2p6"` | bspline (forced) | +0.050448590840 (scalar) |
| `["1s2 2s2 2p6"]` | shooting (forced) | refused: multi-config needs the B-spline MCDHF machinery |

User-visible consequence: `Atom(10, '1s2 2s2 2p6')` and
`Atom(10, ['1s2 2s2 2p6'])` differ by **0.033096651 Ha = 0.9007 eV =
7264.6 cm⁻¹** in total energy while their correction-free energies agree
to 7.9e-6 Ha: the entire difference is the Breit.  Gated by
`test_qed.py::TestBreitRouteReporting::test_breit_does_not_depend_on_the_config_spelling`
(strict xfail carrying all four numbers).

## 3. Code changes and their independent verification

| file | change | verification |
|---|---|---|
| `corrections/breit.py` | `SCALAR_BREIT_NOTE` + `scalar_breit_provenance()` moved here (lower layer); module docstring records the defect, the DBSR_HF numbers and the two refuted repairs | note text checked against DBSR_HF runs cited in §2.3 |
| `corrections/adapters.py` | `BreitCorrection` provenance gains `externally_validated: False` + the note; Uehling description `K₁(2cr)` → `I(2cr)`; NMS description sign fixed | new `TestCorrectionDescriptions` gates the two descriptions; behaviour unchanged (provenance is metadata) |
| `corrections/stack.py` | `finalize_with_corrections` stashes `_breit_provenance` (the dict `MCDHFResult.breit_provenance` reads); new `breit_provenance(result)` accessor for the single-config results | `test_shooting_breit_provenance_names_the_unvalidated_scalar` (PASS) |
| `corrections/qed.py` | `_short_range_expectation` docstring `K₁(2cr)` → `I(2cr)` | docstring only |
| `corrections/qedmod/mohr.py` | `MohrBackend.description` drops the removed "hydrogenic n^-3 fallback" | docstring/string only |
| `methods/mcdhf/breit_ci.py` | "Validation" docstring names the real validators; re-exports `SCALAR_BREIT_NOTE` / `scalar_breit_provenance` from `corrections.breit` (import direction: `methods` → `corrections`) | import check: `SCALAR_BREIT_NOTE is` the same object; `RETARDATION_NOTE`/`BreitOperatorError` untouched |
| `methods/mcdhf/solver.py` | reporting fix of §2.2 (+ `applied_shift_source`) | **applied physics re-measured**: Ne `+0.017340451403924817`, He `+6.377915809485657e-05` — identical to the pre-patch values (0.017340451403924817 / 6.377915809485657e-05), and `reported − applied = 0.0` on both |

The reporting fix does cost one extra CI diagonalisation per MCDHF solve
*when Breit is on*, outside the SCF loop.  It is skipped entirely when
Breit is off (the default for `MCDHFSolver`), and for the per-block
scaffold.

## 4. Test counts

| file | before | after |
|---|---|---|
| `test_qed.py` | 17 passed | 27 passed, 3 xfailed |
| `test_shooting_corrections.py` | 19 passed | 19 passed |
| `test_qed_orbital_shifts.py` | 2 passed | 2 passed |
| `test_corrections_stack.py` | 8 passed | 8 passed, 1 xfailed |
| `test_qed_uehling.py` | 16 passed | 16 passed |
| **total (touched files)** | **62 passed, 0 xfail** | **72 passed, 4 xfailed** |

Every xfail is `strict=True` with the measured number and the named
cause, so each one becomes a *failure* the moment the corresponding code
is fixed.  Regression sweep of the rest of the slice
(`test_corrections_stack` … `test_fac_parity_gates`, 13 files):
**212 passed, 0 failed**.  `test_mcdhf.py` + `test_nist_levels.py`:
**20 passed, 4 xfailed** (the scalar-mode xfail there still xfails — no
XPASS).  `ruff check` clean on all ten touched files.

## 5. Findings NOT fixed, and why

* **F8 (`test_nist_levels.py:179`)** — the file is owned by another
  agent; the orchestrator forbade edits.  Two things it still needs:
  (a) the xfail reason is stale ("the default was flipped back to that
  wrong scalar" — `04cd274` made `breit_mode="operator"` the MCDHF
  default, and with the reporting fix the *scalar* is no longer what
  `E_Breit` means on the operator route); (b) the DBSR_HF literal
  0.016643629692418 is the audit's convergence — the orchestrator's
  re-run gives 0.016643630840490.  Neither changes the xfail's verdict.
* **F11 second half** — putting the FAC/sfac Fe XVII 1s self-energy in
  as a literal with `rel=0.02`.  The only number available is the prose
  "SE 4.28 − Uehling 0.35" in the test's own docstring; I could not
  independently verify it, and rule 4 forbids inventing a reference
  number.  Marked `@pytest.mark.snapshot` instead.
* **F12, F13 (`test_qedmod.py`)** — the `shabaev_local` vs `mohr`
  comparisons are calibrated by construction, and fixing them means
  comparing against QEDMOD v7's own table (`~/code/ASAP/qedmod_v7`),
  which is a backend-porting job, not a test edit.  Out of this slice's
  budget.
* **F15, F17 (`test_qed_self_energy_table.py:121`,
  `test_qed_indelicato_vs_fac.py:51,101`)**, **F18/F19
  (`test_higher_multipoles.py`)** — duplicated/self-comparison coverage
  and the false M1-scaling docstring.  Deleting them is safe but they
  are documentation of past fixes; not reached in this pass.  F19's
  requested home (`tests/test_breit_operator.py`) is another agent's
  file: what it needs is a closed-shell total gate — that now exists in
  `test_qed.py::TestBreitRouteReporting`, so adding a second copy there
  would be duplication.
* **F16 tenth item** — the audit's "10 unmarked D-class" resolves to 9
  *still* D-class items after the F2/F4 fixes (`test_qed.py:97,103,108,
  139,151,205×2,218×2` = 8 + `test_qed_orbital_shifts.py:34`).  All 9 are
  marked.  `test_qed.py:112` is now a strict xfail against DBSR_HF (B,
  not D) and `:156` is a seam identity, so marking either `snapshot`
  would be false.

## 6. Collateral the parent must resolve (not mine to edit)

`tests/test_averaging_decoupling.py:156` now fails:
`comps["E_Breit"] == approx(1.6974, abs=5e-3)` obtains
**0.5698939033522947**.  The pin was the *scalar that was never applied*
on that route (`Atom(Z=27, engine="shooting", averaging="ci")` goes
through `MCDHFSolver` with the operator in the CI).  The applied Breit is
unchanged by my patch — measured `E_Breit + E_QED_total = 0.5698939 +
0.3821873 = 0.9520800` equals the `(with corrections) −
(corrections="none")` total difference 0.9520799656106647.  The file is
not in my slice, so I did not touch it; the honest re-pin is
`0.5698939033522947` (indicative external anchor: DBSR_HF Co 3d7 4s2
gives 0.532518297380245, +7.0 %, consistent with the omitted
retardation, whereas the old 1.6974 is 3.19x it).

Also still needed, outside the scope I was given:
`ShootingResult` (`numerics/shooting/solver.py:798`) and `SCFResult`
(`methods/dirac_fock.py:88`) have no `breit_provenance` property, so the
scalar note is reachable only through
`atomx.corrections.stack.breit_provenance(result)` or
`BreitCorrection().evaluate(result).provenance`.  A four-line property
delegating to the stashed `_breit_provenance` would close that.

## 7. Follow-up: the scalar Breit was FIXED (not deleted), and now cross-validates the operator

An intermediate directive ("operator only, delete the scalar") was
reversed: the user wants the scalar kept.  The scalar is therefore
**restored and fixed**, and it is *complementary* to the operator rather
than redundant with it.

### 7.1 Withdrawn: the "(2L+1) hypothesis is refuted" claim

I withdraw it.  My refutation argued that dividing the ``v = L``
coefficients by ``(2L+1)`` "matches DBSR_HF only by cancelling the
missing retardation".  That argument fails on He: the 1s2 retardation is
nil — the Gaunt-only operator already reproduces DBSR_HF to 5e-9
relative there — so there was nothing for a wrong factor to cancel, yet
the corrected scalar lands +0.13 % from DBSR_HF.  The correction does
real work.

The root cause is the ``(2L+1)`` of the addition theorem for the
Racah-normalised harmonics the ``S^v`` radial integrals use,

    P_L(cos θ12) = (4π/(2L+1)) Σ_q Y_L^q(1) Y_L^{q*}(2),

while DBSR's ``SMU`` (``hf_energy_BR.f90:82``) is written for the
unnormalised ``Y_L`` contraction.  Landing it is one line — divide
``results[L]`` by ``(2L+1)`` in ``_breit_smu_coefficients``, and the
derivation now sits in that function and in the module docstring.  The
remaining difference from DBSR is that ``coef(i,j,k)``
(``dbsr_lib_zconfjj.f90:4098``) is still replaced by the
``oa·ob/((2j_a+1)(2j_b+1))`` weight, plus the ``v ≠ L`` channels being
~50x too small and ``L`` starting at 1 instead of 0.  The closed form of
``coef`` was **not** established: GRASP's ``breit.f90`` is a driver over
generated ``genintbreit1/2`` angular integrals and carries no explicit
factor, so the ``(2L+1)`` reading rests on the three-element ratio
measurement, not on a source.

### 7.2 The three-element acceptance set (all vs DBSR_HF ``mbreit=1``)

| Z | scalar as-is before | scalar after ÷(2L+1) | Gaunt operator | DBSR_HF |
|---|---|---|---|---|
| 2 He 1s2 | 1.9156e-04 (3.004x) | **6.38497e-05 (+0.11 %)** | 6.37792e-05 (+0.000005 %) | 0.000063777432919 |
| 10 Ne 1s2 2s2 2p6 | 5.0437e-02 (3.030x) | **1.65886e-02 (−0.33 %)** | 1.73405e-02 (+4.19 %) | 0.016643630840490 |
| 18 Ar 1s2…3p6 | 4.1004e-01 (3.098x) | **1.33235e-01 (+0.66 %)** | 1.40594e-01 (+6.22 %) | 0.132364619357042 |

(Scalar values on the B-spline DHF orbitals; the shooting flavour gives
6.38569e-05 / 1.65849e-02 / 1.33204e-01.)

So the **fixed scalar is the better closed-shell total** (it keeps the
``v ≠ L`` retardation channels) while the **operator is the only one with
the non-scalar content that splits CI levels**.  That is the architecture
that is now in place, unchanged from before except for the bug:

* single-CSF closed shells (`atom.py:824-836 use_ci_avg`, `len(csfs)==1`)
  → DHF + `CorrectionStack` → the **fixed scalar**;
* multi-CSF / open shells → `MCDHFSolver` → the **operator** in the CI
  blocks (this was already true; the earlier "spelling" framing was
  imprecise — the discriminator is the CSF count);
* `breit_mode="scalar"` is a legitimate comparison path again.

### 7.3 Mutual validation (the prize)

The two implementations share no angular kernel — DBSR ``SMU``
transcription plus ``S^v`` radial integrals on one side, 3j-based Racah
algebra plus Gaunt photon multipoles on the other — so their agreement
is independent evidence.  Every single-CSF run now reports **both**:
the scalar is applied and the operator is attached to the result
(`breit_ci.attach_breit_operator_energy` → ``_E_Breit_operator``), and
the adapter's provenance carries ``gaunt_operator_cross_check`` and
``operator_vs_scalar_ratio``.  Measured ratio (operator/scalar): He
0.99877, Ne 1.04533, Ar 1.05519 — i.e. the documented omitted
retardation, not a disagreement.  Gates:
``test_qed.py::TestBreitRouteReporting::test_breit_spellings_are_two_implementations_both_anchored``
(both routes vs DBSR_HF, and the ratio confined to 1.02–1.07) and
``test_routes_report_the_implementation_that_ran``.

### 7.4 The `np` NameError, and why my run did not catch it

`breit_ci._ci_radial_inputs` used `np.zeros_like` in the same edit that
introduced it; the import was added in the *next* command, so the tree
was broken between the two.  My verification ran after the import
landed, so it was green while the orchestrator's repro hit the
intermediate revision.  That is a sequencing failure, not a coverage
one — but the coverage point stands and is now closed: the acceptance
set below exercises `Atom(2, '1s2', corrections=Corrections(breit=True))`,
the exact path that raised, and `ruff check` (F821) runs on every touched
file in the same command as the tests.

### 7.5 Acceptance, current tree

```
Z=2   str/shoot 6.385693e-05 (scalar, +0.13 %)  str/bspl 6.384971e-05  list/bspl 6.377916e-05 (operator, +0.003 %)
Z=10  str/shoot 1.658490e-02 (scalar, -0.35 %)  str/bspl 1.658861e-02  list/bspl 1.734045e-02 (operator, +4.19 %)
Z=18  str/shoot 1.332044e-01 (scalar, +0.63 %)  str/bspl 1.332345e-01  list/bspl 1.405943e-01 (operator, +6.22 %)
```
On every route ``E_Breit`` equals the shift applied to ``total_energy``.
There is no longer a 3x number anywhere.
