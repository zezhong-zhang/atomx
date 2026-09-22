# S4 remediation — atomic properties and transition rates

Slice: `docs/test_audit/S4_properties_transitions.md` (27 findings).
Agent: remediation pass 2026-09-25.  Every change is test-side unless the
row says otherwise; three `src/` **docstrings** were corrected (no code
behaviour changed anywhere).

Verification: `pytest` over the 17 touched files →
**300 passed, 10 xfailed** (52 s); every `xfail` carries `strict=True`
explicitly and a measured number + named cause.  `ruff check` clean on all
17 test files and the 3 `src/` files.

Tolerance policy: no gate was loosened.  Where a tightened gate failed, the
test became `xfail(strict=True)` with the measurement in the reason (never
a silent skip), or the underlying value was re-derived from a closed form.
No expected value was taken from atomx output unless the test is marked
`@pytest.mark.snapshot` (and says so in its docstring).

---

## 1. Finding-by-finding table

| # | file:line (pre-fix) | change | measured before → after | new gate |
|---|---|---|---|---|
| 1 | `tests/test_transitions.py:239` | add `TestHydrogenicLyAlpha` (NIST/known H 2p→1s A + Ly-α degeneracy); drop the false "gauge consistency" docstring line | strongest A assertion was `A > 0` — passed the pre-fix 1.253677e9 → A(2p1/2)=6.268313e8, A(2p3/2)=6.268384e8 s⁻¹ | `approx(6.2649e8, rel=2e-3)` (measured 5.6e-4 off) + `A_32/A_12 = 1.0000114` (pre-fix 2.0000) + explicit exclusion of 1.253677e9 |
| 2 | `tests/test_polarizability.py:26` | replace `0.5 < α < 1.6` with a basis-convergence/plateau test + a strict xfail against the exact value | 0.892802 at (κmax=2, n_virt=15) — **not** basis-truncated: 1.093864 (3,30) → 1.112394 (4,40) → 1.117768 (5,60) → **1.117810 (6,80) = 1.117810 (7,100)**; exact 1.383192 (Mitroy-Safronova-Clark 2010) | plateau test passes (`α(2,15)<α(3,30)<α(6,80)`, `α(7,100)=α(6,80)` to 1e-5); `xfail(strict=True)`: 1.117810 is 19.2 % low |
| 3 | `tests/test_stark.py:52` | `-10 < dE < 0` → exact −½αF² | band admitted −1e-9 → −5.59871678619299 MHz vs closed form −5.5987165519 MHz | `approx(-0.5·4.5·(F/AU_FIELD)²·HARTREE_MHZ, rel=1e-6)` (residual 4.2e-8 = the src CODATA-2018 literals) |
| 4 | `tests/test_hyperfine.py:279,375` | mark both preset assertions `@pytest.mark.snapshot`; add the untuned reduced-mass gate | preset is *fitted* to 1420.403 (circular) → new: `(1+m_e/m_p)^−3 = −1632.073 ppm` from shipped CODATA, `A_FNM = A_DHF·factor` (rel 1e-12), `A_FNM = 1418.9532307` | `approx(-1632.1, abs=0.5)` ppm, `approx(1418.9532, abs=2e-3)` MHz, raw-DHF gap `610.46 ppm` (±0.05) |
| 5 | `tests/test_M1_E2_rates.py:493` | delete the 3.94e6 pin; two external-anchored strict xfails (+O VII point) | 3.94e6 pin (old broken CI) and current 1.3997e6 are both wrong: anchor 9.74e3 (Z-scaling of the measured O⁶⁺ 1046 s⁻¹) and 3.7178e5 measured at Z=8 | `xfail(strict=True)` vs `approx(9741.6, rel=0.15)` and vs the measured `1046` — see §3 |
| 6 | `tests/test_sternheimer_edm.py:122,230` | 2 red tests → `xfail(strict=True)` + new executable blocker test | residual 1.877e-07 at iter 11 (**53× inside** tol=1e-5); real cause `max_same_kappa_overlap = 1.702e-03 > 1e-03` → `converged=False` → `sternheimer_edm_enhancement` raises | blocker test asserts residual<tol, `not converged`, overlap>threshold, and the ValueError — R_e(Tl) is loudly ungated |
| 7 | `tests/test_sternheimer_edm.py:181` | `isinstance(ratio, float)` → measured value + a falsifiable inequality; tighten H band; fix the src docstring | `ratio = 42.85` (R_e 0.0304260 vs Sandars 7.10018e-4) against a docstring claiming "factor-of-2-3"; `1e-6 < R_e < 1` → 0.0304260 | `ratio > 5` + `approx(42.85, rel=0.05)` (snapshot); `approx(0.0304260, rel=0.05)`; `sternheimer.py` docstring now names 42.85 |
| 8 | `tests/test_M1_E2_rates.py:588` | `gauge_ratio > 0.90` → `abs(gauge_ratio - 1) < 0.10` | accepted a velocity rate 1000× the length rate → still passes at the measured level | two-sided 10 % band |
| 9 | `tests/test_edm.py:17` (+`scaling.py:63`) | Cs R_e: 16× window → Liu-Kelly factor-of-2 + frozen pin; label the ratio tests as formula pins; src docstring "R_e ≈ 80" → 224.45 | 224.4456 measured vs the docstring's 80 (2.8×) and Liu-Kelly 120 (1.87×) | `60 < R_e < 240` (factor 2), `approx(224.4456, rel=1e-4)` (snapshot), `abs(R_e-80)/80 > 1` |
| 10 | `tests/test_pnc.py:17,60` | `Q_W < -70` → Standard-Model closed form; `isfinite` → closed-form kernel + exact identities | −73.2205 measured (old gate accepted −1000); the old radial-ME fixture gives **−0.0** (the `isfinite` check could not see a zero) | `approx(-73.2205, abs=1e-3)` + SM checks of the shipped Q_W(p/n); `ME = G_F/(2√2)·Q_W·c²` for a normalized density (rel 1e-12) + antisymmetry + Q_W-linearity |
| 11 | `tests/test_hyperfine.py:311-358` | add the missing B magnitude gate + the never-referenced `BARN_BOHR2` round-trip | B was gated only by `B(1/2)=0`, `B(2Q)=2B(Q)`, `B≠0` → `⟨r⁻³⟩(2p3/2,Z=1) = 0.041667869` vs exact 1/24; `B(Q=1 b) = -5.951946e-10` vs closed form −5.951775e-10 (2.9e-5 = the (Zα)² shift) | `approx(1/24, rel=1e-3)`, `approx(-(2/5)·barn·⟨r⁻³⟩, rel=1e-3)` (pins prefactor 2/5, sign, barn conversion), `BARN_BOHR2 = 1e-28/a₀²` |
| 12 | `tests/test_polarizability.py:58-68` | delete the private-helper re-derivation; exercise the public `include_bbr=True` path | BBR had zero coverage → α(H 1s)=4.499413 (exact 4.5); static shift −5.888017e-18 vs −½⟨E²⟩·4.5 = −5.888785e-18; dynamic/static = 1.000107 | `approx(4.5, rel=1e-3)`, `approx(-½E²·4.5, rel=1e-3)`, dynamic/static `approx(1, rel=1e-3)` |
| 13 | `test_gauge_invariance.py:71,88`; `test_velocity_gauge_full_order.py:48-110` | tighten 7–8 orders | hydrogenic Z=1,2,3,5,8: 2.08e-11 / 3.84e-12 / 1.09e-12 / 3.37e-12 / 3.55e-11; per-rank L=1…4: 4.2e-10 / 2.9e-10 / 4.3e-07 / 3.7e-04 | `dev < 1e-9` (was 1e-3/5e-3); flat 1e-2 → `{1:1e-8, 2:1e-8, 3:1e-5, 4:1e-3}` with the measured value in the failure message |
| 14 | `tests/test_gauge_invariance.py:217` | 0.20 → the Gu 2008 value 0.10 the docstring already quoted; refresh the Z=2 xfail number | He-like Z=10 measured 0.0369 (2.7× margin); Z=2 measured 0.234 (the reason cited a stale 0.389) | `ratio_dev < 0.10`; Z=2 stays `xfail(strict=True)` |
| 15 | `test_two_photon_vs_literature.py:43` | flat 1 % → per-Z tolerance | 3.77e-5 (Z=1), 1.23e-4 (Z=2), 2.63e-4 (Z=3), 7.06e-4 (Z=5); H vs Goldman-Drake 8.229: 8.228832 | `{1:1e-4, 2:3e-4, 3:6e-4, 5:2e-3}`; H `approx(8.229, rel=2e-4)` |
| 16 | `test_two_photon_vs_literature.py:93-96` | delete the vacuous He "order of magnitude" test (10 < val < 200) | replaced by the existing 10 % He gate at the same Z | — |
| 17 | `tests/test_isotope.py:331-342` | gate the derivation's own units, not a rounded MHz literal | F_Ha(1s) = 0.6669946 (4.9e-4 from 2/3) inside a ±1.3 % band | `approx(2/3, rel=1e-3)` + explicit Ha/Bohr² → MHz/fm² round-trip (rel 1e-12) |
| 18 | `tests/test_edm.py:16-17` | see #9 | 224.446 vs 30 < R_e < 500 | factor-of-2 vs Liu-Kelly 120 |
| 19 | `tests/test_isotope.py:246-248,264-292` | pin the converged K_SMS; add a two-basis convergence test; tighten the momentum-gauge residual | K_SMS(Ne) = −117.8527221416 (ns=50), −117.8527221418 (70), −117.8527221420 (90); gauge residual 7.26e-2 | `approx(-117.852722, rel=1e-6)` (snapshot) + `K(50)=K(70)` to 1e-9; residual `< 0.08` (was 0.1) |
| 20 | `tests/test_zeeman.py:93-97` | record the \|κ\|≥2 NR fallback where it matters | κ=−2 returns 4/3 at Z=1 **and** Z=92 (no Dirac correction); the implemented κ=−1 channel at Z=92 is 1.654846, i.e. 17.3 % below its NR value | asserts the fallback exists at Z=92 (fails loudly if replaced) + `approx(1.654846, rel=1e-6)` |
| 21 | `test_transitions_engine_uniform.py:1-11` | relabel: this file gates the two SCF **orbital sets**, not the machinery (both engines share `radial_eL`); add a kernel oracle | the file was C-degenerate for its own claim; new oracle: `radial_eL(P,Q,P,Q,·,1)=⟨r⟩`, `L=2 → ⟨r²⟩` | `⟨r⟩ = 1.5/6.0`, `⟨r²⟩ = 3.0/42.0` (Bethe-Salpeter closed forms) at `rel=1e-3`; measured −1.8e-5…−3.6e-5 |
| 22 | `test_two_photon.py:16-68,82-90` | mark the Drake-fit/spectrum tests `snapshot`; delete the dead `_H_LIKE_REF`; use the shipped `ALPHA`; **new** shape gate + src docstring | the "reference values" are a local re-typing of the implemented fit; **new defect**: ψ(1/2) = 0 exactly (twin peaks 0.0625 at y=0.1464/0.8536) → the "≤5 % across [0.05,0.95]" claim is false by 100 % | shape test `xfail(strict=True)` carrying the node + peak positions; `two_photon.py` docstring records the defect |
| 23 | `test_fe17_..._bernitt2012.py:171-176` | delete the literal-echo test (3.7==3.7, 0.4==0.4, …) | the constants are used by the real band gate | — |
| 24 | `tests/test_M1_E2_rates.py:378-387` | delete `test_M1_fs_rate_positive` (×3, `A > 0`) | the pre-fix 2000×-too-small rate also passed; the channel's machinery is already pinned by Drake (B-class) and the Z¹² ratio test | — |
| 25 | `tests/test_transitions.py:84-90` | rename `test_m3_s_to_d_forbidden_by_parity` → `test_m3_s_to_s_forbidden_by_triangle`; assert the allowed M3 case it mis-described | the audit's `file:line` pointed at `test_M1_E2_rates.py`; the test actually lives in `test_transitions.py` | `selection_rules(-1,-1,3,'M') is False` (triangle) + `(-1,-3,3,'M') is True` (the allowed s→d5/2) |
| 26 | `tests/test_polarisation_F2.py:55-60` | `val != 0 and isfinite` → the exact value | F2(2→0,L=2) = 0.6324555320336752 = 2/√10 (machine precision) | `approx(2/√10, rel=1e-12)` + `F2(0→2,L=2) == 0.0`; `isfinite` moved to an inelastic channel |
| 27 | `test_stark.py:20`, `test_polarizability.py:66`, `test_two_photon.py:27` | replace retyped physics literals with shipped constants | `5.14220674763e11`, `27.211`, `2.41798934e8`, `8.319/5.14220674763e9`, `137.035999` | `AU_FIELD_VPM = HARTREE_EV/BOHR_RADIUS_M` (E_h/(e a₀)); `HARTREE_MHZ = AU_RATE_PER_S/(2π)/1e6` (ν=E/h); `ALPHA` from `atomx.constants`; only the literature constants (4.5 a₀³, 8.319 V/cm, Drake's fit coefficients) remain literals |

Test count per touched file (collected): `test_transitions` 28, `test_transitions_engine_uniform` 19, `test_M1_E2_rates` 33 (+2 xfail), `test_gauge_invariance` 9 (+1 xfail), `test_velocity_gauge_full_order` 26, `test_hyperfine` 37, `test_polarizability` 5 (+1 xfail), `test_stark` 10, `test_edm` 11, `test_sternheimer_edm` 10 (+2 xfail), `test_pnc` 5, `test_isotope` 39, `test_polarisation_F2` 8, `test_two_photon` 22 (+1 xfail), `test_two_photon_vs_literature` 14, `test_fe17...bernitt2012` 3 (+3 xfail), `test_zeeman` 7.

---

## 2. The Ne IX M1 rate: which of 3.94e6 / 1.3997e6 is right? — **neither**

### The external evidence

1. **Direct measurement (the decisive number).** Crespo López-Urrutia,
   Beiersdorfer, Widmann & Savin, *Phys. Rev. A* **58**, 238 (1998)
   measured the lifetime of the He-like O⁶⁺ `1s2s ³S₁` level as
   956 (+5/−4) μs, i.e. a magnetic-dipole rate to the `1s² ¹S₀` ground
   state of **A = 1046 (+4/−5) s⁻¹** (EBIT, magnetic trapping mode).
2. **Isoelectronic scaling.** The He-like M1 rate follows Z¹⁰; from the
   measured O⁶⁺ point that gives **9.74e3 s⁻¹ at Z=10**.
3. **Independent clean-room code.** I ran FAC 1.1.5 (`~/code/ASAP/fac`,
   `benchmarks/fac_runner.py`) on the same config set
   (`1s2`, `1s1 2*1`) for Ne: the `2³S₁ → 1¹S₀` M1 row has
   `ΔE = 903.913 eV`, `gf = 8.003418e-10`, rate **9458.4 s⁻¹**
   (FAC's own `gf = A·g·λ²/6.6703e15` identity closes to 0.02 %, so the
   printed column is A, not gA).  FAC's M1 kernel reproduces the H-like
   `2s₁/₂ → 1s₁/₂` rate at Z=10 as **2.4943e4 s⁻¹ = Drake's 2.5e-6 Z¹⁰**
   to 0.03 % — the same transition atomx itself reproduces to 0.2 %.
4. **The cited "Lin et al. 7.0e6 s⁻¹" is not usable.**  It is 700× the
   measured-sequence value at Z=10 (7.0e6 / 9.74e3), and the measured O⁶⁺
   rate scaled by Z¹⁰ leaves no room for it.  Whatever that number was,
   the docstring's use of it as the anchor was wrong; I removed it rather
   than re-pinning against it.

### What atomx produces

| Z | ion | atomx tracer | atomx E | external | ratio |
|---|---|---|---|---|---|
| 6 | C V | 9.169284e4 | 299.710 eV | FAC 37.55 | **2441×** |
| 8 | O VII | 3.717801e5 | 561.851 eV | **measured 1046** (FAC 870.09) | **355×** |
| 10 | Ne IX | 1.399676e6 | 906.004 eV | 9.74e3 (FAC 9458.4) | **148×** |

The Z-dependence is wrong as well: atomx's Ne/O ratio is 3.77 where Z¹⁰
gives 9.31 and FAC gives 10.9.  The one-electron limit is sound (atomx
reproduces Drake's H-like `2s₁/₂→1s₁/₂` M1 to 0.2 %), so the fault is in
the many-body M1 line strength, not in `radial_mL`/`angular_mL`/
`einstein_A_mL`.

### Mechanism (measured, read-only diagnosis by a dedicated sub-investigation)

The M1 amplitude is an (αZ)²-order cancellation residue of two ~2e-3
radial integrals: `∫P_2s Q_1s r dr = −2.102121e-3` and
`∫Q_2s P_1s r dr = +1.937630e-3`, i.e. a residual `R = −1.644909e-4`
(**7.83 %** cancellation).  The exact hydrogenic Z=10 pair cancels to
**0.598 %** (`R = −1.222498e-5`), and that pair through the *identical*
kernel gives 2.5101e4 s⁻¹ = Drake's value.  So the poison is the orbital
pair the MCDHF hands to the kernel: the Guerra-averaged occupations have
collapsed to a `1s²` field (`1s 1.999262, 2s/2p 0.001`), so the "2s" is a
ground-state virtual, and the R7 asymmetric mean-field seam leaves the
(1s, 2s) spinors mutually inconsistent at ~8 % where the physics needs
~0.5 %.  Consequences: `R` is ~15× too large and Z-*independent*
(−2.217e-4 / −1.737e-4 / −1.645e-4 at Z=6/8/10) where the physics requires
`R ∝ Z²`.  Two further defects sit in the same path: `calculator.py:693`
gives the closed-shell-J=0 → one-electron-promoted channel a recoupling
factor `√(2/3)` where an independent m-scheme CG/3j check gives exactly 1
(1.5× in the rate, in the opposite direction), and `angular/csf.py:170-183`
takes the population factor from the *initial* occupation, so the
two-electron change `2s² → 1s2s` gets a spurious 1.704971e6 s⁻¹ "M1" line
at 1004.43 eV.  The two errors close the observed factor exactly:
(1.644909e-4/1.10e-5)² / (1/0.8165)² = 223.5/1.5 = **149 ≈ 148**.

### Verdict and what was done

Both atomx numbers are wrong: 3.94e6 (the pin, output of the
pre-`04cd274` configuration-average CI matrix) is 400× the anchor, and
1.3997e6 (the exact CI matrix on the current orbitals) is 148× the anchor
with the wrong Z-scaling.  Re-pinning to either would be pinning an
unvalidated output, so:

* `test_helike_M1_Z10_vs_measured_anchor` — `xfail(strict=True)` against
  `1046·(10/8)¹⁰ = 9741.6 s⁻¹` (rel 0.15), reason carrying 1.3997e6, the
  measurement, the FAC value and the Z-scaling failure;
* `test_helike_M1_O7_vs_measured_o6plus` — **new** `xfail(strict=True)`
  against the measured 1046 s⁻¹ itself (the only directly measured point
  of the sequence), reason carrying 3.7178e5;
* the class docstring's false "0.56× Lin … all along" and the module
  reference list's Lin citation are replaced by the measurement and the
  FAC numbers, plus the mechanism above.

The three code fixes (transition-orbital consistency, the CFP factor, the
occupation index) are **not** applied here: they change pinned E1/M1/E2
numbers across many test files outside this slice and need a full-suite
revalidation.  They are reported for the orchestrator in §4.

---

## 3. Genuine failures that tightening exposed

1. **He-like M1 (findings 5, and new).** §2.  148×/355×/2441×, wrong
   Z-scaling, plus a third defect (spurious 1004.43 eV line).
2. **`R_e`(Tl) does not exist (finding 6).** The audit's diagnosis was
   wrong: the SCF *does* converge (residual 1.877e-07, 53× inside
   `tol=1e-5`, unchanged at `max_iter=120`, `mode='orbital'`,
   `damping=0.3`, `use_physics_grid=False`).  `converged` is set False by
   the same-kappa orthogonality diagnostic at 1.702e-03 > 1e-03
   (`dirac_fock.py`, the missing off-diagonal Lagrange multipliers,
   review H4), and the public EDM entry point then refuses to run.  So
   the flagship `R_e(Tl)` claim has *no number at all* — now loud.
3. **He α(0) is converged, not basis-truncated (finding 2).** The audit
   inferred unconverged basis from the monotonicity test; the sequence
   actually plateaus at 1.117810 a₀³ (unchanged from κmax=6, n_virt=80 to
   κmax=7, n_virt=100), 19.2 % below the exact 1.383192 — a model deficit
   (frozen-core V^{N−1}, no correlation), not a basis one.
4. **Two-photon spectral shape (new, finding 22).** ψ(y) = 0 exactly at
   y = 1/2 (twin peaks at y = 0.1464/0.8536); the exact Spitzer-Greenstein
   2s→1s distribution is single-peaked at y = 1/2.  The docstring's
   "≤ 5 % across [0.05, 0.95]" is false by 100 % at the centre.
5. **Documentation-vs-code splits, now visible.** Cs `R_e`: code 224.4456
   vs docstring "≈ 80" (the 2J+1 denominator) vs Liu-Kelly 120 — the code
   and the twice-stated formula are self-consistent, the worked example
   was wrong (prose fixed).  Sternheimer H: `ratio = 42.85` vs the module
   docstring's "factor-of-2-3" (prose fixed, test now names the number).
6. **He-like Z=10 gauge gate (finding 14).** Measured 0.0369 against the
   old 0.20 threshold: keeping the Gu 2008 0.10 the docstring already
   quoted was possible and closes a 2× silent-regression window.

Findings whose tightened gate *passed* at the measured level, i.e. where
the code was right and only the gate/docstring was wrong: 1, 3, 7, 8, 9,
11, 12, 13, 14, 15, 17, 19, 20, 21, 23, 26, 27.

---

## 4. Not fixed, and why

1. **`src/atomx/methods/dirac_fock.py` same-kappa orthogonality (finding
   6).** Adding the off-diagonal Lagrange multipliers is a DHF-solver
   change requiring its own independent validation (and a full-suite
   re-pin).  The tests now carry the measured overlap and refuse to hide
   the missing `R_e(Tl)`.
2. **The three M1 many-body defects (finding 5).** Same reason: they move
   pinned transition numbers in files outside this slice.  Diagnosis,
   numbers and proposed minimal fixes are in §2 and in the
   `TestHeliumLikeM1` docstring.
3. **He α(0) model deficit (finding 2).** Core relaxation / correlation is
   a model change, not a test change; gated as a strict xfail.
4. **`lande_gJ_hydrogenic` |κ| ≥ 2 (finding 20).** The audit's preferred
   fixes (raise/warn, rename `LandeFactor.dirac` → `dirac_s1`, or add an
   `nr_fallback` flag) touch a public dataclass consumed by test files
   outside this slice.  The test now records the omission at Z=92 with the
   17.3 % size of the dropped correction, and fails loudly if the fallback
   is ever replaced — the rename/warn itself is left to the orchestrator.
5. **Two-photon ψ(y) shape (finding 22).** Fixing the shape means
   implementing a correct Spitzer-Greenstein ψ(y) (and deciding what to do
   about the existing users of `two_photon_spectrum`); the src docstring
   now records the defect and the test is a strict xfail.
6. **`src/atomx/properties/stark.py` CODATA-2018 literals.**
   `_AU_FIELD_VPM = 5.14220674763e11` and `2.41798934e8` MHz/eV are 4.2e-8
   from the shipped CODATA-2022 identities used by the new tests.  I did
   not change published numbers for a 4e-8 unit-conversion staleness;
   the gates are at `rel=1e-6` (10⁷× tighter than the old band) and the
   residual is named in the test docstring.
7. **`test_two_photon.py` Drake-fit tests.** Kept (they pin the shipped
   fit coefficients) but reclassified `@pytest.mark.snapshot`, because
   they compare the implementation against a local re-typing of itself.
   They are *not* counted as coverage of the two-photon rate claim.
8. **`test_two_photon_vs_literature.py::test_sos_he_like_matches_drake`**
   (10 % gate, measured 7.07e-2 at Z=2) — the audit did not flag it and
   the measured value already sits at the stated accuracy claim; left
   unchanged rather than loosened or arbitrarily re-cut.
9. **`test_fe17_3c_3d_vs_bernitt2012.py`** — only the tautological
   band-definition test was deleted (finding 23); the two orbital-cause
   strict xfails were pre-existing and are untouched.

---

## 5. Code changes (all docstring-only, no behaviour change)

| file | change | verification |
|---|---|---|
| `src/atomx/properties/edm/scaling.py:63` | worked example "R_e ≈ 80" → "R_e = 224.45" with the arithmetic (224.45 × 0.75/2 = 84.2 explains the old number) and the 1.87× Liu-Kelly ratio | recomputed from the twice-stated formula with the shipped α: 224.4456 = 10α²·55³·1.9/(0.5·1.5·1); matches `tests/test_edm.py::test_R_e_cs_like_frozen_value` |
| `src/atomx/properties/edm/sternheimer.py:43` | "factor-of-2 accuracy" confined to the heavy atoms the scaling law was calibrated on; the H 1s ratio 42.85 (0.0304260 vs 7.10018e-4) recorded | matches `test_sandars_comparison_present` (both numbers asserted) |
| `src/atomx/properties/transitions/two_photon.py:168` | the false "accurate to ≤ 5 % across [0.05, 0.95]" claim replaced by the measured central node (ψ(1/2) = 0, twin peaks at 0.1464/0.8536) and a pointer to the strict xfail | ψ(1/2) = 0.0 exactly and argmax = 0.1464 measured in `test_two_photon_spectrum_shape_is_not_the_literature_psi` |

No executable line in `src/` was modified.
