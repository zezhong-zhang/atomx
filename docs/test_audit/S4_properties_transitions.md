# S4 — atomic properties and transition rates

Slice: wavelengths, A-coefficients, gf values, hyperfine constants, isotope
shifts, g-factors, polarisabilities, EDM/PNC, UTA moments.
21 files, 371 collected tests (all files in the dispatch list existed; none
skipped).  Reference material: `tests/test_property_regressions.py`
(the H2/H3 gates) is not in the slice but is cited where it supplies a gate
that a slice file's docstring claims.

## 1. Verdict

The slice contains **two genuinely excellent physics gates** — the
length/velocity gauge-invariance tests (`test_velocity_gauge_full_order.py`,
`test_gauge_invariance.py`), which assert an exact theorem and **do not share a
radial kernel between the two gauges** (`radial_eL` integrates
`(P_fP_i+Q_fQ_i)r^L`; `radial_eL_velocity` integrates the Grant §8.4
derivative-free reduction `(c/ω)[(κ_i−κ_f+L)A + (κ_i−κ_f−L)B]r^{L−1}` — distinct
integrands over a shared quadrature), and the UTA moment file
(`test_uta_fac_moments_gates.py`), which hard-codes a clean-room FAC reference
*and* the `ΣS = Σrme²/2` sum rule.  Everything else is dominated by
**magnitude bands that cannot fail** — polarisability `0.5 < α < 1.6` around an
exact value of 1.383, Stark `−10 < ΔE < 0` around an exact −5.599 MHz, EDM
`1e-6 < R_e < 1` while the file's own computed Sandars ratio is 42.9 against a
stated expectation of 2–3 — and by **circular literals**: the H 1s hyperfine
preset was *fitted* to 1420.403 MHz
(`src/atomx/data/hyperfine_presets.py:76` provenance note) and is then asserted
equal to 1420.403 MHz and "within 10 ppm of experiment".  Two gates are red
today in the slow tier (He-like M1 tracer, Tl eEDM), and the named gate for the
NIST Ly-α A-coefficient (`docs/VERIFICATION.md:29` → `tests/test_transitions.py`)
**does not exist in the file it names**: that file's strongest A-rate assertion
is `A > 0`, which the pre-fix buggy value (1.253677e9 s⁻¹) satisfied.  Class mix
over the slice: **A171 B30 C17 D24 E76 F53** (sums exactly to the 371 collected
tests; the per-file rows below add up to the same numbers).

## 2. File table

| file | tests | class mix | worst | note |
|---|---:|---|---|---|
| `test_transitions.py` | 25 | A19 D1 E5 | **S1** | module docstring promises an H 2p→1s A-coefficient *and* a gauge check; neither is in the file. Ne A-rate gated by `> 0`. |
| `test_transitions_engine_uniform.py` | 18 | C16 F2 | S3 | C-degenerate for the machinery claim (both engines share `radial_eL`); a real SCF-orbital gate. |
| `test_M1_E2_rates.py` | 35 | A14 B8 D2 E11 | **S1** | Drake/Garstang oracles are real B-class; the He-like M1 tracer pin is **red** (1.40e6 vs 3.94e6); `gauge_ratio > 0.90` is one-sided. |
| `test_gauge_invariance.py` | 10 | A5 D1 F4 | S2 | theorem at 2e-11 gated at 1e-3; He-like Z=10 gate (0.20) contradicts its own docstring (0.10); Z=2 strict-xfail at 0.389. |
| `test_velocity_gauge_full_order.py` | 26 | A26 | S2 | the strongest file in the slice: 26 theorem assertions, L=1…4, measured 3e-11…2.8e-4 vs a 1e-2 gate. |
| `test_two_photon.py` | 22 | A1 E17 F4 | S3 | "Drake 1986 / FAC reference values" are a local re-typing of the implementation's own fit; the spectrum test re-checks a normalisation the source defines. |
| `test_two_photon_vs_literature.py` | 14 | A2 B8 E2 F2 | S2 | SOS-vs-Drake is a genuine independent check but is gated 1e-2 against a measured 3.8e-5; the He "order of magnitude" test is redundant and 20× wide. |
| `test_hyperfine.py` | 34 | A12 B6 D7 E7 F2 | **S1** | preset ladder is fitted-then-asserted; the documented dominant term (−1632 ppm reduced mass) is invisible to the 0.5 % gate; B has no magnitude test. |
| `test_isotope.py` | 38 | A25 B1 D3 E3 F6 | S2 | virial + exact reduced-mass NMS are A-class; Seltzer F(1s)=2/3 is gated at 1.3 %; Ne K_SMS only within a factor 6. |
| `test_zeeman.py` | 27 | A16 B2 D2 E3 F4 | S2 | the SI↔a.u. Bohr-magneton identity done right; `lande_gJ_hydrogenic` silently returns the NR value for \|κ\|≥2 and the field is named `dirac`. |
| `test_stark.py` | 10 | A4 E4 F2 | **S1** | textbook linear Stark is A-class; the quadratic shift is gated by a band admitting 10 orders of magnitude. |
| `test_polarizability.py` | 4 | E4 | **S1** | He α(0) measured 0.8928 a₀³ vs the exact 1.383 named in the docstring; band 0.5–1.6. BBR public path never exercised. |
| `test_polarisation_F2.py` | 8 | A6 E1 F1 | S4 | three exact closed forms + selection rules; the E2 case (exact 2/√10) asserts only finiteness. |
| `test_g_factor.py` | 7 | A2 B1 C1 E1 F2 | S2 | Breit formula A-class and SCF-vs-analytic at 1e-8; the top-level value is a 1 % band. |
| `test_edm.py` | 11 | E7 F4 | **S1** | every "scaling" test re-derives the implementation's own one-line formula; the shipped Cs docstring value (≈80) is 2.8× its own code (224.4) and the `30 < R_e < 500` gate cannot see it. |
| `test_sternheimer_edm.py` | 11 | A2 D3 E4 F2 | **S1** | **2 red slow tests** (Tl SCF not converged); the Sandars ratio (42.85 for H) is computed and never asserted; H gated 1e-6 < R_e < 1. |
| `test_pnc.py` | 4 | E2 F2 | **S1** | the whole PNC property class: the amplitude raises `NotImplementedError`, Q_W gated by `< -70`, the radial ME by `isfinite`. |
| `test_fe17_3c_3d_vs_bernitt2012.py` | 4 | D3 E1 | S1 | the Bernitt 2σ band gate is strict-xfail (measured 2.517 vs [3.30, 4.10]); one test asserts the band literals equal themselves. |
| `test_spectrum_and_uta.py` | 29 | A13 E4 F12 | S2 | `uta_moments` arithmetic is A-class and exact; everything else is container/C RM plumbing. |
| `test_uta_fac_moments_gates.py` | 7 | A3 B2 D1 F1 | S3 | model file: hard-coded clean-room FAC reference + the `Σ rme²/2` sum rule + the documented Unsöld deviation. |
| `test_review_m9_m13_gates.py` | 27 | A21 B2 D1 F3 | S4 | sympy 3j oracles, CODATA identities, the (1+m_e/m_p)⁻³ = −1632 ppm derivation; 3 docstring-text gates are F. |

## 3. Findings

### S1 — an A/B claim that nothing gates, or a test that passes for a wrong result

**1. The named gate for the NIST Ly-α A-coefficient does not exist in the named
file.**  `docs/VERIFICATION.md:29` ("Hydrogen 2p→1s A coefficient … gate:
`tests/test_transitions.py`") and that file's own docstring
(`tests/test_transitions.py:6`, "E1 A-coefficient for H 2p->1s (exact value:
6.2649e8 s^-1)") promise the gate.  The file's strongest A-rate assertion is
`tests/test_transitions.py:239` `assert tr.A_rate_per_s > 0` in
`test_neon_transitions`.  Measured: the pre-fix wrong value
`A(1s→2p3/2) = 1.253677e9` and the correct `6.268384e8` are **both positive**,
so this file passed with the H2 degeneracy bug installed.  The real gate lives
in `tests/test_property_regressions.py:145` (`rel=0.005` vs 6.2649e8).
*Remediation:* move/duplicate the NIST assertion into `test_transitions.py`
(it costs one `Atom(Z=1)`-free hydrogenic call) and delete the false docstring
line; or correct `VERIFICATION.md:29` to name `test_property_regressions.py`.

**2. Polarisability: the exact value is named in the docstring and the gate
cannot see a 35 % error.**  `tests/test_polarizability.py:16-27`
(`test_helium_polarizability_reasonable_magnitude`) asserts
`0.5 < r.alpha_static < 1.6` (factor 3.2) while its docstring says "Exact
Helium α(0) ≈ 1.383 a₀³".  Measured at the test's own basis
(`Atom(Z=2, config="1s2", engine="bspline", corrections="none")`,
`kappa_max=2, n_virt_max=15`): **α(0) = 0.892802 a₀³ — 35.4 % below exact**;
even at `kappa_max=3, n_virt_max=30` it is 1.0939 (−20.9 %), and the
monotonicity test (`:35-39`) confirms the series is still climbing, i.e. the
window is being passed by a manifestly unconverged number.
*Remediation:* gate the H-like case at the exact 4.5 a₀³ (already done at
`test_review_m9_m13_gates.py:159`), and for He replace the window with
(a) an explicit basis-convergence assertion (`α(kappa_max=3, 30) ≥
α(kappa_max=2, 15)` exists; add `α → 1.383` within a stated basis budget), or
(b) a documented HF-level target ≈1.32 a₀³ with a 2 % band.

**3. Quadratic Stark is gated by a band that admits any small number.**
`tests/test_stark.py:49-52` `assert -10.0 < dE < 0`.  Measured
`quadratic_stark_shift(alpha_a0_cubed=4.5, F_Vpm=1e7) = -5.5987 MHz`; the exact
first-principles value is `−½αF²` = `−0.5 · 4.5 · (1e7/5.14220674763e11)² Ha`
= −5.5987 MHz (α(H 1s) = 4.5 a₀³ is exact).  The gate accepts −1e-9.
*Remediation:* `assert dE == pytest.approx(-0.5*4.5*(1e7/5.14220674763e11)**2
* HARTREE_MHZ, rel=1e-10)` — the constant is already typed in
`test_linear_stark_h_n2_textbook:20`.

**4. The H 1s hyperfine "comparison with Karshenboim theory" is circular.**
`src/atomx/data/hyperfine_presets.py:31-36` states the `h1s` coefficients "are
*empirically tuned* to reproduce the Karshenboim 2005 theoretical total
ν_theory = 1420.403 MHz … against atomx's 80-basis B-spline DHF baseline
A_DHF = 1421.273 MHz"; `:76` is the tuned triple.  `tests/test_hyperfine.py:279`
then asserts `A_final == approx(1420.403, abs=0.002)` ("<1 ppm from Karshenboim
theory") and `:375-381` asserts "<10 ppm from experiment" using the same preset.
Both are D-class fitted constants wearing a B-class citation; neither can fail
for a physics reason.  Compounding it: the raw DHF baseline is
`A(¹H 1s) = 1421.272852 MHz` (`test_review_m9_m13_gates.py:102`), i.e.
**+610 ppm** from the measured 1420.4058, and the dominant documented omission
is the reduced-mass factor `(1+m_e/m_p)⁻³ = −1632 ppm` — the only *rung* test
that checks it is `test_hyperfine.py:364-373` (`A_fnm ≈ 1418.95 ± 0.5`, i.e.
±350 ppm).
*Remediation:* mark both preset assertions `@pytest.mark.snapshot`, and gate
the untuned physics instead: `(M/(M+m_e))³ · A_DHF` against `1418.9532` to
`abs=2e-3` (that exact identity already exists at
`test_review_m9_m13_gates.py:126-127` and should be the only H 1s statement
attributed to Karshenboim).

**5. The He-like M1 tracer pin is red and its docstring claim is false.**
`tests/test_M1_E2_rates.py:493` (slow) pins `A = 3.94e6 ± 5 %`; measured today
through the same code path: **`A = 1.3997e6` (0.200 × Lin et al. 7.0e6)**.
The class docstring `:465-474` still asserts "the tracer sits at 3.94e6 s⁻¹ =
0.56 × Lin … and it has been at that value all along: 3.970e6 before the
2s²/2p² configurations were added, 3.940e6 after".  A factor-2.8 change in a
published observable (Ne IX 2³S₁→1¹S₀ M1) went unnoticed by the default suite
because the only gate is slow-marked.
*Remediation:* re-pin to the measured value with the current date and the
CI-kernel commit named, or convert to a strict `xfail` carrying the measured
1.3997e6 (the pattern already used at `:498-514`); correct the "all along"
sentence either way.

**6. Both Tl eEDM gates fail: the flagship `R_e(Tl)` claim is ungated.**
`tests/test_sternheimer_edm.py:122` and `:230` both fail with
`res.converged == False` (11 iterations, residual 1.877e-07) for
`DiracFock.solve(Z=81, subshells=_thallium_subshells())`.  The file docstring
`:112-118` calls Tl "the central deliverable of HYP-685" and cites "literature
~585"; the gate is `50 < R_e < 5000`.  Because both tests are `slow`, the
default run is green while the only numerical eEDM gate in the repository is
red.
*Remediation:* relax the SCF tolerance or seed for this panel (the residual is
19× the tolerance, not a divergence), then re-pin the measured `R_e(Tl)` and
tighten to a literature band; until then mark the tests `xfail(strict=True)`
with the non-convergence recorded, so the claim is visibly ungated rather than
silently broken.

**7. The Sternheimer↔Sandars comparison is computed and never asserted.**
`src/atomx/properties/edm/sternheimer.py:404-407` returns
`scaling_comparison = {"sandars": R_sandars, "ratio": R_e_total/R_sandars}`.
Measured for H 1s (`kappa_max=4, n_virt_max=20`): **`R_e = 0.0304260`,
`sandars = 7.10018e-4`, `ratio = 42.85`** — an order of magnitude outside the
"factor-of-2-3" agreement the module docstring `:17-23` claims.  The only test
is `tests/test_sternheimer_edm.py:176-181`, which asserts `sandars > 0` and
`isinstance(ratio, float)`; `:71-81` gates H by `1e-6 < R_e < 1.0` (10⁶ band).
*Remediation:* assert `ratio` against a stated band and print the measured
42.85 in the failure message; if 42.85 is believed, the docstring's
"factor-of-2-3" must be corrected and the H bound tightened to the measured
value.

**8. The gauge-quality gate is one-sided.**  `tests/test_M1_E2_rates.py:588`
`assert t.gauge_ratio > 0.90`; `gauge_ratio` is `A_velocity / A_length`
(`src/atomx/properties/transitions/calculator.py:103`), so a velocity rate
1000× the length rate passes.  The same pattern is absent elsewhere, but this
is the only API-level gauge gate for the E2 channel.
*Remediation:* `assert abs(t.gauge_ratio - 1.0) < 0.10`.

**9. The documented Cs `R_e` value is wrong by 2.8× and the gate cannot see it.**
`src/atomx/properties/edm/scaling.py:66-70` states: "For a Cs-like alkali
(Z=55, n_eff ≈ 1.9, J=1/2, L=0) this gives **R_e ≈ 80** — comparable to the
modern MBPT value of 120 within the documented factor-of-2 accuracy of this
formula."  Measured: `electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5,
L=0) = **224.446**` (`scaling.py:95` divides by `J(J+1)(2L+1) = 0.75`, not by
the `2J+1 = 2` the "≈80" figure implies: 224.4 × 0.75/2 = 84.2).  So the
shipped docstring's worked example is 2.8× off its own code, and the ratio to
the named Liu-Kelly 120 is 1.87 — inside "factor of 2", but not the number
printed.  `tests/test_edm.py:16-17` gates `30 < R_e < 500` (16×), so neither
the docstring error nor a 2.8× code regression is detectable.  The same file's
`:20-24` (Z³ ratio at `rel=1e-12`), `:27-31` (1/3) and `:45-51` (Z⁴) are
tautological re-derivations of `scaling.py:95`; `:54-58` gates
`1e-18 < K < 1e-16` (100×) where the measured `K_S(Hg, A=199, closed) =
4.3624e-17` sits 9 % from the "Dzuba–Flambaum ~4e-17" named in
`scaling.py:118-121`.
*Remediation:* fix the docstring to 224 (or fix the formula if 80 was the
intended physics), gate `R_e(Cs) == approx(120, rel=1)` against Liu-Kelly and
`K_S(Hg) == approx(4e-17, rel=0.2)`; label the Z³/Z⁴/1-3 ratio tests as formula
pins.

**10. PNC is a property class with no physics gate at all.**  The public
amplitude entry point raises: `tests/test_pnc.py:24` and `:34` use
`pytest.raises(NotImplementedError, match="opposite-parity")`, so
`pnc_amplitude_dirac_only` and `methods.pnc.compute_pnc` never return a number;
`PNCResult` is never constructed.  `weak_charge_qw` is gated by
`assert weak_charge_qw(55, 78) < -70.0` (`:17`), which accepts −1000; the
standard-model value from the module's own constants
(`src/atomx/properties/pnc/__init__.py:44-45`, Q_W(p)=0.0713, Q_W(n)=−0.9890)
is exactly `55·0.0713 + 78·(−0.9890) = −73.22`.  `weak_hamiltonian_radial_me`
is gated by `np.isfinite(me)` (`:60`).
*Remediation:* gate `weak_charge_qw(55, 78) == approx(-73.22, rel=1e-12)` (it
is a pure closed form of the two shipped constants) and gate the radial ME's
`r→0`/`r→∞` behaviour or its `ρ_nuc → δ(r)` limit instead of finiteness; state
in `__all__`/docs that the amplitude is a scaffold.

**11. Hyperfine B has no magnitude assertion.**  `tests/test_hyperfine.py`
gates B by `B(j=1/2) == 0.0` (`:311-319`), `B(Q=0) == 0.0` (`:321-330`),
`B(2Q) == 2B(Q)` (`:332-342`) and `B_MHz["2p3/2"] != 0` (`:358`) — all exact
identities or existence checks, none of which constrains the `(2j−1)/(2j+2)`
prefactor in `src/atomx/properties/hyperfine/quadrupole.py:6` or the barn→Bohr²
conversion `BARN_BOHR2` (`:47`, never referenced by any test).  A wrong
prefactor or a factor-1000 unit error passes.
*Remediation:* the point-nucleus hydrogenic ratio is analytic — for a Dirac
`p3/2` orbital `B/A = [Q/μ_I] · f(κ) ⟨r⁻³⟩` with `⟨r⁻³⟩` in closed form — so
assert `B/A` for H-like `2p3/2` against the closed form, or against a published
DHF B for ¹³³Cs 6p ²P₃/₂.

**12. The polarisability BBR path (`include_bbr=True`,
`black_body_shift_K`) has zero coverage, and its only test is
self-referential.**  No test in the repository passes `include_bbr` or reads
`black_body_shift_K` (grep over `tests/`: no hits).  The one BBR test,
`tests/test_polarizability.py:58-68`, calls the private
`_bbr_shift_from_alpha` with constant α and re-derives
`−0.5·E²·α0` with its own copy of `8.319/5.14220674763e9`; the source comment
at `src/atomx/properties/polarizability/__init__.py:304-305` says this limit is
"guaranteed" by construction.
*Remediation:* exercise the public path (`dipole_polarizability(res,
include_bbr=True)` on H 1s) and assert `black_body_shift_K` equals
`−½⟨E²⟩_{300K}·α(0)` with `α(0) = 4.5 a₀³` (the exact H value already gated at
`test_review_m9_m13_gates.py:159`), i.e. check the *physical* quantity, not the
helper's own branch.

### S2 — tolerance that cannot distinguish the physics at issue

**13. The gauge theorem is gated 7–8 orders of magnitude looser than it holds.**
Measured (reproduced for this audit):
`test_gauge_invariance.py` params (`ns=120, ks=9`) give
`|A_v/A_l − 1| = 2.08e-11 (Z=1), 3.84e-12 (Z=2), 1.09e-12 (Z=3), 3.37e-12
(Z=5), 3.55e-11 (Z=8)` against gates `1e-3`/`5e-3`
(`tests/test_gauge_invariance.py:71,76,88`).
`test_velocity_gauge_full_order.py` params (`ns=80, ks=9`) give
`3.4e-11 (s→p3/2), 4.96e-10 (p1/2→d3/2), 6.70e-11 (s→d5/2), 5.43e-07 (s→f7/2),
2.85e-04 (s→g9/2)` against a flat gate of `1e-2` (`:48-110`).  The source
docstring (`src/atomx/properties/transitions/multipole.py:88-90`) says the two
forms "produce the *same* Einstein A-coefficient", and the review
(`docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md:811-813`) quotes −4.7e-11 /
+1.1e-10 / −7.0e-7 as the demonstrated level — none of those numbers appear in
the tests.
*Remediation:* tighten to `1e-8` for L≤2 and `1e-3` at L=4, with the measured
values in comments, exactly as `test_transitions_engine_uniform.py` does for
its own tolerances.

**14. The He-like gauge threshold contradicts its own docstring.**
`tests/test_gauge_invariance.py:153` states the Gu 2008 §3.2 threshold for
ground-state E1 is `|A_v/A_l − 1| ≤ 0.10`; `:217` asserts `< 0.20` for He-like
Ne with no justification, while `:162-174` keeps Z=2 at the literature 0.10 as a
strict `xfail` (measured 0.389).
*Remediation:* either state the 0.20 as a deliberate, dated loosening with the
measured value, or keep 0.10 and mark Z=10 `xfail(strict=True)` with its
measured number too — the current mix lets a 2× regression through silently.

**15. Two-photon SOS-vs-literature is gated 500× looser than measured.**  The
SOS path is a genuine independent method (B-spline Dirac sum-over-states,
`two_photon.py:263-357`) and the Drake fit is external.  Measured at the test's
parameters (`ns=60, ks=9, n_omega=100`): `rel = 3.77e-05 (Z=1), 1.23e-04 (Z=2),
2.63e-04 (Z=3), 7.06e-04 (Z=5)`; `tests/test_two_photon_vs_literature.py:43`
gates `rel < 0.01`, and the review quotes 2.1e-5 for H.
*Remediation:* gate `rel < 1e-3` (still 20× the Z=1 measurement but 10× the
worst measured point), and record the measured 8.228832 s⁻¹ for H against
Drake's 8.229142 as a comment.

**16. The He-like "order of magnitude" test is both redundant and vacuous.**
`tests/test_two_photon_vs_literature.py:93-96` asserts `10.0 < val < 200.0`;
measured `two_photon_rate_sos_he_like(2, ns=60, ks=9, n_omega=100) =
47.5919 s⁻¹` (Drake 51.2113).  The same file already gates Z=2 at 10 %
(`:87`), and the Drake fit names the value to 1 %.
*Remediation:* delete it.

**17. Seltzer `F(1s) = 2/3 Ha/Bohr²` — exact — is gated at 1.3 %.**
`tests/test_isotope.py:331-342` derives `F = (2π/3)|ψ(0)|² = 2/3` in its own
docstring and then asserts `F_MHz_per_fm2 == approx(1.57, abs=0.02)`, i.e.
[1.55, 1.59].  Measured: `F_Ha_per_Bohr2["1s1/2"] = 0.666994592`
(rel 4.9e-4 from 2/3), `F_MHz_per_fm2 = 1.5672024`.
*Remediation:* assert `fs.F_Ha_per_Bohr2["1s1/2"] == approx(2/3, rel=1e-3)`
(directly in the units of the derivation) and pin the MHz/fm² conversion
separately with an explicit unit round-trip.

**18. Cs EDM enhancement gated by a 16× window around a *mis-stated* literature
value.**  `tests/test_edm.py:16-17`: `electron_edm_enhancement_sandars(55,
n_eff=1.9) = 224.446` (measured), against the docstring's own "R_e ≈ 80" and the
Liu-Kelly MBPT 120; the gate is `30 < R_e < 500`.  See finding 9 — this is the
same defect seen from the test side.
*Remediation:* `assert R_e == pytest.approx(120, rel=1.0)` (Liu-Kelly, at the
module's stated accuracy), which is what makes the formula falsifiable and
would have caught the 80-vs-224 docstring error.

**19. Ne ab-initio `K_SMS` gated only within a factor 6.**
`tests/test_isotope.py:246-248` `50.0 < abs(sms.K_SMS_au) < 300.0` with no
external reference and no converged value named; the neighbouring gauge test
(`:264-292`) is the only independent handle and is itself gated at `rel=0.1`
while the hydrogenic analogue in this slice achieves 1e-11.
*Remediation:* report and pin the basis-converged `K_SMS(Ne)` (the test already
solves at `ns=50, ks=9`; a two-basis convergence assertion is enough), and
tighten the momentum-gauge consistency check to the measured residual.

**20. `lande_gJ_hydrogenic` returns the non-relativistic value for |κ| ≥ 2 while
being named "Dirac", and no test exposes it.**  `src/atomx/properties/zeeman/
lande.py:166,171` fall back to `lande_gJ_nr`; `LandeFactor.dirac` is then that
number.  `tests/test_zeeman.py:93-97` (`test_p3_2_falls_back_to_nr`) pins the
fallback at Z=1, where the relativistic correction is ~1e-5, so it cannot see
the problem.  For the one κ where Dirac *is* implemented, the correction at
Z=92 is `(2/3)(1+2√(1−(Zα)²)) / 2 = 0.82742` (measured `g = 1.654846`), i.e.
**−17.26 %** — the same order
of effect is simply dropped for p₃/₂/d.  `test_heavy_ion_reduction` (`:79-85`)
only exercises κ=−1.
*Remediation:* raise or warn (`NotImplementedError`/`RuntimeWarning`) for
|κ|≥2, or rename the field to `dirac_s1` and add a `nr_fallback` flag; add a
Z=92 κ=−2 assertion that records the omission.

### S3 — D-class masquerading as validation

**21. `test_transitions_engine_uniform.py` is C-degenerate for the machinery
claim.**  `src/atomx/properties/transitions/multipole.py:35` states "One
implementation of each radial kernel serves both engines", and the test file
repeats it (`:6-10`).  The 2e-3 rate/line-strength agreement therefore measures
the two *SCF* engines (genuinely independent: shooting vs B-spline) and is blind
to any error inside `radial_eL`/`radial_mL` — the exact class of error the M1/E2
prefactor bug (`621fe99`) was.
*Remediation:* keep the file (the orbital-quality gate is real) but relabel its
docstring claims from "the transition pipeline … gates the machinery" to "gates
the orbital set"; add one independent kernel oracle — e.g. the closed-form
hydrogenic `⟨r⟩`, `⟨r²⟩`, `⟨r⁻³⟩` or the exact Rutherford `R¹` — that does not
route through `eval_orbital_pair`.

**22. `test_two_photon.py`'s "reference values" are a local copy of the
implementation.**  `tests/test_two_photon.py:25-39` re-types the Drake 1986 /
FAC fit that `src/atomx/properties/transitions/two_photon.py:66-136` implements;
the only difference is the α literal (CODATA-2018 `137.035999` vs the shipped
`ALPHA`, a ≈7e-10 relative effect on the rate), far below the `rel=1e-6` gate at
`:48`.  Sixteen of the file's 22 tests are this comparison or a self-ratio
(`:64-69`), and `test_two_photon_spectrum_normalisation` (`:82-90`) re-checks a
normalisation whose constant `2/(1/30)` is written into the source
(`two_photon.py:192`) from the same analytic integral.
*Remediation:* keep the fit tests but mark them `@pytest.mark.snapshot`; the
real oracle is `test_two_photon_vs_literature.py::test_sos_h_like_*`, which
should be the file that carries the 8.229 s⁻¹ claim.  The *shape* claim in
`two_photon.py:168-174` ("Spitzer-Greenstein … accurate to ≤5 % across
[0.05,0.95]") has no test at all and should get one against the literature ψ(y).

**23. Literal-echo tests.**  `tests/test_fe17_3c_3d_vs_bernitt2012.py:171-176`
asserts `3.7 == 3.7`, `0.4 == 0.4`, `3.3 == 3.3`, `4.1 == 4.1` (the band
constants against themselves); `tests/test_edm.py:73-77` asserts seven dict keys
exist.
*Remediation:* delete the band-definition test (the constants are in the
module header); the key-existence test can be folded into the scaling test that
already reads the values.

### S4 — E-class noise and misleading names

**24. Tests that assert nothing falsifiable about physics.**
`tests/test_M1_E2_rates.py:378-387` (`test_M1_fs_rate_positive`, ×3)
`assert A > 0` — the pre-fix M1 rate was ~2000× too small and still positive
(the commit `621fe99` message says so).  `:330-333`
(`test_ML_uses_same_prefactor_as_EL`) asserts `einstein_A_mL(...) ==
einstein_A_eL(...)`, which the source implements as a delegation.  `:313-328`
(`test_E2_prefactor_analytic`) re-derives the same expression that
`einstein_A_eL` evaluates.  `tests/test_M1_E2_rates.py:254-264`'s allowed
branch asserts nothing by its own admission (`:262-263`).  H-like
2p₃/₂→2p₁/₂ has an exact analytic M1 rate
(`A = (α⁶/…)(ΔE_FS)³·…`), so `test_M1_fs_rate_positive` is replaceable by a
closed-form assertion, and `test_M1_fs_Z12_scaling` already shows the
machinery works.
*Remediation:* replace with `A(Z=10) == approx(analytic, rel=1e-6)` using the
exact Dirac `2p` splitting, or delete.

**25. A test whose name asserts a parity fact its body explicitly disowns.**
`tests/test_M1_E2_rates.py:84-90` is named
`test_m3_s_to_d_forbidden_by_parity`; the body's own comment (`:87-89`) says
"Actually this is allowed by both — replace with a true M-forbidden", then
asserts the unrelated `selection_rules(-1, -1, 3, 'M') is False`.
*Remediation:* rename to `test_m3_s_to_s_forbidden_by_triangle` (the physics it
actually checks) or assert `angular_mL(-1, 2, 3)`/the correct s→d M3 case.

**26. `test_F2_e2_finite` checks finiteness where an exact value exists.**
`tests/test_polarisation_F2.py:55-60` asserts `val != 0.0 and
math.isfinite(val)`; measured `steffen_alder_F2(2.0, 0.0, L=2) =
0.6324555320 = 2/√10`.  The E1 cases in the same file are correctly gated to
`rel=1e-10`.
*Remediation:* `assert val == pytest.approx(2/math.sqrt(10), rel=1e-12)`, plus
one or two more E2 (J_u,J_l) points from the same Steffen-Alder table.

**27. Numeric literals of the physics are retyped inside tests.**
`tests/test_stark.py:20` retypes `5.14220674763e11`, `27.211`, `2.41798934e8`;
`tests/test_polarizability.py:66` retypes `8.319`, `5.14220674763e9` (the
source re-derives the same two at `polarizability/__init__.py:316`);
`tests/test_two_photon.py:27` retypes `137.035999`.  This is exactly the M12
class already fixed for `src/` by `test_review_m9_m13_gates.py:216-240`, but the
tests themselves were not converted.
*Remediation:* import `HARTREE_EV`, `HARTREE_MHZ`, `ALPHA`, `C` from
`atomx.constants` and keep the *literature* number as the only literal (e.g.
`μ_B/h = 13996.2449 MHz/T`), so a CODATA bump cannot silently desynchronise a test.

## 4. Model tests (imitate these)

1. `tests/test_velocity_gauge_full_order.py` (whole file, 26 assertions) — the
   right way to test a theorem: two algebraically independent reductions of the
   same observable, one flat tolerance, every (κ_i, κ_f) channel and every L.
   Its only weakness is the tolerance (finding 13).
2. `tests/test_zeeman.py:160-162` —
   `assert pytest.approx(13996.24, abs=0.1) == BOHR_MAGNETON_MHZ_PER_TESLA`,
   the SI↔atomic-unit identity with the physical constant written out.
   `tests/test_review_m9_m13_gates.py:237-240` upgrades it by *also* proving the
   stale CODATA-2018 value is detectably different.
3. `tests/test_property_regressions.py:176-198` — the `A ↔ gf` identity with the
   pre-fix factor `(2j_u+1)/(2j_l+1)=2` named in the message, checked on the
   hydrogenic rows *and* on the public `Atom(Z=10)` rows.
4. `tests/test_uta_fac_moments_gates.py:84-105` — a hard-coded clean-room FAC
   reference plus the `ΣS = Σ rme²/2` sum rule, with the fixture's provenance
   named and the weights-only proxy's deviation pinned rather than "fixed".
5. `tests/test_review_m9_m13_gates.py:109-127` — the exact reduced-mass
   identity `(1+m_e/m_p)⁻³ = −1632.1 ppm` computed from the shipped constant and
   then applied to `A_DHF`: a first-principles derivation, not a quoted number.
6. `tests/test_review_m9_m13_gates.py:261-298` — `slater_condon_f` against
   sympy's exact 3j symbols **and** exact rationals, i.e. an oracle that shares
   no code with the implementation.
7. `tests/test_isotope.py:41-59` — virial theorem (`T = −E`) and the point-nucleus
   `E_NMS = T/M` for H 1s, both to `rel=1e-3` against closed forms.
8. `tests/test_gauge_invariance.py:64-71` — combines the theorem with a NIST
   cross-check (`A_l` within 1 % of 6.2649e8) in one test, so a wrong absolute
   scale cannot hide behind a perfect ratio.
9. `tests/test_hyperfine.py:114-130` — `A ∝ g_I` at `rel=1e-10`, an exact
   identity that pins the whole prefactor chain.
10. `tests/test_M1_E2_rates.py:100-117` — Drake's `2.5e-6 Z^10` with the
    citation, the Z values and the ratio printed in the assertion message.
