# S5 — collision and continuum physics

Slice: electron-impact excitation/ionisation, dielectronic and radiative
recombination, autoionisation/Auger, photoionisation, R-matrix, molecular and
continuum spectra.

Files: the 29 listed files, all present, plus **one added file** —
`tests/test_fac_dw_benchmarks.py` (the live DW excitation/EII vs-FAC gate; it is
the test that actually consumes `benchmarks/results_fac_dw_excitation.csv` and
`results_fac_dw_eii.csv`, so it belongs to this slice). Annex (not repeated in
full here): `docs/test_audit/_S5_molecular_subreport.md` for the three molecular
Auger files.

Measured in this session (`.venv`, `MPLCONFIGDIR=/tmp/atomx_n1/mpl`):

| run | result |
|---|---|
| 8 FAC-comparison files (dw benchmarks, born completion, dr×2, auger×3, continuum-vs-fac) | 42 passed, 3 xfailed, 0 skipped, 78 s |
| `test_eii.py test_r_matrix.py` (`-m "not slow"`) | 37 passed, **1 skipped**, 1 deselected, 36 s |
| 13 light files (continuum, photoionization, hydrogenic, born_cb, dw_angular, pair_ladder, scattering_models, cx_models, spectrum_continuum, eii_ea, dr_density, rr, autoionization) | 163 passed, 74 s |
| `test_r_matrix_phase2_vs_burke1971.py` | 6 passed, 80 s |
| molecular trio | 146 passed, 2 skipped (pyscf absent), 1.5 s |
| **total** | **394 passed, 3 xfailed, 3 skipped, 3 deselected** |

Collected cases: 260 in the 26 non-molecular files (one of them the added file),
148 in the molecular trio.

## 1. Verdict

This slice contains the best physics gates in the repo — `test_dw_born_completion.py`
(independent Maxwellian quadrature, exact Bethe closed form, plane-wave Born
limit), `test_r_matrix_phase2_vs_burke1971.py` (a quoted literature table
checked to 1 % on both resonance positions and widths), the Storey–Hummer
absolute RR anchor and the live-vs-recorded-FAC DW panels — and around them a
large shell of tests that can only fail for plumbing reasons. Of 260
non-molecular cases, 94 are first-principles and 23 are external-oracle, but 112
are E/F: positivity, finiteness, order-of-magnitude bands, provider/model
construction and error paths. The single most important structural fact is that
**the whole electron-impact excitation FAC parity panel lives at
E/I = 1.05, 2, 5** (`benchmarks/results_fac_dw_excitation.csv`), i.e. entirely
below the Born-completion point where the 20x/69x/263x defect lived; the
high-energy region is gated only by `test_dw_born_completion.py`, and that gate's
30 % band is *wider than the 29.3 % floor imposed by the Born blend itself*, so
it cannot detect a collapse of the distorted-wave sum (measured: an
`L_max=2, kappa_max=3` truncation with Ω_DW 114x/419x/1571x below Born still
passes at −15.3 %). On the recombination side, `test_rr_vs_fac.py` does not
touch FAC at all while the real FAC PI/RR panel (`results_fac_pi_rr.csv`,
−63 % at 1 eV) is read by no test; the DR "kernel parity" test is a
same-kernel round-trip (measured ratio 0.99970901, identical at all four
temperatures); the fast DR benchmark gate asserts only the CSV's own recorded
numbers and the CSV has not been regenerated since 2026-07-16, before both the
DR-prefactor fix and the R7 orbital fix. Provenance is mostly honest and
unusually well documented (`RESULTS_fac_dw_eii.md` records the sfac binary size,
mtime and deck; fixtures name their FAC version), with two exceptions: the
`facwave` dumps carry no code/version stamp inside the file, and the DW
excitation FAC column has no dated provenance record of its own. I found no case
of a reference file that is atomx output relabelled as external, but I did find
four `external_fac`-marked checks that assert values atomx itself produced (the
DR kernel round-trip, the DR adapter-vs-fixture ratio, the manifold-vs-fixture
ratio, and the two recorded-atomx columns in the DW panels) and one
atomx-generated "cleanroom" fixture whose own JSON says it is self-generated
(annex).

## 2. File table

Class mix is at the collected-case level. A first-principles / B external oracle
/ C internal cross-check / D frozen self-regression / E tautology / F plumbing.

| file | cases | class mix | worst | note |
|---|---|---|---|---|
| `test_continuum.py` | 22 | `A12 C3 E5 F2` | S3 | Dirac dispersion and free-box density are analytic; the H 1s PI test asserts only σ>0 while the same path is Stobbe-gated 30 lines below. |
| `test_continuum_vs_fac.py` | 4 | `A2 B2` | S4 | Stored FAC free-wave dumps + the analytic √(2/π) normalisation, constant to 1.5 % at 1 keV–200 keV; **not marked `external_fac`**; fixture files carry no code stamp. |
| `test_photoionization.py` | 14 | `A6 E1 F7` | S4 | Fano closed-form limits (q→∞, q=0, ρ²=0, asymptote) are exact; one test named "symmetric limit" asserts only F(0)=1. |
| `test_hydrogenic_cross_sections.py` | 12 | `A5 B3 D1 F3` | S3 | NIST Stobbe 6.30e-18 and Storey–Hummer α₁ₛ = 1.596e-13 are model B-class; the He⁺ threshold literal has no source and duplicates the Z² test. |
| `test_eii.py` | 23 | `A1 B6 C3 E11 F2` | **S1** | Six NIST/Younger-bracketed points are real B; the rate coefficient is gated only by positivity + one monotonic step; the near-threshold 1.05I gate is permanently skipped. |
| `test_eii_ea.py` | 4 | `A1 C1 D1 E1` | S3 | Threshold/window logic is genuine; the "detailed-balance composer" test is an identity by construction. |
| `test_dw_angular_coupling.py` | 37 | `A23 C4 E8 F2` | S4 | 23 genuine angular-momentum/selection-rule checks incl. a 6j orthogonality sum rule; two classes (parity, exchange phase) do arithmetic in the test and never call the code. |
| `test_dw_born_completion.py` | 18 | `A14 C4` | **S1** | Model file: exact Bethe closed form, independent Maxwellian quadrature at 4 temperatures, g_i, Born limit — but the Born band is wider than the blend floor (finding 1). |
| `test_born_cb.py` | 12 | `A7 B1 F4` | S2 | Exact ln E/E and 1/E Born asymptotes with hand-derived constants; H 1s magnitude only bounded to 0.1–1.0 Mb. |
| `test_r_matrix.py` | 16 | `A5 B2 E6 F3` | S2 | Spin-filtering and residual charge are A; the two "FAC" thresholds are NIST/atomx-hydrogenic to 3e-6 under a 5e-4 gate. |
| `test_r_matrix_phase2_vs_burke1971.py` | 6 | `A2 B1 E2 F1` | — | The strongest external gate in the slice (Burke 1971 Table 2, 1 % on E_r and Γ); position selection is by proximity to the reference. |
| `test_rr.py` | 9 | `A2 C2 E3 F2` | S3 | The B-spline RR path is cross-checked against the closed-form Stobbe chain (rtol 0.10); two tests are degenerate re-calls of the same function. |
| `test_rr_vs_fac.py` | 1 | `C1` | **S1** | Named "vs FAC", contains no FAC comparison; the real FAC panel is consumed by nothing. |
| `test_dr_vs_fac.py` | 4 | `C1 D1 E2` | **S1** | Kernel "parity" is a same-kernel round-trip (0.99970901 at every T); the CSV gate never recomputes atomx. |
| `test_dr_fac_reference.py` | 3 | `C1 D1 E1` | S3 | Adapter vs the fixture through the same kernel, band [0.80, 1.05] vs measured 0.86–0.96; label parser pinned to the fixture's 193/194 split. |
| `test_dr_manifold_e2e.py` | 4 | `B1 C1 E2` | S2 | Manifold α(T) within ±10 % of a measured ratio, but both sides pass through the same Burgess kernel; the anchor is fixture-dependent (finding 4). |
| `test_dr_density_suppression.py` | 5 | `C1 E1 F3` | S4 | Model behaviour (IPD > binding dissolves the resonance) plus one literal no-op test (`None` vs `None`, `rtol=0`). |
| `test_auger_fac_panel.py` | 4 | `B2 E2` | **S1** | Honest rewrite of a former tautology, but the working gate is a 300x ratio band; the 10 % gate is xfail (measured 1.13–14.8). |
| `test_auger_kll_vs_fac.py` | 2 | `B1 E1` | **S1** | Same five resonances as the above, live but gated only on magnitude [1e13, 1e17]; the 10 % gate is xfail. |
| `test_jj_level_auger.py` | 3 | `B2 F1` | S2 | ILEV 71 known-answer vs FAC 3.927e14 s⁻¹ (ratio 1.017, band ±20 %) is the only per-level Auger KAT; panel band 0.1–10 is xfail. |
| `test_autoionization.py` | 19 | `A5 C1 E4 F9` | S3 | Closed-form Burgess α at 1.927e-11 with the derivation in the docstring, and T_peak = 2E/3; nine plumbing tests. |
| `test_pair_ladder.py` | 3 | `A1 C1 D1` | S3 | One excellent closed-form 1×1 ladder test; the production-scale ratio 0.833 ∈ [0.75, 0.95] is D. (Subject is MBPT, not scattering — included because listed.) |
| `test_scattering_models.py` | 12 | `A2 E4 F6` | S4 | Coulomb-tail ratio and bare potential are A; the rest is model construction and fail-closed error paths (which are worth keeping). |
| `test_cx_models.py` | 11 | `A3 E5 F3` | S2 | LZ limits (P→1 sudden, P→0 adiabatic, peak at P=1/2) are A; the only magnitude checks are monotonicity/ratios. |
| `test_spectrum_continuum.py` | 5 | `A3 E1 F1` | — | Rybicki–Lightman free-free Gaunt limits and the RR edge (zero below to 1e-30) are clean A. |
| `test_fac_dw_benchmarks.py` (**added**) | 7 | `B2 D2 E3` | S3 | The two live-vs-recorded-FAC rms gates are model B; the two live-vs-recorded-atomx drift gates are D wearing `external_fac`. |
| molecular trio (annex) | 148 (57 fn) | `A8 D3 E6 F40` | **S1** | Half of a 148-case slice is argument validation; the "external" fixture anchors are its own inputs (804.456 eV reproduced to 2e-10 eV). |

Aggregate (non-molecular, 260 cases): **A 94, B 23, C 24, D 7, E 63, F 49**.

## 3. Findings

### S1 — tests that would pass for a physically wrong result

**1. The Born-limit gate cannot detect a collapsed DW sum: the 30 % band exceeds
the 29.3 % floor set by the blend itself.**
`tests/test_dw_born_completion.py:66` (`BORN_TOL = 0.30`) and `:170`
(`assert np.all(np.abs(rel) <= BORN_TOL)`).
At 50–200x threshold the geometric tail is unresolved, so the solver forces
`tail = max(tail_floor, omega_partial)` (`solver.py:663-676`) and
`born_blend_weights` returns `w = sqrt(tail/(tail+Ω_partial)) ≥ 1/√2`
(`born.py:272-306`). The shipped value is then
`Ω = (1−w)Ω_DW + wΩ_Born ≥ 0.707 Ω_Born` for **any** Ω_DW, i.e.
`|Ω/Ω_Born − 1| ≤ 0.293 < 0.30`. Measured (Ne He-like, threshold 914.02 eV):

| E/I | Ω (shipped) | Ω_Born | rel | w | implied Ω_DW | Ω_DW/Ω_Born |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.16377 | 0.21224 | −0.228 | 0.744 | 0.0232 | 0.112 |
| 100 | 0.21803 | 0.28463 | −0.234 | 0.758 | 0.0096 | 0.034 |
| 200 | 0.32110 | 0.41823 | −0.232 | 0.766 | 0.0038 | 0.009 |

The implied DW partial-wave sum is still **9x/30x/110x below the Born limit**;
what the commit fixed is the blend engagement, not the kernel. Direct
counterfactual run with `L_max=2, kappa_max=3` (Ω_DW 114x/419x/1571x below
Born at 50/100/200x): shipped rel = −0.153/−0.154/−0.154 → **the test passes**.
Remediation: (a) pin Ω_DW explicitly at fixed `L_max=6, kappa_max=10`
(0.0232/0.0096/0.0038) so a *further* collapse of the partial-wave sum trips,
and (b) state in the docstring that this gate validates the Born completion's
engagement, not the DW sum, since 1−1/√2 = 0.293 of the 0.30 budget is
unreachable by construction. A tighter band is only meaningful after raising
`kappa_max` until `converged=True`.

**2. The EII rate coefficient has no magnitude gate at all.**
`tests/test_eii.py:143-160` — assertions are `rates.shape`, `np.all(rates>0)` and
`rates[1] > rates[0]` at T = 5/20/100 eV for Ar. The sibling quantity
`collision_strength_to_rate` was dimensionally wrong by 2e10 in this session
(H0 defect); the same class of error in `eii_rate_coefficient` (which does
delegate to `maxwellian_rate`) would pass this test. Remediation: mirror
`test_dw_born_completion.py::test_collision_strength_to_rate_matches_closed_form`
— feed a constant σ = 1e-16 cm² on a 60 kT grid and compare against the
first-principles quadrature `sqrt(8/(π m_e))(kT)^{-3/2}∫σE e^{-E/kT}dE` at
T = 1/10/100/1000 eV, plus a `g_i`-type scaling check.

**3. `test_rr_vs_fac.py` compares against no FAC data, and the real FAC PI/RR
panel is read by no test.**
`tests/test_rr_vs_fac.py:60-90` compares the B-spline path against the internal
hydrogenic chain (`milne_rr_cross_section` + `maxwellian_rate`, rtol 0.15,
measured 5–13 %) — a legitimate C, misnamed. Meanwhile
`benchmarks/results_fac_pi_rr.csv` (created by `a894c55`) holds a real sfac
`RRTable` comparison: `Fe25+→Fe24+(1s²)` rel_diff **−0.626 at 1 eV**,
−0.120 at 15.1 eV, and −0.121 for `Fe24+→Fe23+ 1s²2s (ILEV 0→8)` at 16 keV;
`grep` finds **zero** tests referencing that file (`docs/plans/scattering_gap_closeout_plan.md:42`
instead cites `test_rr_vs_fac.py` as the RR gate). Remediation: add
`test_pi_rr_vs_fac.py` in the `test_fac_dw_benchmarks.py` style — import
`bench_fac_pi_rr`, recompute the atomx column live at the CSV's E_eV points and
gate live-vs-recorded-FAC with the measured envelope; or delete the file and
the plan's claim.

**4. DR "kernel parity with FAC" is a same-kernel round-trip.**
`tests/test_dr_vs_fac.py:88-106`, `KERNEL_TOLERANCE = 0.01` (`:46`). The fixture
column `fe17_dr_fac.json::alpha_fac_cm3_s` was itself produced by combining FAC
widths through this same Burgess kernel (the file's `source` says so). Measured
by feeding the fixture's own resonances to `dr_rate_coefficient`:

```
T_MK            1      5      10     50
fixture*Ry^1.5  6.863e-13 3.285e-12 2.040e-12 2.896e-13
kernel(states)  6.861e-13 3.284e-12 2.039e-12 2.895e-13
ratio           0.99970901 0.99970901 0.99970901 0.99970901
```

The ratio is a *constant* to 6 digits at every temperature (the residual is the
`RYDBERG_EV**1.5` conversion), so the test asserts that the code still
reproduces the number it produced when the fixture was written; any error in
`_DR_PREFACTOR_CM3_S_EV_32` cancels. The 1 % gate is 34x the actual residual.
Class C-degenerate, marked `external_fac`. Remediation: keep the test but mark
it `snapshot`, rename it (e.g. `test_kernel_reproduces_recorded_alpha`), tighten
to `rtol=1e-3` (2.9e-4 measured), and put the prefactor check where it belongs
(the closed-form test in `tests/test_plasma_physics.py`).

**5. The fast DR benchmark gate never recomputes atomx, and the CSV predates two
physics fixes.**
`tests/test_dr_vs_fac.py:129-163` reads
`benchmarks/results_fac_dr.csv` and asserts `rms ≤ 0.35` and the ratios recorded
in the note column — deleting `bench_fac_dr.py`'s manifold pipeline could not
fail it. `git log -1 --format=%ad` on the CSV: **2026-07-16** (`c14968d`),
i.e. before the R7 asymmetric-mean-field fix (2026-09-20) and the DR prefactor
fix `833451b` (2026-09-22). The DW CSVs were given exactly this live-recompute
treatment in `a894c55`/`f77be88`; the DR one was not. Remediation: port the
`test_fac_dw_benchmarks.py` pattern (module-scope fixture calling
`bench_fac_dr.run_atomx()`, gate live-vs-recorded atomx at the measured drift
and live-vs-FAC at the measured ratio ±10 %), and regenerate the CSV.

**6. The near-threshold EII gate is permanently skipped; the EII FAC panel stops
at 1.5I.**
`tests/test_eii.py:388-424` skips with "HYP-841 Q1 pending: fresh FAC panel has
no 1.05I shape rows" (measured: `1 skipped` in the run above) because
`results_fac_dw_eii.csv` has only `E/I ∈ {1.5, 2, 5}`. So the region where the
direct EII cross-section rises from zero — and where the DW-Born defect class
bites hardest — has no external reference, and the threshold-convention question
is left open in `benchmarks/RESULTS_fac_dw_eii.md`. Additionally
`test_fresh_three_z_fac_k_shell_gate` (`:347-385`) requests only
`(2.0, 5.0)`, ignoring the 1.5I rows that *are* in the CSV, and skips whenever
the CSV or its provenance paragraph is missing. Remediation: add the 1.05I/1.01I
rows to the panel (the driver already supports `ENERGY_FACTORS`) and gate
absolute σ with a numerical floor instead of a ratio; make the 1.5I rows part of
the requested set.

**7. The only working per-level Auger gates are 300x and 10x bands; every 10 %
gate is xfail.**
`tests/test_auger_fac_panel.py:68-69` (`RATIO_FLOOR = 0.1`, `RATIO_CEIL = 30`)
with measured `atomx/FAC = 1.13/1.38/5.96/7.11/14.82` (`:199`);
`tests/test_jj_level_auger.py:157` (`0.1 ≤ ratio ≤ 10`), currently
`xfail(strict=True)` because ILEV 32 sits at 0.0966. A factor-2 error either way
passes. `docs/VERIFICATION.md` already retracts the "1.005–1.023 vs FAC" claim,
so the README must not carry it. Remediation: keep the order-of-magnitude gate
as a tripwire but (a) change `test_jj_level_auger`'s strict xfail into a
recorded `xfail` with an issue link, (b) add a *differential* gate that does not
depend on the angular factor: the per-channel ratio atomx/FAC must be constant
across the five panel rows for the J-resolved path (measured spread 10.4x → the
target of <2x is the honest next milestone).

**8. "Detailed balance" composer is an algebraic identity.**
`tests/test_eii_ea.py:68-93` builds `d` with `Γ_a = 1e14, A_r = 0` and a twin
with `A_r = 3e12`, then feeds `partial_auger_to_final = {"r1": 3e12}`. Both
branches reduce to `Γ_a·X/(Γ_a+X)` with the same `X` and the same
`_DR_PREFACTOR_CM3_S_EV_32`, so `a_re == a_dr` to machine precision **by
construction** (`rtol=1e-12`); the docstring's "detailed-balance composer" is not
tested against any inverse process. Remediation: assert the *relation* to the
inverse process — `alpha_RE(d)·exp(E_d/kT)·g_i` must equal
`alpha_capture(d)·(2J+1)/(g_f)` for a resonance with a known partial width — or
relabel the test as a branching-arithmetic check.

**9. (annex) The molecular "cleanroom" anchors are the fixture's own inputs.**
`tests/test_molecular_auger_hyp1066_regression.py:158-171` asserts
804.456 ± 0.05 eV; measured 804.455999999798 eV, and the fixture's
`ε_2p = -2.7183936399306443` Ha is the exact solution of
`804.456/HARTREE_EV`. The branch ratios [0.73, 0.23, 0.04] are the injected raw
intensities (ERIs injected as `sqrt(raw)`, exchange identically 0). The file
does not use the `external_fac` marker and its JSON says "Atomx self-generated",
so this is D, but it is presented as a reference-anchored regression. See the
annex, findings F1–F7.

### S2 — inconsistent pins / tolerances that cannot distinguish the physics

**10. Two FAC-derived Fe XVII DR reference sets disagree by 3.9x (1 MK) to 1.8x
(50 MK).**
`tests/data/fe17_dr_fac.json` (194 resonances, dr33+dr34+KLM, AICut 1e-4) vs
`tests/data/fe17_dr33_fac_cleanroom.json` (236 resonances, dr33, AICut 0), both
run through the current kernel:

```
T_MK                 1        5        10       50
kernel(fixture)      6.86e-13 3.28e-12 2.04e-12 2.90e-13
kernel(cleanroom)    2.67e-12 7.30e-12 4.00e-12 5.09e-13
ratio                0.257    0.450    0.510    0.569
median Γ_auger       1.83e13 s^-1 (fixture) vs 7.72e11 s^-1 (cleanroom)
```

The atomx manifold pipeline is therefore simultaneously "0.487 of FAC" at 1 MK
(`test_dr_manifold_e2e.py:150`) and ≈1.9x FAC against the older fixture, and the
"1 MK is Boltzmann suppression" narrative only holds for one of the two.
Remediation: adjudicate which AITable run is the reference (the cleanroom deck
with `AICut 0` is the better documented), retire the other column, and state the
comparison in the surviving fixture against a *true* FAC DRTable/α column rather
than a kernel-combined one.

**11. The "FAC" R-matrix thresholds are NIST/atomx values under a gate 150x
looser than the agreement.**
`tests/test_r_matrix.py:23-29`, `:152-169`. Measured:
`hydrogenic_dirac_energy` with the reduced-mass factor gives H = −13.59846830 eV
and He⁺ = −54.41821051 eV, i.e. 3.3e-6 and 7.9e-6 from the quoted
"FAC" numbers, which in turn sit 8e-7 (H) and 3.3e-7 (He⁺, NIST 54.417760) from
the NIST ionisation energies — the numbers cannot be distinguished from NIST or
from atomx's own hydrogenic output. The 5e-4 gate is ~100x the achieved
agreement, and it is loose in a physically meaningful way: an **infinite-mass**
He⁺ (no finite-nuclear-mass correction) is 1.45e-4 off and **passes**; only H
(5.48e-4) would fail. Remediation: quote the source and the number in the test
(NIST ASD H I 13.598434599702 eV, He II 54.417760 eV), tighten to `rel=1e-5`,
and add an explicit "without the reduced-mass factor this fails" assertion.

**12. The external FAC panels stop below the Born-completion point, and the
"recorded atomx" drift band is 2x its own measured drift.**
`benchmarks/results_fac_dw_excitation.csv` covers `E/I ∈ {1.05, 2, 5}`;
`results_fac_dw_eii.csv` covers `{1.5, 2, 5}`. `tests/test_fac_dw_benchmarks.py:83`
(`EXC_ATOMX_REPRO = 0.05`) vs the module's own measurement of ≤2.6 % (the Z=54
rows were recorded at the pre-CI-fix threshold and are 2.2–2.6 % stale, as the
CSV note admits). Remediation: record one high-energy row (e.g. `E/I = 50`) in
both panels — the DW driver already interpolates from FAC's own CE grid, so this
needs one live `sfac` run to confirm the grid covers it — and tighten
`EXC_ATOMX_REPRO` to 0.03 (or re-record the Z=54 rows at the current threshold).

**13. Ratio-in-a-wide-band comparisons presented as validation.**
`tests/test_eii.py:245` (`1/3 < DW/BEB < 3`), `:336` (`0.05 < DW/BEB < 15`,
i.e. 300x for Ne at 100 eV), `:176` (`1e-19 < σ < 1e-17`), `:214`
(`1e-18 < σ < 1e-17`); `tests/test_auger_kll_vs_fac.py:58-59`
(`[1e13, 1e17]` s⁻¹); `tests/test_cx_models.py:92` (`s3/s2 > 1`).
Each is documented as a sanity band, but they are the *only* gates on those
quantities. Remediation: for each, replace the band with the nearest
first-principles anchor — BEB's own peak/limit for the DW-BEB comparisons (or
drop the comparison), Younger 1981 for He⁺ (already present at `:120`), and the
Olson–Salop `R_c(q)²` ratio 2.25 (currently only `> 1`).

**14. The `facwave` FAC dumps carry no code/version stamp in-file.**
`tests/data/facwave/*.txt` headers contain `n, kappa, energy, dn, SelfEne, vc,
vr, ilast, …` — nothing identifies FAC, its version or the deck;
`tests/test_continuum_vs_fac.py:1-16` supplies that only in prose, and the same
docstring claims the √(2/π) ratio holds "at ALL energies (100 eV .. 200 keV)"
while the four pinned files are 1 keV/10 keV/40 keV/200 keV (`:36-41`).
Remediation: add a `# code = FAC 1.1.5, deck = …` header line to each dump and
correct the docstring range.

**15. `results_fac_pi_rr.csv` / `REPORT_FAC_PARITY.md` disagree with themselves
about the same FAC number.**
The CSV labels its first case `Fe25+ -> Fe24+(1s²) RR` with
`sigma_atomx = 27.95` at 1 eV, while `benchmarks/REPORT_FAC_PARITY.md:85-95`
lists the same FAC number (74.8) under "He-like Fe²⁴⁺ K-shell" with
`sigma_atomx = 688` (`grep` for `t_fac_s`/`t_atomx_s` shows the columns are
benchmark runtimes, not physics). Both files were created in `a894c55`.
Remediation: regenerate the report tables from the CSV in the same commit, or
delete the stale table.

### S3 — D-class masquerading, duplicated coverage

**16. `external_fac` on tests that assert atomx's own recorded output.**
`tests/test_fac_dw_benchmarks.py:197-228` and `:257-277`
(live-vs-recorded-`omega_atomx`/`sigma_atomx`); `tests/test_dr_vs_fac.py:88-106`
(same kernel); `tests/test_dr_fac_reference.py:97-118` (the fixture's labels
drive the parser assertions). Marker contract is documented in
`docs/VERIFICATION.md`: `external_fac` = "asserts atomx against stored external
reference data". Remediation: split each file's tests — `external_fac` only for
the live-vs-FAC columns, `snapshot` for the live-vs-recorded-atomx columns.

**17. Duplicated Auger coverage.**
`tests/test_auger_kll_vs_fac.py:127-150` and
`tests/test_auger_fac_panel.py:177-208` compute the same five FAC panel widths
from the same `Atom(Z=26, charge=16, config=[...])` solve and gate the same
magnitude band; the second file's docstring says it follows the first's
pattern. Remediation: keep one live panel (the one with the J-resolved KAT) and
fold the other's xfail into it.

**18. `test_hydrogenic_cross_sections.py:23-26`** pins He⁺ σ_th = 1.575e-18 cm²
with no source, and the very next test derives the same information from the
H value via σ ∝ Z⁻². Remediation: delete it, or quote the source (it is the
Stobbe/DIRAC value at Z=2) and keep only the ratio test.

### S4 — noise, misnamed and dead tests

**19. `tests/test_dw_angular_coupling.py:166-183` (`TestParityConservation`) and
`:303-320` (`TestExchangePhase`)** perform the arithmetic in the test body and
never call atomx (`assert (0+0+1+1) % 2 == 0`, `assert exp % 2 == 1`). Delete or
assert on `_channel_allowed`/the kernel's own phase output.

**20. `tests/test_dw_angular_coupling.py:106-121`** (`test_1s_to_2p_dipole`)
re-implements `reduced_C` from the same 3j and compares — E-class.
`tests/test_dw_angular_coupling.py:358-405` (`test_positive_definite`,
`test_zero_when_all_products_zero`) assert properties of a sum of squares.

**21. `tests/test_dr_density_suppression.py:30-35`** compares two calls that
differ only by `ne_cm3=None` with `rtol=0, atol=0` — a literal no-op.

**22. `tests/test_photoionization.py:38-43`** is named `test_symmetric_limit`
and asserts only `F(ε=0) = q² = 1`; the symmetry claim is never checked.

**23. `tests/test_autoionization.py:163-172`** (`test_at_T_interpolation`)
interpolates at a grid point (`T = linspace(1, 100, 1001)` contains 10.0) and
compares against the nearest grid value — a tautology.

**24. `tests/test_continuum.py:193-212`** asserts only `σ > 0` at three photon
energies on a path that `:230-239` already gates against Stobbe to 15 %;
`:151-156` (`factor < 10.0`) is an unbounded-band assertion.

## 4. Model tests (worth imitating)

1. **`tests/test_dw_born_completion.py:240-311`** —
   `test_collision_strength_to_rate_matches_closed_form` (+`_magnitude_and_scaling`,
   `_uses_g_i`, `_matches_crm_convention`): an independently written `scipy.quad`
   of the textbook Maxwellian average at T = 1/10/100/1000 eV, agreeing to
   <1e-6, plus the 8.01e-08 cm³/s absolute value and the kT^{-1/2} scaling. This
   is the template for every rate coefficient in the package.
2. **`tests/test_dw_born_completion.py:314-359`**
   (`TestCoulombBetheTopupCoefficient`): the closed form
   `Ω = (4 g_i f/ΔE)·ln[(k_i+k_f)/(k_i−k_f)]` written out in the test and
   matched to `rtol=1e-12`, with the g_i linearity and the *absence* of a hidden
   L dependence asserted separately — the two defects the review found.
3. **`tests/test_r_matrix_phase2_vs_burke1971.py:33-37, 164-204`** — Burke,
   Hibbert & Robb 1971 Table 2 quoted in the test, three resonances, positions
   *and* widths to 1 %, with the configuration identity of each extracted
   resonance asserted first.
4. **`tests/test_hydrogenic_cross_sections.py:68-91`** —
   `test_milne_rr_h1s_absolute_rate_anchor`: Storey & Hummer 1995
   α_A − α_B = 1.596e-13 cm³/s at 10⁴ K, `rel=0.02, abs=0.0` (the `abs=0.0` is
   needed because the default approx floor would swamp 1.6e-13). One absolute
   anchor for the whole Milne chain.
5. **`tests/test_fac_dw_benchmarks.py:231-250, 280-297`** — the live-vs-recorded
   FAC rms gates (rms 3.28 %/max 5.22 % and 2.03 %/3.10 % against gates of
   5 %/10 % and 3.5 %/6 %): recompute atomx at the CSV's own grid points and
   compare against the external column, tolerances calibrated to the measurement
   with ~1.5x headroom.
6. **`tests/test_autoionization.py:80-124`** — the closed-form Burgess α
   (1.92696898e-11 with the derivation and the prefactor spelled out in the
   docstring) and `T_peak = 2E/3` from `d/dT[T^{-3/2}e^{-E/T}] = 0`.
7. **`tests/test_continuum_vs_fac.py:73-99`** — a stored FAC oracle reduced to a
   single analytic number: the ratio P_atomx/P_FAC must equal √(2/π) with a
   *constant* pointwise ratio (rms < 1.5 %) at four energies spanning 1 keV to
   200 keV, which is what pins the relativistic (E+2c²) factor in the
   δ(E)-normalisation.
8. **`tests/test_spectrum_continuum.py:94-124`** — the RR edge: zero below the
   edge to `abs=1e-30` and non-zero above, with the edge energy computed from the
   atom's own 1s binding energy.
9. **`tests/test_pair_ladder.py:52-85`** — `E = −(2J+1)v₀²/(D+V)` for a 1×1
   doubles space, cross-checked against the 200-term geometric series the solver
   claims to resum, with the convergence domain `|V/D| < 1` asserted first.
10. **`tests/test_jj_level_auger.py:100-115`** — a single named channel
    (Fe XVII KLL ILEV 71, J_a = 0) against FAC's AITable 3.927e14 s⁻¹ with the
    measured ratio 1.017 in the docstring and a ±20 % tripwire band.
11. **`tests/test_scattering_models.py:37-46`** — the potential must reduce to a
    pure Coulomb tail: `V(16)/V(32) = 2` to 1e-12.
12. **`tests/test_continuum.py:229-258`** — the finite-box PI cross-section
    tracked against the analytic Stobbe value on a 7-point threshold scan
    (rtol 0.15) plus the below-threshold zero.
13. (annex) **`tests/test_molecular_auger_initial_state.py:400, 412, 456`** —
    the S-metric construction `CᵀSC = I` by hand, the closed-shell spin-coupling
    count `n + 2·n(n−1)/2`, and the Koopmans energy written out sign by sign.
